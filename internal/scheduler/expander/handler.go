package expander

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator/filter"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/samber/lo/mutable"
	corev1 "k8s.io/api/core/v1"
	errors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	karpv1 "sigs.k8s.io/karpenter/pkg/apis/v1"
)

const (
	MaxInFlightNodes           = 15
	WaitingInFlightNodesPeriod = 20 * time.Second
)

type NodeExpander struct {
	client            client.Client
	scheduler         *scheduler.Scheduler
	allocator         *gpuallocator.GpuAllocator
	logger            klog.Logger
	inFlightNodes     map[string][]*tfv1.GPU
	preSchedulePods   map[string]*tfv1.AllocRequest
	preScheduleTimers map[string]*time.Timer
	eventRecorder     record.EventRecorder
	mu                sync.RWMutex
	ctx               context.Context
}

func NewNodeExpander(
	ctx context.Context,
	allocator *gpuallocator.GpuAllocator,
	scheduler *scheduler.Scheduler,
	recorder record.EventRecorder,
) *NodeExpander {

	expander := &NodeExpander{
		client:            allocator.Client,
		scheduler:         scheduler,
		allocator:         allocator,
		logger:            log.FromContext(ctx).WithValues("component", "NodeExpander"),
		inFlightNodes:     make(map[string][]*tfv1.GPU, 10),
		preSchedulePods:   make(map[string]*tfv1.AllocRequest, 20),
		preScheduleTimers: make(map[string]*time.Timer, 20),
		eventRecorder:     recorder,
		ctx:               ctx,
	}
	allocator.RegisterBindHandler(func(req *tfv1.AllocRequest) {
		obj := &corev1.ObjectReference{
			Kind:            "Pod",
			APIVersion:      "v1",
			Namespace:       req.PodMeta.Namespace,
			Name:            req.PodMeta.Name,
			UID:             req.PodMeta.UID,
			ResourceVersion: req.PodMeta.ResourceVersion,
		}
		recorder.Eventf(obj, corev1.EventTypeNormal, "NodeExpansionCheck",
			"new node provisioned and pod scheduled successfully")
		expander.logger.Info("new node provisioned and pod scheduled successfully",
			"namespace", req.PodMeta.Namespace, "pod", req.PodMeta.Name)
		expander.RemovePreSchedulePod(req.PodMeta.Name, true)
	})
	return expander
}

func (e *NodeExpander) ProcessExpansion(ctx context.Context, pod *corev1.Pod) error {
	if pod == nil {
		return fmt.Errorf("pod cannot be nil")
	}
	if _, ok := e.preSchedulePods[pod.Name]; ok {
		e.logger.Info("Pod already in pre-schedule state, skipping expansion check and wait for expansion", "pod", klog.KObj(pod))
		return nil
	}
	if len(e.inFlightNodes) >= MaxInFlightNodes {
		e.logger.Error(nil, "Too many inFlight nodes, skipping expansion to avoid too many nodes provisioned concurrently")
		time.Sleep(WaitingInFlightNodesPeriod)
		return nil
	}

	// Step 1: Simulate scheduling without GPU plugins
	suitableNodes, err := e.simulateSchedulingWithoutGPU(ctx, pod)
	if err != nil {
		e.eventRecorder.Eventf(pod, corev1.EventTypeNormal, "NodeExpansionCheck",
			"can not schedule on any nodes even without GPU constraints, karpenter should take over expansion. error: %w", err)
		e.logger.Info("Pod schedulable but no GPU nodes available, karpenter should take over expansion",
			"namespace", pod.Namespace, "pod", pod.Name, "error", err)
		return nil
	}
	if len(suitableNodes) == 0 {
		e.eventRecorder.Eventf(pod, corev1.EventTypeNormal, "NodeExpansionCheck",
			"can not schedule on any nodes even without GPU constraints, karpenter should take over expansion, 0 fit nodes")
		e.logger.Info("Pod schedulable but no GPU nodes available, karpenter should take over expansion",
			"namespace", pod.Namespace, "pod", pod.Name)
		return nil
	}

	// Step 2: Check if it's a GPU resource issue, include inFlightNodes
	nodeGPUs := e.allocator.GetNodeGpuStore()
	allGpus := []*tfv1.GPU{}
	// Shuffle gpuNodes to avoid always using the same node in the same region
	mutable.Shuffle(suitableNodes)
	for _, gpuNode := range suitableNodes {
		if gpus, ok := nodeGPUs[gpuNode.Name]; ok {
			for _, gpu := range gpus {
				allGpus = append(allGpus, gpu)
			}
		}
	}
	inFlightGPUSnapshot := make([]*tfv1.GPU, 0, len(e.inFlightNodes)*4)
	for _, inFlightGPUs := range e.inFlightNodes {
		for _, gpu := range inFlightGPUs {
			snapshot := gpu.DeepCopy()
			inFlightGPUSnapshot = append(inFlightGPUSnapshot, snapshot)
			allGpus = append(allGpus, snapshot)
		}
	}
	if len(allGpus) == 0 {
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionCheck",
			"all schedulable nodes are none GPU nodes, manual check required")
		e.logger.Info("No GPU nodes can put the Pod, manual check required", "namespace", pod.Namespace, "pod", pod.Name)
		return nil
	}

	// Step 3: Check if it's a GPU resource issue, include inFlightNodes
	allocRequest, satisfied, isResourceIssue := e.checkGPUFitWithInflightNodes(pod, allGpus, inFlightGPUSnapshot)
	if satisfied {
		// GPU free-up during expansion, or satisfied by in-flight nodes, pod can be scheduled now or whiles later
		e.eventRecorder.Eventf(pod, corev1.EventTypeNormal, "NodeExpansionCheck",
			"fit GPU resources, pod should be scheduled now or whiles later")
		return nil
	}
	if !isResourceIssue {
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionCheck",
			"pod scheduling failure not due to GPU resources, manual check required")
		e.logger.Info("Pod scheduling failure not due to GPU resources, manual check required",
			"namespace", pod.Namespace, "pod", pod.Name)
		return nil
	}

	// Step 4: Caused by insufficient GPU resources, try find node util it satisfies the pod
	preScheduled := false
	for _, gpuNode := range suitableNodes {
		// when node is not owned by any known provisioner, skip check, util find a node can be expanded
		if len(gpuNode.OwnerReferences) == 0 {
			continue
		}
		preparedNode, preparedGPUs := e.prepareNewNodesForScheduleAttempt(gpuNode, nodeGPUs[gpuNode.Name])
		if !e.checkGPUFitForNewNode(pod, preparedGPUs) {
			continue
		}

		e.logger.Info("prepare new node for schedule attempt from existing node", "existingNode", gpuNode.Name, "newNode", preparedNode.Name)
		err = e.createGPUNodeClaim(ctx, pod, preparedNode)
		if err != nil {
			return err
		}

		e.addInFlightNodeAndPreSchedulePod(allocRequest, preparedNode, preparedGPUs)
		preScheduled = true
		break
	}
	if !preScheduled {
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionFailed", "failed to satisfy the pending pod, no potential GPU nodes can fit")
		return fmt.Errorf("failed to satisfy the pending pod, no potential GPU nodes can fit")
	}
	return nil
}

func (e *NodeExpander) addInFlightNodeAndPreSchedulePod(allocRequest *tfv1.AllocRequest, node *corev1.Node, gpus []*tfv1.GPU) {
	e.mu.Lock()
	e.inFlightNodes[node.Name] = gpus
	podMeta := allocRequest.PodMeta
	e.preSchedulePods[podMeta.Name] = allocRequest
	// Add timer for each pre-scheduled pod, if not scheduled for 10 minutes, make warning event and remove from mem
	timer := time.AfterFunc((10 * time.Minute), func() {
		currentPod := &corev1.Pod{}
		err := e.client.Get(e.ctx, client.ObjectKey{Name: podMeta.Name, Namespace: podMeta.Namespace}, currentPod)
		if err != nil {
			if errors.IsNotFound(err) || !currentPod.DeletionTimestamp.IsZero() {
				e.RemovePreSchedulePod(podMeta.Name, false)
			}
			e.logger.Error(err, "failed to get pod for node expansion check",
				"namespace", podMeta.Namespace, "pod", podMeta.Name)
			e.RemovePreSchedulePod(podMeta.Name, false)
			return
		}
		if currentPod.Spec.NodeName != "" {
			// already scheduled, remove pre-scheduled pod
			e.eventRecorder.Eventf(currentPod, corev1.EventTypeNormal, "NodeExpansionCheck",
				"new node provisioned and pod scheduled successfully")
			e.logger.Info("new node provisioned and pod scheduled successfully",
				"namespace", podMeta.Namespace, "pod", podMeta.Name)
			e.RemovePreSchedulePod(podMeta.Name, false)
		} else {
			// not scheduled, record warning event and remove pre-scheduled pod
			e.eventRecorder.Eventf(currentPod, corev1.EventTypeWarning, "NodeExpansionCheck",
				"failed to schedule pod after 10 minutes")
			e.logger.Info("failed to schedule pod after 10 minutes",
				"namespace", podMeta.Namespace, "pod", podMeta.Name)
			e.RemovePreSchedulePod(podMeta.Name, false)
		}
	})
	e.preScheduleTimers[podMeta.Name] = timer
	e.mu.Unlock()
}

func (e *NodeExpander) RemoveInFlightNode(nodeName string) {
	if e == nil {
		return
	}
	e.mu.Lock()
	delete(e.inFlightNodes, nodeName)
	e.logger.Info("Removed in-flight node", "node", nodeName, "remaining inflight nodes", len(e.inFlightNodes))
	e.mu.Unlock()
}

func (e *NodeExpander) RemovePreSchedulePod(podName string, stopTimer bool) {
	if e == nil {
		return
	}
	e.mu.Lock()
	if stopTimer {
		if timer, ok := e.preScheduleTimers[podName]; ok {
			timer.Stop()
		}
	}
	delete(e.preScheduleTimers, podName)
	delete(e.preSchedulePods, podName)
	e.logger.Info("Removed pre-scheduled pod", "pod", podName, "remaining pre-scheduled pods", len(e.preSchedulePods))
	e.mu.Unlock()
}

func (e *NodeExpander) prepareNewNodesForScheduleAttempt(
	templateNode *corev1.Node, templateGPUs map[string]*tfv1.GPU,
) (*corev1.Node, []*tfv1.GPU) {
	newPreparedNode := templateNode.DeepCopy()
	newPreparedNode.Name = constants.TensorFusionSystemName + "-" + rand.String(10)
	newPreparedGPUs := []*tfv1.GPU{}
	for _, gpu := range templateGPUs {
		gpuCopy := gpu.DeepCopy()
		gpuCopy.Name = "gpu-" + rand.String(12)
		gpuCopy.Status.Available = gpuCopy.Status.Capacity.DeepCopy()
		newPreparedGPUs = append(newPreparedGPUs, gpuCopy)
	}
	return newPreparedNode, newPreparedGPUs
}

func (e *NodeExpander) simulateSchedulingWithoutGPU(ctx context.Context, pod *corev1.Pod) ([]*corev1.Node, error) {
	state := framework.NewCycleState()
	state.SetRecordPluginMetrics(false)
	podsToActivate := framework.NewPodsToActivate()
	state.Write(framework.PodsToActivateKey, podsToActivate)
	state.Write(fwk.StateKey(constants.SchedulerSimulationKey), &gpuallocator.SimulateSchedulingFilterDetail{
		FilterStageDetails: []filter.FilterDetail{},
	})

	// simulate schedulingCycle non side effect part
	fwkInstance := e.scheduler.Profiles[pod.Spec.SchedulerName]
	if fwkInstance == nil {
		log.FromContext(ctx).Error(nil, "scheduler framework not found", "pod", pod.Name, "namespace", pod.Namespace)
		return nil, fmt.Errorf("scheduler framework not found")
	}
	if pod.Labels == nil {
		return nil, fmt.Errorf("pod labels is nil, pod: %s", pod.Name)
	}

	// Disable the tensor fusion component label to simulate scheduling without GPU plugins
	// NOTE: must apply patch after `go mod vendor`, FindNodesThatFitPod is not exported from Kubernetes
	// Run `git apply ./patches/scheduler-sched-one.patch` once or `bash scripts/patch-scheduler.sh`
	if !utils.IsTensorFusionPod(pod) {
		return nil, fmt.Errorf("pod to check expansion is not a tensor fusion worker pod: %s", pod.Name)
	}
	delete(pod.Labels, constants.LabelComponent)
	scheduleResult, _, err := e.scheduler.FindNodesThatFitPod(ctx, fwkInstance, state, pod)
	pod.Labels[constants.LabelComponent] = constants.ComponentWorker
	if len(scheduleResult) == 0 {
		return nil, err
	}
	result := []*corev1.Node{}
	for _, nodeInfo := range scheduleResult {
		result = append(result, nodeInfo.Node())
	}
	return result, nil
}

func (e *NodeExpander) checkGPUFitWithInflightNodes(pod *corev1.Pod, gpus []*tfv1.GPU, inflightSnapshot []*tfv1.GPU) (
	allocRequest *tfv1.AllocRequest,
	satisfied bool,
	isResourceIssue bool,
) {
	// NOTE: a known issue, if cpu/mem not enough or affinity not satisfied for pre-scheduled pods inside inFlightNodes,
	// it will not be considered, when inflight created and the Pod still not be able to schedule on new node,
	// wait next scheduling check and node expansion period (k8s move UnscheduleQueue to ActiveQueue every 5 minutes)
	for _, alloc := range e.preSchedulePods {
		preScheduledPodPreAllocated := false
		for _, gpu := range inflightSnapshot {
			if gpu.Status.Available.Tflops.Cmp(alloc.Request.Tflops) > 0 &&
				gpu.Status.Available.Vram.Cmp(alloc.Request.Vram) > 0 {
				gpu.Status.Available.Tflops.Sub(alloc.Request.Tflops)
				gpu.Status.Available.Vram.Sub(alloc.Request.Vram)
				preScheduledPodPreAllocated = true
				break
			}
		}
		// this is unexpected, all pre-scheduled pod should be able to place into inFlight node
		// possible happen when new node added to cluster and removed from inFlight nodes, simultaneously,
		// new Pods added and also unschedulable, trigger node expansion before previous Pod scheduled
		if !preScheduledPodPreAllocated {
			e.logger.Info("[Warning] pre-scheduled pod can not set into InFlight node anymore, remove queue and retry later",
				"pod", alloc.PodMeta.Name, "namespace", alloc.PodMeta.Namespace)
			e.RemovePreSchedulePod(alloc.PodMeta.Name, true)
		}
	}

	// Get allocation request
	e.mu.RLock()
	defer e.mu.RUnlock()
	allocRequest, _, err := e.allocator.ComposeAllocationRequest(pod)
	if err != nil {
		return nil, false, true
	}

	quotaStore := e.allocator.GetQuotaStore()
	if err := quotaStore.CheckSingleQuotaAvailable(allocRequest); err != nil {
		e.logger.Error(err, "can not schedule pod due to single workload quotas issue")
		return allocRequest, false, false
	}

	// Check total quota with pre-scheduled pods
	toScheduleResource := &tfv1.GPUResourceUsage{
		Requests: tfv1.Resource{
			Tflops: resource.Quantity{},
			Vram:   resource.Quantity{},
		},
		Limits: tfv1.Resource{
			Tflops: resource.Quantity{},
			Vram:   resource.Quantity{},
		},
		Workers: int32(len(e.preSchedulePods)),
	}
	for _, alloc := range e.preSchedulePods {
		toScheduleResource.Requests.Tflops.Add(alloc.Request.Tflops)
		toScheduleResource.Requests.Vram.Add(alloc.Request.Vram)
		toScheduleResource.Limits.Tflops.Add(alloc.Limit.Tflops)
		toScheduleResource.Limits.Vram.Add(alloc.Limit.Vram)
	}
	if err := quotaStore.CheckTotalQuotaWithPreScheduled(allocRequest, toScheduleResource); err != nil {
		e.logger.Error(err, "can not schedule pod due to namespace level quotas issue")
		return allocRequest, false, false
	}

	// Check if existing + inflight nodes can satisfy the request
	filteredGPUs, _, err := e.allocator.Filter(allocRequest, gpus, false)
	if err != nil || len(filteredGPUs) == 0 {
		return allocRequest, false, true
	}
	return allocRequest, true, false
}

func (e *NodeExpander) checkGPUFitForNewNode(pod *corev1.Pod, gpus []*tfv1.GPU) bool {
	allocRequest, _, err := e.allocator.ComposeAllocationRequest(pod)
	if err != nil {
		return false
	}
	filteredGPUs, _, err := e.allocator.Filter(allocRequest, gpus, false)
	if err != nil || len(filteredGPUs) == 0 {
		return false
	}
	e.logger.Info("GPU fit for new node", "pod", pod.Name, "namespace", pod.Namespace)
	return true
}

func (e *NodeExpander) createGPUNodeClaim(ctx context.Context, pod *corev1.Pod, preparedNode *corev1.Node) error {
	owners := preparedNode.GetOwnerReferences()
	isKarpenterNodeClaim := false
	isGPUNodeClaim := false
	controlledBy := &metav1.OwnerReference{}
	for _, owner := range owners {
		controlledBy = &owner
		// Karpenter owner reference is not controller reference
		if owner.Kind == constants.KarpenterNodeClaimKind {
			isKarpenterNodeClaim = true
			break
		} else if owner.Kind == tfv1.GPUNodeClaimKind {
			isGPUNodeClaim = true
			break
		}
	}
	if !isKarpenterNodeClaim && !isGPUNodeClaim {
		e.logger.Info("node is not owned by any known provisioner, skip expansion", "node", preparedNode.Name)
		return fmt.Errorf("node is not owned by any known provisioner, skip expansion")
	}
	e.logger.Info("start expanding node from existing template node", "tmplNode", preparedNode.Name)
	if isKarpenterNodeClaim {
		// Check if controllerMeta's parent is GPUNodeClaim using unstructured object
		return e.handleKarpenterNodeClaim(ctx, pod, preparedNode, controlledBy)
	} else if isGPUNodeClaim {
		// Running in Provisioning mode, clone the parent GPUNodeClaim and apply
		e.logger.Info("node is controlled by GPUNodeClaim, cloning another to expand node", "tmplNode", preparedNode.Name)
		return e.cloneGPUNodeClaim(ctx, pod, preparedNode, controlledBy)
	}
	return nil
}

// handleKarpenterNodeClaim handles the case where the controller is a Karpenter NodeClaim
// It checks if the NodeClaim's parent is a GPUNodeClaim and handles accordingly
func (e *NodeExpander) handleKarpenterNodeClaim(ctx context.Context, pod *corev1.Pod, preparedNode *corev1.Node, controlledBy *metav1.OwnerReference) error {
	// Get the NodeClaim using unstructured object to query its parent
	nodeClaim := &karpv1.NodeClaim{}
	nodeClaimKey := client.ObjectKey{Name: controlledBy.Name}
	if err := e.client.Get(ctx, nodeClaimKey, nodeClaim); err != nil {
		e.logger.Error(err, "failed to get NodeClaim", "nodeClaimName", controlledBy.Name)
		return fmt.Errorf("failed to get NodeClaim %s: %w", controlledBy.Name, err)
	}

	// Check if the NodeClaim has owner references
	ownerRefs := nodeClaim.GetOwnerReferences()
	var nodeClaimParent *metav1.OwnerReference
	hasNodePoolParent := false

	for _, owner := range ownerRefs {
		if owner.Kind == constants.KarpenterNodePoolKind {
			hasNodePoolParent = true
		}
		if owner.Controller != nil && *owner.Controller {
			nodeClaimParent = &owner
			break
		}
	}

	if nodeClaimParent != nil && nodeClaimParent.Kind == tfv1.GPUNodeClaimKind {
		// Parent is GPUNodeClaim, clone it and let cloudprovider module create real GPUNode
		e.logger.Info("NodeClaim parent is GPUNodeClaim, cloning another to expand node",
			"nodeClaimName", controlledBy.Name, "gpuNodeClaimParent", nodeClaimParent.Name)
		return e.cloneGPUNodeClaim(ctx, pod, preparedNode, nodeClaimParent)
	} else if hasNodePoolParent {
		// owned by Karpenter node pool, create NodeClaim directly with special label identifier
		e.logger.Info("NodeClaim owned by Karpenter Pool, creating Karpenter NodeClaim to expand node",
			"nodeClaimName", controlledBy.Name)
		return e.createKarpenterNodeClaimDirect(ctx, pod, preparedNode, nodeClaim)
	} else {
		return fmt.Errorf("NodeClaim has no valid parent, can not expand node, should not happen")
	}
}

// cloneGPUNodeClaim clones a GPUNodeClaim and lets the cloudprovider module create real GPUNode
func (e *NodeExpander) cloneGPUNodeClaim(ctx context.Context, pod *corev1.Pod, preparedNode *corev1.Node, gpuNodeClaimOwner *metav1.OwnerReference) error {
	// Get the original GPUNodeClaim
	originalGPUNodeClaim := &tfv1.GPUNodeClaim{}
	gpuNodeClaimKey := client.ObjectKey{Name: gpuNodeClaimOwner.Name}
	if err := e.client.Get(ctx, gpuNodeClaimKey, originalGPUNodeClaim); err != nil {
		e.logger.Error(err, "failed to get original GPUNodeClaim", "gpuNodeClaimName", gpuNodeClaimOwner.Name)
		return fmt.Errorf("failed to get original GPUNodeClaim %s: %w", gpuNodeClaimOwner.Name, err)
	}

	// Clone the GPUNodeClaim with a new name
	if originalGPUNodeClaim.Labels == nil {
		return fmt.Errorf("original GPUNodeClaim %s has no labels, can not clone for expansion", gpuNodeClaimOwner.Name)
	}
	newGPUNodeClaim := originalGPUNodeClaim.DeepCopy()
	if newGPUNodeClaim.Labels == nil {
		newGPUNodeClaim.Labels = make(map[string]string, 2)
	}
	newGPUNodeClaim.Labels[constants.KarpenterExpansionLabel] = preparedNode.Name
	newGPUNodeClaim.Name = originalGPUNodeClaim.Labels[constants.LabelKeyOwner] + "-" + rand.String(8)

	// Create the new GPUNodeClaim
	if err := e.client.Create(ctx, newGPUNodeClaim); err != nil {
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionFailed", "failed to create new GPUNodeClaim: %v", err)
		return fmt.Errorf("failed to create new GPUNodeClaim: %w", err)
	}
	e.eventRecorder.Eventf(pod, corev1.EventTypeNormal, "NodeExpansionCompleted", "created new GPUNodeClaim for node expansion: %s", newGPUNodeClaim.Name)
	e.logger.Info("created new GPUNodeClaim for node expansion", "pod", pod.Name, "namespace", pod.Namespace, "gpuNodeClaim", newGPUNodeClaim.Name, "sourceNode", preparedNode.Name)
	return nil
}

// createKarpenterNodeClaimDirect creates a Karpenter NodeClaim directly with special label identifier
// when running GPUPool in AutoSelect mode and Karpenter manage its Nodes, no GPUNodeClaim is created
func (e *NodeExpander) createKarpenterNodeClaimDirect(ctx context.Context, pod *corev1.Pod, preparedNode *corev1.Node, nodeClaim *karpv1.NodeClaim) error {
	// Create NodeClaim from the prepared node
	newNodeClaim := &karpv1.NodeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:            preparedNode.Name,
			Labels:          make(map[string]string, 16),
			Annotations:     make(map[string]string, 4),
			OwnerReferences: nodeClaim.OwnerReferences,
		},
		Spec: nodeClaim.Spec,
	}
	// Add special label to indicate this is for node expansion of "preparedNode"
	// When GPUNode controller reconciles, check and call RemoveInFlightNode
	newNodeClaim.Labels[constants.KarpenterExpansionLabel] = newNodeClaim.Name

	// Pass through labels and annotations
	for k, v := range nodeClaim.Labels {
		if isNotAutoAddedKarpenterKeys(k) {
			newNodeClaim.Labels[k] = v
		}
	}
	for k, v := range nodeClaim.Annotations {
		if isNotAutoAddedKarpenterKeys(k) {
			newNodeClaim.Annotations[k] = v
		}
	}

	// Create the NodeClaim
	if err := e.client.Create(ctx, newNodeClaim); err != nil {
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionFailed", "failed to create new NodeClaim: %v", err)
		return fmt.Errorf("failed to create NodeClaim: %w", err)
	}

	e.eventRecorder.Eventf(pod, corev1.EventTypeNormal, "NodeExpansionCompleted", "created new NodeClaim for node expansion: %s", newNodeClaim.Name)
	e.logger.Info("created new NodeClaim for node expansion", "pod", pod.Name, "namespace", pod.Namespace, "nodeClaim", newNodeClaim.Name)
	return nil
}

func isNotAutoAddedKarpenterKeys(k string) bool {
	if strings.HasPrefix(k, "karpenter.") {
		// others are cloud provider's label and annotation, should not copy, wait for cloud provider to add
		return strings.HasPrefix(k, "karpenter.sh") || strings.HasPrefix(k, "karpenter.k8s.io")
	}
	return true
}
