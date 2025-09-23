package expander

import (
	"context"
	"fmt"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/karpenter"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator/filter"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type NodeExpander struct {
	client          client.Client
	scheduler       *scheduler.Scheduler
	allocator       *gpuallocator.GpuAllocator
	logger          klog.Logger
	inFlightNodes   map[string][]*tfv1.GPU
	preSchedulePods map[string]*corev1.Pod
	eventRecorder   record.EventRecorder
	mu              sync.RWMutex
}

func NewNodeExpander(
	ctx context.Context,
	allocator *gpuallocator.GpuAllocator,
	scheduler *scheduler.Scheduler,
	recorder record.EventRecorder,
) *NodeExpander {

	expander := &NodeExpander{
		client:          allocator.Client,
		scheduler:       scheduler,
		allocator:       allocator,
		logger:          log.FromContext(ctx).WithValues("component", "NodeExpander"),
		inFlightNodes:   make(map[string][]*tfv1.GPU, 10),
		preSchedulePods: make(map[string]*corev1.Pod, 20),
		eventRecorder:   recorder,
	}
	allocator.RegisterBindHandler(func(req *tfv1.AllocRequest) {
		expander.RemovePreSchedulePod(req.PodMeta.Name)
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

	// Step 1: Simulate scheduling without GPU plugins
	gpuNodes, err := e.simulateSchedulingWithoutGPU(ctx, pod)
	if err != nil {
		e.eventRecorder.Eventf(pod, corev1.EventTypeNormal, "NodeExpansionCheck",
			"can not schedule on any nodes even without GPU constraints, manual check required. error: %w", err)
		e.logger.Info("Pod schedulable but no GPU nodes available, manual check required",
			"namespace", pod.Namespace, "pod", pod.Name, "error", err)
		return nil
	}
	if len(gpuNodes) == 0 {
		e.eventRecorder.Eventf(pod, corev1.EventTypeNormal, "NodeExpansionCheck",
			"can not schedule on any nodes, manual check required, 0 fit nodes")
		e.logger.Info("Pod schedulable but no GPU nodes available, manual check required",
			"namespace", pod.Namespace, "pod", pod.Name)
		return nil
	}

	// Step 2: Check if it's a GPU resource issue, include inFlightNodes
	nodeGPUs := e.allocator.GetNodeGpuStore()
	allGpus := []*tfv1.GPU{}
	for _, gpuNode := range gpuNodes {
		if gpus, ok := nodeGPUs[gpuNode.Name]; ok {
			for _, gpu := range gpus {
				allGpus = append(allGpus, gpu)
			}
		}
	}
	for _, inFlightGPUs := range e.inFlightNodes {
		allGpus = append(allGpus, inFlightGPUs...)
	}
	if len(allGpus) == 0 {
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionCheck",
			"all schedulable nodes are none GPU nodes, manual check required")
		e.logger.Info("No GPU nodes can put the Pod, manual check required", "namespace", pod.Namespace, "pod", pod.Name)
		return nil
	}

	// Step 3: Check if it's a GPU resource issue, include inFlightNodes
	satisfied, isResourceIssue := e.checkGPUFitWithInflightNodes(pod, allGpus)
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
	for _, gpuNode := range gpuNodes {
		preparedNode, preparedGPUs, err := e.tryExpandGPUNodesFromTemplate(ctx, pod, gpuNode, nodeGPUs[gpuNode.Name])
		if err != nil {
			return fmt.Errorf("node expansion failed: %w", err)
		}
		if !e.checkGPUFitForNewNode(pod, preparedGPUs) {
			continue
		}

		err = e.createGPUNodeClaim(ctx, pod, preparedNode)
		if err != nil {
			return err
		}

		e.addInFlightNodeAndPreSchedulePod(pod, preparedNode, preparedGPUs)
		preScheduled = true
		break
	}
	if !preScheduled {
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionFailed", "failed to satisfy the pending pod, no potential GPU nodes can fit")
		return fmt.Errorf("failed to satisfy the pending pod, no potential GPU nodes can fit")
	}
	return nil
}

func (e *NodeExpander) addInFlightNodeAndPreSchedulePod(pod *corev1.Pod, node *corev1.Node, gpus []*tfv1.GPU) {
	e.mu.Lock()
	e.inFlightNodes[node.Name] = gpus
	e.preSchedulePods[pod.Name] = pod
	// TODO add timer for each pre-scheduled pod, if not scheduled for 10 minutes, make warning event
	e.mu.Unlock()
}

func (e *NodeExpander) RemoveInFlightNode(nodeName string) {
	if e == nil {
		return
	}
	e.mu.Lock()
	delete(e.inFlightNodes, nodeName)
	e.mu.Unlock()
}

func (e *NodeExpander) RemovePreSchedulePod(podName string) {
	if e == nil {
		return
	}
	e.mu.Lock()
	delete(e.preSchedulePods, podName)
	e.mu.Unlock()
}

func (e *NodeExpander) tryExpandGPUNodesFromTemplate(
	ctx context.Context, pod *corev1.Pod,
	templateNode *corev1.Node, templateGPUs map[string]*tfv1.GPU,
) (*corev1.Node, []*tfv1.GPU, error) {
	// TODO: copy the resource and make a new node with new GPUs
	return nil, nil, nil
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

	// Disable the tensor fusion label to simulate scheduling without GPU plugins
	// NOTE: must apply patch after `go mod vendor`, FindNodesThatFitPod is not exported from Kubernetes
	// Run `git apply ./patches/scheduler-sched-one.patch` once or `bash scripts/patch-scheduler.sh`
	pod.Labels[constants.TensorFusionEnabledLabelKey] = constants.FalseStringValue
	scheduleResult, _, err := e.scheduler.FindNodesThatFitPod(ctx, fwkInstance, state, pod)
	pod.Labels[constants.TensorFusionEnabledLabelKey] = constants.TrueStringValue
	if len(scheduleResult) == 0 {
		return nil, err
	}
	result := []*corev1.Node{}
	for _, nodeInfo := range scheduleResult {
		result = append(result, nodeInfo.Node())
	}
	return result, nil
}

func (e *NodeExpander) checkGPUFitWithInflightNodes(pod *corev1.Pod, gpus []*tfv1.GPU) (satisfied bool, isResourceIssue bool) {

	// TODO reduce available capacity by looping pre-scheduled Pods
	// e.preSchedulePods
	// TODO consider cpu, mem resources for all pre-scheduled pods, ignore other affinity antiAffinity
	// if not scheduled after 10 minutes, make warning event, and remove from PreScheduled queue,
	// wait next Expansion to check why
	// TODO inFlight node can not be greater than a threshold, eg 15, to avoid exceeded cloud vendor limit
	// If too many inflight node during process queue, sleep interval and wait next Expansion

	// Get allocation request
	e.mu.RLock()
	defer e.mu.RUnlock()
	allocRequest, _, err := e.allocator.ComposeAllocationRequest(pod)
	if err != nil {
		return false, true
	}

	quotaStore := e.allocator.GetQuotaStore()
	if err := quotaStore.CheckSingleQuotaAvailable(allocRequest); err != nil {
		e.logger.Error(err, "can not schedule pod due to single workload quotas issue")
		return false, false
	}

	// TODO change nil to sum of all pre-scheduled pods
	if err := quotaStore.CheckTotalQuotaWithPreScheduled(allocRequest, nil); err != nil {
		e.logger.Error(err, "can not schedule pod due to namespace level quotas issue")
		return false, false
	}

	// Check if existing + inflight nodes can satisfy the request
	filteredGPUs, _, err := e.allocator.Filter(allocRequest, gpus, false)
	if err != nil || len(filteredGPUs) == 0 {
		return false, true
	}
	return true, false
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

func (e *NodeExpander) createGPUNodeClaim(ctx context.Context, pod *corev1.Pod, tmplNode *corev1.Node) error {
	// TODO: Create new NodeClaim based on the selected node
	// Judge different node pool management mode, auto detect the best way to expand node
	newNodeClaim, err := karpenter.CreateNodeClaimFromNode(ctx, e.client, tmplNode)
	if err != nil {
		return fmt.Errorf("failed to create NodeClaim: %w", err)
	}

	if err := e.client.Create(ctx, newNodeClaim); err != nil {
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionFailed", "failed to create new GPUNodeClaim for pod %w", err)
		return fmt.Errorf("failed to create NodeClaim: %w", err)
	}
	log.FromContext(ctx).Info("created NodeClaim", "nodeClaim", newNodeClaim.Name)
	return nil
}
