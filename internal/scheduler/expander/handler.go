package expander

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator/filter"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/awslabs/operatorpkg/status"
	"github.com/samber/lo/mutable"
	corev1 "k8s.io/api/core/v1"
	errors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/tools/record"
	resourcehelper "k8s.io/component-helpers/resource"
	schedulingcorev1 "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	karpv1 "sigs.k8s.io/karpenter/pkg/apis/v1"
)

const (
	WaitingInFlightNodesPeriod = 20 * time.Second
	insufficientCapacityReason = "InsufficientCapacityError"
)

type NodeExpander struct {
	client                    client.Client
	scheduler                 *scheduler.Scheduler
	allocator                 *gpuallocator.GpuAllocator
	logger                    klog.Logger
	inFlightNodes             map[string][]*tfv1.GPU
	inFlightNodeClaims        sync.Map
	failedExpansionCandidates map[string]map[string]struct{}
	preSchedulePods           map[string]*tfv1.AllocRequest
	preScheduleTimers         map[string]*time.Timer
	eventRecorder             record.EventRecorder
	eventReader               client.Reader
	activatePod               func(*corev1.Pod)
	mu                        sync.RWMutex
	ctx                       context.Context
}

type inFlightNodeClaim struct {
	podKey       client.ObjectKey
	podUID       types.UID
	nodeClaimUID types.UID
	candidate    string
}

type schedulerFitPodAPI interface {
	UpdateNodeInfoSnapshot(ctx context.Context) error
	FindNodesThatFitPod(
		ctx context.Context,
		schedFramework framework.Framework,
		state fwk.CycleState,
		pod *corev1.Pod,
	) ([]fwk.NodeInfo, framework.Diagnosis, error)
}

func NewNodeExpander(
	ctx context.Context,
	allocator *gpuallocator.GpuAllocator,
	scheduler *scheduler.Scheduler,
	eventReader client.Reader,
	recorder record.EventRecorder,
) *NodeExpander {

	expander := &NodeExpander{
		client:                    allocator.Client,
		scheduler:                 scheduler,
		allocator:                 allocator,
		logger:                    log.FromContext(ctx).WithValues("component", "NodeExpander"),
		inFlightNodes:             make(map[string][]*tfv1.GPU, 10),
		failedExpansionCandidates: make(map[string]map[string]struct{}),
		preSchedulePods:           make(map[string]*tfv1.AllocRequest, 20),
		preScheduleTimers:         make(map[string]*time.Timer, 20),
		inFlightNodeClaims:        sync.Map{},
		eventRecorder:             recorder,
		eventReader:               eventReader,
		ctx:                       ctx,
	}
	if scheduler != nil {
		expander.activatePod = func(pod *corev1.Pod) {
			scheduler.SchedulingQueue.Activate(expander.logger, map[string]*corev1.Pod{string(pod.UID): pod})
		}
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

		removed := expander.RemovePreSchedulePod(req.PodMeta.Name, true)
		if removed {
			expander.clearFailedExpansionCandidates(req.PodMeta.Namespace, req.PodMeta.Name, req.PodMeta.UID)
			recorder.Eventf(obj, corev1.EventTypeNormal, "NodeExpansionCheck",
				"new node provisioned and pod scheduled successfully")
		}
	})

	// Start checking inFlightNodeClaims every minute to avoid stuck in inFlightNodes
	go func() {
		ticker := time.NewTicker(time.Minute)
		defer ticker.Stop()
		for {
			select {
			case <-expander.ctx.Done():
				return
			case <-ticker.C:
			}
			expander.inFlightNodeClaims.Range(func(key, value any) bool {
				karpenterNodeClaim := &karpv1.NodeClaim{}
				name := key.(string)
				if err := expander.client.Get(expander.ctx, client.ObjectKey{Name: name}, karpenterNodeClaim); err != nil {
					if errors.IsNotFound(err) {
						expander.handleTerminatedInFlightNodeClaim(name, value, expander.hasInsufficientCapacityEvent(name, value))
						expander.logger.Info("karpenter node claim not found, remove from inFlightNodeClaims and inFlightNodes", "nodeClaimName", name)
						return true
					}
					expander.logger.Error(err, "failed to get karpenter node claim", "nodeClaimName", name)
					return true
				}
				expander.cleanupInFlightNodeClaim(karpenterNodeClaim)
				return true
			})
		}
	}()

	return expander
}

func (e *NodeExpander) cleanupInFlightNodeClaim(nodeClaim *karpv1.NodeClaim) {
	if nodeClaim == nil {
		return
	}
	if nodeClaim.StatusConditions().IsTrue(status.ConditionReady) || nodeClaim.Status.NodeName != "" {
		value, _ := e.inFlightNodeClaims.Load(nodeClaim.Name)
		e.inFlightNodeClaims.Delete(nodeClaim.Name)
		e.RemoveInFlightNode(nodeClaim.Name)
		e.clearFailedExpansionCandidatesForClaim(value)
		e.logger.Info("karpenter node claim ready, remove from inFlightNodeClaims and inFlightNodes", "nodeClaimName", nodeClaim.Name)
		return
	}
	if !nodeClaim.DeletionTimestamp.IsZero() {
		value, _ := e.inFlightNodeClaims.Load(nodeClaim.Name)
		e.handleTerminatedInFlightNodeClaim(nodeClaim.Name, value, e.hasInsufficientCapacityEvent(nodeClaim.Name, value))
		e.logger.Info("karpenter node claim is deleted, remove from inFlightNodeClaims and inFlightNodes", "nodeClaimName", nodeClaim.Name)
		return
	}
	e.mu.RLock()
	_, inFlight := e.inFlightNodes[nodeClaim.Name]
	e.mu.RUnlock()
	if !inFlight {
		value, _ := e.inFlightNodeClaims.Load(nodeClaim.Name)
		e.inFlightNodeClaims.Delete(nodeClaim.Name)
		e.clearFailedExpansionCandidatesForClaim(value)
		e.logger.Info("karpenter node claim has been provisioned, remove from inFlightNodeClaims", "nodeClaimName", nodeClaim.Name)
	}
}

func (e *NodeExpander) handleTerminatedInFlightNodeClaim(name string, value any, insufficientCapacity bool) {
	e.inFlightNodeClaims.Delete(name)
	e.RemoveInFlightNode(name)

	claim, ok := value.(inFlightNodeClaim)
	if !ok {
		return
	}
	_ = e.RemovePreSchedulePod(claim.podKey.Name, true)

	pod := &corev1.Pod{}
	if err := e.client.Get(e.ctx, claim.podKey, pod); err != nil || pod.UID != claim.podUID {
		e.clearFailedExpansionCandidates(claim.podKey.Namespace, claim.podKey.Name, claim.podUID)
		return
	}
	if insufficientCapacity {
		e.addFailedExpansionCandidate(claim.podKey, claim.podUID, claim.candidate)
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionCandidateFailed",
			"Karpenter NodeClaim %s failed due to insufficient capacity; trying the next expansion preference", name)
	}
	if e.activatePod != nil {
		e.activatePod(pod)
		e.logger.Info("reactivated pod after Karpenter NodeClaim terminated",
			"pod", klog.KObj(pod), "nodeClaimName", name, "insufficientCapacity", insufficientCapacity)
	}
}

func (e *NodeExpander) hasInsufficientCapacityEvent(name string, value any) bool {
	claim, ok := value.(inFlightNodeClaim)
	if !ok || e.eventReader == nil {
		return false
	}
	matchingFields := client.MatchingFields{
		"reason":              insufficientCapacityReason,
		"involvedObject.kind": constants.KarpenterNodeClaimKind,
	}
	if claim.nodeClaimUID != "" {
		matchingFields["involvedObject.uid"] = string(claim.nodeClaimUID)
	} else {
		matchingFields["involvedObject.name"] = name
	}
	events := &corev1.EventList{}
	if err := e.eventReader.List(e.ctx, events, matchingFields, client.Limit(1)); err != nil {
		e.logger.Error(err, "failed to list events for terminated Karpenter NodeClaim", "nodeClaimName", name)
		return false
	}
	for i := range events.Items {
		event := &events.Items[i]
		if event.InvolvedObject.Kind != constants.KarpenterNodeClaimKind {
			continue
		}
		if claim.nodeClaimUID != "" && event.InvolvedObject.UID == claim.nodeClaimUID {
			return true
		}
		if claim.nodeClaimUID == "" && event.InvolvedObject.Name == name {
			return true
		}
	}
	return false
}

func (e *NodeExpander) clearFailedExpansionCandidatesForClaim(value any) {
	claim, ok := value.(inFlightNodeClaim)
	if !ok {
		return
	}
	e.clearFailedExpansionCandidates(claim.podKey.Namespace, claim.podKey.Name, claim.podUID)
}

func (e *NodeExpander) GetNodeScalerInfo() any {
	if e == nil {
		return map[string]any{}
	}
	e.mu.RLock()
	defer e.mu.RUnlock()

	// Deep-copy the live maps because the caller (HTTP handler) JSON-encodes
	// the result *after* RUnlock. Returning live references would race with
	// addInFlightNode / addPreSchedulePod / RemovePreSchedulePod and trip
	// `concurrent map iteration and map write`.
	inFlightNodesCopy := make(map[string][]*tfv1.GPU, len(e.inFlightNodes))
	for nodeName, gpus := range e.inFlightNodes {
		gpuCopies := make([]*tfv1.GPU, 0, len(gpus))
		for _, g := range gpus {
			if g == nil {
				continue
			}
			gpuCopies = append(gpuCopies, g.DeepCopy())
		}
		inFlightNodesCopy[nodeName] = gpuCopies
	}

	preSchedulePodsCopy := make(map[string]*tfv1.AllocRequest, len(e.preSchedulePods))
	for podName, req := range e.preSchedulePods {
		if req == nil {
			continue
		}
		preSchedulePodsCopy[podName] = req.DeepCopy()
	}

	inFlightNodeClaimSnapshot := make(map[string]any)
	e.inFlightNodeClaims.Range(func(key, value interface{}) bool {
		claim, ok := value.(inFlightNodeClaim)
		if !ok {
			inFlightNodeClaimSnapshot[key.(string)] = value
			return true
		}
		inFlightNodeClaimSnapshot[key.(string)] = map[string]string{
			"pod":       claim.podKey.String(),
			"podUID":    string(claim.podUID),
			"candidate": claim.candidate,
		}
		return true
	})
	failedExpansionCandidatesCopy := make(map[string][]string, len(e.failedExpansionCandidates))
	for podKey, candidates := range e.failedExpansionCandidates {
		values := make([]string, 0, len(candidates))
		for candidate := range candidates {
			values = append(values, candidate)
		}
		sort.Strings(values)
		failedExpansionCandidatesCopy[podKey] = values
	}
	return map[string]any{
		"inFlightNodes":             inFlightNodesCopy,
		"inFlightNodeClaims":        inFlightNodeClaimSnapshot,
		"failedExpansionCandidates": failedExpansionCandidatesCopy,
		"preSchedulePods":           preSchedulePodsCopy,
		"preScheduleTimerNum":       len(e.preScheduleTimers),
	}
}

func (e *NodeExpander) ProcessExpansion(ctx context.Context, pod *corev1.Pod) error {
	if pod == nil {
		return fmt.Errorf("pod cannot be nil")
	}
	// Read maps under the read lock to avoid `concurrent map read and map
	// write` against addPreSchedulePod / addInFlightNode / RemovePreSchedulePod.
	e.mu.RLock()
	_, alreadyInPreSchedule := e.preSchedulePods[pod.Name]
	inFlightCount := len(e.inFlightNodes)
	e.mu.RUnlock()

	if alreadyInPreSchedule {
		e.logger.Info("Pod already in pre-schedule state, skipping expansion check and wait for expansion", "pod", klog.KObj(pod))
		return nil
	}
	if inFlightCount >= config.GetMaxInFlightNodes() {
		e.logger.Error(nil, "Too many inFlight nodes, skipping expansion to avoid too many nodes provisioned concurrently")
		time.Sleep(WaitingInFlightNodesPeriod)
		return nil
	}

	// Step 1: Simulate scheduling without GPU plugins
	gpuNodesPassedOtherFilters, err := e.simulateSchedulingWithoutGPU(ctx, pod)
	if err != nil {
		e.eventRecorder.Eventf(pod, corev1.EventTypeNormal, "NodeExpansionCheck",
			"can not schedule on any nodes even without GPU constraints, manual check required. error: %v", err)
		e.logger.Info("Pod schedulable but no GPU nodes available, manual check required",
			"namespace", pod.Namespace, "pod", pod.Name, "error", err)
		return nil
	}
	if len(gpuNodesPassedOtherFilters) == 0 {
		e.eventRecorder.Eventf(pod, corev1.EventTypeNormal, "NodeExpansionCheck",
			"can not schedule on any nodes, manual check required, 0 fit nodes")
		e.logger.Info("Pod schedulable but no GPU nodes available, manual check required",
			"namespace", pod.Namespace, "pod", pod.Name)
		return nil
	}

	// Step 2: Check if it's a GPU resource issue, include inFlightNodes
	nodeGPUs := e.allocator.GetNodeGpuStore()
	allGpus := []*tfv1.GPU{}
	// Shuffle gpuNodes to avoid always using the same node in the same region
	mutable.Shuffle(gpuNodesPassedOtherFilters)
	for _, gpuNode := range gpuNodesPassedOtherFilters {
		if gpus, ok := nodeGPUs[gpuNode.Name]; ok {
			for _, gpu := range gpus {
				allGpus = append(allGpus, gpu)
			}
		}
	}
	// Snapshot inFlightNodes under read lock to avoid concurrent map access
	// against addInFlightNode / RemoveInFlightNode.
	e.mu.RLock()
	inFlightGPUSnapshot := make(map[string]*tfv1.GPU, len(e.inFlightNodes)*4)
	for _, inFlightGPUs := range e.inFlightNodes {
		for _, gpu := range inFlightGPUs {
			snapshot := gpu.DeepCopy()
			inFlightGPUSnapshot[gpu.Name] = snapshot
			allGpus = append(allGpus, snapshot)
		}
	}
	e.mu.RUnlock()
	if len(allGpus) == 0 {
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionCheck",
			"all schedulable nodes are none GPU nodes, manual check required")
		e.logger.Info("No GPU nodes can put the Pod, manual check required", "namespace", pod.Namespace, "pod", pod.Name)
		return nil
	}

	// Step 3: Check if it's a GPU resource issue, include inFlightNodes
	allocRequest, satisfied, isResourceIssue, onlyCanBeFlightGPU := e.checkGPUFitWithInflightNodes(pod, allGpus, inFlightGPUSnapshot)
	if satisfied {
		if onlyCanBeFlightGPU {
			e.addPreSchedulePod(allocRequest)
			// Pod should be scheduled after new node is provisioned
			e.eventRecorder.Eventf(pod, corev1.EventTypeNormal, "NodeExpansionCheck",
				"fit in-flight GPU resources, pod should be scheduled after new node is provisioned")
		} else {
			// GPU free-up during expansion, or satisfied by in-flight nodes, pod can be scheduled now or whiles later
			e.eventRecorder.Eventf(pod, corev1.EventTypeNormal, "NodeExpansionCheck",
				"fit GPU resources, pod should be scheduled now or whiles later on existing/provisioning nodes")
		}
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
	for _, gpuNode := range gpuNodesPassedOtherFilters {
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

		e.addInFlightNode(preparedNode, preparedGPUs)
		e.addPreSchedulePod(allocRequest)
		preScheduled = true
		break
	}
	if !preScheduled {
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionFailed", "failed to satisfy the pending pod, no potential GPU nodes can fit")
		return fmt.Errorf("failed to satisfy the pending pod, no potential GPU nodes can fit")
	}
	return nil
}

func expansionPodKey(namespace, name string, uid types.UID) string {
	return expansionPodPrefix(namespace, name) + string(uid)
}

func expansionPodPrefix(namespace, name string) string {
	return namespace + "/" + name + "/"
}

func (e *NodeExpander) addFailedExpansionCandidate(podKey client.ObjectKey, podUID types.UID, candidate string) {
	if candidate == "" {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	key := expansionPodKey(podKey.Namespace, podKey.Name, podUID)
	if e.failedExpansionCandidates[key] == nil {
		e.failedExpansionCandidates[key] = make(map[string]struct{})
	}
	e.failedExpansionCandidates[key][candidate] = struct{}{}
}

func (e *NodeExpander) clearFailedExpansionCandidates(namespace, name string, uid types.UID) {
	e.mu.Lock()
	delete(e.failedExpansionCandidates, expansionPodKey(namespace, name, uid))
	e.mu.Unlock()
}

// ClearFailedExpansionCandidatesForPod removes all candidate history for a
// deleted Pod, including entries left by an older UID with the same name.
func (e *NodeExpander) ClearFailedExpansionCandidatesForPod(namespace, name string) {
	if e == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	prefix := expansionPodPrefix(namespace, name)
	for key := range e.failedExpansionCandidates {
		if strings.HasPrefix(key, prefix) {
			delete(e.failedExpansionCandidates, key)
		}
	}
}

const relaxedExpansionCandidate = "<relaxed>"

type expansionPreference struct {
	candidate    string
	requirements map[string][]string
}

func preferredExpansionCandidates(pod *corev1.Pod) []expansionPreference {
	if pod == nil || pod.Spec.Affinity == nil || pod.Spec.Affinity.NodeAffinity == nil {
		return []expansionPreference{{candidate: relaxedExpansionCandidate}}
	}
	terms := append([]corev1.PreferredSchedulingTerm(nil),
		pod.Spec.Affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution...)
	sort.SliceStable(terms, func(i, j int) bool {
		return terms[i].Weight > terms[j].Weight
	})

	candidates := make([]expansionPreference, 0, len(terms)+1)
	seen := make(map[string]struct{}, len(terms)+1)
	for _, term := range terms {
		requirements := make(map[string][]string)
		for _, expression := range term.Preference.MatchExpressions {
			if expression.Operator != corev1.NodeSelectorOpIn ||
				(expression.Key != corev1.LabelInstanceTypeStable &&
					expression.Key != corev1.LabelTopologyZone &&
					expression.Key != corev1.LabelTopologyRegion) {
				continue
			}
			requirements[expression.Key] = append([]string(nil), expression.Values...)
		}
		candidate := expansionRequirementsKey(requirements)
		if candidate == "" {
			continue
		}
		if _, exists := seen[candidate]; exists {
			continue
		}
		seen[candidate] = struct{}{}
		candidates = append(candidates, expansionPreference{candidate: candidate, requirements: requirements})
	}
	candidates = append(candidates, expansionPreference{candidate: relaxedExpansionCandidate})
	return candidates
}

func expansionRequirementsKey(requirements map[string][]string) string {
	keys := []string{corev1.LabelInstanceTypeStable, corev1.LabelTopologyZone, corev1.LabelTopologyRegion}
	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		values := append([]string(nil), requirements[key]...)
		if len(values) == 0 {
			continue
		}
		sort.Strings(values)
		parts = append(parts, key+"="+strings.Join(values, ","))
	}
	return strings.Join(parts, ";")
}

func (e *NodeExpander) expansionPreferencesToTry(pod *corev1.Pod) []expansionPreference {
	candidates := preferredExpansionCandidates(pod)
	podKey := expansionPodKey(pod.Namespace, pod.Name, pod.UID)
	prefix := expansionPodPrefix(pod.Namespace, pod.Name)

	e.mu.Lock()
	defer e.mu.Unlock()
	// A recreated Pod starts a new expansion attempt. Drop candidate history
	// from older UIDs while preserving the current Pod's failures.
	for key := range e.failedExpansionCandidates {
		if strings.HasPrefix(key, prefix) && key != podKey {
			delete(e.failedExpansionCandidates, key)
		}
	}
	failed := e.failedExpansionCandidates[podKey]
	remaining := make([]expansionPreference, 0, len(candidates))
	for _, candidate := range candidates {
		if _, exists := failed[candidate.candidate]; !exists {
			remaining = append(remaining, candidate)
		}
	}
	return remaining
}

func (e *NodeExpander) addInFlightNode(node *corev1.Node, gpus []*tfv1.GPU) {
	e.mu.Lock()
	e.inFlightNodes[node.Name] = gpus
	e.mu.Unlock()
}

func (e *NodeExpander) addPreSchedulePod(allocRequest *tfv1.AllocRequest) {
	e.mu.Lock()
	defer e.mu.Unlock()
	podMeta := allocRequest.PodMeta
	e.preSchedulePods[podMeta.Name] = allocRequest
	// Add timer for each pre-scheduled pod, if not scheduled for 10 minutes, make warning event and remove from mem
	timer := time.AfterFunc((10 * time.Minute), func() {
		currentPod := &corev1.Pod{}
		err := e.client.Get(e.ctx, client.ObjectKey{Name: podMeta.Name, Namespace: podMeta.Namespace}, currentPod)
		if err != nil {
			if errors.IsNotFound(err) {
				_ = e.RemovePreSchedulePod(podMeta.Name, false)
				e.clearFailedExpansionCandidates(podMeta.Namespace, podMeta.Name, podMeta.UID)
			}
			e.logger.Error(err, "failed to get pod for node expansion check",
				"namespace", podMeta.Namespace, "pod", podMeta.Name)
			_ = e.RemovePreSchedulePod(podMeta.Name, false)
			return
		}
		if !currentPod.DeletionTimestamp.IsZero() {
			_ = e.RemovePreSchedulePod(podMeta.Name, false)
			e.clearFailedExpansionCandidates(podMeta.Namespace, podMeta.Name, podMeta.UID)
			return
		}
		if currentPod.Spec.NodeName != "" {
			// already scheduled, remove pre-scheduled pod
			e.eventRecorder.Eventf(currentPod, corev1.EventTypeNormal, "NodeExpansionCheck",
				"new node provisioned and pod scheduled successfully")
			e.logger.Info("new node provisioned and pod scheduled successfully",
				"namespace", podMeta.Namespace, "pod", podMeta.Name)
			_ = e.RemovePreSchedulePod(podMeta.Name, false)
			e.clearFailedExpansionCandidates(podMeta.Namespace, podMeta.Name, podMeta.UID)
		} else {
			// not scheduled, record warning event and remove pre-scheduled pod
			e.eventRecorder.Eventf(currentPod, corev1.EventTypeWarning, "NodeExpansionCheck",
				"failed to schedule pod after 10 minutes")
			e.logger.Info("failed to schedule pod after 10 minutes",
				"namespace", podMeta.Namespace, "pod", podMeta.Name)
			_ = e.RemovePreSchedulePod(podMeta.Name, false)
		}
	})
	e.preScheduleTimers[podMeta.Name] = timer
}

func (e *NodeExpander) RemoveInFlightNode(nodeName string) {
	if e == nil {
		return
	}
	e.mu.Lock()
	if _, ok := e.inFlightNodes[nodeName]; ok {
		delete(e.inFlightNodes, nodeName)
		e.logger.Info("Removed in-flight node", "node", nodeName, "remaining inflight nodes", len(e.inFlightNodes))
	}
	e.mu.Unlock()
}

func (e *NodeExpander) RemovePreSchedulePod(podName string, stopTimer bool) bool {
	if e == nil {
		return false
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if stopTimer {
		if timer, ok := e.preScheduleTimers[podName]; ok {
			timer.Stop()
		}
	}
	delete(e.preScheduleTimers, podName)

	if _, ok := e.preSchedulePods[podName]; ok {
		delete(e.preSchedulePods, podName)
		e.logger.Info("Removed pre-scheduled pod", "pod", podName, "remaining pre-scheduled pods", len(e.preSchedulePods))
		return true
	}
	return false
}

func (e *NodeExpander) prepareNewNodesForScheduleAttempt(
	templateNode *corev1.Node, templateGPUs map[string]*tfv1.GPU,
) (*corev1.Node, []*tfv1.GPU) {
	newPreparedNode := templateNode.DeepCopy()
	newPreparedNode.Name = constants.TensorFusionSystemName + "-" + rand.String(10)
	if newPreparedNode.Labels != nil {
		newPreparedNode.Labels[constants.KubernetesHostNameLabel] = newPreparedNode.Name
	}
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
	// FindNodesThatFitPod is from our scheduler vendor patch.
	// If patch is missing, return a clear remediation message.
	if !utils.IsTensorFusionPod(pod) {
		return nil, fmt.Errorf("pod to check expansion is not a tensor fusion worker pod: %s", pod.Name)
	}
	fitAPI, ok := any(e.scheduler).(schedulerFitPodAPI)
	if !ok {
		return nil, fmt.Errorf("scheduler patch missing FindNodesThatFitPod/UpdateNodeInfoSnapshot: run `make vendor` or `bash scripts/patch-scheduler.sh`")
	}
	if err := fitAPI.UpdateNodeInfoSnapshot(ctx); err != nil {
		return nil, fmt.Errorf("refresh scheduler snapshot before expansion simulation: %w", err)
	}
	delete(pod.Labels, constants.LabelComponent)
	scheduleResult, _, err := fitAPI.FindNodesThatFitPod(ctx, fwkInstance, state, pod)
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

func (e *NodeExpander) checkGPUFitWithInflightNodes(pod *corev1.Pod, potentialGpus []*tfv1.GPU, inflightSnapshot map[string]*tfv1.GPU) (
	allocRequest *tfv1.AllocRequest,
	satisfied bool,
	isResourceIssue bool,
	onlyCanBeFlightGPU bool,
) {
	// NOTE: a known issue, if cpu/mem not enough or affinity not satisfied for pre-scheduled pods inside inFlightNodes,
	// it will not be considered, when inflight created and the Pod still not be able to schedule on new node,
	// wait next scheduling check and node expansion period (k8s move UnscheduleQueue to ActiveQueue every 5 minutes)
	//
	// Snapshot pre-scheduled pods under the read lock so that the loop body —
	// which calls RemovePreSchedulePod (write lock) — does not race with the
	// iteration. Without the snapshot Go would fatal with
	// `concurrent map read and map write`.
	e.mu.RLock()
	preScheduleSnapshot := make([]*tfv1.AllocRequest, 0, len(e.preSchedulePods))
	for _, alloc := range e.preSchedulePods {
		preScheduleSnapshot = append(preScheduleSnapshot, alloc)
	}
	e.mu.RUnlock()

	for _, alloc := range preScheduleSnapshot {
		preScheduledPodPreAllocated := false
		for _, gpu := range inflightSnapshot {
			reqTflops := alloc.Request.Tflops
			if !alloc.Request.ComputePercent.IsZero() {
				reqTflops = *utils.ComputePercentToTflops(gpu.Status.Capacity.Tflops, alloc.Request)
			}
			if gpu.Status.Available.Tflops.Cmp(reqTflops) >= 0 &&
				gpu.Status.Available.Vram.Cmp(alloc.Request.Vram) >= 0 {
				gpu.Status.Available.Tflops.Sub(reqTflops)
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
			_ = e.RemovePreSchedulePod(alloc.PodMeta.Name, true)
		}
	}

	// Get allocation request
	e.mu.RLock()
	defer e.mu.RUnlock()
	allocRequest, _, err := e.allocator.ComposeAllocationRequest(pod)
	if err != nil {
		return nil, false, true, false
	}

	quotaStore := e.allocator.GetQuotaStore()
	if err := quotaStore.CheckSingleQuotaAvailable(allocRequest); err != nil {
		e.logger.Error(err, "can not schedule pod due to single workload quotas issue")
		return allocRequest, false, false, false
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
		return allocRequest, false, false, false
	}

	// Check if existing + inflight nodes can satisfy the request
	filteredGPUs, _, err := e.allocator.Filter(allocRequest, potentialGpus, false)
	if err != nil || len(filteredGPUs) == 0 {
		return allocRequest, false, true, false
	}

	onlyCanBeFlightGPU = true
	for _, gpu := range filteredGPUs {
		if _, ok := inflightSnapshot[gpu.Name]; !ok {
			onlyCanBeFlightGPU = false
			break
		}
	}
	return allocRequest, true, false, onlyCanBeFlightGPU
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
	e.logger.Info("start expanding node from existing template node", "newNodeClaimName", preparedNode.Name)
	if isKarpenterNodeClaim {
		// Check if controllerMeta's parent is GPUNodeClaim using unstructured object
		return e.handleKarpenterNodeClaim(ctx, pod, preparedNode, controlledBy)
	} else if isGPUNodeClaim {
		// Running in Provisioning mode, clone the parent GPUNodeClaim and apply
		e.logger.Info("node is controlled by GPUNodeClaim, cloning another to expand node", "newNode", preparedNode.Name)
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
			"controlledBy", controlledBy.Name, "gpuNodeClaimParent", nodeClaimParent.Name)
		return e.cloneGPUNodeClaim(ctx, pod, preparedNode, nodeClaimParent)
	} else if hasNodePoolParent {
		// owned by Karpenter node pool, create NodeClaim directly with special label identifier
		e.logger.Info("NodeClaim owned by Karpenter Pool, creating Karpenter NodeClaim to expand node",
			"controlledBy", controlledBy.Name)
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
	spec := nodeClaim.DeepCopy().Spec
	spec.Resources.Requests = nodeClaimRequestsForPod(pod)
	podRequirements := podKarpenterRequirementValues(pod, preparedNode)
	labels := make(map[string]string, 4)
	for _, owner := range nodeClaim.OwnerReferences {
		if owner.Kind != constants.KarpenterNodePoolKind {
			continue
		}
		nodePool := &karpv1.NodePool{}
		if err := e.client.Get(ctx, client.ObjectKey{Name: owner.Name}, nodePool); err != nil {
			return fmt.Errorf("failed to get Karpenter NodePool %s: %w", owner.Name, err)
		}
		poolRequirements := append([]karpv1.NodeSelectorRequirementWithMinValues(nil), nodePool.Spec.Template.Spec.Requirements...)
		poolKeys := make(map[string]struct{}, len(poolRequirements))
		for _, requirement := range poolRequirements {
			poolKeys[requirement.Key] = struct{}{}
		}
		for key, value := range nodePool.Spec.Template.Labels {
			labels[key] = value
		}
		labels[karpv1.NodePoolLabelKey] = nodePool.Name
		if nodeClassRef := nodePool.Spec.Template.Spec.NodeClassRef; nodeClassRef != nil {
			labels[karpv1.NodeClassLabelKey(nodeClassRef.GroupKind())] = nodeClassRef.Name
		}
		for key := range labels {
			// Karpenter layers NodeClaim labels into its requirements. Do not
			// append a stale requirement for the same key from the source claim.
			poolKeys[key] = struct{}{}
		}
		for _, requirement := range spec.Requirements {
			if _, exists := poolKeys[requirement.Key]; exists {
				continue
			}
			// A source NodeClaim contains provider-resolved instance type, zone,
			// and possibly region values for the existing node. The replacement
			// must use NodePool alternatives and let Karpenter choose when the
			// NodePool does not constrain a location.
			if requirement.Key == corev1.LabelInstanceTypeStable ||
				requirement.Key == corev1.LabelTopologyZone ||
				requirement.Key == corev1.LabelTopologyRegion {
				continue
			}
			poolRequirements = append(poolRequirements, requirement)
		}
		spec.Requirements = poolRequirements
		break
	}
	labelRequirements := make(map[string][]string, len(labels))
	for key, value := range labels {
		labelRequirements[key] = []string{value}
	}
	if !mergeKarpenterRequirements(&spec, labelRequirements) {
		return fmt.Errorf("karpenter node pool labels do not intersect with its requirements")
	}
	if !mergeKarpenterRequirements(&spec, podRequirements) {
		return fmt.Errorf("pod requirements do not intersect with Karpenter NodePool requirements")
	}

	var preference expansionPreference
	preferenceFound := false
	for _, candidate := range e.expansionPreferencesToTry(pod) {
		candidateSpec := (&karpv1.NodeClaim{Spec: spec}).DeepCopy().Spec
		if !mergeKarpenterRequirements(&candidateSpec, candidate.requirements) {
			continue
		}
		spec = candidateSpec
		preference = candidate
		preferenceFound = true
		break
	}
	if !preferenceFound {
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionCandidatesExhausted",
			"all Karpenter expansion preferences, including the relaxed fallback, have failed")
		return fmt.Errorf("all Karpenter expansion preferences have failed for pod %s/%s", pod.Namespace, pod.Name)
	}

	// Create NodeClaim from the prepared node
	newNodeClaim := &karpv1.NodeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:            preparedNode.Name,
			Labels:          labels,
			Annotations:     make(map[string]string, 4),
			OwnerReferences: nodeClaim.OwnerReferences,
		},
		Spec: spec,
	}
	// Add special label to indicate this is for node expansion of "preparedNode"
	// When GPUNode controller reconciles, check and call RemoveInFlightNode
	newNodeClaim.Labels[constants.KarpenterExpansionLabel] = newNodeClaim.Name

	// Keep source annotations, but rebuild labels from the NodePool template.
	// Source NodeClaim labels include provider-resolved instance properties that
	// would constrain the replacement to the same offering.
	for k, v := range nodeClaim.Annotations {
		// Expansion copies should not inherit do-not-disrupt protection; otherwise reclaimed nodes can get stuck.
		if shouldSkipAnnotationCopy(k, v) {
			continue
		}
		if isNotAutoAddedKarpenterKeys(k) {
			newNodeClaim.Annotations[k] = v
		}
	}

	// Create the NodeClaim
	if err := e.client.Create(ctx, newNodeClaim); err != nil {
		e.eventRecorder.Eventf(pod, corev1.EventTypeWarning, "NodeExpansionFailed", "failed to create new NodeClaim: %v", err)
		return fmt.Errorf("failed to create NodeClaim: %w", err)
	}
	e.inFlightNodeClaims.Store(newNodeClaim.Name, inFlightNodeClaim{
		podKey:       client.ObjectKey{Namespace: pod.Namespace, Name: pod.Name},
		podUID:       pod.UID,
		nodeClaimUID: newNodeClaim.UID,
		candidate:    preference.candidate,
	})
	e.eventRecorder.Eventf(pod, corev1.EventTypeNormal, "NodeExpansionCompleted", "created new NodeClaim for node expansion: %s", newNodeClaim.Name)
	e.logger.Info("created new NodeClaim for node expansion", "pod", pod.Name, "namespace", pod.Namespace, "nodeClaim", newNodeClaim.Name)

	return nil
}

func nodeClaimRequestsForPod(pod *corev1.Pod) corev1.ResourceList {
	requests := resourcehelper.PodRequests(pod, resourcehelper.PodResourcesOptions{})
	for name := range requests {
		if strings.HasPrefix(string(name), constants.Domain+"/") {
			delete(requests, name)
		}
	}
	requests[corev1.ResourcePods] = resource.MustParse("1")
	return requests
}

func podKarpenterRequirementValues(pod *corev1.Pod, preparedNode *corev1.Node) map[string][]string {
	valuesByKey := make(map[string][]string)
	if pod == nil {
		return valuesByKey
	}
	setValues := func(key string, values ...string) {
		if key != corev1.LabelInstanceTypeStable &&
			key != corev1.LabelTopologyZone &&
			key != corev1.LabelTopologyRegion {
			return
		}
		unique := make([]string, 0, len(values))
		seen := make(map[string]struct{}, len(values))
		for _, value := range values {
			if value != "" {
				if _, exists := seen[value]; !exists {
					unique = append(unique, value)
					seen[value] = struct{}{}
				}
			}
		}
		if current, exists := valuesByKey[key]; exists {
			allowed := make(map[string]struct{}, len(unique))
			for _, value := range unique {
				allowed[value] = struct{}{}
			}
			intersection := make([]string, 0, len(current))
			for _, value := range current {
				if _, ok := allowed[value]; ok {
					intersection = append(intersection, value)
				}
			}
			valuesByKey[key] = intersection
			return
		}
		valuesByKey[key] = unique
	}
	for key, value := range pod.Spec.NodeSelector {
		setValues(key, value)
	}
	if pod.Spec.Affinity != nil && pod.Spec.Affinity.NodeAffinity != nil &&
		pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
		terms := pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
		for i := range terms {
			if preparedNode != nil {
				matches, err := schedulingcorev1.MatchNodeSelectorTerms(preparedNode, &corev1.NodeSelector{
					NodeSelectorTerms: []corev1.NodeSelectorTerm{terms[i]},
				})
				if err != nil || !matches {
					continue
				}
			}
			for _, expression := range terms[i].MatchExpressions {
				if expression.Operator == corev1.NodeSelectorOpIn {
					setValues(expression.Key, expression.Values...)
				}
			}
			// NodeSelectorTerms are ORed. A single NodeClaim expresses an AND
			// set of requirements, so use the term represented by the prepared
			// node instead of broadening the claim with values from all terms.
			break
		}
	}
	return valuesByKey
}

func mergeKarpenterRequirements(spec *karpv1.NodeClaimSpec, valuesByKey map[string][]string) bool {
	if spec == nil {
		return false
	}
	merged := append([]karpv1.NodeSelectorRequirementWithMinValues(nil), spec.Requirements...)
	for i := range merged {
		merged[i].Values = append([]string(nil), merged[i].Values...)
	}
	for key, values := range valuesByKey {
		values = uniqueStrings(values)
		if len(values) == 0 {
			continue
		}
		matched := false
		minValues := 0
		remaining := make([]karpv1.NodeSelectorRequirementWithMinValues, 0, len(merged)+1)
		for _, requirement := range merged {
			if requirement.Key != key {
				remaining = append(remaining, requirement)
				continue
			}
			matched = true
			var compatible bool
			values, compatible = intersectKarpenterRequirementValues(requirement, values)
			if !compatible {
				return false
			}
			if requirement.MinValues != nil && *requirement.MinValues > minValues {
				minValues = *requirement.MinValues
			}
		}
		if minValues > len(values) {
			return false
		}
		sort.Strings(values)
		mergedRequirement := karpv1.NodeSelectorRequirementWithMinValues{
			Key: key, Operator: corev1.NodeSelectorOpIn, Values: values,
		}
		if matched && minValues > 0 {
			mergedRequirement.MinValues = &minValues
		}
		merged = append(remaining, mergedRequirement)
	}
	spec.Requirements = merged
	sort.Slice(spec.Requirements, func(i, j int) bool {
		if spec.Requirements[i].Key == spec.Requirements[j].Key {
			return spec.Requirements[i].Operator < spec.Requirements[j].Operator
		}
		return spec.Requirements[i].Key < spec.Requirements[j].Key
	})
	return true
}

func intersectKarpenterRequirementValues(requirement karpv1.NodeSelectorRequirementWithMinValues, values []string) ([]string, bool) {
	allowed := make(map[string]struct{}, len(requirement.Values))
	for _, value := range requirement.Values {
		allowed[value] = struct{}{}
	}
	result := make([]string, 0, len(values))
	switch requirement.Operator {
	case corev1.NodeSelectorOpIn:
		for _, value := range values {
			if _, exists := allowed[value]; exists {
				result = append(result, value)
			}
		}
	case corev1.NodeSelectorOpNotIn:
		for _, value := range values {
			if _, excluded := allowed[value]; !excluded {
				result = append(result, value)
			}
		}
	case corev1.NodeSelectorOpExists:
		result = append(result, values...)
	case corev1.NodeSelectorOpDoesNotExist:
		return nil, false
	case corev1.NodeSelectorOpGt, corev1.NodeSelectorOpLt, karpv1.NodeSelectorOpGte, karpv1.NodeSelectorOpLte:
		if len(requirement.Values) != 1 {
			return nil, false
		}
		boundary, err := strconv.Atoi(requirement.Values[0])
		if err != nil {
			return nil, false
		}
		for _, value := range values {
			number, err := strconv.Atoi(value)
			if err != nil {
				continue
			}
			if requirementMatchesNumber(requirement.Operator, number, boundary) {
				result = append(result, value)
			}
		}
	default:
		return nil, false
	}
	return result, len(result) > 0
}

func requirementMatchesNumber(operator corev1.NodeSelectorOperator, value, boundary int) bool {
	switch operator {
	case corev1.NodeSelectorOpGt:
		return value > boundary
	case corev1.NodeSelectorOpLt:
		return value < boundary
	case karpv1.NodeSelectorOpGte:
		return value >= boundary
	case karpv1.NodeSelectorOpLte:
		return value <= boundary
	default:
		return false
	}
}

func uniqueStrings(values []string) []string {
	unique := make([]string, 0, len(values))
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if value == "" {
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		unique = append(unique, value)
	}
	return unique
}

func isNotAutoAddedKarpenterKeys(k string) bool {
	if strings.HasPrefix(k, "karpenter.") {
		// others are cloud provider's label and annotation, should not copy, wait for cloud provider to add
		return strings.HasPrefix(k, "karpenter.sh") || strings.HasPrefix(k, "karpenter.k8s.io")
	}
	return true
}

func shouldSkipAnnotationCopy(k, v string) bool {
	return k == "karpenter.sh/do-not-disrupt" && strings.EqualFold(v, "true")
}
