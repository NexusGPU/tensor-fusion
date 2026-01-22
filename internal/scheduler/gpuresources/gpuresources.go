package gpuresources

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gang"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/indexallocator"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/quota"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const Name = "GPUResourcesFit"
const CycleStateAllocateRequest = "allocateRequest"
const CycleStateGPUSchedulingResult = "gpuSchedulingResult"
const SchedulerSimulationKey = "schedulerSimulation"

var _ framework.PreFilterPlugin = &GPUFit{}
var _ framework.FilterPlugin = &GPUFit{}
var _ framework.ScorePlugin = &GPUFit{}
var _ framework.PermitPlugin = &GPUFit{}
var _ framework.ReservePlugin = &GPUFit{}
var _ framework.PostBindPlugin = &GPUFit{}
var _ framework.EnqueueExtensions = &GPUFit{}

type GPUFit struct {
	logger         *klog.Logger
	fh             framework.Handle
	client         client.Client
	allocator      *gpuallocator.GpuAllocator
	indexAllocator *indexallocator.IndexAllocator
	gangManager    *gang.Manager
	ctx            context.Context
	cfg            *config.GPUFitConfig
}

type GPUSchedulingStateData struct {
	// PreFilter stage compose valid nodes and their GPUs
	NodeGPUs map[string][]*tfv1.GPU

	// Score stage compose each node's each GPU's score,
	// node store is sum of GPU score
	ValidNodeGPUScore map[string]map[string]int

	ValidNodeNotMatchingGPUScore map[string]map[string]int

	// In Reserve stage, bind GPUs to pod, update allocator cache
	// In PostBind stage, fetch final GPUs call Pod patch API to update annotation
	FinalGPUs []string

	// Preempt pods
	PreemptPods sync.Map

	// IsPreemption
	IsPreemption bool

	// ScoringStrategy
	ScoringStrategy gpuallocator.Strategy

	// Gang scheduling info for Permit stage
	GangWaitingInfo *gang.WaitingPodInfo
}

func (p *GPUSchedulingStateData) Clone() fwk.StateData {
	return p
}

type PluginFactoryFunc func(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error)

func NewWithDeps(allocator *gpuallocator.GpuAllocator, indexAllocator *indexallocator.IndexAllocator, gangManager *gang.Manager, client client.Client) PluginFactoryFunc {
	return func(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error) {
		target := &config.GPUFitConfig{}
		if unknown, ok := obj.(*runtime.Unknown); ok {
			if err := json.Unmarshal(unknown.Raw, target); err != nil {
				return nil, err
			}
		}
		lh := klog.FromContext(ctx).WithValues("plugin", Name)
		lh.Info("Creating new GPUFit plugin")
		c := &GPUFit{
			logger:         &lh,
			fh:             handle,
			cfg:            target,
			allocator:      allocator,
			indexAllocator: indexAllocator,
			gangManager:    gangManager,
			ctx:            ctx,
			client:         client,
		}
		lh.Info("Created new GPUFit plugin", "plugin", c)

		// Set framework handle on gang manager so it can call Allow/Reject on waiting pods
		if gangManager != nil {
			gangManager.SetFrameworkHandle(handle)
		}

		allocator.SetMaxWorkerPerNode(target.MaxWorkerPerNode)
		return c, nil
	}
}

func (s *GPUFit) Name() string {
	return Name
}

func (s *GPUFit) PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, _ []fwk.NodeInfo) (*framework.PreFilterResult, *fwk.Status) {
	// Handle progressive migration case
	if utils.IsProgressiveMigration() && utils.HasGPUResourceRequest(pod) {
		nodeNames := s.allocator.ListNonUsingNodes()
		s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeNormal, "ScheduleWithNativeGPU",
			"Scheduling non-TF workload for progressive migration",
			"use native GPU resources, available native GPU nodes: "+strconv.Itoa(len(nodeNames)))
		return &framework.PreFilterResult{
			NodeNames: nodeNames,
		}, fwk.NewStatus(fwk.Success, "progressive migration for native resources claim")
	}

	// Skip non tensor-fusion mode
	if !utils.IsTensorFusionWorker(pod) {
		return nil, fwk.NewStatus(fwk.Skip, "skip for non tensor-fusion mode")
	}

	// Check gang scheduling pre-filter
	if s.gangManager != nil {
		if err := s.gangManager.PreFilter(ctx, pod); err != nil {
			s.logger.Info("Gang scheduling PreFilter failed", "pod", pod.Name, "error", err)
			return nil, fwk.NewStatus(fwk.Unschedulable, "gang scheduling pre-filter: "+err.Error())
		}
	}

	// Handle tensor-fusion mode scheduling
	s.logger.Info("checking GPU node resources for pod", "pod", pod.Name)
	allocRequest, reason, err := utils.ComposeAllocationRequest(s.ctx, pod)
	if err != nil {
		return nil, fwk.NewStatus(fwk.Error, reason)
	}
	state.Write(CycleStateAllocateRequest, allocRequest)

	simulateScheduleFilterDetail, err := state.Read(constants.SchedulerSimulationKey)
	isSimulateSchedule := err == nil

	filteredGPUs, filterDetails, err := s.allocator.CheckQuotaAndFilter(ctx, allocRequest, isSimulateSchedule)

	if isSimulateSchedule {
		filterState := simulateScheduleFilterDetail.(*gpuallocator.SimulateSchedulingFilterDetail)
		filterState.FilterStageDetails = filterDetails
		state.Write(constants.SchedulerSimulationKey, filterState)
	}

	if err != nil {
		metrics.SetSchedulerMetrics(allocRequest.PoolName, false)
		s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeWarning, "GPUQuotaOrCapacityNotEnough",
			"check quota and filter", "TensorFusion schedule failed, no enough resource or quotas: "+err.Error())
		s.logger.Error(err, "failed to check quota and filter", "pod", pod.Name)

		if quotaErr, ok := err.(*quota.QuotaExceededError); ok {
			if quotaErr.Unresolvable {
				return nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, quotaErr.Error())
			} else {
				return nil, fwk.NewStatus(fwk.Unschedulable, err.Error())
			}
		} else {
			return nil, fwk.NewStatus(fwk.Unschedulable, err.Error())
		}
	}

	// For partitioned mode, match partition template if not already specified
	if allocRequest.Isolation == tfv1.IsolationModePartitioned && allocRequest.PartitionTemplateID == "" {
		matchedGPU, partitionMatch, err := s.allocator.GetMatchedPartition(allocRequest, filteredGPUs)
		if err != nil {
			metrics.SetSchedulerMetrics(allocRequest.PoolName, false)
			s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeWarning, "PartitionTemplateMatchFailed",
				"match partition template", "Failed to match partition template: "+err.Error())
			s.logger.Error(err, "failed to match partition template", "pod", pod.Name)
			return nil, fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("no suitable partition template: %v", err))
		}

		// Set partition template ID in alloc request
		allocRequest.PartitionTemplateID = partitionMatch.TemplateID
		s.logger.Info("Matched partition template in PreFilter",
			"pod", pod.Name,
			"gpu", matchedGPU.Name,
			"template", allocRequest.PartitionTemplateID,
			"score", partitionMatch.Score)

		// Update state with the updated alloc request
		state.Write(CycleStateAllocateRequest, allocRequest)
	}

	validNodesValidGPUs := lo.GroupBy(filteredGPUs, func(gpu *tfv1.GPU) string {
		return gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
	})
	validNodeNonMatchingGPUs := make(map[string][]*tfv1.GPU, len(validNodesValidGPUs))

	cnt := 0
	allGPUNodeNames := sets.New[string]()
	nodeGPUs := s.allocator.GetNodeGpuStore()
	for k := range nodeGPUs {
		allGPUNodeNames.Insert(k)
	}
	for k, matchedGPUs := range validNodesValidGPUs {
		cnt++

		// get all GPUs on this node
		allGPUs := nodeGPUs[k]

		// all GPUs on this node matched, skip check non-matched GPU score
		total := len(allGPUs)
		matched := len(matchedGPUs)
		if total == matched {
			continue
		}

		preAllocSize := total - matched
		if preAllocSize <= 0 {
			s.logger.Error(nil, "Filtering GPU error, unexpected less than 0", "pod",
				pod.Name, "node", k, "totalGPU count", total, "matchedGPU count", matched)
			preAllocSize = 2
		}
		// range if it's not in validNodesValidGPUs, add to validNodeNonMatchingGPUs
		validNodeNonMatchingGPUs[k] = make([]*tfv1.GPU, 0, preAllocSize)
		for gpuName, gpu := range allGPUs {
			seen := false
			// just loop because the number always <= 8/16
			for _, matchedGPU := range matchedGPUs {
				if gpuName == matchedGPU.Name {
					seen = true
					break
				}
			}
			if !seen {
				validNodeNonMatchingGPUs[k] = append(validNodeNonMatchingGPUs[k], gpu)
			}
		}
	}
	s.logger.Info("filtered valid node GPUs", "nodes count", cnt, "pod", pod.Name)

	strategy := s.allocator.GetScoringStrategy(s.cfg, allocRequest)
	// assign score based on different strategies
	score := s.allocator.Score(ctx, strategy, allocRequest, validNodesValidGPUs)
	// if some GPUs are filtered out but Node is valid, assign score for calculating node average score
	notMatchingGPUScore := s.allocator.Score(ctx, strategy, allocRequest, validNodeNonMatchingGPUs)

	s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeNormal, "PreScheduleDone", "pre filter for TensorFusion workload",
		"TensorFusion pre schedule done, valid GPU node count: "+strconv.Itoa(cnt))

	if s.logger.V(6).Enabled() {
		jsonStr, _ := json.Marshal(validNodesValidGPUs)
		scoreJsonStr, _ := json.Marshal(score)
		s.logger.V(6).Info("PreFilterResult", "validNodeGPUs", jsonStr, "score", scoreJsonStr)
	}

	state.Write(CycleStateGPUSchedulingResult, &GPUSchedulingStateData{
		NodeGPUs:                     validNodesValidGPUs,
		ValidNodeGPUScore:            score,
		ValidNodeNotMatchingGPUScore: notMatchingGPUScore,
		FinalGPUs:                    []string{},
		PreemptPods:                  sync.Map{},
		ScoringStrategy:              strategy,
		IsPreemption:                 false,
	})

	return &framework.PreFilterResult{
		NodeNames: allGPUNodeNames,
	}, fwk.NewStatus(fwk.Success)
}

func (s *GPUFit) PreFilterExtensions() framework.PreFilterExtensions {
	return s
}

func (s *GPUFit) AddPod(ctx context.Context, state fwk.CycleState, pod *v1.Pod, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	stateData, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		return fwk.NewStatus(fwk.Error, err.Error())
	}
	stateDataParsed := stateData.(*GPUSchedulingStateData)
	if pods, ok := stateDataParsed.PreemptPods.Load(nodeInfo.Node().Name); ok {
		podsParsed := pods.(sets.Set[types.NamespacedName])

		nameNs := types.NamespacedName{
			Namespace: podInfoToAdd.GetPod().Namespace,
			Name:      podInfoToAdd.GetPod().Name,
		}
		if podsParsed.Has(nameNs) {
			podsParsed.Delete(nameNs)
		}
	}
	return fwk.NewStatus(fwk.Success, "")
}

func (s *GPUFit) RemovePod(ctx context.Context, state fwk.CycleState, pod *v1.Pod, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	stateData, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		if fwk.ErrNotFound == err {
			stateData = &GPUSchedulingStateData{
				PreemptPods: sync.Map{},
			}
			state.Write(CycleStateGPUSchedulingResult, stateData)
		} else {
			return fwk.NewStatus(fwk.Error, err.Error())
		}
	}
	stateDataParsed := stateData.(*GPUSchedulingStateData)
	stateDataParsed.IsPreemption = true
	if pods, ok := stateDataParsed.PreemptPods.Load(nodeInfo.Node().Name); ok {
		parsedPods := pods.(sets.Set[types.NamespacedName])
		parsedPods.Insert(types.NamespacedName{
			Namespace: podInfoToRemove.GetPod().Namespace,
			Name:      podInfoToRemove.GetPod().Name,
		})
	} else {
		stateDataParsed.PreemptPods.Store(nodeInfo.Node().Name, sets.New(types.NamespacedName{
			Namespace: podInfoToRemove.GetPod().Namespace,
			Name:      podInfoToRemove.GetPod().Name,
		}))
	}
	return fwk.NewStatus(fwk.Success, "")
}

func (s *GPUFit) Filter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if !utils.IsTensorFusionWorker(pod) {
		return fwk.NewStatus(fwk.Success, "skip for non tensor-fusion mode")
	}

	filterResult, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		return fwk.NewStatus(fwk.Error, err.Error())
	}

	// k8s will RemoveAll Pods, and run Filter for high priority pod,
	// then Scheduler framework will reprieve victims one by one until filter returns unschedulable
	if filterResult.(*GPUSchedulingStateData).IsPreemption {
		allocRequest, err := state.Read(CycleStateAllocateRequest)
		allocRequestParsed := allocRequest.(*tfv1.AllocRequest)
		if err != nil {
			return fwk.NewStatus(fwk.Error, err.Error())
		}
		podsToPreempt, ok := filterResult.(*GPUSchedulingStateData).PreemptPods.Load(nodeInfo.Node().Name)
		if !ok {
			return fwk.NewStatus(fwk.Unschedulable, "no pods to preempt")
		}
		podsToPreemptParsed := podsToPreempt.(sets.Set[types.NamespacedName])
		err = s.allocator.CheckQuotaAndFilterSingleNodePreempt(
			nodeInfo.Node().Name, allocRequestParsed, podsToPreemptParsed)
		if err != nil {
			return fwk.NewStatus(fwk.Unschedulable, err.Error())
		}
		return fwk.NewStatus(fwk.Success, "")
	}

	nodeName := nodeInfo.Node().Name

	// Check if there are higher priority nominated pods waiting for this node's GPU resources
	// This ensures that low priority pods don't steal GPU resources from pods that have already
	// won preemption and are waiting for victims to terminate
	if status := s.checkNominatedPodsGPUReservation(pod, nodeName, filterResult.(*GPUSchedulingStateData)); !status.IsSuccess() {
		return status
	}

	if _, ok := filterResult.(*GPUSchedulingStateData).NodeGPUs[nodeName]; !ok {
		return fwk.NewStatus(fwk.Unschedulable, "GPU not fit")
	}
	return fwk.NewStatus(fwk.Success, "")
}

// checkNominatedPodsGPUReservation checks if there are higher priority TensorFusion pods
// that have nominated this node and are waiting for GPU resources.
// If the current pod has lower priority, it should not be scheduled to avoid stealing
// resources that are essentially "reserved" for the nominated pod.
func (s *GPUFit) checkNominatedPodsGPUReservation(pod *v1.Pod, nodeName string, schedulingData *GPUSchedulingStateData) *fwk.Status {
	nominatedPodInfos := s.fh.NominatedPodsForNode(nodeName)
	if len(nominatedPodInfos) == 0 {
		return fwk.NewStatus(fwk.Success, "")
	}

	currentPodPriority := int32(0)
	if pod.Spec.Priority != nil {
		currentPodPriority = *pod.Spec.Priority
	}

	availableGPUs := schedulingData.NodeGPUs[nodeName]
	if len(availableGPUs) == 0 {
		return fwk.NewStatus(fwk.Success, "")
	}

	// Calculate total available GPU resources on this node
	totalAvailableTflops := resource.Quantity{}
	totalAvailableVram := resource.Quantity{}
	for _, gpu := range availableGPUs {
		if gpu.Status.Available != nil {
			totalAvailableTflops.Add(gpu.Status.Available.Tflops)
			totalAvailableVram.Add(gpu.Status.Available.Vram)
		}
	}

	// Calculate resources needed by higher priority nominated pods
	reservedTflops := resource.Quantity{}
	reservedVram := resource.Quantity{}

	for _, nominatedPodInfo := range nominatedPodInfos {
		nominatedPod := nominatedPodInfo.GetPod()

		// Skip if it's the same pod
		if nominatedPod.UID == pod.UID {
			continue
		}

		// Only consider TensorFusion worker pods
		if !utils.IsTensorFusionWorker(nominatedPod) {
			continue
		}

		nominatedPodPriority := int32(0)
		if nominatedPod.Spec.Priority != nil {
			nominatedPodPriority = *nominatedPod.Spec.Priority
		}

		// Reserve resources for nominated pods with higher priority
		// For equal priority: also reserve to give nominated pods precedence (they won preemption)
		if nominatedPodPriority < currentPodPriority {
			continue
		}

		// CRITICAL: If nominated pod has higher priority, ALWAYS block lower priority pods
		// This prevents preempted pods from restarting and stealing resources
		if nominatedPodPriority > currentPodPriority {
			s.logger.Info("Blocking lower priority pod to protect nominated pod's resources",
				"currentPod", pod.Name,
				"currentPriority", currentPodPriority,
				"nominatedPod", nominatedPod.Name,
				"nominatedPriority", nominatedPodPriority,
				"node", nodeName)
		}

		// Get the nominated pod's GPU resource requirements
		nominatedAllocReq, _, err := s.allocator.ComposeAllocationRequest(nominatedPod)
		if err != nil {
			s.logger.V(4).Info("Failed to compose allocation request for nominated pod",
				"nominatedPod", nominatedPod.Name, "error", err)
			continue
		}

		// CRITICAL FIX: For multi-GPU nominated pods with higher priority,
		// block ALL lower priority pods on this node to prevent partial resource stealing
		// This prevents the issue where victim pods restart and occupy 1 GPU while
		// the 2-GPU nominated pod is waiting for both GPUs
		if nominatedAllocReq.Count > 1 && nominatedPodPriority > currentPodPriority {
			s.logger.Info("Blocking lower priority pod completely for multi-GPU nominated pod",
				"currentPod", pod.Name,
				"currentPriority", currentPodPriority,
				"nominatedPod", nominatedPod.Name,
				"nominatedPriority", nominatedPodPriority,
				"nominatedGPUCount", nominatedAllocReq.Count,
				"node", nodeName)
			return fwk.NewStatus(fwk.Unschedulable,
				fmt.Sprintf("Node reserved for multi-GPU higher priority nominated pod %s (requires %d GPUs)", nominatedPod.Name, nominatedAllocReq.Count))
		}

		// Add to reserved resources
		reservedTflops.Add(nominatedAllocReq.Request.Tflops)
		reservedVram.Add(nominatedAllocReq.Request.Vram)

		s.logger.V(4).Info("Reserving GPU resources for nominated pod",
			"currentPod", pod.Name,
			"nominatedPod", nominatedPod.Name,
			"nominatedPriority", nominatedPodPriority,
			"currentPriority", currentPodPriority,
			"reservedTflops", nominatedAllocReq.Request.Tflops.String(),
			"reservedVram", nominatedAllocReq.Request.Vram.String())
	}

	// If no resources need to be reserved, allow scheduling
	if reservedTflops.IsZero() && reservedVram.IsZero() {
		return fwk.NewStatus(fwk.Success, "")
	}

	// Check if there are enough resources after reservation
	remainingTflops := totalAvailableTflops.DeepCopy()
	remainingVram := totalAvailableVram.DeepCopy()
	remainingTflops.Sub(reservedTflops)
	remainingVram.Sub(reservedVram)

	// Get current pod's requirements
	currentAllocReq, _, err := s.allocator.ComposeAllocationRequest(pod)
	if err != nil {
		return fwk.NewStatus(fwk.Error, "failed to compose allocation request: "+err.Error())
	}

	// Check if remaining resources are sufficient
	if remainingTflops.Cmp(currentAllocReq.Request.Tflops) < 0 ||
		remainingVram.Cmp(currentAllocReq.Request.Vram) < 0 {
		s.logger.Info("GPU resources reserved for higher priority nominated pods",
			"currentPod", pod.Name,
			"node", nodeName,
			"currentPriority", currentPodPriority,
			"reservedTflops", reservedTflops.String(),
			"reservedVram", reservedVram.String(),
			"requiredTflops", currentAllocReq.Request.Tflops.String(),
			"requiredVram", currentAllocReq.Request.Vram.String())
		return fwk.NewStatus(fwk.Unschedulable,
			fmt.Sprintf("GPU resources reserved for higher priority nominated pods on node %s", nodeName))
	}

	return fwk.NewStatus(fwk.Success, "")
}

func (s *GPUFit) Score(
	ctx context.Context,
	state fwk.CycleState,
	pod *v1.Pod,
	nodeInfo fwk.NodeInfo,
) (int64, *fwk.Status) {
	// Skip non tensor-fusion mode scheduling
	if !utils.IsTensorFusionWorker(pod) {
		return 0, fwk.NewStatus(fwk.Success, "")
	}

	if state == nil {
		return 0, fwk.NewStatus(fwk.Error, "invalid schedule state")
	}
	filterResult, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		return 0, fwk.NewStatus(fwk.Error, err.Error())
	}
	scheduledState := filterResult.(*GPUSchedulingStateData)
	gpuScoreMap, ok := scheduledState.ValidNodeGPUScore[nodeInfo.Node().Name]
	if !ok {
		return 0, fwk.NewStatus(fwk.Unschedulable, "not valid node")
	}
	// normalize to 0-100, when node has more GPUs but filtered out,
	// should consider it as 100 when strategy is compact_first, and consider as 0 when is low_load_first
	sum := 0
	for _, score := range gpuScoreMap {
		sum += score
	}

	notMatchingGPUScoreMap, ok := scheduledState.ValidNodeNotMatchingGPUScore[nodeInfo.Node().Name]
	if ok {
		for _, score := range notMatchingGPUScoreMap {
			sum += score
		}
	}
	return int64(sum / (len(gpuScoreMap) + len(notMatchingGPUScoreMap))), nil
}

func (s *GPUFit) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

func (s *GPUFit) Reserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	if !utils.IsTensorFusionWorker(pod) {
		return fwk.NewStatus(fwk.Success, "skip for non tensor-fusion mode")
	}

	s.logger.Info("Reserving pod for GPU resources", "pod", pod.Name, "node", nodeName)
	allocRequest, err := state.Read(CycleStateAllocateRequest)
	if err != nil {
		return fwk.NewStatus(fwk.Error, err.Error())
	}

	schedulingResultRaw, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		return fwk.NewStatus(fwk.Error, err.Error())
	}

	// set final GPUs and try update GPU allocator cache
	schedulingResult := schedulingResultRaw.(*GPUSchedulingStateData)
	validGPUs, ok := schedulingResult.NodeGPUs[nodeName]
	if !ok {
		return fwk.NewStatus(fwk.Unschedulable, "not valid node")
	}

	// find top N score GPUs in this node
	neededGPUs := allocRequest.(*tfv1.AllocRequest).Count

	// when needed GPUs equals to valid GPUs, just return all GPUs on this node
	if neededGPUs == uint(len(validGPUs)) {
		schedulingResult.FinalGPUs = lo.Map(validGPUs, func(gpu *tfv1.GPU, _ int) string {
			return gpu.Name
		})
	} else if neededGPUs < uint(len(validGPUs)) {
		// try scoring GPU from single node level
		// TODO: consider NUMA topology on one node when neededGPUs > 1
		gpuScoreEntries := make([]lo.Entry[string, int], 0, len(validGPUs))
		for _, gpu := range validGPUs {
			score := schedulingResult.ScoringStrategy.Score(gpu, false)
			gpuScoreEntries = append(gpuScoreEntries, lo.Entry[string, int]{Key: gpu.Name, Value: score})
		}
		slices.SortFunc(gpuScoreEntries, func(i, j lo.Entry[string, int]) int {
			return j.Value - i.Value
		})
		schedulingResult.FinalGPUs = lo.Map(gpuScoreEntries[:neededGPUs], func(entry lo.Entry[string, int], _ int) string {
			return entry.Key
		})
	} else {
		return fwk.NewStatus(fwk.Error, "not enough GPUs on nominated node:"+nodeName)
	}

	// reserve GPU resources inside memory and asynchronously update GPU custom resource
	allocReq := allocRequest.(*tfv1.AllocRequest)
	_, err = s.allocator.Bind(
		schedulingResult.FinalGPUs,
		allocReq,
	)
	if err != nil {
		return fwk.NewStatus(fwk.Error, err.Error())
	}

	return fwk.NewStatus(fwk.Success, "")
}

// Permit implements the gang scheduling logic
func (s *GPUFit) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	if !utils.IsTensorFusionWorker(pod) {
		return fwk.NewStatus(fwk.Success, "skip for non tensor-fusion mode"), 0
	}

	// Check if gang scheduling is enabled for this pod
	if s.gangManager == nil {
		return fwk.NewStatus(fwk.Success, "gang manager not initialized"), 0
	}

	gangConfig := s.gangManager.ParseGangConfig(pod)
	if !gangConfig.Enabled {
		return fwk.NewStatus(fwk.Success, "gang scheduling not enabled"), 0
	}

	s.logger.Info("Permit stage: checking gang scheduling", "pod", pod.Name, "node", nodeName, "group", gangConfig.GroupKey)

	// Get alloc request for gang info
	allocRequestRaw, err := state.Read(CycleStateAllocateRequest)
	if err != nil {
		return fwk.NewStatus(fwk.Error, "failed to read alloc request: "+err.Error()), 0
	}
	allocReq := allocRequestRaw.(*tfv1.AllocRequest)

	// Get scheduling result for GPU info
	schedulingResultRaw, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		return fwk.NewStatus(fwk.Error, "failed to read scheduling result: "+err.Error()), 0
	}
	schedulingResult := schedulingResultRaw.(*GPUSchedulingStateData)

	// Call gang manager Permit
	status, waitTime, waitingInfo := s.gangManager.Permit(ctx, pod, nodeName, allocReq)

	// Store waiting info in state for later use (e.g., Unreserve)
	schedulingResult.GangWaitingInfo = waitingInfo
	state.Write(CycleStateGPUSchedulingResult, schedulingResult)

	switch status {
	case gang.PermitAllow:
		s.logger.Info("Gang scheduling: all members ready, proceeding", "pod", pod.Name, "group", gangConfig.GroupKey)
		s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeNormal, gang.EventReasonGangScheduled,
			"Gang scheduling complete", "All gang members ready for group %s", gangConfig.GroupKey)
		return fwk.NewStatus(fwk.Success, "gang scheduling complete"), 0

	case gang.PermitWait:
		s.logger.Info("Gang scheduling: waiting for members", "pod", pod.Name, "group", gangConfig.GroupKey, "waitTime", waitTime)
		s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeNormal, gang.EventReasonGangWaiting,
			"Waiting for gang members", "Waiting for other members in group %s", gangConfig.GroupKey)
		return fwk.NewStatus(fwk.Wait, "waiting for gang members"), waitTime

	case gang.PermitReject:
		s.logger.Info("Gang scheduling: rejected (timeout or insufficient members)", "pod", pod.Name, "group", gangConfig.GroupKey)
		s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeWarning, gang.EventReasonGangTimeout,
			"Gang scheduling timeout", "Gang scheduling timeout for group %s", gangConfig.GroupKey)
		return fwk.NewStatus(fwk.Unschedulable, "gang scheduling timeout"), 0

	default:
		return fwk.NewStatus(fwk.Error, "unknown permit status"), 0
	}
}

func (s *GPUFit) Unreserve(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) {
	if !utils.IsTensorFusionWorker(pod) {
		return
	}

	s.logger.Info("Un-reserving pod for GPU resources", "pod", pod.Name, "node", nodeName)
	schedulingResultRaw, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		s.logger.Error(err, "failed to read gpu scheduling result", "pod", pod.Name)
		return
	}
	schedulingResult := schedulingResultRaw.(*GPUSchedulingStateData)

	// Notify gang manager about unreserve
	if s.gangManager != nil {
		s.gangManager.Unreserve(ctx, pod)
	}

	s.allocator.Dealloc(tfv1.NameNamespace{
		Name:      pod.Labels[constants.WorkloadKey],
		Namespace: pod.Namespace,
	}, schedulingResult.FinalGPUs, pod.ObjectMeta)
}

func (s *GPUFit) PostBind(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) {
	if !utils.IsTensorFusionWorker(pod) {
		return
	}

	s.logger.Info("PostBinding pod for GPU resources", "pod", pod.Name, "node", nodeName)
	gpuSchedulingResult, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		s.logger.Error(err, "failed to read gpu scheduling result", "pod", pod.Name)
		return
	}
	// write the allocated GPU info to Pod in bindingCycle, before default binder changing the Pod nodeName info
	gpuIDs := strings.Join(gpuSchedulingResult.(*GPUSchedulingStateData).FinalGPUs, ",")
	s.logger.Info("PostBinding pod for GPU resources", "pod", pod.Name, "node", nodeName, "gpuIDs", gpuIDs)

	index, err := utils.ParsePodIndexResourceClaim(pod)
	if err != nil {
		s.logger.Error(err, "failed to parse pod index annotation", "pod", pod.Name)
		return
	}

	indexAvailable := s.indexAllocator.CheckNodeIndexAndTryOccupy(pod, index)

	// Build patch operations
	patchOps := []map[string]any{
		{
			"op":    "add",
			"path":  "/metadata/annotations/" + utils.EscapeJSONPointer(constants.GPUDeviceIDsAnnotation),
			"value": gpuIDs,
		},
	}
	if indexAvailable {
		patchOps = append(patchOps, map[string]any{
			"op":    "add",
			"path":  "/metadata/annotations/" + utils.EscapeJSONPointer(constants.PodIndexAnnotation),
			"value": strconv.Itoa(index),
		})
	} else {
		s.logger.Info("Index is not available on node, spawn a goroutine to patch it asynchronously", "pod", pod.Name, "node", nodeName, "index", index)
		// spawn a goroutine to patch
		s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeNormal, "PodIndexAllocationPending", "Pod index allocation pending",
			fmt.Sprintf("Index %d will be patched into pod after released by other pod on the same node: %s", index, nodeName))
		s.indexAllocator.AsyncCheckNodeIndexAvailableAndAssign(pod, index)
	}

	// Add partition template ID annotation if in partitioned mode
	allocRequestRaw, err := state.Read(CycleStateAllocateRequest)
	if err == nil {
		allocRequest := allocRequestRaw.(*tfv1.AllocRequest)
		if allocRequest.Isolation == tfv1.IsolationModePartitioned && allocRequest.PartitionTemplateID != "" {
			patchOps = append(patchOps, map[string]any{
				"op":    "add",
				"path":  "/metadata/annotations/" + utils.EscapeJSONPointer(constants.PartitionTemplateIDAnnotation),
				"value": allocRequest.PartitionTemplateID,
			})
			s.logger.Info("Adding partition template ID annotation", "pod", pod.Name, "templateID", allocRequest.PartitionTemplateID)
		}
	}

	// Convert patch operations to JSON
	patchBytes, err := json.Marshal(patchOps)
	if err != nil {
		s.logger.Error(err, "failed to marshal patch operations", "pod", pod.Name)
		return
	}

	// Patch pod annotations with retry
	err = retry.OnError(wait.Backoff{
		Duration: 1 * time.Second,
		Factor:   2,
		Jitter:   0.1,
		Steps:    3,
	}, func(err error) bool {
		return true
	}, func() error {
		err = s.client.Patch(s.ctx, pod, client.RawPatch(types.JSONPatchType, patchBytes))
		if err != nil {
			s.logger.Error(err, "failed to patch pod annotations", "pod", pod.Name)
			s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeWarning, "GPUDeviceAllocatedFailed",
				"Attach GPU device ID info failed", "Can not add GPU device IDs: "+gpuIDs)
		} else {
			s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeNormal, "GPUDeviceAllocated",
				"Attach GPU device ID info", "Attach TensorFusion GPU device IDs to Pod: "+gpuIDs)
		}
		return nil
	})
	if err != nil {
		if indexAvailable {
			s.indexAllocator.RemoveNodeIndexQueueForPod(types.NamespacedName{
				Namespace: pod.Namespace,
				Name:      pod.Name,
			})
		}
		s.logger.Error(err, "failed to patch pod annotations in post binding stage", "pod", pod.Name)
		return
	}
}

func (s *GPUFit) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	// EventResource format must be: <plural-resource-name>.<version>.<group>
	// Example: gpus.v1.tensor-fusion.ai
	// The scheduler's eventhandlers.go will use DynInformerFactory to watch this resource
	gpuResource := fwk.EventResource("gpus.v1.tensor-fusion.ai")
	return []fwk.ClusterEventWithHint{
		{
			Event: fwk.ClusterEvent{
				Resource:   gpuResource,
				ActionType: fwk.Add | fwk.Update,
			},
			QueueingHintFn: s.queueingHint,
		},
	}, nil
}

// convertToGPU converts an any to *tfv1.GPU, handling both typed and unstructured objects
func convertToGPU(obj any) (*tfv1.GPU, error) {
	if obj == nil {
		return nil, nil
	}

	// Try direct type assertion first (fastest path)
	if gpu, ok := obj.(*tfv1.GPU); ok {
		return gpu, nil
	}

	// Try to convert from unstructured.Unstructured using DefaultUnstructuredConverter
	if unstructuredObj, ok := obj.(*unstructured.Unstructured); ok {
		gpu := &tfv1.GPU{}
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(unstructuredObj.Object, gpu); err != nil {
			return nil, fmt.Errorf("failed to convert unstructured to GPU: %w", err)
		}
		return gpu, nil
	}
	return nil, fmt.Errorf("cannot convert %T to *tfv1.GPU", obj)
}

func (s *GPUFit) queueingHint(logger klog.Logger, pod *v1.Pod, oldObj, newObj any) (fwk.QueueingHint, error) {
	// Only process TensorFusion worker pods
	if !utils.IsTensorFusionWorker(pod) {
		return fwk.QueueSkip, nil
	}

	oldGPU, err := convertToGPU(oldObj)
	if err != nil {
		logger.V(5).Info("Failed to convert oldObj to GPU, skip", "error", err)
		return fwk.QueueSkip, nil
	}

	newGPU, err := convertToGPU(newObj)
	if err != nil {
		logger.V(5).Info("Failed to convert newObj to GPU, skip", "error", err)
		return fwk.QueueSkip, nil
	}

	// Calculate resource increase
	var increaseTflops, increaseVram resource.Quantity
	if oldGPU == nil && newGPU != nil {
		// Add event: use available resources as increase
		if newGPU.Status.Available != nil {
			increaseTflops = newGPU.Status.Available.Tflops.DeepCopy()
			increaseVram = newGPU.Status.Available.Vram.DeepCopy()
		}
	} else if oldGPU != nil && newGPU != nil {
		// Update event: calculate difference in available resources
		if oldGPU.Status.Available != nil && newGPU.Status.Available != nil {
			increaseTflops = newGPU.Status.Available.Tflops.DeepCopy()
			increaseVram = newGPU.Status.Available.Vram.DeepCopy()
			increaseTflops.Sub(oldGPU.Status.Available.Tflops)
			increaseVram.Sub(oldGPU.Status.Available.Vram)
		}
	}

	// If resource decreased, skip
	if increaseTflops.Cmp(resource.MustParse("0")) <= 0 && increaseVram.Cmp(resource.MustParse("0")) <= 0 {
		return fwk.QueueSkip, nil
	}

	// OPTIMIZATION: For nominated pods (pods that won preemption), immediately requeue
	// when any GPU resource becomes available on their nominated node.
	// This significantly reduces the delay after preemption.
	if pod.Status.NominatedNodeName != "" && newGPU != nil {
		gpuNodeName := newGPU.Status.NodeSelector[constants.KubernetesHostNameLabel]
		if gpuNodeName == pod.Status.NominatedNodeName {
			logger.Info("GPU resource released on nominated node, immediately requeue preempting pod",
				"pod", klog.KObj(pod),
				"nominatedNode", pod.Status.NominatedNodeName,
				"gpu", newGPU.Name,
				"increaseTflops", increaseTflops.String(),
				"increaseVram", increaseVram.String())
			return fwk.Queue, nil
		}
	}

	// Compose allocation request for the pod passed in by scheduler framework
	allocRequest, _, err := utils.ComposeAllocationRequest(s.ctx, pod)
	if err != nil {
		logger.V(5).Info("Failed to compose allocation request for pod, skip",
			"pod", klog.KObj(pod), "error", err)
		return fwk.QueueSkip, nil
	}

	// Calculate total request for this pod (multiply by count)
	podTotalTflops := allocRequest.Request.Tflops
	podTotalVram := allocRequest.Request.Vram

	// Queue if resource increase >= pod's request
	if increaseTflops.Cmp(podTotalTflops) >= 0 && increaseVram.Cmp(podTotalVram) >= 0 {
		logger.V(4).Info("GPU resource increase sufficient for pod, requeue unscheduled pod",
			"pod", klog.KObj(pod),
			"increaseTflops", increaseTflops.String(),
			"increaseVram", increaseVram.String(),
			"podRequestTflops", podTotalTflops.String(),
			"podRequestVram", podTotalVram.String())
		return fwk.Queue, nil
	}

	return fwk.QueueSkip, nil
}
