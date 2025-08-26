package gpuresources

import (
	"context"
	"encoding/json"
	"sort"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
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
var _ framework.ReservePlugin = &GPUFit{}
var _ framework.PostBindPlugin = &GPUFit{}

type GPUFit struct {
	logger    *klog.Logger
	fh        framework.Handle
	client    client.Client
	allocator *gpuallocator.GpuAllocator
	ctx       context.Context
	cfg       *config.GPUFitConfig
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
}

func (p *GPUSchedulingStateData) Clone() framework.StateData {
	return p
}

type PluginFactoryFunc func(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error)

func NewWithDeps(allocator *gpuallocator.GpuAllocator, client client.Client) PluginFactoryFunc {
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
			logger:    &lh,
			fh:        handle,
			cfg:       target,
			allocator: allocator,
			ctx:       ctx,
			client:    client,
		}
		lh.Info("Created new GPUFit plugin", "plugin", c)

		allocator.SetMaxWorkerPerNode(target.MaxWorkerPerNode)
		return c, nil
	}
}

func (s *GPUFit) Name() string {
	return Name
}

func (s *GPUFit) PreFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod) (*framework.PreFilterResult, *framework.Status) {
	// Handle progressive migration case
	if utils.IsProgressiveMigration() && utils.HasGPUResourceRequest(pod) {
		nodeNames := s.allocator.ListNonUsingNodes()
		s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeNormal, "ScheduleWithNativeGPU",
			"Scheduling non-TF workload for progressive migration",
			"use native GPU resources, available native GPU nodes: "+strconv.Itoa(len(nodeNames)))
		return &framework.PreFilterResult{
			NodeNames: nodeNames,
		}, framework.NewStatus(framework.Success, "progressive migration for native resources claim")
	}

	// Skip non tensor-fusion mode
	if !utils.IsTensorFusionWorker(pod) {
		return nil, framework.NewStatus(framework.Skip, "skip for non tensor-fusion mode")
	}

	// Handle tensor-fusion mode scheduling
	s.logger.Info("checking GPU node resources for pod", "pod", pod.Name)
	allocRequest, reason, err := s.allocator.ComposeAllocationRequest(pod)
	if err != nil {
		return nil, framework.NewStatus(framework.Error, reason)
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
		return nil, framework.NewStatus(framework.Unschedulable, err.Error())
	}

	validNodesValidGPUs := lo.GroupBy(filteredGPUs, func(gpu *tfv1.GPU) string {
		return gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
	})
	validNodeNonMatchingGPUs := make(map[string][]*tfv1.GPU, len(validNodesValidGPUs))

	nodeNames := sets.New[string]()
	nodeGPUs := s.allocator.GetNodeGpuStore()
	for k, matchedGPUs := range validNodesValidGPUs {
		nodeNames.Insert(k)

		// get all GPUs on this node
		allGPUs := nodeGPUs[k]

		// all GPUs on this node matched, skip check non-matched GPU score
		total := len(allGPUs)
		matched := len(matchedGPUs)
		if total == matched {
			continue
		}

		// range if it's not in validNodesValidGPUs, add to validNodeNonMatchingGPUs
		validNodeNonMatchingGPUs[k] = make([]*tfv1.GPU, 0, total-matched)
		for gpuName, gpu := range allGPUs {
			seen := false
			// just loop because the number always <= 8
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
	s.logger.Info("filtered valid node GPUs", "nodes count", nodeNames.Len(), "pod", pod.Name)

	// assign score based on different strategies
	score := s.allocator.Score(ctx, s.cfg, allocRequest, validNodesValidGPUs)

	// if some GPUs are filtered out but Node is valid, assign score for calculating node average score
	notMatchingGPUScore := s.allocator.Score(ctx, s.cfg, allocRequest, validNodeNonMatchingGPUs)

	s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeNormal, "PreScheduleDone", "pre filter for TensorFusion workload",
		"TensorFusion pre schedule done, valid GPU node count: "+strconv.Itoa(nodeNames.Len()))

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
	})

	return &framework.PreFilterResult{
		NodeNames: nodeNames,
	}, framework.NewStatus(framework.Success)
}

func (s *GPUFit) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

func (s *GPUFit) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	if !utils.IsTensorFusionWorker(pod) {
		return framework.NewStatus(framework.Success, "skip for non tensor-fusion mode")
	}

	filterResult, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	nodeName := nodeInfo.GetName()
	if _, ok := filterResult.(*GPUSchedulingStateData).NodeGPUs[nodeName]; !ok {
		return framework.NewStatus(framework.Unschedulable, "no valid node found, gpu capacity not enough")
	}
	return framework.NewStatus(framework.Success, "")
}

func (s *GPUFit) Score(
	ctx context.Context,
	state *framework.CycleState,
	pod *v1.Pod,
	nodeInfo *framework.NodeInfo,
) (int64, *framework.Status) {
	// Skip non tensor-fusion mode scheduling
	if !utils.IsTensorFusionWorker(pod) {
		return 0, framework.NewStatus(framework.Success, "")
	}

	if state == nil {
		return 0, framework.NewStatus(framework.Error, "no valid node found, gpu capacity not enough")
	}
	filterResult, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, err.Error())
	}
	scheduledState := filterResult.(*GPUSchedulingStateData)
	gpuScoreMap, ok := scheduledState.ValidNodeGPUScore[nodeInfo.GetName()]
	if !ok {
		return 0, framework.NewStatus(framework.Unschedulable, "no valid node found, gpu capacity not enough")
	}
	// normalize to 0-100, when node has more GPUs but filtered out,
	// should consider it as 100 when strategy is compact_first, and consider as 0 when is low_load_first
	sum := 0
	for _, score := range gpuScoreMap {
		sum += score
	}

	notMatchingGPUScoreMap, ok := scheduledState.ValidNodeNotMatchingGPUScore[nodeInfo.GetName()]
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

func (s *GPUFit) Reserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	if !utils.IsTensorFusionWorker(pod) {
		return framework.NewStatus(framework.Success, "skip for non tensor-fusion mode")
	}

	s.logger.Info("Reserving pod for GPU resources", "pod", pod.Name, "node", nodeName)
	allocRequest, err := state.Read(CycleStateAllocateRequest)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	schedulingResultRaw, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	// set final GPUs and try update GPU allocator cache
	schedulingResult := schedulingResultRaw.(*GPUSchedulingStateData)
	gpuScoreMap, ok := schedulingResult.ValidNodeGPUScore[nodeName]
	if !ok {
		return framework.NewStatus(framework.Unschedulable, "no valid node found, gpu capacity not enough")
	}

	// find top N score GPUs in this node
	neededGPUs := allocRequest.(*tfv1.AllocRequest).Count

	gpuScoreEntries := lo.Entries(gpuScoreMap)
	sort.Slice(gpuScoreEntries, func(i, j int) bool {
		return gpuScoreEntries[i].Value < gpuScoreEntries[j].Value
	})

	schedulingResult.FinalGPUs = lo.Map(gpuScoreEntries[:neededGPUs], func(entry lo.Entry[string, int], _ int) string {
		return entry.Key
	})
	state.Write(CycleStateGPUSchedulingResult, schedulingResult)

	_, err = s.allocator.Bind(
		schedulingResult.FinalGPUs,
		allocRequest.(*tfv1.AllocRequest),
	)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	return framework.NewStatus(framework.Success, "")
}

func (s *GPUFit) Unreserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
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

	s.allocator.Dealloc(tfv1.NameNamespace{
		Name:      pod.Labels[constants.WorkloadKey],
		Namespace: pod.Namespace,
	}, schedulingResult.FinalGPUs, pod.ObjectMeta)
}

func (s *GPUFit) PostBind(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
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
	patch := []byte(`[{
		"op": "add",
		"path": "/metadata/annotations/` + utils.EscapeJSONPointer(constants.GPUDeviceIDsAnnotation) + `",
		"value": "` + gpuIDs + `"}]`)

	err = s.client.Patch(s.ctx, pod, client.RawPatch(types.JSONPatchType, patch))
	if err != nil {
		s.logger.Error(err, "failed to patch gpu device ids", "pod", pod.Name)
		s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeWarning, "GPUDeviceAllocatedFailed",
			"Attach GPU device ID info failed", "Can not add GPU device IDs: "+gpuIDs)
	} else {
		s.fh.EventRecorder().Eventf(pod, pod, v1.EventTypeNormal, "GPUDeviceAllocated",
			"Attach GPU device ID info", "Attach TensorFusion GPU device IDs to Pod: "+gpuIDs)
	}
}
