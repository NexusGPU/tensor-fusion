package gpuresources

import (
	"context"
	"encoding/json"
	"sort"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
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

var _ framework.PreFilterPlugin = &GPUFit{}
var _ framework.FilterPlugin = &GPUFit{}
var _ framework.ScorePlugin = &GPUFit{}
var _ framework.ReservePlugin = &GPUFit{}
var _ framework.PreBindPlugin = &GPUFit{}

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
	NodeGPUs map[string][]tfv1.GPU

	// Score stage compose each node's each GPU's score,
	// node store is sum of GPU score
	ValidNodeGPUScore map[string]map[string]int

	// In Reserve stage, bind GPUs to pod, update allocator cache
	// In PreBind stage, fetch final GPUs call Pod patch API to update annotation
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
		nodeNames := s.allocator.ListNonTensorFusionNodes()
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
	state.Write(CycleStateAllocateRequest, &allocRequest)

	filteredGPUs, err := s.allocator.CheckQuotaAndFilter(ctx, &allocRequest)
	if err != nil {
		s.logger.Error(err, "failed to check quota and filter", "pod", pod.Name)
		return nil, framework.NewStatus(framework.Unschedulable, err.Error())
	}

	validNodeGPUs := lo.GroupBy(filteredGPUs, func(gpu tfv1.GPU) string {
		return gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
	})
	// remove nodes that don't have enough GPUs that meet the request
	nodeNames := sets.New[string]()
	for k, v := range validNodeGPUs {
		if len(v) < int(allocRequest.Count) {
			delete(validNodeGPUs, k)
		} else {
			nodeNames.Insert(k)
		}
	}
	s.logger.Info("filtered valid node GPUs", "validNodeGPU count", len(validNodeGPUs), "nodeNames count", nodeNames.Len(), "pod", pod.Name)

	// assign score based on different strategies
	score := s.allocator.Score(ctx, s.cfg, allocRequest, validNodeGPUs)

	if s.logger.V(6).Enabled() {
		jsonStr, _ := json.Marshal(validNodeGPUs)
		scoreJsonStr, _ := json.Marshal(score)
		s.logger.V(6).Info("PreFilterResult", "validNodeGPUs", jsonStr, "score", scoreJsonStr)
	}

	state.Write(CycleStateGPUSchedulingResult, &GPUSchedulingStateData{
		NodeGPUs:          validNodeGPUs,
		ValidNodeGPUScore: score,
		FinalGPUs:         []string{},
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

func (s *GPUFit) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
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
	gpuScoreMap, ok := scheduledState.ValidNodeGPUScore[nodeName]
	if !ok {
		return 0, framework.NewStatus(framework.Unschedulable, "no valid node found, gpu capacity not enough")
	}
	// normalize to 0-100 when each node has
	sum := 0
	for _, score := range gpuScoreMap {
		sum += score
	}
	return int64(sum / len(gpuScoreMap)), nil
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

func (s *GPUFit) PreBind(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	if !utils.IsTensorFusionWorker(pod) {
		return framework.NewStatus(framework.Success, "skip for non tensor-fusion mode")
	}

	s.logger.Info("PreBinding pod for GPU resources", "pod", pod.Name, "node", nodeName)
	gpuSchedulingResult, err := state.Read(CycleStateGPUSchedulingResult)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	// write the allocated GPU info to Pod in bindingCycle, before default binder changing the Pod nodeName info
	gpuIDs := strings.Join(gpuSchedulingResult.(*GPUSchedulingStateData).FinalGPUs, ",")
	s.logger.Info("PreBinding pod for GPU resources", "pod", pod.Name, "node", nodeName, "gpuIDs", gpuIDs)
	patch := []byte(`[{
		"op": "add",
		"path": "/metadata/annotations/` + utils.EscapeJSONPointer(constants.GPUDeviceIDsAnnotation) + `",
		"value": "` + gpuIDs + `"}]`)

	err = s.client.Patch(s.ctx, pod, client.RawPatch(types.JSONPatchType, patch))
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	return framework.NewStatus(framework.Success, "")
}
