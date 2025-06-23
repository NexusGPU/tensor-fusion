package gpuresources

import (
	"context"
	"encoding/json"
	"strconv"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const Name = "GPUResourceFit"

var _ framework.FilterPlugin = &GPUFit{}
var _ framework.ScorePlugin = &GPUFit{}
var _ framework.ReservePlugin = &GPUFit{}

type GPUFit struct {
	logger    *klog.Logger
	fh        framework.Handle
	podLister cache.Indexer
	pdbLister cache.Indexer
	client    *rest.RESTClient
	allocator *gpuallocator.GpuAllocator

	cfg *GPUFitConfig
}

type GPUFitConfig struct {
	MaxWorkerPerNode int `json:"maxWorkerPerNode"`
}

type PluginFactoryFunc func(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error)

func NewWithDeps(allocator *gpuallocator.GpuAllocator) PluginFactoryFunc {
	return func(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error) {
		target := &GPUFitConfig{}
		if unknown, ok := obj.(*runtime.Unknown); ok {
			if err := json.Unmarshal(unknown.Raw, target); err != nil {
				return nil, err
			}
		}
		lh := klog.FromContext(ctx).WithValues("plugin", Name)
		c := &GPUFit{
			logger:    &lh,
			fh:        handle,
			cfg:       target,
			allocator: allocator,
		}
		return c, nil
	}
}

func (s *GPUFit) Name() string {
	return Name
}

func (s *GPUFit) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	if pod.Labels == nil || pod.Labels[constants.LabelComponent] != constants.ComponentWorker {
		return framework.NewStatus(framework.Success)
	}

	allocRequest, reason, err := composeAllocationRequest(pod)
	if err != nil {
		return framework.NewStatus(framework.Unschedulable, reason)
	}

	// TODO judge nodeInfo with better simpler way rather than loop all nodes
	_, err = s.allocator.Filter(ctx, allocRequest)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	return framework.NewStatus(framework.Success, "")
}

func composeAllocationRequest(pod *v1.Pod) (gpuallocator.AllocRequest, string, error) {
	gpuResource, err := utils.GetGPUResource(pod, true)
	if err != nil {
		return gpuallocator.AllocRequest{}, "invalid gpu resource annotation", err
	}

	count, err := strconv.ParseUint(pod.Annotations[constants.GpuCountKey], 10, 64)
	if err != nil {
		return gpuallocator.AllocRequest{}, "invalid gpu count annotation", err
	}
	allocRequest := gpuallocator.AllocRequest{
		PoolName: pod.Annotations[constants.GpuPoolKey],
		Request:  gpuResource,

		Count:    uint(count),
		GPUModel: pod.Annotations[constants.GPUModelAnnotation],
		WorkloadNameNamespace: tfv1.NameNamespace{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
	}
	return allocRequest, "", nil
}

func (s *GPUFit) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	// TODO rank nodeInfo based on GPU resources
	return 0, nil
}

func (s *GPUFit) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

func (s *GPUFit) Reserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	allocRequest, reason, err := composeAllocationRequest(pod)
	if err != nil {
		return framework.NewStatus(framework.Unschedulable, reason)
	}
	// TODO bind to node and GPUs
	_, err = s.allocator.Bind(ctx, []*tfv1.GPU{}, allocRequest, "")
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	return framework.NewStatus(framework.Success, "")
}

func (s *GPUFit) Unreserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {

	allocRequest, reason, err := composeAllocationRequest(pod)
	if err != nil {
		s.logger.Error(err, "failed to compose allocation request", "pod", pod.Name, "reason", reason)
		return
	}
	// TODO get allocated GPU info as 4th param
	s.allocator.Dealloc(ctx, tfv1.NameNamespace{
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}, allocRequest.Request, []types.NamespacedName{}, pod.Name)
}
