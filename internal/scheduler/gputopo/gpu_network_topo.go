package scheduler

import (
	"context"

	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

const Name = "GPUNetworkTopologyAware"

var _ framework.FilterPlugin = &GPUNetworkTopologyAware{}
var _ framework.ScorePlugin = &GPUNetworkTopologyAware{}
var _ framework.ReservePlugin = &GPUNetworkTopologyAware{}

type GPUNetworkTopologyAware struct {
	logger    *klog.Logger
	fh        framework.Handle
	allocator *gpuallocator.GpuAllocator
	podLister cache.Indexer
	pdbLister cache.Indexer
	client    *rest.RESTClient
}

type PluginFactoryFunc func(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error)

func NewWithDeps(allocator *gpuallocator.GpuAllocator) PluginFactoryFunc {
	return func(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error) {
		lh := klog.FromContext(ctx).WithValues("plugin", Name)
		c := &GPUNetworkTopologyAware{
			logger:    &lh,
			fh:        handle,
			allocator: allocator,
		}
		return c, nil
	}
}

func (s *GPUNetworkTopologyAware) Name() string {
	return Name
}

func (s *GPUNetworkTopologyAware) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	return 0, nil
}

func (s *GPUNetworkTopologyAware) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

func (s *GPUNetworkTopologyAware) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	return nil
}

func (s *GPUNetworkTopologyAware) Reserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	return nil
}

func (s *GPUNetworkTopologyAware) Unreserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
}
