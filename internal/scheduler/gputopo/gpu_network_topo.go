package scheduler

import (
	"context"
	"encoding/json"

	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const Name = "GPUNetworkTopologyAware"

var _ framework.FilterPlugin = &GPUNetworkTopologyAware{}

type GPUNetworkTopologyAware struct {
	logger    *klog.Logger
	fh        framework.Handle
	allocator *gpuallocator.GpuAllocator
	client    client.Client
	ctx       context.Context
	cfg       *config.GPUNetworkTopologyAwareConfig
}

type PluginFactoryFunc func(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error)

func NewWithDeps(allocator *gpuallocator.GpuAllocator, client client.Client) PluginFactoryFunc {
	return func(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error) {
		target := &config.GPUNetworkTopologyAwareConfig{}
		if unknown, ok := obj.(*runtime.Unknown); ok {
			if err := json.Unmarshal(unknown.Raw, target); err != nil {
				return nil, err
			}
		}
		lh := klog.FromContext(ctx).WithValues("plugin", Name)
		c := &GPUNetworkTopologyAware{
			logger:    &lh,
			fh:        handle,
			allocator: allocator,
			client:    client,
			cfg:       target,
			ctx:       ctx,
		}
		return c, nil
	}
}

func (s *GPUNetworkTopologyAware) Name() string {
	return Name
}

func (s *GPUNetworkTopologyAware) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	return framework.NewStatus(framework.Success, "")
}
