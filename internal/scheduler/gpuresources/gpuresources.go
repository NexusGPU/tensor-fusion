package gpuresources

import (
	"context"
	"encoding/json"

	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
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
	return nil
}

func (s *GPUFit) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	return 0, nil
}

func (s *GPUFit) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

func (s *GPUFit) Reserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
	return nil
}

func (s *GPUFit) Unreserve(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) {
}
