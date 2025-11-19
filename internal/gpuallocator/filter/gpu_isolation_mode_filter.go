package filter

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// GPUIsolationModeFilter filters GPUs based on their isolation mode
type GPUIsolationModeFilter struct {
	requiredIsolationMode tfv1.IsolationModeType
}

// NewGPUIsolationModeFilter creates a new filter that matches GPUs with the specified isolation mode
func NewGPUIsolationModeFilter(isolationMode tfv1.IsolationModeType) *GPUIsolationModeFilter {
	return &GPUIsolationModeFilter{
		requiredIsolationMode: isolationMode,
	}
}

// Filter implements GPUFilter interface
func (f *GPUIsolationModeFilter) Filter(ctx context.Context, workerPodKey tfv1.NameNamespace, gpus []*tfv1.GPU) ([]*tfv1.GPU, error) {
	if f.requiredIsolationMode == "" {
		return gpus, nil
	}

	filtered := make([]*tfv1.GPU, 0, len(gpus))
	for _, gpu := range gpus {
		if gpu.Status.IsolationMode == "" || gpu.Status.IsolationMode == f.requiredIsolationMode {
			filtered = append(filtered, gpu)
		}
	}
	return filtered, nil
}

func (f *GPUIsolationModeFilter) Name() string {
	return "GPUIsolationModeFilter"
}
