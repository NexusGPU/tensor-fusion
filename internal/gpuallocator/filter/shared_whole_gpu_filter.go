package filter

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/samber/lo"
)

// SharedWholeGPUFilter only keeps completely idle, unpartitioned physical GPUs.
// Assumed allocations are already applied to the GPU copies before filters run,
// so an in-flight reservation also makes Available differ from Capacity.
type SharedWholeGPUFilter struct{}

func NewSharedWholeGPUFilter() *SharedWholeGPUFilter {
	return &SharedWholeGPUFilter{}
}

func (f *SharedWholeGPUFilter) Filter(
	_ context.Context,
	_ tfv1.NameNamespace,
	gpus []*tfv1.GPU,
) ([]*tfv1.GPU, error) {
	return lo.Filter(gpus, func(gpu *tfv1.GPU, _ int) bool {
		if gpu == nil || gpu.Status.Capacity == nil || gpu.Status.Available == nil {
			return false
		}
		if len(gpu.Status.AllocatedPartitions) > 0 {
			return false
		}
		return gpu.Status.Available.Tflops.Equal(gpu.Status.Capacity.Tflops) &&
			gpu.Status.Available.Vram.Equal(gpu.Status.Capacity.Vram)
	}), nil
}

func (f *SharedWholeGPUFilter) Name() string {
	return "SharedWholeGPUFilter"
}
