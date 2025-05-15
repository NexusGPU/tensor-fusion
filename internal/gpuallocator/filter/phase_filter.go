package filter

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/samber/lo"
)

// PhaseFilter filters GPUs based on their operational phase
type PhaseFilter struct {
	allowedPhases []tfv1.TensorFusionGPUPhase
}

// NewPhaseFilter creates a new PhaseFilter with the specified allowed phases
func NewPhaseFilter(allowedPhases ...tfv1.TensorFusionGPUPhase) *PhaseFilter {
	return &PhaseFilter{
		allowedPhases: allowedPhases,
	}
}

// Filter implements GPUFilter.Filter
func (f *PhaseFilter) Filter(_ context.Context, gpus []tfv1.GPU) ([]tfv1.GPU, error) {
	return lo.Filter(gpus, func(gpu tfv1.GPU, _ int) bool {
		return lo.Contains(f.allowedPhases, gpu.Status.Phase)
	}), nil
}
