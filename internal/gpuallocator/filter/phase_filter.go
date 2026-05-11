package filter

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/samber/lo"
	"sigs.k8s.io/controller-runtime/pkg/log"
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
func (f *PhaseFilter) Filter(ctx context.Context, workerPodKey tfv1.NameNamespace, gpus []*tfv1.GPU) ([]*tfv1.GPU, error) {
	validPhase := 0
	filteredGPUs := lo.Filter(gpus, func(gpu *tfv1.GPU, _ int) bool {
		// Exclude GPUs claimed by external device plugins (e.g. nvidia-device-plugin, HAMI).
		// In progressive migration mode, only GPUs explicitly owned by tensor-fusion are eligible.
		// In normal mode, GPUs with a non-empty UsedBy that isn't tensor-fusion are also excluded
		// to prevent double-scheduling with external provisioners.
		if gpu.Status.UsedBy != "" && gpu.Status.UsedBy != tfv1.UsedByTensorFusion {
			return false
		}

		ok := lo.Contains(f.allowedPhases, gpu.Status.Phase)
		if ok {
			validPhase = validPhase + 1
		}
		return ok
	})
	log.FromContext(ctx).V(6).Info("PhaseFilter", "validPhase", validPhase, "total", len(gpus), "workerPodKey", workerPodKey)
	return filteredGPUs, nil
}

func (f *PhaseFilter) Name() string {
	return "PhaseFilter"
}
