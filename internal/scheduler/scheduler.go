package scheduler

import (
	"context"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type Strategy interface {
	SelectGPU(gpus []tfv1.GPU) (*tfv1.GPU, error)
}

// NewStrategy creates a strategy based on the placement mode
func NewStrategy(placementMode tfv1.PlacementMode) Strategy {
	switch placementMode {
	case tfv1.PlacementModeLowLoadFirst:
		return LowLoadFirst{}
	default:
		// CompactFirst is the default strategy
		return CompactFirst{}
	}
}

type Scheduler struct {
	client.Client
	filterRegistry *FilterRegistry
}

func NewScheduler(client client.Client) Scheduler {
	// Create base filter registry with common filters
	baseRegistry := NewFilterRegistry().With(
		NewPhaseFilter(tfv1.TensorFusionGPUPhaseRunning),
	)

	return Scheduler{
		Client:         client,
		filterRegistry: baseRegistry,
	}
}

// Schedule schedules a request to a gpu.
func (s *Scheduler) Schedule(ctx context.Context, poolName string, request tfv1.Resource) (*tfv1.GPU, error) {
	gpus := &tfv1.GPUList{}

	if err := s.List(ctx, gpus, client.MatchingLabels{constants.GpuPoolKey: poolName}); err != nil {
		return nil, fmt.Errorf("list GPUs: %w", err)
	}

	filterRegistry := s.filterRegistry.With(NewResourceFilter(request))

	// Apply the filters in sequence
	filteredGPUs, err := filterRegistry.Apply(ctx, gpus.Items)
	if err != nil {
		return nil, fmt.Errorf("apply filters: %w", err)
	}

	if len(filteredGPUs) == 0 {
		return nil, fmt.Errorf("no gpus available in pool %s after filtering", poolName)
	}

	pool := &tfv1.GPUPool{}
	if err := s.Get(ctx, client.ObjectKey{Name: poolName}, pool); err != nil {
		return nil, fmt.Errorf("get pool %s: %w", poolName, err)
	}
	schedulingConfigTemplate := &tfv1.SchedulingConfigTemplate{}
	if pool.Spec.SchedulingConfigTemplate != nil {
		if err := s.Get(ctx, client.ObjectKey{Name: *pool.Spec.SchedulingConfigTemplate}, schedulingConfigTemplate); err != nil {
			return nil, fmt.Errorf("get scheduling config template %s: %w", *pool.Spec.SchedulingConfigTemplate, err)
		}
	}

	strategy := NewStrategy(schedulingConfigTemplate.Spec.Placement.Mode)
	gpu, err := strategy.SelectGPU(filteredGPUs)
	if err != nil {
		return nil, fmt.Errorf("select GPU: %w", err)
	}

	// reduce available resource on the GPU status

	gpu.Status.Available.Tflops.Sub(request.Tflops)
	gpu.Status.Available.Vram.Sub(request.Vram)

	// update the GPU object
	if err := s.Status().Update(ctx, gpu); err != nil {
		return nil, fmt.Errorf("update GPU: %w", err)
	}

	return gpu, nil
}

// Release releases a request from a gpu.
func (s *Scheduler) Release(ctx context.Context, request tfv1.Resource, gpu *tfv1.GPU) error {
	gpu.Status.Available.Tflops.Add(request.Tflops)
	gpu.Status.Available.Vram.Add(request.Vram)

	// update the GPU object
	if err := s.Status().Update(ctx, gpu); err != nil {
		return fmt.Errorf("update GPU: %w", err)
	}
	return nil
}
