package scheduler

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestFilters(t *testing.T) {
	// Create test GPUs
	gpus := []tfv1.GPU{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "gpu-1",
				Labels: map[string]string{
					"location": "us-west",
				},
			},
			Status: tfv1.GPUStatus{
				Phase:    tfv1.TensorFusionGPUPhaseRunning,
				GPUModel: "A100",
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("10"),
					Vram:   resource.MustParse("40Gi"),
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "gpu-2",
				Labels: map[string]string{
					"location": "us-east",
				},
			},
			Status: tfv1.GPUStatus{
				Phase:    tfv1.TensorFusionGPUPhaseRunning,
				GPUModel: "A100",
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("5"),
					Vram:   resource.MustParse("20Gi"),
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "gpu-3",
				Labels: map[string]string{
					"location": "us-west",
				},
			},
			Status: tfv1.GPUStatus{
				Phase:    tfv1.TensorFusionGPUPhasePending,
				GPUModel: "H100",
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("20"),
					Vram:   resource.MustParse("80Gi"),
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "gpu-4",
				Labels: map[string]string{
					"location": "eu-west",
				},
			},
			Status: tfv1.GPUStatus{
				Phase:    tfv1.TensorFusionGPUPhaseRunning,
				GPUModel: "RTX4090",
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("3"),
					Vram:   resource.MustParse("24Gi"),
				},
			},
		},
	}

	ctx := context.Background()

	t.Run("PhaseFilter", func(t *testing.T) {
		filter := NewPhaseFilter(tfv1.TensorFusionGPUPhaseRunning)
		result, err := filter.Filter(ctx, gpus)
		assert.NoError(t, err)
		assert.Len(t, result, 3)
		for _, gpu := range result {
			assert.Equal(t, tfv1.TensorFusionGPUPhaseRunning, gpu.Status.Phase)
		}
	})

	t.Run("ResourceFilter", func(t *testing.T) {
		filter := NewResourceFilter(tfv1.Resource{
			Tflops: resource.MustParse("8"),
			Vram:   resource.MustParse("30Gi"),
		})
		result, err := filter.Filter(ctx, gpus)
		assert.NoError(t, err)
		assert.Len(t, result, 2)
		// Should include gpu-1 and gpu-3
		assert.ElementsMatch(t, []string{"gpu-1", "gpu-3"}, []string{result[0].Name, result[1].Name})
	})

	t.Run("FilterRegistry with multiple filters", func(t *testing.T) {
		// Create registry and chain filters with With method
		registry := NewFilterRegistry().
			With(NewPhaseFilter(tfv1.TensorFusionGPUPhaseRunning)).
			With(NewResourceFilter(tfv1.Resource{
				Tflops: resource.MustParse("8"),
				Vram:   resource.MustParse("30Gi"),
			}))

		// Apply filters
		result, err := registry.Apply(ctx, gpus)
		assert.NoError(t, err)
		assert.Len(t, result, 1)
		assert.Equal(t, "gpu-1", result[0].Name)
	})

	t.Run("FilterRegistry immutability", func(t *testing.T) {
		// Test that the original registry is not modified
		baseRegistry := NewFilterRegistry().
			With(NewPhaseFilter(tfv1.TensorFusionGPUPhaseRunning))

		// Create a new registry with additional filter
		extendedRegistry := baseRegistry.
			With(NewResourceFilter(tfv1.Resource{
				Tflops: resource.MustParse("8"),
				Vram:   resource.MustParse("30Gi"),
			}))

		// Apply base registry filters
		baseResult, err := baseRegistry.Apply(ctx, gpus)
		assert.NoError(t, err)
		assert.Len(t, baseResult, 3) // Only phase filter applied

		// Apply extended registry filters
		extendedResult, err := extendedRegistry.Apply(ctx, gpus)
		assert.NoError(t, err)
		assert.Len(t, extendedResult, 1) // Phase and model filters applied
	})
}

// Example of a custom filter implementation
type CustomFilter struct {
	// Custom filter properties
}

func (f *CustomFilter) Filter(_ context.Context, gpus []tfv1.GPU) ([]tfv1.GPU, error) {
	// Custom filtering logic
	return gpus, nil
}
