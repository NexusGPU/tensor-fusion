package workload

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestRecommendationForWorkerPreservesNonTargetResources(t *testing.T) {
	state := NewWorkloadState()
	state.Spec.AutoScalingConfig.AutoSetResources = &tfv1.AutoSetResources{
		Enable:         true,
		TargetResource: tfv1.ScalingTargetResourceVRAM,
	}
	current := &tfv1.Resources{
		Requests: tfv1.Resource{
			Tflops: *resource.NewQuantity(10, resource.DecimalSI),
			Vram:   *resource.NewQuantity(10, resource.BinarySI),
		},
		Limits: tfv1.Resource{
			Tflops: *resource.NewQuantity(20, resource.DecimalSI),
			Vram:   *resource.NewQuantity(20, resource.BinarySI),
		},
	}
	recommendation := &tfv1.Resources{
		Requests: tfv1.Resource{
			Tflops: *resource.NewQuantity(99, resource.DecimalSI),
			Vram:   *resource.NewQuantity(30, resource.BinarySI),
		},
		Limits: tfv1.Resource{
			Tflops: *resource.NewQuantity(100, resource.DecimalSI),
			Vram:   *resource.NewQuantity(40, resource.BinarySI),
		},
	}

	got := recommendationForWorker(state, current, recommendation)
	if got.Requests.Tflops.Cmp(current.Requests.Tflops) != 0 || got.Limits.Tflops.Cmp(current.Limits.Tflops) != 0 {
		t.Fatalf("non-target TFLOPS changed: got request=%s limit=%s", got.Requests.Tflops.String(), got.Limits.Tflops.String())
	}
	if got.Requests.Vram.Cmp(recommendation.Requests.Vram) != 0 || got.Limits.Vram.Cmp(recommendation.Limits.Vram) != 0 {
		t.Fatalf("target VRAM changed unexpectedly: got request=%s limit=%s", got.Requests.Vram.String(), got.Limits.Vram.String())
	}
}
