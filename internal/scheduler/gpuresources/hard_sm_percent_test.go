package gpuresources

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestEffectiveHardSMPercentUsesGPUCRCapacity(t *testing.T) {
	t.Parallel()

	percent, err := effectiveHardSMPercent(resource.MustParse("10"), []*tfv1.GPU{
		{
			Status: tfv1.GPUStatus{
				Capacity: &tfv1.Resource{Tflops: resource.MustParse("71")},
			},
		},
	})
	if err != nil {
		t.Fatalf("calculate hard SM percent: %v", err)
	}
	if percent != 15 {
		t.Fatalf("hard SM percent = %d, want 15", percent)
	}
}
