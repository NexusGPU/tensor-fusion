package gpuresources

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestReserveNominatedSharedGPUs(t *testing.T) {
	candidates := []*tfv1.GPU{
		{ObjectMeta: metav1.ObjectMeta{Name: "gpu-a"}, Status: tfv1.GPUStatus{
			Capacity:  &tfv1.Resource{Tflops: resource.MustParse("100"), Vram: resource.MustParse("24Gi")},
			Available: &tfv1.Resource{Tflops: resource.MustParse("100"), Vram: resource.MustParse("24Gi")},
		}},
		{ObjectMeta: metav1.ObjectMeta{Name: "gpu-b"}, Status: tfv1.GPUStatus{
			Capacity:  &tfv1.Resource{Tflops: resource.MustParse("100"), Vram: resource.MustParse("24Gi")},
			Available: &tfv1.Resource{Tflops: resource.MustParse("100"), Vram: resource.MustParse("24Gi")},
		}},
	}

	remaining, reserved := reserveNominatedSharedGPUs(candidates, []*tfv1.AllocRequest{{Count: 1}}, nil)
	if reserved != 1 {
		t.Fatalf("reserved = %d, want 1", reserved)
	}
	if len(remaining) != 1 {
		t.Fatalf("remaining GPU count = %d, want 1", len(remaining))
	}
	if remaining[0].Name != "gpu-b" {
		t.Fatalf("remaining GPU = %q, want gpu-b", remaining[0].Name)
	}
}

func TestReserveNominatedSharedGPUsPrefersConcreteNames(t *testing.T) {
	candidates := []*tfv1.GPU{
		{ObjectMeta: metav1.ObjectMeta{Name: "gpu-a"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "gpu-b"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "gpu-c"}},
	}

	remaining, reserved := reserveNominatedSharedGPUs(candidates, []*tfv1.AllocRequest{
		{Count: 1, GPUNames: []string{"gpu-c"}},
	}, nil)
	if reserved != 1 {
		t.Fatalf("reserved = %d, want 1", reserved)
	}
	if len(remaining) != 2 {
		t.Fatalf("remaining GPU count = %d, want 2", len(remaining))
	}
	if remaining[0].Name != "gpu-a" || remaining[1].Name != "gpu-b" {
		t.Fatalf("remaining GPUs = [%q %q], want [gpu-a gpu-b]", remaining[0].Name, remaining[1].Name)
	}
}

func TestReserveNominatedSharedGPUsHandlesMultipleCards(t *testing.T) {
	candidates := []*tfv1.GPU{
		{ObjectMeta: metav1.ObjectMeta{Name: "gpu-a"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "gpu-b"}},
		{ObjectMeta: metav1.ObjectMeta{Name: "gpu-c"}},
	}

	remaining, reserved := reserveNominatedSharedGPUs(candidates, []*tfv1.AllocRequest{{Count: 2}}, nil)
	if reserved != 2 {
		t.Fatalf("reserved = %d, want 2", reserved)
	}
	if len(remaining) != 1 {
		t.Fatalf("remaining GPU count = %d, want 1", len(remaining))
	}
}
