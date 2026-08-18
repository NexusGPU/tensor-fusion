package filter

import (
	"context"
	"encoding/json"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGPUIsolationModeFilter_RespectsCapabilities(t *testing.T) {
	hardSupported := mustMarshalCaps(t, gpuVirtualizationCapabilities{
		SupportsHardIsolation: true,
	})
	hardUnsupported := mustMarshalCaps(t, gpuVirtualizationCapabilities{
		SupportsPartitioning: true,
	})

	filter := NewGPUIsolationModeFilter(tfv1.IsolationModeHard)
	gpus := []*tfv1.GPU{
		{
			Status: tfv1.GPUStatus{
				UUID:          "gpu-hard-supported",
				IsolationMode: tfv1.IsolationModeHard,
			},
		},
		{
			Status: tfv1.GPUStatus{
				UUID:          "gpu-hard-unsupported",
				IsolationMode: tfv1.IsolationModeHard,
			},
		},
		{
			Status: tfv1.GPUStatus{
				UUID:          "gpu-mode-mismatch",
				IsolationMode: tfv1.IsolationModeSoft,
			},
		},
	}
	gpus[0].Annotations = map[string]string{constants.GPUVirtualizationCapabilitiesAnnotation: hardSupported}
	gpus[1].Annotations = map[string]string{constants.GPUVirtualizationCapabilitiesAnnotation: hardUnsupported}
	gpus[2].Annotations = map[string]string{constants.GPUVirtualizationCapabilitiesAnnotation: hardSupported}

	filtered, err := filter.Filter(context.Background(), tfv1.NameNamespace{}, gpus)
	if err != nil {
		t.Fatalf("filter returned error: %v", err)
	}
	if len(filtered) != 1 {
		t.Fatalf("expected 1 gpu, got %d", len(filtered))
	}
	if filtered[0].Status.UUID != "gpu-hard-supported" {
		t.Fatalf("unexpected gpu selected: %s", filtered[0].Status.UUID)
	}
}

func TestGPUIsolationModeFilter_BackwardCompatibilityWithoutCapabilities(t *testing.T) {
	filter := NewGPUIsolationModeFilter(tfv1.IsolationModeSoft)
	gpus := []*tfv1.GPU{
		{
			Status: tfv1.GPUStatus{
				UUID:          "gpu-legacy",
				IsolationMode: tfv1.IsolationModeSoft,
			},
		},
	}

	filtered, err := filter.Filter(context.Background(), tfv1.NameNamespace{}, gpus)
	if err != nil {
		t.Fatalf("filter returned error: %v", err)
	}
	if len(filtered) != 1 {
		t.Fatalf("expected legacy gpu to pass when capabilities are missing")
	}
}

func TestGPUIsolationModeFilter_ExplicitUnsupportedIsolation(t *testing.T) {
	unsupportedAll := mustMarshalCaps(t, gpuVirtualizationCapabilities{})
	filter := NewGPUIsolationModeFilter(tfv1.IsolationModeHard)
	gpus := []*tfv1.GPU{
		{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					constants.GPUVirtualizationCapabilitiesAnnotation: unsupportedAll,
				},
			},
			Status: tfv1.GPUStatus{
				UUID:          "gpu-explicit-unsupported",
				IsolationMode: tfv1.IsolationModeHard,
			},
		},
	}

	filtered, err := filter.Filter(context.Background(), tfv1.NameNamespace{}, gpus)
	if err != nil {
		t.Fatalf("filter returned error: %v", err)
	}
	if len(filtered) != 0 {
		t.Fatalf("expected gpu to be rejected when annotation explicitly marks capabilities unsupported")
	}
}

func TestGPUIsolationModeFilter_SharedAcceptsAnyNodeRuntimeMode(t *testing.T) {
	filter := NewGPUIsolationModeFilter(tfv1.IsolationModeShared)
	gpus := []*tfv1.GPU{
		{Status: tfv1.GPUStatus{UUID: "shared", IsolationMode: tfv1.IsolationModeShared}},
		{Status: tfv1.GPUStatus{UUID: "soft", IsolationMode: tfv1.IsolationModeSoft}},
		{Status: tfv1.GPUStatus{UUID: "hard", IsolationMode: tfv1.IsolationModeHard}},
		{Status: tfv1.GPUStatus{UUID: "partitioned", IsolationMode: tfv1.IsolationModePartitioned}},
	}

	filtered, err := filter.Filter(context.Background(), tfv1.NameNamespace{}, gpus)
	if err != nil {
		t.Fatalf("filter returned error: %v", err)
	}
	if len(filtered) != len(gpus) {
		t.Fatalf("expected all runtime modes to be eligible for shared, got %d", len(filtered))
	}
}

func TestGPUIsolationModeFilter_DynamicUsesRecoveredActiveMode(t *testing.T) {
	filter := NewGPUIsolationModeFilter(tfv1.IsolationModeSoft)
	gpus := []*tfv1.GPU{
		{Status: tfv1.GPUStatus{UUID: "idle", IsolationPolicy: tfv1.IsolationModePolicyDynamic}},
		{Status: tfv1.GPUStatus{UUID: "soft", IsolationPolicy: tfv1.IsolationModePolicyDynamic, ActiveIsolationMode: tfv1.IsolationModeSoft}},
		{Status: tfv1.GPUStatus{UUID: "hard", IsolationPolicy: tfv1.IsolationModePolicyDynamic, ActiveIsolationMode: tfv1.IsolationModeHard}},
		{Status: tfv1.GPUStatus{UUID: "conflict", IsolationPolicy: tfv1.IsolationModePolicyDynamic, DynamicIsolationConflict: true}},
	}
	filtered, err := filter.Filter(context.Background(), tfv1.NameNamespace{}, gpus)
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered) != 2 || filtered[0].Status.UUID != "idle" || filtered[1].Status.UUID != "soft" {
		t.Fatalf("unexpected Dynamic candidates: %#v", filtered)
	}
}

func TestSharedWholeGPUFilter_OnlyKeepsIdleUnpartitionedGPUs(t *testing.T) {
	res := func(tflops, vram string) *tfv1.Resource {
		return &tfv1.Resource{Tflops: resource.MustParse(tflops), Vram: resource.MustParse(vram)}
	}
	gpus := []*tfv1.GPU{
		{ObjectMeta: metav1.ObjectMeta{Name: "idle"}, Status: tfv1.GPUStatus{Capacity: res("100", "24Gi"), Available: res("100", "24Gi")}},
		{ObjectMeta: metav1.ObjectMeta{Name: "used"}, Status: tfv1.GPUStatus{Capacity: res("100", "24Gi"), Available: res("80", "20Gi")}},
		{ObjectMeta: metav1.ObjectMeta{Name: "partitioned"}, Status: tfv1.GPUStatus{
			Capacity: res("100", "24Gi"), Available: res("100", "24Gi"),
			AllocatedPartitions: map[string]tfv1.AllocatedPartition{"pod": {PodUID: "pod"}},
		}},
	}

	filtered, err := NewSharedWholeGPUFilter().Filter(context.Background(), tfv1.NameNamespace{}, gpus)
	if err != nil {
		t.Fatalf("filter returned error: %v", err)
	}
	if len(filtered) != 1 || filtered[0].Name != "idle" {
		t.Fatalf("expected only idle GPU, got %#v", filtered)
	}
}

func mustMarshalCaps(t *testing.T, caps gpuVirtualizationCapabilities) string {
	t.Helper()
	data, err := json.Marshal(caps)
	if err != nil {
		t.Fatalf("marshal caps: %v", err)
	}
	return string(data)
}
