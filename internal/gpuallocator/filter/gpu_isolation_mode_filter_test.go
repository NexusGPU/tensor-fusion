package filter

import (
	"context"
	"encoding/json"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
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

func mustMarshalCaps(t *testing.T, caps gpuVirtualizationCapabilities) string {
	t.Helper()
	data, err := json.Marshal(caps)
	if err != nil {
		t.Fatalf("marshal caps: %v", err)
	}
	return string(data)
}
