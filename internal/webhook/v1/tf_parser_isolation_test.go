package v1

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestValidateIsolationAndExecutionMode(t *testing.T) {
	tests := []struct {
		name    string
		profile *tfv1.WorkloadProfileSpec
		wantErr bool
	}{
		{
			name: "hard remote is allowed",
			profile: &tfv1.WorkloadProfileSpec{
				Isolation:  tfv1.IsolationModeHard,
				IsLocalGPU: false,
			},
			wantErr: false,
		},
		{
			name: "hard local without sidecar is allowed",
			profile: &tfv1.WorkloadProfileSpec{
				Isolation:     tfv1.IsolationModeHard,
				IsLocalGPU:    true,
				SidecarWorker: false,
			},
			wantErr: false,
		},
		{
			name: "hard local sidecar is allowed",
			profile: &tfv1.WorkloadProfileSpec{
				Isolation:     tfv1.IsolationModeHard,
				IsLocalGPU:    true,
				SidecarWorker: true,
			},
			wantErr: false,
		},
		{
			name: "soft local is allowed",
			profile: &tfv1.WorkloadProfileSpec{
				Isolation:  tfv1.IsolationModeSoft,
				IsLocalGPU: true,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateIsolationAndExecutionMode(tt.profile)
			if tt.wantErr && err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("expected no error, got %v", err)
			}
		})
	}
}

func TestParseNativeGPUResourceForSharedUsesCountOnly(t *testing.T) {
	originalManager := provider.GetManager()
	manager := provider.NewManager(nil)
	manager.UpdateProvider(&tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "nvidia-provider"},
		Spec: tfv1.ProviderConfigSpec{
			Vendor:             constants.AcceleratorVendorNvidia,
			InUseResourceNames: []string{string(constants.NvidiaGPUKey)},
		},
	})
	provider.SetGlobalManagerForTesting(manager)
	t.Cleanup(func() { provider.SetGlobalManagerForTesting(originalManager) })

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}},
		Spec: corev1.PodSpec{Containers: []corev1.Container{
			{
				Name: "main",
				Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
					constants.NvidiaGPUKey: resource.MustParse("2"),
				}},
			},
		}},
	}
	profile := &tfv1.WorkloadProfile{Spec: tfv1.WorkloadProfileSpec{Isolation: tfv1.IsolationModeShared}}

	if err := parseGPUResourcesAnnotations(pod, profile); err != nil {
		t.Fatalf("parseGPUResourcesAnnotations() error = %v", err)
	}
	if profile.Spec.GPUCount != 2 {
		t.Fatalf("GPUCount = %d, want 2", profile.Spec.GPUCount)
	}
	if !profile.Spec.Resources.Requests.Tflops.IsZero() ||
		!profile.Spec.Resources.Requests.ComputePercent.IsZero() ||
		!profile.Spec.Resources.Requests.Vram.IsZero() ||
		!profile.Spec.Resources.Limits.Tflops.IsZero() ||
		!profile.Spec.Resources.Limits.ComputePercent.IsZero() ||
		!profile.Spec.Resources.Limits.Vram.IsZero() {
		t.Fatalf("shared native GPU migration should use count only, got resources %#v", profile.Spec.Resources)
	}
	if got := pod.Annotations[constants.ContainerGPUCountAnnotation]; got != `{"main":2}` {
		t.Fatalf("container GPU counts = %q, want %q", got, `{"main":2}`)
	}
}
