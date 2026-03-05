package controller

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestBuildProviderHypervisorExtraEnv(t *testing.T) {
	t.Helper()

	s := runtime.NewScheme()
	obj := &unstructured.Unstructured{
		Object: map[string]any{
			"apiVersion": "tensor-fusion.ai/v1",
			"kind":       "ProviderConfig",
			"metadata": map[string]any{
				"name": "ascend-provider",
			},
			"spec": map[string]any{
				"hypervisor": map[string]any{
					"extraEnv": []any{
						map[string]any{
							"name":  "ASCEND_RUNTIME_OPTIONS",
							"value": "VIRTUAL",
						},
						map[string]any{
							"name": "ASCEND_VISIBLE_DEVICES",
						},
						map[string]any{
							"name": "   ",
						},
					},
				},
			},
		},
	}
	obj.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   "tensor-fusion.ai",
		Version: "v1",
		Kind:    "ProviderConfig",
	})

	r := &GPUNodeReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).WithRuntimeObjects(obj).Build(),
	}

	got := r.buildProviderHypervisorExtraEnv(context.Background(), "ascend-provider")
	want := []corev1.EnvVar{
		{
			Name:  "ASCEND_RUNTIME_OPTIONS",
			Value: "VIRTUAL",
		},
		{
			Name:  "ASCEND_VISIBLE_DEVICES",
			Value: "",
		},
	}

	if len(got) != len(want) {
		t.Fatalf("unexpected env count: got=%d want=%d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("env mismatch at %d: got=%+v want=%+v", i, got[i], want[i])
		}
	}
}

func TestBuildProviderHypervisorExtraEnvNotFound(t *testing.T) {
	t.Helper()

	s := runtime.NewScheme()
	r := &GPUNodeReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).WithRuntimeObjects(&unstructured.Unstructured{
			Object: map[string]any{
				"apiVersion": "tensor-fusion.ai/v1",
				"kind":       "ProviderConfig",
				"metadata": map[string]any{
					"name": "other-provider",
				},
			},
		}).Build(),
	}

	got := r.buildProviderHypervisorExtraEnv(context.Background(), "ascend-provider")
	if got != nil {
		t.Fatalf("expected nil envs when provider not found, got=%v", got)
	}
}

func TestBuildProviderHypervisorExtraEnvEmptyName(t *testing.T) {
	t.Helper()

	s := runtime.NewScheme()
	r := &GPUNodeReconciler{
		Client: fake.NewClientBuilder().WithScheme(s).Build(),
	}

	got := r.buildProviderHypervisorExtraEnv(context.Background(), " ")
	if got != nil {
		t.Fatalf("expected nil envs for empty provider name, got=%v", got)
	}
}
