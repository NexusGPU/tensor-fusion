package utils

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func vectorSidecar(t *testing.T, pool *tfv1.GPUPool) v1.Container {
	t.Helper()
	spec := v1.PodSpec{}
	AddTFHypervisorConfAfterTemplate(context.Background(), &spec, pool, "", false)
	for _, container := range spec.Containers {
		if container.Name == constants.TFContainerVector {
			return container
		}
	}
	t.Fatal("hypervisor pod has no Vector sidecar")
	return v1.Container{}
}

// The database password reaches the hypervisor's Vector sidecar by reference
// to the Secret the operator was told about, and only then: without one the
// sidecar is composed as before.
func TestVectorSidecarTakesTheDatabasePasswordFromASecret(t *testing.T) {
	pool := goldenHashPool()
	pool.Spec.ComponentConfig.Hypervisor.EnableVector = true
	pool.Spec.ComponentConfig.Hypervisor.PodTemplate = nil

	t.Setenv(constants.VectorGreptimePasswordSecretEnv, "")
	for _, env := range vectorSidecar(t, pool).Env {
		if env.Name == constants.VectorGreptimePasswordEnv {
			t.Fatalf("no Secret was named, but the sidecar refers to one: %+v", env)
		}
	}

	t.Setenv(constants.VectorGreptimePasswordSecretEnv, "release-greptimedb-secret")
	sidecar := vectorSidecar(t, pool)
	found := false
	for _, env := range sidecar.Env {
		if env.Name != constants.VectorGreptimePasswordEnv {
			continue
		}
		found = true
		if env.Value != "" || env.ValueFrom == nil || env.ValueFrom.SecretKeyRef == nil {
			t.Fatalf("the password must be a Secret reference, got %+v", env)
		}
		if ref := env.ValueFrom.SecretKeyRef; ref.Name != "release-greptimedb-secret" || ref.Key != constants.GreptimePasswordSecretKey {
			t.Fatalf("wrong Secret reference: %+v", ref)
		}
	}
	if !found {
		t.Fatal("the sidecar does not get the password variable")
	}

	// Both Vector sidecars have the same memory limit: this one by default,
	// the controller's in the chart.
	if got, want := sidecar.Resources.Limits[v1.ResourceMemory], resource.MustParse("512Mi"); !got.Equal(want) {
		t.Errorf("memory limit of the sidecar is %s, want %s", got.String(), want.String())
	}
}
