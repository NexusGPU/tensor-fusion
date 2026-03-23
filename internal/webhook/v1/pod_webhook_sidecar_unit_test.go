package v1

import (
	"fmt"
	"testing"

	corev1 "k8s.io/api/core/v1"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

func TestAddConnectionForLocalSidecarWorker(t *testing.T) {
	container := &corev1.Container{
		Env: []corev1.EnvVar{
			{
				Name:  "UNCHANGED",
				Value: "keep",
			},
			{
				Name:  constants.ConnectionInfoEnv,
				Value: "stale",
			},
		},
	}

	addConnectionForLocalSidecarWorker(container)

	expectedConnection := fmt.Sprintf(
		"shmem+%s+%s+1",
		constants.ConnectionSharedMemName,
		constants.ConnectionSharedMemSize,
	)

	if got := findEnvValue(container.Env, constants.ConnectionInfoEnv); got != expectedConnection {
		t.Fatalf("unexpected %s: got %q want %q", constants.ConnectionInfoEnv, got, expectedConnection)
	}

	if got := findEnvValue(container.Env, constants.DisableVMSharedMemEnv); got != "0" {
		t.Fatalf("unexpected %s: got %q want %q", constants.DisableVMSharedMemEnv, got, "0")
	}

	if got := findEnvValue(container.Env, "UNCHANGED"); got != "keep" {
		t.Fatalf("unexpected UNCHANGED env value: got %q want %q", got, "keep")
	}
}

func findEnvValue(envs []corev1.EnvVar, name string) string {
	for _, env := range envs {
		if env.Name == name {
			return env.Value
		}
	}
	return ""
}
