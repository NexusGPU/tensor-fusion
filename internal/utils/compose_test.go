package utils_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

func TestAddTFHypervisorConfAfterTemplate(t *testing.T) {
	tests := []struct {
		name              string
		enableVector      bool
		hypervisorImage   string
		expectInitCount   int
		expectVolumeCount int
	}{
		{
			name:              "without vector",
			enableVector:      false,
			hypervisorImage:   "test-image:latest",
			expectInitCount:   2, // init-shm + init-runtime
			expectVolumeCount: 7,
		},
		{
			name:              "with vector",
			enableVector:      true,
			hypervisorImage:   "test-image:latest",
			expectInitCount:   2, // init-shm + init-runtime
			expectVolumeCount: 7,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			spec := &corev1.PodSpec{}
			pool := &tfv1.GPUPool{
				Spec: tfv1.GPUPoolSpec{
					ComponentConfig: &tfv1.ComponentConfig{
						Hypervisor: &tfv1.HypervisorConfig{
							Image:        tt.hypervisorImage,
							EnableVector: tt.enableVector,
						},
					},
				},
			}

			utils.AddTFHypervisorConfAfterTemplate(ctx, spec, pool, "NVIDIA", false)

			require.Len(t, spec.InitContainers, tt.expectInitCount, "unexpected number of init containers")
			require.Len(t, spec.Volumes, tt.expectVolumeCount, "unexpected number of volumes")
			require.True(t, spec.HostPID)
			require.NotNil(t, spec.TerminationGracePeriodSeconds)
		})
	}
}

func TestSetWorkerContainerSpec(t *testing.T) {
	tests := []struct {
		name             string
		workerImage      string
		disabledFeatures string
		sharedMemMode    bool
		expectCommand    []string
	}{
		{
			name:             "basic worker config",
			workerImage:      "worker:latest",
			disabledFeatures: "",
			sharedMemMode:    false,
			expectCommand: []string{
				"./tensor-fusion-worker",
				"-p",
				"8000",
			},
		},
		{
			name:             "worker with shared memory mode",
			workerImage:      "worker:latest",
			disabledFeatures: "",
			sharedMemMode:    true,
			expectCommand: []string{
				"/bin/bash",
				"-c",
				"touch /dev/shm/tf_shm && chmod 666 /dev/shm/tf_shm && exec ./tensor-fusion-worker -n shmem -m tf_shm -M 256",
			},
		},
		{
			name:             "worker with disabled start-worker feature",
			workerImage:      "worker:latest",
			disabledFeatures: "start-worker",
			sharedMemMode:    false,
			expectCommand: []string{
				"sleep",
				"infinity",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			container := &corev1.Container{}
			workloadProfile := &tfv1.WorkloadProfileSpec{}
			workerConfig := &tfv1.WorkerConfig{
				Image: tt.workerImage,
			}
			hypervisorConfig := &tfv1.HypervisorConfig{}

			utils.SetWorkerContainerSpec(container, workloadProfile, workerConfig, hypervisorConfig, tt.disabledFeatures, tt.sharedMemMode)

			require.NotEmpty(t, container.Name)
			if tt.workerImage != "" {
				require.Equal(t, tt.workerImage, container.Image)
			}

			// Verify command is set correctly
			require.NotEmpty(t, container.Command, "container command should not be empty")
			require.Equal(t, tt.expectCommand, container.Command, "container command should match expected value")

			// Verify shared memory mode specific setup
			if tt.sharedMemMode && tt.disabledFeatures == "" {
				require.Len(t, container.Command, 3, "shared memory mode should use bash -c with script")
				require.Equal(t, "/bin/bash", container.Command[0], "should use bash")
				require.Equal(t, "-c", container.Command[1], "should use -c flag")
				require.Contains(t, container.Command[2], "touch /dev/shm/tf_shm", "should create shared memory file")
				require.Contains(t, container.Command[2], "chmod 666 /dev/shm/tf_shm", "should set file permissions")
				require.Contains(t, container.Command[2], "exec ./tensor-fusion-worker", "should exec worker")
				require.Contains(t, container.Command[2], "-n shmem", "should use shmem mode")
				require.Contains(t, container.Command[2], "-m tf_shm", "should specify shared memory name")
				require.Contains(t, container.Command[2], "-M 256", "should specify shared memory size")
			}
		})
	}
}
