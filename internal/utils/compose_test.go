package utils_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
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
			expectInitCount:   1,
			expectVolumeCount: 7,
		},
		{
			name:              "with vector",
			enableVector:      true,
			hypervisorImage:   "test-image:latest",
			expectInitCount:   1,
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

			utils.AddTFHypervisorConfAfterTemplate(ctx, spec, pool)

			require.Len(t, spec.InitContainers, tt.expectInitCount, "unexpected number of init containers")
			require.Len(t, spec.Volumes, tt.expectVolumeCount, "unexpected number of volumes")
			require.True(t, spec.HostPID)
			require.NotNil(t, spec.TerminationGracePeriodSeconds)
		})
	}
}

func TestAddTFNodeDiscoveryConfAfterTemplate(t *testing.T) {
	tests := []struct {
		name                                 string
		compatibleWithNvidiaContainerToolkit bool
		nodeDiscoveryImage                   string
		gpuNodeName                          string
		expectInitContainers                 int
		validateToolkitInit                  bool
	}{
		{
			name:                                 "with nvidia toolkit validation",
			compatibleWithNvidiaContainerToolkit: true,
			nodeDiscoveryImage:                   "node-discovery:latest",
			gpuNodeName:                          "test-node",
			expectInitContainers:                 1,
			validateToolkitInit:                  true,
		},
		{
			name:                                 "without nvidia toolkit validation",
			compatibleWithNvidiaContainerToolkit: false,
			nodeDiscoveryImage:                   "node-discovery:latest",
			gpuNodeName:                          "test-node",
			expectInitContainers:                 0,
			validateToolkitInit:                  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			tmpl := &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{},
			}
			pool := &tfv1.GPUPool{
				Spec: tfv1.GPUPoolSpec{
					ComponentConfig: &tfv1.ComponentConfig{
						NodeDiscovery: &tfv1.NodeDiscoveryConfig{
							Image: tt.nodeDiscoveryImage,
						},
					},
				},
			}

			utils.AddTFNodeDiscoveryConfAfterTemplate(ctx, tmpl, pool, tt.gpuNodeName, tt.compatibleWithNvidiaContainerToolkit)

			require.Len(t, tmpl.Spec.InitContainers, tt.expectInitContainers, "unexpected number of init containers")

			if tt.validateToolkitInit {
				// Verify toolkit-validation init container
				toolkitInit := tmpl.Spec.InitContainers[0]
				require.Equal(t, constants.TFInitContainerNameToolkitValidation, toolkitInit.Name)
				require.Equal(t, tt.nodeDiscoveryImage, toolkitInit.Image)
				require.Equal(t, []string{"sh", "-c"}, toolkitInit.Command)
				require.Contains(t, toolkitInit.Args[0], "/run/nvidia/validations/toolkit-ready")
				require.NotNil(t, toolkitInit.SecurityContext)
				require.True(t, *toolkitInit.SecurityContext.Privileged)

				// Verify volume mount
				require.Len(t, toolkitInit.VolumeMounts, 1)
				require.Equal(t, "run-nvidia-validations", toolkitInit.VolumeMounts[0].Name)
				require.Equal(t, "/run/nvidia/validations", toolkitInit.VolumeMounts[0].MountPath)
				require.NotNil(t, toolkitInit.VolumeMounts[0].MountPropagation)
				require.Equal(t, corev1.MountPropagationHostToContainer, *toolkitInit.VolumeMounts[0].MountPropagation)

				// Verify volume exists
				var nvidiaVolume *corev1.Volume
				for i := range tmpl.Spec.Volumes {
					if tmpl.Spec.Volumes[i].Name == "run-nvidia-validations" {
						nvidiaVolume = &tmpl.Spec.Volumes[i]
						break
					}
				}
				require.NotNil(t, nvidiaVolume, "run-nvidia-validations volume not found")
				require.NotNil(t, nvidiaVolume.HostPath)
				require.Equal(t, "/run/nvidia/validations", nvidiaVolume.HostPath.Path)
				require.NotNil(t, nvidiaVolume.HostPath.Type)
				require.Equal(t, corev1.HostPathDirectoryOrCreate, *nvidiaVolume.HostPath.Type)
			}

			// Verify basic node discovery settings
			require.Equal(t, corev1.RestartPolicyOnFailure, tmpl.Spec.RestartPolicy)
			require.NotNil(t, tmpl.Spec.TerminationGracePeriodSeconds)
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

func TestNodeDiscoveryInitContainerImageFallback(t *testing.T) {
	t.Run("use node discovery image when specified", func(t *testing.T) {
		ctx := context.Background()
		tmpl := &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{},
		}
		pool := &tfv1.GPUPool{
			Spec: tfv1.GPUPoolSpec{
				ComponentConfig: &tfv1.ComponentConfig{
					NodeDiscovery: &tfv1.NodeDiscoveryConfig{
						Image: "node-discovery:v1.0",
					},
				},
			},
		}

		utils.AddTFNodeDiscoveryConfAfterTemplate(ctx, tmpl, pool, "test-node", true)

		require.Len(t, tmpl.Spec.InitContainers, 1)
		require.Equal(t, constants.TFInitContainerNameToolkitValidation, tmpl.Spec.InitContainers[0].Name)
		require.Equal(t, "node-discovery:v1.0", tmpl.Spec.InitContainers[0].Image)
	})

	t.Run("fallback to main container image when node discovery image is empty", func(t *testing.T) {
		ctx := context.Background()
		tmpl := &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name:  "node-discovery",
						Image: "fallback-discovery:v2.0",
					},
				},
			},
		}
		pool := &tfv1.GPUPool{
			Spec: tfv1.GPUPoolSpec{
				ComponentConfig: &tfv1.ComponentConfig{
					NodeDiscovery: &tfv1.NodeDiscoveryConfig{
						Image: "",
					},
				},
			},
		}

		utils.AddTFNodeDiscoveryConfAfterTemplate(ctx, tmpl, pool, "test-node", true)

		require.Len(t, tmpl.Spec.InitContainers, 1)
		require.Equal(t, constants.TFInitContainerNameToolkitValidation, tmpl.Spec.InitContainers[0].Name)
		require.Equal(t, "fallback-discovery:v2.0", tmpl.Spec.InitContainers[0].Image)
	})
}

func TestComposeNvidiaDriverProbeJob(t *testing.T) {
	node := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-1",
		},
	}
	pool := &tfv1.GPUPool{
		Spec: tfv1.GPUPoolSpec{
			ComponentConfig: &tfv1.ComponentConfig{
				Hypervisor: &tfv1.HypervisorConfig{
					Image: "hypervisor:latest",
				},
			},
		},
	}

	job, err := utils.ComposeNvidiaDriverProbeJob(node, pool)
	require.NoError(t, err)
	require.NotNil(t, job)
	require.Equal(t, corev1.RestartPolicyOnFailure, job.Spec.Template.Spec.RestartPolicy)
	require.Equal(t, "node-1", job.Spec.Template.Spec.NodeName)
	require.Len(t, job.Spec.Template.Spec.Containers, 1)
	container := job.Spec.Template.Spec.Containers[0]
	require.Equal(t, "hypervisor:latest", container.Image)
	require.NotEmpty(t, container.Args)
	require.Contains(t, container.Args[0], "toolkit-ready")
}
