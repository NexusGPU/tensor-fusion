package utils_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

func TestAddTFHypervisorConfAfterTemplate(t *testing.T) {
	tests := []struct {
		name                                 string
		compatibleWithNvidiaContainerToolkit bool
		enableVector                         bool
		hypervisorImage                      string
		expectInitContainers                 int
		expectVolumes                        int
		validateToolkitInit                  bool
	}{
		{
			name:                                 "with nvidia toolkit validation",
			compatibleWithNvidiaContainerToolkit: true,
			enableVector:                         false,
			hypervisorImage:                      "test-image:latest",
			expectInitContainers:                 2, // init-shm + toolkit-validation
			expectVolumes:                        8, // 7 base volumes + run-nvidia-validations
			validateToolkitInit:                  true,
		},
		{
			name:                                 "without nvidia toolkit validation",
			compatibleWithNvidiaContainerToolkit: false,
			enableVector:                         false,
			hypervisorImage:                      "test-image:latest",
			expectInitContainers:                 1, // init-shm only
			expectVolumes:                        7, // 7 base volumes
			validateToolkitInit:                  false,
		},
		{
			name:                                 "with nvidia toolkit and vector enabled",
			compatibleWithNvidiaContainerToolkit: true,
			enableVector:                         true,
			hypervisorImage:                      "test-image:latest",
			expectInitContainers:                 2, // init-shm + toolkit-validation
			expectVolumes:                        8, // 7 base volumes + run-nvidia-validations
			validateToolkitInit:                  true,
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

			utils.AddTFHypervisorConfAfterTemplate(ctx, spec, pool, tt.compatibleWithNvidiaContainerToolkit)

			require.Len(t, spec.InitContainers, tt.expectInitContainers, "unexpected number of init containers")
			require.Len(t, spec.Volumes, tt.expectVolumes, "unexpected number of volumes")

			if tt.validateToolkitInit {
				// Find the toolkit-validation init container
				var toolkitInit *corev1.Container
				for i := range spec.InitContainers {
					if spec.InitContainers[i].Name == constants.TFInitContainerNameToolkitValidation {
						toolkitInit = &spec.InitContainers[i]
						break
					}
				}
				require.NotNil(t, toolkitInit, "toolkit-validation init container not found")
				require.Equal(t, tt.hypervisorImage, toolkitInit.Image)
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
				for i := range spec.Volumes {
					if spec.Volumes[i].Name == "run-nvidia-validations" {
						nvidiaVolume = &spec.Volumes[i]
						break
					}
				}
				require.NotNil(t, nvidiaVolume, "run-nvidia-validations volume not found")
				require.NotNil(t, nvidiaVolume.HostPath)
				require.Equal(t, "/run/nvidia/validations", nvidiaVolume.HostPath.Path)
				require.NotNil(t, nvidiaVolume.HostPath.Type)
				require.Equal(t, corev1.HostPathDirectoryOrCreate, *nvidiaVolume.HostPath.Type)
			}

			// Verify basic hypervisor settings
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
	}{
		{
			name:             "basic worker config",
			workerImage:      "worker:latest",
			disabledFeatures: "",
			sharedMemMode:    false,
		},
		{
			name:             "worker with shared memory",
			workerImage:      "worker:latest",
			disabledFeatures: "",
			sharedMemMode:    true,
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
		})
	}
}

func TestHypervisorInitContainerImageFallback(t *testing.T) {
	t.Run("use hypervisor image when specified", func(t *testing.T) {
		ctx := context.Background()
		spec := &corev1.PodSpec{}
		pool := &tfv1.GPUPool{
			Spec: tfv1.GPUPoolSpec{
				ComponentConfig: &tfv1.ComponentConfig{
					Hypervisor: &tfv1.HypervisorConfig{
						Image: "hypervisor:v1.0",
					},
				},
			},
		}

		utils.AddTFHypervisorConfAfterTemplate(ctx, spec, pool, true)

		var toolkitInit *corev1.Container
		for i := range spec.InitContainers {
			if spec.InitContainers[i].Name == constants.TFInitContainerNameToolkitValidation {
				toolkitInit = &spec.InitContainers[i]
				break
			}
		}
		require.NotNil(t, toolkitInit)
		require.Equal(t, "hypervisor:v1.0", toolkitInit.Image)
	})

	t.Run("fallback to main container image when hypervisor image is empty", func(t *testing.T) {
		ctx := context.Background()
		spec := &corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "hypervisor",
					Image: "fallback-image:v2.0",
				},
			},
		}
		pool := &tfv1.GPUPool{
			Spec: tfv1.GPUPoolSpec{
				ComponentConfig: &tfv1.ComponentConfig{
					Hypervisor: &tfv1.HypervisorConfig{
						Image: "",
					},
				},
			},
		}

		utils.AddTFHypervisorConfAfterTemplate(ctx, spec, pool, true)

		var toolkitInit *corev1.Container
		for i := range spec.InitContainers {
			if spec.InitContainers[i].Name == constants.TFInitContainerNameToolkitValidation {
				toolkitInit = &spec.InitContainers[i]
				break
			}
		}
		require.NotNil(t, toolkitInit)
		require.Equal(t, "fallback-image:v2.0", toolkitInit.Image)
	})
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
