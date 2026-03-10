package utils_test

import (
	"context"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

func hasNvidiaVisibleEnv(envs []corev1.EnvVar) bool {
	for _, env := range envs {
		if env.Name == constants.NvidiaVisibleAllDeviceEnv {
			return true
		}
	}
	return false
}

func envValue(envs []corev1.EnvVar, name string) (string, bool) {
	for _, env := range envs {
		if env.Name == name {
			return env.Value, true
		}
	}
	return "", false
}

func hasVolumeMount(mounts []corev1.VolumeMount, name, path string) bool {
	for _, mount := range mounts {
		if mount.Name == name && mount.MountPath == path {
			return true
		}
	}
	return false
}

func hasHostPathVolume(volumes []corev1.Volume, name, path string) bool {
	for _, volume := range volumes {
		if volume.Name == name && volume.HostPath != nil && volume.HostPath.Path == path {
			return true
		}
	}
	return false
}

func hasIndexResourceLimit(resources corev1.ResourceRequirements) bool {
	for key := range resources.Limits {
		if strings.HasPrefix(string(key), constants.PodIndexAnnotation+constants.PodIndexDelimiter) {
			return true
		}
	}
	return false
}

func ptr(s string) *string {
	return &s
}

var _ = Describe("Compose Utils", func() {
	AfterEach(func() {
		provider.SetGlobalManagerForTesting(nil)
	})

	Describe("AddTFHypervisorConfAfterTemplate", func() {
		DescribeTable("configures hypervisor correctly",
			func(enableVector bool, hypervisorImage string, expectInitCount, expectVolumeCount int) {
				ctx := context.Background()
				spec := &corev1.PodSpec{}
				pool := &tfv1.GPUPool{
					Spec: tfv1.GPUPoolSpec{
						ComponentConfig: &tfv1.ComponentConfig{
							Hypervisor: &tfv1.HypervisorConfig{
								Image:        hypervisorImage,
								EnableVector: enableVector,
							},
						},
					},
				}

				utils.AddTFHypervisorConfAfterTemplate(ctx, spec, pool, "NVIDIA", false)

				Expect(spec.InitContainers).To(HaveLen(expectInitCount), "unexpected number of init containers")
				Expect(spec.Volumes).To(HaveLen(expectVolumeCount), "unexpected number of volumes")
				Expect(spec.HostPID).To(BeTrue())
				Expect(spec.TerminationGracePeriodSeconds).NotTo(BeNil())
			},
			Entry("without vector", false, "test-image:latest", 2, 7),
			Entry("with vector", true, "test-image:latest", 2, 7),
		)

		It("should inject NVIDIA_VISIBLE_DEVICES only for NVIDIA vendor", func() {
			ctx := context.Background()
			pool := &tfv1.GPUPool{
				Spec: tfv1.GPUPoolSpec{
					ComponentConfig: &tfv1.ComponentConfig{
						Hypervisor: &tfv1.HypervisorConfig{
							Image: "test-image:latest",
						},
					},
				},
			}

			nvidiaSpec := &corev1.PodSpec{}
			utils.AddTFHypervisorConfAfterTemplate(ctx, nvidiaSpec, pool, "NVIDIA", false)
			Expect(hasNvidiaVisibleEnv(nvidiaSpec.Containers[0].Env)).To(BeTrue())

			ascendSpec := &corev1.PodSpec{}
			utils.AddTFHypervisorConfAfterTemplate(ctx, ascendSpec, pool, "Ascend", false)
			Expect(hasNvidiaVisibleEnv(ascendSpec.Containers[0].Env)).To(BeFalse())
		})
	})

	Describe("AddTFDefaultClientConfBeforePatch", func() {
		newPool := func() *tfv1.GPUPool {
			return &tfv1.GPUPool{
				Spec: tfv1.GPUPoolSpec{
					ComponentConfig: &tfv1.ComponentConfig{
						Client: &tfv1.ClientConfig{Image: "client:latest"},
					},
				},
			}
		}

		It("should not inject initContainer or sidecar container in local mode", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}},
				Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "main"}}},
			}

			utils.AddTFDefaultClientConfBeforePatch(context.Background(), pod, newPool(), utils.TensorFusionInfo{
				Profile: &tfv1.WorkloadProfileSpec{
					IsLocalGPU:    true,
					SidecarWorker: true,
				},
			}, []int{0})

			Expect(pod.Spec.InitContainers).To(BeEmpty())
			Expect(pod.Spec.Containers).To(HaveLen(1))

			volumeNames := make([]string, 0, len(pod.Spec.Volumes))
			for _, volume := range pod.Spec.Volumes {
				volumeNames = append(volumeNames, volume.Name)
			}
			Expect(volumeNames).To(ContainElement(constants.DataVolumeName))
			Expect(volumeNames).NotTo(ContainElement(constants.TransportShmVolumeName))
		})

		It("should not inject NVIDIA_VISIBLE_DEVICES for non-NVIDIA local workloads", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}},
				Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "main"}}},
			}

			utils.AddTFDefaultClientConfBeforePatch(context.Background(), pod, newPool(), utils.TensorFusionInfo{
				Profile: &tfv1.WorkloadProfileSpec{
					IsLocalGPU: true,
					GPUVendor:  "Ascend",
				},
			}, []int{0})

			Expect(hasNvidiaVisibleEnv(pod.Spec.Containers[0].Env)).To(BeFalse())
		})

		It("should inject client initContainer in remote mode", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}},
				Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "main"}}},
			}

			utils.AddTFDefaultClientConfBeforePatch(context.Background(), pod, newPool(), utils.TensorFusionInfo{
				Profile: &tfv1.WorkloadProfileSpec{IsLocalGPU: false},
			}, []int{0})

			Expect(pod.Spec.InitContainers).To(HaveLen(1))
			Expect(pod.Spec.InitContainers[0].Name).To(Equal(constants.TFContainerNameClient))
			Expect(pod.Spec.Containers).To(HaveLen(1))
		})

		It("should auto set runtimeClassName for Ascend remote mode", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}},
				Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "main"}}},
			}

			utils.AddTFDefaultClientConfBeforePatch(context.Background(), pod, newPool(), utils.TensorFusionInfo{
				Profile: &tfv1.WorkloadProfileSpec{
					IsLocalGPU: false,
					GPUVendor:  constants.AcceleratorVendorHuaweiAscendNPU,
					Isolation:  tfv1.IsolationModeShared,
				},
			}, []int{0})

			Expect(pod.Spec.RuntimeClassName).NotTo(BeNil())
			Expect(*pod.Spec.RuntimeClassName).To(Equal(constants.AscendRuntimeClassName))
		})

		It("should auto set runtimeClassName for Ascend partitioned local mode", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}},
				Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "main"}}},
			}

			utils.AddTFDefaultClientConfBeforePatch(context.Background(), pod, newPool(), utils.TensorFusionInfo{
				Profile: &tfv1.WorkloadProfileSpec{
					IsLocalGPU: true,
					GPUVendor:  constants.AcceleratorVendorHuaweiAscendNPU,
					Isolation:  tfv1.IsolationModePartitioned,
				},
			}, []int{0})

			Expect(pod.Spec.RuntimeClassName).NotTo(BeNil())
			Expect(*pod.Spec.RuntimeClassName).To(Equal(constants.AscendRuntimeClassName))
		})

		It("should keep existing runtimeClassName", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{}},
				Spec: corev1.PodSpec{
					RuntimeClassName: ptr("custom-runtime"),
					Containers:       []corev1.Container{{Name: "main"}},
				},
			}

			utils.AddTFDefaultClientConfBeforePatch(context.Background(), pod, newPool(), utils.TensorFusionInfo{
				Profile: &tfv1.WorkloadProfileSpec{
					IsLocalGPU: false,
					GPUVendor:  constants.AcceleratorVendorHuaweiAscendNPU,
					Isolation:  tfv1.IsolationModeShared,
				},
			}, []int{0})

			Expect(pod.Spec.RuntimeClassName).NotTo(BeNil())
			Expect(*pod.Spec.RuntimeClassName).To(Equal("custom-runtime"))
		})
	})

	Describe("SetWorkerContainerSpec", func() {
		DescribeTable("configures worker container correctly",
			func(vendor, workerImage, disabledFeatures string, sharedMemMode bool, expectCommand []string, expectNvidiaVisibleEnv, expectLdPreload bool) {
				container := &corev1.Container{}
				workloadProfile := &tfv1.WorkloadProfileSpec{
					GPUVendor: vendor,
				}
				workerConfig := &tfv1.WorkerConfig{
					Image: workerImage,
				}
				hypervisorConfig := &tfv1.HypervisorConfig{}

				utils.SetWorkerContainerSpec(container, workloadProfile, workerConfig, hypervisorConfig, disabledFeatures, sharedMemMode)

				Expect(container.Name).NotTo(BeEmpty())
				if workerImage != "" {
					Expect(container.Image).To(Equal(workerImage))
				}

				// Verify command is set correctly
				Expect(container.Command).NotTo(BeEmpty(), "container command should not be empty")
				Expect(container.Command).To(Equal(expectCommand), "container command should match expected value")
				Expect(hasNvidiaVisibleEnv(container.Env)).To(Equal(expectNvidiaVisibleEnv))
				_, hasLdPreload := envValue(container.Env, constants.LdPreloadEnv)
				Expect(hasLdPreload).To(Equal(expectLdPreload))

				// Verify shared memory mode specific setup
				if sharedMemMode && disabledFeatures == "" {
					Expect(container.Command).To(HaveLen(3), "shared memory mode should use bash -c with script")
					Expect(container.Command[0]).To(Equal("/bin/bash"), "should use bash")
					Expect(container.Command[1]).To(Equal("-c"), "should use -c flag")
					Expect(container.Command[2]).To(ContainSubstring("touch /dev/shm/tf_shm"), "should create shared memory file")
					Expect(container.Command[2]).To(ContainSubstring("chmod 666 /dev/shm/tf_shm"), "should set file permissions")
					Expect(container.Command[2]).To(ContainSubstring("exec ./tensor-fusion-worker"), "should exec worker")
					Expect(container.Command[2]).To(ContainSubstring("-n shmem"), "should use shmem mode")
					Expect(container.Command[2]).To(ContainSubstring("-m tf_shm"), "should specify shared memory name")
					Expect(container.Command[2]).To(ContainSubstring("-M 256"), "should specify shared memory size")
				}
			},
			Entry("basic worker config", "NVIDIA", "worker:latest", "", false, []string{
				"./tensor-fusion-worker",
				"-p",
				"8000",
			}, true, true),
			Entry("worker with shared memory mode", "NVIDIA", "worker:latest", "", true, []string{
				"/bin/bash",
				"-c",
				"touch /dev/shm/tf_shm && chmod 666 /dev/shm/tf_shm && exec ./tensor-fusion-worker -n shmem -m tf_shm -M 256",
			}, true, true),
			Entry("worker with disabled start-worker feature", "NVIDIA", "worker:latest", "start-worker", false, []string{
				"sleep",
				"infinity",
			}, true, true),
			Entry("worker without nvidia visible env for Ascend", "Ascend", "worker:latest", "", false, []string{
				"./tensor-fusion-worker",
				"-p",
				"8000",
			}, false, false),
		)

		It("should apply provider runtime mounts and ld library path for Ascend remote worker", func() {
			providerMgr := provider.NewManager(nil)
			providerMgr.UpdateProvider(&tfv1.ProviderConfig{
				ObjectMeta: metav1.ObjectMeta{Name: "ascend-provider"},
				Spec: tfv1.ProviderConfigSpec{
					Vendor: constants.AcceleratorVendorHuaweiAscendNPU,
					Hypervisor: &tfv1.ProviderHypervisorConfig{
						LDLibraryPath: constants.AscendLDLibraryPath,
						HostPathMounts: []tfv1.ProviderHypervisorHostPathMount{
							{
								Name:      constants.AscendDriverVolumeName,
								HostPath:  constants.AscendDriverHostPath,
								MountPath: constants.AscendDriverHostPath,
								ReadOnly:  true,
							},
							{
								Name:      constants.AscendDCMIVolumeName,
								HostPath:  constants.AscendDCMIHostPath,
								MountPath: constants.AscendDCMIHostPath,
								ReadOnly:  true,
							},
						},
					},
					Images: tfv1.ProviderImages{
						Middleware:   "ascend-hypervisor:latest",
						RemoteWorker: "ascend-worker:latest",
					},
				},
			})
			provider.SetGlobalManagerForTesting(providerMgr)

			spec := corev1.PodSpec{Containers: []corev1.Container{{}}}
			workload := &tfv1.TensorFusionWorkload{
				Spec: tfv1.WorkloadProfileSpec{
					GPUVendor: constants.AcceleratorVendorHuaweiAscendNPU,
				},
			}

			utils.AddWorkerConfAfterTemplate(
				context.Background(),
				&spec,
				&workload.Spec,
				&tfv1.WorkerConfig{Image: "worker:latest"},
				&tfv1.HypervisorConfig{},
				workload,
			)

			ldLibraryPath, ok := envValue(spec.Containers[0].Env, "LD_LIBRARY_PATH")
			Expect(ok).To(BeTrue())
			Expect(ldLibraryPath).To(Equal(constants.AscendLDLibraryPath))
			Expect(spec.RuntimeClassName).NotTo(BeNil())
			Expect(*spec.RuntimeClassName).To(Equal(constants.AscendRuntimeClassName))
			_, hasLdPreload := envValue(spec.Containers[0].Env, constants.LdPreloadEnv)
			Expect(hasLdPreload).To(BeFalse())
			Expect(hasIndexResourceLimit(spec.Containers[0].Resources)).To(BeTrue())
			Expect(hasVolumeMount(spec.Containers[0].VolumeMounts, constants.AscendDriverVolumeName, constants.AscendDriverHostPath)).To(BeTrue())
			Expect(hasVolumeMount(spec.Containers[0].VolumeMounts, constants.AscendDCMIVolumeName, constants.AscendDCMIHostPath)).To(BeTrue())
			Expect(hasHostPathVolume(spec.Volumes, constants.AscendDriverVolumeName, constants.AscendDriverHostPath)).To(BeTrue())
			Expect(hasHostPathVolume(spec.Volumes, constants.AscendDCMIVolumeName, constants.AscendDCMIHostPath)).To(BeTrue())
		})
	})
})
