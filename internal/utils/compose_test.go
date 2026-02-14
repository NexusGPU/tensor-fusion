package utils_test

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

var _ = Describe("Compose Utils", func() {
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
	})

	Describe("SetWorkerContainerSpec", func() {
		DescribeTable("configures worker container correctly",
			func(workerImage, disabledFeatures string, sharedMemMode bool, expectCommand []string) {
				container := &corev1.Container{}
				workloadProfile := &tfv1.WorkloadProfileSpec{}
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
			Entry("basic worker config", "worker:latest", "", false, []string{
				"./tensor-fusion-worker",
				"-p",
				"8000",
			}),
			Entry("worker with shared memory mode", "worker:latest", "", true, []string{
				"/bin/bash",
				"-c",
				"touch /dev/shm/tf_shm && chmod 666 /dev/shm/tf_shm && exec ./tensor-fusion-worker -n shmem -m tf_shm -M 256",
			}),
			Entry("worker with disabled start-worker feature", "worker:latest", "start-worker", false, []string{
				"sleep",
				"infinity",
			}),
		)
	})
})
