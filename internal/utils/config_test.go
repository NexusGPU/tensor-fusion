package utils_test

import (
	"os"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

var _ = Describe("Config Utils", func() {
	Describe("ExtractPodWorkerInfo", func() {
		Describe("DeviceUUIDs", func() {
			DescribeTable("extracts device UUIDs correctly",
				func(annotations map[string]string, expected []string) {
					pod := &corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:        "test-pod",
							Annotations: annotations,
						},
					}

					info := utils.ExtractPodWorkerInfo(pod)

					Expect(info.DeviceUUIDs).To(Equal(expected))
				},
				Entry("no GPU IDs annotation", map[string]string{}, []string(nil)),
				Entry("single GPU ID", map[string]string{
					constants.GPUDeviceIDsAnnotation: "gpu-0001",
				}, []string{"gpu-0001"}),
				Entry("multiple GPU IDs", map[string]string{
					constants.GPUDeviceIDsAnnotation: "gpu-0001,gpu-0002,gpu-0003",
				}, []string{"gpu-0001", "gpu-0002", "gpu-0003"}),
				Entry("GPU IDs with spaces", map[string]string{
					constants.GPUDeviceIDsAnnotation: "gpu-0001 , gpu-0002 , gpu-0003",
				}, []string{"gpu-0001", "gpu-0002", "gpu-0003"}),
				Entry("GPU IDs with empty segments", map[string]string{
					constants.GPUDeviceIDsAnnotation: "gpu-0001,,gpu-0002,",
				}, []string{"gpu-0001", "gpu-0002"}),
				Entry("empty GPU IDs annotation", map[string]string{
					constants.GPUDeviceIDsAnnotation: "",
				}, []string{}),
			)
		})

		Describe("IsolationMode", func() {
			DescribeTable("extracts isolation mode correctly",
				func(annotations map[string]string, expected string) {
					pod := &corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:        "test-pod",
							Annotations: annotations,
						},
					}

					info := utils.ExtractPodWorkerInfo(pod)

					Expect(info.IsolationMode).To(Equal(expected))
				},
				Entry("no isolation mode - default to soft", map[string]string{}, string(tfv1.IsolationModeSoft)),
				Entry("explicit soft isolation", map[string]string{
					constants.IsolationModeAnnotation: string(tfv1.IsolationModeSoft),
				}, string(tfv1.IsolationModeSoft)),
				Entry("hard isolation", map[string]string{
					constants.IsolationModeAnnotation: string(tfv1.IsolationModeHard),
				}, string(tfv1.IsolationModeHard)),
				Entry("shared isolation", map[string]string{
					constants.IsolationModeAnnotation: string(tfv1.IsolationModeShared),
				}, string(tfv1.IsolationModeShared)),
			)
		})

		Describe("MemoryLimit", func() {
			DescribeTable("extracts memory limit correctly",
				func(annotations map[string]string, expected uint64) {
					pod := &corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:        "test-pod",
							Annotations: annotations,
						},
					}

					info := utils.ExtractPodWorkerInfo(pod)

					Expect(info.MemoryLimitBytes).To(Equal(expected))
				},
				Entry("no VRAM limit", map[string]string{}, uint64(0)),
				Entry("VRAM limit in Gi", map[string]string{
					constants.VRAMLimitAnnotation: "8Gi",
				}, uint64(8*1024*1024*1024)),
				Entry("VRAM limit in Mi", map[string]string{
					constants.VRAMLimitAnnotation: "512Mi",
				}, uint64(512*1024*1024)),
				Entry("invalid VRAM limit", map[string]string{
					constants.VRAMLimitAnnotation: "invalid",
				}, uint64(0)),
			)
		})

		Describe("ComputeLimit", func() {
			DescribeTable("extracts compute limit correctly",
				func(annotations map[string]string, expected uint32) {
					pod := &corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:        "test-pod",
							Annotations: annotations,
						},
					}

					info := utils.ExtractPodWorkerInfo(pod)

					Expect(info.ComputeLimitUnits).To(Equal(expected))
				},
				Entry("no compute limit", map[string]string{}, uint32(0)),
				Entry("compute limit 50 percent", map[string]string{
					constants.ComputeLimitAnnotation: "50",
				}, uint32(50)),
				Entry("compute limit 100 percent", map[string]string{
					constants.ComputeLimitAnnotation: "100",
				}, uint32(100)),
				Entry("invalid compute limit", map[string]string{
					constants.ComputeLimitAnnotation: "invalid",
				}, uint32(0)),
			)
		})

		Describe("TemplateID", func() {
			DescribeTable("extracts template ID correctly",
				func(annotations map[string]string, expected string) {
					pod := &corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:        "test-pod",
							Annotations: annotations,
						},
					}

					info := utils.ExtractPodWorkerInfo(pod)

					Expect(info.TemplateID).To(Equal(expected))
				},
				Entry("no template ID", map[string]string{}, ""),
				Entry("partition template ID takes precedence", map[string]string{
					constants.PartitionTemplateIDAnnotation: "partition-uuid-123",
					constants.WorkloadProfileAnnotation:     "workload-profile-456",
				}, "partition-uuid-123"),
				Entry("falls back to workload profile", map[string]string{
					constants.WorkloadProfileAnnotation: "workload-profile-456",
				}, "workload-profile-456"),
				Entry("only partition template ID", map[string]string{
					constants.PartitionTemplateIDAnnotation: "partition-uuid-789",
				}, "partition-uuid-789"),
			)
		})

		It("should extract complete worker info correctly", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "complete-worker-pod",
					Namespace: "default",
					Annotations: map[string]string{
						constants.GPUDeviceIDsAnnotation:        "gpu-a100-001, gpu-a100-002",
						constants.IsolationModeAnnotation:       string(tfv1.IsolationModeHard),
						constants.VRAMLimitAnnotation:           "16Gi",
						constants.ComputeLimitAnnotation:        "75",
						constants.PartitionTemplateIDAnnotation: "mig-1g-5gb",
					},
				},
			}

			info := utils.ExtractPodWorkerInfo(pod)

			Expect(info.DeviceUUIDs).To(Equal([]string{"gpu-a100-001", "gpu-a100-002"}))
			Expect(info.IsolationMode).To(Equal(string(tfv1.IsolationModeHard)))
			Expect(info.MemoryLimitBytes).To(Equal(uint64(16 * 1024 * 1024 * 1024)))
			Expect(info.ComputeLimitUnits).To(Equal(uint32(75)))
			Expect(info.TemplateID).To(Equal("mig-1g-5gb"))
		})
	})

	Describe("GetGPUResource", func() {
		Describe("Request", func() {
			DescribeTable("gets GPU resource request correctly",
				func(annotations map[string]string, expectTflops, expectVram, expectPercent string, expectError bool) {
					pod := &corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:        "test-pod",
							Namespace:   "default",
							Annotations: annotations,
						},
					}

					res, err := utils.GetGPUResource(pod, true)

					if expectError {
						Expect(err).To(HaveOccurred())
						Expect(err.Error()).To(ContainSubstring("mutually exclusive"))
						return
					}

					Expect(err).NotTo(HaveOccurred())
					if expectTflops != "" {
						Expect(res.Tflops.String()).To(Equal(expectTflops))
					}
					if expectVram != "" {
						Expect(res.Vram.String()).To(Equal(expectVram))
					}
					if expectPercent != "" {
						Expect(res.ComputePercent.String()).To(Equal(expectPercent))
					}
				},
				Entry("tflops and vram request", map[string]string{
					constants.TFLOPSRequestAnnotation: "100",
					constants.VRAMRequestAnnotation:   "8Gi",
				}, "100", "8Gi", "", false),
				Entry("compute percent and vram request", map[string]string{
					constants.ComputeRequestAnnotation: "50",
					constants.VRAMRequestAnnotation:    "4Gi",
				}, "", "4Gi", "50", false),
				Entry("both tflops and compute percent - mutual exclusion error", map[string]string{
					constants.TFLOPSRequestAnnotation:  "100",
					constants.ComputeRequestAnnotation: "50",
					constants.VRAMRequestAnnotation:    "8Gi",
				}, "", "", "", true),
				Entry("no annotations", map[string]string{}, "", "", "", false),
				Entry("only vram", map[string]string{
					constants.VRAMRequestAnnotation: "8Gi",
				}, "", "8Gi", "", false),
			)
		})

		Describe("Limit", func() {
			DescribeTable("gets GPU resource limit correctly",
				func(annotations map[string]string, expectTflops, expectVram, expectPercent string, expectError bool) {
					pod := &corev1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name:        "test-pod",
							Namespace:   "default",
							Annotations: annotations,
						},
					}

					res, err := utils.GetGPUResource(pod, false)

					if expectError {
						Expect(err).To(HaveOccurred())
						Expect(err.Error()).To(ContainSubstring("mutually exclusive"))
						return
					}

					Expect(err).NotTo(HaveOccurred())
					if expectTflops != "" {
						Expect(res.Tflops.String()).To(Equal(expectTflops))
					}
					if expectVram != "" {
						Expect(res.Vram.String()).To(Equal(expectVram))
					}
					if expectPercent != "" {
						Expect(res.ComputePercent.String()).To(Equal(expectPercent))
					}
				},
				Entry("tflops and vram limit", map[string]string{
					constants.TFLOPSLimitAnnotation: "200",
					constants.VRAMLimitAnnotation:   "16Gi",
				}, "200", "16Gi", "", false),
				Entry("compute percent and vram limit", map[string]string{
					constants.ComputeLimitAnnotation: "100",
					constants.VRAMLimitAnnotation:    "8Gi",
				}, "", "8Gi", "100", false),
				Entry("both tflops and compute percent limit - mutual exclusion error", map[string]string{
					constants.TFLOPSLimitAnnotation:  "200",
					constants.ComputeLimitAnnotation: "100",
					constants.VRAMLimitAnnotation:    "16Gi",
				}, "", "", "", true),
			)
		})
	})

	Describe("GetEnvOrDefault", func() {
		const testEnvKey = "TEST_ENV_KEY_FOR_UTILS_TEST"

		AfterEach(func() {
			_ = os.Unsetenv(testEnvKey)
		})

		DescribeTable("returns correct value",
			func(envValue string, setEnv bool, defaultValue, expected string) {
				_ = os.Unsetenv(testEnvKey)

				if setEnv {
					Expect(os.Setenv(testEnvKey, envValue)).To(Succeed())
				}

				result := utils.GetEnvOrDefault(testEnvKey, defaultValue)

				Expect(result).To(Equal(expected))
			},
			Entry("env not set, returns default", "", false, "default-value", "default-value"),
			Entry("env set to value, returns env value", "actual-value", true, "default-value", "actual-value"),
			Entry("env set to empty, returns default", "", true, "default-value", "default-value"),
		)
	})

	Describe("NewShortID", func() {
		DescribeTable("generates ID with correct length",
			func(length int) {
				id := utils.NewShortID(length)

				Expect(id).To(HaveLen(length))
				// Verify it only contains allowed characters from ShortUUIDAlphabet
				for _, char := range id {
					Expect(constants.ShortUUIDAlphabet).To(ContainSubstring(string(char)),
						"ID should only contain characters from ShortUUIDAlphabet")
				}
			},
			Entry("short ID length 8", 8),
			Entry("short ID length 12", 12),
			Entry("short ID length 4", 4),
		)

		It("should generate unique IDs", func() {
			ids := make(map[string]struct{})
			for i := 0; i < 100; i++ {
				id := utils.NewShortID(12)
				_, exists := ids[id]
				Expect(exists).To(BeFalse(), "Generated IDs should be unique")
				ids[id] = struct{}{}
			}
		})

		It("should return full ID when length exceeds max", func() {
			id := utils.NewShortID(100)
			Expect(id).NotTo(BeEmpty())
		})
	})

	Describe("IsDebugMode", func() {
		const debugEnvKey = "DEBUG"

		AfterEach(func() {
			_ = os.Unsetenv(debugEnvKey)
		})

		It("should return false when debug mode not set", func() {
			_ = os.Unsetenv(debugEnvKey)
			Expect(utils.IsDebugMode()).To(BeFalse())
		})

		It("should return true when debug mode set to true", func() {
			Expect(os.Setenv(debugEnvKey, "true")).To(Succeed())
			Expect(utils.IsDebugMode()).To(BeTrue())
		})

		It("should return false when debug mode set to false", func() {
			Expect(os.Setenv(debugEnvKey, "false")).To(Succeed())
			Expect(utils.IsDebugMode()).To(BeFalse())
		})

		It("should return false when debug mode set to other value", func() {
			Expect(os.Setenv(debugEnvKey, "1")).To(Succeed())
			Expect(utils.IsDebugMode()).To(BeFalse())
		})
	})

	Describe("SetProgressiveMigration", func() {
		var original bool

		BeforeEach(func() {
			original = utils.IsProgressiveMigration()
		})

		AfterEach(func() {
			utils.SetProgressiveMigration(original)
		})

		It("should set to true", func() {
			utils.SetProgressiveMigration(true)
			Expect(utils.IsProgressiveMigration()).To(BeTrue())
		})

		It("should set to false", func() {
			utils.SetProgressiveMigration(false)
			Expect(utils.IsProgressiveMigration()).To(BeFalse())
		})
	})

	Describe("GetSelfServiceAccountNameShort", func() {
		It("should return at least the last part of the full name", func() {
			fullName := utils.GetSelfServiceAccountNameFull()
			shortName := utils.GetSelfServiceAccountNameShort()

			if fullName != "" {
				Expect(shortName).NotTo(BeEmpty())
			}
		})
	})
})
