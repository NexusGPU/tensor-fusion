package metrics

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var _ = Describe("SetWorkerMetricsByWorkload", func() {
	var (
		pod         *corev1.Pod
		capacityMap map[string]tfv1.Resource
	)

	BeforeEach(func() {
		capacityMap = map[string]tfv1.Resource{
			"A100": {
				Tflops: resource.MustParse("312"),
				Vram:   resource.MustParse("80Gi"),
			},
			"H100": {
				Tflops: resource.MustParse("756"),
				Vram:   resource.MustParse("80Gi"),
			},
		}

		workerMetricsLock.Lock()
		workerMetricsMap = make(map[string]*WorkerResourceMetrics, 8)
		workerMetricsLock.Unlock()

		gpuAllocationMetricsLock.Lock()
		gpuAllocationMetricsMap = make(map[string]*GPUAllocationMetrics, 8)
		gpuAllocationMetricsLock.Unlock()
	})

	Context("Using Tflops annotations", func() {
		It("should correctly set tflops when using tflops annotations", func() {
			pod = &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-worker-tflops",
					Namespace: "default",
					Labels: map[string]string{
						constants.WorkloadKey: "test-workload",
					},
					Annotations: map[string]string{
						constants.TFLOPSRequestAnnotation: "100",
						constants.TFLOPSLimitAnnotation:   "200",
						constants.VRAMRequestAnnotation:   "10Gi",
						constants.VRAMLimitAnnotation:     "20Gi",
						constants.GpuCountAnnotation:      "1",
						constants.GpuPoolKey:              "default-pool",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, capacityMap)

			workerMetricsLock.RLock()
			metrics, exists := workerMetricsMap[pod.Name]
			workerMetricsLock.RUnlock()

			Expect(exists).To(BeTrue())
			Expect(metrics.TflopsRequest).To(BeNumerically("~", 100.0, 0.01))
			Expect(metrics.TflopsLimit).To(BeNumerically("~", 200.0, 0.01))
		})
	})

	Context("Using ComputePercent annotations (auto-migration)", func() {
		It("should convert 100% compute-percent to full A100 tflops", func() {
			pod = &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-worker-compute-percent-a100",
					Namespace: "default",
					Labels: map[string]string{
						constants.WorkloadKey: "test-workload",
					},
					Annotations: map[string]string{
						constants.ComputeRequestAnnotation: "100",
						constants.ComputeLimitAnnotation:   "100",
						constants.VRAMRequestAnnotation:    "80Gi",
						constants.VRAMLimitAnnotation:      "80Gi",
						constants.GpuCountAnnotation:       "1",
						constants.GPUModelAnnotation:       "A100",
						constants.GpuPoolKey:               "default-pool",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, capacityMap)

			workerMetricsLock.RLock()
			metrics, exists := workerMetricsMap[pod.Name]
			workerMetricsLock.RUnlock()

			Expect(exists).To(BeTrue())
			Expect(metrics.TflopsRequest).To(BeNumerically("~", 312.0, 0.01))
			Expect(metrics.TflopsLimit).To(BeNumerically("~", 312.0, 0.01))
		})

		It("should convert 50% compute-percent to half H100 tflops", func() {
			pod = &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-worker-compute-percent-h100-half",
					Namespace: "default",
					Labels: map[string]string{
						constants.WorkloadKey: "test-workload",
					},
					Annotations: map[string]string{
						constants.ComputeRequestAnnotation: "50",
						constants.ComputeLimitAnnotation:   "50",
						constants.VRAMRequestAnnotation:    "40Gi",
						constants.VRAMLimitAnnotation:      "40Gi",
						constants.GpuCountAnnotation:       "1",
						constants.GPUModelAnnotation:       "H100",
						constants.GpuPoolKey:               "default-pool",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, capacityMap)

			workerMetricsLock.RLock()
			metrics, exists := workerMetricsMap[pod.Name]
			workerMetricsLock.RUnlock()

			Expect(exists).To(BeTrue())
			Expect(metrics.TflopsRequest).To(BeNumerically("~", 378.0, 0.01))
			Expect(metrics.TflopsLimit).To(BeNumerically("~", 378.0, 0.01))
		})
	})

	Context("Edge cases", func() {
		It("should set tflops to 0 when compute-percent is used but gpu-model is missing", func() {
			pod = &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-worker-no-model",
					Namespace: "default",
					Labels: map[string]string{
						constants.WorkloadKey: "test-workload",
					},
					Annotations: map[string]string{
						constants.ComputeRequestAnnotation: "100",
						constants.ComputeLimitAnnotation:   "100",
						constants.VRAMRequestAnnotation:    "80Gi",
						constants.VRAMLimitAnnotation:      "80Gi",
						constants.GpuCountAnnotation:       "1",
						constants.GpuPoolKey:               "default-pool",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, capacityMap)

			workerMetricsLock.RLock()
			metrics, exists := workerMetricsMap[pod.Name]
			workerMetricsLock.RUnlock()

			Expect(exists).To(BeTrue())
			Expect(metrics.TflopsRequest).To(BeNumerically("==", 0.0))
			Expect(metrics.TflopsLimit).To(BeNumerically("==", 0.0))
		})

		It("should set tflops to 0 when gpu-model is not found in capacity map", func() {
			pod = &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-worker-unknown-model",
					Namespace: "default",
					Labels: map[string]string{
						constants.WorkloadKey: "test-workload",
					},
					Annotations: map[string]string{
						constants.ComputeRequestAnnotation: "100",
						constants.ComputeLimitAnnotation:   "100",
						constants.VRAMRequestAnnotation:    "80Gi",
						constants.VRAMLimitAnnotation:      "80Gi",
						constants.GpuCountAnnotation:       "1",
						constants.GPUModelAnnotation:       "UnknownGPU",
						constants.GpuPoolKey:               "default-pool",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, capacityMap)

			workerMetricsLock.RLock()
			metrics, exists := workerMetricsMap[pod.Name]
			workerMetricsLock.RUnlock()

			Expect(exists).To(BeTrue())
			Expect(metrics.TflopsRequest).To(BeNumerically("==", 0.0))
			Expect(metrics.TflopsLimit).To(BeNumerically("==", 0.0))
		})

		It("should correctly convert different compute-percent for request and limit", func() {
			pod = &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-worker-different-percent",
					Namespace: "default",
					Labels: map[string]string{
						constants.WorkloadKey: "test-workload",
					},
					Annotations: map[string]string{
						constants.ComputeRequestAnnotation: "30",
						constants.ComputeLimitAnnotation:   "100",
						constants.VRAMRequestAnnotation:    "24Gi",
						constants.VRAMLimitAnnotation:      "80Gi",
						constants.GpuCountAnnotation:       "1",
						constants.GPUModelAnnotation:       "A100",
						constants.GpuPoolKey:               "default-pool",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, capacityMap)

			workerMetricsLock.RLock()
			metrics, exists := workerMetricsMap[pod.Name]
			workerMetricsLock.RUnlock()

			Expect(exists).To(BeTrue())
			Expect(metrics.TflopsRequest).To(BeNumerically("~", 93.6, 0.01))
			Expect(metrics.TflopsLimit).To(BeNumerically("~", 312.0, 0.01))
		})

		It("should update existing metrics when called multiple times", func() {
			pod = &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-worker-update",
					Namespace: "default",
					Labels: map[string]string{
						constants.WorkloadKey: "test-workload",
					},
					Annotations: map[string]string{
						constants.ComputeRequestAnnotation: "50",
						constants.ComputeLimitAnnotation:   "50",
						constants.VRAMRequestAnnotation:    "40Gi",
						constants.VRAMLimitAnnotation:      "40Gi",
						constants.GpuCountAnnotation:       "1",
						constants.GPUModelAnnotation:       "A100",
						constants.GpuPoolKey:               "default-pool",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, capacityMap)

			workerMetricsLock.RLock()
			metrics1 := workerMetricsMap[pod.Name]
			workerMetricsLock.RUnlock()
			Expect(metrics1).NotTo(BeNil())
			Expect(metrics1.TflopsRequest).To(BeNumerically("~", 156.0, 0.01))
			Expect(metrics1.TflopsLimit).To(BeNumerically("~", 156.0, 0.01))

			// Update annotation to 100%
			pod.Annotations[constants.ComputeRequestAnnotation] = "100"
			pod.Annotations[constants.ComputeLimitAnnotation] = "100"

			SetWorkerMetricsByWorkload(pod, capacityMap)

			workerMetricsLock.RLock()
			metrics2 := workerMetricsMap[pod.Name]
			workerMetricsLock.RUnlock()

			Expect(metrics2).NotTo(BeNil())
			Expect(metrics2.TflopsRequest).To(BeNumerically("~", 312.0, 0.01))
			Expect(metrics2.TflopsLimit).To(BeNumerically("~", 312.0, 0.01))
		})
	})
})
