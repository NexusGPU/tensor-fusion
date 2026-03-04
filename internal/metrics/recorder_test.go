package metrics

import (
	"bytes"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

var _ = Describe("SetWorkerMetricsByWorkload", func() {
	var (
		pod      *corev1.Pod
		gpuStore map[types.NamespacedName]*tfv1.GPU
	)

	BeforeEach(func() {
		// Create GPU store with GPU ID to model mapping
		gpuStore = make(map[types.NamespacedName]*tfv1.GPU)

		// Create A100 GPU
		gpuA100 := &tfv1.GPU{
			ObjectMeta: metav1.ObjectMeta{
				Name: "gpu-a100-1",
			},
			Status: tfv1.GPUStatus{
				GPUModel: "A100",
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("312"),
					Vram:   resource.MustParse("80Gi"),
				},
			},
		}
		gpuStore[types.NamespacedName{Name: "gpu-a100-1"}] = gpuA100

		// Create H100 GPU
		gpuH100 := &tfv1.GPU{
			ObjectMeta: metav1.ObjectMeta{
				Name: "gpu-h100-1",
			},
			Status: tfv1.GPUStatus{
				GPUModel: "H100",
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("756"),
					Vram:   resource.MustParse("80Gi"),
				},
			},
		}
		gpuStore[types.NamespacedName{Name: "gpu-h100-1"}] = gpuH100

		// Create another A100 GPU for update test
		gpuA100_2 := &tfv1.GPU{
			ObjectMeta: metav1.ObjectMeta{
				Name: "gpu-a100-2",
			},
			Status: tfv1.GPUStatus{
				GPUModel: "A100",
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("312"),
					Vram:   resource.MustParse("80Gi"),
				},
			},
		}
		gpuStore[types.NamespacedName{Name: "gpu-a100-2"}] = gpuA100_2

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
						constants.GPUDeviceIDsAnnotation:  "gpu-a100-1",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, gpuStore)

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
						constants.GPUDeviceIDsAnnotation:   "gpu-a100-1",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, gpuStore)

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
						constants.GPUDeviceIDsAnnotation:   "gpu-h100-1",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, gpuStore)

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
						constants.GPUDeviceIDsAnnotation:   "gpu-missing-1",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, gpuStore)

			workerMetricsLock.RLock()
			metrics, exists := workerMetricsMap[pod.Name]
			workerMetricsLock.RUnlock()

			Expect(exists).To(BeTrue())
			Expect(metrics.TflopsRequest).To(BeNumerically("==", 0.0))
			Expect(metrics.TflopsLimit).To(BeNumerically("==", 0.0))
		})

		It("should set tflops to 0 when gpu-model is not found in capacity map", func() {
			// Create a GPU with unknown model for this test
			gpuUnknown := &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gpu-unknown-1",
				},
				Status: tfv1.GPUStatus{
					GPUModel: "UnknownGPU",
				},
			}
			gpuStore[types.NamespacedName{Name: "gpu-unknown-1"}] = gpuUnknown

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
						constants.GPUDeviceIDsAnnotation:   "gpu-unknown-1",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, gpuStore)

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
						constants.GPUDeviceIDsAnnotation:   "gpu-a100-1",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, gpuStore)

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
						constants.GPUDeviceIDsAnnotation:   "gpu-a100-2",
					},
				},
			}

			SetWorkerMetricsByWorkload(pod, gpuStore)

			workerMetricsLock.RLock()
			metrics1 := workerMetricsMap[pod.Name]
			workerMetricsLock.RUnlock()
			Expect(metrics1).NotTo(BeNil())
			Expect(metrics1.TflopsRequest).To(BeNumerically("~", 156.0, 0.01))
			Expect(metrics1.TflopsLimit).To(BeNumerically("~", 156.0, 0.01))

			// Update annotation to 100%
			pod.Annotations[constants.ComputeRequestAnnotation] = "100"
			pod.Annotations[constants.ComputeLimitAnnotation] = "100"

			SetWorkerMetricsByWorkload(pod, gpuStore)

			workerMetricsLock.RLock()
			metrics2 := workerMetricsMap[pod.Name]
			workerMetricsLock.RUnlock()

			Expect(metrics2).NotTo(BeNil())
			Expect(metrics2.TflopsRequest).To(BeNumerically("~", 312.0, 0.01))
			Expect(metrics2.TflopsLimit).To(BeNumerically("~", 312.0, 0.01))
		})
	})
})

var _ = Describe("SetGPUMetrics", func() {
	BeforeEach(func() {
		gpuResourceMetricsLock.Lock()
		gpuResourceMetricsMap = make(map[string]*GPUResourceMetrics, 8)
		gpuResourceMetricsLock.Unlock()
	})

	Context("Normal cases", func() {
		It("should correctly set per-GPU card metrics", func() {
			gpuList := []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-uuid-1"},
					Status: tfv1.GPUStatus{
						Phase:    tfv1.TensorFusionGPUPhaseRunning,
						GPUModel: "A100",
						Capacity: &tfv1.Resource{
							Tflops: resource.MustParse("312"),
							Vram:   resource.MustParse("80Gi"),
						},
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("200"),
							Vram:   resource.MustParse("60Gi"),
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-uuid-2"},
					Status: tfv1.GPUStatus{
						Phase:    tfv1.TensorFusionGPUPhaseRunning,
						GPUModel: "H100",
						Capacity: &tfv1.Resource{
							Tflops: resource.MustParse("756"),
							Vram:   resource.MustParse("80Gi"),
						},
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("756"),
							Vram:   resource.MustParse("80Gi"),
						},
					},
				},
			}

			SetGPUMetrics(gpuList, "node-1", "pool-default")

			gpuResourceMetricsLock.RLock()
			defer gpuResourceMetricsLock.RUnlock()

			Expect(gpuResourceMetricsMap).To(HaveLen(2))

			m1 := gpuResourceMetricsMap["gpu-uuid-1"]
			Expect(m1).NotTo(BeNil())
			Expect(m1.NodeName).To(Equal("node-1"))
			Expect(m1.PoolName).To(Equal("pool-default"))
			Expect(m1.GPUModel).To(Equal("A100"))
			Expect(m1.Phase).To(Equal("Running"))
			Expect(m1.CapacityTflops).To(BeNumerically("~", 312.0, 0.01))
			Expect(m1.AvailableTflops).To(BeNumerically("~", 200.0, 0.01))
			Expect(m1.AllocatedTflops).To(BeNumerically("~", 112.0, 0.01))
			Expect(m1.AllocatedTflopsPercent).To(BeNumerically("~", 112.0/312.0*100, 0.01))
			Expect(m1.CapacityVramBytes).To(BeNumerically("~", 80.0*1024*1024*1024, 1))
			Expect(m1.AvailableVramBytes).To(BeNumerically("~", 60.0*1024*1024*1024, 1))
			Expect(m1.AllocatedVramBytes).To(BeNumerically("~", 20.0*1024*1024*1024, 1))
			Expect(m1.AllocatedVramPercent).To(BeNumerically("~", 25.0, 0.01))

			m2 := gpuResourceMetricsMap["gpu-uuid-2"]
			Expect(m2).NotTo(BeNil())
			Expect(m2.GPUModel).To(Equal("H100"))
			Expect(m2.AllocatedTflops).To(BeNumerically("~", 0.0, 0.01))
			Expect(m2.AllocatedTflopsPercent).To(BeNumerically("~", 0.0, 0.01))
			Expect(m2.AllocatedVramBytes).To(BeNumerically("~", 0.0, 0.01))
			Expect(m2.AllocatedVramPercent).To(BeNumerically("~", 0.0, 0.01))
		})
	})

	Context("Skip invalid GPUs", func() {
		It("should skip GPUs with nil capacity or available", func() {
			gpuList := []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-no-capacity"},
					Status: tfv1.GPUStatus{
						Phase:    tfv1.TensorFusionGPUPhasePending,
						GPUModel: "A100",
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("100"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-no-available"},
					Status: tfv1.GPUStatus{
						Phase:    tfv1.TensorFusionGPUPhasePending,
						GPUModel: "A100",
						Capacity: &tfv1.Resource{
							Tflops: resource.MustParse("312"),
							Vram:   resource.MustParse("80Gi"),
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-both-nil"},
					Status: tfv1.GPUStatus{
						Phase:    tfv1.TensorFusionGPUPhasePending,
						GPUModel: "A100",
					},
				},
			}

			SetGPUMetrics(gpuList, "node-1", "pool-default")

			gpuResourceMetricsLock.RLock()
			defer gpuResourceMetricsLock.RUnlock()

			Expect(gpuResourceMetricsMap).To(BeEmpty())
		})
	})

	Context("Update existing metrics", func() {
		It("should overwrite metrics on subsequent calls", func() {
			gpuList := []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-uuid-1"},
					Status: tfv1.GPUStatus{
						Phase:    tfv1.TensorFusionGPUPhaseRunning,
						GPUModel: "A100",
						Capacity: &tfv1.Resource{
							Tflops: resource.MustParse("312"),
							Vram:   resource.MustParse("80Gi"),
						},
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("312"),
							Vram:   resource.MustParse("80Gi"),
						},
					},
				},
			}

			SetGPUMetrics(gpuList, "node-1", "pool-default")

			gpuResourceMetricsLock.RLock()
			Expect(gpuResourceMetricsMap["gpu-uuid-1"].AllocatedTflops).To(BeNumerically("~", 0.0, 0.01))
			gpuResourceMetricsLock.RUnlock()

			gpuList[0].Status.Available = &tfv1.Resource{
				Tflops: resource.MustParse("200"),
				Vram:   resource.MustParse("60Gi"),
			}

			SetGPUMetrics(gpuList, "node-1", "pool-default")

			gpuResourceMetricsLock.RLock()
			defer gpuResourceMetricsLock.RUnlock()
			Expect(gpuResourceMetricsMap["gpu-uuid-1"].AllocatedTflops).To(BeNumerically("~", 112.0, 0.01))
			Expect(gpuResourceMetricsMap["gpu-uuid-1"].AvailableTflops).To(BeNumerically("~", 200.0, 0.01))
		})
	})

	Context("Stale GPU cleanup", func() {
		It("should remove metrics for GPUs no longer in the list for the same node", func() {
			gpuList := []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						Phase: tfv1.TensorFusionGPUPhaseRunning, GPUModel: "A100",
						Capacity:  &tfv1.Resource{Tflops: resource.MustParse("312"), Vram: resource.MustParse("80Gi")},
						Available: &tfv1.Resource{Tflops: resource.MustParse("312"), Vram: resource.MustParse("80Gi")},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-2"},
					Status: tfv1.GPUStatus{
						Phase: tfv1.TensorFusionGPUPhaseRunning, GPUModel: "A100",
						Capacity:  &tfv1.Resource{Tflops: resource.MustParse("312"), Vram: resource.MustParse("80Gi")},
						Available: &tfv1.Resource{Tflops: resource.MustParse("312"), Vram: resource.MustParse("80Gi")},
					},
				},
			}
			SetGPUMetrics(gpuList, "node-1", "pool-default")

			// gpu-2 disappears from node-1
			SetGPUMetrics(gpuList[:1], "node-1", "pool-default")

			gpuResourceMetricsLock.RLock()
			defer gpuResourceMetricsLock.RUnlock()
			Expect(gpuResourceMetricsMap).To(HaveLen(1))
			Expect(gpuResourceMetricsMap).To(HaveKey("gpu-1"))
			Expect(gpuResourceMetricsMap).NotTo(HaveKey("gpu-2"))
		})

		It("should not affect GPUs on other nodes", func() {
			gpuList1 := []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-n1-1"},
					Status: tfv1.GPUStatus{
						Phase: tfv1.TensorFusionGPUPhaseRunning, GPUModel: "A100",
						Capacity:  &tfv1.Resource{Tflops: resource.MustParse("312"), Vram: resource.MustParse("80Gi")},
						Available: &tfv1.Resource{Tflops: resource.MustParse("312"), Vram: resource.MustParse("80Gi")},
					},
				},
			}
			gpuList2 := []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-n2-1"},
					Status: tfv1.GPUStatus{
						Phase: tfv1.TensorFusionGPUPhaseRunning, GPUModel: "H100",
						Capacity:  &tfv1.Resource{Tflops: resource.MustParse("756"), Vram: resource.MustParse("80Gi")},
						Available: &tfv1.Resource{Tflops: resource.MustParse("756"), Vram: resource.MustParse("80Gi")},
					},
				},
			}
			SetGPUMetrics(gpuList1, "node-1", "pool-default")
			SetGPUMetrics(gpuList2, "node-2", "pool-default")

			// node-1 loses all GPUs
			SetGPUMetrics([]tfv1.GPU{}, "node-1", "pool-default")

			gpuResourceMetricsLock.RLock()
			defer gpuResourceMetricsLock.RUnlock()
			Expect(gpuResourceMetricsMap).To(HaveLen(1))
			Expect(gpuResourceMetricsMap).To(HaveKey("gpu-n2-1"))
		})
	})

	Context("Zero capacity edge case", func() {
		It("should not divide by zero when capacity is 0", func() {
			gpuList := []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-zero-cap"},
					Status: tfv1.GPUStatus{
						Phase:    tfv1.TensorFusionGPUPhaseRunning,
						GPUModel: "A100",
						Capacity: &tfv1.Resource{
							Tflops: resource.MustParse("0"),
							Vram:   resource.MustParse("0"),
						},
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("0"),
							Vram:   resource.MustParse("0"),
						},
					},
				},
			}

			SetGPUMetrics(gpuList, "node-1", "pool-default")

			gpuResourceMetricsLock.RLock()
			defer gpuResourceMetricsLock.RUnlock()

			m := gpuResourceMetricsMap["gpu-zero-cap"]
			Expect(m).NotTo(BeNil())
			Expect(m.AllocatedTflopsPercent).To(BeNumerically("==", 0.0))
			Expect(m.AllocatedVramPercent).To(BeNumerically("==", 0.0))
		})
	})
})

var _ = Describe("RemoveGPUMetricsByNode", func() {
	BeforeEach(func() {
		gpuResourceMetricsLock.Lock()
		gpuResourceMetricsMap = make(map[string]*GPUResourceMetrics, 8)
		gpuResourceMetricsLock.Unlock()
	})

	It("should remove all GPU metrics by node name", func() {
		gpuList1 := []tfv1.GPU{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-n1-1"},
				Status: tfv1.GPUStatus{
					Phase: tfv1.TensorFusionGPUPhaseRunning, GPUModel: "A100",
					Capacity:  &tfv1.Resource{Tflops: resource.MustParse("312"), Vram: resource.MustParse("80Gi")},
					Available: &tfv1.Resource{Tflops: resource.MustParse("312"), Vram: resource.MustParse("80Gi")},
				},
			},
		}
		gpuList2 := []tfv1.GPU{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-n2-1"},
				Status: tfv1.GPUStatus{
					Phase: tfv1.TensorFusionGPUPhaseRunning, GPUModel: "H100",
					Capacity:  &tfv1.Resource{Tflops: resource.MustParse("756"), Vram: resource.MustParse("80Gi")},
					Available: &tfv1.Resource{Tflops: resource.MustParse("756"), Vram: resource.MustParse("80Gi")},
				},
			},
		}
		SetGPUMetrics(gpuList1, "node-1", "pool-default")
		SetGPUMetrics(gpuList2, "node-2", "pool-default")

		RemoveGPUMetricsByNode("node-1")

		gpuResourceMetricsLock.RLock()
		defer gpuResourceMetricsLock.RUnlock()
		Expect(gpuResourceMetricsMap).To(HaveLen(1))
		Expect(gpuResourceMetricsMap).To(HaveKey("gpu-n2-1"))
	})
})

var _ = Describe("RecordMetrics GPU output", func() {
	BeforeEach(func() {
		workerMetricsLock.Lock()
		workerMetricsMap = make(map[string]*WorkerResourceMetrics, 8)
		workerMetricsLock.Unlock()

		nodeMetricsLock.Lock()
		nodeMetricsMap = make(map[string]*NodeResourceMetrics, 8)
		nodeMetricsLock.Unlock()

		poolMetricsLock.Lock()
		poolMetricsMap = make(map[string]*PoolResourceMetrics, 4)
		poolMetricsLock.Unlock()

		gpuAllocationMetricsLock.Lock()
		gpuAllocationMetricsMap = make(map[string]*GPUAllocationMetrics, 8)
		gpuAllocationMetricsLock.Unlock()

		gpuResourceMetricsLock.Lock()
		gpuResourceMetricsMap = make(map[string]*GPUResourceMetrics, 8)
		gpuResourceMetricsLock.Unlock()
	})

	It("should include tf_gpu_metrics in output", func() {
		gpuList := []tfv1.GPU{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-uuid-rec-1"},
				Status: tfv1.GPUStatus{
					Phase:    tfv1.TensorFusionGPUPhaseRunning,
					GPUModel: "A100",
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("312"),
						Vram:   resource.MustParse("80Gi"),
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("200"),
						Vram:   resource.MustParse("60Gi"),
					},
				},
			},
		}
		SetGPUMetrics(gpuList, "test-node", "test-pool")

		var buf bytes.Buffer
		mr := &MetricsRecorder{
			HourlyUnitPriceMap: map[string]float64{},
			WorkerUnitPriceMap: map[string]map[string]RawBillingPricing{},
		}
		mr.RecordMetrics(&buf)

		output := buf.String()
		Expect(output).To(ContainSubstring("tf_gpu_metrics"))
		Expect(output).To(ContainSubstring("gpu-uuid-rec-1"))
		Expect(output).To(ContainSubstring("test-node"))
		Expect(output).To(ContainSubstring("test-pool"))
		Expect(output).To(ContainSubstring("A100"))
		Expect(output).To(ContainSubstring("capacity_tflops"))
		Expect(output).To(ContainSubstring("available_tflops"))
		Expect(output).To(ContainSubstring("allocated_tflops"))
	})

	It("should not output tf_gpu_metrics when map is empty and other maps are also empty", func() {
		var buf bytes.Buffer
		mr := &MetricsRecorder{
			HourlyUnitPriceMap: map[string]float64{},
			WorkerUnitPriceMap: map[string]map[string]RawBillingPricing{},
		}
		mr.RecordMetrics(&buf)

		output := buf.String()
		Expect(strings.TrimSpace(output)).To(BeEmpty())
	})
})
