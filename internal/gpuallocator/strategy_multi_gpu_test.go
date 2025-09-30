package gpuallocator

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var _ = Describe("Multi-GPU Selection", func() {
	var gpus []*tfv1.GPU
	var nodeGpuStore map[string]map[string]*tfv1.GPU

	BeforeEach(func() {
		// Create test GPUs with different node labels
		gpus = []*tfv1.GPU{
			// Node 1 GPUs
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gpu-1-1",
					Labels: map[string]string{
						constants.LabelKeyOwner: "node-1",
					},
				},
				Status: tfv1.GPUStatus{
					NodeSelector: map[string]string{
						constants.KubernetesHostNameLabel: "node-1",
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("10"),
						Vram:   resource.MustParse("40Gi"),
					},
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("20"),
						Vram:   resource.MustParse("80Gi"),
					},
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gpu-1-2",
					Labels: map[string]string{
						constants.LabelKeyOwner: "node-1",
					},
				},
				Status: tfv1.GPUStatus{
					NodeSelector: map[string]string{
						constants.KubernetesHostNameLabel: "node-1",
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("12"),
						Vram:   resource.MustParse("42Gi"),
					},
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("20"),
						Vram:   resource.MustParse("80Gi"),
					},
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gpu-1-3",
					Labels: map[string]string{
						constants.LabelKeyOwner: "node-1",
					},
				},
				Status: tfv1.GPUStatus{
					NodeSelector: map[string]string{
						constants.KubernetesHostNameLabel: "node-1",
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("14"),
						Vram:   resource.MustParse("44Gi"),
					},
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("20"),
						Vram:   resource.MustParse("80Gi"),
					},
				},
			},
			// Node 2 GPUs
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gpu-2-1",
					Labels: map[string]string{
						constants.LabelKeyOwner: "node-2",
					},
				},
				Status: tfv1.GPUStatus{
					NodeSelector: map[string]string{
						constants.KubernetesHostNameLabel: "node-2",
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("20"),
						Vram:   resource.MustParse("80Gi"),
					},
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("25"),
						Vram:   resource.MustParse("100Gi"),
					},
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gpu-2-2",
					Labels: map[string]string{
						constants.LabelKeyOwner: "node-2",
					},
				},
				Status: tfv1.GPUStatus{
					NodeSelector: map[string]string{
						constants.KubernetesHostNameLabel: "node-2",
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("18"),
						Vram:   resource.MustParse("75Gi"),
					},
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("25"),
						Vram:   resource.MustParse("100Gi"),
					},
				},
			},
			// Node 3 GPU (only one)
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "gpu-3-1",
					Labels: map[string]string{
						constants.LabelKeyOwner: "node-3",
					},
				},
				Status: tfv1.GPUStatus{
					NodeSelector: map[string]string{
						constants.KubernetesHostNameLabel: "node-3",
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("30"),
						Vram:   resource.MustParse("100Gi"),
					},
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("40"),
						Vram:   resource.MustParse("120Gi"),
					},
				},
			},
		}

		// Create nodeGpuStore for testing
		nodeGpuStore = make(map[string]map[string]*tfv1.GPU)
		for _, gpu := range gpus {
			nodeName := gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
			if nodeGpuStore[nodeName] == nil {
				nodeGpuStore[nodeName] = make(map[string]*tfv1.GPU)
			}
			nodeGpuStore[nodeName][gpu.Name] = gpu
		}
	})

	Describe("LowLoadFirst Strategy", func() {
		It("should select GPUs with multi-GPU requirements", func() {
			strategy := LowLoadFirst{
				cfg: &config.GPUFitConfig{
					VramWeight:   0.5,
					TflopsWeight: 0.5,
				},
				nodeGpuStore: nodeGpuStore,
			}

			// Test selecting 2 GPUs
			selected, err := strategy.SelectGPUs(gpus, 2)
			Expect(err).To(Succeed())
			Expect(selected).To(HaveLen(2))

			// Should select from node-1 as it has highest total node score (168.75)
			Expect(selected[0].Labels[constants.LabelKeyOwner]).To(Equal("node-1"))
			Expect(selected[1].Labels[constants.LabelKeyOwner]).To(Equal("node-1"))

			// Should select GPUs in order of highest individual scores first
			Expect(selected[0].Name).To(Equal("gpu-1-3")) // 62.5 score
			Expect(selected[1].Name).To(Equal("gpu-1-2")) // 56.25 score

			// Test selecting 3 GPUs
			selected, err = strategy.SelectGPUs(gpus, 3)
			Expect(err).To(Succeed())
			Expect(selected).To(HaveLen(3))

			// Should select from node-1 as it's the only node with 3 GPUs
			Expect(selected[0].Labels[constants.LabelKeyOwner]).To(Equal("node-1"))
			Expect(selected[1].Labels[constants.LabelKeyOwner]).To(Equal("node-1"))
			Expect(selected[2].Labels[constants.LabelKeyOwner]).To(Equal("node-1"))

			// Should select GPUs in order of highest individual scores first
			Expect(selected[0].Name).To(Equal("gpu-1-3")) // 62.5 score
			Expect(selected[1].Name).To(Equal("gpu-1-2")) // 56.25 score
			Expect(selected[2].Name).To(Equal("gpu-1-1")) // 50 score

			// Test selecting more GPUs than available on any node
			selected, err = strategy.SelectGPUs(gpus, 4)
			Expect(err).To(HaveOccurred())
			Expect(selected).To(BeNil())
		})
	})

	Describe("CompactFirst Strategy", func() {
		It("should select GPUs with multi-GPU requirements", func() {
			strategy := CompactFirst{
				cfg: &config.GPUFitConfig{
					VramWeight:   0.5,
					TflopsWeight: 0.5,
				},
				nodeGpuStore: nodeGpuStore,
			}

			// Test selecting 2 GPUs
			selected, err := strategy.SelectGPUs(gpus, 2)
			Expect(err).To(Succeed())
			Expect(selected).To(HaveLen(2))

			// Should select from node-1 as it has lower resources
			Expect(selected[0].Labels[constants.LabelKeyOwner]).To(Equal("node-1"))
			Expect(selected[1].Labels[constants.LabelKeyOwner]).To(Equal("node-1"))

			// Should select GPUs in order of lowest resources first
			Expect(selected[0].Name).To(Equal("gpu-1-1"))
			Expect(selected[1].Name).To(Equal("gpu-1-2"))

			// Test selecting 3 GPUs
			selected, err = strategy.SelectGPUs(gpus, 3)
			Expect(err).To(Succeed())
			Expect(selected).To(HaveLen(3))

			// Should select from node-1 as it's the only node with 3 GPUs
			Expect(selected[0].Labels[constants.LabelKeyOwner]).To(Equal("node-1"))
			Expect(selected[1].Labels[constants.LabelKeyOwner]).To(Equal("node-1"))
			Expect(selected[2].Labels[constants.LabelKeyOwner]).To(Equal("node-1"))

			// Should select GPUs in order of lowest resources first
			Expect(selected[0].Name).To(Equal("gpu-1-1"))
			Expect(selected[1].Name).To(Equal("gpu-1-2"))
			Expect(selected[2].Name).To(Equal("gpu-1-3"))

			// Test selecting more GPUs than available on any node
			selected, err = strategy.SelectGPUs(gpus, 4)
			Expect(err).To(HaveOccurred())
			Expect(selected).To(BeNil())
		})
	})
})
