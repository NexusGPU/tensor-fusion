package gpuallocator

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestMultiGPUSelection(t *testing.T) {
	// Create test GPUs with different node labels
	gpus := []tfv1.GPU{
		// Node 1 GPUs
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "gpu-1-1",
				Labels: map[string]string{
					constants.LabelKeyOwner: "node-1",
				},
			},
			Status: tfv1.GPUStatus{
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("10"),
					Vram:   resource.MustParse("40Gi"),
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
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("12"),
					Vram:   resource.MustParse("42Gi"),
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
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("14"),
					Vram:   resource.MustParse("44Gi"),
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
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("20"),
					Vram:   resource.MustParse("80Gi"),
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
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("18"),
					Vram:   resource.MustParse("75Gi"),
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
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("30"),
					Vram:   resource.MustParse("100Gi"),
				},
			},
		},
	}

	t.Run("LowLoadFirst_MultiGPU", func(t *testing.T) {
		strategy := LowLoadFirst{}

		// Test selecting 2 GPUs
		selected, err := strategy.SelectGPUs(gpus, 2)
		assert.NoError(t, err)
		assert.Equal(t, 2, len(selected))

		// Should select from node-2 as it has higher resources
		assert.Equal(t, "node-2", selected[0].Labels[constants.LabelKeyOwner])
		assert.Equal(t, "node-2", selected[1].Labels[constants.LabelKeyOwner])

		// Should select GPUs in order of highest resources first
		assert.Equal(t, "gpu-2-1", selected[0].Name)
		assert.Equal(t, "gpu-2-2", selected[1].Name)

		// Test selecting 3 GPUs
		selected, err = strategy.SelectGPUs(gpus, 3)
		assert.NoError(t, err)
		assert.Equal(t, 3, len(selected))

		// Should select from node-1 as it's the only node with 3 GPUs
		assert.Equal(t, "node-1", selected[0].Labels[constants.LabelKeyOwner])
		assert.Equal(t, "node-1", selected[1].Labels[constants.LabelKeyOwner])
		assert.Equal(t, "node-1", selected[2].Labels[constants.LabelKeyOwner])

		// Should select GPUs in order of highest resources first
		assert.Equal(t, "gpu-1-3", selected[0].Name)
		assert.Equal(t, "gpu-1-2", selected[1].Name)
		assert.Equal(t, "gpu-1-1", selected[2].Name)

		// Test selecting more GPUs than available on any node
		selected, err = strategy.SelectGPUs(gpus, 4)
		assert.Error(t, err)
		assert.Nil(t, selected)
	})

	t.Run("CompactFirst_MultiGPU", func(t *testing.T) {
		strategy := CompactFirst{}

		// Test selecting 2 GPUs
		selected, err := strategy.SelectGPUs(gpus, 2)
		assert.NoError(t, err)
		assert.Equal(t, 2, len(selected))

		// Should select from node-1 as it has lower resources
		assert.Equal(t, "node-1", selected[0].Labels[constants.LabelKeyOwner])
		assert.Equal(t, "node-1", selected[1].Labels[constants.LabelKeyOwner])

		// Should select GPUs in order of lowest resources first
		assert.Equal(t, "gpu-1-1", selected[0].Name)
		assert.Equal(t, "gpu-1-2", selected[1].Name)

		// Test selecting 3 GPUs
		selected, err = strategy.SelectGPUs(gpus, 3)
		assert.NoError(t, err)
		assert.Equal(t, 3, len(selected))

		// Should select from node-1 as it's the only node with 3 GPUs
		assert.Equal(t, "node-1", selected[0].Labels[constants.LabelKeyOwner])
		assert.Equal(t, "node-1", selected[1].Labels[constants.LabelKeyOwner])
		assert.Equal(t, "node-1", selected[2].Labels[constants.LabelKeyOwner])

		// Should select GPUs in order of lowest resources first
		assert.Equal(t, "gpu-1-1", selected[0].Name)
		assert.Equal(t, "gpu-1-2", selected[1].Name)
		assert.Equal(t, "gpu-1-3", selected[2].Name)

		// Test selecting more GPUs than available on any node
		selected, err = strategy.SelectGPUs(gpus, 4)
		assert.Error(t, err)
		assert.Nil(t, selected)
	})
}
