package gpuallocator

import (
	"context"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func newCapacityMissingTestGPU(name, owner string, phase tfv1.TensorFusionGPUPhase) *tfv1.GPU {
	return &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				constants.LabelKeyOwner: owner,
			},
		},
		Status: tfv1.GPUStatus{
			Phase:    phase,
			GPUModel: "NVIDIA-Test-GPU",
			Capacity: &tfv1.Resource{
				Tflops: resource.MustParse("100"),
				Vram:   resource.MustParse("16Gi"),
			},
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("100"),
				Vram:   resource.MustParse("16Gi"),
			},
		},
	}
}

// A GPU marked missing by hypervisor discovery must not contribute phantom
// capacity to GPUNode level Total/Available statistics.
func TestRefreshGPUNodeCapacityExcludesMissingGPUs(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	_ = tfv1.AddToScheme(scheme)

	nodeName := "capacity-missing-test-node"
	node := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{Name: nodeName},
	}
	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: "capacity-missing-test-pool"},
	}

	healthyGPU := newCapacityMissingTestGPU("gpu-healthy", nodeName, tfv1.TensorFusionGPUPhaseRunning)
	missingGPU := newCapacityMissingTestGPU("gpu-missing", nodeName, tfv1.TensorFusionGPUPhaseUnknown)
	missingGPU.Annotations = map[string]string{
		constants.GPUMissingSinceAnnotationKey: time.Now().Format(time.RFC3339),
	}

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPU{}, &tfv1.GPUNode{}).
		WithObjects(node, pool, healthyGPU, missingGPU).Build()

	allocator := NewGpuAllocator(ctx, nil, k8sClient, time.Second)

	_, err := RefreshGPUNodeCapacity(ctx, k8sClient, node, pool, allocator, nil)
	assert.NoError(t, err)

	expectedTflops := resource.MustParse("100")
	expectedVram := resource.MustParse("16Gi")
	assert.Equal(t, expectedTflops.String(), node.Status.TotalTFlops.String(),
		"missing GPU capacity must be excluded from TotalTFlops")
	assert.Equal(t, expectedVram.String(), node.Status.TotalVRAM.String(),
		"missing GPU capacity must be excluded from TotalVRAM")
	assert.Equal(t, expectedTflops.String(), node.Status.AvailableTFlops.String(),
		"missing GPU must be excluded from AvailableTFlops")
	assert.Equal(t, expectedVram.String(), node.Status.AvailableVRAM.String(),
		"missing GPU must be excluded from AvailableVRAM")
}
