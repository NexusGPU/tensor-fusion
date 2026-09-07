package gpuallocator

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func TestHandleGPUUpdateRefreshesNodeAndPoolIndexes(t *testing.T) {
	oldGPU := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "gpu-1",
			Labels: map[string]string{constants.GpuPoolKey: "pool-old"},
		},
		Status: tfv1.GPUStatus{
			NodeSelector: map[string]string{constants.KubernetesHostNameLabel: "node-old"},
			Capacity: &tfv1.Resource{
				Tflops: resource.MustParse("10"),
				Vram:   resource.MustParse("1Gi"),
			},
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("10"),
				Vram:   resource.MustParse("1Gi"),
			},
		},
	}
	newGPU := oldGPU.DeepCopy()
	newGPU.Labels[constants.GpuPoolKey] = "pool-new"
	newGPU.Status.NodeSelector[constants.KubernetesHostNameLabel] = "node-new"

	allocator := &GpuAllocator{
		gpuStore: map[types.NamespacedName]*tfv1.GPU{
			{Name: oldGPU.Name}: oldGPU,
		},
		nodeGpuStore: map[string]map[string]*tfv1.GPU{
			"node-old": {oldGPU.Name: oldGPU},
		},
		poolGpuStore: map[string]map[string]*tfv1.GPU{
			"pool-old": {oldGPU.Name: oldGPU},
		},
		nodeWorkerStore: map[string]map[types.NamespacedName]struct{}{
			"node-old": {},
		},
	}

	allocator.handleGPUUpdate(context.Background(), newGPU)

	_, oldNodeExists := allocator.nodeGpuStore["node-old"][oldGPU.Name]
	_, oldPoolExists := allocator.poolGpuStore["pool-old"][oldGPU.Name]
	newNodeGPU, newNodeExists := allocator.nodeGpuStore["node-new"][oldGPU.Name]
	newPoolGPU, newPoolExists := allocator.poolGpuStore["pool-new"][oldGPU.Name]

	require.False(t, oldNodeExists)
	require.False(t, oldPoolExists)
	require.True(t, newNodeExists)
	require.True(t, newPoolExists)
	require.Same(t, oldGPU, newNodeGPU)
	require.Same(t, oldGPU, newPoolGPU)
	_, oldWorkerExists := allocator.nodeWorkerStore["node-old"]
	require.False(t, oldWorkerExists)
}
