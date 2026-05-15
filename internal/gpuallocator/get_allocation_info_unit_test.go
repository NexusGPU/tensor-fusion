package gpuallocator

import (
	"sync"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// newAllocatorForGetInfoTest builds an isolated GpuAllocator with the three
// stores GetAllocationInfo reads from, without running the constructor (which
// depends on real K8s client / indexAllocator).
func newAllocatorForGetInfoTest() *GpuAllocator {
	cap := &tfv1.Resource{
		Tflops: resource.MustParse("100"),
		Vram:   resource.MustParse("80Gi"),
	}
	gpu := &tfv1.GPU{}
	gpu.Name = "gpu-1"
	gpu.Status.Capacity = cap.DeepCopy()
	gpu.Status.Available = cap.DeepCopy()

	gpuKey := types.NamespacedName{Name: "gpu-1"}
	allocReq := &tfv1.AllocRequest{PodMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod-1"}}

	return &GpuAllocator{
		gpuStore:        map[types.NamespacedName]*tfv1.GPU{gpuKey: gpu},
		nodeWorkerStore: map[string]map[types.NamespacedName]struct{}{"node-a": {gpuKey: {}}},
		uniqueAllocation: map[string]*tfv1.AllocRequest{
			"uid-1": allocReq,
		},
	}
}

func TestGetAllocationInfo_ReturnsDeepCopies(t *testing.T) {
	s := newAllocatorForGetInfoTest()
	gpuKey := types.NamespacedName{Name: "gpu-1"}

	gpuStore, nodeWorkerStore, uniqueAllocation := s.GetAllocationInfo()

	// 1. gpuStore: mutating the returned GPU must not affect internal store.
	returnedGPU := gpuStore[gpuKey]
	assert.NotNil(t, returnedGPU)
	returnedGPU.Status.Available.Tflops = resource.MustParse("0")
	originalInternal := s.gpuStore[gpuKey]
	assert.NotEqual(t, "0", originalInternal.Status.Available.Tflops.String(),
		"GetAllocationInfo must return a deep copy of gpuStore values")

	// 2. nodeWorkerStore: mutating the returned inner map must not affect internal.
	innerCopy := nodeWorkerStore["node-a"]
	innerCopy[types.NamespacedName{Name: "extra"}] = struct{}{}
	assert.NotContains(t, s.nodeWorkerStore["node-a"], types.NamespacedName{Name: "extra"},
		"GetAllocationInfo must return a copied inner map for nodeWorkerStore")

	// 3. uniqueAllocation: mutating the returned value must not affect internal.
	returnedReq := uniqueAllocation["uid-1"]
	assert.NotNil(t, returnedReq)
	returnedReq.PodMeta.Name = "tampered"
	assert.Equal(t, "pod-1", s.uniqueAllocation["uid-1"].PodMeta.Name,
		"GetAllocationInfo must return a deep copy of uniqueAllocation values")
}

func TestGetAllocationInfo_ConcurrentReadIsSafe(t *testing.T) {
	s := newAllocatorForGetInfoTest()
	const goroutines = 8
	const iterations = 50

	var wg sync.WaitGroup
	wg.Add(goroutines)
	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				gpuStore, _, _ := s.GetAllocationInfo()
				// Mutating snapshots in parallel must not race with the allocator's
				// internal state (RWMutex + deep copy is what protects us).
				if g, ok := gpuStore[types.NamespacedName{Name: "gpu-1"}]; ok && g.Status.Available != nil {
					g.Status.Available.Tflops = resource.MustParse("0")
				}
			}
		}()
	}
	wg.Wait()

	// Internal state must remain intact.
	internalGPU := s.gpuStore[types.NamespacedName{Name: "gpu-1"}]
	assert.Equal(t, "100", internalGPU.Status.Available.Tflops.String())
}
