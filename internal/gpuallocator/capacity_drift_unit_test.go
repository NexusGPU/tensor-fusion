package gpuallocator

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const testGPU1Name = "gpu-1"

func qty(s string) resource.Quantity {
	return resource.MustParse(s)
}

func makeGPU(name string, capTflops, capVram, availTflops, availVram string) *tfv1.GPU {
	g := &tfv1.GPU{}
	g.Name = name
	g.Status.Capacity = &tfv1.Resource{Tflops: qty(capTflops), Vram: qty(capVram)}
	g.Status.Available = &tfv1.Resource{Tflops: qty(availTflops), Vram: qty(availVram)}
	return g
}

func TestClampGPUAvailableToCapacity(t *testing.T) {
	t.Run("clamp Available > Capacity to Capacity", func(t *testing.T) {
		g := makeGPU("g", "71", "24Gi", "142", "48Gi")
		clampGPUAvailableToCapacity(g)
		assert.Equal(t, "71", g.Status.Available.Tflops.String())
		assert.Equal(t, "24Gi", g.Status.Available.Vram.String())
	})

	t.Run("Available <= Capacity is left alone", func(t *testing.T) {
		g := makeGPU("g", "71", "24Gi", "30", "10Gi")
		clampGPUAvailableToCapacity(g)
		assert.Equal(t, "30", g.Status.Available.Tflops.String())
		assert.Equal(t, "10Gi", g.Status.Available.Vram.String())
	})

	t.Run("nil capacity/available is a no-op (no panic)", func(t *testing.T) {
		g := &tfv1.GPU{}
		clampGPUAvailableToCapacity(g)

		g.Status.Capacity = &tfv1.Resource{Tflops: qty("71"), Vram: qty("24Gi")}
		clampGPUAvailableToCapacity(g) // available still nil
		g.Status.Available = &tfv1.Resource{Tflops: qty("100"), Vram: qty("48Gi")}
		clampGPUAvailableToCapacity(g)
		assert.Equal(t, "71", g.Status.Available.Tflops.String())
		assert.Equal(t, "24Gi", g.Status.Available.Vram.String())
	})

	t.Run("mixed: only Vram exceeds", func(t *testing.T) {
		g := makeGPU("g", "71", "24Gi", "30", "100Gi")
		clampGPUAvailableToCapacity(g)
		assert.Equal(t, "30", g.Status.Available.Tflops.String())
		assert.Equal(t, "24Gi", g.Status.Available.Vram.String())
	})
}

func TestRecomputeGPUAvailableFromAllocations(t *testing.T) {
	mkReq := func(podName string, gpuNames []string, tflops, vram string) *tfv1.AllocRequest {
		return &tfv1.AllocRequest{
			PodMeta:  metav1.ObjectMeta{Name: podName, Namespace: "ns"},
			GPUNames: gpuNames,
			Request:  tfv1.Resource{Tflops: qty(tflops), Vram: qty(vram)},
		}
	}

	t.Run("subtracts requests targeting this GPU", func(t *testing.T) {
		s := &GpuAllocator{
			uniqueAllocation: map[string]*tfv1.AllocRequest{
				"uid-a": mkReq("pod-a", []string{testGPU1Name}, "20", "8Gi"),
				"uid-b": mkReq("pod-b", []string{testGPU1Name}, "10", "4Gi"),
				"uid-c": mkReq("pod-c", []string{"gpu-2"}, "30", "8Gi"), // different GPU
			},
		}
		g := makeGPU(testGPU1Name, "71", "24Gi", "71", "24Gi")
		s.recomputeGPUAvailableFromAllocations(g)
		// 71 - 20 - 10 = 41
		assert.Equal(t, "41", g.Status.Available.Tflops.String())
		// 24Gi - 8Gi - 4Gi = 12Gi
		assert.Equal(t, "12Gi", g.Status.Available.Vram.String())
	})

	t.Run("no allocations -> Available equals Capacity", func(t *testing.T) {
		s := &GpuAllocator{uniqueAllocation: map[string]*tfv1.AllocRequest{}}
		g := makeGPU(testGPU1Name, "71", "24Gi", "0", "0")
		s.recomputeGPUAvailableFromAllocations(g)
		assert.Equal(t, "71", g.Status.Available.Tflops.String())
		assert.Equal(t, "24Gi", g.Status.Available.Vram.String())
	})

	t.Run("nil capacity is a no-op (no panic)", func(t *testing.T) {
		s := &GpuAllocator{uniqueAllocation: map[string]*tfv1.AllocRequest{}}
		g := &tfv1.GPU{}
		g.Name = testGPU1Name
		// Status.Capacity is nil
		s.recomputeGPUAvailableFromAllocations(g)
		assert.Nil(t, g.Status.Capacity)
	})
}

// TestHandleGPUUpdateCapacityDiff_HypervisorRestartPattern reproduces the exact
// failure mode observed on dev for gpu-f5d00867: GPU CR is wiped (capacity=0),
// the worker pod survives, and the hypervisor later restores capacity. Without
// the recompute, Available would jump from 0 to Capacity even though the worker
// is still holding allocation -> on subsequent Dealloc, Available > Capacity.
func TestHandleGPUUpdateCapacityDiff_HypervisorRestartPattern(t *testing.T) {
	gpuName := "gpu-f5d00867"
	// Worker is allocated full GPU (vram=24Gi, tflops via ComputePercent absent here
	// -> use plain Request.Tflops=71 for simplicity).
	s := &GpuAllocator{
		uniqueAllocation: map[string]*tfv1.AllocRequest{
			"worker-uid": {
				PodMeta:  metav1.ObjectMeta{Name: "worker", Namespace: "ns", UID: "worker-uid"},
				GPUNames: []string{gpuName},
				Request:  tfv1.Resource{Tflops: qty("71"), Vram: qty("24Gi")},
			},
		},
	}
	// "old" reflects in-memory state right after the GPU CR was re-created with
	// empty status (handleGPUCreate initialized Available=Capacity=zero).
	old := makeGPU(gpuName, "0", "0", "0", "0")
	// "gpu" is what the hypervisor publishes after restart: full capacity.
	incoming := makeGPU(gpuName, "71", "24Gi", "71", "24Gi")

	s.handleGPUUpdateCapacityDiff(old, incoming)

	// After the fix, Available reflects the active allocation (worker holds
	// the full GPU), so Available should be zero, not 71/24Gi.
	assert.Equal(t, "0", old.Status.Available.Tflops.String(),
		"Available.Tflops should be 0 (worker holds full GPU); raw-diff math would give 71")
	assert.Equal(t, "0", old.Status.Available.Vram.String(),
		"Available.Vram should be 0; raw-diff math would give 24Gi")
	// Capacity correctly updated.
	assert.Equal(t, "71", old.Status.Capacity.Tflops.String())
	assert.Equal(t, "24Gi", old.Status.Capacity.Vram.String())
}

// TestHandleGPUUpdateCapacityDiff_NormalGrowthStillWorks ensures the recompute
// only triggers on the zero-capacity transition, not on routine updates.
func TestHandleGPUUpdateCapacityDiff_NormalGrowthStillWorks(t *testing.T) {
	gpuName := testGPU1Name
	s := &GpuAllocator{uniqueAllocation: map[string]*tfv1.AllocRequest{}}
	old := makeGPU(gpuName, "60", "20Gi", "50", "16Gi") // 10/4Gi already used
	incoming := makeGPU(gpuName, "71", "24Gi", "71", "24Gi")

	s.handleGPUUpdateCapacityDiff(old, incoming)

	// Diff +11/+4Gi added to Available -> 50+11=61, 16Gi+4Gi=20Gi
	assert.Equal(t, "61", old.Status.Available.Tflops.String())
	assert.Equal(t, "20Gi", old.Status.Available.Vram.String())
	assert.Equal(t, "71", old.Status.Capacity.Tflops.String())
	assert.Equal(t, "24Gi", old.Status.Capacity.Vram.String())
}

// TestHandleGPUUpdateCapacityDiff_ClampPreventsOverCapacity guards against the
// observable >Capacity symptom even if some upstream code path misbehaves.
func TestHandleGPUUpdateCapacityDiff_ClampPreventsOverCapacity(t *testing.T) {
	gpuName := testGPU1Name
	s := &GpuAllocator{uniqueAllocation: map[string]*tfv1.AllocRequest{}}
	// Pathological: incoming capacity is smaller than old Available (could
	// happen if Capacity is downgraded). Available must clamp to new Capacity.
	old := makeGPU(gpuName, "100", "40Gi", "100", "40Gi")
	incoming := makeGPU(gpuName, "71", "24Gi", "71", "24Gi")

	s.handleGPUUpdateCapacityDiff(old, incoming)

	assert.True(t, old.Status.Available.Tflops.Cmp(old.Status.Capacity.Tflops) <= 0,
		"Available.Tflops must not exceed Capacity.Tflops after diff")
	assert.True(t, old.Status.Available.Vram.Cmp(old.Status.Capacity.Vram) <= 0,
		"Available.Vram must not exceed Capacity.Vram after diff")
}
