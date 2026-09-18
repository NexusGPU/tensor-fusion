package gpuallocator

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
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

// makeAllocReq builds a committed allocation record referencing gpuNames.
func makeAllocReq(podName string, gpuNames []string, tflops, vram string) *tfv1.AllocRequest {
	return &tfv1.AllocRequest{
		PodMeta:  metav1.ObjectMeta{Name: podName, Namespace: "ns", UID: types.UID(podName + "-uid")},
		GPUNames: gpuNames,
		Request:  tfv1.Resource{Tflops: qty(tflops), Vram: qty(vram)},
	}
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

// TestHandleGPUUpdateCapacityDiff_NormalGrowthStillWorks ensures a routine
// capacity growth still lands on the same value the old delta math produced:
// the rebuild derives it from the ledger, which is equivalent when nothing has
// drifted.
func TestHandleGPUUpdateCapacityDiff_NormalGrowthStillWorks(t *testing.T) {
	gpuName := testGPU1Name
	// Available (50) is Capacity (60) minus one committed 10/4Gi holder, and
	// the ledger is what the rebuild trusts - so register that holder here.
	s := newTestAllocator()
	s.uniqueAllocation = map[string]*tfv1.AllocRequest{
		"uid-a": makeAllocReq("pod-a", []string{gpuName}, "10", "4Gi"),
	}
	old := makeGPU(gpuName, "60", "20Gi", "50", "16Gi")
	incoming := makeGPU(gpuName, "71", "24Gi", "71", "24Gi")

	s.handleGPUUpdateCapacityDiff(old, incoming)

	// Same result as the old delta math: 71-10=61, 24Gi-4Gi=20Gi.
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

// Regression for the "capacity is correct but Available stays 681" report.
// 681 = 835 - (989 - 835): Available had already drifted by the time the
// capacity downgrade was observed (e.g. a percent-based holder released after
// the card shrank, which leaves a residual of percent x delta), and the raw
// delta math carried the drift over instead of repairing it. 681 != 835
// permanently blocks shared whole-GPU allocation, which requires
// Available == Capacity.
func TestHandleGPUUpdateCapacityDiff_RebuildsDriftedAvailable(t *testing.T) {
	gpuName := "gpu-drifted"
	s := newTestAllocator()
	old := makeGPU(gpuName, "989", "141Gi", "835", "141Gi")
	incoming := makeGPU(gpuName, "835", "141Gi", "835", "141Gi")

	s.handleGPUUpdateCapacityDiff(old, incoming)

	assert.Equal(t, "835", old.Status.Capacity.Tflops.String())
	assert.Equal(t, "835", old.Status.Available.Tflops.String(),
		"idle GPU must be rebuilt to capacity; raw delta math left it at 681")
	assert.True(t, old.Status.Available.Vram.Equal(old.Status.Capacity.Vram))
}

// The rebuild must keep partition holders accounted for: partitioned requests
// normally carry zero tflops/vram (the template defines their size), so a naive
// rebuild would advertise the whole card as free.
func TestHandleGPUUpdateCapacityDiff_KeepsPartitionedUsage(t *testing.T) {
	const model = "test-partition-model"
	MutatePartitionConfigForTesting(func(c *partitionConfig) {
		c.Templates[model] = map[string]config.PartitionTemplateInfo{
			"part-half": {
				TemplateID:      "part-half",
				Name:            "half",
				ComputePercent:  50,
				MemoryGigabytes: 71,
			},
		}
	})

	gpuName := "gpu-partitioned"
	s := newTestAllocator()
	s.uniqueAllocation = map[string]*tfv1.AllocRequest{
		"uid-p": {
			PodMeta:             metav1.ObjectMeta{Name: "pod-p", Namespace: "ns", UID: "uid-p"},
			GPUNames:            []string{gpuName},
			Isolation:           tfv1.IsolationModePartitioned,
			PartitionTemplateID: "part-half",
		},
	}
	old := makeGPU(gpuName, "100", "141Gi", "100", "141Gi")
	old.Status.GPUModel = model
	incoming := makeGPU(gpuName, "200", "141Gi", "200", "141Gi")
	incoming.Status.GPUModel = model

	s.handleGPUUpdateCapacityDiff(old, incoming)

	// Half of the new 200 TFlops capacity, 71Gi of the 141Gi card.
	assert.True(t, old.Status.Available.Tflops.Equal(qty("100")),
		"expected 50%% of the new capacity, got %s", old.Status.Available.Tflops.String())
	assert.True(t, old.Status.Available.Vram.Equal(qty("70Gi")),
		"expected the 141Gi card minus 71Gi, got %s", old.Status.Available.Vram.String())
}

// correctIdleGPUAvailableLocked is the 3-minute backstop: it repairs cards that
// are idle per both the status snapshot and the ledger, and leaves every card
// with a committed holder untouched.
func TestCorrectIdleGPUAvailableLocked(t *testing.T) {
	idle := makeGPU("gpu-idle", "835", "141Gi", "681", "141Gi")
	busy := makeGPU("gpu-busy", "835", "141Gi", "681", "141Gi")
	busy.Status.RunningApps = []*tfv1.RunningAppDetail{{Name: "wl", Namespace: "ns"}}
	ledgerOnly := makeGPU("gpu-ledger-only", "835", "141Gi", "681", "141Gi")
	noAvailable := makeGPU("gpu-no-available", "835", "141Gi", "0", "0")
	noAvailable.Status.Available = nil

	s := newTestAllocator()
	s.uniqueAllocation = map[string]*tfv1.AllocRequest{
		"uid-x": makeAllocReq("pod-x", []string{"gpu-ledger-only"}, "1", "1Gi"),
	}
	s.gpuStore = map[types.NamespacedName]*tfv1.GPU{
		{Name: "gpu-idle"}:         idle,
		{Name: "gpu-busy"}:         busy,
		{Name: "gpu-ledger-only"}:  ledgerOnly,
		{Name: "gpu-no-available"}: noAvailable,
	}

	assert.Equal(t, 2, s.correctIdleGPUAvailableLocked())

	assert.True(t, idle.Status.Available.Tflops.Equal(idle.Status.Capacity.Tflops),
		"idle drifted GPU must be restored to capacity")
	assert.True(t, idle.Status.Available.Vram.Equal(idle.Status.Capacity.Vram))
	assert.Contains(t, s.dirtyQueue, types.NamespacedName{Name: "gpu-idle"},
		"correction must be synced back to the CR")

	assert.NotNil(t, noAvailable.Status.Available, "a nil Available on an idle card is repaired too")
	assert.True(t, noAvailable.Status.Available.Tflops.Equal(noAvailable.Status.Capacity.Tflops))

	assert.Equal(t, "681", busy.Status.Available.Tflops.String(),
		"cards with a running app must be left alone")
	assert.Equal(t, "681", ledgerOnly.Status.Available.Tflops.String(),
		"cards still referenced by the ledger must be left alone")
}

// A rebuilt Available has to reach the CR: the sync loop only publishes dirty
// GPUs, and an idle card has no allocation event to mark it later, so without
// this the scheduler (which filters on CR copies, requiring
// Available == Capacity for shared whole-GPU placement) would keep blocking the
// card even though the allocator already corrected its own bookkeeping.
func TestHandleGPUUpdatePublishesRebuiltAvailable(t *testing.T) {
	gpuName := "gpu-cap-change"
	key := types.NamespacedName{Name: gpuName}
	s := newTestAllocator()
	s.gpuStore[key] = makeGPU(gpuName, "989", "141Gi", "681", "141Gi")
	incoming := makeGPU(gpuName, "835", "141Gi", "681", "141Gi")

	s.handleGPUUpdate(context.Background(), incoming)

	stored := s.gpuStore[key]
	assert.Equal(t, "835", stored.Status.Capacity.Tflops.String())
	assert.Equal(t, "835", stored.Status.Available.Tflops.String())
	assert.Contains(t, s.dirtyQueue, key, "rebuilt Available must be queued for the CR")
}
