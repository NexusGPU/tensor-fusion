package gpuallocator

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func sharedTestGPU(name, node string, mode tfv1.IsolationModeType) *tfv1.GPU {
	gpu := makeGPU(name, "100", "24Gi", "100", "24Gi")
	gpu.Status.IsolationMode = mode
	gpu.Status.NodeSelector = map[string]string{constants.KubernetesHostNameLabel: node}
	return gpu
}

func TestSharedAllocationConsumesAndRestoresWholeGPU(t *testing.T) {
	s := newTestAllocator()
	gpu := sharedTestGPU("gpu-1", "node-1", tfv1.IsolationModeSoft)
	req := &tfv1.AllocRequest{
		Isolation: tfv1.IsolationModeShared,
		Request:   tfv1.Resource{Tflops: qty("10"), Vram: qty("2Gi")},
		PodMeta:   metav1.ObjectMeta{UID: "pod-1"},
	}

	assert.NoError(t, s.applyAllocationToGPU(gpu, req, gpu.Name))
	assert.True(t, gpu.Status.Available.Tflops.IsZero())
	assert.True(t, gpu.Status.Available.Vram.IsZero())
	assert.Error(t, s.applyAllocationToGPU(gpu, req, gpu.Name), "shared allocation must be exclusive")

	s.releaseAllocationFromGPU(gpu, req, gpu.Name)
	assert.Equal(t, "100", gpu.Status.Available.Tflops.String())
	assert.Equal(t, "24Gi", gpu.Status.Available.Vram.String())
}

func TestDeallocSharedAllocationRestoresWholeGPU(t *testing.T) {
	s := newTestAllocator()
	gpu := sharedTestGPU("gpu-1", "node-1", tfv1.IsolationModeShared)
	gpu.Status.Available.Tflops = qty("0")
	gpu.Status.Available.Vram = qty("0")
	s.gpuStore[types.NamespacedName{Name: gpu.Name}] = gpu

	podMeta := metav1.ObjectMeta{Name: "worker-1", Namespace: "default", UID: "pod-1"}
	req := &tfv1.AllocRequest{
		Isolation:             tfv1.IsolationModeShared,
		Request:               tfv1.Resource{Tflops: qty("10"), Vram: qty("2Gi")},
		PodMeta:               podMeta,
		GPUNames:              []string{gpu.Name},
		WorkloadNameNamespace: tfv1.NameNamespace{Name: "workload-1", Namespace: "default"},
	}
	s.uniqueAllocation[string(podMeta.UID)] = req
	s.nodeWorkerStore["node-1"] = map[types.NamespacedName]struct{}{
		{Name: podMeta.Name, Namespace: podMeta.Namespace}: {},
	}

	s.Dealloc(req.WorkloadNameNamespace, req.GPUNames, podMeta)

	assert.Equal(t, "100", gpu.Status.Available.Tflops.String())
	assert.Equal(t, "24Gi", gpu.Status.Available.Vram.String())
}

func TestRecomputeSharedAllocationKeepsGPUExclusive(t *testing.T) {
	s := newTestAllocator()
	gpu := sharedTestGPU("gpu-1", "node-1", tfv1.IsolationModeHard)
	s.uniqueAllocation["pod-1"] = &tfv1.AllocRequest{
		Isolation: tfv1.IsolationModeShared,
		GPUNames:  []string{gpu.Name},
		Request:   tfv1.Resource{Tflops: qty("10"), Vram: qty("2Gi")},
	}

	s.recomputeGPUAvailableFromAllocations(gpu)
	assert.True(t, gpu.Status.Available.Tflops.IsZero())
	assert.True(t, gpu.Status.Available.Vram.IsZero())
}

func TestSharedSimulationFiltersOnlyWholeIdleGPUsWithoutMutation(t *testing.T) {
	s := newTestAllocator()
	full := sharedTestGPU("gpu-full", "node-1", tfv1.IsolationModeSoft)
	partial := sharedTestGPU("gpu-partial", "node-1", tfv1.IsolationModeHard)
	partial.Status.Available.Tflops = qty("90")

	filtered, details, err := s.Filter(&tfv1.AllocRequest{
		Isolation: tfv1.IsolationModeShared,
		Request:   tfv1.Resource{Tflops: qty("10"), Vram: qty("2Gi")},
	}, []*tfv1.GPU{full, partial}, true)

	assert.NoError(t, err)
	assert.Len(t, filtered, 1)
	assert.Equal(t, full.Name, filtered[0].Name)
	assert.Equal(t, "90", partial.Status.Available.Tflops.String(), "simulation must not mutate input GPUs")
	foundWholeGPUFilter := false
	for _, detail := range details {
		if detail.FilterName == "SharedWholeGPUFilter" {
			foundWholeGPUFilter = true
			break
		}
	}
	assert.True(t, foundWholeGPUFilter)
}

func TestSharedPreemptionSimulationRestoresWholeGPUOnlyForSharedVictim(t *testing.T) {
	makeAllocator := func(gpu *tfv1.GPU) *GpuAllocator {
		s := newTestAllocator()
		s.gpuStore[types.NamespacedName{Name: gpu.Name}] = gpu
		s.nodeGpuStore = map[string]map[string]*tfv1.GPU{
			"node-1": {gpu.Name: gpu},
		}
		return s
	}
	incoming := &tfv1.AllocRequest{
		Isolation: tfv1.IsolationModeShared,
		Request:   tfv1.Resource{Tflops: qty("10"), Vram: qty("2Gi")},
		Count:     1,
	}

	fullyAllocated := sharedTestGPU("gpu-shared-victim", "node-1", tfv1.IsolationModeHard)
	fullyAllocated.Status.Available = &tfv1.Resource{}
	filtered, _, err := makeAllocator(fullyAllocated).FilterWithPreempt(incoming, []*tfv1.AllocRequest{
		{
			Isolation: tfv1.IsolationModeShared,
			GPUNames:  []string{fullyAllocated.Name},
			Request:   tfv1.Resource{Tflops: qty("1"), Vram: qty("1Gi")},
		},
	})
	assert.NoError(t, err)
	assert.Len(t, filtered, 1)
	assert.Equal(t, "100", filtered[0].Status.Available.Tflops.String())
	assert.Equal(t, "24Gi", filtered[0].Status.Available.Vram.String())

	partiallyUsed := sharedTestGPU("gpu-soft-victim", "node-1", tfv1.IsolationModeHard)
	partiallyUsed.Status.Available = &tfv1.Resource{Tflops: qty("80"), Vram: qty("20Gi")}
	filtered, _, err = makeAllocator(partiallyUsed).FilterWithPreempt(incoming, []*tfv1.AllocRequest{
		{
			Isolation: tfv1.IsolationModeHard,
			GPUNames:  []string{partiallyUsed.Name},
			Request:   tfv1.Resource{Tflops: qty("10"), Vram: qty("2Gi")},
		},
	})
	assert.NoError(t, err)
	assert.Empty(t, filtered, "releasing one slice must not make a still-used GPU eligible for shared")
}

func TestDynamicPreemptionOnlyReusesSameModeGPU(t *testing.T) {
	gpu := sharedTestGPU("gpu-soft-victim", "node-1", tfv1.IsolationModeSoft)
	gpu.Status.IsolationPolicy = tfv1.IsolationModePolicyDynamic
	gpu.Status.ActiveIsolationMode = tfv1.IsolationModeSoft
	gpu.Status.Available = &tfv1.Resource{Tflops: qty("20"), Vram: qty("4Gi")}
	s := newTestAllocator()
	s.isolationPolicy = tfv1.IsolationModePolicyDynamic
	s.gpuStore[types.NamespacedName{Name: gpu.Name}] = gpu
	s.nodeGpuStore = map[string]map[string]*tfv1.GPU{"node-1": {gpu.Name: gpu}}
	victim := &tfv1.AllocRequest{
		Isolation: tfv1.IsolationModeSoft,
		GPUNames:  []string{gpu.Name},
		Request:   tfv1.Resource{Tflops: qty("80"), Vram: qty("20Gi")},
	}
	incoming := func(mode tfv1.IsolationModeType) *tfv1.AllocRequest {
		return &tfv1.AllocRequest{
			Isolation: mode,
			Count:     1,
			Request:   tfv1.Resource{Tflops: qty("90"), Vram: qty("22Gi")},
		}
	}

	filtered, _, err := s.FilterWithPreempt(incoming(tfv1.IsolationModeSoft), []*tfv1.AllocRequest{victim})
	assert.NoError(t, err)
	assert.Len(t, filtered, 1, "same-mode preemption should reuse the released GPU")
	assert.Equal(t, tfv1.IsolationModeType(tfv1.IsolationModeSoft), filtered[0].Status.ActiveIsolationMode)

	filtered, _, err = s.FilterWithPreempt(incoming(tfv1.IsolationModeHard), []*tfv1.AllocRequest{victim})
	assert.NoError(t, err)
	assert.Empty(t, filtered, "cross-mode preemption must not change the GPU mode lock")
}

func TestCompactPlacementKeepsExistingGPUAndNodeBinpackWithoutSharedPriority(t *testing.T) {
	usedHard := sharedTestGPU("hard-used", "node-used", tfv1.IsolationModeHard)
	usedHard.Status.Available = &tfv1.Resource{Tflops: qty("50"), Vram: qty("12Gi")}
	idleHard := sharedTestGPU("hard-idle", "node-used", tfv1.IsolationModeHard)
	idleSharedA := sharedTestGPU("shared-idle-a", "node-idle", tfv1.IsolationModeShared)
	idleSharedB := sharedTestGPU("shared-idle-b", "node-idle", tfv1.IsolationModeShared)
	all := []*tfv1.GPU{usedHard, idleHard, idleSharedA, idleSharedB}
	nodeStore := map[string]map[string]*tfv1.GPU{
		"node-used": {usedHard.Name: usedHard, idleHard.Name: idleHard},
		"node-idle": {idleSharedA.Name: idleSharedA, idleSharedB.Name: idleSharedB},
	}
	strategy := CompactFirst{
		cfg:          &config.GPUFitConfig{TflopsWeight: 0.5, VramWeight: 0.5},
		nodeGpuStore: nodeStore,
	}
	s := newTestAllocator()

	hardCandidates, _, err := s.Filter(&tfv1.AllocRequest{
		Isolation: tfv1.IsolationModeHard,
		Request:   tfv1.Resource{Tflops: qty("10"), Vram: qty("2Gi")},
		Count:     1,
	}, all, false)
	assert.NoError(t, err)
	selected, err := strategy.SelectGPUs(hardCandidates, 1)
	assert.NoError(t, err)
	assert.Equal(t, usedHard.Name, selected[0].Name, "hard slices should pack onto the used GPU")

	sharedCandidates, _, err := s.Filter(&tfv1.AllocRequest{
		Isolation: tfv1.IsolationModeShared,
		Request:   tfv1.Resource{Tflops: qty("10"), Vram: qty("2Gi")},
		Count:     1,
	}, all, false)
	assert.NoError(t, err)
	selected, err = strategy.SelectGPUs(sharedCandidates, 1)
	assert.NoError(t, err)
	assert.Equal(t, idleHard.Name, selected[0].Name,
		"shared-labelled nodes must not outrank an already-used hard node")
}
