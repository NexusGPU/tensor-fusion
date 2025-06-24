package gpuallocator

import (
	"context"
	"sync"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/quota"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// Test constants
const (
	TestNamespace = "test-ns"
	TestPoolName  = "test-pool"
	TestWorkload  = "test-workload"
)

// Test helper functions
//
//nolint:unparam // Test helper - namespace parameter is consistent by design
func createTestQuota(namespace string, tflops, vram int64, workers int32) *tfv1.GPUResourceQuota {
	return &tfv1.GPUResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-quota",
			Namespace: namespace,
		},
		Spec: tfv1.GPUResourceQuotaSpec{
			Total: tfv1.GPUResourceQuotaTotal{
				RequestsTFlops: resource.NewQuantity(tflops, resource.DecimalSI),
				RequestsVRAM:   resource.NewQuantity(vram, resource.BinarySI),
				LimitsTFlops:   resource.NewQuantity(tflops, resource.DecimalSI),
				LimitsVRAM:     resource.NewQuantity(vram, resource.BinarySI),
				Workers:        &workers,
			},
		},
	}
}

//nolint:unparam // Test helper - namespace parameter is consistent by design
func createAllocRequest(tflops, vram int64, count uint) AllocRequest {
	return AllocRequest{
		PoolName: TestPoolName,
		WorkloadNameNamespace: tfv1.NameNamespace{
			Namespace: TestNamespace,
			Name:      TestWorkload,
		},
		Request: tfv1.Resource{
			Tflops: *resource.NewQuantity(tflops, resource.DecimalSI),
			Vram:   *resource.NewQuantity(vram, resource.BinarySI),
		},
		Count: count,
	}
}

//nolint:unparam // Test helper - pool parameter is consistent by design
func createAvailableGPU(name, pool string, tflops, vram int64) tfv1.GPU {
	return tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				constants.LabelKeyOwner: "node1",
				constants.GpuPoolKey:    pool,
			},
		},
		Status: tfv1.GPUStatus{
			Phase: tfv1.TensorFusionGPUPhaseRunning,
			Capacity: &tfv1.Resource{
				Tflops: *resource.NewQuantity(tflops, resource.DecimalSI),
				Vram:   *resource.NewQuantity(vram, resource.BinarySI),
			},
			Available: &tfv1.Resource{
				Tflops: *resource.NewQuantity(tflops, resource.DecimalSI),
				Vram:   *resource.NewQuantity(vram, resource.BinarySI),
			},
			RunningApps: []*tfv1.RunningAppDetail{},
		},
	}
}

//nolint:unparam // Test helper - name parameter is consistent by design
func createTestGPUPool(name string) *tfv1.GPUPool {
	return &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: tfv1.GPUPoolSpec{},
		Status: tfv1.GPUPoolStatus{
			Phase: tfv1.TensorFusionPoolPhaseRunning,
		},
	}
}

func objectsFromGPUs(gpus []tfv1.GPU) []client.Object {
	objects := make([]client.Object, len(gpus))
	for i := range gpus {
		objects[i] = &gpus[i]
	}
	return objects
}

func createWorkerPod(namespace, name, tflops, vram string) v1.Pod {
	return v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				constants.LabelComponent: constants.ComponentWorker,
				constants.WorkloadKey:    TestWorkload,
			},
			Annotations: map[string]string{
				constants.TFLOPSRequestAnnotation: tflops,
				constants.VRAMRequestAnnotation:   vram,
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}
}

// Shared test fixtures to reduce duplication
type QuotaTestFixture struct {
	quotaStore *QuotaStore
	quota      *tfv1.GPUResourceQuota
	entry      *QuotaStoreEntry
}

func setupQuotaTest(tflops, vram int64, workers int32) *QuotaTestFixture {
	qs := NewQuotaStore(nil)
	quotaObj := createTestQuota(TestNamespace, tflops, vram, workers)

	calc := quota.NewCalculator()
	entry := &QuotaStoreEntry{
		quota:        quotaObj,
		currentUsage: calc.CreateZeroUsage(),
		available:    calc.CopyUsageToTotal(quotaObj),
	}
	qs.quotaStore[TestNamespace] = entry

	return &QuotaTestFixture{
		quotaStore: qs,
		quota:      quotaObj,
		entry:      entry,
	}
}

// Helper functions for backward compatibility
func createZeroUsage() *tfv1.GPUResourceUsage {
	calc := quota.NewCalculator()
	return calc.CreateZeroUsage()
}

func createUsageFromQuota(quotaObj *tfv1.GPUResourceQuota) *tfv1.GPUResourceUsage {
	calc := quota.NewCalculator()
	return calc.CopyUsageToTotal(quotaObj)
}

// Common assertion helper
type QuotaExpectation struct {
	TFlops  int64
	VRAM    int64
	Workers int32
}

func assertQuotaState(t *testing.T, usage *tfv1.GPUResourceUsage, expected QuotaExpectation, desc string) {
	assert.Equal(t, expected.TFlops, usage.RequestsTFlops.Value(), "%s - TFlops mismatch", desc)
	assert.Equal(t, expected.VRAM, usage.RequestsVRAM.Value(), "%s - VRAM mismatch", desc)
	assert.Equal(t, expected.Workers, *usage.Workers, "%s - Workers mismatch", desc)
}

// Core QuotaStore Tests - Table-driven approach
func TestQuotaStore_BasicOperations(t *testing.T) {
	tests := []struct {
		name        string
		quotaConfig QuotaExpectation
		request     struct {
			tflops, vram int64
			count        uint
		}
		expectError bool
		finalUsage  QuotaExpectation
		finalAvail  QuotaExpectation
		testFunc    func(*testing.T, *QuotaTestFixture)
	}{
		{
			name:        "allocation within quota",
			quotaConfig: QuotaExpectation{TFlops: 100, VRAM: 1000, Workers: 10},
			request: struct {
				tflops, vram int64
				count        uint
			}{30, 300, 2},
			expectError: false,
			finalUsage:  QuotaExpectation{TFlops: 60, VRAM: 600, Workers: 2},
			finalAvail:  QuotaExpectation{TFlops: 40, VRAM: 400, Workers: 8},
			testFunc: func(t *testing.T, fixture *QuotaTestFixture) {
				req := createAllocRequest(30, 300, 2)
				err := fixture.quotaStore.checkQuotaAvailable(TestNamespace, req)
				require.NoError(t, err)
				fixture.quotaStore.AllocateQuota(TestNamespace, req)
			},
		},
		{
			name:        "allocation exceeds quota",
			quotaConfig: QuotaExpectation{TFlops: 100, VRAM: 1000, Workers: 10},
			request: struct {
				tflops, vram int64
				count        uint
			}{60, 600, 2}, // 60*2=120 > 100
			expectError: true,
			testFunc: func(t *testing.T, fixture *QuotaTestFixture) {
				req := createAllocRequest(60, 600, 2)
				err := fixture.quotaStore.checkQuotaAvailable(TestNamespace, req)
				require.Error(t, err)
				assert.Contains(t, err.Error(), "total.requests.tflops")
			},
		},
		{
			name:        "allocation and deallocation cycle",
			quotaConfig: QuotaExpectation{TFlops: 100, VRAM: 1000, Workers: 10},
			request: struct {
				tflops, vram int64
				count        uint
			}{30, 300, 2},
			expectError: false,
			finalUsage:  QuotaExpectation{TFlops: 0, VRAM: 0, Workers: 0},
			finalAvail:  QuotaExpectation{TFlops: 100, VRAM: 1000, Workers: 10},
			testFunc: func(t *testing.T, fixture *QuotaTestFixture) {
				req := createAllocRequest(30, 300, 2)
				fixture.quotaStore.AllocateQuota(TestNamespace, req)
				fixture.quotaStore.DeallocateQuota(TestNamespace, req.Request, int32(req.Count))
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fixture := setupQuotaTest(tt.quotaConfig.TFlops, tt.quotaConfig.VRAM, tt.quotaConfig.Workers)

			tt.testFunc(t, fixture)

			if !tt.expectError {
				usage, available, exists := fixture.quotaStore.GetQuotaStatus(TestNamespace)
				require.True(t, exists)
				assertQuotaState(t, usage, tt.finalUsage, "final usage")
				assertQuotaState(t, available, tt.finalAvail, "final available")
			}
		})
	}
}

func TestQuotaStore_ConcurrentOperations(t *testing.T) {
	fixture := setupQuotaTest(100, 1000, 20)

	var wg sync.WaitGroup
	errors := make(chan error, 20)
	successes := make(chan bool, 20)

	// Launch 10 goroutines trying to allocate concurrently
	for range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := createAllocRequest(8, 80, 1)

			fixture.quotaStore.storeMutex.Lock()
			err := fixture.quotaStore.checkQuotaAvailable(TestNamespace, req)
			if err == nil {
				fixture.quotaStore.AllocateQuota(TestNamespace, req)
				successes <- true
			} else {
				errors <- err
			}
			fixture.quotaStore.storeMutex.Unlock()
		}()
	}

	wg.Wait()
	close(errors)
	close(successes)

	successCount := len(successes)
	errorCount := len(errors)

	assert.LessOrEqual(t, successCount, 12, "Should not exceed quota capacity")
	assert.Equal(t, 10, successCount+errorCount, "All requests should be processed")
}

func TestQuotaStore_BoundaryConditions(t *testing.T) {
	tests := []struct {
		name           string
		quotaTFlops    int64
		quotaVRAM      int64
		quotaWorkers   int32
		requestTFlops  int64
		requestVRAM    int64
		requestWorkers uint
		expectError    bool
	}{
		{
			name:           "exact quota boundary",
			quotaTFlops:    100,
			quotaVRAM:      1000,
			quotaWorkers:   10,
			requestTFlops:  10,
			requestVRAM:    100,
			requestWorkers: 10,
			expectError:    false,
		},
		{
			name:           "exceed quota by one worker",
			quotaTFlops:    100,
			quotaVRAM:      1000,
			quotaWorkers:   10,
			requestTFlops:  9,
			requestVRAM:    90,
			requestWorkers: 11,
			expectError:    true,
		},
		{
			name:           "zero quota allocation",
			quotaTFlops:    0,
			quotaVRAM:      0,
			quotaWorkers:   0,
			requestTFlops:  1,
			requestVRAM:    1,
			requestWorkers: 1,
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			qs := NewQuotaStore(nil)
			quota := createTestQuota(TestNamespace, tt.quotaTFlops, tt.quotaVRAM, tt.quotaWorkers)

			entry := &QuotaStoreEntry{
				quota:        quota,
				currentUsage: createZeroUsage(),
				available:    createUsageFromQuota(quota),
			}
			qs.quotaStore[TestNamespace] = entry

			req := createAllocRequest(tt.requestTFlops, tt.requestVRAM, tt.requestWorkers)
			err := qs.checkQuotaAvailable(TestNamespace, req)

			if tt.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// Integration Tests with GpuAllocator
func TestGPUAllocator_QuotaIntegration(t *testing.T) {
	t.Run("successful allocation within quota", func(t *testing.T) {
		scheme := runtime.NewScheme()
		require.NoError(t, tfv1.AddToScheme(scheme))

		quota := createTestQuota(TestNamespace, 100, 1000, 10)
		gpus := []tfv1.GPU{
			createAvailableGPU("gpu1", TestPoolName, 50, 500),
			createAvailableGPU("gpu2", TestPoolName, 50, 500),
		}

		testPool := createTestGPUPool(TestPoolName)
		allObjects := objectsFromGPUs(gpus)
		allObjects = append(allObjects, testPool)

		client := fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(allObjects...).
			WithLists(&tfv1.GPUResourceQuotaList{Items: []tfv1.GPUResourceQuota{*quota}}).
			Build()

		ctx := context.Background()
		allocator := NewGpuAllocator(ctx, client, 0)

		err := allocator.initGPUAndQuotaStore(ctx)
		require.NoError(t, err)

		req := createAllocRequest(30, 300, 2)
		allocatedGPUs, err := allocator.Alloc(ctx, req)
		require.NoError(t, err)
		require.Len(t, allocatedGPUs, 2)

		usage, available, exists := allocator.quotaStore.GetQuotaStatus(TestNamespace)
		require.True(t, exists)
		assert.Equal(t, int64(60), usage.RequestsTFlops.Value())
		assert.Equal(t, int64(600), usage.RequestsVRAM.Value())
		assert.Equal(t, int32(2), *usage.Workers)
		assert.Equal(t, int64(40), available.RequestsTFlops.Value())
		assert.Equal(t, int64(400), available.RequestsVRAM.Value())
		assert.Equal(t, int32(8), *available.Workers)
	})

	t.Run("allocation exceeds quota", func(t *testing.T) {
		scheme := runtime.NewScheme()
		require.NoError(t, tfv1.AddToScheme(scheme))

		quota := createTestQuota(TestNamespace, 100, 1000, 10)
		gpus := []tfv1.GPU{
			createAvailableGPU("gpu1", TestPoolName, 100, 1000),
			createAvailableGPU("gpu2", TestPoolName, 100, 1000),
		}

		testPool := createTestGPUPool(TestPoolName)
		allObjects := objectsFromGPUs(gpus)
		allObjects = append(allObjects, testPool)

		client := fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(allObjects...).
			WithLists(&tfv1.GPUResourceQuotaList{Items: []tfv1.GPUResourceQuota{*quota}}).
			Build()

		ctx := context.Background()
		allocator := NewGpuAllocator(ctx, client, 0)

		err := allocator.initGPUAndQuotaStore(ctx)
		require.NoError(t, err)

		req := createAllocRequest(60, 600, 2) // 60*2=120 > 100 quota
		_, err = allocator.Alloc(ctx, req)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "total.requests.tflops")
	})
}

func TestGPUAllocator_ConcurrentQuotaEnforcement(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))

	quota := createTestQuota(TestNamespace, 50, 500, 5)
	gpus := []tfv1.GPU{
		createAvailableGPU("gpu1", TestPoolName, 20, 200),
		createAvailableGPU("gpu2", TestPoolName, 20, 200),
		createAvailableGPU("gpu3", TestPoolName, 20, 200),
		createAvailableGPU("gpu4", TestPoolName, 20, 200),
	}

	testPool := createTestGPUPool(TestPoolName)
	allObjects := objectsFromGPUs(gpus)
	allObjects = append(allObjects, testPool)

	client := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(allObjects...).
		WithLists(&tfv1.GPUResourceQuotaList{Items: []tfv1.GPUResourceQuota{*quota}}).
		Build()

	ctx := context.Background()
	allocator := NewGpuAllocator(ctx, client, 0)

	err := allocator.initGPUAndQuotaStore(ctx)
	require.NoError(t, err)

	var wg sync.WaitGroup
	results := make(chan error, 10)

	// Launch 6 concurrent allocation attempts, but quota only allows 5 workers
	for range 6 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			req := createAllocRequest(10, 100, 1) // Each request uses 10 TFlops
			_, err := allocator.Alloc(ctx, req)
			results <- err
		}()
	}

	wg.Wait()
	close(results)

	// Count successes and failures
	successes := 0
	failures := 0
	for err := range results {
		if err == nil {
			successes++
		} else {
			failures++
		}
	}

	// Should allow exactly 5 allocations (50 TFlops / 10 TFlops each)
	assert.Equal(t, 5, successes, "Should allow exactly 5 allocations")
	assert.Equal(t, 1, failures, "Should reject 1 allocation due to quota")

	// Verify final quota state
	usage, available, exists := allocator.quotaStore.GetQuotaStatus(TestNamespace)
	require.True(t, exists)
	assert.Equal(t, int64(50), usage.RequestsTFlops.Value(), "Should use full quota")
	assert.Equal(t, int64(0), available.RequestsTFlops.Value(), "No quota should remain")
	assert.Equal(t, int32(5), *usage.Workers, "Should have 5 workers allocated")
}

func TestGPUAllocator_QuotaReconciliation(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))
	require.NoError(t, v1.AddToScheme(scheme))

	quota := createTestQuota(TestNamespace, 100, 1000, 10)
	gpus := []tfv1.GPU{
		createAvailableGPU("gpu1", TestPoolName, 50, 500),
		createAvailableGPU("gpu2", TestPoolName, 50, 500),
	}

	// Create worker pods that should be counted in quota usage
	workerPods := []v1.Pod{
		createWorkerPod(TestNamespace, "worker1", "20", "200"),
		createWorkerPod(TestNamespace, "worker2", "30", "300"),
	}

	testPool := createTestGPUPool(TestPoolName)
	allObjects := objectsFromGPUs(gpus)
	allObjects = append(allObjects, testPool)

	client := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(allObjects...).
		WithLists(
			&tfv1.GPUResourceQuotaList{Items: []tfv1.GPUResourceQuota{*quota}},
			&v1.PodList{Items: workerPods},
		).
		Build()

	ctx := context.Background()
	allocator := NewGpuAllocator(ctx, client, 0)

	err := allocator.initGPUAndQuotaStore(ctx)
	require.NoError(t, err)

	// Manually trigger reconciliation
	allocator.reconcileAllocationState(ctx)

	// Verify quota usage reflects the worker pods
	usage, available, exists := allocator.quotaStore.GetQuotaStatus(TestNamespace)
	require.True(t, exists)

	assert.Equal(t, int64(50), usage.RequestsTFlops.Value())     // 20+30
	assert.Equal(t, int64(500), usage.RequestsVRAM.Value())      // 200+300
	assert.Equal(t, int32(2), *usage.Workers)                    // 2 pods
	assert.Equal(t, int64(50), available.RequestsTFlops.Value()) // 100-50
	assert.Equal(t, int64(500), available.RequestsVRAM.Value())  // 1000-500
	assert.Equal(t, int32(8), *available.Workers)                // 10-2
}

func TestGPUAllocator_QuotaDeallocation(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))

	quota := createTestQuota(TestNamespace, 100, 1000, 10)
	gpus := []tfv1.GPU{
		createAvailableGPU("gpu1", TestPoolName, 50, 500),
		createAvailableGPU("gpu2", TestPoolName, 50, 500),
	}

	testPool := createTestGPUPool(TestPoolName)
	allObjects := objectsFromGPUs(gpus)
	allObjects = append(allObjects, testPool)

	client := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(allObjects...).
		WithLists(&tfv1.GPUResourceQuotaList{Items: []tfv1.GPUResourceQuota{*quota}}).
		Build()

	ctx := context.Background()
	allocator := NewGpuAllocator(ctx, client, 0)

	err := allocator.initGPUAndQuotaStore(ctx)
	require.NoError(t, err)

	// Allocate GPUs
	req := createAllocRequest(30, 300, 2)
	allocatedGPUs, err := allocator.Alloc(ctx, req)
	require.NoError(t, err)
	require.Len(t, allocatedGPUs, 2)

	// Verify allocation
	usage, available, exists := allocator.quotaStore.GetQuotaStatus(TestNamespace)
	require.True(t, exists)
	assert.Equal(t, int64(60), usage.RequestsTFlops.Value())
	assert.Equal(t, int64(40), available.RequestsTFlops.Value())

	// Deallocate GPUs
	gpuNames := make([]types.NamespacedName, len(allocatedGPUs))
	for i, gpu := range allocatedGPUs {
		gpuNames[i] = types.NamespacedName{Name: gpu.Name}
	}

	allocator.Dealloc(ctx, req.WorkloadNameNamespace, req.Request, gpuNames)

	// Verify deallocation
	usage, available, exists = allocator.quotaStore.GetQuotaStatus(TestNamespace)
	require.True(t, exists)
	assert.Equal(t, int64(0), usage.RequestsTFlops.Value())
	assert.Equal(t, int64(100), available.RequestsTFlops.Value())
	assert.Equal(t, int32(0), *usage.Workers)
	assert.Equal(t, int32(10), *available.Workers)
}

// Calculator-specific Tests
func TestQuotaCalculator_ResourceOperations(t *testing.T) {
	calc := quota.NewCalculator()

	t.Run("resource operation creation", func(t *testing.T) {
		baseResource := tfv1.Resource{
			Tflops: *resource.NewQuantity(10, resource.DecimalSI),
			Vram:   *resource.NewQuantity(100, resource.BinarySI),
		}

		op := calc.NewResourceOperation(baseResource, 3)
		assert.Equal(t, int64(30), op.TFlops.Value())
		assert.Equal(t, int64(300), op.VRAM.Value())
		assert.Equal(t, int32(3), op.Workers)
	})

	t.Run("usage percentage calculation", func(t *testing.T) {
		quota := createTestQuota(TestNamespace, 100, 1000, 10)
		usage := createZeroUsage()
		usage.RequestsTFlops.Set(25)
		usage.RequestsVRAM.Set(400)
		*usage.Workers = 3

		percentages := calc.CalculateUsagePercent(quota, usage)
		assert.Equal(t, int64(25), percentages["requests.tflops"])
		assert.Equal(t, int64(40), percentages["requests.vram"])
		assert.Equal(t, int64(30), percentages["workers"])
	})

	t.Run("available percent calculation", func(t *testing.T) {
		quota := createTestQuota(TestNamespace, 100, 1000, 10)
		usage := createZeroUsage()
		usage.RequestsTFlops.Set(25)
		usage.RequestsVRAM.Set(400)
		*usage.Workers = 3

		percent := calc.CalculateAvailablePercent(quota, usage)
		assert.Equal(t, int64(75), *percent.RequestsTFlops)
		assert.Equal(t, int64(60), *percent.RequestsVRAM)
		assert.Equal(t, int64(70), *percent.Workers)
	})

	t.Run("quota exceeded detection", func(t *testing.T) {
		quota := createTestQuota(TestNamespace, 100, 1000, 10)

		// Within limits
		usage := createZeroUsage()
		usage.RequestsTFlops.Set(50)
		assert.False(t, calc.IsQuotaExceeded(quota, usage))

		// Exceeds limits
		usage.RequestsTFlops.Set(150)
		assert.True(t, calc.IsQuotaExceeded(quota, usage))
	})

	t.Run("deallocation clamping", func(t *testing.T) {
		usage := createZeroUsage()
		usage.RequestsTFlops.Set(30)
		usage.RequestsVRAM.Set(300)
		*usage.Workers = 2

		// Try to deallocate more than available
		operation := &quota.ResourceOperation{
			TFlops:  *resource.NewQuantity(50, resource.DecimalSI),
			VRAM:    *resource.NewQuantity(500, resource.BinarySI),
			Workers: 5,
		}

		actual := calc.CalculateActualDeallocation(usage, operation)
		assert.Equal(t, int64(30), actual.TFlops.Value())
		assert.Equal(t, int64(300), actual.VRAM.Value())
		assert.Equal(t, int32(2), actual.Workers)
	})
}

func TestQuotaCalculator_EdgeCases(t *testing.T) {
	calc := quota.NewCalculator()

	t.Run("zero division safety", func(t *testing.T) {
		quota := createTestQuota(TestNamespace, 0, 0, 0)
		usage := createZeroUsage()

		// Should not panic on zero totals
		percentages := calc.CalculateUsagePercent(quota, usage)
		assert.Empty(t, percentages)

		percent := calc.CalculateAvailablePercent(quota, usage)
		assert.Nil(t, percent.RequestsTFlops)
		assert.Nil(t, percent.RequestsVRAM)
		assert.Nil(t, percent.Workers)
	})

	t.Run("negative usage protection", func(t *testing.T) {
		usage := createZeroUsage()
		usage.RequestsTFlops.Set(10)

		// Try to subtract more than available
		toSubtract := *resource.NewQuantity(20, resource.DecimalSI)
		calc.SafeSub(usage.RequestsTFlops, toSubtract)

		// Should clamp to zero
		assert.Equal(t, int64(0), usage.RequestsTFlops.Value())
	})

	t.Run("nil quantity handling", func(t *testing.T) {
		// Should not panic with nil quantities
		var nilQty *resource.Quantity
		testQty := *resource.NewQuantity(10, resource.DecimalSI)

		calc.SafeAdd(nilQty, testQty) // Should not panic
		calc.SafeSub(nilQty, testQty) // Should not panic
	})
}

func TestQuotaStore_ValidationRules(t *testing.T) {
	qs := NewQuotaStore(nil)

	t.Run("valid quota configuration", func(t *testing.T) {
		quota := createTestQuota(TestNamespace, 100, 1000, 10)
		err := qs.validateQuotaConfig(quota)
		assert.NoError(t, err)
	})

	t.Run("negative values validation", func(t *testing.T) {
		quota := createTestQuota(TestNamespace, -10, 1000, 10)
		err := qs.validateQuotaConfig(quota)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "requests.tflops cannot be negative")
	})

	t.Run("limits less than requests validation", func(t *testing.T) {
		quota := createTestQuota(TestNamespace, 100, 1000, 10)
		// Set limits lower than requests
		quota.Spec.Total.LimitsTFlops = resource.NewQuantity(50, resource.DecimalSI)

		err := qs.validateQuotaConfig(quota)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "limits.tflops cannot be less than requests.tflops")
	})

	t.Run("invalid alert threshold validation", func(t *testing.T) {
		quota := createTestQuota(TestNamespace, 100, 1000, 10)
		invalidThreshold := int32(150)
		quota.Spec.Total.AlertThresholdPercent = &invalidThreshold

		err := qs.validateQuotaConfig(quota)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "alertThresholdPercent must be between 0 and 100")
	})
}
