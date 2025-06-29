package gpuallocator

import (
	"context"
	"fmt"
	"sync"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	quota "github.com/NexusGPU/tensor-fusion/internal/quota"
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
	TestQuotaName = "test-quota"
)

var podMeta = metav1.ObjectMeta{Namespace: TestNamespace, Name: TestWorkload, UID: "test-pod"}

func createAllocRequest(tflops, vram int64, count uint) *tfv1.AllocRequest {
	return &tfv1.AllocRequest{
		PoolName: TestPoolName,
		WorkloadNameNamespace: tfv1.NameNamespace{
			Namespace: TestNamespace,
			Name:      TestWorkload,
		},
		Request: tfv1.Resource{
			Tflops: *resource.NewQuantity(tflops, resource.DecimalSI),
			Vram:   *resource.NewQuantity(vram, resource.BinarySI),
		},
		Limit: tfv1.Resource{
			Tflops: *resource.NewQuantity(tflops, resource.DecimalSI),
			Vram:   *resource.NewQuantity(vram, resource.BinarySI),
		},
		Count: count,
	}
}

func createAvailableGPU(name string, tflops, vram int64) tfv1.GPU {
	return tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				constants.LabelKeyOwner: "node1",
				constants.GpuPoolKey:    TestPoolName,
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

func createTestGPUPool() *tfv1.GPUPool {
	return &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{
			Name: TestPoolName,
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
			UID:       types.UID(fmt.Sprintf("%s-uid", name)),
			Labels: map[string]string{
				constants.LabelComponent: constants.ComponentWorker,
				constants.WorkloadKey:    TestWorkload,
			},
			Annotations: map[string]string{
				constants.TFLOPSRequestAnnotation: tflops,
				constants.VRAMRequestAnnotation:   vram,
				constants.TFLOPSLimitAnnotation:   tflops,
				constants.VRAMLimitAnnotation:     vram,
				constants.GpuCountAnnotation:      "1",
				constants.GpuPoolKey:              TestPoolName,
			},
		},
		Spec: v1.PodSpec{
			NodeName: "test-node", // Pods need to be scheduled
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}
}

// Shared test fixtures to reduce duplication
type QuotaTestFixture struct {
	quotaStore *quota.QuotaStore
	quota      *tfv1.GPUResourceQuota
	entry      *quota.QuotaStoreEntry
}

func initAllocator(t *testing.T, allocator *GpuAllocator) {
	err := allocator.InitGPUAndQuotaStore()
	require.NoError(t, err)
	allocator.ReconcileAllocationState()
}

func setupQuotaTest(tflops, vram int64, workers int32) *QuotaTestFixture {
	qs := quota.NewQuotaStore(nil, context.Background())
	quotaObj := createTestQuota(tflops, vram, workers)

	calc := quota.NewCalculator()
	entry := &quota.QuotaStoreEntry{
		Quota:        quotaObj,
		CurrentUsage: calc.CreateZeroUsage(),
	}
	qs.QuotaStore[TestNamespace] = entry

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

// Common assertion helper
type QuotaExpectation struct {
	TFlops  int64
	VRAM    int64
	Workers int32
}

func assertQuotaState(t *testing.T, usage *tfv1.GPUResourceUsage, expected QuotaExpectation, desc string) {
	assert.Equal(t, expected.TFlops, usage.Requests.Tflops.Value(), "%s - TFlops mismatch", desc)
	assert.Equal(t, expected.VRAM, usage.Requests.Vram.Value(), "%s - VRAM mismatch", desc)
	assert.Equal(t, expected.Workers, usage.Workers, "%s - Workers mismatch", desc)
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
			finalUsage:  QuotaExpectation{TFlops: 60, VRAM: 600, Workers: 1},
			finalAvail:  QuotaExpectation{TFlops: 40, VRAM: 400, Workers: 9},
			testFunc: func(t *testing.T, fixture *QuotaTestFixture) {
				req := createAllocRequest(30, 300, 2)
				err := fixture.quotaStore.CheckQuotaAvailable(TestNamespace, req)
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
				err := fixture.quotaStore.CheckQuotaAvailable(TestNamespace, req)
				require.Error(t, err)
				assert.Contains(t, err.Error(), "total.max.tflops.request")
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
				fixture.quotaStore.DeallocateQuota(TestNamespace, req)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fixture := setupQuotaTest(tt.quotaConfig.TFlops, tt.quotaConfig.VRAM, tt.quotaConfig.Workers)

			tt.testFunc(t, fixture)

			if !tt.expectError {
				usage, exists := fixture.quotaStore.GetQuotaStatus(TestNamespace)
				require.True(t, exists)
				assertQuotaState(t, usage, tt.finalUsage, "final usage")
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

			// TODO
			// fixture.quotaStore.storeMutex.Lock()
			// defer fixture.quotaStore.storeMutex.Unlock()
			err := fixture.quotaStore.CheckQuotaAvailable(TestNamespace, req)
			if err == nil {
				fixture.quotaStore.AllocateQuota(TestNamespace, req)
				successes <- true
			} else {
				errors <- err
			}

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
		name          string
		quotaTFlops   int64
		quotaVRAM     int64
		quotaWorkers  int32
		requestTFlops int64
		requestVRAM   int64
		requestGPUs   uint
		expectError   bool
	}{
		{
			name:          "exact quota boundary",
			quotaTFlops:   100,
			quotaVRAM:     1000,
			quotaWorkers:  10,
			requestTFlops: 10,
			requestVRAM:   100,
			requestGPUs:   10,
			expectError:   false,
		},
		{
			name:          "zero quota allocation",
			quotaTFlops:   0,
			quotaVRAM:     0,
			quotaWorkers:  0,
			requestTFlops: 1,
			requestVRAM:   1,
			requestGPUs:   1,
			expectError:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			qs := quota.NewQuotaStore(nil, context.Background())
			quotaObj := createTestQuota(tt.quotaTFlops, tt.quotaVRAM, tt.quotaWorkers)

			entry := &quota.QuotaStoreEntry{
				Quota:        quotaObj,
				CurrentUsage: createZeroUsage(),
			}
			qs.QuotaStore[TestNamespace] = entry

			req := createAllocRequest(tt.requestTFlops, tt.requestVRAM, tt.requestGPUs)
			err := qs.CheckQuotaAvailable(TestNamespace, req)

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

		quota := createTestQuota(100, 1000, 10)
		gpus := []tfv1.GPU{
			createAvailableGPU("gpu1", 50, 500),
			createAvailableGPU("gpu2", 50, 500),
		}

		testPool := createTestGPUPool()
		allObjects := objectsFromGPUs(gpus)
		allObjects = append(allObjects, testPool)

		client := fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(allObjects...).
			WithLists(&tfv1.GPUResourceQuotaList{Items: []tfv1.GPUResourceQuota{*quota}}).
			Build()

		ctx := context.Background()
		allocator := NewGpuAllocator(ctx, client, 0)

		initAllocator(t, allocator)

		req := createAllocRequest(30, 300, 2)
		req.PodMeta = podMeta
		allocatedGPUs, err := allocator.Alloc(req)
		require.NoError(t, err)
		require.Len(t, allocatedGPUs, 2)

		usage, exists := allocator.quotaStore.GetQuotaStatus(TestNamespace)
		require.True(t, exists)
		assert.Equal(t, int64(60), usage.Requests.Tflops.Value())
		assert.Equal(t, int64(600), usage.Requests.Vram.Value())
		assert.Equal(t, int32(1), usage.Workers)
	})

	t.Run("allocation exceeds quota", func(t *testing.T) {
		scheme := runtime.NewScheme()
		require.NoError(t, tfv1.AddToScheme(scheme))

		quota := createTestQuota(100, 1000, 10)
		gpus := []tfv1.GPU{
			createAvailableGPU("gpu1", 100, 1000),
			createAvailableGPU("gpu2", 100, 1000),
		}

		testPool := createTestGPUPool()
		allObjects := objectsFromGPUs(gpus)
		allObjects = append(allObjects, testPool)

		client := fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(allObjects...).
			WithLists(&tfv1.GPUResourceQuotaList{Items: []tfv1.GPUResourceQuota{*quota}}).
			Build()

		ctx := context.Background()
		allocator := NewGpuAllocator(ctx, client, 0)

		initAllocator(t, allocator)

		req := createAllocRequest(60, 600, 2) // 60*2=120 > 100 quota
		req.PodMeta = podMeta
		_, err := allocator.Alloc(req)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "total.max.tflops")
	})
}

func TestGPUAllocator_ConcurrentQuotaEnforcement(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))

	quota := createTestQuota(50, 500, 5)
	gpus := []tfv1.GPU{
		createAvailableGPU("gpu1", 20, 200),
		createAvailableGPU("gpu2", 20, 200),
		createAvailableGPU("gpu3", 20, 200),
		createAvailableGPU("gpu4", 20, 200),
	}

	testPool := createTestGPUPool()
	allObjects := objectsFromGPUs(gpus)
	allObjects = append(allObjects, testPool)

	client := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(allObjects...).
		WithLists(&tfv1.GPUResourceQuotaList{Items: []tfv1.GPUResourceQuota{*quota}}).
		Build()

	ctx := context.Background()
	allocator := NewGpuAllocator(ctx, client, 0)

	initAllocator(t, allocator)

	var wg sync.WaitGroup
	results := make(chan error, 10)

	// Launch 6 concurrent allocation attempts, but quota only allows 5 workers
	for i := range 6 {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			req := createAllocRequest(10, 100, 1) // Each request uses 10 TFlops
			// Create unique pod metadata for each goroutine
			req.PodMeta = metav1.ObjectMeta{
				Name:      fmt.Sprintf("test-pod-%d", index),
				Namespace: TestNamespace,
				UID:       types.UID(fmt.Sprintf("test-uid-%d", index)),
			}
			_, err := allocator.Alloc(req)
			results <- err
		}(i)
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
	usage, exists := allocator.quotaStore.GetQuotaStatus(TestNamespace)
	require.True(t, exists)
	assert.Equal(t, int64(50), usage.Requests.Tflops.Value(), "Should use full quota")
	assert.Equal(t, int32(5), usage.Workers, "Should have 5 workers allocated")
}

func TestGPUAllocator_QuotaReconciliation(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))
	require.NoError(t, v1.AddToScheme(scheme))

	quota := createTestQuota(100, 1000, 10)
	gpus := []tfv1.GPU{
		createAvailableGPU("gpu1", 50, 500),
		createAvailableGPU("gpu2", 50, 500),
	}

	// Create worker pods that should be counted in quota usage
	workerPods := []v1.Pod{
		createWorkerPod(TestNamespace, "worker1", "20", "200"),
		createWorkerPod(TestNamespace, "worker2", "30", "300"),
	}

	testPool := createTestGPUPool()
	allObjects := objectsFromGPUs(gpus)
	allObjects = append(allObjects, testPool)
	allObjects = append(allObjects, &tfv1.TensorFusionWorkload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      TestWorkload,
			Namespace: TestNamespace,
		},
	})

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

	initAllocator(t, allocator)

	// Verify quota usage reflects the worker pods
	usage, exists := allocator.quotaStore.GetQuotaStatus(TestNamespace)
	require.True(t, exists)

	assert.Equal(t, int64(50), usage.Requests.Tflops.Value()) // 20+30
	assert.Equal(t, int64(500), usage.Requests.Vram.Value())  // 200+300
	assert.Equal(t, int32(2), usage.Workers)                  // 2 pods
}

func TestGPUAllocator_QuotaDeallocation(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))

	quota := createTestQuota(100, 1000, 10)
	gpus := []tfv1.GPU{
		createAvailableGPU("gpu1", 50, 500),
		createAvailableGPU("gpu2", 50, 500),
	}

	testPool := createTestGPUPool()
	allObjects := objectsFromGPUs(gpus)
	allObjects = append(allObjects, testPool)

	client := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(allObjects...).
		WithLists(&tfv1.GPUResourceQuotaList{Items: []tfv1.GPUResourceQuota{*quota}}).
		Build()

	ctx := context.Background()
	allocator := NewGpuAllocator(ctx, client, 0)

	initAllocator(t, allocator)

	// Allocate GPUs
	req := createAllocRequest(30, 300, 2)
	req.PodMeta = podMeta
	allocatedGPUs, err := allocator.Alloc(req)
	require.NoError(t, err)
	require.Len(t, allocatedGPUs, 2)

	// Verify allocation
	usage, exists := allocator.quotaStore.GetQuotaStatus(TestNamespace)
	require.True(t, exists)
	assert.Equal(t, int64(60), usage.Requests.Tflops.Value())
	assert.Equal(t, int64(600), usage.Requests.Vram.Value()) // 300 * 2 = 600
	assert.Equal(t, int32(1), usage.Workers)

	// Deallocate GPUs
	gpuNames := make([]string, len(allocatedGPUs))
	for i, gpu := range allocatedGPUs {
		gpuNames[i] = gpu.Name
	}

	allocator.Dealloc(req.WorkloadNameNamespace, gpuNames, podMeta)

	// Verify deallocation
	usage, exists = allocator.quotaStore.GetQuotaStatus(TestNamespace)
	require.True(t, exists)
	assert.Equal(t, int64(0), usage.Requests.Tflops.Value())
	assert.Equal(t, int32(0), usage.Workers)
}

func createTestQuota(tflops, vram int64, workers int32) *tfv1.GPUResourceQuota {
	return &tfv1.GPUResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      TestQuotaName,
			Namespace: TestNamespace,
		},
		Spec: tfv1.GPUResourceQuotaSpec{
			Total: tfv1.GPUResourceQuotaTotal{
				Requests: &tfv1.Resource{
					Tflops: *resource.NewQuantity(tflops, resource.DecimalSI),
					Vram:   *resource.NewQuantity(vram, resource.BinarySI),
				},
				Limits: &tfv1.Resource{
					Tflops: *resource.NewQuantity(tflops, resource.DecimalSI),
					Vram:   *resource.NewQuantity(vram, resource.BinarySI),
				},
				MaxWorkers: &workers,
			},
		},
	}
}
