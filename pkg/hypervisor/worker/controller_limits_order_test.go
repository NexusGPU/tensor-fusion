package worker

import (
	"sync"
	"sync/atomic"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/stretchr/testify/require"
)

// A limit in the mapping that does not match what the tracked worker and the
// device controller say (a change whose event was not delivered, a value left
// behind by anything else) is put right by the next pass of the sync loop.
func TestSyncPassRestoresLimitsThatDoNotMatchTheCurrentState(t *testing.T) {
	estimated := discoveredDevice("gpu-0", "NVIDIA GeForce RTX 3090", 0, 44.08, 0)
	w, _ := newLimitTestController(t, &fakeWorkerBackend{}, estimated)
	devices := w.deviceController.(*fakeDeviceController)
	worker := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("8Gi", "40"), estimated.UUID)
	w.workerChangeHandler().OnAdd(worker)
	view := openLimiterView(t, w, worker)
	upLimit, memLimit := loadLimiterView(view, 0)
	require.Equal(t, uint32(91), upLimit)
	require.Equal(t, 8*gib, memLimit)

	// Something left other values in the mapping.
	view.WithState(func(state *workerstate.SharedDeviceState) {
		atomic.StoreUint32(&state.V2.Devices[0].DeviceInfo.UpLimit, 100)
		atomic.StoreUint64(&state.V2.Devices[0].DeviceInfo.MemLimit, 80*gib)
	})
	w.syncSharedMemoryState()
	upLimit, memLimit = loadLimiterView(view, 0)
	require.Equal(t, uint32(91), upLimit)
	require.Equal(t, 8*gib, memLimit)

	// The capacity changed and nobody was told.
	devices.setDevice(discoveredDevice("gpu-0", "NVIDIA GeForce RTX 3090", 0, 44.08, 71.2))
	w.syncSharedMemoryState()
	upLimit, _ = loadLimiterView(view, 0)
	require.Equal(t, uint32(57), upLimit)

	// A pass that finds everything in place changes nothing.
	w.syncSharedMemoryState()
	upLimit, memLimit = loadLimiterView(view, 0)
	require.Equal(t, uint32(57), upLimit)
	require.Equal(t, 8*gib, memLimit)
}

// Changes of a worker's limit and of its device's capacity arrive on different
// goroutines. Whatever their order, the mapping ends at the limits of the
// state they left behind: a limit derived before a change is never written
// after the one derived from it.
func TestConcurrentLimitChangesEndAtTheCurrentState(t *testing.T) {
	estimated := discoveredDevice("gpu-0", "NVIDIA GeForce RTX 3090", 0, 44.08, 0)
	listed := discoveredDevice("gpu-0", "NVIDIA GeForce RTX 3090", 0, 44.08, 71.2)
	for round := range 20 {
		w, _ := newLimitTestController(t, &fakeWorkerBackend{}, estimated)
		devices := w.deviceController.(*fakeDeviceController)
		handler := w.workerChangeHandler()
		tracked := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("8Gi", "40"), estimated.UUID)
		handler.OnAdd(tracked)
		view := openLimiterView(t, w, tracked)

		const changes = 200
		var wg sync.WaitGroup
		wg.Add(3)
		go func() {
			defer wg.Done()
			previous := estimated
			for i := range changes {
				next := estimated
				if i%2 == 0 {
					next = listed
				}
				devices.setDevice(next)
				w.onDeviceUpdate(previous, next)
				previous = next
			}
		}()
		var last atomic.Pointer[api.WorkerInfo]
		last.Store(tracked)
		go func() {
			defer wg.Done()
			previous := tracked
			for i := range changes {
				tflops := "40"
				if i%2 == 0 {
					tflops = "20"
				}
				next := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("8Gi", tflops), estimated.UUID)
				handler.OnUpdate(previous, next)
				previous = next
				last.Store(next)
			}
		}()
		go func() {
			defer wg.Done()
			for range changes / 4 {
				w.syncSharedMemoryState()
			}
		}()
		wg.Wait()

		device, _ := devices.GetDevice("gpu-0")
		want := computeUpLimit(last.Load(), device)
		got, _ := loadLimiterView(view, 0)
		require.Equal(t, want, got, "round %d: limit in the mapping after all changes were delivered", round)
	}
}

// Applying limits that are already in place, which the sync loop does for
// every soft worker in every pass, allocates nothing.
func TestApplyingUnchangedLimitsDoesNotAllocate(t *testing.T) {
	first := discoveredDevice("gpu-0", "NVIDIA A100-SXM4-80GB", 0, 19.49184, 312)
	second := discoveredDevice("gpu-1", "NVIDIA A100-SXM4-80GB", 1, 19.49184, 312)
	w, _ := newLimitTestController(t, &fakeWorkerBackend{}, first, second)
	worker := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("8Gi", "40"), first.UUID, second.UUID)
	w.workerChangeHandler().OnAdd(worker)
	upLimit, _ := readMappedLimits(t, w, worker, 1)
	require.Equal(t, uint32(13), upLimit)

	allocations := testing.AllocsPerRun(200, func() {
		w.mu.RLock()
		w.applyWorkerLimitsLocked(w.workers[worker.WorkerUID])
		w.mu.RUnlock()
	})
	require.Zero(t, allocations)
}
