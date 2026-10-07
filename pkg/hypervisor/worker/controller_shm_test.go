package worker

import (
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/stretchr/testify/require"
)

func newSharedMemoryTestController(t *testing.T) (*WorkerController, *api.WorkerInfo) {
	t.Helper()
	info := &api.WorkerInfo{
		WorkerUID: "first-uid", Namespace: "namespace", WorkerName: "worker",
		IsolationMode: tfv1.IsolationModeSoft, Status: api.WorkerStatusDeviceAllocating,
	}
	controller := &WorkerController{
		backend: &fakeWorkerBackend{}, deviceController: &fakeDeviceController{},
		allocationController: &fakeWorkerAllocationController{allocations: map[string]*api.WorkerAllocation{
			info.WorkerUID: {WorkerInfo: info, DeviceInfos: []*api.DeviceInfo{{UUID: "gpu-0", TotalMemoryBytes: 1 << 30}}},
		}},
		workers: map[string]*api.WorkerInfo{}, shmHandles: map[string]*workerstate.SharedMemoryHandle{},
		shmBasePath: t.TempDir(), nowFunc: time.Now,
	}
	t.Cleanup(func() {
		controller.mu.Lock()
		defer controller.mu.Unlock()
		for uid := range controller.shmHandles {
			controller.closeSharedMemoryLocked(uid)
		}
	})
	controller.workerChangeHandler().OnAdd(info)
	return controller, info
}

func TestWorkerSharedMemoryConcurrentInitPreservesCounters(t *testing.T) {
	w, info := newSharedMemoryTestController(t)
	original := w.getShmHandle(info.WorkerUID)
	require.NotNil(t, original)
	const count = 32
	errs := make([]error, count)
	var wg sync.WaitGroup
	for i := range count {
		wg.Go(func() {
			errs[i] = w.WithWorkerSharedMemory(info.WorkerUID, func(state *workerstate.SharedDeviceState) {
				state.SetPodMemoryUsed(0, state.V2.Devices[0].DeviceInfo.PodMemoryUsed+1)
			})
		})
	}
	wg.Wait()
	for _, err := range errs {
		require.NoError(t, err)
	}
	// A duplicate informer Add and a later background preparation must reuse
	// the handle already used by HTTP /pod and /process requests.
	w.workerChangeHandler().OnAdd(info)
	require.Same(t, original, w.getShmHandle(info.WorkerUID))
	require.Equal(t, uint64(count), original.GetState().V2.Devices[0].DeviceInfo.PodMemoryUsed)
}

func TestWorkerSharedMemoryRemovalFencesLateRequests(t *testing.T) {
	w, info := newSharedMemoryTestController(t)
	original := w.getShmHandle(info.WorkerUID)
	entered, release, done := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	go func() {
		done <- w.WithWorkerSharedMemory(info.WorkerUID, func(state *workerstate.SharedDeviceState) {
			close(entered)
			<-release
			state.SetPodMemoryUsed(0, 512)
		})
	}()
	<-entered
	removed := make(chan struct{})
	go func() { w.workerChangeHandler().OnRemove(info); close(removed) }()
	close(release)
	require.NoError(t, <-done)
	<-removed
	require.Nil(t, original.GetState(), "removal must close the hypervisor's mapping")
	require.Error(t, w.WithWorkerSharedMemory(info.WorkerUID, nil))
	w.cleanupOrphanedSharedMemory()
	w.syncSharedMemoryState() // The allocator still has a stale entry in this fixture.
	_, err := os.Stat(filepath.Join(w.shmBasePath, info.Namespace, info.WorkerName, workerstate.ShmPathSuffix))
	require.ErrorIs(t, err, os.ErrNotExist)
}

func TestWorkerSharedMemorySameNameReplacementRejectsOldUID(t *testing.T) {
	w, info := newSharedMemoryTestController(t)
	handler := w.workerChangeHandler()
	shmPath := filepath.Join(w.shmBasePath, info.Namespace, info.WorkerName, workerstate.ShmPathSuffix)
	before, err := os.Stat(shmPath)
	require.NoError(t, err)
	require.NoError(t, w.WithWorkerSharedMemory(info.WorkerUID, func(state *workerstate.SharedDeviceState) {
		state.SetPodMemoryUsed(0, 512)
	}))
	handler.OnRemove(info)
	replacement := *info
	replacement.WorkerUID = "replacement-uid"
	w.allocationController.(*fakeWorkerAllocationController).allocations[replacement.WorkerUID] = &api.WorkerAllocation{
		WorkerInfo: &replacement, DeviceInfos: []*api.DeviceInfo{{UUID: "gpu-0", TotalMemoryBytes: 2 << 30}},
	}
	handler.OnAdd(&replacement)
	after, err := os.Stat(shmPath)
	require.NoError(t, err)
	require.False(t, os.SameFile(before, after))
	require.Error(t, w.WithWorkerSharedMemory(info.WorkerUID, nil))
	handler.OnRemove(info) // A delayed old deletion must not remove the replacement.
	w.cleanupOrphanedSharedMemory()
	require.NoError(t, w.WithWorkerSharedMemory(replacement.WorkerUID, func(state *workerstate.SharedDeviceState) {
		require.Equal(t, uint64(0), state.V2.Devices[0].DeviceInfo.PodMemoryUsed)
		require.Equal(t, uint64(2<<30), state.V2.Devices[0].DeviceInfo.MemLimit)
	}))
	current, err := os.Stat(shmPath)
	require.NoError(t, err)
	require.True(t, os.SameFile(after, current))
}

func TestWorkerSharedMemoryTerminationRejectsInitialization(t *testing.T) {
	w, info := newSharedMemoryTestController(t)
	terminated := *info
	terminated.Status = api.WorkerStatusTerminated
	w.workerChangeHandler().OnUpdate(info, &terminated)
	require.Nil(t, w.getShmHandle(info.WorkerUID))
	require.Error(t, w.WithWorkerSharedMemory(info.WorkerUID, nil))
	w.syncSharedMemoryState()
	require.Nil(t, w.getShmHandle(info.WorkerUID))
}

func TestWorkerSharedMemoryPreservesHardAndSharedClientConfig(t *testing.T) {
	for _, mode := range []tfv1.IsolationModeType{tfv1.IsolationModeHard, tfv1.IsolationModeShared} {
		t.Run(string(mode), func(t *testing.T) {
			w, first := newSharedMemoryTestController(t)
			w.workerChangeHandler().OnRemove(first)
			info := *first
			info.WorkerUID = string(mode) + "-uid"
			info.IsolationMode = mode
			w.allocationController.(*fakeWorkerAllocationController).allocations[info.WorkerUID] = &api.WorkerAllocation{
				WorkerInfo: &info, DeviceInfos: []*api.DeviceInfo{{
					UUID: "gpu-1234", TotalMemoryBytes: 1 << 30,
					Properties: map[string]string{"totalComputeUnits": "10", "computeCapability": "7.0"},
				}},
			}
			w.workerChangeHandler().OnAdd(&info)
			require.NoError(t, w.WithWorkerSharedMemory(info.WorkerUID, func(state *workerstate.SharedDeviceState) {
				require.Equal(t, "GPU-1234", state.V2.Devices[0].GetUUID())
				require.Equal(t, uint64(1<<30), state.V2.Devices[0].DeviceInfo.MemLimit)
			}))
		})
	}
}

func TestFirstWorkerNotificationCanAlreadyBeTerminated(t *testing.T) {
	allocations := &fakeWorkerAllocationController{}
	w := &WorkerController{
		allocationController: allocations,
		workers:              map[string]*api.WorkerInfo{}, shmHandles: map[string]*workerstate.SharedMemoryHandle{},
	}
	info := &api.WorkerInfo{WorkerUID: "failed-before-add", Status: api.WorkerStatusTerminated}
	w.workerChangeHandler().OnAdd(info)
	require.Equal(t, []string{info.WorkerUID}, allocations.deallocated)
	require.Error(t, w.WithWorkerSharedMemory(info.WorkerUID, nil))
}
