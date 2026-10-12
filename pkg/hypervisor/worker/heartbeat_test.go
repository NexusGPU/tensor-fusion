package worker

import (
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/stretchr/testify/require"
)

// blockingDeviceController hangs in the process sample until released, the way
// a provider call into a stuck driver does.
type blockingDeviceController struct {
	fakeDeviceController
	entered chan struct{}
	release chan struct{}
}

func (d *blockingDeviceController) GetProcessInformation() ([]api.ProcessInformation, error) {
	select {
	case d.entered <- struct{}{}:
	default:
	}
	<-d.release
	return nil, nil
}

// A limiter takes the hypervisor for gone when the heartbeat is two seconds
// old. Nothing slow may therefore stand between the clock and the heartbeat:
// not a provider call that hangs, and not the worker controller's own lock.
func TestHeartbeatDoesNotDependOnDeviceSamplingOrLocks(t *testing.T) {
	w, info := newSharedMemoryTestController(t)
	require.NoError(t, w.WithWorkerSharedMemory(info.WorkerUID, nil))
	now := time.Unix(1_800_000_000, 0)
	w.nowFunc = func() time.Time { return now }

	devices := &blockingDeviceController{entered: make(chan struct{}, 1), release: make(chan struct{})}
	w.deviceController = devices
	syncDone := make(chan struct{})
	go func() {
		defer close(syncDone)
		w.syncSharedMemoryState()
	}()
	select {
	case <-devices.entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the sync pass never reached the process sample")
	}

	// The sync pass is stuck in the provider. The heartbeat still advances.
	for range 3 {
		now = now.Add(time.Second)
		w.writeHeartbeats()
		require.Equal(t, uint64(now.Unix()), workerHeartbeat(t, w, info.WorkerUID))
	}

	// It does not need the worker controller's lock either.
	w.mu.Lock()
	now = now.Add(time.Second)
	written := make(chan struct{})
	go func() {
		defer close(written)
		w.writeHeartbeats()
	}()
	select {
	case <-written:
	case <-time.After(5 * time.Second):
		t.Fatal("writing the heartbeat waited for the worker controller's lock")
	}
	w.mu.Unlock()
	require.Equal(t, uint64(now.Unix()), workerHeartbeat(t, w, info.WorkerUID))

	close(devices.release)
	<-syncDone
}

// A mapping is stamped the moment it is opened, and no longer once it is closed.
func TestHeartbeatFollowsTheOpenMappings(t *testing.T) {
	w, info := newSharedMemoryTestController(t)
	// Adding the worker opened its mapping already.
	require.Len(t, *w.heartbeatHandles.Load(), 1)
	require.NotZero(t, workerHeartbeat(t, w, info.WorkerUID))
	now := time.Unix(1_800_000_000, 0)
	w.nowFunc = func() time.Time { return now }

	// Reopened after a restart: stamped at once, which ends the limiter's
	// stand-in for the hypervisor without waiting for a tick.
	w.mu.Lock()
	w.closeSharedMemoryLocked(info.WorkerUID)
	w.mu.Unlock()
	require.Empty(t, *w.heartbeatHandles.Load())
	now = now.Add(10 * time.Second)
	require.NoError(t, w.WithWorkerSharedMemory(info.WorkerUID, func(state *workerstate.SharedDeviceState) {
		require.Equal(t, uint64(now.Unix()), state.GetLastHeartbeat())
	}))

	// A pass over closed mappings touches nothing.
	handles := *w.heartbeatHandles.Load()
	w.mu.Lock()
	w.closeSharedMemoryLocked(info.WorkerUID)
	w.mu.Unlock()
	stale := handles
	w.heartbeatHandles.Store(&stale)
	w.writeHeartbeats()
}
