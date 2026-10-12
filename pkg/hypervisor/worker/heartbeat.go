package worker

import (
	"context"
	"time"

	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
)

// The heartbeat tells the limiter in every worker that the hypervisor is alive.
// It has a resolution of one second, and a limiter takes the hypervisor for
// gone, and enforces from its own clock, once the heartbeat is two seconds old.
//
// It is therefore written by a loop of its own that does nothing else: no
// provider call, no API call, no lock shared with anything slow. It reads the
// mappings from a snapshot published whenever they change, so not even the
// worker controller's own lock can hold it up. A slow device sample delays
// usage accounting, never the heartbeat.
const heartbeatInterval = 500 * time.Millisecond

// publishHeartbeatHandlesLocked republishes the mappings the heartbeat loop
// serves. The caller holds w.mu for writing and calls it after every change of
// w.shmHandles.
func (w *WorkerController) publishHeartbeatHandlesLocked() {
	handles := make([]*workerstate.SharedMemoryHandle, 0, len(w.shmHandles))
	for _, handle := range w.shmHandles {
		handles = append(handles, handle)
	}
	w.heartbeatHandles.Store(&handles)
}

// writeHeartbeats stamps every mapping with the current time. A mapping that
// was closed in the meantime, or whose backing file no longer covers it, is
// skipped by its handle.
func (w *WorkerController) writeHeartbeats() {
	handles := w.heartbeatHandles.Load()
	if handles == nil {
		return
	}
	now := uint64(w.now().Unix())
	for _, handle := range *handles {
		handle.WithState(func(state *workerstate.SharedDeviceState) { state.UpdateHeartbeat(now) })
	}
}

func (w *WorkerController) startHeartbeatLoop(ctx context.Context) {
	w.syncWG.Add(1)
	go func() {
		defer w.syncWG.Done()
		ticker := time.NewTicker(heartbeatInterval)
		defer ticker.Stop()
		for {
			w.writeHeartbeats()
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
		}
	}()
}
