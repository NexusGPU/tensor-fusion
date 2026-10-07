package single_node

import (
	"os/exec"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/stretchr/testify/require"
)

type stopTestAllocator struct {
	framework.WorkerAllocationController
	deallocate func(string) error
}

func (a *stopTestAllocator) DeallocateWorker(uid string) error {
	return a.deallocate(uid)
}

func TestStopWorkerReleasesAllocationAfterProcessExit(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	info := &api.WorkerInfo{
		WorkerUID: "process-worker",
		WorkerRunningInfo: &api.WorkerRunningInfo{
			Type: api.WorkerRuntimeTypeProcess, Executable: "sleep", Args: []string{"30"},
		},
	}
	require.NoError(t, b.StartWorker(info))
	b.processesMu.RLock()
	cmd, done := b.processes[info.WorkerUID].cmd, b.processes[info.WorkerUID].done
	b.processesMu.RUnlock()
	released := false
	b.allocationController = &stopTestAllocator{deallocate: func(uid string) error {
		require.Equal(t, info.WorkerUID, uid)
		select {
		case <-done:
		default:
			t.Fatal("GPU was released before the process exit callback finished")
		}
		require.NotNil(t, cmd.ProcessState, "child must be reaped before releasing its GPU")
		released = true
		return nil
	}}
	// There is deliberately no subscriber: explicit stop must release even
	// when worker registration and removal have never reached a consumer.
	require.NoError(t, b.StopWorker(info.WorkerUID))
	require.True(t, released)
	require.Empty(t, b.ListWorkers())
	persisted, err := b.fileState.LoadWorkers()
	require.NoError(t, err)
	require.Empty(t, persisted)
}

func TestStartWorkerCannotReplaceLiveProcessOrStartAfterStop(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	info := &api.WorkerInfo{
		WorkerUID: "process-worker",
		WorkerRunningInfo: &api.WorkerRunningInfo{
			Type: api.WorkerRuntimeTypeProcess, Executable: "sleep", Args: []string{"30"},
		},
	}
	require.NoError(t, b.StartWorker(info))
	pid := b.ListWorkers()[0].WorkerRunningInfo.PID
	require.Error(t, b.StartWorker(info))
	require.Equal(t, pid, b.ListWorkers()[0].WorkerRunningInfo.PID)
	require.NoError(t, b.Stop())
	require.Error(t, b.StartWorker(&api.WorkerInfo{WorkerUID: "after-stop"}))
}

func TestFileRemovalStopsManagedProcessBeforeNotifyingRemoval(t *testing.T) {
	for _, change := range []string{"removed", "deletion marker", "terminal status"} {
		t.Run(change, func(t *testing.T) {
			b := newWorkerSyncTestBackend(t)
			info := newProcessWorker()
			require.NoError(t, b.StartWorker(info))
			b.processesMu.RLock()
			done := b.processes[info.WorkerUID].done
			b.processesMu.RUnlock()
			added := make(chan string, 1)
			removed := make(chan bool, 1)
			notifyRemoval := func(*api.WorkerInfo) {
				select {
				case <-done:
					removed <- true
				default:
					removed <- false
				}
			}
			require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
				OnAdd:    func(info *api.WorkerInfo) { added <- info.WorkerUID },
				OnRemove: notifyRemoval,
				OnUpdate: func(_, current *api.WorkerInfo) {
					if current.Status == api.WorkerStatusTerminated {
						notifyRemoval(current)
					}
				},
			}))
			require.Equal(t, info.WorkerUID, receiveWorker(t, added))
			if change == "removed" {
				require.NoError(t, b.fileState.RemoveWorker(info.WorkerUID))
			} else {
				current := b.ListWorkers()[0]
				if change == "deletion marker" {
					current.DeletedAt = time.Now().Unix()
				} else {
					current.Status = api.WorkerStatusTerminated
				}
				require.NoError(t, b.fileState.AddWorker(current))
			}
			b.discoverWorkers()
			select {
			case exited := <-removed:
				require.True(t, exited, "file removal must stop the managed process before releasing its GPU")
			case <-time.After(time.Second):
				t.Fatal("removed worker was not reconciled")
			}
		})
	}
}

func TestDelayedProcessExitDoesNotOverwriteReplacement(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	old := exec.Command("sh", "-c", "exit 1")
	require.NoError(t, old.Start())
	replacement := exec.Command("sh", "-c", "exit 0")
	info := &api.WorkerInfo{WorkerUID: "worker", WorkerRunningInfo: &api.WorkerRunningInfo{IsRunning: true, PID: 123}}
	b.workers[info.WorkerUID] = info
	b.processes[info.WorkerUID] = &processState{cmd: replacement, isRunning: true}
	b.waitForProcess(info.WorkerUID, old, nil, make(chan struct{}))
	require.Same(t, replacement, b.processes[info.WorkerUID].cmd)
	require.True(t, b.processes[info.WorkerUID].isRunning)
	require.True(t, b.workers[info.WorkerUID].WorkerRunningInfo.IsRunning)
}
