package single_node

import (
	"errors"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/stretchr/testify/require"
)

func newProcessWorker() *api.WorkerInfo {
	return &api.WorkerInfo{
		WorkerUID: "process-worker",
		WorkerRunningInfo: &api.WorkerRunningInfo{
			Type: api.WorkerRuntimeTypeProcess, Executable: "sleep", Args: []string{"30"},
		},
	}
}

func TestStartWorkerPersistenceFailureDoesNotStartProcess(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	blocked := filepath.Join(t.TempDir(), "not-a-directory")
	require.NoError(t, os.WriteFile(blocked, nil, 0600))
	b.fileState = NewFileStateManager(blocked)
	require.Error(t, b.StartWorker(newProcessWorker()))
	b.processesMu.RLock()
	count := len(b.processes)
	b.processesMu.RUnlock()
	require.Zero(t, count, "a failed StartWorker must not leave a child running")
	require.Empty(t, b.ListWorkers())
}

func TestGPUProcessStartsAfterPreparation(t *testing.T) {
	for _, mode := range []tfv1.IsolationModeType{tfv1.IsolationModeSoft, tfv1.IsolationModeHard} {
		t.Run(string(mode), func(t *testing.T) {
			b := newWorkerSyncTestBackend(t)
			info := newProcessWorker()
			info.IsolationMode, info.AllocatedDevices = mode, []string{"gpu-0"}
			prepared := filepath.Join(t.TempDir(), "prepared")
			result := filepath.Join(t.TempDir(), "result")
			info.WorkerRunningInfo.Executable = "sh"
			info.WorkerRunningInfo.Args = []string{"-c", `test -f "$PREPARED" && printf ready > "$RESULT"`}
			info.WorkerRunningInfo.Env = map[string]string{"PREPARED": prepared, "RESULT": result}
			var added atomic.Bool
			require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
				OnAdd: func(*api.WorkerInfo) { added.Store(true) },
				OnPrepare: func(uid string) error {
					require.Equal(t, info.WorkerUID, uid)
					require.True(t, added.Load(), "the worker must be registered before preparation")
					return os.WriteFile(prepared, []byte("ready"), 0600)
				},
			}))
			require.NoError(t, b.StartWorker(info))
			require.Eventually(t, func() bool {
				content, err := os.ReadFile(result)
				return err == nil && string(content) == "ready"
			}, time.Second, time.Millisecond)
		})
	}
}

func TestStartWorkerPreparationFailureRollsBack(t *testing.T) {
	for _, missing := range []bool{false, true} {
		name := "failed preparation"
		if missing {
			name = "missing preparation handler"
		}
		t.Run(name, func(t *testing.T) {
			b := newWorkerSyncTestBackend(t)
			info := newProcessWorker()
			info.IsolationMode, info.AllocatedDevices = tfv1.IsolationModeHard, []string{"gpu-0"}
			failure := errors.New("shared memory is unavailable")
			if !missing {
				require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
					OnPrepare: func(string) error { return failure },
				}))
			}
			var released atomic.Bool
			b.allocationController = &stopTestAllocator{deallocate: func(string) error { released.Store(true); return nil }}
			err := b.StartWorker(info)
			require.Error(t, err)
			if !missing {
				require.ErrorIs(t, err, failure)
			}
			require.True(t, released.Load(), "failed startup must return its reserved GPU")
			require.Empty(t, b.ListWorkers())
			workers, err := b.fileState.LoadWorkers()
			require.NoError(t, err)
			require.Empty(t, workers)
			b.processesMu.RLock()
			count := len(b.processes)
			b.processesMu.RUnlock()
			require.Zero(t, count)
		})
	}
}

func TestStartWorkerExecFailureRollsBack(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	info := newProcessWorker()
	info.WorkerRunningInfo.Executable = filepath.Join(t.TempDir(), "does-not-exist")
	require.Error(t, b.StartWorker(info))
	require.Empty(t, b.ListWorkers())
	workers, err := b.fileState.LoadWorkers()
	require.NoError(t, err)
	require.Empty(t, workers)
}

func TestStartWorkerPIDPersistenceFailureStopsProcessAndRetainsCleanup(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	info := newProcessWorker()
	info.IsolationMode, info.AllocatedDevices = tfv1.IsolationModeHard, []string{"gpu-0"}
	blocked := filepath.Join(b.stateDir, workersFile+".tmp")
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnPrepare: func(string) error {
			// The initial worker write succeeded. Fail the PID write and cleanup
			// write after preparation, without changing the worker's persisted data.
			return os.Mkdir(blocked, 0755)
		},
	}))
	var released atomic.Bool
	b.allocationController = &stopTestAllocator{deallocate: func(string) error { released.Store(true); return nil }}
	require.Error(t, b.StartWorker(info))
	b.processesMu.RLock()
	count := len(b.processes)
	b.processesMu.RUnlock()
	require.Zero(t, count, "PID persistence failure must stop and reap the child")
	current := b.ListWorkers()
	require.Len(t, current, 1, "failed cleanup must remain visible for retry")
	require.False(t, current[0].WorkerRunningInfo.IsRunning)
	require.Zero(t, current[0].WorkerRunningInfo.PID)
	require.False(t, released.Load(), "a failed state deletion must not forget GPU ownership")
	require.NoError(t, os.Remove(blocked))
	require.NoError(t, b.StopWorker(info.WorkerUID))
	require.True(t, released.Load())
	require.Empty(t, b.ListWorkers())
}

func TestRestartProcessWaitsForPreparation(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	info := newProcessWorker()
	info.IsolationMode, info.AllocatedDevices = tfv1.IsolationModeSoft, []string{"gpu-0"}
	info.WorkerRunningInfo.Executable = "sh"
	info.WorkerRunningInfo.Args = []string{"-c", "exit 1"}
	failure := errors.New("preparation unavailable on restart")
	var fail atomic.Bool
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnPrepare: func(string) error {
			if fail.Load() {
				return failure
			}
			return nil
		},
	}))
	require.NoError(t, b.StartWorker(info))
	b.processesMu.RLock()
	done := b.processes[info.WorkerUID].done
	b.processesMu.RUnlock()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("test child did not exit")
	}
	fail.Store(true)
	require.ErrorIs(t, b.restartProcess(info.WorkerUID), failure)
	b.processesMu.RLock()
	cmd := b.processes[info.WorkerUID].cmd
	b.processesMu.RUnlock()
	require.Nil(t, cmd, "a restart must not bypass preparation")
}
