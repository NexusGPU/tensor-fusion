package single_node

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/stretchr/testify/require"
)

func newWorkerSyncTestBackend(t *testing.T) *SingleNodeBackend {
	t.Helper()
	b := NewSingleNodeBackend(t.Context(), nil, nil, WithStateDir(t.TempDir()))
	t.Cleanup(func() { require.NoError(t, b.Stop()) })
	return b
}

func receiveWorker(t *testing.T, ch <-chan string) string {
	t.Helper()
	select {
	case uid := <-ch:
		return uid
	case <-time.After(time.Second):
		t.Fatal("worker lifecycle notification was lost")
		return ""
	}
}

func TestWorkerSyncRestoresInitialFileState(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	info := &api.WorkerInfo{WorkerUID: "running-before-restart", Status: api.WorkerStatusRunning}
	require.NoError(t, b.fileState.AddWorker(info))
	added := make(chan string, 1)
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnAdd: func(info *api.WorkerInfo) { added <- info.WorkerUID },
	}))
	require.NoError(t, b.Start())
	require.Equal(t, info.WorkerUID, receiveWorker(t, added))
}

func TestWorkerSyncRemovesStoppedWorker(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	info := &api.WorkerInfo{WorkerUID: "stopped-worker"}
	added, removed := make(chan string, 1), make(chan string, 1)
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnAdd:    func(info *api.WorkerInfo) { added <- info.WorkerUID },
		OnRemove: func(info *api.WorkerInfo) { removed <- info.WorkerUID },
	}))
	require.NoError(t, b.StartWorker(info))
	require.Equal(t, info.WorkerUID, receiveWorker(t, added))
	require.NoError(t, b.StopWorker(info.WorkerUID))
	require.Equal(t, info.WorkerUID, receiveWorker(t, removed))
}

func TestWorkerSyncCoalescesBurstWithoutLosingDeletion(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	first := &api.WorkerInfo{WorkerUID: "first"}
	entered, release := make(chan struct{}), make(chan struct{})
	var unblock sync.Once
	t.Cleanup(func() { unblock.Do(func() { close(release) }) })
	removed := make(chan string, 1)
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnAdd: func(info *api.WorkerInfo) {
			if info.WorkerUID == first.WorkerUID {
				close(entered)
				<-release
			}
		},
		OnRemove: func(info *api.WorkerInfo) { removed <- info.WorkerUID },
	}))
	require.NoError(t, b.StartWorker(first))
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("initial Add never reached the subscriber")
	}
	for i := range 40 {
		require.NoError(t, b.StartWorker(&api.WorkerInfo{WorkerUID: fmt.Sprintf("worker-%d", i)}))
	}
	require.NoError(t, b.StopWorker(first.WorkerUID))
	unblock.Do(func() { close(release) })
	require.Equal(t, first.WorkerUID, receiveWorker(t, removed))
}

func TestWorkerSyncDetectsMetadataChangeAndFileDeletion(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	info := &api.WorkerInfo{WorkerUID: "worker", Status: api.WorkerStatusRunning, AllocatedDevices: []string{"gpu-0"}}
	require.NoError(t, b.fileState.AddWorker(info))
	require.NoError(t, b.loadState())
	added, updated, removed := make(chan string, 1), make(chan string, 1), make(chan string, 1)
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnAdd: func(info *api.WorkerInfo) { added <- info.WorkerUID },
		OnUpdate: func(oldInfo, newInfo *api.WorkerInfo) {
			if oldInfo.AllocatedDevices[0] == "gpu-0" && newInfo.AllocatedDevices[0] == "gpu-1" {
				updated <- newInfo.WorkerUID
			}
		},
		OnRemove: func(info *api.WorkerInfo) { removed <- info.WorkerUID },
	}))
	require.Equal(t, info.WorkerUID, receiveWorker(t, added))
	info.AllocatedDevices = []string{"gpu-1"}
	require.NoError(t, b.fileState.AddWorker(info))
	b.discoverWorkers()
	require.Equal(t, info.WorkerUID, receiveWorker(t, updated))
	require.NoError(t, b.fileState.RemoveWorker(info.WorkerUID))
	b.discoverWorkers()
	require.Equal(t, info.WorkerUID, receiveWorker(t, removed))
}
