package single_node

import (
	"fmt"
	"sync"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/stretchr/testify/require"
)

func TestFileStateConcurrentWorkerUpdates(t *testing.T) {
	fsm := NewFileStateManager(t.TempDir())
	testConcurrentStateUpdates(t, func(uid string) error {
		return fsm.AddWorker(&api.WorkerInfo{WorkerUID: uid})
	}, fsm.RemoveWorker, fsm.LoadWorkers)
}

func TestFileStateConcurrentDeviceUpdates(t *testing.T) {
	fsm := NewFileStateManager(t.TempDir())
	testConcurrentStateUpdates(t, func(uid string) error {
		return fsm.AddDevice(&api.DeviceInfo{UUID: uid})
	}, fsm.RemoveDevice, fsm.LoadDevices)
}

func testConcurrentStateUpdates[T any](
	t *testing.T, add, remove func(string) error, load func() (map[string]T, error),
) {
	t.Helper()
	const count = 32
	for i := range count {
		require.NoError(t, add(fmt.Sprintf("old-%d", i)))
	}
	var wg sync.WaitGroup
	start := make(chan struct{})
	errs := make([]error, count*2)
	for i := range count {
		wg.Go(func() { <-start; errs[i] = add(fmt.Sprintf("new-%d", i)) })
		wg.Go(func() { <-start; errs[count+i] = remove(fmt.Sprintf("old-%d", i)) })
	}
	close(start)
	wg.Wait()
	for _, err := range errs {
		require.NoError(t, err)
	}
	records, err := load()
	require.NoError(t, err)
	require.Len(t, records, count)
	for i := range count {
		require.Contains(t, records, fmt.Sprintf("new-%d", i))
		require.NotContains(t, records, fmt.Sprintf("old-%d", i))
	}
}
