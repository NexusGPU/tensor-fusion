package kubernetes

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

type syncTestAllocations struct {
	mu      sync.Mutex
	workers map[string]*api.WorkerAllocation
}

func (a *syncTestAllocations) AllocateWorkerDevices(info *api.WorkerInfo) (*api.WorkerAllocation, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if existing := a.workers[info.WorkerUID]; existing != nil {
		return existing, nil
	}
	allocation := &api.WorkerAllocation{WorkerInfo: info}
	a.workers[info.WorkerUID] = allocation
	return allocation, nil
}
func (a *syncTestAllocations) DeallocateWorker(uid string) error {
	a.mu.Lock()
	defer a.mu.Unlock()
	delete(a.workers, uid)
	return nil
}
func (a *syncTestAllocations) RetryPendingCleanup() error                             { return nil }
func (a *syncTestAllocations) RecoverPartitionedWorker(*api.WorkerInfo, string) error { return nil }
func (a *syncTestAllocations) GetWorkerAllocation(uid string) (*api.WorkerAllocation, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	allocation, ok := a.workers[uid]
	return allocation, ok
}
func (a *syncTestAllocations) GetDeviceAllocations() map[string][]*api.WorkerAllocation {
	a.mu.Lock()
	defer a.mu.Unlock()
	result := map[string][]*api.WorkerAllocation{}
	for _, allocation := range a.workers {
		result["gpu-0"] = append(result["gpu-0"], allocation)
	}
	return result
}

func newWorkerSyncTestBackend(t *testing.T) *KubeletBackend {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	cache := &PodCacheManager{
		ctx: ctx, nodeName: "node", cachedPod: map[string]*corev1.Pod{},
		indexToWorkerInfo: map[int]*api.WorkerInfo{}, stopCh: make(chan struct{}), workerChangedCh: make(chan struct{}, 1),
		indexSubscribers: map[int]map[*workerInfoSubscriber]struct{}{},
		podSubscribers:   map[string]chan<- *api.WorkerInfo{},
	}
	b := &KubeletBackend{
		ctx: ctx, podCacher: cache, workers: map[string]*api.WorkerInfo{}, subscribers: map[string]struct{}{},
		allocationController: &syncTestAllocations{workers: map[string]*api.WorkerAllocation{}},
	}
	t.Cleanup(cancel)
	return b
}

func TestWorkerSyncRetainsDeletionWhenNotificationsOverflow(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	first := createTestPodWithIndex(1)
	entered, release := make(chan struct{}), make(chan struct{})
	var unblock sync.Once
	t.Cleanup(func() { unblock.Do(func() { close(release) }) })
	removed := make(chan string, 1)
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnAdd: func(info *api.WorkerInfo) {
			if info.WorkerUID == string(first.UID) {
				close(entered)
				<-release
			}
		},
		OnRemove: func(info *api.WorkerInfo) { removed <- info.WorkerUID },
	}))
	b.podCacher.onPodAdd(first)
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("first Add was not delivered")
	}
	// Simulate a slow SDK call while a burst fills the former 16-event buffer.
	for i := 2; i <= 42; i++ {
		b.podCacher.onPodAdd(createTestPodWithIndex(i))
	}
	b.podCacher.onPodDelete(first)
	unblock.Do(func() { close(release) })
	select {
	case uid := <-removed:
		require.Equal(t, string(first.UID), uid)
	case <-time.After(time.Second):
		t.Fatal("dropped deletion left an obsolete worker allocation")
	}
	require.Eventually(t, func() bool { return len(b.ListWorkers()) == 41 }, time.Second, time.Millisecond)
}

func TestWorkerSyncLoadsPodsCachedBeforeRegistration(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	pod := createTestPodWithIndex(1)
	b.podCacher.onPodAdd(pod)
	added := make(chan string, 1)
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnAdd: func(info *api.WorkerInfo) { added <- info.WorkerUID },
	}))
	select {
	case uid := <-added:
		require.Equal(t, string(pod.UID), uid)
	case <-time.After(time.Second):
		t.Fatal("existing worker was never recovered")
	}
}

func TestWorkerSyncCleansAllocationWhoseAddAndDeleteWereCoalesced(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	pod := createTestPodWithIndex(1)
	b.podCacher.onPodAdd(pod)
	info, _, err := b.podCacher.extractWorkerInfo(pod)
	require.NoError(t, err)
	_, err = b.allocationController.AllocateWorkerDevices(info)
	require.NoError(t, err)
	b.podCacher.onPodDelete(pod)
	removed := make(chan struct{}, 1)
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnRemove: func(info *api.WorkerInfo) {
			_ = b.allocationController.DeallocateWorker(info.WorkerUID)
			removed <- struct{}{}
		},
	}))
	select {
	case <-removed:
	case <-time.After(time.Second):
		t.Fatal("allocation survived an Add/Delete burst")
	}
	_, exists := b.allocationController.GetWorkerAllocation(info.WorkerUID)
	require.False(t, exists)
}

func TestWorkerSyncCleansDeletedPodEvenIfAnnotationsBecomeInvalid(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	pod := createTestPodWithIndex(1)
	added, removed := make(chan struct{}, 1), make(chan struct{}, 1)
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnAdd:    func(*api.WorkerInfo) { added <- struct{}{} },
		OnRemove: func(*api.WorkerInfo) { removed <- struct{}{} },
	}))
	b.podCacher.onPodAdd(pod)
	select {
	case <-added:
	case <-time.After(time.Second):
		t.Fatal("Add was not delivered")
	}
	invalid := pod.DeepCopy()
	invalid.Annotations[constants.GpuCountAnnotation] = "not-a-count"
	_, _, err := b.podCacher.extractWorkerInfo(invalid)
	require.Error(t, err)
	b.podCacher.onPodDelete(invalid)
	select {
	case <-removed:
	case <-time.After(time.Second):
		t.Fatal("invalid annotations prevented deletion cleanup")
	}
	require.Empty(t, b.ListWorkers())
}

func TestDevicePluginPreparesCurrentPodBeforeReturningAllocation(t *testing.T) {
	for _, mode := range []string{"soft", "hard"} {
		for _, pluginIndex := range []int{0, 1} { // Legacy index and v2 index_0.
			t.Run(mode+"/"+[]string{"legacy", "v2"}[pluginIndex], func(t *testing.T) {
				b := newWorkerSyncTestBackend(t)
				pod := createTestPodWithIndex(1)
				pod.Annotations[constants.IsolationModeAnnotation] = mode
				b.podCacher.onPodAdd(pod)
				base := t.TempDir()
				id := workerstate.NewPodIdentifier(pod.Namespace, pod.Name)
				old, err := workerstate.PrepareWorkerSharedMemory(base, id, "previous-uid", []workerstate.DeviceConfig{
					{DeviceIdx: 0, DeviceUUID: "gpu-0", MemLimit: 4 << 30},
				}, false)
				require.NoError(t, err)
				defer func() { _ = old.Close() }()
				old.GetState().SetPodMemoryUsed(0, 512)
				var added atomic.Bool
				require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
					OnAdd: func(*api.WorkerInfo) { added.Store(true) },
					OnPrepare: func(uid string) error {
						require.True(t, added.Load(), "preparation raced worker registration")
						handle, err := workerstate.PrepareWorkerSharedMemory(base, id, uid, []workerstate.DeviceConfig{
							{DeviceIdx: 0, DeviceUUID: "gpu-0", MemLimit: 2 << 30},
						}, false)
						if err != nil {
							return err
						}
						return handle.Close()
					},
				}))
				plugins := NewDevicePlugins(t.Context(), nil, b.allocateWorkerDevices, b.podCacher)
				dp := plugins[pluginIndex]
				response, err := dp.Allocate(t.Context(), &pluginapi.AllocateRequest{
					ContainerRequests: []*pluginapi.ContainerAllocateRequest{{DevicesIds: []string{dp.deviceID(0)}}},
				})
				require.NoError(t, err)
				require.Len(t, response.ContainerResponses, 1)
				// A process may open SHM immediately after the device-plugin RPC.
				current, err := workerstate.OpenSharedMemoryHandle(base, id)
				require.NoError(t, err)
				defer func() { _ = current.Close() }()
				require.Equal(t, uint64(2<<30), current.GetState().V2.Devices[0].DeviceInfo.MemLimit)
				require.Equal(t, uint64(0), current.GetState().V2.Devices[0].DeviceInfo.PodMemoryUsed)
				target, err := os.Readlink(filepath.Join(id.ToPath(base), workerstate.ShmPathSuffix))
				require.NoError(t, err)
				require.Contains(t, target, string(pod.UID))
			})
		}
	}
}

func TestDeviceAllocationRetriesPreparationWithoutAllocatingAgain(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	pod := createTestPodWithIndex(1)
	b.podCacher.onPodAdd(pod)
	info, _, err := b.podCacher.extractWorkerInfo(pod)
	require.NoError(t, err)
	var attempts int
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnPrepare: func(string) error {
			attempts++
			if attempts == 1 {
				return errors.New("temporary shared memory failure")
			}
			return nil
		},
	}))
	_, err = b.allocateWorkerDevices(t.Context(), info)
	require.ErrorContains(t, err, "temporary shared memory failure")
	first, exists := b.allocationController.GetWorkerAllocation(info.WorkerUID)
	require.True(t, exists)
	second, err := b.allocateWorkerDevices(t.Context(), info)
	require.NoError(t, err)
	require.Same(t, first, second)
	require.Equal(t, 2, attempts)
}

func TestDeviceAllocationCleansPodDeletedDuringPreparation(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	pod := createTestPodWithIndex(1)
	b.podCacher.onPodAdd(pod)
	info, _, err := b.podCacher.extractWorkerInfo(pod)
	require.NoError(t, err)
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnPrepare: func(string) error { b.podCacher.onPodDelete(pod); return nil },
	}))
	_, err = b.allocateWorkerDevices(t.Context(), info)
	require.ErrorContains(t, err, "stopped during allocation")
	_, exists := b.allocationController.GetWorkerAllocation(info.WorkerUID)
	require.False(t, exists, "late allocation must be cleaned even if the delete event ran first")
}

func TestDevicePluginCancellationRemovesPendingWorkerSubscription(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	dp := NewDevicePlugins(t.Context(), nil, b.allocateWorkerDevices, b.podCacher)[1]
	done := make(chan error, 1)
	go func() {
		_, err := dp.Allocate(ctx, &pluginapi.AllocateRequest{
			ContainerRequests: []*pluginapi.ContainerAllocateRequest{{DevicesIds: []string{dp.deviceID(0)}}},
		})
		done <- err
	}()
	require.Eventually(t, func() bool {
		b.podCacher.subscribersMu.RLock()
		defer b.podCacher.subscribersMu.RUnlock()
		return len(b.podCacher.indexSubscribers[1]) == 1
	}, time.Second, time.Millisecond)
	cancel()
	select {
	case err := <-done:
		require.ErrorIs(t, err, context.Canceled)
	case <-time.After(time.Second):
		t.Fatal("canceled RPC left a worker wait running")
	}
	b.podCacher.subscribersMu.RLock()
	require.Empty(t, b.podCacher.indexSubscribers)
	b.podCacher.subscribersMu.RUnlock()
}

func TestDeviceAllocationDoesNotReleaseTerminatingPod(t *testing.T) {
	b := newWorkerSyncTestBackend(t)
	pod := createTestPodWithIndex(1)
	b.podCacher.onPodAdd(pod)
	info, _, err := b.podCacher.extractWorkerInfo(pod)
	require.NoError(t, err)
	require.NoError(t, b.RegisterWorkerUpdateHandler(framework.WorkerChangeHandler{
		OnPrepare: func(string) error {
			terminating := pod.DeepCopy()
			now := metav1.Now()
			terminating.DeletionTimestamp = &now
			b.podCacher.onPodAdd(terminating)
			return nil
		},
		OnRemove: func(info *api.WorkerInfo) { _ = b.allocationController.DeallocateWorker(info.WorkerUID) },
	}))
	_, err = b.allocateWorkerDevices(t.Context(), info)
	require.ErrorContains(t, err, "terminating")
	_, exists := b.allocationController.GetWorkerAllocation(info.WorkerUID)
	require.True(t, exists, "TerminationGracePeriod must not make GPU ownership available early")
	b.podCacher.onPodDelete(pod)
	require.Eventually(t, func() bool {
		_, exists := b.allocationController.GetWorkerAllocation(info.WorkerUID)
		return !exists
	}, time.Second, time.Millisecond)
}
