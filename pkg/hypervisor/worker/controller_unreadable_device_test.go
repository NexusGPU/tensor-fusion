package worker

import (
	"sync/atomic"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/stretchr/testify/require"
)

// A device whose process list could not be read in one sample says nothing
// about its processes. What its workers were last seen holding stays, their
// utilization is not taken for zero, and everything is accounted again with
// the next sample that can be read.
func TestUnreadableProcessListKeepsTheAccountingOfItsDevice(t *testing.T) {
	const namespace = "ns"
	shmBasePath := t.TempDir()
	handle, err := workerstate.CreateSharedMemoryHandle(shmBasePath, workerstate.NewPodIdentifier(namespace, "pod-a"),
		[]workerstate.DeviceConfig{
			{DeviceIdx: 0, DeviceUUID: "GPU-0", UpLimit: 50, MemLimit: 10 << 30},
			{DeviceIdx: 1, DeviceUUID: "GPU-1", UpLimit: 50, MemLimit: 10 << 30},
		})
	require.NoError(t, err)
	t.Cleanup(func() { _ = handle.Close() })

	worker := &api.WorkerInfo{
		WorkerUID: "uid-a", Namespace: namespace, WorkerName: "pod-a", IsolationMode: tfv1.IsolationModeHard,
	}
	now := time.Unix(1_710_000_000, 0)
	devices := &fakeDeviceController{}
	controller := &WorkerController{
		backend: &fakeWorkerBackend{mappings: map[uint32]*framework.ProcessMappingInfo{
			101: {Namespace: namespace, PodName: "pod-a"},
			102: {Namespace: namespace, PodName: "pod-a"},
			900: {Namespace: "other", PodName: "not-a-worker"},
		}},
		deviceController: devices,
		allocationController: &fakeWorkerAllocationController{allocations: map[string]*api.WorkerAllocation{
			"uid-a": {WorkerInfo: worker, DeviceInfos: []*api.DeviceInfo{{UUID: "gpu-0", Index: 0}, {UUID: "gpu-1", Index: 1}}},
		}},
		workers:     map[string]*api.WorkerInfo{"uid-a": worker},
		shmBasePath: shmBasePath,
		shmHandles:  map[string]*workerstate.SharedMemoryHandle{"uid-a": handle},
		nowFunc:     func() time.Time { return now },
	}
	memory := func() (gpu0, gpu1 uint64) {
		handle.WithState(func(state *workerstate.SharedDeviceState) {
			gpu0 = atomic.LoadUint64(&state.V2.Devices[0].DeviceInfo.PodMemoryUsed)
			gpu1 = atomic.LoadUint64(&state.V2.Devices[1].DeviceInfo.PodMemoryUsed)
		})
		return gpu0, gpu1
	}
	sync := func(rows ...api.ProcessInformation) {
		devices.processInfos = rows
		now = now.Add(sharedMemorySyncInterval)
		controller.syncSharedMemoryState()
	}

	process := func(pid, device string, memory uint64, percent float64) api.ProcessInformation {
		return api.ProcessInformation{
			ProcessID: pid, DeviceUUID: device, MemoryUsedBytes: memory, ComputeUtilizationPercent: percent,
		}
	}
	sync(
		process("101", "GPU-0", 1<<30, 20),
		process("102", "GPU-1", 2<<30, 30),
		process("900", "GPU-1", 1<<20, 5),
	)
	gpu0, gpu1 := memory()
	require.Equal(t, uint64(1<<30), gpu0)
	require.Equal(t, uint64(2<<30), gpu1)
	require.True(t, controller.processUtilization.Load().Unmetered["gpu-1"])

	// GPU-1 cannot be read; GPU-0 can, and its process grew.
	sync(
		process("101", "GPU-0", 3<<30, 25),
		api.ProcessInformation{DeviceUUID: "GPU-1", ComputeUtilizationUnavailable: true, ProcessListUnavailable: true},
	)
	gpu0, gpu1 = memory()
	require.Equal(t, uint64(3<<30), gpu0, "a readable device is accounted as usual")
	require.Equal(t, uint64(2<<30), gpu1, "memory on a device that could not be read is neither zeroed nor guessed")
	snapshot := controller.processUtilization.Load()
	require.Equal(t, map[string]map[string]float64{"gpu-0": {"uid-a": 25}}, snapshot.Devices,
		"the unreadable device has no per-process utilization in this sample, not a zero")
	require.True(t, snapshot.Unmetered["gpu-1"], "what ran unmetered there is still assumed to")
	require.True(t, controller.unreadableDevices["gpu-1"])

	// The marker is not a process: no metrics row is made of it, and the
	// workers of the device have none in this sample.
	metrics, err := controller.GetWorkerMetrics()
	require.NoError(t, err)
	require.Contains(t, metrics, "gpu-0")
	require.NotContains(t, metrics, "gpu-1")

	// Readable again and the process is gone: now the memory is released.
	sync(process("101", "GPU-0", 3<<30, 25))
	gpu0, gpu1 = memory()
	require.Equal(t, uint64(3<<30), gpu0)
	require.Zero(t, gpu1)
	require.Empty(t, controller.unreadableDevices)
	require.False(t, controller.processUtilization.Load().Unmetered["gpu-1"])
}

// Keeping the previous accounting is for the workers allocated on the device
// that cannot be read, and for a bounded time. A worker on a readable device
// is accounted from what is seen, whatever happens to other devices; and a
// device that stays unreadable is accounted as holding nothing once the bound
// has passed, until it can be read again.
func TestUnreadableProcessListAffectsOnlyItsOwnDeviceAndExpires(t *testing.T) {
	const namespace = "ns"
	shmBasePath := t.TempDir()
	open := func(pod, device string) *workerstate.SharedMemoryHandle {
		handle, err := workerstate.CreateSharedMemoryHandle(shmBasePath, workerstate.NewPodIdentifier(namespace, pod),
			[]workerstate.DeviceConfig{{DeviceIdx: 0, DeviceUUID: device, UpLimit: 50, MemLimit: 10 << 30}})
		require.NoError(t, err)
		t.Cleanup(func() { _ = handle.Close() })
		return handle
	}
	healthy, broken := open("pod-a", "GPU-0"), open("pod-b", "GPU-1")
	workerA := &api.WorkerInfo{
		WorkerUID: "uid-a", Namespace: namespace, WorkerName: "pod-a", IsolationMode: tfv1.IsolationModeHard,
	}
	workerB := &api.WorkerInfo{
		WorkerUID: "uid-b", Namespace: namespace, WorkerName: "pod-b", IsolationMode: tfv1.IsolationModeHard,
	}
	now := time.Unix(1_710_000_000, 0)
	devices := &fakeDeviceController{}
	controller := &WorkerController{
		backend: &fakeWorkerBackend{mappings: map[uint32]*framework.ProcessMappingInfo{
			101: {Namespace: namespace, PodName: "pod-a"},
			201: {Namespace: namespace, PodName: "pod-b"},
		}},
		deviceController: devices,
		allocationController: &fakeWorkerAllocationController{allocations: map[string]*api.WorkerAllocation{
			"uid-a": {WorkerInfo: workerA, DeviceInfos: []*api.DeviceInfo{{UUID: "gpu-0", Index: 0}}},
			"uid-b": {WorkerInfo: workerB, DeviceInfos: []*api.DeviceInfo{{UUID: "gpu-1", Index: 0}}},
		}},
		workers:     map[string]*api.WorkerInfo{"uid-a": workerA, "uid-b": workerB},
		shmBasePath: shmBasePath,
		shmHandles:  map[string]*workerstate.SharedMemoryHandle{"uid-a": healthy, "uid-b": broken},
		nowFunc:     func() time.Time { return now },
	}
	memory := func(handle *workerstate.SharedMemoryHandle) (used uint64) {
		handle.WithState(func(state *workerstate.SharedDeviceState) {
			used = atomic.LoadUint64(&state.V2.Devices[0].DeviceInfo.PodMemoryUsed)
		})
		return used
	}
	unreadable := api.ProcessInformation{
		DeviceUUID: "GPU-1", ComputeUtilizationUnavailable: true, ProcessListUnavailable: true,
	}
	sync := func(after time.Duration, rows ...api.ProcessInformation) {
		devices.processInfos = rows
		now = now.Add(after)
		controller.syncSharedMemoryState()
	}

	sync(sharedMemorySyncInterval,
		api.ProcessInformation{ProcessID: "101", DeviceUUID: "GPU-0", MemoryUsedBytes: 9 << 30},
		api.ProcessInformation{ProcessID: "201", DeviceUUID: "GPU-1", MemoryUsedBytes: 4 << 30})
	require.Equal(t, uint64(9<<30), memory(healthy))
	require.Equal(t, uint64(4<<30), memory(broken))

	// The other device becomes unreadable and the process on the healthy one
	// exits: its worker's figure follows in the same pass.
	sync(sharedMemorySyncInterval, unreadable)
	require.Zero(t, memory(healthy), "a worker on a readable device is not held up by another device")
	require.Equal(t, uint64(4<<30), memory(broken), "the worker on the unreadable device keeps its figure")

	// Still within the bound.
	sync(unreadableMemoryMaxAge-time.Second, unreadable)
	require.Equal(t, uint64(4<<30), memory(broken))
	require.False(t, controller.unreadableExpired["gpu-1"])

	// Past the bound: nothing is accounted on the device any more.
	sync(2*time.Second, unreadable)
	require.Zero(t, memory(broken))
	require.True(t, controller.unreadableExpired["gpu-1"])

	// Readable again: accounted from what is seen, and the bound starts anew
	// the next time.
	sync(sharedMemorySyncInterval, api.ProcessInformation{ProcessID: "201", DeviceUUID: "GPU-1", MemoryUsedBytes: 2 << 30})
	require.Equal(t, uint64(2<<30), memory(broken))
	require.Empty(t, controller.unreadableSince)
	sync(sharedMemorySyncInterval, unreadable)
	require.Equal(t, uint64(2<<30), memory(broken))
}

// The provider is asked for the process list by the sync loop only, once per
// pass. Neither the heartbeat nor an on-demand reader within the sample's
// lifetime asks again: the call reads the driver's utilization samples and
// takes milliseconds per device.
func TestProcessListIsReadOncePerSyncPass(t *testing.T) {
	now := time.Unix(1_710_000_000, 0)
	devices := &fakeDeviceController{processInfos: []api.ProcessInformation{{ProcessID: "101", DeviceUUID: "GPU-0"}}}
	worker := &api.WorkerInfo{
		WorkerUID: "uid-a", Namespace: "ns", WorkerName: "pod-a", IsolationMode: tfv1.IsolationModeSoft,
	}
	controller := &WorkerController{
		backend: &fakeWorkerBackend{mappings: map[uint32]*framework.ProcessMappingInfo{
			101: {Namespace: "ns", PodName: "pod-a"},
		}},
		deviceController: devices,
		allocationController: &fakeWorkerAllocationController{allocations: map[string]*api.WorkerAllocation{
			"uid-a": {WorkerInfo: worker, DeviceInfos: []*api.DeviceInfo{{UUID: "gpu-0", Index: 0}}},
		}},
		workers:     map[string]*api.WorkerInfo{"uid-a": worker},
		shmBasePath: t.TempDir(),
		shmHandles:  map[string]*workerstate.SharedMemoryHandle{},
		nowFunc:     func() time.Time { return now },
	}

	for pass := 1; pass <= 3; pass++ {
		controller.syncSharedMemoryState()
		require.Equal(t, pass, devices.processCalls)
		// Readers between two passes share the sample of the pass.
		for range 5 {
			_, err := controller.GetWorkerMetrics()
			require.NoError(t, err)
			controller.writeHeartbeats()
		}
		require.Equal(t, pass, devices.processCalls)
		now = now.Add(sharedMemorySyncInterval)
	}
}
