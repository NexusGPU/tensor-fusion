package worker

import (
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/stretchr/testify/require"
)

func newRecoveryTestController(t *testing.T) (*WorkerController, *fakeDeviceController) {
	t.Helper()
	devices := &fakeDeviceController{devices: map[string]*api.DeviceInfo{
		"gpu-0": {UUID: "gpu-0", TotalMemoryBytes: 4 << 30},
	}}
	w := &WorkerController{
		backend: &fakeWorkerBackend{}, deviceController: devices,
		allocationController: NewAllocationController(devices),
		workers:              map[string]*api.WorkerInfo{}, shmHandles: map[string]*workerstate.SharedMemoryHandle{},
		shmBasePath: t.TempDir(), nowFunc: time.Now,
	}
	t.Cleanup(func() {
		for uid := range w.shmHandles {
			w.closeSharedMemoryLocked(uid)
		}
	})
	return w, devices
}

func TestRecoverCheckpointedPendingWorker(t *testing.T) {
	for _, mode := range []tfv1.IsolationModeType{
		tfv1.IsolationModeSoft, tfv1.IsolationModeHard, tfv1.IsolationModeShared,
	} {
		t.Run(string(mode), func(t *testing.T) {
			w, _ := newRecoveryTestController(t)
			info := &api.WorkerInfo{
				WorkerUID: "checkpointed", Namespace: "ns", WorkerName: "worker", IsolationMode: mode,
				Status: api.WorkerStatusDeviceAllocating, AllocationConfirmed: true,
				AllocatedDevices: []string{"gpu-0"},
			}
			if mode == tfv1.IsolationModeSoft || mode == tfv1.IsolationModeHard {
				// A container may already use a legacy mapping while another
				// container keeps the Pod Pending during the hypervisor upgrade.
				handle, err := workerstate.CreateSharedMemoryHandle(w.shmBasePath,
					workerstate.NewPodIdentifier(info.Namespace, info.WorkerName),
					[]workerstate.DeviceConfig{{DeviceIdx: 0, MemLimit: 4 << 30}})
				require.NoError(t, err)
				handle.GetState().SetPodMemoryUsed(0, 123)
				require.NoError(t, handle.Close())
			}
			w.workerChangeHandler().OnAdd(info)
			allocation, ok := w.allocationController.GetWorkerAllocation(info.WorkerUID)
			require.True(t, ok, "kubelet can reuse its checkpoint without calling Allocate again")
			require.Len(t, allocation.DeviceInfos, 1)
			require.Len(t, w.allocationController.GetDeviceAllocations()["gpu-0"], 1)
			if mode == tfv1.IsolationModeSoft || mode == tfv1.IsolationModeHard {
				handle := w.getShmHandle(info.WorkerUID)
				require.NotNil(t, handle)
				require.Equal(t, uint64(123), handle.GetState().V2.Devices[0].DeviceInfo.PodMemoryUsed)
			}
			w.workerChangeHandler().OnRemove(info)
			_, ok = w.allocationController.GetWorkerAllocation(info.WorkerUID)
			require.False(t, ok)
		})
	}
}

func TestRecoverWorkerWhenRunningUpdateArrives(t *testing.T) {
	w, _ := newRecoveryTestController(t)
	pending := &api.WorkerInfo{
		WorkerUID: "worker", Namespace: "ns", WorkerName: "worker", IsolationMode: tfv1.IsolationModeSoft,
		Status: api.WorkerStatusDeviceAllocating, AllocatedDevices: []string{"gpu-0"},
	}
	handler := w.workerChangeHandler()
	handler.OnAdd(pending)
	_, ok := w.allocationController.GetWorkerAllocation(pending.WorkerUID)
	require.False(t, ok, "a new Pending Pod must wait for device-plugin admission")
	running := *pending
	running.Status = api.WorkerStatusRunning
	handler.OnUpdate(pending, &running)
	w.syncSharedMemoryState()
	_, ok = w.allocationController.GetWorkerAllocation(pending.WorkerUID)
	require.True(t, ok)
	require.NotNil(t, w.getShmHandle(pending.WorkerUID))
}

func TestRecoveryRetriesDeviceDiscoveryWithoutAnotherPodEvent(t *testing.T) {
	w, devices := newRecoveryTestController(t)
	device := devices.devices["gpu-0"]
	delete(devices.devices, "gpu-0")
	info := &api.WorkerInfo{
		WorkerUID: "worker", Namespace: "ns", WorkerName: "worker", IsolationMode: tfv1.IsolationModeSoft,
		Status: api.WorkerStatusRunning, AllocatedDevices: []string{"gpu-0"},
	}
	w.workerChangeHandler().OnAdd(info)
	_, ok := w.allocationController.GetWorkerAllocation(info.WorkerUID)
	require.False(t, ok)
	devices.devices["gpu-0"] = device
	w.syncSharedMemoryState()
	_, ok = w.allocationController.GetWorkerAllocation(info.WorkerUID)
	require.True(t, ok)
	w.syncSharedMemoryState()
	require.Len(t, w.allocationController.GetDeviceAllocations()["gpu-0"], 1, "retries must not double-count the GPU")
	w.workerChangeHandler().OnRemove(info)
	w.recoverExistingWorkerAllocation(info) // A stale sync snapshot must not resurrect the allocation.
	_, ok = w.allocationController.GetWorkerAllocation(info.WorkerUID)
	require.False(t, ok)
}
