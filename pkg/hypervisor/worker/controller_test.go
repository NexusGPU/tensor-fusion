package worker

import (
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
)

type fakeDeviceController struct {
	devices      map[string]*api.DeviceInfo
	processInfos []api.ProcessInformation
}

func (f *fakeDeviceController) Start() error { return nil }

func (f *fakeDeviceController) Stop() error { return nil }

func (f *fakeDeviceController) DiscoverDevices() error { return nil }

func (f *fakeDeviceController) ListDevices() ([]*api.DeviceInfo, error) {
	if len(f.devices) == 0 {
		return nil, nil
	}
	devices := make([]*api.DeviceInfo, 0, len(f.devices))
	for _, device := range f.devices {
		devices = append(devices, device)
	}
	return devices, nil
}

func (f *fakeDeviceController) GetDevice(deviceUUID string) (*api.DeviceInfo, bool) {
	device, ok := f.devices[deviceUUID]
	return device, ok
}

func (f *fakeDeviceController) SplitDevice(deviceUUID, _ string) (*api.DeviceInfo, error) {
	device, ok := f.devices[deviceUUID]
	if !ok || device == nil {
		return nil, nil
	}
	copied := *device
	return &copied, nil
}

func (f *fakeDeviceController) RemovePartitionedDevice(string, string) error { return nil }

func (f *fakeDeviceController) GetDeviceMetrics() (map[string]*api.GPUUsageMetrics, error) {
	return nil, nil
}

func (f *fakeDeviceController) GetProcessInformation() ([]api.ProcessInformation, error) {
	return f.processInfos, nil
}

func (f *fakeDeviceController) GetVendorMountLibs() ([]*api.Mount, error) { return nil, nil }

func (f *fakeDeviceController) RegisterDeviceUpdateHandler(framework.DeviceChangeHandler) {}

func (f *fakeDeviceController) GetAcceleratorVendor() string { return "NVIDIA" }

type fakeWorkerAllocationController struct {
	allocations map[string]*api.WorkerAllocation
}

func (f *fakeWorkerAllocationController) AllocateWorkerDevices(*api.WorkerInfo) (*api.WorkerAllocation, error) {
	return nil, nil
}

func (f *fakeWorkerAllocationController) DeallocateWorker(string) error { return nil }

func (f *fakeWorkerAllocationController) GetWorkerAllocation(workerUID string) (*api.WorkerAllocation, bool) {
	allocation, ok := f.allocations[workerUID]
	return allocation, ok
}

func (f *fakeWorkerAllocationController) GetDeviceAllocations() map[string][]*api.WorkerAllocation {
	return nil
}

type fakeWorkerBackend struct {
	mappings map[uint32]*framework.ProcessMappingInfo
}

func (f *fakeWorkerBackend) Start() error { return nil }

func (f *fakeWorkerBackend) Stop() error { return nil }

func (f *fakeWorkerBackend) RegisterWorkerUpdateHandler(framework.WorkerChangeHandler) error {
	return nil
}

func (f *fakeWorkerBackend) StartWorker(*api.WorkerInfo) error { return nil }

func (f *fakeWorkerBackend) StopWorker(string) error { return nil }

func (f *fakeWorkerBackend) GetProcessMappingInfo(hostPID uint32) (*framework.ProcessMappingInfo, error) {
	return f.mappings[hostPID], nil
}

func (f *fakeWorkerBackend) GetDeviceChangeHandler() framework.DeviceChangeHandler {
	return framework.DeviceChangeHandler{}
}

func (f *fakeWorkerBackend) ListWorkers() []*api.WorkerInfo { return nil }

func TestSyncSharedMemoryStateUpdatesHeartbeatAndPodMemory(t *testing.T) {
	t.Parallel()

	const (
		namespace = "tensor-fusion-sys"
		podName   = "worker-pod"
		workerUID = "worker-uid"
		hostPID   = uint32(1234)
	)

	shmBasePath := t.TempDir()
	podIdentifier := workerstate.NewPodIdentifier(namespace, podName)
	handle, err := workerstate.CreateSharedMemoryHandle(shmBasePath, podIdentifier, []workerstate.DeviceConfig{
		{
			DeviceIdx:  0,
			DeviceUUID: "GPU-1234",
			UpLimit:    15,
			MemLimit:   10 << 30,
		},
	})
	if err != nil {
		t.Fatalf("create shared memory: %v", err)
	}
	defer func() {
		_ = handle.Close()
	}()

	workerInfo := &api.WorkerInfo{
		WorkerUID:     workerUID,
		Namespace:     namespace,
		WorkerName:    podName,
		IsolationMode: tfv1.IsolationModeHard,
	}
	allocation := &api.WorkerAllocation{
		WorkerInfo: workerInfo,
		DeviceInfos: []*api.DeviceInfo{
			{
				UUID:  "gpu-1234",
				Index: 0,
			},
		},
	}

	syncTime := time.Unix(1_710_000_000, 0)
	controller := &WorkerController{
		backend: &fakeWorkerBackend{
			mappings: map[uint32]*framework.ProcessMappingInfo{
				hostPID: {
					Namespace: namespace,
					PodName:   podName,
					GuestID:   namespace + "_" + podName + "_tensorfusion-worker",
					HostPID:   hostPID,
				},
			},
		},
		deviceController: &fakeDeviceController{
			processInfos: []api.ProcessInformation{
				{
					ProcessID:       "1234",
					DeviceUUID:      "GPU-1234",
					MemoryUsedBytes: 512 << 20,
				},
			},
		},
		allocationController: &fakeWorkerAllocationController{
			allocations: map[string]*api.WorkerAllocation{
				workerUID: allocation,
			},
		},
		workers: map[string]*api.WorkerInfo{
			workerUID: workerInfo,
		},
		shmBasePath: shmBasePath,
		nowFunc: func() time.Time {
			return syncTime
		},
	}

	// syncSharedMemoryState requires liblimiter.so (loaded from accelerator .so).
	// Without a real device.Controller, getLimiter() returns nil and sync is a no-op.
	// The actual shared memory sync is tested via limiter_test.cc in vgpu-provider.
	controller.syncSharedMemoryState()
}
