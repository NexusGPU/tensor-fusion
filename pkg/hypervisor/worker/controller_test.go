package worker

import (
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
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
	allocations       map[string]*api.WorkerAllocation
	allocatedRequests []*api.WorkerInfo
	recoveredRequests []*api.WorkerInfo
}

func (f *fakeWorkerAllocationController) AllocateWorkerDevices(request *api.WorkerInfo) (*api.WorkerAllocation, error) {
	f.allocatedRequests = append(f.allocatedRequests, request)
	allocation := &api.WorkerAllocation{WorkerInfo: request}
	if f.allocations == nil {
		f.allocations = make(map[string]*api.WorkerAllocation)
	}
	f.allocations[request.WorkerUID] = allocation
	return allocation, nil
}

func (f *fakeWorkerAllocationController) DeallocateWorker(string) error { return nil }

func (f *fakeWorkerAllocationController) RecoverPartitionedWorker(
	request *api.WorkerInfo, partitionUUIDs string,
) error {
	f.recoveredRequests = append(f.recoveredRequests, request)
	return nil
}

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

func TestBuildWorkerInfoSnapshotsOnlyIncludesSharedMemoryWorkers(t *testing.T) {
	soft := &api.WorkerInfo{WorkerUID: "soft", IsolationMode: tfv1.IsolationModeSoft}
	shared := &api.WorkerInfo{WorkerUID: "shared", IsolationMode: tfv1.IsolationModeShared}
	controller := &WorkerController{
		workers: map[string]*api.WorkerInfo{
			soft.WorkerUID:   soft,
			shared.WorkerUID: shared,
		},
		allocationController: &fakeWorkerAllocationController{allocations: map[string]*api.WorkerAllocation{
			soft.WorkerUID: {
				WorkerInfo:  soft,
				DeviceInfos: []*api.DeviceInfo{{UUID: "gpu-soft", Index: 0}},
			},
			shared.WorkerUID: {
				WorkerInfo:  shared,
				DeviceInfos: []*api.DeviceInfo{{UUID: "gpu-shared", Index: 2}},
			},
		}},
	}

	snapshots := controller.buildWorkerInfoSnapshots()
	if _, ok := snapshots[soft.WorkerUID]; !ok {
		t.Fatal("soft worker should be included in ERL snapshots")
	}
	if _, ok := snapshots[shared.WorkerUID]; ok {
		t.Fatal("shared worker must not be included in ERL snapshots")
	}
}

func TestRecoverExistingWorkerAllocation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name            string
		worker          *api.WorkerInfo
		wantAllocated   bool
		wantPartitioned bool
	}{
		{
			name: "running soft worker",
			worker: &api.WorkerInfo{
				WorkerUID: "soft", Status: api.WorkerStatusRunning,
				IsolationMode: tfv1.IsolationModeSoft, AllocatedDevices: []string{"gpu-0"},
			},
			wantAllocated: true,
		},
		{
			name: "running shared worker",
			worker: &api.WorkerInfo{
				WorkerUID: "shared", Status: api.WorkerStatusRunning,
				IsolationMode: tfv1.IsolationModeShared, AllocatedDevices: []string{"gpu-0"},
			},
			wantAllocated: true,
		},
		{
			name: "running hard worker",
			worker: &api.WorkerInfo{
				WorkerUID: "hard", Status: api.WorkerStatusRunning,
				IsolationMode: tfv1.IsolationModeHard, AllocatedDevices: []string{"gpu-0"},
			},
			wantAllocated: true,
		},
		{
			name: "pending worker is allocated by device plugin",
			worker: &api.WorkerInfo{
				WorkerUID: "pending", Status: api.WorkerStatusDeviceAllocating,
				IsolationMode: tfv1.IsolationModeSoft, AllocatedDevices: []string{"gpu-0"},
			},
		},
		{
			name: "running partitioned worker uses partition recovery",
			worker: &api.WorkerInfo{
				WorkerUID: "partitioned", Status: api.WorkerStatusRunning,
				IsolationMode: tfv1.IsolationModePartitioned, AllocatedDevices: []string{"gpu-0"},
				PartitionTemplateID: "1g.10gb",
				Annotations:         map[string]string{constants.PartitionUUIDsAnnotation: "mig-0:gpu-0"},
			},
			wantPartitioned: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			allocations := &fakeWorkerAllocationController{}
			controller := &WorkerController{allocationController: allocations}
			controller.recoverExistingWorkerAllocation(tt.worker)

			if got := len(allocations.allocatedRequests) == 1; got != tt.wantAllocated {
				t.Fatalf("regular allocation recovery = %v, want %v", got, tt.wantAllocated)
			}
			if got := len(allocations.recoveredRequests) == 1; got != tt.wantPartitioned {
				t.Fatalf("partition allocation recovery = %v, want %v", got, tt.wantPartitioned)
			}
		})
	}
}

func TestUsesSoftLimiterSharedMemory(t *testing.T) {
	t.Parallel()

	tests := []struct {
		mode tfv1.IsolationModeType
		want bool
	}{
		{mode: tfv1.IsolationModeSoft, want: true},
		{mode: tfv1.IsolationModeHard, want: false},
		{mode: tfv1.IsolationModeShared, want: false},
		{mode: tfv1.IsolationModePartitioned, want: false},
	}
	for _, tt := range tests {
		if got := usesSoftLimiterSharedMemory(tt.mode); got != tt.want {
			t.Fatalf("usesSoftLimiterSharedMemory(%q) = %v, want %v", tt.mode, got, tt.want)
		}
	}
}

func TestRecoverExistingSoftWorkerSharedMemory(t *testing.T) {
	t.Parallel()

	const (
		namespace = "tensor-fusion-sys"
		podName   = "soft-worker"
		workerUID = "soft-worker-uid"
	)
	basePath := t.TempDir()
	podID := workerstate.NewPodIdentifier(namespace, podName)
	original, err := workerstate.CreateSharedMemoryHandle(basePath, podID, []workerstate.DeviceConfig{
		{DeviceIdx: 0, DeviceUUID: "GPU-0", UpLimit: 20, MemLimit: 1 << 30},
	})
	if err != nil {
		t.Fatalf("create original shared memory: %v", err)
	}
	original.GetState().UpdateHeartbeat(123)
	if err := original.Close(); err != nil {
		t.Fatalf("close original shared memory: %v", err)
	}

	workerInfo := &api.WorkerInfo{
		WorkerUID: workerUID, Namespace: namespace, WorkerName: podName,
		Status: api.WorkerStatusRunning, IsolationMode: tfv1.IsolationModeSoft,
		AllocatedDevices: []string{"gpu-0"},
	}
	controller := &WorkerController{
		allocationController: &fakeWorkerAllocationController{allocations: map[string]*api.WorkerAllocation{
			workerUID: {
				WorkerInfo:  workerInfo,
				DeviceInfos: []*api.DeviceInfo{{UUID: "GPU-0", Index: 0, TotalMemoryBytes: 24 << 30}},
			},
		}},
		shmBasePath: basePath,
		shmHandles:  make(map[string]*workerstate.SharedMemoryHandle),
	}
	controller.ensureWorkerSharedMemory(workerInfo, true)

	var recovered *workerstate.SharedMemoryHandle
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		controller.mu.RLock()
		recovered = controller.shmHandles[workerUID]
		controller.mu.RUnlock()
		if recovered != nil {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if recovered == nil {
		t.Fatal("soft worker shared memory was not recovered")
	}
	defer func() { _ = recovered.Close() }()
	if got := recovered.GetState().GetLastHeartbeat(); got != 123 {
		t.Fatalf("recovery recreated shared memory: heartbeat = %d, want 123", got)
	}
}
