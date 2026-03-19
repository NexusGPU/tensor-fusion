package worker

import (
	"strings"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
)

func TestAllocateWorkerDevicesRejectsMissingAllocatedDevices(t *testing.T) {
	t.Parallel()

	controller := NewAllocationController(&fakeDeviceController{})

	allocation, err := controller.AllocateWorkerDevices(&api.WorkerInfo{
		WorkerUID:     "worker-without-gpu",
		IsolationMode: tfv1.IsolationModeHard,
	})
	if err == nil {
		t.Fatal("expected allocation error for worker without allocated devices")
	}
	if allocation != nil {
		t.Fatalf("expected nil allocation, got %#v", allocation)
	}
	if !strings.Contains(err.Error(), "no allocated devices") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestAllocateWorkerDevicesPinsNvidiaVisibleDevicesByUUID(t *testing.T) {
	t.Parallel()

	controller := NewAllocationController(&fakeDeviceController{
		devices: map[string]*api.DeviceInfo{
			"gpu-1": {
				UUID:       "GPU-bbb",
				Vendor:     constants.AcceleratorVendorNvidia,
				Index:      1,
				DeviceNode: map[string]string{"/dev/nvidia1": "/dev/nvidia1"},
			},
			"gpu-0": {
				UUID:       "GPU-aaa",
				Vendor:     constants.AcceleratorVendorNvidia,
				Index:      0,
				DeviceNode: map[string]string{"/dev/nvidia0": "/dev/nvidia0"},
			},
		},
	})

	allocation, err := controller.AllocateWorkerDevices(&api.WorkerInfo{
		WorkerUID:        "worker-nvidia-shared",
		AllocatedDevices: []string{"gpu-1", "gpu-0"},
		IsolationMode:    tfv1.IsolationModeShared,
	})
	if err != nil {
		t.Fatalf("allocate worker devices: %v", err)
	}
	if allocation.Envs[constants.NvidiaVisibleAllDeviceEnv] != "GPU-aaa,GPU-bbb" {
		t.Fatalf(
			"unexpected %s: %q",
			constants.NvidiaVisibleAllDeviceEnv,
			allocation.Envs[constants.NvidiaVisibleAllDeviceEnv],
		)
	}
}

func TestAllocateWorkerDevicesDoesNotPinNvidiaVisibleDevicesForPartitionedMode(t *testing.T) {
	t.Parallel()

	controller := NewAllocationController(&fakeDeviceController{
		devices: map[string]*api.DeviceInfo{
			"gpu-0": {
				UUID:       "GPU-aaa",
				Vendor:     constants.AcceleratorVendorNvidia,
				Index:      0,
				DeviceNode: map[string]string{"/dev/nvidia0": "/dev/nvidia0"},
			},
		},
	})

	allocation, err := controller.AllocateWorkerDevices(&api.WorkerInfo{
		WorkerUID:           "worker-nvidia-partitioned",
		AllocatedDevices:    []string{"gpu-0"},
		IsolationMode:       tfv1.IsolationModePartitioned,
		PartitionTemplateID: "mig-1g-10gb",
	})
	if err != nil {
		t.Fatalf("allocate worker devices: %v", err)
	}
	if _, exists := allocation.Envs[constants.NvidiaVisibleAllDeviceEnv]; exists {
		t.Fatalf("did not expect %s for partitioned mode", constants.NvidiaVisibleAllDeviceEnv)
	}
}
