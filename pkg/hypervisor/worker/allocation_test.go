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

func TestAllocateWorkerDevicesPinsMthreadsVisibleDevicesByIndex(t *testing.T) {
	t.Parallel()

	controller := NewAllocationController(&fakeDeviceController{
		devices: map[string]*api.DeviceInfo{
			"mt-1": {
				UUID:       "gpu-mt-bbb",
				Vendor:     constants.AcceleratorVendorMThreads,
				Index:      1,
				DeviceNode: map[string]string{"/dev/mtgpu.1": "/dev/mtgpu.1"},
			},
			"mt-0": {
				UUID:       "gpu-mt-aaa",
				Vendor:     constants.AcceleratorVendorMThreads,
				Index:      0,
				DeviceNode: map[string]string{"/dev/mtgpu.0": "/dev/mtgpu.0"},
			},
		},
	})

	allocation, err := controller.AllocateWorkerDevices(&api.WorkerInfo{
		WorkerUID:        "worker-mthreads-shared",
		AllocatedDevices: []string{"mt-1", "mt-0"},
		IsolationMode:    tfv1.IsolationModeShared,
	})
	if err != nil {
		t.Fatalf("allocate worker devices: %v", err)
	}
	if got := allocation.Envs[constants.MthreadsVisibleDevicesEnv]; got != "0,1" {
		t.Fatalf("unexpected %s: %q", constants.MthreadsVisibleDevicesEnv, got)
	}
	if _, exists := allocation.Envs[constants.NvidiaVisibleAllDeviceEnv]; exists {
		t.Fatalf("did not expect %s for MThreads vendor", constants.NvidiaVisibleAllDeviceEnv)
	}
}

func TestAllocateWorkerDevicesPinsAscendVisibleDevicesByIndex(t *testing.T) {
	t.Parallel()

	controller := NewAllocationController(&fakeDeviceController{
		devices: map[string]*api.DeviceInfo{
			"npu-0": {
				UUID:       "npu-aaa",
				Vendor:     constants.AcceleratorVendorHuaweiAscendNPU,
				Index:      0,
				DeviceNode: map[string]string{"/dev/davinci0": "/dev/davinci0"},
			},
		},
	})

	allocation, err := controller.AllocateWorkerDevices(&api.WorkerInfo{
		WorkerUID:        "worker-ascend-shared",
		AllocatedDevices: []string{"npu-0"},
		IsolationMode:    tfv1.IsolationModeSoft,
	})
	if err != nil {
		t.Fatalf("allocate worker devices: %v", err)
	}
	if got := allocation.Envs[constants.AscendVisibleDevicesEnv]; got != "0" {
		t.Fatalf("unexpected %s: %q", constants.AscendVisibleDevicesEnv, got)
	}
}

func TestAllocateWorkerDevicesPinsPpuVisibleDevicesByIndex(t *testing.T) {
	t.Parallel()

	controller := NewAllocationController(&fakeDeviceController{
		devices: map[string]*api.DeviceInfo{
			"ppu-1": {
				UUID:       "ppu-GPU-bbb",
				Vendor:     constants.AcceleratorVendorAlibabaPPU,
				Index:      1,
				DeviceNode: map[string]string{"/dev/alixpu_ppu1": "/dev/alixpu_ppu1"},
			},
			"ppu-0": {
				UUID:       "ppu-GPU-aaa",
				Vendor:     constants.AcceleratorVendorAlibabaPPU,
				Index:      0,
				DeviceNode: map[string]string{"/dev/alixpu_ppu0": "/dev/alixpu_ppu0"},
			},
		},
	})

	allocation, err := controller.AllocateWorkerDevices(&api.WorkerInfo{
		WorkerUID:        "worker-ppu-soft",
		AllocatedDevices: []string{"ppu-1", "ppu-0"},
		IsolationMode:    tfv1.IsolationModeSoft,
	})
	if err != nil {
		t.Fatalf("allocate worker devices: %v", err)
	}
	if got := allocation.Envs[constants.PpuVisibleDevicesEnv]; got != "0,1" {
		t.Fatalf("unexpected %s: %q", constants.PpuVisibleDevicesEnv, got)
	}
	if _, exists := allocation.Envs[constants.NvidiaVisibleAllDeviceEnv]; exists {
		t.Fatalf("did not expect %s for PPU vendor", constants.NvidiaVisibleAllDeviceEnv)
	}
}

func TestAllocateWorkerDevicesDoesNotPinMthreadsVisibleDevicesForPartitionedMode(t *testing.T) {
	t.Parallel()

	// Partitioned mode: AssignPartition populates DeviceEnv with the
	// partition-scoped MTHREADS_VISIBLE_DEVICES. The canonicalize block must
	// not overwrite it.
	controller := NewAllocationController(&fakeDeviceController{
		devices: map[string]*api.DeviceInfo{
			"mt-0": {
				UUID:   "gpu-mt-aaa",
				Vendor: constants.AcceleratorVendorMThreads,
				Index:  0,
				// SplitDevice in the fake controller is what populates DeviceEnv
				// in real flows; here we just confirm the canonicalize branch
				// is skipped for partitioned mode.
				DeviceNode: map[string]string{"/dev/mtgpu.0": "/dev/mtgpu.0"},
			},
		},
	})

	allocation, err := controller.AllocateWorkerDevices(&api.WorkerInfo{
		WorkerUID:           "worker-mthreads-partitioned",
		AllocatedDevices:    []string{"mt-0"},
		IsolationMode:       tfv1.IsolationModePartitioned,
		PartitionTemplateID: "musa-1g-4gb",
	})
	if err != nil {
		t.Fatalf("allocate worker devices: %v", err)
	}
	if _, exists := allocation.Envs[constants.MthreadsVisibleDevicesEnv]; exists {
		t.Fatalf("did not expect %s for partitioned mode", constants.MthreadsVisibleDevicesEnv)
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
