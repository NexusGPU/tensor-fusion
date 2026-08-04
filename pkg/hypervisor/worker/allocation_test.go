package worker

import (
	"strings"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"k8s.io/apimachinery/pkg/api/resource"
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

func TestAllocateWorkerDevicesSetsHardSMPercentLimit(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		isolation   tfv1.IsolationModeType
		limit       tfv1.Resource
		annotations map[string]string
		want        string
		wantEnv     bool
	}{
		{
			name:      "compute percent is preserved",
			isolation: tfv1.IsolationModeHard,
			limit: tfv1.Resource{
				ComputePercent: resource.MustParse("30"),
			},
			want:    "30",
			wantEnv: true,
		},
		{
			name:      "absolute tflops is converted using selected GPU",
			isolation: tfv1.IsolationModeHard,
			limit: tfv1.Resource{
				Tflops: resource.MustParse("10"),
			},
			want:    "25",
			wantEnv: true,
		},
		{
			name:      "scheduler capacity conversion overrides device discovery",
			isolation: tfv1.IsolationModeHard,
			limit: tfv1.Resource{
				Tflops: resource.MustParse("10"),
			},
			annotations: map[string]string{
				constants.EffectiveHardSMPercentAnnotation: "15",
			},
			want:    "15",
			wantEnv: true,
		},
		{
			name:      "soft allocation does not set hard limiter",
			isolation: tfv1.IsolationModeSoft,
			limit: tfv1.Resource{
				Tflops: resource.MustParse("10"),
			},
			wantEnv: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			controller := NewAllocationController(&fakeDeviceController{
				devices: map[string]*api.DeviceInfo{
					"gpu-0": {
						UUID:      "GPU-aaa",
						Vendor:    constants.AcceleratorVendorNvidia,
						MaxTflops: 40,
					},
				},
			})
			allocation, err := controller.AllocateWorkerDevices(&api.WorkerInfo{
				WorkerUID:        "worker-hard-limit-" + strings.ReplaceAll(tt.name, " ", "-"),
				AllocatedDevices: []string{"gpu-0"},
				IsolationMode:    tt.isolation,
				Limits:           tt.limit,
				Annotations:      tt.annotations,
			})
			if err != nil {
				t.Fatalf("allocate worker devices: %v", err)
			}
			got, exists := allocation.Envs[constants.HardSMLimiterEnv]
			if exists != tt.wantEnv {
				t.Fatalf("%s presence = %v, want %v", constants.HardSMLimiterEnv, exists, tt.wantEnv)
			}
			if got != tt.want {
				t.Fatalf("%s = %q, want %q", constants.HardSMLimiterEnv, got, tt.want)
			}
		})
	}
}

func TestAllocateWorkerDevicesPinsIndexBasedVisibleDevices(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		devices       map[string]*api.DeviceInfo
		workerUID     string
		allocated     []string
		isolationMode tfv1.IsolationModeType
		wantEnvs      map[string]string
	}{
		{
			name: "MThreads pins hook and runtime filter envs by index",
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
			workerUID:     "worker-mthreads-shared",
			allocated:     []string{"mt-1", "mt-0"},
			isolationMode: tfv1.IsolationModeShared,
			wantEnvs: map[string]string{
				constants.MthreadsVisibleDevicesEnv: "0,1",
				constants.MusaVisibleDevicesEnv:     "0,1",
			},
		},
		{
			name: "Ascend pins hook env by index",
			devices: map[string]*api.DeviceInfo{
				"npu-0": {
					UUID:       "npu-aaa",
					Vendor:     constants.AcceleratorVendorHuaweiAscendNPU,
					Index:      0,
					DeviceNode: map[string]string{"/dev/davinci0": "/dev/davinci0"},
				},
			},
			workerUID:     "worker-ascend-shared",
			allocated:     []string{"npu-0"},
			isolationMode: tfv1.IsolationModeSoft,
			wantEnvs: map[string]string{
				constants.AscendVisibleDevicesEnv: "0",
			},
		},
		{
			name: "PPU pins hook and CUDA runtime filter envs by index",
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
			workerUID:     "worker-ppu-soft",
			allocated:     []string{"ppu-1", "ppu-0"},
			isolationMode: tfv1.IsolationModeSoft,
			wantEnvs: map[string]string{
				constants.PpuVisibleDevicesEnv:  "0,1",
				constants.CudaVisibleDevicesEnv: "0,1",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			controller := NewAllocationController(&fakeDeviceController{devices: tt.devices})
			allocation, err := controller.AllocateWorkerDevices(&api.WorkerInfo{
				WorkerUID:        tt.workerUID,
				AllocatedDevices: tt.allocated,
				IsolationMode:    tt.isolationMode,
			})
			if err != nil {
				t.Fatalf("allocate worker devices: %v", err)
			}
			for envName, want := range tt.wantEnvs {
				if got := allocation.Envs[envName]; got != want {
					t.Fatalf("unexpected %s: %q, want %q", envName, got, want)
				}
			}
			if _, exists := allocation.Envs[constants.NvidiaVisibleAllDeviceEnv]; exists {
				t.Fatalf("did not expect %s for non-NVIDIA vendor", constants.NvidiaVisibleAllDeviceEnv)
			}
		})
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
