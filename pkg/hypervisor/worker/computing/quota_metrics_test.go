package computing

import (
	"errors"
	"math"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
)

type quotaMetricsDeviceController struct {
	framework.DeviceController
	metrics map[string]*api.GPUUsageMetrics
	err     error
}

func (d *quotaMetricsDeviceController) GetDeviceMetrics() (map[string]*api.GPUUsageMetrics, error) {
	return d.metrics, d.err
}

func TestERLPreservesLimitsWithoutValidUtilization(t *testing.T) {
	for _, tc := range []struct {
		name    string
		metrics map[string]*api.GPUUsageMetrics
		err     error
	}{
		{name: "query failed", err: errors.New("driver unavailable")},
		{name: "device absent", metrics: map[string]*api.GPUUsageMetrics{"GPU-other": {ComputePercentage: 0}}},
		{name: "nil sample", metrics: map[string]*api.GPUUsageMetrics{"GPU-test": nil}},
		{name: "negative", metrics: map[string]*api.GPUUsageMetrics{"GPU-test": {ComputePercentage: -1}}},
		{name: "over 100", metrics: map[string]*api.GPUUsageMetrics{"GPU-test": {ComputePercentage: 101}}},
		{name: "NaN", metrics: map[string]*api.GPUUsageMetrics{"GPU-test": {ComputePercentage: math.NaN()}}},
		{name: "infinity", metrics: map[string]*api.GPUUsageMetrics{"GPU-test": {ComputePercentage: math.Inf(1)}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			handle, err := workerstate.CreateSharedMemoryHandle(t.TempDir(),
				&workerstate.PodIdentifier{Namespace: "test", Name: "worker"},
				[]workerstate.DeviceConfig{{DeviceIdx: 0, DeviceUUID: "GPU-test", UpLimit: 50}})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := handle.Close(); err != nil {
					t.Error(err)
				}
			})
			device := &quotaMetricsDeviceController{}
			controller := NewQuotaController(device, nil).(*Controller)
			controller.workerInfoFn = func() map[string]*WorkerInfoSnapshot {
				return map[string]*WorkerInfoSnapshot{"worker": {Devices: []DeviceSnapshot{
					{DeviceUUID: "GPU-test", DeviceIdx: 0, UpLimit: 50},
				}}}
			}
			controller.shmHandleFn = func(string) *workerstate.SharedMemoryHandle { return handle }
			// Establish a valid sample before a prolonged telemetry outage.
			device.metrics = map[string]*api.GPUUsageMetrics{"GPU-test": {ComputePercentage: 50}}
			controller.updateERLControllers()
			before := *controller.erlStates["worker:gpu-test"]
			var rate, capacity float64
			handle.WithState(func(s *workerstate.SharedDeviceState) {
				rate = s.V2.Devices[0].DeviceInfo.GetERLTokenRefillRate()
				capacity = s.V2.Devices[0].DeviceInfo.GetERLTokenCapacity()
			})
			device.metrics, device.err = tc.metrics, tc.err
			for range 20 {
				controller.updateERLControllers()
			}
			if after := *controller.erlStates["worker:gpu-test"]; after != before {
				t.Fatalf("invalid telemetry changed controller: before=%+v after=%+v", before, after)
			}
			handle.WithState(func(s *workerstate.SharedDeviceState) {
				info := &s.V2.Devices[0].DeviceInfo
				if info.GetERLTokenRefillRate() != rate || info.GetERLTokenCapacity() != capacity {
					t.Fatal("invalid telemetry changed shared-memory limits")
				}
			})
			// A genuine idle sample must still permit the normal rate increase.
			device.err = nil
			device.metrics = map[string]*api.GPUUsageMetrics{"GPU-test": {ComputePercentage: 0}}
			controller.updateERLControllers()
			if controller.erlStates["worker:gpu-test"].currentRate <= before.currentRate {
				t.Fatal("valid idle sample did not resume rate control")
			}
		})
	}
}
