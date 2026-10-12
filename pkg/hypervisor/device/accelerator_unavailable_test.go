package device

import (
	"math"
	"testing"
	"unsafe"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/stretchr/testify/require"
)

// The provider marks telemetry it could not obtain in-band, with values outside
// the 0-100 range of the field (provider/accelerator.h). The markers must not
// reach any consumer as numbers.

// A library that returns no markers cannot say which fields of an entry it
// read, so an entry with an unusable utilization is dropped as a whole.
func TestDeviceMetricsOfUnreadableDeviceAreDropped(t *testing.T) {
	old := getDeviceMetrics
	t.Cleanup(func() { getDeviceMetrics = old })
	getDeviceMetrics = func(uuids **byte, count uintptr, out *DeviceMetrics) Result {
		metrics := (*[64]DeviceMetrics)(unsafe.Pointer(out))
		names := []string{"GPU-ok", "GPU-unreadable", "GPU-idle", "GPU-garbage"}
		require.Equal(t, len(names), int(count))
		for i, name := range names {
			metrics[i] = DeviceMetrics{}
			copy(metrics[i].DeviceUUID[:], name)
		}
		metrics[0].UtilizationPercent, metrics[0].MemoryUsedBytes, metrics[0].PowerUsageWatts = 62, 4<<30, 210
		// ACCEL_DEVICE_UTILIZATION_UNAVAILABLE: every other field is zero.
		metrics[1].UtilizationPercent = math.MaxUint32
		// A measured idle device is a sample like any other.
		metrics[2].UtilizationPercent, metrics[2].TemperatureCelsius = 0, 31
		// Outside the contract without being the marker.
		metrics[3].UtilizationPercent = 250
		return ResultSuccess
	}

	accelerator := &AcceleratorInterface{}
	metrics, err := accelerator.GetDeviceMetrics([]string{"GPU-ok", "GPU-unreadable", "GPU-idle", "GPU-garbage"})
	require.NoError(t, err)
	byUUID := map[string]*api.GPUUsageMetrics{}
	for _, m := range metrics {
		byUUID[m.DeviceUUID] = m
	}
	require.Len(t, byUUID, 2, "unreadable devices have no sample at all")
	require.Equal(t, 62.0, byUUID["GPU-ok"].ComputePercentage)
	require.Equal(t, int64(210), byUUID["GPU-ok"].PowerUsage)
	require.Contains(t, byUUID, "GPU-idle")
	require.Zero(t, byUUID["GPU-idle"].ComputePercentage)

	// The device controller keys them by device and has no entry for the rest.
	controller := &Controller{
		accelerator: accelerator,
		devices: map[string]*api.DeviceInfo{
			"gpu-ok": {UUID: "gpu-ok"}, "gpu-unreadable": {UUID: "gpu-unreadable"},
			"gpu-idle": {UUID: "gpu-idle"}, "gpu-garbage": {UUID: "gpu-garbage"},
		},
		nativeUUIDs: map[string]string{
			"gpu-ok": "GPU-ok", "gpu-unreadable": "GPU-unreadable", "gpu-idle": "GPU-idle", "gpu-garbage": "GPU-garbage",
		},
	}
	// Map iteration decides the request order; answer by name instead of position.
	getDeviceMetrics = func(uuids **byte, count uintptr, out *DeviceMetrics) Result {
		requested := unsafe.Slice(uuids, int(count))
		metrics := unsafe.Slice(out, int(count))
		for i := range requested {
			name := cString(requested[i])
			metrics[i] = DeviceMetrics{}
			copy(metrics[i].DeviceUUID[:], name)
			switch name {
			case "GPU-ok":
				metrics[i].UtilizationPercent = 62
			case "GPU-unreadable":
				metrics[i].UtilizationPercent = math.MaxUint32
			case "GPU-garbage":
				metrics[i].UtilizationPercent = 250
			}
		}
		return ResultSuccess
	}
	deviceMetrics, err := controller.GetDeviceMetrics()
	require.NoError(t, err)
	require.Len(t, deviceMetrics, 2)
	require.Contains(t, deviceMetrics, "gpu-ok")
	require.Contains(t, deviceMetrics, "gpu-idle")
}

func cString(ptr *byte) string {
	if ptr == nil {
		return ""
	}
	length := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(ptr), length)) != 0 {
		length++
	}
	return string(unsafe.Slice(ptr, length))
}

func TestProcessUtilizationNotSuppliedIsFlagged(t *testing.T) {
	old := getProcessInformation
	t.Cleanup(func() { getProcessInformation = old })
	reported := []struct {
		pid     string
		percent float64
	}{
		{"100", 37.5},         // measured
		{"200", 0},            // measured idle
		{"300", -1},           // ACCEL_COMPUTE_UTILIZATION_UNAVAILABLE
		{"400", math.NaN()},   // outside the contract
		{"500", -0.5},         // outside the contract
		{"600", math.Inf(1)},  // outside the contract
		{"700", 100},          // boundary
		{"800", math.Inf(-1)}, // outside the contract
	}
	getProcessInformation = func(out *ProcessInformation, maxCount uintptr, count *uintptr) Result {
		infos := unsafe.Slice(out, int(maxCount))
		for i, process := range reported {
			infos[i] = ProcessInformation{ComputeUtilizationPercent: process.percent, MemoryUsedBytes: uint64(i+1) << 20}
			copy(infos[i].ProcessID[:], process.pid)
			copy(infos[i].DeviceUUID[:], "GPU-0")
		}
		*count = uintptr(len(reported))
		return ResultSuccess
	}

	accelerator := &AcceleratorInterface{}
	accelerator.telemetryMarkers.Store(true)
	processes, err := accelerator.GetProcessInformation()
	require.NoError(t, err)
	require.Len(t, processes, len(reported))
	wantUnavailable := map[string]bool{"300": true, "400": true, "500": true, "600": true, "800": true}
	sum := 0.0
	for i, process := range processes {
		require.Equal(t, wantUnavailable[process.ProcessID], process.ComputeUtilizationUnavailable,
			"pid %s", process.ProcessID)
		if process.ComputeUtilizationUnavailable {
			require.Zero(t, process.ComputeUtilizationPercent, "pid %s", process.ProcessID)
		}
		// Memory is reported either way.
		require.Equal(t, uint64(i+1)<<20, process.MemoryUsedBytes)
		sum += process.ComputeUtilizationPercent
	}
	require.Equal(t, 137.5, sum, "a marker must never take part in a sum")
}

// What the provider does not report is derived from the device: memory in use
// as a share of the device's memory, and utilization as TFLOPS in use.
func TestDeviceMetricsCarryMemoryShareAndTflopsInUse(t *testing.T) {
	old := getDeviceMetrics
	t.Cleanup(func() { getDeviceMetrics = old })
	getDeviceMetrics = func(uuids **byte, count uintptr, out *DeviceMetrics) Result {
		requested := unsafe.Slice(uuids, int(count))
		metrics := unsafe.Slice(out, int(count))
		for i := range requested {
			metrics[i] = DeviceMetrics{UtilizationPercent: 40, MemoryUsedBytes: 6 << 30}
			copy(metrics[i].DeviceUUID[:], cString(requested[i]))
		}
		return ResultSuccess
	}
	listed := &api.DeviceInfo{UUID: "gpu-3090", TotalMemoryBytes: 24 << 30, ProviderEstimatedTflops: 44.08}
	listed.ResolveTflopsCapacity(71.2, true)
	estimated := &api.DeviceInfo{UUID: "gpu-a6000", TotalMemoryBytes: 48 << 30, ProviderEstimatedTflops: 38.7072}
	estimated.ResolveTflopsCapacity(0, false)
	unknown := &api.DeviceInfo{UUID: "gpu-unknown"}
	controller := &Controller{
		accelerator: &AcceleratorInterface{},
		devices:     map[string]*api.DeviceInfo{listed.UUID: listed, estimated.UUID: estimated, unknown.UUID: unknown},
		nativeUUIDs: map[string]string{listed.UUID: "GPU-3090", estimated.UUID: "GPU-A6000", unknown.UUID: "GPU-unknown"},
	}

	metrics, err := controller.GetDeviceMetrics()
	require.NoError(t, err)
	require.Len(t, metrics, 3)
	// 40% of 71.2 FP16 TFLOPS, not of the provider's 44.08.
	require.InDelta(t, 28.48, metrics["gpu-3090"].ComputeTflops, 1e-9)
	require.Equal(t, 25.0, metrics["gpu-3090"].MemoryPercentage)
	// Without a table entry the capacity is the provider's estimate.
	require.InDelta(t, 0.4*38.7072, metrics["gpu-a6000"].ComputeTflops, 1e-9)
	require.Equal(t, 12.5, metrics["gpu-a6000"].MemoryPercentage)
	// Nothing known about the device: nothing derived.
	require.Zero(t, metrics["gpu-unknown"].ComputeTflops)
	require.Zero(t, metrics["gpu-unknown"].MemoryPercentage)
	require.Equal(t, 40.0, metrics["gpu-unknown"].ComputePercentage)
}
