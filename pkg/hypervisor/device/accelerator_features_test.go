package device

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"unsafe"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/stretchr/testify/require"
)

// stubInit replaces the library's initialization and feature selection for one
// test. selection nil stands for a library that does not export
// AccelSetFeatures.
func stubInit(t *testing.T, selection func(requested uint64, enabled *uint64) Result) {
	t.Helper()
	oldInit, oldSet := accelInit, setFeatures
	t.Cleanup(func() { accelInit, setFeatures = oldInit, oldSet })
	accelInit = func() Result { return ResultSuccess }
	setFeatures = selection
}

func stubProcessRows(t *testing.T, fill func(rows []ProcessInformation) (int, Result)) {
	t.Helper()
	old := getProcessInformation
	t.Cleanup(func() { getProcessInformation = old })
	getProcessInformation = func(out *ProcessInformation, maxCount uintptr, count *uintptr) Result {
		written, result := fill(unsafe.Slice(out, int(maxCount)))
		*count = uintptr(written)
		return result
	}
}

func processRow(pid, device string, percent float64, memory uint64) ProcessInformation {
	row := ProcessInformation{ComputeUtilizationPercent: percent, MemoryUsedBytes: memory}
	copy(row.ProcessID[:], pid)
	copy(row.DeviceUUID[:], device)
	return row
}

// The features are selected after every initialization, and are only active
// when the library says so.
func TestFeatureSelection(t *testing.T) {
	for _, tc := range []struct {
		name      string
		selection func(requested uint64, enabled *uint64) Result
		want      bool
	}{
		{name: "library without the function", selection: nil, want: false},
		{name: "selection refused", selection: func(_ uint64, enabled *uint64) Result {
			*enabled = featureTelemetryMarkers
			return ResultErrorOperationFailed
		}, want: false},
		{name: "feature unknown to the library", selection: func(_ uint64, enabled *uint64) Result {
			*enabled = 0
			return ResultSuccess
		}, want: false},
		{name: "another feature only", selection: func(_ uint64, enabled *uint64) Result {
			*enabled = 1 << 5
			return ResultSuccess
		}, want: false},
		{name: "feature enabled", selection: func(requested uint64, enabled *uint64) Result {
			*enabled = requested & featureTelemetryMarkers
			return ResultSuccess
		}, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls, requested := 0, uint64(0)
			var selection func(uint64, *uint64) Result
			if tc.selection != nil {
				selection = func(r uint64, enabled *uint64) Result {
					calls++
					requested = r
					return tc.selection(r, enabled)
				}
			}
			stubInit(t, selection)

			accelerator := &AcceleratorInterface{libPath: "stub"}
			// Left over from an earlier initialization: must not survive.
			accelerator.telemetryMarkers.Store(true)
			require.NoError(t, accelerator.initialize())
			require.Equal(t, tc.want, accelerator.TelemetryMarkers())
			if tc.selection != nil {
				require.Equal(t, 1, calls)
				require.Equal(t, featureTelemetryMarkers, requested&featureTelemetryMarkers)
			}

			// The library forgets the selection when it is shut down, so it is
			// made again after every initialization.
			require.NoError(t, accelerator.initialize())
			require.Equal(t, tc.want, accelerator.TelemetryMarkers())
			if tc.selection != nil {
				require.Equal(t, 2, calls)
			}
		})
	}
}

func TestFeaturesAreNotSelectedWhenInitializationFails(t *testing.T) {
	selected := false
	stubInit(t, func(_ uint64, enabled *uint64) Result {
		selected = true
		*enabled = featureTelemetryMarkers
		return ResultSuccess
	})
	accelInit = func() Result { return ResultErrorInternal }

	accelerator := &AcceleratorInterface{libPath: "stub"}
	accelerator.telemetryMarkers.Store(true)
	require.Error(t, accelerator.initialize())
	require.False(t, selected)
	require.False(t, accelerator.TelemetryMarkers())
}

// Without the feature the library does not measure per-process utilization:
// the field is always zero and must not be taken for a measurement.
func TestProcessUtilizationIsNotSuppliedWithoutTheFeature(t *testing.T) {
	stubProcessRows(t, func(rows []ProcessInformation) (int, Result) {
		rows[0] = processRow("100", "GPU-0", 0, 1<<20)
		rows[1] = processRow("200", "GPU-0", 0, 2<<20)
		// Not what such a library returns, and not believed either.
		rows[2] = processRow("300", "GPU-1", 55, 3<<20)
		return 3, ResultSuccess
	})

	processes, err := (&AcceleratorInterface{}).GetProcessInformation()
	require.NoError(t, err)
	require.Len(t, processes, 3)
	for i, process := range processes {
		require.True(t, process.ComputeUtilizationUnavailable, "pid %s", process.ProcessID)
		require.Zero(t, process.ComputeUtilizationPercent, "pid %s", process.ProcessID)
		require.False(t, process.ProcessListUnavailable)
		require.Equal(t, uint64(i+1)<<20, process.MemoryUsedBytes, "memory is reported either way")
	}
}

// With the feature, a device whose process list could not be read is one row
// without a process, and a device without a utilization sample marks each of
// its processes.
func TestProcessMarkersWithTheFeature(t *testing.T) {
	stubProcessRows(t, func(rows []ProcessInformation) (int, Result) {
		rows[0] = processRow("100", "GPU-0", 40, 1<<30)
		rows[1] = processRow("200", "GPU-0", 0, 2<<30)
		rows[2] = processRow("", "GPU-1", computeUtilizationUnavailable, 0)
		rows[3] = processRow("300", "GPU-2", computeUtilizationUnavailable, 3<<30)
		return 4, ResultSuccess
	})
	accelerator := &AcceleratorInterface{}
	accelerator.telemetryMarkers.Store(true)

	processes, err := accelerator.GetProcessInformation()
	require.NoError(t, err)
	require.Equal(t, []api.ProcessInformation{
		{ProcessID: "100", DeviceUUID: "GPU-0", ComputeUtilizationPercent: 40, MemoryUsedBytes: 1 << 30},
		{ProcessID: "200", DeviceUUID: "GPU-0", MemoryUsedBytes: 2 << 30},
		{DeviceUUID: "GPU-1", ComputeUtilizationUnavailable: true, ProcessListUnavailable: true},
		{ProcessID: "300", DeviceUUID: "GPU-2", ComputeUtilizationUnavailable: true, MemoryUsedBytes: 3 << 30},
	}, processes)
}

// Utilization and memory of a device are marked one by one; what was read of
// the entry is kept, and a marker never becomes a number.
func TestDeviceMetricsMarkersWithTheFeature(t *testing.T) {
	old := getDeviceMetrics
	t.Cleanup(func() { getDeviceMetrics = old })
	getDeviceMetrics = func(uuids **byte, count uintptr, out *DeviceMetrics) Result {
		requested, metrics := unsafe.Slice(uuids, int(count)), unsafe.Slice(out, int(count))
		for i := range requested {
			name := cString(requested[i])
			metrics[i] = DeviceMetrics{
				UtilizationPercent: 30, MemoryUsedBytes: 6 << 30, PowerUsageWatts: 180, TemperatureCelsius: 55,
			}
			copy(metrics[i].DeviceUUID[:], name)
			switch name {
			case "GPU-no-utilization":
				metrics[i].UtilizationPercent = math.MaxUint32
			case "GPU-no-memory":
				metrics[i].MemoryUsedBytes = math.MaxUint64
			case "GPU-closed":
				metrics[i].UtilizationPercent, metrics[i].MemoryUsedBytes = math.MaxUint32, math.MaxUint64
			}
		}
		return ResultSuccess
	}
	accelerator := &AcceleratorInterface{}
	accelerator.telemetryMarkers.Store(true)
	device := func(uuid string) *api.DeviceInfo {
		info := &api.DeviceInfo{UUID: uuid, TotalMemoryBytes: 24 << 30}
		info.ResolveTflopsCapacity(100, true)
		return info
	}
	controller := &Controller{
		accelerator: accelerator,
		devices: map[string]*api.DeviceInfo{
			"gpu-ok": device("gpu-ok"), "gpu-no-utilization": device("gpu-no-utilization"),
			"gpu-no-memory": device("gpu-no-memory"), "gpu-closed": device("gpu-closed"),
		},
		nativeUUIDs: map[string]string{
			"gpu-ok": "GPU-ok", "gpu-no-utilization": "GPU-no-utilization",
			"gpu-no-memory": "GPU-no-memory", "gpu-closed": "GPU-closed",
		},
	}

	metrics, err := controller.GetDeviceMetrics()
	require.NoError(t, err)
	require.Len(t, metrics, 3, "a device that could not be opened has no sample")
	require.NotContains(t, metrics, "gpu-closed")

	ok := metrics["gpu-ok"]
	require.False(t, ok.ComputeUnavailable || ok.MemoryUnavailable)
	require.Equal(t, 30.0, ok.ComputePercentage)
	require.Equal(t, 30.0, ok.ComputeTflops)
	require.Equal(t, 25.0, ok.MemoryPercentage)

	noUtilization := metrics["gpu-no-utilization"]
	require.True(t, noUtilization.ComputeUnavailable)
	require.False(t, noUtilization.MemoryUnavailable)
	require.Zero(t, noUtilization.ComputePercentage)
	require.Zero(t, noUtilization.ComputeTflops)
	require.Equal(t, uint64(6<<30), noUtilization.MemoryBytes)
	require.Equal(t, 25.0, noUtilization.MemoryPercentage)
	require.Equal(t, int64(180), noUtilization.PowerUsage)
	require.Equal(t, 55.0, noUtilization.Temperature)

	noMemory := metrics["gpu-no-memory"]
	require.True(t, noMemory.MemoryUnavailable)
	require.False(t, noMemory.ComputeUnavailable)
	require.Zero(t, noMemory.MemoryBytes)
	require.Zero(t, noMemory.MemoryPercentage)
	require.Equal(t, 30.0, noMemory.ComputePercentage)
	require.Equal(t, int64(180), noMemory.PowerUsage)
}

// A library whose rows do not fit returns none of them. The buffer grows until
// they fit, is kept at that size, and gives up at its limit.
func TestProcessListGrowsItsBuffer(t *testing.T) {
	rowsAvailable := 3000
	var offered []int
	stubProcessRows(t, func(rows []ProcessInformation) (int, Result) {
		offered = append(offered, len(rows))
		if len(rows) < rowsAvailable {
			return 0, ResultErrorResourceExhausted
		}
		for i := range rowsAvailable {
			rows[i] = processRow("7", "GPU-0", 0, 1<<20)
		}
		return rowsAvailable, ResultSuccess
	})
	accelerator := &AcceleratorInterface{}

	processes, err := accelerator.GetProcessInformation()
	require.NoError(t, err)
	require.Len(t, processes, rowsAvailable)
	require.Equal(t, []int{1024, 2048, 4096}, offered)

	// The next call starts with the buffer that was large enough.
	offered = nil
	_, err = accelerator.GetProcessInformation()
	require.NoError(t, err)
	require.Equal(t, []int{4096}, offered)

	// More rows than the limit is an error, not a truncated list, and the
	// limit is not exceeded.
	rowsAvailable, offered = maxProcessRows+1, nil
	_, err = accelerator.GetProcessInformation()
	require.Error(t, err)
	require.Equal(t, []int{4096, 8192, maxProcessRows}, offered)

	// It recovers as soon as the rows fit again.
	rowsAvailable = 10
	processes, err = accelerator.GetProcessInformation()
	require.NoError(t, err)
	require.Len(t, processes, 10)
}

// The buffer the library writes into is not allocated per call.
func TestProcessListBufferIsReused(t *testing.T) {
	stubProcessRows(t, func(rows []ProcessInformation) (int, Result) { return 0, ResultSuccess })
	accelerator := &AcceleratorInterface{}
	_, err := accelerator.GetProcessInformation()
	require.NoError(t, err)
	first := &accelerator.processRows[0]
	allocations := testing.AllocsPerRun(100, func() {
		if _, err := accelerator.GetProcessInformation(); err != nil {
			t.Fatal(err)
		}
	})
	require.Same(t, first, &accelerator.processRows[0])
	require.LessOrEqual(t, allocations, 1.0, "only the (empty) result may be allocated")
}

// exampleLibrary returns the path of a library built by `make build-provider`,
// or skips the test when it was not built.
func exampleLibrary(t *testing.T, name string) string {
	t.Helper()
	path := filepath.Join("..", "..", "..", "provider", "build", name)
	if _, err := os.Stat(path); err != nil {
		t.Skipf("%s not built (make build-provider)", name)
	}
	return path
}

// Loaded for real: the example provider agrees to the feature, and the same
// provider built without AccelSetFeatures, as a library released before the
// function existed, loads and works with the feature off.
func TestFeatureSelectionWithTheExampleProvider(t *testing.T) {
	current := exampleLibrary(t, "libaccelerator_example.so")
	legacy := exampleLibrary(t, "libaccelerator_example_legacy.so")

	accelerator, err := NewAcceleratorInterface(current)
	require.NoError(t, err)
	require.True(t, accelerator.TelemetryMarkers())
	_, err = accelerator.GetProcessInformation()
	require.NoError(t, err)
	require.NoError(t, accelerator.Close())
	require.False(t, accelerator.TelemetryMarkers(), "the selection ends with the shutdown")
	// A new initialization selects the features again.
	require.NoError(t, accelerator.Load())
	require.True(t, accelerator.TelemetryMarkers())
	require.NoError(t, accelerator.Close())

	old, err := NewAcceleratorInterface(legacy)
	require.NoError(t, err)
	require.False(t, old.TelemetryMarkers())
	require.Nil(t, setFeatures, "the binding of the previously loaded library must not be left behind")
	devices, err := old.GetAllDevices()
	require.NoError(t, err)
	require.NotEmpty(t, devices)
	_, err = old.GetProcessInformation()
	require.NoError(t, err)
	require.NoError(t, old.Close())

	// The bindings are per process: leave the current library bound.
	accelerator, err = NewAcceleratorInterface(current)
	require.NoError(t, err)
	require.True(t, accelerator.TelemetryMarkers())
	require.NoError(t, accelerator.Close())
}

// A library that did not agree to return markers returns none: a row without a
// process id and a memory figure of all ones are then what that library
// reported, not statements about an unreadable device.
func TestNoMarkerIsRecognisedWithoutTheFeature(t *testing.T) {
	stubProcessRows(t, func(rows []ProcessInformation) (int, Result) {
		rows[0] = processRow("", "GPU-0", 0, 5<<20)
		rows[1] = processRow("100", "GPU-0", computeUtilizationUnavailable, 1<<20)
		return 2, ResultSuccess
	})
	oldMetrics := getDeviceMetrics
	t.Cleanup(func() { getDeviceMetrics = oldMetrics })
	getDeviceMetrics = func(uuids **byte, count uintptr, out *DeviceMetrics) Result {
		metrics := unsafe.Slice(out, int(count))
		metrics[0] = DeviceMetrics{UtilizationPercent: 30, MemoryUsedBytes: math.MaxUint64, PowerUsageWatts: 180}
		copy(metrics[0].DeviceUUID[:], "GPU-0")
		return ResultSuccess
	}

	for _, markers := range []bool{false, true} {
		accelerator := &AcceleratorInterface{}
		accelerator.telemetryMarkers.Store(markers)

		processes, err := accelerator.GetProcessInformation()
		require.NoError(t, err)
		require.Len(t, processes, 2)
		require.Equal(t, markers, processes[0].ProcessListUnavailable, "markers=%v", markers)
		if !markers {
			require.Equal(t, uint64(5<<20), processes[0].MemoryUsedBytes, "the row is passed on as reported")
		}
		require.True(t, processes[1].ComputeUtilizationUnavailable)
		require.False(t, processes[1].ProcessListUnavailable)

		metrics, err := accelerator.GetDeviceMetrics([]string{"GPU-0"})
		require.NoError(t, err)
		require.Len(t, metrics, 1)
		require.Equal(t, markers, metrics[0].MemoryUnavailable, "markers=%v", markers)
		require.Equal(t, 30.0, metrics[0].ComputePercentage)
		if !markers {
			require.Equal(t, uint64(math.MaxUint64), metrics[0].MemoryBytes)
		}
	}
}
