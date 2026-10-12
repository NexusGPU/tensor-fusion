package metrics

import (
	"bytes"
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

type usageTestDevices struct {
	framework.DeviceController
	devices map[string]*api.DeviceInfo
	metrics map[string]*api.GPUUsageMetrics
}

func (d *usageTestDevices) GetDevice(uuid string) (*api.DeviceInfo, bool) {
	device, ok := d.devices[uuid]
	return device, ok
}

func (d *usageTestDevices) GetDeviceMetrics() (map[string]*api.GPUUsageMetrics, error) {
	return d.metrics, nil
}

type usageTestWorkers struct {
	framework.WorkerController
	workers []*api.WorkerInfo
	metrics map[string]map[string]map[string]*api.WorkerMetrics
	usage   map[string]map[string]api.WorkerComputeUsage
}

func (w *usageTestWorkers) ListWorkers() ([]*api.WorkerInfo, error) { return w.workers, nil }

func (w *usageTestWorkers) GetWorkerMetrics() (map[string]map[string]map[string]*api.WorkerMetrics, error) {
	return w.metrics, nil
}

func (w *usageTestWorkers) GetWorkerComputeUsage() map[string]map[string]api.WorkerComputeUsage {
	copied := map[string]map[string]api.WorkerComputeUsage{}
	for device, workers := range w.usage {
		copied[device] = map[string]api.WorkerComputeUsage{}
		for worker, usage := range workers {
			copied[device][worker] = usage
		}
	}
	return copied
}

// instantOnlyWorkers is a worker controller that does not integrate usage.
type instantOnlyWorkers struct {
	framework.WorkerController
	inner *usageTestWorkers
}

func (w *instantOnlyWorkers) ListWorkers() ([]*api.WorkerInfo, error) { return w.inner.workers, nil }

func (w *instantOnlyWorkers) GetWorkerMetrics() (map[string]map[string]map[string]*api.WorkerMetrics, error) {
	return w.inner.metrics, nil
}

type usageTestAllocations struct {
	framework.WorkerAllocationController
	workers []*api.WorkerInfo
}

func (a *usageTestAllocations) GetWorkerAllocation(uid string) (*api.WorkerAllocation, bool) {
	for _, worker := range a.workers {
		if worker.WorkerUID == uid {
			return &api.WorkerAllocation{WorkerInfo: worker}, true
		}
	}
	return nil, false
}

// usageLine is one influx line: its tags and its fields, as written.
type usageLine struct {
	measurement string
	tags        map[string]string
	fields      map[string]float64
}

func parseUsageLines(t *testing.T, raw []byte) []usageLine {
	t.Helper()
	var lines []usageLine
	for _, text := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if text == "" {
			continue
		}
		parts := strings.Split(text, " ")
		require.Len(t, parts, 3, "line %q", text)
		head := strings.Split(parts[0], ",")
		line := usageLine{measurement: head[0], tags: map[string]string{}, fields: map[string]float64{}}
		for _, tag := range head[1:] {
			key, value, _ := strings.Cut(tag, "=")
			line.tags[key] = value
		}
		for _, field := range strings.Split(parts[1], ",") {
			key, value, _ := strings.Cut(field, "=")
			number, err := strconv.ParseFloat(strings.TrimSuffix(strings.TrimSuffix(value, "i"), "u"), 64)
			require.NoError(t, err, "field %q of line %q", field, text)
			line.fields[key] = number
		}
		lines = append(lines, line)
	}
	return lines
}

func rtx3090() *api.DeviceInfo {
	device := &api.DeviceInfo{UUID: "gpu-0", TotalMemoryBytes: 24 << 30, ProviderEstimatedTflops: 44.08}
	device.ResolveTflopsCapacity(71.2, true)
	return device
}

func TestDeviceUsageRowsCarryTflopsAndMemoryShare(t *testing.T) {
	unknownCapacity := &api.DeviceInfo{UUID: "gpu-1", TotalMemoryBytes: 48 << 30}
	devices := &usageTestDevices{
		devices: map[string]*api.DeviceInfo{"gpu-0": rtx3090(), "gpu-1": unknownCapacity},
		// As the device controller returns them: memory share and TFLOPS in use
		// derived from the device.
		metrics: map[string]*api.GPUUsageMetrics{
			"gpu-0": {
				DeviceUUID: "gpu-0", MemoryBytes: 6 << 30, MemoryPercentage: 25, ComputePercentage: 50, ComputeTflops: 35.6,
			},
			"gpu-1": {DeviceUUID: "gpu-1", MemoryBytes: 12 << 30, MemoryPercentage: 25, ComputePercentage: 50},
		},
	}
	recorder := &HypervisorMetricsRecorder{nodeName: "node", gpuPool: "pool", deviceController: devices}
	var out bytes.Buffer
	recorder.RecordDeviceMetrics(&out)

	byDevice := map[string]usageLine{}
	for _, line := range parseUsageLines(t, out.Bytes()) {
		require.Equal(t, "tf_gpu_usage", line.measurement)
		byDevice[line.tags["uuid"]] = line
	}
	require.Len(t, byDevice, 2)
	require.Equal(t, 35.6, byDevice["gpu-0"].fields["compute_tflops"])
	require.Equal(t, 50.0, byDevice["gpu-0"].fields["compute_percentage"])
	require.Equal(t, 25.0, byDevice["gpu-0"].fields["memory_percentage"])
	require.Equal(t, float64(6<<30), byDevice["gpu-0"].fields["memory_bytes"])
	require.NotContains(t, byDevice["gpu-1"].fields, "compute_tflops", "unknown capacity: the field is left out, not zero")
	require.Equal(t, 50.0, byDevice["gpu-1"].fields["compute_percentage"])
}

// A value the provider could not read in a sample is left out of the row; the
// rest of what it read of the device is still written.
func TestDeviceUsageRowsLeaveOutWhatWasNotRead(t *testing.T) {
	devices := &usageTestDevices{
		devices: map[string]*api.DeviceInfo{"gpu-0": rtx3090(), "gpu-1": rtx3090()},
		metrics: map[string]*api.GPUUsageMetrics{
			"gpu-0": {DeviceUUID: "gpu-0", ComputeUnavailable: true, MemoryBytes: 6 << 30, MemoryPercentage: 25,
				Temperature: 55, PowerUsage: 180},
			"gpu-1": {DeviceUUID: "gpu-1", MemoryUnavailable: true, ComputePercentage: 50, ComputeTflops: 35.6,
				Temperature: 60, PowerUsage: 200},
		},
	}
	recorder := &HypervisorMetricsRecorder{nodeName: "node", gpuPool: "pool", deviceController: devices}
	var out bytes.Buffer
	recorder.RecordDeviceMetrics(&out)

	byDevice := map[string]usageLine{}
	for _, line := range parseUsageLines(t, out.Bytes()) {
		byDevice[line.tags["uuid"]] = line
	}
	require.Len(t, byDevice, 2)

	noUtilization := byDevice["gpu-0"].fields
	require.NotContains(t, noUtilization, "compute_percentage")
	require.NotContains(t, noUtilization, "compute_tflops")
	require.Equal(t, float64(6<<30), noUtilization["memory_bytes"])
	require.Equal(t, 25.0, noUtilization["memory_percentage"])
	require.Equal(t, 55.0, noUtilization["temperature"])
	require.Equal(t, 180.0, noUtilization["power_usage"])

	noMemory := byDevice["gpu-1"].fields
	require.NotContains(t, noMemory, "memory_bytes")
	require.NotContains(t, noMemory, "memory_percentage")
	require.Equal(t, 50.0, noMemory["compute_percentage"])
	require.Equal(t, 35.6, noMemory["compute_tflops"])
	require.Equal(t, 60.0, noMemory["temperature"])
}

type failingWriter struct{ err error }

func (w *failingWriter) Write(data []byte) (int, error) {
	if w.err != nil {
		return 0, w.err
	}
	return len(data), nil
}

// A metrics file that cannot be written loses records, nothing else: the
// recorder carries on and writes again as soon as it can.
func TestMetricsWriteFailureIsNotFatal(t *testing.T) {
	devices := &usageTestDevices{
		devices: map[string]*api.DeviceInfo{"gpu-0": rtx3090()},
		metrics: map[string]*api.GPUUsageMetrics{"gpu-0": {DeviceUUID: "gpu-0", ComputePercentage: 50}},
	}
	recorder := &HypervisorMetricsRecorder{nodeName: "node", gpuPool: "pool", deviceController: devices}
	writer := &failingWriter{err: errors.New("no space left on device")}
	for range 3 {
		recorder.RecordDeviceMetrics(writer)
	}
	require.True(t, recorder.writeFailing.Load())
	writer.err = nil
	recorder.RecordDeviceMetrics(writer)
	require.False(t, recorder.writeFailing.Load())
}

func TestWorkerUsageRowsCarryIntervalAverageAndTflops(t *testing.T) {
	limited := &api.WorkerInfo{
		WorkerUID: "uid-limited", WorkerName: "limited", Namespace: "ns", WorkloadName: "app",
		Limits: tfv1.Resource{Vram: resource.MustParse("8Gi")},
	}
	unlimited := &api.WorkerInfo{WorkerUID: "uid-unlimited", WorkerName: "unlimited", Namespace: "ns"}
	unmeasured := &api.WorkerInfo{WorkerUID: "uid-unmeasured", WorkerName: "unmeasured", Namespace: "ns"}
	all := []*api.WorkerInfo{limited, unlimited, unmeasured}
	workers := &usageTestWorkers{
		workers: all,
		metrics: map[string]map[string]map[string]*api.WorkerMetrics{"gpu-0": {
			// Two processes of one worker; what they read at this instant is 90%.
			"uid-limited": {
				"1": {MemoryBytes: 1 << 30, ComputePercentage: 60},
				"2": {MemoryBytes: 1 << 30, ComputePercentage: 30},
			},
			"uid-unlimited":  {"3": {MemoryBytes: 6 << 30, ComputePercentage: 10}},
			"uid-unmeasured": {"4": {MemoryBytes: 3 << 30, ComputeUnavailable: true}},
		}},
		usage: map[string]map[string]api.WorkerComputeUsage{"gpu-0": {
			"uid-limited":   {PercentSeconds: 1000, Seconds: 100},
			"uid-unlimited": {PercentSeconds: 0, Seconds: 100},
		}},
	}
	devices := &usageTestDevices{devices: map[string]*api.DeviceInfo{"gpu-0": rtx3090()}}
	recorder := &HypervisorMetricsRecorder{
		nodeName: "node", gpuPool: "pool", deviceController: devices, workerController: workers,
		allocationController: &usageTestAllocations{workers: all},
	}
	record := func() map[string]usageLine {
		t.Helper()
		var out bytes.Buffer
		recorder.RecordWorkerMetrics(&out)
		rows := map[string]usageLine{}
		for _, line := range parseUsageLines(t, out.Bytes()) {
			require.Equal(t, "tf_worker_usage", line.measurement)
			require.Equal(t, "gpu-0", line.tags["uuid"])
			rows[line.tags["pod_name"]] = line
		}
		return rows
	}

	// First record: no earlier reading to average against, so the instant is used.
	first := record()
	require.Len(t, first, 3)
	require.Equal(t, 90.0, first["limited"].fields["compute_percentage"])
	require.InDelta(t, 0.9*71.2, first["limited"].fields["compute_tflops"], 1e-9)
	// Memory against the worker's limit, or against the device without one.
	require.Equal(t, 25.0, first["limited"].fields["memory_percentage"])
	require.Equal(t, 25.0, first["unlimited"].fields["memory_percentage"])
	require.Equal(t, float64(2<<30), first["limited"].fields["memory_bytes"])
	// Not measured: both compute fields are left out, memory is still reported.
	require.NotContains(t, first["unmeasured"].fields, "compute_percentage")
	require.NotContains(t, first["unmeasured"].fields, "compute_tflops")
	require.Equal(t, 12.5, first["unmeasured"].fields["memory_percentage"])
	require.Equal(t, "uid-limited", first["limited"].tags["worker"])
	require.Equal(t, "app", first["limited"].tags["workload"])

	// A minute later: the worker averaged 40% of the device over that minute,
	// whatever its processes read in the instant of the record.
	workers.usage["gpu-0"]["uid-limited"] = api.WorkerComputeUsage{PercentSeconds: 1000 + 40*60, Seconds: 160}
	workers.usage["gpu-0"]["uid-unlimited"] = api.WorkerComputeUsage{PercentSeconds: 0, Seconds: 160}
	second := record()
	require.InDelta(t, 40, second["limited"].fields["compute_percentage"], 1e-9)
	// TFLOPS in use = 40% x 71.2 FP16 TFLOPS.
	require.InDelta(t, 28.48, second["limited"].fields["compute_tflops"], 1e-9)
	// A measured idle worker is written as zero, which is not "unknown".
	require.Contains(t, second["unlimited"].fields, "compute_tflops")
	require.Zero(t, second["unlimited"].fields["compute_tflops"])
	require.Zero(t, second["unlimited"].fields["compute_percentage"])
	require.NotContains(t, second["unmeasured"].fields, "compute_tflops")

	// Nothing was measured during the next interval: the fields are left out
	// instead of repeating the instant or reporting zero.
	third := record()
	require.NotContains(t, third["limited"].fields, "compute_percentage")
	require.NotContains(t, third["limited"].fields, "compute_tflops")
	require.Contains(t, third["limited"].fields, "memory_bytes")

	// A device whose capacity is unknown has a utilization but no TFLOPS figure.
	devices.devices["gpu-0"] = &api.DeviceInfo{UUID: "gpu-0", TotalMemoryBytes: 24 << 30}
	workers.usage["gpu-0"]["uid-limited"] = api.WorkerComputeUsage{PercentSeconds: 1000 + 40*60 + 10*60, Seconds: 220}
	fourth := record()
	require.InDelta(t, 10, fourth["limited"].fields["compute_percentage"], 1e-9)
	require.NotContains(t, fourth["limited"].fields, "compute_tflops")

	// A worker controller without integrated usage: the instant, as before.
	recorder.workerController = &instantOnlyWorkers{inner: workers}
	devices.devices["gpu-0"] = rtx3090()
	instant := record()
	require.Equal(t, 90.0, instant["limited"].fields["compute_percentage"])
	require.InDelta(t, 0.9*71.2, instant["limited"].fields["compute_tflops"], 1e-9)
}
