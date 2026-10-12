package worker

import (
	"errors"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/stretchr/testify/require"
)

// The process sample used for memory accounting also yields per-worker compute
// utilization. A provider that cannot supply it says so explicitly; that must
// stay distinguishable from workers that were measured idle.
func TestProcessSamplePublishesPerWorkerComputeUtilization(t *testing.T) {
	sampledAt := time.Unix(1_710_000_000, 0)
	devices := &fakeDeviceController{}
	controller := &WorkerController{
		backend: &fakeWorkerBackend{mappings: map[uint32]*framework.ProcessMappingInfo{
			101: {Namespace: "ns", PodName: "pod-a"},
			102: {Namespace: "ns", PodName: "pod-a"},
			201: {Namespace: "ns", PodName: "pod-b"},
			301: {Namespace: "other", PodName: "not-a-worker"},
		}},
		deviceController: devices,
		workers: map[string]*api.WorkerInfo{
			"uid-a": {WorkerUID: "uid-a", Namespace: "ns", WorkerName: "pod-a", IsolationMode: tfv1.IsolationModeSoft},
			"uid-b": {WorkerUID: "uid-b", Namespace: "ns", WorkerName: "pod-b", IsolationMode: tfv1.IsolationModeSoft},
		},
		nowFunc: func() time.Time { return sampledAt },
	}
	collect := func() { controller.collectWorkerMemoryUsage(controller.buildWorkerLookupMap()) }

	require.Nil(t, controller.processUtilization.Load(), "nothing is published before the first sample")

	// The provider could not supply per-process utilization for the device and
	// says so: the device is not covered, which is not the same as idle.
	devices.processInfos = []api.ProcessInformation{
		{ProcessID: "101", DeviceUUID: "GPU-0", MemoryUsedBytes: 1 << 20, ComputeUtilizationUnavailable: true},
		{ProcessID: "201", DeviceUUID: "GPU-0", MemoryUsedBytes: 1 << 20, ComputeUtilizationUnavailable: true},
	}
	collect()
	snapshot := controller.processUtilization.Load()
	require.NotNil(t, snapshot)
	require.Equal(t, sampledAt, snapshot.SampledAt)
	require.Empty(t, snapshot.Devices, "utilization that was not supplied is not a measurement")

	// Supplied zeros are a measurement: every process read idle in this sample.
	devices.processInfos = []api.ProcessInformation{
		{ProcessID: "101", DeviceUUID: "GPU-0", MemoryUsedBytes: 1 << 20},
		{ProcessID: "201", DeviceUUID: "GPU-0", MemoryUsedBytes: 1 << 20},
	}
	sampledAt = sampledAt.Add(time.Second)
	collect()
	snapshot = controller.processUtilization.Load()
	require.Equal(t, map[string]map[string]float64{"gpu-0": {"uid-a": 0, "uid-b": 0}}, snapshot.Devices)
	require.Equal(t, map[string]float64{"gpu-0": 0}, snapshot.Totals)

	// Values are summed per worker and device. A process outside any worker,
	// such as a desktop compositor, is listed but not charged to one.
	devices.processInfos = []api.ProcessInformation{
		{ProcessID: "101", DeviceUUID: "GPU-0", ComputeUtilizationPercent: 20},
		{ProcessID: "102", DeviceUUID: "GPU-0", ComputeUtilizationPercent: 5},
		{ProcessID: "201", DeviceUUID: "GPU-0", ComputeUtilizationPercent: 0},
		{ProcessID: "301", DeviceUUID: "GPU-0", ComputeUtilizationPercent: 40},
		{ProcessID: "201", DeviceUUID: "GPU-1", ComputeUtilizationPercent: 0},
		{ProcessID: "101", DeviceUUID: "GPU-2", ComputeUtilizationUnavailable: true},
	}
	sampledAt = sampledAt.Add(time.Second)
	collect()
	snapshot = controller.processUtilization.Load()
	require.Equal(t, sampledAt, snapshot.SampledAt)
	require.Equal(t, map[string]map[string]float64{
		"gpu-0": {"uid-a": 25, "uid-b": 0},
		"gpu-1": {"uid-b": 0},
	}, snapshot.Devices)
	// Everything on the device, worker or not, counts into its total.
	require.Equal(t, map[string]float64{"gpu-0": 65, "gpu-1": 0}, snapshot.Totals)
	// A process outside every worker is load the token buckets do not see.
	require.Equal(t, map[string]bool{"gpu-0": true}, snapshot.Unmetered)

	// A failed sample keeps the previous snapshot; its timestamp lets it age out.
	devices.processErr = errors.New("provider unavailable")
	collect()
	require.Same(t, snapshot, controller.processUtilization.Load())
}

// Utilization the provider did not supply is flagged in worker metrics instead
// of being reported as a measured zero or a negative number.
func TestWorkerMetricsFlagUnsuppliedComputeUtilization(t *testing.T) {
	devices := &fakeDeviceController{processInfos: []api.ProcessInformation{
		{ProcessID: "101", DeviceUUID: "GPU-0", MemoryUsedBytes: 1 << 30, ComputeUtilizationPercent: 35},
		{ProcessID: "201", DeviceUUID: "GPU-1", MemoryUsedBytes: 2 << 30, ComputeUtilizationUnavailable: true},
	}}
	controller := &WorkerController{
		backend: &fakeWorkerBackend{mappings: map[uint32]*framework.ProcessMappingInfo{
			101: {GuestPID: 7, PodUID: "uid-a"},
			201: {GuestPID: 9, PodUID: "uid-a"},
		}},
		deviceController: devices,
		workers:          map[string]*api.WorkerInfo{"uid-a": {WorkerUID: "uid-a", Namespace: "ns", WorkerName: "pod-a"}},
	}

	metrics, err := controller.GetWorkerMetrics()
	require.NoError(t, err)
	measured := metrics["gpu-0"]["uid-a"]["7"]
	require.NotNil(t, measured)
	require.False(t, measured.ComputeUnavailable)
	require.Equal(t, 35.0, measured.ComputePercentage)
	unsupplied := metrics["gpu-1"]["uid-a"]["9"]
	require.NotNil(t, unsupplied)
	require.True(t, unsupplied.ComputeUnavailable)
	require.Zero(t, unsupplied.ComputePercentage)
	require.Equal(t, uint64(2<<30), unsupplied.MemoryBytes, "memory is reported either way")
}

// The provider's process list is not free to obtain. The sync loop samples it
// once per interval and every other reader within that interval reuses it.
func TestProcessListIsSampledOncePerInterval(t *testing.T) {
	now := time.Unix(1_710_000_000, 0)
	devices := &fakeDeviceController{processInfos: []api.ProcessInformation{
		{ProcessID: "101", DeviceUUID: "GPU-0", MemoryUsedBytes: 1 << 30},
	}}
	controller := &WorkerController{
		backend: &fakeWorkerBackend{mappings: map[uint32]*framework.ProcessMappingInfo{
			101: {GuestPID: 7, PodUID: "uid-a"},
		}},
		deviceController: devices,
		workers:          map[string]*api.WorkerInfo{"uid-a": {WorkerUID: "uid-a", Namespace: "ns", WorkerName: "pod-a"}},
		nowFunc:          func() time.Time { return now },
	}
	metrics := func() {
		t.Helper()
		result, err := controller.GetWorkerMetrics()
		require.NoError(t, err)
		require.Len(t, result["gpu-0"]["uid-a"], 1)
	}

	// Without a recent sample a reader takes one itself.
	metrics()
	require.Equal(t, 1, devices.processCalls)
	// The sync loop always samples; readers in the same interval do not.
	controller.collectWorkerMemoryUsage(controller.buildWorkerLookupMap())
	require.Equal(t, 2, devices.processCalls)
	for range 5 {
		metrics()
	}
	require.Equal(t, 2, devices.processCalls)
	now = now.Add(sharedMemorySyncInterval)
	controller.collectWorkerMemoryUsage(controller.buildWorkerLookupMap())
	metrics()
	require.Equal(t, 3, devices.processCalls)

	// A sample that is too old is not served, and neither is one after a failure.
	now = now.Add(processSampleMaxAge + time.Millisecond)
	metrics()
	require.Equal(t, 4, devices.processCalls)
	now = now.Add(processSampleMaxAge + time.Millisecond)
	devices.processErr = errors.New("provider unavailable")
	_, err := controller.GetWorkerMetrics()
	require.Error(t, err)
}
