package worker

import (
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/computing"
	"github.com/stretchr/testify/require"
)

// usageTestQuota is a quota controller that attributes utilization to workers.
type usageTestQuota struct {
	framework.QuotaController
	// worker UID -> share of the device (0-1); absent: no signal.
	attributed map[string]float64
}

func (q *usageTestQuota) AttributedUtilization(
	workerUID, _ string, _ time.Time, _ time.Duration,
) (float64, bool) {
	share, ok := q.attributed[workerUID]
	return share, ok
}

func usageTestAllocation(uid string, mode tfv1.IsolationModeType, devices ...string) *api.WorkerAllocation {
	allocation := &api.WorkerAllocation{WorkerInfo: &api.WorkerInfo{WorkerUID: uid, IsolationMode: mode}}
	for _, device := range devices {
		allocation.DeviceInfos = append(allocation.DeviceInfos, &api.DeviceInfo{UUID: device})
	}
	return allocation
}

func processSampleOf(
	now time.Time, percents map[string]map[string]float64, uncovered ...string,
) *computing.ProcessUtilizationSnapshot {
	processes := []api.ProcessInformation{}
	for device, workers := range percents {
		for range workers {
			processes = append(processes, api.ProcessInformation{DeviceUUID: device})
		}
		if len(workers) == 0 {
			processes = append(processes, api.ProcessInformation{DeviceUUID: device})
		}
	}
	for _, device := range uncovered {
		processes = append(processes, api.ProcessInformation{DeviceUUID: device, ComputeUtilizationUnavailable: true})
	}
	snapshot := computing.NewProcessUtilizationSnapshot(now, processes)
	for device, workers := range percents {
		for worker, percent := range workers {
			snapshot.AddWorkerProcess(worker, device, percent)
		}
	}
	return snapshot
}

// A worker's utilization is integrated over time from the best signal there is
// for it, so a reader gets the average over its own interval.
func TestWorkerComputeUsageIsIntegratedOverTime(t *testing.T) {
	quota := &usageTestQuota{attributed: map[string]float64{"soft": 0.40}}
	w := &WorkerController{quotaController: quota}
	allocations := map[string]*api.WorkerAllocation{
		"soft":     usageTestAllocation("soft", tfv1.IsolationModeSoft, "GPU-0"),
		"hard":     usageTestAllocation("hard", tfv1.IsolationModeHard, "GPU-0"),
		"idle":     usageTestAllocation("idle", tfv1.IsolationModeShared, "GPU-0"),
		"unknown":  usageTestAllocation("unknown", tfv1.IsolationModeShared, "GPU-1"),
		"two-gpus": usageTestAllocation("two-gpus", tfv1.IsolationModeHard, "GPU-0", "GPU-2"),
	}
	now := time.Unix(1_800_000_000, 0)
	sample := func(softProcesses, hard float64) *computing.ProcessUtilizationSnapshot {
		return processSampleOf(now, map[string]map[string]float64{
			// The soft worker's processes read differently from what the control
			// loop attributed to it; the attribution wins.
			"GPU-0": {"soft": softProcesses, "hard": hard, "two-gpus": 10},
			"GPU-2": {"two-gpus": 60},
		}, "GPU-1")
	}

	// The first sample only starts the clock.
	w.integrateWorkerUsage(now, allocations, sample(90, 20))
	require.Empty(t, w.GetWorkerComputeUsage())
	start := w.GetWorkerComputeUsage()

	// 10 s at 20% for the hard worker, then 10 s at 50%.
	for range 20 {
		now = now.Add(sharedMemorySyncInterval)
		w.integrateWorkerUsage(now, allocations, sample(90, 20))
	}
	middle := w.GetWorkerComputeUsage()
	quota.attributed["soft"] = 0.60
	for range 20 {
		now = now.Add(sharedMemorySyncInterval)
		w.integrateWorkerUsage(now, allocations, sample(5, 50))
	}
	end := w.GetWorkerComputeUsage()

	mean := func(from, to map[string]map[string]api.WorkerComputeUsage, device, worker string) (float64, bool) {
		return to[device][worker].MeanPercentSince(from[device][worker])
	}
	got, ok := mean(start, end, "gpu-0", "hard")
	require.True(t, ok)
	require.InDelta(t, 35, got, 1e-9, "20 s average of the hard worker")
	got, _ = mean(start, middle, "gpu-0", "hard")
	require.InDelta(t, 20, got, 1e-9)
	got, _ = mean(middle, end, "gpu-0", "hard")
	require.InDelta(t, 50, got, 1e-9)

	got, ok = mean(start, end, "gpu-0", "soft")
	require.True(t, ok)
	require.InDelta(t, 50, got, 1e-9, "the soft worker follows the control loop's attribution")

	got, ok = mean(start, end, "gpu-0", "idle")
	require.True(t, ok, "a worker without a process on a measured device is measured idle")
	require.Zero(t, got)

	_, ok = mean(start, end, "gpu-1", "unknown")
	require.False(t, ok, "a device the provider did not cover yields no measurement")
	require.NotContains(t, end, "gpu-1")

	got, _ = mean(start, end, "gpu-0", "two-gpus")
	require.InDelta(t, 10, got, 1e-9)
	got, _ = mean(start, end, "gpu-2", "two-gpus")
	require.InDelta(t, 60, got, 1e-9, "usage is kept per device")
	require.InDelta(t, 20, end["gpu-0"]["hard"].Seconds, 1e-9)

	// Without a signal the clock of that worker stops; nothing is accounted.
	delete(quota.attributed, "soft")
	before := w.GetWorkerComputeUsage()
	for range 10 {
		now = now.Add(sharedMemorySyncInterval)
		w.integrateWorkerUsage(now, allocations, nil)
	}
	after := w.GetWorkerComputeUsage()
	_, ok = mean(before, after, "gpu-0", "soft")
	require.False(t, ok, "no signal: not measured, which is not idle")
	_, ok = mean(before, after, "gpu-0", "hard")
	require.False(t, ok)

	// A soft worker the control loop has no signal for falls back to its processes.
	for range 4 {
		now = now.Add(sharedMemorySyncInterval)
		w.integrateWorkerUsage(now, allocations, sample(30, 50))
	}
	got, ok = mean(after, w.GetWorkerComputeUsage(), "gpu-0", "soft")
	require.True(t, ok)
	require.InDelta(t, 30, got, 1e-9)

	// A long pause between two samples is not accounted to either of them.
	paused := w.GetWorkerComputeUsage()
	now = now.Add(time.Minute)
	w.integrateWorkerUsage(now, allocations, sample(30, 100))
	require.Equal(t, paused, w.GetWorkerComputeUsage())

	// Workers that released the device are forgotten.
	delete(allocations, "hard")
	delete(allocations, "two-gpus")
	now = now.Add(sharedMemorySyncInterval)
	w.integrateWorkerUsage(now, allocations, sample(30, 50))
	final := w.GetWorkerComputeUsage()
	require.NotContains(t, final["gpu-0"], "hard")
	require.NotContains(t, final, "gpu-2")
	require.Contains(t, final["gpu-0"], "soft")
}

func TestWorkerMetricsCarryTflopsInUse(t *testing.T) {
	devices := &fakeDeviceController{
		devices: map[string]*api.DeviceInfo{
			"gpu-0": discoveredDevice("gpu-0", "NVIDIA GeForce RTX 3090", 0, 44.08, 71.2),
			"gpu-1": discoveredDevice("gpu-1", "NVIDIA RTX A6000", 1, 0, 0),
		},
		processInfos: []api.ProcessInformation{
			{ProcessID: "101", DeviceUUID: "GPU-0", ComputeUtilizationPercent: 25},
			{ProcessID: "102", DeviceUUID: "GPU-1", ComputeUtilizationPercent: 25},
			{ProcessID: "103", DeviceUUID: "GPU-0", ComputeUtilizationUnavailable: true},
		},
	}
	controller := &WorkerController{
		backend: &fakeWorkerBackend{mappings: map[uint32]*framework.ProcessMappingInfo{
			101: {GuestPID: 1, PodUID: "uid-a"}, 102: {GuestPID: 2, PodUID: "uid-a"}, 103: {GuestPID: 3, PodUID: "uid-a"},
		}},
		deviceController: devices,
		workers:          map[string]*api.WorkerInfo{"uid-a": {WorkerUID: "uid-a", Namespace: "ns", WorkerName: "pod-a"}},
	}
	metrics, err := controller.GetWorkerMetrics()
	require.NoError(t, err)
	// A quarter of an RTX 3090's 71.2 FP16 TFLOPS, not of the provider's 44.08.
	require.InDelta(t, 17.8, metrics["gpu-0"]["uid-a"]["1"].ComputeTflops, 1e-9)
	require.Zero(t, metrics["gpu-1"]["uid-a"]["2"].ComputeTflops, "no capacity known for the device")
	require.Zero(t, metrics["gpu-0"]["uid-a"]["3"].ComputeTflops)
	require.True(t, metrics["gpu-0"]["uid-a"]["3"].ComputeUnavailable)
}

func TestUsageMath(t *testing.T) {
	_, ok := api.WorkerComputeUsage{PercentSeconds: 100, Seconds: 10}.MeanPercentSince(
		api.WorkerComputeUsage{PercentSeconds: 100, Seconds: 10})
	require.False(t, ok)
	mean, ok := api.WorkerComputeUsage{PercentSeconds: 300, Seconds: 20}.MeanPercentSince(
		api.WorkerComputeUsage{PercentSeconds: 100, Seconds: 10})
	require.True(t, ok)
	require.Equal(t, 20.0, mean)

	device := discoveredDevice("gpu-0", "NVIDIA A100-SXM4-80GB", 0, 19.49184, 312)
	tflops, ok := api.TflopsInUse(50, device)
	require.True(t, ok)
	require.Equal(t, 156.0, tflops)
	_, ok = api.TflopsInUse(50, nil)
	require.False(t, ok)
	_, ok = api.TflopsInUse(50, &api.DeviceInfo{})
	require.False(t, ok)
	_, ok = api.TflopsInUse(-1, device)
	require.False(t, ok)
}
