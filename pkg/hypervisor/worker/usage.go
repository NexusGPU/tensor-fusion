package worker

import (
	"math"
	"strings"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/computing"
)

// Compute usage per worker.
//
// A worker's utilization of a device is sampled every sync interval and
// integrated over time, so that whoever reports it (usage metrics once a
// minute) reports the average over its own interval instead of whatever the
// device happened to be doing in the instant it looked.
//
// The utilization of one sample is, per worker and device:
//
//  1. For a soft-isolated worker, what the quota control loop attributed to it
//     in its latest interval: the worker's own share of the device, derived
//     from per-process utilization calibrated against the tokens it spent, from
//     its share of the tokens spent on the device, or from the device's
//     utilization when it is alone on it.
//  2. Otherwise the sum of the utilization the provider reports for the
//     worker's processes on the device, provided it supplied that for the
//     device. A worker without a process on the device is using none of it.
//  3. Otherwise nothing: the sample is skipped and no time is accounted, so the
//     interval reads as "not measured" rather than as idle.

const (
	// A gap between two samples longer than this is not accounted to either.
	usageSampleMaxGap = 4 * sharedMemorySyncInterval
	// The control loop runs every 500 ms; an attribution older than this no
	// longer describes the present.
	attributedUtilizationMaxAge = 2 * time.Second
)

var _ framework.WorkerComputeUsageProvider = &WorkerController{}

// attributedUtilizationSource is what the quota controller offers for (1).
type attributedUtilizationSource interface {
	AttributedUtilization(workerUID, deviceUUID string, now time.Time, maxAge time.Duration) (float64, bool)
}

// workerUsageTracker holds the accumulated usage; its zero value is ready.
type workerUsageTracker struct {
	mu         sync.Mutex
	lastSample time.Time
	// device UUID (lower case) -> worker UID -> usage
	usage map[string]map[string]*api.WorkerComputeUsage
}

// GetWorkerComputeUsage implements framework.WorkerComputeUsageProvider.
func (w *WorkerController) GetWorkerComputeUsage() map[string]map[string]api.WorkerComputeUsage {
	w.usage.mu.Lock()
	defer w.usage.mu.Unlock()
	result := make(map[string]map[string]api.WorkerComputeUsage, len(w.usage.usage))
	for deviceUUID, workers := range w.usage.usage {
		result[deviceUUID] = make(map[string]api.WorkerComputeUsage, len(workers))
		for workerUID, usage := range workers {
			result[deviceUUID][workerUID] = *usage
		}
	}
	return result
}

// integrateWorkerUsage accounts the time since the previous sample to the
// utilization every allocated worker has now. processes is this tick's
// per-process sample, nil when it could not be taken.
func (w *WorkerController) integrateWorkerUsage(
	now time.Time, allocations map[string]*api.WorkerAllocation, processes *computing.ProcessUtilizationSnapshot,
) {
	attributed, _ := w.quotaController.(attributedUtilizationSource)

	w.usage.mu.Lock()
	defer w.usage.mu.Unlock()
	elapsed := now.Sub(w.usage.lastSample).Seconds()
	if w.usage.lastSample.IsZero() || elapsed <= 0 || elapsed > usageSampleMaxGap.Seconds() {
		elapsed = 0
	}
	w.usage.lastSample = now

	seen := make(map[string]map[string]bool, len(w.usage.usage))
	for workerUID, allocation := range allocations {
		if allocation == nil || allocation.WorkerInfo == nil {
			continue
		}
		for _, deviceInfo := range allocation.DeviceInfos {
			if deviceInfo == nil {
				continue
			}
			deviceUUID := strings.ToLower(deviceInfo.UUID)
			if seen[deviceUUID] == nil {
				seen[deviceUUID] = make(map[string]bool)
			}
			seen[deviceUUID][workerUID] = true

			percent, measured := 0.0, false
			if attributed != nil && usesSoftLimiterSharedMemory(allocation.WorkerInfo.IsolationMode) {
				if share, ok := attributed.AttributedUtilization(workerUID, deviceUUID, now, attributedUtilizationMaxAge); ok {
					percent, measured = share*100, true
				}
			}
			if !measured && processes != nil {
				if workers := processes.Devices[deviceUUID]; workers != nil {
					percent, measured = workers[workerUID], true
				}
			}
			if !measured || elapsed == 0 {
				continue
			}
			if w.usage.usage == nil {
				w.usage.usage = make(map[string]map[string]*api.WorkerComputeUsage)
			}
			if w.usage.usage[deviceUUID] == nil {
				w.usage.usage[deviceUUID] = make(map[string]*api.WorkerComputeUsage)
			}
			usage := w.usage.usage[deviceUUID][workerUID]
			if usage == nil {
				usage = &api.WorkerComputeUsage{}
				w.usage.usage[deviceUUID][workerUID] = usage
			}
			usage.PercentSeconds += math.Min(math.Max(percent, 0), 100) * elapsed
			usage.Seconds += elapsed
		}
	}

	// Forget workers that no longer hold the device.
	for deviceUUID, workers := range w.usage.usage {
		for workerUID := range workers {
			if !seen[deviceUUID][workerUID] {
				delete(workers, workerUID)
			}
		}
		if len(workers) == 0 {
			delete(w.usage.usage, deviceUUID)
		}
	}
}
