package worker

import (
	"strconv"
	"strings"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/samber/lo"
	"k8s.io/klog/v2"
)

// computeUpLimit is the compute limit of a worker on a device as a percentage
// (1-100) of that device. It is the only place the hypervisor derives one, for
// every isolation mode:
//
//   - A hard-isolated worker carries the percentage the scheduler derived when
//     it placed the worker; that one is used when present.
//   - A limit given as a percentage is used as is.
//   - A limit given in TFLOPS is a share of the device's TFLOPS capacity
//     (api.DeviceInfo.TflopsCapacity, the capacity published on the GPU
//     resource), converted by the function the scheduler uses. Deriving it here
//     keeps in-place limit changes correct without a second writer.
//
// Without a limit, or without a known capacity to convert against, the worker
// is not limited.
func computeUpLimit(workerInfo *api.WorkerInfo, deviceInfo *api.DeviceInfo) uint32 {
	if workerInfo == nil {
		return 100
	}
	if workerInfo.IsolationMode == tfv1.IsolationModeHard {
		if raw := workerInfo.Annotations[constants.EffectiveHardSMPercentAnnotation]; raw != "" {
			if parsed, err := strconv.ParseUint(raw, 10, 32); err == nil && parsed >= 1 && parsed <= 100 {
				return uint32(parsed)
			}
		}
	}
	if workerInfo.Limits.ComputePercent.Value() > 0 {
		return uint32(workerInfo.Limits.ComputePercent.Value())
	}
	if deviceInfo == nil {
		return 100
	}
	if percent, ok := utils.TflopsToComputePercent(
		workerInfo.Limits.Tflops.AsApproximateFloat64(), deviceInfo.EffectiveTflopsCapacity(),
	); ok {
		return percent
	}
	return 100
}

// computeBurstWindow is the compute burst window a worker asks for itself, in
// seconds, or nil when it asks for none and the pool's applies (see
// constants.ComputeBurstWindowAnnotation). The admission webhook only lets a
// usable value through, but the annotation can be changed afterwards: a value
// that is negative or above the maximum is brought to the nearest bound, one
// that is not a duration counts as not set, and either is logged once per
// value.
func (w *WorkerController) computeBurstWindow(worker *api.WorkerInfo) *float64 {
	window, set, err := utils.ComputeBurstWindow(worker.Annotations)
	if err == nil {
		w.burstWindowReported.Delete(worker.WorkerUID)
	} else {
		raw := worker.Annotations[constants.ComputeBurstWindowAnnotation]
		if previous, reported := w.burstWindowReported.Swap(worker.WorkerUID, raw); !reported || previous != raw {
			applied := "the burst window of the pool applies"
			if set {
				applied = "a burst window of " + window.String() + " applies"
			}
			klog.Warningf("Worker %s/%s: %v; %s", worker.Namespace, worker.WorkerName, err, applied)
		}
	}
	if !set {
		return nil
	}
	seconds := window.Seconds()
	return &seconds
}

// allocatedDevices returns the devices of an allocation with the TFLOPS
// capacity the device controller holds for them now. An allocation keeps each
// device as it was when the worker was admitted, while the capacity can be
// re-resolved later (the hardware table changed). Everything else, the slot
// index in particular, stays as allocated.
func (w *WorkerController) allocatedDevices(allocation *api.WorkerAllocation) []*api.DeviceInfo {
	if allocation == nil {
		return nil
	}
	devices := make([]*api.DeviceInfo, 0, len(allocation.DeviceInfos))
	for _, allocated := range allocation.DeviceInfos {
		if allocated == nil {
			continue
		}
		if w.deviceController != nil {
			current, exists := w.deviceController.GetDevice(allocated.UUID)
			if exists && current != nil && (current.TflopsCapacity != allocated.TflopsCapacity ||
				current.TflopsCapacitySource != allocated.TflopsCapacitySource) {
				refreshed := *allocated
				refreshed.TflopsCapacity = current.TflopsCapacity
				refreshed.TflopsCapacitySource = current.TflopsCapacitySource
				allocated = &refreshed
			}
		}
		devices = append(devices, allocated)
	}
	return devices
}

// refreshWorkerLimits applies an in-place change of a running worker's limits.
// The allocation and the shared-memory mapping are created once, when the
// worker is admitted, so both are brought up to date here.
func (w *WorkerController) refreshWorkerLimits(worker *api.WorkerInfo) {
	if worker == nil || worker.Status == api.WorkerStatusTerminated {
		return
	}
	// The lock keeps a newer update from being overwritten by an older one:
	// only the worker that is currently tracked is applied.
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.stopped || w.workers[worker.WorkerUID] != worker {
		return
	}
	if refresher, ok := w.allocationController.(framework.WorkerAllocationRefresher); ok {
		refresher.RefreshWorkerInfo(worker)
	}
	w.applyWorkerLimitsLocked(worker)
}

// onDeviceUpdate re-derives the limits of the workers on a device whose TFLOPS
// capacity changed: a limit in TFLOPS is a share of that capacity.
func (w *WorkerController) onDeviceUpdate(oldDevice, newDevice *api.DeviceInfo) {
	if oldDevice == nil || newDevice == nil || oldDevice.TflopsCapacity == newDevice.TflopsCapacity {
		return
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.stopped {
		return
	}
	for _, worker := range w.workers {
		if worker == nil || worker.Status == api.WorkerStatusTerminated ||
			!lo.Contains(worker.AllocatedDevices, newDevice.UUID) {
			continue
		}
		w.applyWorkerLimitsLocked(worker)
	}
}

// limitWriter derives a worker's limits and writes them to its mapping in one
// critical section, so that of two applications racing for the same worker the
// one that writes last also derived last: an older derivation can never
// overwrite a newer one. It is reused for every application and allocates
// nothing; mu guards every field and is taken with w.mu held, never before it.
type limitWriter struct {
	mu         sync.Mutex
	controller *WorkerController
	worker     *api.WorkerInfo
	allocation *api.WorkerAllocation
	// write bound once, so that handing it to the mapping allocates nothing.
	apply func(*workerstate.SharedDeviceState)
}

func (l *limitWriter) write(state *workerstate.SharedDeviceState) {
	for _, allocated := range l.allocation.DeviceInfos {
		if allocated == nil {
			continue
		}
		// The allocation keeps each device as it was when the worker was
		// admitted; the capacity a TFLOPS limit is a share of is the one the
		// device controller holds now.
		current := allocated
		if l.controller.deviceController != nil {
			if device, exists := l.controller.deviceController.GetDevice(allocated.UUID); exists && device != nil {
				current = device
			}
		}
		upLimit, memLimit := computeUpLimit(l.worker, current), workerMemoryLimit(l.worker, allocated)
		if state.SetDeviceLimits(int(allocated.Index), upLimit, memLimit) {
			klog.Infof("Updated limits of worker %s/%s device %d: compute=%d%% memory=%d bytes",
				l.worker.Namespace, l.worker.WorkerName, allocated.Index, upLimit, memLimit)
		}
	}
}

// applyWorkerLimitsLocked publishes the worker's limits to its soft limiter
// mapping without recreating it. The caller must hold w.mu (shared or
// exclusive) and pass the worker that is tracked under it. Hard and shared
// workers are skipped: their limits take effect when the container starts.
//
// It runs when a mapping is opened, on every event that can change a limit
// (worker update, device capacity update), and in every pass of the sync loop.
// Each run derives the limits from what the tracked worker and the device
// controller say at that moment and writes them under limitWriter.mu, so the
// order in which events are delivered does not matter, and a limit that does
// not match the current state for any reason is put right within one sync
// interval. Writing an unchanged limit is skipped by the mapping, and a run
// allocates nothing.
func (w *WorkerController) applyWorkerLimitsLocked(worker *api.WorkerInfo) {
	if !usesSoftLimiterSharedMemory(worker.IsolationMode) {
		return
	}
	handle := w.shmHandles[worker.WorkerUID]
	if handle == nil {
		return
	}
	allocation, exists := w.allocationController.GetWorkerAllocation(worker.WorkerUID)
	if !exists || allocation == nil || len(allocation.DeviceInfos) == 0 {
		return
	}
	limits := &w.limits
	limits.mu.Lock()
	if limits.apply == nil {
		limits.controller, limits.apply = w, limits.write
	}
	limits.worker, limits.allocation = worker, allocation
	handle.WithState(limits.apply)
	limits.worker, limits.allocation = nil, nil
	limits.mu.Unlock()
}

// workerDeviceConfigs is the limiter configuration of a worker as it stands
// now: the devices are fixed by the allocation, the limits come from the
// tracked worker, and the capacity they are converted with from the device
// controller.
func (w *WorkerController) workerDeviceConfigs(
	worker *api.WorkerInfo, allocation *api.WorkerAllocation,
) []workerstate.DeviceConfig {
	return buildWorkerDeviceConfigs(&api.WorkerAllocation{
		WorkerInfo: worker, DeviceInfos: w.allocatedDevices(allocation),
	})
}

// workerMemoryLimit is the device memory a worker may use on a device: its
// VRAM limit, or all of the device's memory without one.
func workerMemoryLimit(worker *api.WorkerInfo, device *api.DeviceInfo) uint64 {
	if limit := worker.Limits.Vram.Value(); limit > 0 {
		return uint64(limit)
	}
	return device.TotalMemoryBytes
}

func buildWorkerDeviceConfigs(allocation *api.WorkerAllocation) []workerstate.DeviceConfig {
	if allocation == nil || allocation.WorkerInfo == nil {
		return nil
	}
	configs := make([]workerstate.DeviceConfig, 0, len(allocation.DeviceInfos))
	for _, deviceInfo := range allocation.DeviceInfos {
		if deviceInfo == nil {
			continue
		}
		memLimit := workerMemoryLimit(allocation.WorkerInfo, deviceInfo)
		smCount := uint32(0)
		if deviceInfo.Properties != nil {
			if v, err := strconv.ParseUint(deviceInfo.Properties["totalComputeUnits"], 10, 32); err == nil {
				smCount = uint32(v)
			}
		}
		deviceUUID := normalizeDeviceUUID(deviceInfo.UUID)
		totalCores := smCount * 128
		if allocation.WorkerInfo.IsolationMode != tfv1.IsolationModeSoft {
			// Preserve the legacy hard/shared client's UUID and CUDA core layout.
			deviceUUID = deviceInfo.UUID
			if strings.HasPrefix(deviceUUID, "gpu-") {
				deviceUUID = "GPU-" + strings.TrimPrefix(deviceUUID, "gpu-")
			}
			totalCores = smCount * coresPerSM(deviceInfo.Properties["computeCapability"])
		}
		configs = append(configs, workerstate.DeviceConfig{
			DeviceIdx:  uint32(deviceInfo.Index),
			DeviceUUID: deviceUUID,
			UpLimit:    computeUpLimit(allocation.WorkerInfo, deviceInfo),
			MemLimit:   memLimit,
			SMCount:    totalCores,
		})
	}
	return configs
}

func coresPerSM(computeCapability string) uint32 {
	parts := strings.Split(strings.TrimSpace(computeCapability), ".")
	if len(parts) != 2 {
		return 0
	}

	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0
	}
	minor, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0
	}

	switch (major * 10) + minor {
	case 20:
		return 32
	case 21:
		return 48
	case 30, 32, 35, 37:
		return 192
	case 50, 52, 53:
		return 128
	case 60:
		return 64
	case 61, 62:
		return 128
	case 70, 72, 75, 80:
		return 64
	case 86, 87, 89, 90, 100, 101, 103, 110, 120, 121:
		return 128
	default:
		return 0
	}
}

func normalizeDeviceUUID(uuid string) string {
	if strings.HasPrefix(uuid, "gpu-") {
		return strings.ToUpper(strings.TrimPrefix(uuid, "gpu-"))
	}
	if strings.HasPrefix(uuid, "GPU-") {
		return strings.TrimPrefix(uuid, "GPU-")
	}
	return strings.ToUpper(uuid)
}
