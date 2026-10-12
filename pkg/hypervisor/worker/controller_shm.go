package worker

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/computing"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"k8s.io/klog/v2"
)

const (
	sharedMemorySyncInterval = 500 * time.Millisecond
	shmCleanupInterval       = 5 * time.Minute
)

func (w *WorkerController) startSharedMemorySyncLoop(ctx context.Context) {
	if w.backend == nil {
		return
	}

	w.syncWG.Add(1)
	go func() {
		defer w.syncWG.Done()

		ticker := time.NewTicker(sharedMemorySyncInterval)
		defer ticker.Stop()

		for {
			w.syncSharedMemoryState()

			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
		}
	}()
}

func (w *WorkerController) stopSharedMemorySyncLoop() {
	if w.syncCancel == nil {
		return
	}

	w.syncCancel()
	w.syncWG.Wait()
	w.syncCancel = nil
}

// startSharedMemoryCleanupLoop runs a periodic cleanup of orphaned shared memory files
// for workers that no longer exist. Runs every 5 minutes.
func (w *WorkerController) startSharedMemoryCleanupLoop(ctx context.Context) {
	if w.shmBasePath == "" {
		return
	}

	w.syncWG.Add(1)
	go func() {
		defer w.syncWG.Done()

		ticker := time.NewTicker(shmCleanupInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				w.cleanupOrphanedSharedMemory()
			case <-ctx.Done():
				return
			}
		}
	}()
}

// cleanupOrphanedSharedMemory removes shared memory files for workers that no longer exist.
// Directory structure: {shmBasePath}/{namespace}/{podName}/shm
func (w *WorkerController) cleanupOrphanedSharedMemory() {
	// Keep the liveness check and unlink in the same critical section as
	// initialization and OnAdd. A stale snapshot can unlink a new Pod's file.
	w.mu.Lock()
	defer w.mu.Unlock()
	activeWorkers := make(map[string]bool)
	for _, worker := range w.workers {
		activeWorkers[worker.Namespace+"/"+worker.WorkerName] = true
	}

	namespaces, err := os.ReadDir(w.shmBasePath)
	if err != nil {
		return
	}

	cleanedCount := 0

	for _, nsEntry := range namespaces {
		if !nsEntry.IsDir() {
			continue
		}
		nsPath := filepath.Join(w.shmBasePath, nsEntry.Name())
		pods, err := os.ReadDir(nsPath)
		if err != nil {
			continue
		}
		for _, podEntry := range pods {
			if !podEntry.IsDir() {
				continue
			}
			if activeWorkers[nsEntry.Name()+"/"+podEntry.Name()] {
				continue
			}

			podPath := filepath.Join(nsPath, podEntry.Name())
			files, readErr := os.ReadDir(podPath)
			if readErr != nil {
				continue
			}
			for _, file := range files {
				if file.Name() == workerstate.ShmPathSuffix || strings.HasPrefix(file.Name(), ".shm-") {
					_ = os.Remove(filepath.Join(podPath, file.Name()))
				}
			}
			_ = os.Remove(podPath)
			cleanedCount++
		}
	}

	// Clean up stale shm handles
	for uid := range w.shmHandles {
		if _, exists := w.workers[uid]; !exists {
			w.closeSharedMemoryLocked(uid)
		}
	}

	if cleanedCount > 0 {
		klog.Infof("Shared memory cleanup: removed %d orphaned entries", cleanedCount)
	}
}

// getShmHandle returns the Go shm handle for a worker, or nil if not found.
func (w *WorkerController) getShmHandle(workerUID string) *workerstate.SharedMemoryHandle {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.shmHandles[workerUID]
}

func (w *WorkerController) syncSharedMemoryState() {
	if w.backend == nil || w.shmBasePath == "" {
		return
	}
	// Device discovery can fail during initial recovery. Retry admitted workers
	// from the current cache, without waiting for another Pod update.
	workers, _ := w.ListWorkers()
	for _, worker := range workers {
		w.recoverExistingWorkerAllocation(worker)
	}

	workerAllocations := w.workerAllocations()
	if len(workerAllocations) == 0 {
		return
	}

	workerLookup := w.buildWorkerLookupMap()
	memoryByWorkerDevice, unreadableDevices, sampledAtSecs := w.collectWorkerMemoryUsage(workerLookup)
	// The per-process sample of this tick, when there is one, also feeds the
	// usage account.
	var processes *computing.ProcessUtilizationSnapshot
	if memoryByWorkerDevice != nil {
		processes = w.processUtilization.Load()
	}
	w.integrateWorkerUsage(w.now(), workerAllocations, processes)
	now := uint64(w.now().Unix())

	for workerUID, allocation := range workerAllocations {
		workerInfo := allocation.WorkerInfo
		if workerInfo == nil {
			continue
		}
		// Retry preparation after device allocation arrives. This uses the
		// managed sync loop instead of per-Pod goroutines that outlive deletion.
		if usesSoftLimiterSharedMemory(workerInfo.IsolationMode) || workerInfo.IsolationMode == tfv1.IsolationModeHard {
			if err := w.WithWorkerSharedMemory(workerUID, nil); err != nil {
				klog.V(4).Infof("Shared memory not ready for worker %s: %v", workerUID, err)
				continue
			}
		}

		// Get shm handle for this worker
		w.mu.RLock()
		handle := w.shmHandles[workerUID]
		// Limits are also applied when they change. Applying them again here
		// costs nothing while they match, and puts right whatever does not.
		if tracked := w.workers[workerUID]; handle != nil && tracked != nil && !w.stopped {
			w.applyWorkerLimitsLocked(tracked)
		}
		w.mu.RUnlock()
		if handle == nil {
			continue
		}

		deviceMemoryUsage := memoryByWorkerDevice[workerUID]
		handle.WithState(func(state *workerstate.SharedDeviceState) {
			state.UpdateHeartbeat(now)
			// nil means collection failed; an empty, non-nil map is a valid
			// sample with no GPU processes. Keep the last usage on failure, as
			// it was published: with the time of the sample it came from.
			if memoryByWorkerDevice == nil {
				return
			}
			samples := podMemorySamples(allocation.DeviceInfos, deviceMemoryUsage, unreadableDevices, sampledAtSecs)
			for _, sample := range samples {
				publishPodMemoryUsed(state, sample.index, sample.used, sample.sampledAtSecs)
			}
		})
	}
}

// podMemorySample is what one device slot of a worker's mapping is told about
// the memory the worker holds on that device: the figure, and when the process
// sample it comes from was taken (see processSample.takenAtSecs). The time is
// zero for a figure that does not come from a sample.
type podMemorySample struct {
	index         int
	used          uint64
	sampledAtSecs float64
}

// podMemorySamples returns what one process sample, taken at sampledAtSecs,
// says about the device slots of a worker: deviceMemoryUsage is what the
// worker's processes hold per device (lower-case UUID).
//
// unreadable holds the devices whose process list could not be read in this
// sample, each with whether what was known before is still kept (see
// noteUnreadableDevices). A slot whose device is kept has no entry. Nothing is
// known about a device whose process list could not be read: what the worker
// was last seen holding there stays, for a bounded time, and so does the time
// of the sample that figure came from. Publishing the kept figure with the time
// of this sample would tell a reader that it is current, which is the one thing
// that is not known. This applies to the workers allocated on that device only;
// an unreadable device does not hold up the accounting of a worker on another
// one.
//
// Once the bound has passed, the slot gets a figure again, but one that no
// query produced: it is published without a time (sampledAtSecs zero), and the
// limiter checks against it alone.
func podMemorySamples(
	deviceInfos []*api.DeviceInfo, deviceMemoryUsage map[string]uint64, unreadable map[string]bool, sampledAtSecs float64,
) []podMemorySample {
	// Total across every physical GPU this worker's processes touched. Used
	// as a fallback for single-device pods whose process landed on a GPU
	// other than the nominally-allocated one (e.g. shared-pool pods can run
	// on any visible card), so the per-UUID lookup below would otherwise miss.
	var totalUsage uint64
	for _, used := range deviceMemoryUsage {
		totalUsage += used
	}
	singleDevice := len(deviceInfos) == 1
	samples := make([]podMemorySample, 0, len(deviceInfos))
	for _, deviceInfo := range deviceInfos {
		if deviceInfo == nil {
			continue
		}
		deviceUUID := strings.ToLower(deviceInfo.UUID)
		keep, unread := unreadable[deviceUUID]
		if keep {
			continue
		}
		used := deviceMemoryUsage[deviceUUID]
		if used == 0 && singleDevice {
			used = totalUsage
		}
		sample := podMemorySample{index: int(deviceInfo.Index), used: used, sampledAtSecs: sampledAtSecs}
		if unread {
			sample.sampledAtSecs = 0
		}
		samples = append(samples, sample)
	}
	return samples
}

// publishPodMemoryUsed is the one place that tells a worker's limiter how much
// device memory the worker holds on the device of slot index: used bytes, from
// a process sample taken at sampledAtSecs (fractional Unix seconds, read
// before the provider was queried), or a figure without a time when
// sampledAtSecs is zero.
//
// The limiter admits allocations against this figure, and the figure is as old
// as its sample: allocations made since are not in it. The time of the sample
// is published next to it (pod_memory_sampled_at of the slot), so that a
// limiter can add what its process allocated after the sample was taken.
//
// Every field of the mapping is written with an atomic store of its own:
// nothing makes two fields change together, and a reader can see the mapping
// after any of the stores. Their order is what lets it tell a figure and a time
// that belong together; it is kept in one place, SetPodMemorySample, together
// with the rule that the time of a sample differs from the one published
// before it. The sync loop is the only caller and runs in one goroutine, so
// there is one writer per slot. A slot for which nothing is published in a pass
// (see podMemorySamples) keeps both fields as they are.
func publishPodMemoryUsed(state *workerstate.SharedDeviceState, index int, used uint64, sampledAtSecs float64) bool {
	return state.SetPodMemorySample(index, used, sampledAtSecs)
}

// WithWorkerSharedMemory serializes initialization with Pod removal and name
// reuse. Both HTTP initialization and the background sync use this same handle.
func (w *WorkerController) WithWorkerSharedMemory(
	workerUID string, fn func(*workerstate.SharedDeviceState),
) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	worker := w.workers[workerUID]
	if w.stopped || worker == nil || worker.Status == api.WorkerStatusTerminated {
		return fmt.Errorf("worker %s is no longer active", workerUID)
	}
	for uid, other := range w.workers {
		if uid != workerUID && other.Namespace == worker.Namespace && other.WorkerName == worker.WorkerName &&
			other.Status != api.WorkerStatusTerminated {
			return fmt.Errorf("waiting for previous worker at %s/%s to be removed", worker.Namespace, worker.WorkerName)
		}
	}
	allocation, exists := w.allocationController.GetWorkerAllocation(workerUID)
	if !exists || allocation == nil || allocation.WorkerInfo == nil || len(allocation.DeviceInfos) == 0 {
		return fmt.Errorf("worker %s has no device allocation", workerUID)
	}
	handle := w.shmHandles[workerUID]
	if handle != nil && handle.Faulted() {
		// The backing file is writable by the workload's processes, and a
		// mapping it no longer covers is not used again. Only this worker is
		// affected: drop its mapping and publish a fresh segment below.
		klog.Warningf("Shared memory of worker %s/%s (%s) is no longer backed by its file; republishing it",
			worker.Namespace, worker.WorkerName, workerUID)
		w.closeSharedMemoryLocked(workerUID)
		handle = nil
	}
	if handle == nil {
		var err error
		handle, err = workerstate.PrepareWorkerSharedMemory(
			w.shmBasePath, workerstate.NewPodIdentifier(worker.Namespace, worker.WorkerName),
			workerUID, w.workerDeviceConfigs(worker, allocation),
			worker.Status == api.WorkerStatusRunning || worker.AllocationConfirmed,
		)
		if err != nil {
			return err
		}
		w.shmHandles[workerUID] = handle
		w.publishHeartbeatHandlesLocked()
		// The mapping is alive from this moment, for a new worker before its
		// first process starts and for a reopened one without waiting for a tick.
		heartbeat := uint64(w.now().Unix())
		handle.WithState(func(state *workerstate.SharedDeviceState) { state.UpdateHeartbeat(heartbeat) })
		// An existing mapping is opened as it is: it outlives the hypervisor
		// process, so its limits can predate the worker's latest update, a
		// hardware table change or this release. Bring them up to date at the
		// moment it is opened; later changes arrive as events. The compute
		// limits it carried are noted first: the refill rate a control loop
		// left in the mapping was published for those.
		found := make(map[int]uint32, len(allocation.DeviceInfos))
		handle.WithState(func(state *workerstate.SharedDeviceState) {
			for _, device := range allocation.DeviceInfos {
				if device == nil {
					continue
				}
				if upLimit, known := state.GetDeviceUpLimit(int(device.Index)); known {
					found[int(device.Index)] = upLimit
				}
			}
		})
		if w.foundUpLimits == nil {
			w.foundUpLimits = make(map[string]map[int]uint32)
		}
		w.foundUpLimits[workerUID] = found
		w.applyWorkerLimitsLocked(worker)
	}
	if fn != nil && !handle.WithState(fn) {
		if handle.Faulted() {
			return fmt.Errorf("shared memory for worker %s is unusable and will be republished", workerUID)
		}
		return fmt.Errorf("shared memory for worker %s is closed", workerUID)
	}
	return nil
}

func (w *WorkerController) closeSharedMemoryLocked(workerUID string) {
	if handle := w.shmHandles[workerUID]; handle != nil {
		_ = handle.Close()
		delete(w.shmHandles, workerUID)
		delete(w.foundUpLimits, workerUID)
		w.publishHeartbeatHandlesLocked()
	}
}
