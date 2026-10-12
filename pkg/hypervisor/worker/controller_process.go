package worker

import (
	"strconv"
	"strings"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/computing"
	"k8s.io/klog/v2"
)

// unreadableMemoryMaxAge is how long the memory a worker was last seen holding
// on a device is kept while that device's process list cannot be read. The
// figure gates allocations inside the workload: kept too long after the
// processes are gone it refuses memory the worker is entitled to, dropped too
// early it admits memory on top of what is still held. A process list that
// stays unreadable this long is not a passing driver hiccup (those last a
// sample or two); the device is then accounted as holding nothing for the
// worker, which is said in the log, until it can be read again.
const unreadableMemoryMaxAge = 60 * time.Second

// GetWorkerMetrics returns current worker metrics for all workers
// Returns map keyed by device UUID, then by worker UID, then by process ID
func (w *WorkerController) GetWorkerMetrics() (map[string]map[string]map[string]*api.WorkerMetrics, error) {
	// Step 1: Build worker lookup map: "namespace/podName" -> WorkerUID
	workerLookup := w.buildWorkerLookupMap()
	workerUIDs := w.buildWorkerUIDSet()

	// Step 2: Get all process information from device controller. The sync loop
	// samples it every interval; a sample that recent is reused instead of
	// querying the provider a second time.
	sample, err := w.processInformation(processSampleMaxAge)
	if err != nil {
		return nil, err
	}
	// A device whose process list could not be read has no rows here: nothing
	// is reported for its workers in this sample, which is not the same as
	// reporting them idle.
	processInfos := sample.processes

	if len(processInfos) == 0 {
		return make(map[string]map[string]map[string]*api.WorkerMetrics), nil
	}

	// Step 3: Map processes to workers and build result
	// Result structure: map[DeviceUUID]map[WorkerUID]map[ProcessID]*WorkerMetrics
	result := make(map[string]map[string]map[string]*api.WorkerMetrics)

	for _, procInfo := range processInfos {
		// Parse hostPID from ProcessID string
		hostPID, err := strconv.ParseUint(procInfo.ProcessID, 10, 32)
		if err != nil {
			klog.V(4).Infof("Failed to parse process ID %s: %v", procInfo.ProcessID, err)
			continue
		}

		// Get pod identifier from process environment using backend
		mappingInfo, err := w.backend.GetProcessMappingInfo(uint32(hostPID))
		if err != nil {
			// Process may not be a TensorFusion worker, skip silently
			klog.V(5).Infof("Failed to get process mapping info for process %d: %v", hostPID, err)
			continue
		}

		// Resolve the owning worker by the cgroup-derived pod UID; the environ
		// identity only counts for processes without one (see resolveWorkerUID).
		workerUID, found := resolveWorkerUID(mappingInfo, workerLookup, workerUIDs)
		if !found {
			klog.V(5).Infof(
				"Worker not found for process %d (ns=%q pod=%q podUID=%q)",
				hostPID, mappingInfo.Namespace, mappingInfo.PodName, mappingInfo.PodUID,
			)
			continue
		}

		// Normalize device UUID to lowercase for consistency
		deviceUUID := strings.ToLower(procInfo.DeviceUUID)

		// Initialize nested maps if needed
		if result[deviceUUID] == nil {
			result[deviceUUID] = make(map[string]map[string]*api.WorkerMetrics)
		}
		if result[deviceUUID][workerUID] == nil {
			result[deviceUUID][workerUID] = make(map[string]*api.WorkerMetrics)
		}

		// Create WorkerMetrics for this process
		// Use container PID as the process ID for display (more meaningful to users)
		processIDStr := strconv.FormatUint(uint64(mappingInfo.GuestPID), 10)
		workerMetrics := &api.WorkerMetrics{
			DeviceUUID:       deviceUUID,
			WorkerUID:        workerUID,
			ProcessID:        processIDStr,
			MemoryBytes:      procInfo.MemoryUsedBytes,
			MemoryPercentage: procInfo.MemoryUtilizationPercent,
		}
		// Utilization the provider did not supply is flagged, never reported
		// as a measured zero.
		if procInfo.ComputeUtilizationUnavailable {
			workerMetrics.ComputeUnavailable = true
		} else {
			workerMetrics.ComputePercentage = procInfo.ComputeUtilizationPercent
			// TFLOPS in use: the process's share of the device's capacity.
			if device, exists := w.deviceController.GetDevice(deviceUUID); exists {
				workerMetrics.ComputeTflops, _ = api.TflopsInUse(procInfo.ComputeUtilizationPercent, device)
			}
		}
		result[deviceUUID][workerUID][processIDStr] = workerMetrics
	}

	return result, nil
}

// buildWorkerLookupMap builds a map from "namespace/podName" to WorkerUID
// This is used to map processes back to workers
func (w *WorkerController) buildWorkerLookupMap() map[string]string {
	w.mu.RLock()
	defer w.mu.RUnlock()

	lookup := make(map[string]string, len(w.workers))
	for _, worker := range w.workers {
		// Use namespace/podName as key to look up WorkerUID
		key := worker.Namespace + "/" + worker.WorkerName
		lookup[key] = worker.WorkerUID
	}
	return lookup
}

// buildWorkerUIDSet returns the set of WorkerUIDs (== pod.UID) currently tracked
// by this hypervisor. Used to validate cgroup-derived pod UIDs during memory
// attribution.
func (w *WorkerController) buildWorkerUIDSet() map[string]struct{} {
	w.mu.RLock()
	defer w.mu.RUnlock()

	set := make(map[string]struct{}, len(w.workers))
	for _, worker := range w.workers {
		set[worker.WorkerUID] = struct{}{}
	}
	return set
}

// softWorkerUIDSet returns the workers whose compute is limited through token
// buckets.
func (w *WorkerController) softWorkerUIDSet() map[string]struct{} {
	w.mu.RLock()
	defer w.mu.RUnlock()

	set := make(map[string]struct{}, len(w.workers))
	for _, worker := range w.workers {
		if usesSoftLimiterSharedMemory(worker.IsolationMode) {
			set[worker.WorkerUID] = struct{}{}
		}
	}
	return set
}

// resolveWorkerUID maps a process to its owning worker UID. The cgroup-derived
// pod UID is authoritative: the kubelet assigns the cgroup and a process cannot
// leave it, whereas POD_NAME/POD_NAMESPACE are whatever the process was started
// with (and are dropped by some runtimes, e.g. vLLM's spawned EngineCore
// subprocess). A process in a pod cgroup therefore belongs to that pod or to no
// worker at all. The environ-derived namespace/podName key is consulted only
// when the backend read the cgroup and found the process in no pod. When it
// could not read the cgroup the pod is unknown, and the process belongs to no
// worker until it can. Returns ("", false) when the process cannot be
// attributed to a tracked worker.
func resolveWorkerUID(
	mappingInfo *framework.ProcessMappingInfo,
	workerLookup map[string]string,
	workerUIDs map[string]struct{},
) (string, bool) {
	if mappingInfo == nil {
		return "", false
	}
	if mappingInfo.PodUID != "" {
		if _, ok := workerUIDs[mappingInfo.PodUID]; ok {
			return mappingInfo.PodUID, true
		}
		return "", false
	}
	if mappingInfo.PodUIDUnknown {
		return "", false
	}
	if mappingInfo.Namespace != "" && mappingInfo.PodName != "" {
		if uid, ok := workerLookup[mappingInfo.Namespace+"/"+mappingInfo.PodName]; ok {
			return uid, true
		}
	}
	return "", false
}

// processSample is one reading of the provider's process list.
type processSample struct {
	// takenAt is the controller's clock read immediately before the provider
	// was queried. Whatever a worker's processes held before that moment is in
	// the sample; what they allocated or released while the query ran may or
	// may not be. A reader that adds what it has allocated since takenAt to a
	// figure of the sample therefore never counts too little.
	takenAt time.Time
	// processes holds the rows that are processes.
	processes []api.ProcessInformation
	// unreadable holds the devices (lower-case UUID) whose process list the
	// provider could not read in this sample. Nothing is known about their
	// processes; they are not without processes.
	unreadable map[string]bool
}

// newProcessSample separates the processes of one provider reading from the
// rows that mark a device whose process list could not be read.
func newProcessSample(takenAt time.Time, rows []api.ProcessInformation) *processSample {
	sample := &processSample{takenAt: takenAt, processes: rows}
	for i := range rows {
		if !rows[i].ProcessListUnavailable {
			continue
		}
		// Rare: only then the rows are copied without the markers.
		sample.processes = make([]api.ProcessInformation, 0, len(rows)-1)
		sample.unreadable = make(map[string]bool)
		for j := range rows {
			if rows[j].ProcessListUnavailable {
				sample.unreadable[strings.ToLower(rows[j].DeviceUUID)] = true
			} else {
				sample.processes = append(sample.processes, rows[j])
			}
		}
		break
	}
	return sample
}

// takenAtSecs is takenAt as fractional seconds of the wall clock since the Unix
// epoch: the clock and the representation of the refill timestamps in the
// workers' mappings (computing.Controller.nowSecs), so that a limiter can
// compare it with its own clock readings the same way.
func (s *processSample) takenAtSecs() float64 {
	return float64(s.takenAt.UnixMicro()) / 1e6
}

// processSampleMaxAge is how old a sample may be for an on-demand reader. The
// sync loop takes a fresh one every sharedMemorySyncInterval.
const processSampleMaxAge = 2 * sharedMemorySyncInterval

// processInformation returns the provider's process list. A sample younger
// than maxAge is reused, so the provider is queried once per sync interval no
// matter how many readers there are; maxAge 0 always takes a fresh one. The
// result is shared and must not be modified.
//
// The provider call reads the driver's utilization samples, which takes
// milliseconds per device. It is made here and nowhere else: once per sync
// interval, never from the refill or heartbeat loops.
func (w *WorkerController) processInformation(maxAge time.Duration) (*processSample, error) {
	// Read before the provider is queried, never after: see processSample.takenAt.
	now := w.now()
	if maxAge > 0 {
		if sample := w.processSample.Load(); sample != nil {
			if age := now.Sub(sample.takenAt); age >= 0 && age <= maxAge {
				return sample, nil
			}
		}
	}
	rows, err := w.deviceController.GetProcessInformation()
	if err != nil {
		return nil, err
	}
	sample := newProcessSample(now, rows)
	w.processSample.Store(sample)
	return sample, nil
}

// noteUnreadableDevices keeps track of the devices whose process list cannot
// be read, reporting each change once, and returns them, each with whether the
// previous memory accounting is still to be kept: true for those that became
// unreadable less than unreadableMemoryMaxAge ago, false for those past the
// bound. A device that is not in the result was read.
func (w *WorkerController) noteUnreadableDevices(unreadable map[string]bool, now time.Time) map[string]bool {
	w.unreadableMu.Lock()
	defer w.unreadableMu.Unlock()
	var keep map[string]bool
	for deviceUUID := range unreadable {
		if keep == nil {
			keep = make(map[string]bool, len(unreadable))
		}
		since, known := w.unreadableSince[deviceUUID]
		if !known {
			if w.unreadableSince == nil {
				w.unreadableSince = make(map[string]time.Time)
			}
			since = now
			w.unreadableSince[deviceUUID] = now
			klog.Warningf("Process list of device %s could not be read: keeping the memory accounting of its workers "+
				"for up to %s and measuring their utilization by other means until it can", deviceUUID, unreadableMemoryMaxAge)
		}
		if age := now.Sub(since); age >= 0 && age <= unreadableMemoryMaxAge {
			keep[deviceUUID] = true
			continue
		}
		keep[deviceUUID] = false
		if !w.unreadableExpired[deviceUUID] {
			if w.unreadableExpired == nil {
				w.unreadableExpired = make(map[string]bool)
			}
			w.unreadableExpired[deviceUUID] = true
			klog.Warningf("Process list of device %s has been unreadable for more than %s: its workers are accounted "+
				"as holding no memory on it until it can be read again", deviceUUID, unreadableMemoryMaxAge)
		}
	}
	for deviceUUID := range w.unreadableSince {
		if !unreadable[deviceUUID] {
			delete(w.unreadableSince, deviceUUID)
			delete(w.unreadableExpired, deviceUUID)
			klog.Infof("Process list of device %s can be read again", deviceUUID)
		}
	}
	w.unreadableDevices = unreadable
	return keep
}

// collectWorkerMemoryUsage takes this tick's process sample and returns the
// device memory held per worker and device, and the devices (lower-case UUID)
// whose process list could not be read, true for those whose previous
// accounting is to be kept and false for those past the bound (see
// noteUnreadableDevices): nothing is known about what the workers hold on
// either. The memory map is nil when no sample could be taken.
// sampledAtSecs is when the sample behind the figures was taken (see
// processSample.takenAtSecs); it travels with them to where they are published.
func (w *WorkerController) collectWorkerMemoryUsage(
	workerLookup map[string]string,
) (memory map[string]map[string]uint64, keepPrevious map[string]bool, sampledAtSecs float64) {
	sample, err := w.processInformation(0)
	if err != nil {
		// Worth a line when it starts, not twice a second.
		if reason := err.Error(); w.processSampleError.Swap(&reason) == nil {
			klog.Warningf("Process information unavailable, keeping the last memory accounting: %v", err)
		}
		klog.V(4).Infof("Failed to collect process information for shared memory sync: %v", err)
		return nil, nil, 0
	}
	if w.processSampleError.Swap(nil) != nil {
		klog.Infof("Process information is available again")
	}
	processInfos := sample.processes
	keepPrevious = w.noteUnreadableDevices(sample.unreadable, w.now())

	workerUIDs := w.buildWorkerUIDSet()
	softWorkers := w.softWorkerUIDSet()
	result := make(map[string]map[string]uint64)
	// The same sample also carries per-process compute utilization, which the
	// quota controller needs per worker. A failed sample keeps the previous
	// snapshot, which then simply ages out.
	utilization := computing.NewProcessUtilizationSnapshot(w.now(), processInfos)
	// What ran unmetered on a device that cannot be read now is still assumed
	// to run there.
	if previous := w.processUtilization.Load(); previous != nil {
		for deviceUUID := range sample.unreadable {
			if previous.Unmetered[deviceUUID] {
				utilization.MarkUnmetered(deviceUUID)
			}
		}
	}
	for _, procInfo := range processInfos {
		hostPID, err := strconv.ParseUint(procInfo.ProcessID, 10, 32)
		if err != nil {
			klog.V(4).Infof("Failed to parse process ID %s during shared memory sync: %v", procInfo.ProcessID, err)
			continue
		}

		mappingInfo, err := w.backend.GetProcessMappingInfo(uint32(hostPID))
		if err != nil || mappingInfo == nil {
			// Whatever it is, no token bucket accounts for what it does.
			utilization.MarkUnmetered(procInfo.DeviceUUID)
			continue
		}
		// Resolve the owning worker by the cgroup-derived pod UID, so usage is
		// charged to the pod the process really runs in. This also covers a
		// GPU-holding process that stripped its environment (e.g. vLLM's spawned
		// EngineCore drops POD_NAME/POD_NAMESPACE), whose memory would otherwise
		// never be attributed and the pod's nvidia-smi would report 0.
		workerUID, found := resolveWorkerUID(mappingInfo, workerLookup, workerUIDs)
		if !found {
			utilization.MarkUnmetered(procInfo.DeviceUUID)
			continue
		}
		if _, soft := softWorkers[workerUID]; !soft {
			utilization.MarkUnmetered(procInfo.DeviceUUID)
		}

		deviceUUID := strings.ToLower(procInfo.DeviceUUID)
		if result[workerUID] == nil {
			result[workerUID] = make(map[string]uint64)
		}
		result[workerUID][deviceUUID] += procInfo.MemoryUsedBytes
		utilization.AddWorkerProcess(workerUID, deviceUUID, procInfo.ComputeUtilizationPercent)
	}
	// Published only once complete: readers never see it change.
	w.processUtilization.Store(utilization)

	return result, keepPrevious, sample.takenAtSecs()
}
