package worker

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/computing"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/samber/lo"
	"k8s.io/klog/v2"
)

const (
	sharedMemorySyncInterval = 500 * time.Millisecond
	shmCleanupInterval       = 5 * time.Minute
)

type WorkerController struct {
	mode    api.IsolationMode
	policy  tfv1.IsolationModePolicyType
	backend framework.Backend

	deviceController     framework.DeviceController
	allocationController framework.WorkerAllocationController
	quotaController      framework.QuotaController

	mu         sync.RWMutex
	workers    map[string]*api.WorkerInfo
	shmHandles map[string]*workerstate.SharedMemoryHandle // workerUID -> shm handle
	stopped    bool

	shmBasePath string
	nowFunc     func() time.Time

	syncCancel context.CancelFunc
	syncWG     sync.WaitGroup
}

func NewWorkerController(
	deviceController framework.DeviceController,
	allocationController framework.WorkerAllocationController,
	mode api.IsolationMode,
	backend framework.Backend,
) framework.WorkerController {
	return NewWorkerControllerWithPolicy(
		deviceController, allocationController, mode, tfv1.IsolationModePolicyStatic, backend,
	)
}

func NewWorkerControllerWithPolicy(
	deviceController framework.DeviceController,
	allocationController framework.WorkerAllocationController,
	mode api.IsolationMode,
	policy tfv1.IsolationModePolicyType,
	backend framework.Backend,
) framework.WorkerController {
	quotaController := computing.NewQuotaController(deviceController, backend)

	wc := &WorkerController{
		deviceController:     deviceController,
		allocationController: allocationController,
		mode:                 mode,
		policy:               policy,
		backend:              backend,
		quotaController:      quotaController,

		workers:    make(map[string]*api.WorkerInfo, 32),
		shmHandles: make(map[string]*workerstate.SharedMemoryHandle, 8),
		shmBasePath: filepath.Join(
			constants.TFDataPath,
			strings.TrimPrefix(constants.SharedMemMountSubPath, "/"),
		),
		nowFunc: time.Now,
	}

	// Wire up providers so QuotaController can access worker allocations and shm handles
	if qc, ok := quotaController.(*computing.Controller); ok {
		qc.SetWorkerInfoProvider(wc.buildWorkerInfoSnapshots)
		qc.SetShmHandleProvider(wc.getShmHandle)
	}

	return wc
}

func (w *WorkerController) workerChangeHandler() framework.WorkerChangeHandler {
	return framework.WorkerChangeHandler{
		OnPrepare: func(workerUID string) error { return w.WithWorkerSharedMemory(workerUID, nil) },
		OnAdd: func(worker *api.WorkerInfo) {
			w.mu.Lock()
			w.workers[worker.WorkerUID] = worker
			if worker.Status == api.WorkerStatusTerminated {
				w.closeSharedMemoryLocked(worker.WorkerUID)
			}
			w.mu.Unlock()
			// A busy backend may observe only the final Pod state. Allocation
			// can already exist even though no earlier Add reached this handler.
			if worker.Status == api.WorkerStatusTerminated {
				if err := w.allocationController.DeallocateWorker(worker.WorkerUID); err != nil {
					klog.Errorf("Failed to deallocate terminated worker %s: %v", worker.WorkerUID, err)
				}
				return
			}

			w.recoverExistingWorkerAllocation(worker)

			// Existing soft processes keep their limiter state in shared memory.
			// Reopen that mapping after a hypervisor restart so heartbeat updates continue;
			// otherwise already-running CUDA kernels are eventually denied as unhealthy.
			if usesSoftLimiterSharedMemory(worker.IsolationMode) || worker.IsolationMode == tfv1.IsolationModeHard {
				_ = w.WithWorkerSharedMemory(worker.WorkerUID, nil)
			}
		},
		OnRemove: func(worker *api.WorkerInfo) {
			// Fence initialization before deallocating: a delayed HTTP request or
			// sync pass must not recreate shared memory for a removed worker.
			w.mu.Lock()
			delete(w.workers, worker.WorkerUID)
			w.closeSharedMemoryLocked(worker.WorkerUID)
			w.mu.Unlock()
			// Deallocate worker devices first
			if err := w.allocationController.DeallocateWorker(worker.WorkerUID); err != nil {
				klog.Errorf("Failed to deallocate worker %s: %v", worker.WorkerUID, err)
			}
			// Drop per-worker ERL/PID state so it cannot bias the next pod
			// scheduled on the same GPU. Safe to call for any isolation mode;
			// the quota controller handles non-soft modes as a no-op internally.
			if w.quotaController != nil {
				w.quotaController.CleanupWorker(worker.WorkerUID)
			}
		},
		OnUpdate: func(oldWorker, newWorker *api.WorkerInfo) {
			w.mu.Lock()
			w.workers[newWorker.WorkerUID] = newWorker
			if newWorker.Status == api.WorkerStatusTerminated {
				w.closeSharedMemoryLocked(newWorker.WorkerUID)
			}
			w.mu.Unlock()
			// Check if worker transitioned to Terminated state (Succeeded or Failed)
			// If so, deallocate devices including partitions
			if oldWorker.Status != api.WorkerStatusTerminated && newWorker.Status == api.WorkerStatusTerminated {
				if err := w.allocationController.DeallocateWorker(newWorker.WorkerUID); err != nil {
					klog.Errorf("Failed to deallocate worker %s on termination: %v", newWorker.WorkerUID, err)
				}
			}
			w.recoverExistingWorkerAllocation(newWorker)
		},
	}
}

func (w *WorkerController) Start() error {
	err := w.backend.RegisterWorkerUpdateHandler(w.workerChangeHandler())
	if err != nil {
		return err
	}

	// Only soft isolation consumes the shared-memory ERL token bucket.
	if w.policy == tfv1.IsolationModePolicyDynamic || usesSoftLimiterSharedMemory(w.mode) {
		if err := w.quotaController.StartSoftQuotaLimiter(); err != nil {
			klog.Fatalf("Failed to start soft quota limiter: %v", err)
		}
		klog.Info("Soft quota limiter started")
	}

	// Start backend after all handlers are registered
	err = w.backend.Start()
	if err != nil {
		return err
	}
	klog.Info("Worker backend started")

	ctx, cancel := context.WithCancel(context.Background())
	w.syncCancel = cancel
	w.startSharedMemorySyncLoop(ctx)
	w.startSharedMemoryCleanupLoop(ctx)
	return nil
}

func usesSoftLimiterSharedMemory(mode tfv1.IsolationModeType) bool {
	return mode == tfv1.IsolationModeSoft
}

// recoverExistingWorkerAllocation restores admitted Pods after restart. Kubelet
// may reuse a checkpointed allocation while the Pod is still Pending. New Pods
// without that evidence must continue through normal device-plugin admission.
func (w *WorkerController) recoverExistingWorkerAllocation(worker *api.WorkerInfo) {
	if worker == nil || worker.Status == api.WorkerStatusTerminated || len(worker.AllocatedDevices) == 0 ||
		(worker.Status != api.WorkerStatusRunning && !worker.AllocationConfirmed) {
		return
	}
	// Serialize recovery with removal, including background retries using an
	// older snapshot. A late retry must never resurrect a deleted allocation.
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.stopped || w.workers[worker.WorkerUID] != worker {
		return
	}
	if _, exists := w.allocationController.GetWorkerAllocation(worker.WorkerUID); exists {
		return
	}

	if worker.IsolationMode == tfv1.IsolationModePartitioned && worker.PartitionTemplateID != "" {
		if partitionUUIDs := worker.Annotations[constants.PartitionUUIDsAnnotation]; partitionUUIDs != "" {
			if err := w.allocationController.RecoverPartitionedWorker(worker, partitionUUIDs); err != nil {
				klog.Errorf("Failed to recover partitioned allocation for existing worker %s/%s: %v",
					worker.Namespace, worker.WorkerName, err)
			}
		}
		return
	}

	if _, err := w.allocationController.AllocateWorkerDevices(worker); err != nil {
		klog.Errorf("Failed to recover allocation for existing worker %s/%s: %v",
			worker.Namespace, worker.WorkerName, err)
		return
	}
	klog.Infof("Recovered allocation for existing worker %s/%s", worker.Namespace, worker.WorkerName)
}

func (w *WorkerController) Stop() error {
	w.mu.Lock()
	w.stopped = true
	w.mu.Unlock()
	w.stopSharedMemorySyncLoop()
	_ = w.backend.Stop()
	_ = w.quotaController.StopSoftQuotaLimiter()
	w.mu.Lock()
	defer w.mu.Unlock()
	for uid := range w.shmHandles {
		w.closeSharedMemoryLocked(uid)
	}
	return nil
}

func (w *WorkerController) ListWorkers() ([]*api.WorkerInfo, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return lo.Values(w.workers), nil
}

// GetWorkerMetrics returns current worker metrics for all workers
// Returns map keyed by device UUID, then by worker UID, then by process ID
func (w *WorkerController) GetWorkerMetrics() (map[string]map[string]map[string]*api.WorkerMetrics, error) {
	// Step 1: Build worker lookup map: "namespace/podName" -> WorkerUID
	workerLookup := w.buildWorkerLookupMap()
	workerUIDs := w.buildWorkerUIDSet()

	// Step 2: Get all process information from device controller
	processInfos, err := w.deviceController.GetProcessInformation()
	if err != nil {
		return nil, err
	}

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

		// Resolve the owning worker by environ identity, falling back to the
		// cgroup-derived pod UID for processes that stripped POD_NAME/POD_NAMESPACE
		// from their environment (e.g. vLLM's spawned EngineCore subprocess).
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
		result[deviceUUID][workerUID][processIDStr] = &api.WorkerMetrics{
			DeviceUUID:        deviceUUID,
			WorkerUID:         workerUID,
			ProcessID:         processIDStr,
			MemoryBytes:       procInfo.MemoryUsedBytes,
			MemoryPercentage:  procInfo.MemoryUtilizationPercent,
			ComputePercentage: procInfo.ComputeUtilizationPercent,
			// ComputeTflops can be calculated if we have device max TFlops and utilization
			// For now, leave it as 0 since we don't have that info here
			ComputeTflops: 0,
		}
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
// attribution when a process has no usable POD_NAME in its environment.
func (w *WorkerController) buildWorkerUIDSet() map[string]struct{} {
	w.mu.RLock()
	defer w.mu.RUnlock()

	set := make(map[string]struct{}, len(w.workers))
	for _, worker := range w.workers {
		set[worker.WorkerUID] = struct{}{}
	}
	return set
}

// resolveWorkerUID maps a process to its owning worker UID. It prefers the
// environ-derived namespace/podName key, then falls back to the cgroup-derived
// pod UID (environ-independent) for processes that stripped POD_NAME/POD_NAMESPACE
// from their environment (e.g. vLLM's spawned EngineCore subprocess). Returns
// ("", false) when the process cannot be attributed to a tracked worker.
func resolveWorkerUID(
	mappingInfo *framework.ProcessMappingInfo,
	workerLookup map[string]string,
	workerUIDs map[string]struct{},
) (string, bool) {
	if mappingInfo == nil {
		return "", false
	}
	if mappingInfo.Namespace != "" && mappingInfo.PodName != "" {
		if uid, ok := workerLookup[mappingInfo.Namespace+"/"+mappingInfo.PodName]; ok {
			return uid, true
		}
	}
	if mappingInfo.PodUID != "" {
		if _, ok := workerUIDs[mappingInfo.PodUID]; ok {
			return mappingInfo.PodUID, true
		}
	}
	return "", false
}

// buildWorkerInfoSnapshots builds a map of worker info snapshots for ERL updates.
// This is called by the QuotaController to get current worker allocations and device info.
func (w *WorkerController) buildWorkerInfoSnapshots() map[string]*computing.WorkerInfoSnapshot {
	w.mu.RLock()
	defer w.mu.RUnlock()

	result := make(map[string]*computing.WorkerInfoSnapshot, len(w.workers))
	for workerUID, workerInfo := range w.workers {
		if workerInfo == nil || !usesSoftLimiterSharedMemory(workerInfo.IsolationMode) {
			continue
		}
		allocation, exists := w.allocationController.GetWorkerAllocation(workerUID)
		if !exists || allocation == nil || allocation.WorkerInfo == nil {
			continue
		}

		snapshot := &computing.WorkerInfoSnapshot{
			Namespace:  workerInfo.Namespace,
			WorkerName: workerInfo.WorkerName,
		}

		for _, deviceInfo := range allocation.DeviceInfos {
			if deviceInfo == nil {
				continue
			}
			snapshot.Devices = append(snapshot.Devices, computing.DeviceSnapshot{
				DeviceUUID: deviceInfo.UUID,
				DeviceIdx:  int(deviceInfo.Index),
				UpLimit:    computeUpLimit(workerInfo, deviceInfo),
			})
		}

		result[workerUID] = snapshot
	}
	return result
}

// computeUpLimit calculates the compute limit percentage (0-100) for a worker on a device.
func computeUpLimit(workerInfo *api.WorkerInfo, deviceInfo *api.DeviceInfo) uint32 {
	if workerInfo == nil {
		return 100
	}
	if workerInfo.Limits.ComputePercent.Value() > 0 {
		return uint32(workerInfo.Limits.ComputePercent.Value())
	}
	if workerInfo.Limits.Tflops.Value() > 0 && deviceInfo != nil && deviceInfo.MaxTflops > 0 {
		percent := math.Ceil(workerInfo.Limits.Tflops.AsApproximateFloat64() / deviceInfo.MaxTflops * 100.0)
		if percent < 1 {
			return 1
		}
		if percent > 100 {
			return 100
		}
		return uint32(percent)
	}
	return 100
}

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
	memoryByWorkerDevice := w.collectWorkerMemoryUsage(workerLookup)
	now := uint64(w.nowFunc().Unix())

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
		w.mu.RUnlock()
		if handle == nil {
			continue
		}

		deviceMemoryUsage := memoryByWorkerDevice[workerUID]
		handle.WithState(func(state *workerstate.SharedDeviceState) {
			state.UpdateHeartbeat(now)
			// nil means collection failed; an empty, non-nil map is a valid
			// sample with no GPU processes. Keep the last usage on failure.
			if memoryByWorkerDevice == nil {
				return
			}

			// Total across every physical GPU this worker's processes touched. Used
			// as a fallback for single-device pods whose process landed on a GPU
			// other than the nominally-allocated one (e.g. shared-pool pods can run
			// on any visible card), so the per-UUID lookup below would otherwise miss.
			var totalUsage uint64
			for _, used := range deviceMemoryUsage {
				totalUsage += used
			}
			singleDevice := len(allocation.DeviceInfos) == 1
			for _, deviceInfo := range allocation.DeviceInfos {
				if deviceInfo == nil {
					continue
				}
				deviceUUID := strings.ToLower(deviceInfo.UUID)
				used := deviceMemoryUsage[deviceUUID]
				if used == 0 && singleDevice {
					used = totalUsage
				}
				state.SetPodMemoryUsed(int(deviceInfo.Index), used)
			}
		})
	}
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
	if handle == nil {
		var err error
		handle, err = workerstate.PrepareWorkerSharedMemory(
			w.shmBasePath, workerstate.NewPodIdentifier(worker.Namespace, worker.WorkerName),
			workerUID, buildWorkerDeviceConfigs(allocation),
			worker.Status == api.WorkerStatusRunning || worker.AllocationConfirmed,
		)
		if err != nil {
			return err
		}
		w.shmHandles[workerUID] = handle
	}
	if fn != nil && !handle.WithState(fn) {
		return fmt.Errorf("shared memory for worker %s is closed", workerUID)
	}
	return nil
}

func (w *WorkerController) closeSharedMemoryLocked(workerUID string) {
	if handle := w.shmHandles[workerUID]; handle != nil {
		_ = handle.Close()
		delete(w.shmHandles, workerUID)
	}
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
		memLimit := deviceInfo.TotalMemoryBytes
		if allocation.WorkerInfo.Limits.Vram.Value() > 0 {
			memLimit = uint64(allocation.WorkerInfo.Limits.Vram.Value())
		}
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

func (w *WorkerController) workerAllocations() map[string]*api.WorkerAllocation {
	w.mu.RLock()
	defer w.mu.RUnlock()

	allocations := make(map[string]*api.WorkerAllocation, len(w.workers))
	for workerUID := range w.workers {
		allocation, exists := w.allocationController.GetWorkerAllocation(workerUID)
		if !exists || allocation == nil || allocation.WorkerInfo == nil {
			continue
		}
		allocations[workerUID] = allocation
	}
	return allocations
}

func (w *WorkerController) collectWorkerMemoryUsage(workerLookup map[string]string) map[string]map[string]uint64 {
	processInfos, err := w.deviceController.GetProcessInformation()
	if err != nil {
		klog.V(4).Infof("Failed to collect process information for shared memory sync: %v", err)
		return nil
	}

	workerUIDs := w.buildWorkerUIDSet()
	result := make(map[string]map[string]uint64)
	for _, procInfo := range processInfos {
		hostPID, err := strconv.ParseUint(procInfo.ProcessID, 10, 32)
		if err != nil {
			klog.V(4).Infof("Failed to parse process ID %s during shared memory sync: %v", procInfo.ProcessID, err)
			continue
		}

		mappingInfo, err := w.backend.GetProcessMappingInfo(uint32(hostPID))
		if err != nil || mappingInfo == nil {
			continue
		}
		// Resolve the owning worker. Prefer environ-derived namespace/podName,
		// but fall back to the cgroup-derived pod UID when the GPU-holding
		// process stripped its environment (e.g. vLLM's spawned EngineCore drops
		// POD_NAME/POD_NAMESPACE). Without the fallback such a process' memory is
		// never attributed and the pod's nvidia-smi reports 0.
		workerUID, found := resolveWorkerUID(mappingInfo, workerLookup, workerUIDs)
		if !found {
			continue
		}

		deviceUUID := strings.ToLower(procInfo.DeviceUUID)
		if result[workerUID] == nil {
			result[workerUID] = make(map[string]uint64)
		}
		result[workerUID][deviceUUID] += procInfo.MemoryUsedBytes
	}

	return result
}
