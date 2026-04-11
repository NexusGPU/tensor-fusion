package worker

import (
	"context"
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
	backend framework.Backend

	deviceController     framework.DeviceController
	allocationController framework.WorkerAllocationController
	quotaController      framework.QuotaController

	mu         sync.RWMutex
	workers    map[string]*api.WorkerInfo
	shmHandles map[string]*workerstate.SharedMemoryHandle // workerUID -> shm handle

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
	quotaController := computing.NewQuotaController(deviceController, backend)

	wc := &WorkerController{
		deviceController:     deviceController,
		allocationController: allocationController,
		mode:                 mode,
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

func (w *WorkerController) Start() error {
	// Register worker update handler
	handler := framework.WorkerChangeHandler{
		OnAdd: func(worker *api.WorkerInfo) {
			w.mu.Lock()
			w.workers[worker.WorkerUID] = worker
			w.mu.Unlock()

			// For soft isolation, proactively create shared memory when a worker pod appears.
			// Unlike hard/sidecar mode where the worker process calls /process-init to trigger
			// shm creation, soft mode injects the limiter directly into the business container
			// which only reads shm passively.
			if w.mode == tfv1.IsolationModeSoft {
				w.ensureSoftWorkerSharedMemory(worker)
			}
		},
		OnRemove: func(worker *api.WorkerInfo) {
			// Deallocate worker devices first
			if err := w.allocationController.DeallocateWorker(worker.WorkerUID); err != nil {
				klog.Errorf("Failed to deallocate worker %s: %v", worker.WorkerUID, err)
			}
			w.mu.Lock()
			defer w.mu.Unlock()
			delete(w.workers, worker.WorkerUID)
		},
		OnUpdate: func(oldWorker, newWorker *api.WorkerInfo) {
			// Check if worker transitioned to Terminated state (Succeeded or Failed)
			// If so, deallocate devices including partitions
			if oldWorker.Status != api.WorkerStatusTerminated && newWorker.Status == api.WorkerStatusTerminated {
				if err := w.allocationController.DeallocateWorker(newWorker.WorkerUID); err != nil {
					klog.Errorf("Failed to deallocate worker %s on termination: %v", newWorker.WorkerUID, err)
				}
			}
			w.mu.Lock()
			defer w.mu.Unlock()
			w.workers[newWorker.WorkerUID] = newWorker
		},
	}

	err := w.backend.RegisterWorkerUpdateHandler(handler)
	if err != nil {
		return err
	}

	// Start soft quota limiter
	if w.mode == tfv1.IsolationModeSoft {
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

func (w *WorkerController) Stop() error {
	w.stopSharedMemorySyncLoop()
	_ = w.backend.Stop()
	_ = w.quotaController.StopSoftQuotaLimiter()
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

		// Skip if namespace or podName is empty (not a TensorFusion worker)
		if mappingInfo.GuestID == "" {
			continue
		}

		// Look up WorkerUID using namespace/podName
		workerKey := mappingInfo.Namespace + "/" + mappingInfo.PodName
		workerUID, found := workerLookup[workerKey]
		if !found {
			// Process belongs to a pod not tracked by this hypervisor
			klog.V(5).Infof("Worker not found for key %s (process %d)", workerKey, hostPID)
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

// buildWorkerInfoSnapshots builds a map of worker info snapshots for ERL updates.
// This is called by the QuotaController to get current worker allocations and device info.
func (w *WorkerController) buildWorkerInfoSnapshots() map[string]*computing.WorkerInfoSnapshot {
	w.mu.RLock()
	defer w.mu.RUnlock()

	result := make(map[string]*computing.WorkerInfoSnapshot, len(w.workers))
	for workerUID, workerInfo := range w.workers {
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
	activeWorkers := make(map[string]bool)
	w.mu.RLock()
	for _, worker := range w.workers {
		activeWorkers[worker.Namespace+"/"+worker.WorkerName] = true
	}
	w.mu.RUnlock()

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

			shmPath := filepath.Join(nsPath, podEntry.Name(), workerstate.ShmPathSuffix)
			if _, statErr := os.Stat(shmPath); statErr != nil {
				continue
			}

			// Clean up shm file
			_ = os.Remove(shmPath)
			_ = os.Remove(filepath.Join(nsPath, podEntry.Name()))
			cleanedCount++
		}
	}

	// Clean up stale shm handles
	w.mu.Lock()
	for uid, handle := range w.shmHandles {
		if _, exists := w.workers[uid]; !exists {
			_ = handle.Close()
			delete(w.shmHandles, uid)
		}
	}
	w.mu.Unlock()

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

		// Get shm handle for this worker
		w.mu.RLock()
		handle := w.shmHandles[workerUID]
		w.mu.RUnlock()
		if handle == nil {
			continue
		}

		state := handle.GetState()
		if state == nil {
			continue
		}

		state.UpdateHeartbeat(now)

		deviceMemoryUsage := memoryByWorkerDevice[workerUID]
		for _, deviceInfo := range allocation.DeviceInfos {
			if deviceInfo == nil {
				continue
			}
			deviceUUID := strings.ToLower(deviceInfo.UUID)
			state.SetPodMemoryUsed(int(deviceInfo.Index), deviceMemoryUsage[deviceUUID])
		}
	}
}

// ensureSoftWorkerSharedMemory creates shared memory for a soft isolation pod.
// Called when a new worker pod is detected. Uses pure Go shm (no C limiter dependency).
func (w *WorkerController) ensureSoftWorkerSharedMemory(worker *api.WorkerInfo) {
	go func() {
		// Retry for up to 30 seconds waiting for allocation
		for i := 0; i < 30; i++ {
			allocation, exists := w.allocationController.GetWorkerAllocation(worker.WorkerUID)
			if exists && allocation != nil && allocation.WorkerInfo != nil && len(allocation.DeviceInfos) > 0 {
				configs := buildSoftDeviceConfigs(allocation)
				if len(configs) == 0 {
					return
				}
				podId := workerstate.NewPodIdentifier(worker.Namespace, worker.WorkerName)
				handle, err := workerstate.CreateSharedMemoryHandle(w.shmBasePath, podId, configs)
				if err != nil {
					klog.Errorf("Failed to create shared memory for soft worker %s/%s: %v", worker.Namespace, worker.WorkerName, err)
					return
				}
				// Store handle for later use (heartbeat, memory sync)
				w.mu.Lock()
				w.shmHandles[worker.WorkerUID] = handle
				w.mu.Unlock()
				klog.Infof("Created shared memory for soft worker %s/%s with %d devices",
					worker.Namespace, worker.WorkerName, len(configs))
				return
			}
			time.Sleep(1 * time.Second)
		}
		klog.Warningf("Timed out waiting for allocation for soft worker %s/%s", worker.Namespace, worker.WorkerName)
	}()
}

func buildSoftDeviceConfigs(allocation *api.WorkerAllocation) []workerstate.DeviceConfig {
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
		configs = append(configs, workerstate.DeviceConfig{
			DeviceIdx:  uint32(deviceInfo.Index),
			DeviceUUID: normalizeDeviceUUID(deviceInfo.UUID),
			UpLimit:    computeUpLimit(allocation.WorkerInfo, deviceInfo),
			MemLimit:   memLimit,
			SMCount:    smCount * 128,
		})
	}
	return configs
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

	result := make(map[string]map[string]uint64)
	for _, procInfo := range processInfos {
		hostPID, err := strconv.ParseUint(procInfo.ProcessID, 10, 32)
		if err != nil {
			klog.V(4).Infof("Failed to parse process ID %s during shared memory sync: %v", procInfo.ProcessID, err)
			continue
		}

		mappingInfo, err := w.backend.GetProcessMappingInfo(uint32(hostPID))
		if err != nil || mappingInfo == nil || mappingInfo.GuestID == "" {
			continue
		}

		workerKey := mappingInfo.Namespace + "/" + mappingInfo.PodName
		workerUID, found := workerLookup[workerKey]
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
