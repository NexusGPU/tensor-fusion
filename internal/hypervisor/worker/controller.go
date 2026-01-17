package worker

import (
	"strconv"
	"strings"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/worker/computing"
	"github.com/samber/lo"
	"k8s.io/klog/v2"
)

type WorkerController struct {
	mode    api.IsolationMode
	backend framework.Backend

	deviceController     framework.DeviceController
	allocationController framework.WorkerAllocationController
	quotaController      framework.QuotaController

	mu      sync.RWMutex
	workers map[string]*api.WorkerInfo
}

func NewWorkerController(
	deviceController framework.DeviceController,
	allocationController framework.WorkerAllocationController,
	mode api.IsolationMode,
	backend framework.Backend,
) framework.WorkerController {
	quotaController := computing.NewQuotaController(deviceController)
	return &WorkerController{
		deviceController:     deviceController,
		allocationController: allocationController,
		mode:                 mode,
		backend:              backend,
		quotaController:      quotaController,

		workers: make(map[string]*api.WorkerInfo, 32),
	}
}

func (w *WorkerController) Start() error {
	// Register worker update handler
	handler := framework.WorkerChangeHandler{
		OnAdd: func(worker *api.WorkerInfo) {
			w.mu.Lock()
			defer w.mu.Unlock()
			w.workers[worker.WorkerUID] = worker
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
	return nil
}

func (w *WorkerController) Stop() error {
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
