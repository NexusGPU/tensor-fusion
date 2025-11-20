package worker

import (
	"sync"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/worker/computing"
	"k8s.io/klog/v2"
)

type WorkerController struct {
	mode    api.IsolationMode
	backend framework.Backend

	deviceController framework.DeviceController
	quotaController  framework.QuotaController

	mu                  sync.RWMutex
	workers             map[string]bool // worker UID -> exists
	workerWatchStop     chan struct{}
	workerWatchStopOnce sync.Once
}

func NewWorkerController(
	deviceController framework.DeviceController, mode api.IsolationMode, backend framework.Backend) framework.WorkerController {
	quotaController := computing.NewQuotaController(deviceController)
	return &WorkerController{
		deviceController: deviceController,
		mode:             mode,
		backend:          backend,
		quotaController:  quotaController,
		workers:          make(map[string]bool),
		workerWatchStop:  make(chan struct{}),
	}
}

func (w *WorkerController) Start() error {
	err := w.backend.Start()
	if err != nil {
		return err
	}
	klog.Info("Worker backend started")

	// Start watching workers from backend
	workerCh, stopCh, err := w.backend.ListAndWatchWorkers()
	if err != nil {
		return err
	}

	// Start worker watcher goroutine
	go func() {
		for {
			select {
			case <-w.workerWatchStop:
				return
			case <-stopCh:
				return
			case workers, ok := <-workerCh:
				if !ok {
					return
				}
				// Update worker cache
				w.mu.Lock()
				w.workers = make(map[string]bool)
				for _, workerUID := range workers {
					w.workers[workerUID] = true
				}
				w.mu.Unlock()
				klog.V(4).Infof("Updated worker list: %d workers", len(workers))
			}
		}
	}()

	// Start soft quota limiter
	if err := w.quotaController.StartSoftQuotaLimiter(); err != nil {
		klog.Fatalf("Failed to start soft quota limiter: %v", err)
	}
	klog.Info("Soft quota limiter started")

	return nil
}

func (w *WorkerController) Stop() error {
	w.workerWatchStopOnce.Do(func() {
		close(w.workerWatchStop)
	})
	_ = w.backend.Stop()
	_ = w.quotaController.StopSoftQuotaLimiter()
	return nil
}

func (w *WorkerController) GetWorkerAllocation(workerUID string) (*api.DeviceAllocation, error) {
	allocations, err := w.deviceController.GetDeviceAllocations("")
	if err != nil {
		return nil, err
	}
	// Find allocation for this worker
	for _, allocation := range allocations {
		if allocation.PodUID == workerUID || allocation.WorkerID == workerUID {
			return allocation, nil
		}
	}
	return nil, nil
}

func (w *WorkerController) GetWorkerMetricsUpdates() (<-chan *api.DeviceAllocation, error) {
	ch := make(chan *api.DeviceAllocation, 1)
	// TODO: Implement proper worker metrics updates channel with periodic updates
	// Channel will be closed when controller is stopped
	return ch, nil
}

func (w *WorkerController) GetWorkerMetrics() (map[string]map[string]map[string]*api.WorkerMetrics, error) {
	// Get all allocations to know which workers exist
	allocations, err := w.deviceController.GetDeviceAllocations("")
	if err != nil {
		return nil, err
	}

	// Get process compute and memory utilization from device controller
	// Try to cast to concrete type to access accelerator methods
	type acceleratorExposer interface {
		GetProcessComputeUtilization() ([]api.ComputeUtilization, error)
		GetProcessMemoryUtilization() ([]api.MemoryUtilization, error)
	}

	var computeUtils []api.ComputeUtilization
	var memUtils []api.MemoryUtilization

	if exposer, ok := w.deviceController.(acceleratorExposer); ok {
		var err error
		computeUtils, err = exposer.GetProcessComputeUtilization()
		if err != nil {
			computeUtils = []api.ComputeUtilization{}
		}
		memUtils, err = exposer.GetProcessMemoryUtilization()
		if err != nil {
			memUtils = []api.MemoryUtilization{}
		}
	} else {
		// Fallback to empty metrics if interface not available
		computeUtils = []api.ComputeUtilization{}
		memUtils = []api.MemoryUtilization{}
	}

	// Build worker to process mapping
	workerToProcesses, err := w.backend.GetWorkerToProcessMap()
	if err != nil {
		workerToProcesses = make(map[string][]string)
	}

	// Build process to metrics mapping
	processMetrics := make(map[string]map[string]*api.WorkerMetrics) // processID -> deviceUUID -> metrics

	// Aggregate compute metrics by process
	for _, computeUtil := range computeUtils {
		if processMetrics[computeUtil.ProcessID] == nil {
			processMetrics[computeUtil.ProcessID] = make(map[string]*api.WorkerMetrics)
		}
		if processMetrics[computeUtil.ProcessID][computeUtil.DeviceUUID] == nil {
			processMetrics[computeUtil.ProcessID][computeUtil.DeviceUUID] = &api.WorkerMetrics{
				DeviceUUID:        computeUtil.DeviceUUID,
				ProcessID:         computeUtil.ProcessID,
				ComputePercentage: computeUtil.UtilizationPercent,
				ComputeTflops:     computeUtil.TflopsUsed,
			}
		} else {
			processMetrics[computeUtil.ProcessID][computeUtil.DeviceUUID].ComputePercentage += computeUtil.UtilizationPercent
			processMetrics[computeUtil.ProcessID][computeUtil.DeviceUUID].ComputeTflops += computeUtil.TflopsUsed
		}
	}

	// Aggregate memory metrics by process
	for _, memUtil := range memUtils {
		if processMetrics[memUtil.ProcessID] == nil {
			processMetrics[memUtil.ProcessID] = make(map[string]*api.WorkerMetrics)
		}
		if processMetrics[memUtil.ProcessID][memUtil.DeviceUUID] == nil {
			processMetrics[memUtil.ProcessID][memUtil.DeviceUUID] = &api.WorkerMetrics{
				DeviceUUID:  memUtil.DeviceUUID,
				ProcessID:   memUtil.ProcessID,
				MemoryBytes: memUtil.UsedBytes,
			}
		} else {
			processMetrics[memUtil.ProcessID][memUtil.DeviceUUID].MemoryBytes += memUtil.UsedBytes
		}
	}

	// Build result: deviceUUID -> workerUID -> processID -> metrics
	result := make(map[string]map[string]map[string]*api.WorkerMetrics)

	// Map processes to workers
	for workerUID, processIDs := range workerToProcesses {
		for _, processID := range processIDs {
			if deviceMetrics, exists := processMetrics[processID]; exists {
				for deviceUUID, metrics := range deviceMetrics {
					if result[deviceUUID] == nil {
						result[deviceUUID] = make(map[string]map[string]*api.WorkerMetrics)
					}
					if result[deviceUUID][workerUID] == nil {
						result[deviceUUID][workerUID] = make(map[string]*api.WorkerMetrics)
					}
					result[deviceUUID][workerUID][processID] = metrics
					metrics.WorkerUID = workerUID
				}
			}
		}
	}

	// Also include allocations that might not have process mappings yet
	for _, allocation := range allocations {
		workerUID := allocation.WorkerID
		if workerUID == "" {
			workerUID = allocation.PodUID
		}
		if workerUID == "" {
			continue
		}

		if result[allocation.DeviceUUID] == nil {
			result[allocation.DeviceUUID] = make(map[string]map[string]*api.WorkerMetrics)
		}
		if result[allocation.DeviceUUID][workerUID] == nil {
			result[allocation.DeviceUUID][workerUID] = make(map[string]*api.WorkerMetrics)
		}
	}

	return result, nil
}

func (w *WorkerController) ListWorkers() ([]string, error) {
	// First check cache (updated by ListAndWatchWorkers)
	w.mu.RLock()
	cachedWorkers := make([]string, 0, len(w.workers))
	for workerUID := range w.workers {
		cachedWorkers = append(cachedWorkers, workerUID)
	}
	w.mu.RUnlock()

	// If cache has workers, return them
	if len(cachedWorkers) > 0 {
		return cachedWorkers, nil
	}

	// If cache is empty, directly query device allocations to get immediate results
	// This ensures we hit the key logic path and return accurate results
	allocations, err := w.deviceController.GetDeviceAllocations("")
	if err != nil {
		return cachedWorkers, err
	}

	// Extract unique worker UIDs from allocations
	workerSet := make(map[string]bool)
	for _, allocation := range allocations {
		workerUID := allocation.WorkerID
		if workerUID == "" {
			workerUID = allocation.PodUID
		}
		if workerUID != "" {
			workerSet[workerUID] = true
		}
	}

	// Update cache with discovered workers
	w.mu.Lock()
	for workerUID := range workerSet {
		w.workers[workerUID] = true
	}
	w.mu.Unlock()

	// Return list of workers
	workers := make([]string, 0, len(workerSet))
	for workerUID := range workerSet {
		workers = append(workers, workerUID)
	}
	return workers, nil
}
