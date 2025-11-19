package worker

import (
	"context"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/worker/computing"
	"k8s.io/klog/v2"
)

type WorkerController struct {
	workerToProcesses  map[string]string // worker UID -> process ID
	processToNsProcess map[string]string // process ID -> linux Namespaced process ID in container
	mode               api.IsolationMode
	backend            framework.Backend

	deviceController framework.DeviceController
	quotaController  framework.QuotaController
	// TODO: Add worker store to track workers and their allocations
}

func NewWorkerController(
	deviceController framework.DeviceController, mode api.IsolationMode, backend framework.Backend) framework.WorkerController {
	quotaController := computing.NewQuotaController(deviceController)
	return &WorkerController{
		deviceController: deviceController, mode: mode, backend: backend,
		quotaController: quotaController,
	}
}

func (w *WorkerController) Start() error {
	err := w.backend.Start()
	if err != nil {
		return err
	}
	klog.Info("Worker backend started")

	// Start soft quota limiter
	if err := w.quotaController.StartSoftQuotaLimiter(); err != nil {
		klog.Fatalf("Failed to start soft quota limiter: %v", err)
	}
	klog.Info("Soft quota limiter started")

	return nil
}

func (w *WorkerController) Stop() error {
	w.backend.Stop()
	w.quotaController.StopSoftQuotaLimiter()
	return nil
}

func (w *WorkerController) GetWorkerAllocation(ctx context.Context, workerUID string) (*api.DeviceAllocation, error) {
	allocations, err := w.deviceController.GetDeviceAllocations(ctx, "")
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

func (w *WorkerController) GetWorkerMetricsUpdates(ctx context.Context) (<-chan *api.DeviceAllocation, error) {
	// TODO: Implement proper worker metrics updates channel
	ch := make(chan *api.DeviceAllocation)
	return ch, nil
}

func (w *WorkerController) GetWorkerMetrics(ctx context.Context) (map[string]map[string]map[string]*api.WorkerMetrics, error) {
	// TODO: Implement worker metrics collection from device controller
	// This should collect metrics from all devices for all workers
	result := make(map[string]map[string]map[string]*api.WorkerMetrics)
	return result, nil
}

func (w *WorkerController) ListWorkers(ctx context.Context) ([]string, error) {
	// TODO: Implement worker listing from device controller
	// Get all allocations and extract unique worker UIDs
	allocations, err := w.deviceController.GetDeviceAllocations(ctx, "")
	if err != nil {
		return nil, err
	}
	workerSet := make(map[string]bool)
	for _, allocation := range allocations {
		if allocation.PodUID != "" {
			workerSet[allocation.PodUID] = true
		}
		if allocation.WorkerID != "" {
			workerSet[allocation.WorkerID] = true
		}
	}
	workers := make([]string, 0, len(workerSet))
	for workerUID := range workerSet {
		workers = append(workers, workerUID)
	}
	return workers, nil
}
