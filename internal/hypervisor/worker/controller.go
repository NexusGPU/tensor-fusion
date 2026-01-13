package worker

import (
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

func (w *WorkerController) GetWorkerMetrics() (map[string]map[string]map[string]*api.WorkerMetrics, error) {
	// TODO: implement this
	// Get all allocations to know which workers exist
	// find process and then get metrics by host processes
	// w.deviceController.GetProcessMetrics()
	return nil, nil
}
