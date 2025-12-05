package worker

import (
	"maps"
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

	deviceController framework.DeviceController
	quotaController  framework.QuotaController

	mu                sync.RWMutex
	workers           map[string]*api.WorkerInfo
	workerAllocations map[string]*api.WorkerAllocation
}

func NewWorkerController(
	deviceController framework.DeviceController, mode api.IsolationMode, backend framework.Backend) framework.WorkerController {
	quotaController := computing.NewQuotaController(deviceController)
	return &WorkerController{
		deviceController: deviceController,
		mode:             mode,
		backend:          backend,
		quotaController:  quotaController,

		workers:           make(map[string]*api.WorkerInfo, 32),
		workerAllocations: make(map[string]*api.WorkerAllocation, 32),
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

// AllocateWorker implements framework.WorkerController
func (w *WorkerController) AllocateWorkerDevices(request *api.WorkerInfo) (*api.WorkerAllocation, error) {
	// Validate devices exist
	w.mu.Lock()
	defer w.mu.Unlock()

	deviceInfos := make([]*api.DeviceInfo, 0, len(request.AllocatedDevices))

	// partitioned mode, call split device
	isPartitioned := request.IsolationMode == tfv1.IsolationModePartitioned && request.PartitionTemplateID != ""

	for _, deviceUUID := range request.AllocatedDevices {
		if device, exists := w.deviceController.GetDevice(deviceUUID); exists {
			if isPartitioned {
				deviceInfo, err := w.deviceController.SplitDevice(deviceUUID, request.PartitionTemplateID)
				if err != nil {
					return nil, err
				}
				deviceInfos = append(deviceInfos, deviceInfo)
			} else {
				deviceInfos = append(deviceInfos, device)
			}
		}
	}

	mounts, err := w.deviceController.GetVendorMountLibs()
	if err != nil {
		klog.Errorf("failed to get vendor mount libs for worker allocation of %s: %v,", request.WorkerUID, err)
		return nil, err
	}

	envs := make(map[string]string, 8)
	devices := make(map[string]*api.DeviceSpec, 8)
	for _, deviceInfo := range deviceInfos {
		maps.Copy(envs, deviceInfo.DeviceEnv)
		for devNode, guestPath := range deviceInfo.DeviceNode {
			if _, exists := devices[devNode]; exists {
				continue
			}
			devices[devNode] = &api.DeviceSpec{
				HostPath:    devNode,
				GuestPath:   guestPath,
				Permissions: "rwm",
			}
		}
	}

	allocation := &api.WorkerAllocation{
		WorkerInfo:  request,
		DeviceInfos: deviceInfos,
		Envs:        envs,
		Mounts:      mounts,
		Devices:     lo.Values(devices),
	}

	w.workerAllocations[request.WorkerUID] = allocation
	for _, deviceUUID := range request.AllocatedDevices {
		w.deviceController.AddDeviceAllocation(deviceUUID, allocation)
	}
	return allocation, nil
}

func (w *WorkerController) DeallocateWorker(workerUID string) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	allocation, exists := w.workerAllocations[workerUID]
	if !exists {
		klog.Errorf("worker allocation not found for worker, can not deallocate worker %s", workerUID)
		return nil
	}
	delete(w.workerAllocations, workerUID)
	for _, deviceUUID := range allocation.WorkerInfo.AllocatedDevices {
		w.deviceController.RemoveDeviceAllocation(deviceUUID, allocation)
	}
	return nil
}

func (w *WorkerController) ListWorkers() ([]*api.WorkerInfo, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return lo.Values(w.workers), nil
}

func (w *WorkerController) GetWorkerAllocation(workerUID string) (*api.WorkerAllocation, bool) {
	w.mu.RLock()
	defer w.mu.RUnlock()
	allocation, exists := w.workerAllocations[workerUID]
	return allocation, exists
}

func (w *WorkerController) GetWorkerMetrics() (map[string]map[string]map[string]*api.WorkerMetrics, error) {
	// TODO: implement this
	// Get all allocations to know which workers exist
	// find process and then get metrics by host processes
	// w.deviceController.GetProcessMetrics()
	return nil, nil
}
