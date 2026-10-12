package worker

import (
	"context"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
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
	// foundUpLimits holds, per worker and device slot, the compute limit the
	// worker's mapping carried when it was opened, before the current limits
	// were applied to it; see computing.DeviceSnapshot.FoundUpLimit.
	foundUpLimits map[string]map[int]uint32
	stopped       bool

	shmBasePath string
	nowFunc     func() time.Time

	// Per-process compute utilization from the latest successful process
	// sample, read by the quota controller. Nil until the first sample.
	processUtilization atomic.Pointer[computing.ProcessUtilizationSnapshot]
	// The provider's latest process list, shared by the sync loop that samples
	// it and by on-demand readers such as worker metrics.
	processSample atomic.Pointer[processSample]
	// processSampleError holds the error of the last failed process sample, nil
	// after a successful one, so that a failure is logged when it starts.
	processSampleError atomic.Pointer[string]
	// unreadableDevices holds the devices whose process list could not be read
	// in the last sample, to log when that changes.
	unreadableMu      sync.Mutex
	unreadableDevices map[string]bool
	// unreadableSince says since when each of them could not be read, and
	// unreadableExpired which ones passed unreadableMemoryMaxAge.
	unreadableSince   map[string]time.Time
	unreadableExpired map[string]bool
	// limits writes worker limits to the mappings, one worker at a time.
	limits limitWriter
	// burstWindowReported holds, by worker UID, the burst window annotation
	// that was last reported as not usable as written; see computeBurstWindow.
	burstWindowReported sync.Map
	// Compute utilization per worker and device, integrated over time; see usage.go.
	usage workerUsageTracker
	// The mappings the heartbeat loop stamps; see heartbeat.go.
	heartbeatHandles atomic.Pointer[[]*workerstate.SharedMemoryHandle]

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
		qc.SetProcessUtilizationProvider(wc.processUtilization.Load)
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
			w.burstWindowReported.Delete(worker.WorkerUID)
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
			w.refreshWorkerLimits(newWorker)
		},
	}
}

func (w *WorkerController) Start() error {
	err := w.backend.RegisterWorkerUpdateHandler(w.workerChangeHandler())
	if err != nil {
		return err
	}
	// A device's TFLOPS capacity can be re-resolved while workers run on it.
	w.deviceController.RegisterDeviceUpdateHandler(framework.DeviceChangeHandler{OnUpdate: w.onDeviceUpdate})

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
	w.startHeartbeatLoop(ctx)
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

		// Read from the tracked worker in every control interval, like the
		// limit: an annotation changed on a running Pod applies within one.
		burstWindow := w.computeBurstWindow(workerInfo)
		for _, deviceInfo := range w.allocatedDevices(allocation) {
			snapshot.Devices = append(snapshot.Devices, computing.DeviceSnapshot{
				DeviceUUID:  deviceInfo.UUID,
				DeviceIdx:   int(deviceInfo.Index),
				UpLimit:     computeUpLimit(workerInfo, deviceInfo),
				BurstWindow: burstWindow,
				// A nil map reads as zero: not known.
				FoundUpLimit: w.foundUpLimits[workerUID][int(deviceInfo.Index)],
			})
		}

		result[workerUID] = snapshot
	}
	return result
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

func (w *WorkerController) now() time.Time {
	if w.nowFunc != nil {
		return w.nowFunc()
	}
	return time.Now()
}
