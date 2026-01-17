package single_node

import (
	"context"
	"os"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/google/uuid"
	"github.com/samber/lo"
	"k8s.io/klog/v2"
)

type SingleNodeBackend struct {
	ctx                  context.Context
	deviceController     framework.DeviceController
	allocationController framework.WorkerAllocationController
	fileState            *FileStateManager
	mu                   sync.RWMutex
	workers              map[string]*api.WorkerInfo
	stopCh               chan struct{}
	stopOnce             sync.Once

	// Worker watching
	subscribersMu sync.RWMutex
	subscribers   map[string]chan *api.WorkerInfo
	workerHandler *framework.WorkerChangeHandler
}

func NewSingleNodeBackend(ctx context.Context, deviceController framework.DeviceController, allocationController framework.WorkerAllocationController) *SingleNodeBackend {
	stateDir := os.Getenv("TENSOR_FUSION_STATE_DIR")
	if stateDir == "" {
		stateDir = "/tmp/tensor-fusion-state"
	}
	return &SingleNodeBackend{
		ctx:                  ctx,
		deviceController:     deviceController,
		allocationController: allocationController,
		fileState:            NewFileStateManager(stateDir),
		workers:              make(map[string]*api.WorkerInfo),
		stopCh:               make(chan struct{}),
		subscribers:          make(map[string]chan *api.WorkerInfo),
	}
}

func (b *SingleNodeBackend) Start() error {
	// Load initial state from files
	if err := b.loadState(); err != nil {
		klog.Warningf("Failed to load initial state: %v", err)
	}

	// Start periodic worker discovery
	go b.periodicWorkerDiscovery()
	return nil
}

func (b *SingleNodeBackend) Stop() error {
	// Use sync.Once to ensure stopCh is only closed once
	b.stopOnce.Do(func() {
		close(b.stopCh)
	})

	// Close all subscriber channels
	b.subscribersMu.Lock()
	for id, ch := range b.subscribers {
		close(ch)
		delete(b.subscribers, id)
	}
	b.subscribersMu.Unlock()

	return nil
}

// loadState loads workers and devices from file state
func (b *SingleNodeBackend) loadState() error {
	workers, err := b.fileState.LoadWorkers()
	if err != nil {
		return err
	}

	b.mu.Lock()
	b.workers = workers
	b.mu.Unlock()

	return nil
}

// discoverWorkers discovers workers from file state and notifies subscribers of changes
func (b *SingleNodeBackend) discoverWorkers() {
	workers, err := b.fileState.LoadWorkers()
	if err != nil {
		klog.Errorf("Failed to load workers from file state: %v", err)
		return
	}

	b.mu.Lock()
	// Find new and updated workers
	for uid, worker := range workers {
		oldWorker, exists := b.workers[uid]
		if !exists {
			// New worker
			b.workers[uid] = worker
			b.mu.Unlock()
			b.notifySubscribers(worker)
			b.mu.Lock()
		} else if !workersEqual(oldWorker, worker) {
			// Updated worker
			b.workers[uid] = worker
			b.mu.Unlock()
			b.notifySubscribers(worker)
			b.mu.Lock()
		}
	}

	// Find removed workers
	for uid := range b.workers {
		if _, exists := workers[uid]; !exists {
			delete(b.workers, uid)
		}
	}
	b.mu.Unlock()
}

// notifySubscribers notifies all subscribers of a worker change
func (b *SingleNodeBackend) notifySubscribers(worker *api.WorkerInfo) {
	b.subscribersMu.RLock()
	defer b.subscribersMu.RUnlock()

	for _, ch := range b.subscribers {
		select {
		case ch <- worker:
		default:
			klog.Warningf("Channel is full, skipping notification for worker change %s", worker.WorkerUID)
		}
	}
}

// workersEqual checks if two workers are equal (simple comparison)
func workersEqual(w1, w2 *api.WorkerInfo) bool {
	if w1 == nil && w2 == nil {
		return true
	}
	if w1 == nil || w2 == nil {
		return false
	}
	return w1.WorkerUID == w2.WorkerUID &&
		w1.Status == w2.Status &&
		len(w1.AllocatedDevices) == len(w2.AllocatedDevices)
}

func (b *SingleNodeBackend) periodicWorkerDiscovery() {
	// Run initial discovery immediately
	b.discoverWorkers()

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-b.stopCh:
			return
		case <-b.ctx.Done():
			return
		case <-ticker.C:
			b.discoverWorkers()
		}
	}
}

func (b *SingleNodeBackend) RegisterWorkerUpdateHandler(handler framework.WorkerChangeHandler) error {
	b.workerHandler = &handler

	// Create channel for this subscriber
	workerCh := make(chan *api.WorkerInfo, 16)
	subscriberID := uuid.NewString()

	// Register subscriber
	b.subscribersMu.Lock()
	b.subscribers[subscriberID] = workerCh
	b.subscribersMu.Unlock()

	// Start bridge goroutine to convert channel messages to handler calls
	go func() {
		defer func() {
			b.subscribersMu.Lock()
			delete(b.subscribers, subscriberID)
			b.subscribersMu.Unlock()
		}()

		for {
			select {
			case <-b.ctx.Done():
				return
			case <-b.stopCh:
				return
			case worker, ok := <-workerCh:
				if !ok {
					return
				}
				if worker == nil {
					continue
				}

				// Determine if this is add, update, or remove
				b.mu.Lock()
				oldWorker, exists := b.workers[worker.WorkerUID]

				if worker.DeletedAt > 0 {
					// Worker was deleted
					if exists && handler.OnRemove != nil {
						handler.OnRemove(worker)
					}
					delete(b.workers, worker.WorkerUID)
				} else if !exists {
					// New worker
					b.workers[worker.WorkerUID] = worker
					if handler.OnAdd != nil {
						handler.OnAdd(worker)
					}
				} else {
					// Updated worker
					b.workers[worker.WorkerUID] = worker
					if handler.OnUpdate != nil {
						handler.OnUpdate(oldWorker, worker)
					}
				}
				b.mu.Unlock()
			}
		}
	}()
	return nil
}

func (b *SingleNodeBackend) StartWorker(worker *api.WorkerInfo) error {
	if err := b.fileState.AddWorker(worker); err != nil {
		return err
	}

	b.mu.Lock()
	b.workers[worker.WorkerUID] = worker
	b.mu.Unlock()

	b.notifySubscribers(worker)
	klog.Infof("Worker started: %s", worker.WorkerUID)
	return nil
}

func (b *SingleNodeBackend) StopWorker(workerUID string) error {
	if err := b.fileState.RemoveWorker(workerUID); err != nil {
		return err
	}

	b.mu.Lock()
	delete(b.workers, workerUID)
	b.mu.Unlock()

	klog.Infof("Worker stopped: %s", workerUID)
	return nil
}

func (b *SingleNodeBackend) GetProcessMappingInfo(hostPID uint32) (*framework.ProcessMappingInfo, error) {
	// For single node mode, we don't have Kubernetes pod info
	// Return minimal info with hostPID
	return &framework.ProcessMappingInfo{
		HostPID:  hostPID,
		GuestPID: hostPID,
	}, nil
}

func (b *SingleNodeBackend) GetDeviceChangeHandler() framework.DeviceChangeHandler {
	return framework.DeviceChangeHandler{
		OnAdd: func(device *api.DeviceInfo) {
			if err := b.fileState.AddDevice(device); err != nil {
				klog.Errorf("Failed to save device to file state: %v", err)
			} else {
				klog.Infof("Device added: %s", device.UUID)
			}
		},
		OnRemove: func(device *api.DeviceInfo) {
			if err := b.fileState.RemoveDevice(device.UUID); err != nil {
				klog.Errorf("Failed to remove device from file state: %v", err)
			} else {
				klog.Infof("Device removed: %s", device.UUID)
			}
		},
		OnUpdate: func(oldDevice, newDevice *api.DeviceInfo) {
			if err := b.fileState.UpdateDevice(newDevice); err != nil {
				klog.Errorf("Failed to update device in file state: %v", err)
			} else {
				klog.Infof("Device updated: %s", newDevice.UUID)
			}
		},
	}
}

func (b *SingleNodeBackend) ListWorkers() []*api.WorkerInfo {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return lo.Values(b.workers)
}
