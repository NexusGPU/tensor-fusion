package single_node

import (
	"context"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"k8s.io/klog/v2"
)

type SingleNodeBackend struct {
	ctx              context.Context
	deviceController framework.DeviceController
	mu               sync.RWMutex
	workers          map[string]*WorkerState // worker UID -> state
	stopCh           chan struct{}
}

type WorkerState struct {
	UID         string
	ProcessIDs  []string
	CreatedAt   time.Time
	LastUpdated time.Time
}

func NewSingleNodeBackend(ctx context.Context, deviceController framework.DeviceController) *SingleNodeBackend {
	return &SingleNodeBackend{
		ctx:              ctx,
		deviceController: deviceController,
		workers:          make(map[string]*WorkerState),
		stopCh:           make(chan struct{}),
	}
}

func (b *SingleNodeBackend) Start() error {
	// Start periodic worker discovery
	go b.periodicWorkerDiscovery()
	return nil
}

func (b *SingleNodeBackend) Stop() error {
	close(b.stopCh)
	return nil
}

func (b *SingleNodeBackend) periodicWorkerDiscovery() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-b.stopCh:
			return
		case <-b.ctx.Done():
			return
		case <-ticker.C:
			// Discover workers from device allocations
			allocations, err := b.deviceController.GetDeviceAllocations(b.ctx, "")
			if err != nil {
				klog.Errorf("Failed to get device allocations: %v", err)
				continue
			}

			b.mu.Lock()
			// Update worker states from allocations
			for _, allocation := range allocations {
				workerUID := allocation.WorkerID
				if workerUID == "" {
					workerUID = allocation.PodUID
				}
				if workerUID == "" {
					continue
				}

				if _, exists := b.workers[workerUID]; !exists {
					b.workers[workerUID] = &WorkerState{
						UID:         workerUID,
						ProcessIDs:  []string{},
						CreatedAt:   time.Now(),
						LastUpdated: time.Now(),
					}
				} else {
					b.workers[workerUID].LastUpdated = time.Now()
				}
			}

			// Remove workers that no longer have allocations
			activeWorkers := make(map[string]bool)
			for _, allocation := range allocations {
				workerUID := allocation.WorkerID
				if workerUID == "" {
					workerUID = allocation.PodUID
				}
				if workerUID != "" {
					activeWorkers[workerUID] = true
				}
			}

			for workerUID := range b.workers {
				if !activeWorkers[workerUID] {
					delete(b.workers, workerUID)
				}
			}
			b.mu.Unlock()
		}
	}
}

func (b *SingleNodeBackend) ListAndWatchWorkers(ctx context.Context, stopCh <-chan struct{}) ([]string, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	workers := make([]string, 0, len(b.workers))
	for workerUID := range b.workers {
		workers = append(workers, workerUID)
	}
	return workers, nil
}

func (b *SingleNodeBackend) GetWorkerToProcessMap(ctx context.Context) (map[string][]string, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	result := make(map[string][]string)
	for workerUID, state := range b.workers {
		result[workerUID] = append([]string{}, state.ProcessIDs...)
	}
	return result, nil
}

func (b *SingleNodeBackend) StartWorker(ctx context.Context, workerUID string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if _, exists := b.workers[workerUID]; !exists {
		b.workers[workerUID] = &WorkerState{
			UID:         workerUID,
			ProcessIDs:  []string{},
			CreatedAt:   time.Now(),
			LastUpdated: time.Now(),
		}
	}
	return nil
}

func (b *SingleNodeBackend) StopWorker(ctx context.Context, workerUID string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	delete(b.workers, workerUID)
	return nil
}

func (b *SingleNodeBackend) ReconcileDevices(ctx context.Context, devices []string) error {
	// In single node mode, we don't need to reconcile with external systems
	// Devices are managed locally
	return nil
}
