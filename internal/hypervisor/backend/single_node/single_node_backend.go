package single_node

import (
	"context"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"k8s.io/klog/v2"
)

type SingleNodeBackend struct {
	ctx               context.Context
	deviceController  framework.DeviceController
	mu                sync.RWMutex
	workers           map[string]*WorkerState // worker UID -> state
	stopCh            chan struct{}
	stopOnce          sync.Once
	workerCh          chan []string
	workerChCloseOnce sync.Once
	workerStopCh      chan struct{}
	workerStopOnce    sync.Once
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
	// Use sync.Once to ensure stopCh is only closed once
	b.stopOnce.Do(func() {
		close(b.stopCh)
	})
	// Close worker watch stop channel (safe to close even if nil)
	if b.workerStopCh != nil {
		b.workerStopOnce.Do(func() {
			close(b.workerStopCh)
		})
	}
	return nil
}

// discoverWorkers discovers workers from device allocations and updates the internal state
func (b *SingleNodeBackend) discoverWorkers() {
	// Discover workers from device allocations
	allocations, err := b.deviceController.GetDeviceAllocations("")
	if err != nil {
		klog.Errorf("Failed to get device allocations: %v", err)
		return
	}

	b.mu.Lock()
	defer b.mu.Unlock()

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

func (b *SingleNodeBackend) ListAndWatchWorkers() (<-chan []string, <-chan struct{}, error) {
	// Initialize channels if not already created
	if b.workerCh == nil {
		b.workerCh = make(chan []string, 1)
		b.workerStopCh = make(chan struct{})
	}

	// Send initial worker list and watch for changes
	go func() {
		defer b.workerChCloseOnce.Do(func() {
			close(b.workerCh)
		})

		// Trigger immediate discovery before sending initial list
		b.discoverWorkers()

		// Send initial list
		b.mu.RLock()
		workers := make([]string, 0, len(b.workers))
		for workerUID := range b.workers {
			workers = append(workers, workerUID)
		}
		b.mu.RUnlock()

		select {
		case b.workerCh <- workers:
		case <-b.ctx.Done():
			return
		case <-b.workerStopCh:
			return
		}

		// Watch for changes via periodic discovery (already running in background)
		// The periodic discovery will update b.workers, but we don't have a direct
		// notification mechanism, so we'll poll periodically
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-b.ctx.Done():
				return
			case <-b.workerStopCh:
				return
			case <-ticker.C:
				// Trigger discovery before sending update
				b.discoverWorkers()

				b.mu.RLock()
				workers := make([]string, 0, len(b.workers))
				for workerUID := range b.workers {
					workers = append(workers, workerUID)
				}
				b.mu.RUnlock()

				select {
				case b.workerCh <- workers:
				case <-b.ctx.Done():
					return
				case <-b.workerStopCh:
					return
				}
			}
		}
	}()

	return b.workerCh, b.workerStopCh, nil
}

func (b *SingleNodeBackend) GetWorkerToProcessMap() (map[string][]string, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	result := make(map[string][]string)
	for workerUID, state := range b.workers {
		result[workerUID] = append([]string{}, state.ProcessIDs...)
	}
	return result, nil
}

func (b *SingleNodeBackend) StartWorker(workerUID string) error {
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

func (b *SingleNodeBackend) StopWorker(workerUID string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	delete(b.workers, workerUID)
	return nil
}

func (b *SingleNodeBackend) ReconcileDevices(devices []string) error {
	// In single node mode, we don't need to reconcile with external systems
	// Devices are managed locally
	return nil
}
