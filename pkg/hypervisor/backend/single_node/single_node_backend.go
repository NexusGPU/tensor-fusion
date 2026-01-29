package single_node

import (
	"context"
	"math"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/google/uuid"
	"github.com/samber/lo"
	"k8s.io/klog/v2"
)

type processState struct {
	cmd        *exec.Cmd
	retryCount int64
	lastRetry  time.Time
	mu         sync.Mutex
}

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

	// Process management
	processesMu sync.RWMutex
	processes   map[string]*processState
}

func NewSingleNodeBackend(
	ctx context.Context,
	deviceController framework.DeviceController,
	allocationController framework.WorkerAllocationController,
) *SingleNodeBackend {
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
		processes:            make(map[string]*processState),
	}
}

func (b *SingleNodeBackend) Start() error {
	// Load initial state from files
	if err := b.loadState(); err != nil {
		klog.Warningf("Failed to load initial state: %v", err)
	}

	// Start periodic worker discovery
	go b.periodicWorkerDiscovery()

	// Start process reconcile loop
	go b.processReconcileLoop()

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

	// Stop all processes
	b.processesMu.Lock()
	for workerUID, ps := range b.processes {
		if ps.cmd != nil && ps.cmd.Process != nil {
			_ = ps.cmd.Process.Kill()
			klog.Infof("Killed process for worker: %s", workerUID)
		}
	}
	b.processes = make(map[string]*processState)
	b.processesMu.Unlock()

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
	// If worker has process runtime info, start the process
	if worker.WorkerRunningInfo != nil && worker.WorkerRunningInfo.Type == api.WorkerRuntimeTypeProcess {
		if err := b.startProcess(worker); err != nil {
			return err
		}
	}

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

func (b *SingleNodeBackend) startProcess(worker *api.WorkerInfo) error {
	runningInfo := worker.WorkerRunningInfo
	if runningInfo.Executable == "" {
		return nil
	}

	cmd := exec.Command(runningInfo.Executable, runningInfo.Args...)
	cmd.Env = os.Environ()
	for k, v := range runningInfo.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	if runningInfo.WorkingDir != "" {
		cmd.Dir = runningInfo.WorkingDir
	}

	if err := cmd.Start(); err != nil {
		return err
	}

	pid := uint32(cmd.Process.Pid)
	runningInfo.PID = pid
	runningInfo.IsRunning = true
	runningInfo.Restarts = 0

	// Store process state
	b.processesMu.Lock()
	b.processes[worker.WorkerUID] = &processState{
		cmd:        cmd,
		retryCount: 0,
		lastRetry:  time.Now(),
	}
	b.processesMu.Unlock()

	// Start goroutine to wait for process
	go b.waitForProcess(worker.WorkerUID, cmd)

	klog.Infof("Started process for worker %s: PID=%d", worker.WorkerUID, pid)
	return nil
}

func (b *SingleNodeBackend) waitForProcess(workerUID string, cmd *exec.Cmd) {
	err := cmd.Wait()
	exitCode := 0
	if err != nil {
		if exitError, ok := err.(*exec.ExitError); ok {
			exitCode = exitError.ExitCode()
		}
	}

	b.processesMu.Lock()
	ps, exists := b.processes[workerUID]
	if !exists {
		b.processesMu.Unlock()
		return
	}
	b.processesMu.Unlock()

	// Update worker info
	b.mu.Lock()
	worker, exists := b.workers[workerUID]
	if !exists {
		b.mu.Unlock()
		// Worker was removed, clean up process state
		b.processesMu.Lock()
		delete(b.processes, workerUID)
		b.processesMu.Unlock()
		return
	}

	if worker.WorkerRunningInfo != nil {
		worker.WorkerRunningInfo.IsRunning = false
		worker.WorkerRunningInfo.ExitCode = exitCode
		worker.WorkerRunningInfo.PID = 0
	}
	b.mu.Unlock()

	// Check again if worker still exists before updating file state and retry state
	b.mu.RLock()
	_, stillExists := b.workers[workerUID]
	b.mu.RUnlock()

	if !stillExists {
		// Worker was removed, clean up process state
		b.processesMu.Lock()
		delete(b.processes, workerUID)
		b.processesMu.Unlock()
		klog.Infof("Process for removed worker %s exited, cleaning up", workerUID)
		return
	}

	// Update file state only if worker still exists
	_ = b.fileState.AddWorker(worker)

	// Update retry state
	ps.mu.Lock()
	ps.retryCount++
	ps.lastRetry = time.Now()
	ps.cmd = nil
	ps.mu.Unlock()

	// Notify subscribers of the status change
	b.notifySubscribers(worker)

	klog.Warningf("Process for worker %s exited with code %d, will retry (attempt %d)", workerUID, exitCode, ps.retryCount)
}

func (b *SingleNodeBackend) StopWorker(workerUID string) error {
	// First remove from workers map to prevent waitForProcess from updating retry state
	b.mu.Lock()
	delete(b.workers, workerUID)
	b.mu.Unlock()

	// Stop process if running
	b.processesMu.Lock()
	ps, exists := b.processes[workerUID]
	if exists {
		if ps.cmd != nil && ps.cmd.Process != nil {
			_ = ps.cmd.Process.Signal(syscall.SIGTERM)
			// Give it a moment to gracefully shutdown
			time.Sleep(100 * time.Millisecond)
			if ps.cmd.ProcessState == nil || !ps.cmd.ProcessState.Exited() {
				_ = ps.cmd.Process.Kill()
			}
			klog.Infof("Stopped process for worker: %s", workerUID)
		}
		delete(b.processes, workerUID)
	}
	b.processesMu.Unlock()

	if err := b.fileState.RemoveWorker(workerUID); err != nil {
		return err
	}

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

func (b *SingleNodeBackend) processReconcileLoop() {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-b.stopCh:
			return
		case <-b.ctx.Done():
			return
		case <-ticker.C:
			b.reconcileProcesses()
		}
	}
}

func (b *SingleNodeBackend) reconcileProcesses() {
	// Collect workers that need retry
	type retryInfo struct {
		workerUID string
		worker    *api.WorkerInfo
		ps        *processState
	}
	var toRetry []retryInfo

	b.processesMu.Lock()
	for workerUID, ps := range b.processes {
		// Check if worker still exists
		b.mu.RLock()
		worker, exists := b.workers[workerUID]
		b.mu.RUnlock()

		if !exists {
			// Worker was removed, clean up
			if ps.cmd != nil && ps.cmd.Process != nil {
				_ = ps.cmd.Process.Kill()
			}
			delete(b.processes, workerUID)
			continue
		}

		// Skip if process is running
		if ps.cmd != nil && ps.cmd.Process != nil {
			// Check if process is still alive
			if err := ps.cmd.Process.Signal(syscall.Signal(0)); err != nil {
				// Process is dead, mark it
				ps.mu.Lock()
				ps.cmd = nil
				ps.mu.Unlock()
			} else {
				continue
			}
		}

		// Process is not running, check if we should retry
		ps.mu.Lock()
		retryCount := ps.retryCount
		lastRetry := ps.lastRetry
		ps.mu.Unlock()

		// Calculate backoff delay
		backoffDelay := calculateBackoffDelay(retryCount)
		timeSinceLastRetry := time.Since(lastRetry)

		if timeSinceLastRetry >= backoffDelay {
			// Ready to retry
			if worker.WorkerRunningInfo != nil && worker.WorkerRunningInfo.Type == api.WorkerRuntimeTypeProcess {
				toRetry = append(toRetry, retryInfo{
					workerUID: workerUID,
					worker:    worker,
					ps:        ps,
				})
			}
		}
	}
	b.processesMu.Unlock()

	// Retry processes outside of locks to avoid deadlock
	for _, info := range toRetry {
		if err := b.retryStartProcess(info.worker, info.ps); err != nil {
			klog.Errorf("Failed to retry start process for worker %s: %v", info.workerUID, err)
			info.ps.mu.Lock()
			info.ps.lastRetry = time.Now()
			info.ps.mu.Unlock()
		}
	}
}

func (b *SingleNodeBackend) retryStartProcess(worker *api.WorkerInfo, ps *processState) error {
	runningInfo := worker.WorkerRunningInfo
	if runningInfo.Executable == "" {
		return nil
	}

	cmd := exec.Command(runningInfo.Executable, runningInfo.Args...)
	cmd.Env = os.Environ()
	for k, v := range runningInfo.Env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	if runningInfo.WorkingDir != "" {
		cmd.Dir = runningInfo.WorkingDir
	}

	if err := cmd.Start(); err != nil {
		return err
	}

	pid := uint32(cmd.Process.Pid)
	runningInfo.PID = pid
	runningInfo.IsRunning = true
	runningInfo.Restarts++

	ps.mu.Lock()
	ps.cmd = cmd
	ps.lastRetry = time.Now()
	ps.mu.Unlock()

	// Start goroutine to wait for process
	go b.waitForProcess(worker.WorkerUID, cmd)

	// Update worker in map
	b.mu.Lock()
	b.workers[worker.WorkerUID] = worker
	b.mu.Unlock()

	// Update file state
	_ = b.fileState.AddWorker(worker)

	// Notify subscribers
	b.notifySubscribers(worker)

	klog.Infof("Retried start process for worker %s: PID=%d (restart #%d)", worker.WorkerUID, pid, runningInfo.Restarts)
	return nil
}

func calculateBackoffDelay(retryCount int64) time.Duration {
	const (
		baseDelay = 3 * time.Second
		maxDelay  = 60 * time.Second
		factor    = 2.0
	)

	if retryCount <= 0 {
		return baseDelay
	}

	backoff := float64(baseDelay) * math.Pow(factor, float64(retryCount-1))
	if backoff > float64(maxDelay) {
		backoff = float64(maxDelay)
	}

	return time.Duration(backoff)
}
