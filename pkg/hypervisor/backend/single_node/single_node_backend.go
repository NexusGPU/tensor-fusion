package single_node

import (
	"context"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/google/uuid"
	"github.com/samber/lo"
	"k8s.io/klog/v2"
)

type processState struct {
	cmd           *exec.Cmd
	retryCount    int64
	lastRetry     time.Time
	lastExitCode  int
	lastExitError string
	isRunning     bool
	mu            sync.Mutex

	// Copy of runtime info for restart, avoid race with worker map
	executable string
	args       []string
	env        map[string]string
	workingDir string
	logDir     string
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
	logDir      string
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
	logDir := os.Getenv(constants.TFLogPathEnv)
	if logDir == "" {
		logDir = filepath.Join(stateDir, "logs")
	}
	// Ensure log directory exists
	_ = os.MkdirAll(logDir, 0755)

	if lvl := os.Getenv(constants.TFLogLevelEnv); lvl != "" {
		var v string
		switch lvl {
		case "trace":
			v = "6"
		case "debug":
			v = "4"
		case "info":
			v = "2"
		case "warn", "error":
			v = "0"
		default:
			v = "2"
		}
		if err := flag.Set("v", v); err != nil {
			klog.Warningf("Failed to set klog level from env %s: "+
				"%v (flag -v might be undefined or used for other purpose)",
				constants.TFLogLevelEnv, err)
		} else {
			klog.Infof("Set klog level to v=%s from env %s=%s", v, constants.TFLogLevelEnv, lvl)
		}
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
		logDir:               logDir,
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

// buildCmd creates exec.Cmd with proper environment and log redirection
func (b *SingleNodeBackend) buildCmd(ps *processState) (*exec.Cmd, io.Closer, error) {
	if ps.executable == "" {
		return nil, nil, fmt.Errorf("executable is empty")
	}

	cmd := exec.Command(ps.executable, ps.args...)
	cmd.Env = os.Environ()
	for k, v := range ps.env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}
	if ps.workingDir != "" {
		cmd.Dir = ps.workingDir
	}

	// Set up log file for stdout/stderr
	var logFile *os.File
	if ps.logDir != "" {
		logPath := filepath.Join(ps.logDir, fmt.Sprintf("worker-%d.log", time.Now().UnixNano()))
		var err error
		logFile, err = os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			klog.Warningf("Failed to create log file %s: %v, using os.Stderr", logPath, err)
			cmd.Stdout = os.Stdout
			cmd.Stderr = os.Stderr
		} else {
			cmd.Stdout = logFile
			cmd.Stderr = logFile
			klog.V(2).Infof("Process logs will be written to: %s", logPath)
		}
	} else {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	return cmd, logFile, nil
}

func (b *SingleNodeBackend) startProcess(worker *api.WorkerInfo) error {
	runningInfo := worker.WorkerRunningInfo
	if runningInfo == nil {
		return fmt.Errorf("worker %s has no running info", worker.WorkerUID)
	}
	if runningInfo.Executable == "" {
		return fmt.Errorf("executable is empty for worker %s", worker.WorkerUID)
	}

	// Create process state with copied runtime info
	ps := &processState{
		retryCount: 0,
		lastRetry:  time.Now(),
		isRunning:  false,
		executable: runningInfo.Executable,
		args:       append([]string{}, runningInfo.Args...),
		env:        make(map[string]string),
		workingDir: runningInfo.WorkingDir,
		logDir:     b.logDir,
	}
	for k, v := range runningInfo.Env {
		ps.env[k] = v
	}

	cmd, logFile, err := b.buildCmd(ps)
	if err != nil {
		return fmt.Errorf("failed to build cmd for worker %s: %w", worker.WorkerUID, err)
	}

	klog.Infof("Starting process for worker %s: %s %v", worker.WorkerUID, ps.executable, ps.args)

	if err := cmd.Start(); err != nil {
		if logFile != nil {
			_ = logFile.Close()
		}
		return fmt.Errorf("failed to start process for worker %s: %w", worker.WorkerUID, err)
	}

	pid := uint32(cmd.Process.Pid)
	runningInfo.PID = pid
	runningInfo.IsRunning = true
	runningInfo.Restarts = 0

	ps.cmd = cmd
	ps.isRunning = true

	// Store process state
	b.processesMu.Lock()
	b.processes[worker.WorkerUID] = ps
	b.processesMu.Unlock()

	// Start goroutine to wait for process exit
	go b.waitForProcess(worker.WorkerUID, cmd, logFile)

	klog.Infof("✓ Process started for worker %s: PID=%d, executable=%s", worker.WorkerUID, pid, ps.executable)
	return nil
}

func (b *SingleNodeBackend) waitForProcess(workerUID string, cmd *exec.Cmd, logFile io.Closer) {
	err := cmd.Wait()

	// Close log file if exists
	if logFile != nil {
		_ = logFile.Close()
	}

	exitCode := 0
	exitError := ""
	if err != nil {
		exitError = err.Error()
		if ee, ok := err.(*exec.ExitError); ok {
			exitCode = ee.ExitCode()
			// Include stderr if available
			if len(ee.Stderr) > 0 {
				exitError = fmt.Sprintf("%s: %s", err.Error(), string(ee.Stderr))
			}
		}
	}

	// Update process state atomically - this is the source of truth for process status
	b.processesMu.Lock()
	ps, exists := b.processes[workerUID]
	if !exists {
		b.processesMu.Unlock()
		klog.Warningf("⚠ Process exited but processState not found for worker %s (already cleaned up)", workerUID)
		return
	}

	ps.mu.Lock()
	ps.cmd = nil
	ps.isRunning = false
	ps.lastExitCode = exitCode
	ps.lastExitError = exitError
	ps.retryCount++
	ps.lastRetry = time.Now()
	ps.mu.Unlock()
	b.processesMu.Unlock()

	// Log process exit prominently
	klog.Warningf("═══════════════════════════════════════════════════════════════")
	klog.Warningf("⛔ PROCESS EXITED: worker=%s exitCode=%d retryCount=%d", workerUID, exitCode, ps.retryCount)
	if exitError != "" {
		klog.Warningf("   Exit reason: %s", exitError)
	}
	klog.Warningf("═══════════════════════════════════════════════════════════════")

	// Update worker info in workers map (best effort, may have been removed)
	b.mu.Lock()
	worker, workerExists := b.workers[workerUID]
	if workerExists && worker.WorkerRunningInfo != nil {
		worker.WorkerRunningInfo.IsRunning = false
		worker.WorkerRunningInfo.ExitCode = exitCode
		worker.WorkerRunningInfo.PID = 0
	}
	b.mu.Unlock()

	// Update file state and notify if worker still exists
	if workerExists {
		_ = b.fileState.AddWorker(worker)
		b.notifySubscribers(worker)
	}
}

func (b *SingleNodeBackend) StopWorker(workerUID string) error {
	klog.Infof("Stopping worker: %s", workerUID)

	// Stop process if running - must do this BEFORE removing from maps
	// to prevent race with reconcile loop
	b.processesMu.Lock()
	ps, exists := b.processes[workerUID]
	if exists {
		ps.mu.Lock()
		if ps.cmd != nil && ps.cmd.Process != nil {
			klog.Infof("Sending SIGTERM to process PID=%d for worker %s", ps.cmd.Process.Pid, workerUID)
			_ = ps.cmd.Process.Signal(syscall.SIGTERM)
			ps.mu.Unlock()
			b.processesMu.Unlock()

			// Wait outside lock for graceful shutdown
			time.Sleep(100 * time.Millisecond)

			// Check again and force kill if needed
			b.processesMu.Lock()
			ps, exists = b.processes[workerUID]
			if exists {
				ps.mu.Lock()
				if ps.cmd != nil && ps.cmd.Process != nil {
					if ps.cmd.ProcessState == nil || !ps.cmd.ProcessState.Exited() {
						klog.Infof("Force killing process for worker %s", workerUID)
						_ = ps.cmd.Process.Kill()
					}
				}
				ps.mu.Unlock()
			}
		} else {
			ps.mu.Unlock()
		}
		// Remove from processes map
		delete(b.processes, workerUID)
	}
	b.processesMu.Unlock()

	// Remove from workers map
	b.mu.Lock()
	delete(b.workers, workerUID)
	b.mu.Unlock()

	if err := b.fileState.RemoveWorker(workerUID); err != nil {
		klog.Errorf("Failed to remove worker %s from file state: %v", workerUID, err)
		return err
	}

	klog.Infof("✓ Worker stopped: %s", workerUID)
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
	// Collect workerUIDs that need retry - only store UIDs, not pointers
	var toRetry []string

	b.processesMu.RLock()
	for workerUID, ps := range b.processes {
		ps.mu.Lock()
		isRunning := ps.isRunning
		cmd := ps.cmd
		retryCount := ps.retryCount
		lastRetry := ps.lastRetry
		executable := ps.executable
		ps.mu.Unlock()

		// Skip if already running
		if isRunning && cmd != nil {
			continue
		}

		// Skip if no executable configured
		if executable == "" {
			continue
		}

		// Calculate backoff delay
		backoffDelay := calculateBackoffDelay(retryCount)
		timeSinceLastRetry := time.Since(lastRetry)

		if timeSinceLastRetry >= backoffDelay {
			toRetry = append(toRetry, workerUID)
		}
	}
	b.processesMu.RUnlock()

	// Retry processes outside of locks
	for _, workerUID := range toRetry {
		if err := b.restartProcess(workerUID); err != nil {
			klog.Errorf("Failed to restart process for worker %s: %v", workerUID, err)
		}
	}
}

// restartProcess restarts a process for the given worker
func (b *SingleNodeBackend) restartProcess(workerUID string) error {
	b.processesMu.Lock()
	ps, exists := b.processes[workerUID]
	if !exists {
		b.processesMu.Unlock()
		return fmt.Errorf("process state not found for worker %s", workerUID)
	}

	ps.mu.Lock()
	// Double check not already running
	if ps.isRunning && ps.cmd != nil {
		ps.mu.Unlock()
		b.processesMu.Unlock()
		return nil
	}
	ps.mu.Unlock()
	b.processesMu.Unlock()

	// Build and start new cmd
	cmd, logFile, err := b.buildCmd(ps)
	if err != nil {
		return fmt.Errorf("failed to build cmd: %w", err)
	}

	klog.Infof("Restarting process for worker %s: %s %v (retry #%d)", workerUID, ps.executable, ps.args, ps.retryCount)

	if err := cmd.Start(); err != nil {
		if logFile != nil {
			_ = logFile.Close()
		}
		// Update lastRetry to trigger backoff
		ps.mu.Lock()
		ps.lastRetry = time.Now()
		ps.mu.Unlock()
		return fmt.Errorf("failed to start process: %w", err)
	}

	pid := uint32(cmd.Process.Pid)

	// Update process state
	ps.mu.Lock()
	ps.cmd = cmd
	ps.isRunning = true
	ps.lastRetry = time.Now()
	restartCount := ps.retryCount
	ps.mu.Unlock()

	// Update worker running info
	b.mu.Lock()
	worker, workerExists := b.workers[workerUID]
	if workerExists && worker.WorkerRunningInfo != nil {
		worker.WorkerRunningInfo.PID = pid
		worker.WorkerRunningInfo.IsRunning = true
		worker.WorkerRunningInfo.Restarts = int(restartCount)
		worker.WorkerRunningInfo.ExitCode = 0
	}
	b.mu.Unlock()

	// Start goroutine to wait for process exit
	go b.waitForProcess(workerUID, cmd, logFile)

	// Update file state
	if workerExists {
		_ = b.fileState.AddWorker(worker)
		b.notifySubscribers(worker)
	}

	klog.Infof("✓ Process restarted for worker %s: PID=%d (restart #%d)", workerUID, pid, restartCount)
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
