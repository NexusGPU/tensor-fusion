package single_node

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"syscall"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/google/uuid"
	"k8s.io/klog/v2"
)

// processState holds per-worker process info. All fields are protected by
// SingleNodeBackend.processesMu — no per-process lock is needed because:
//   - reconcileProcesses (the only RLock reader) runs on a single goroutine
//   - waitForProcess and restartProcess acquire the exclusive Lock
//   - StopWorker acquires the exclusive Lock
type processState struct {
	cmd           *exec.Cmd
	done          chan struct{}
	retryCount    int64
	lastRetry     time.Time
	lastExitCode  int
	lastExitError string
	isRunning     bool

	// Set at creation; only env is updated (under processesMu) on restart.
	executable string
	args       []string
	env        map[string]string
	workingDir string
	logDir     string
}

type workerSubscriber struct {
	handler framework.WorkerChangeHandler
	changes chan struct{}
	mu      sync.Mutex
	workers map[string]*api.WorkerInfo
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
	lifecycleMu          sync.Mutex // serializes start, restart and stop
	processWG            sync.WaitGroup

	// Worker watching
	subscribersMu sync.RWMutex
	subscribers   map[string]*workerSubscriber
	workerHandler *workerSubscriber

	// Process management
	processesMu sync.RWMutex
	processes   map[string]*processState
	logDir      string
	stateDir    string
}

// BackendOption is a functional option for configuring SingleNodeBackend
type BackendOption func(*SingleNodeBackend)

// WithStateDir sets a custom state directory for the backend
func WithStateDir(stateDir string) BackendOption {
	return func(b *SingleNodeBackend) {
		b.stateDir = stateDir
	}
}

func NewSingleNodeBackend(
	ctx context.Context,
	deviceController framework.DeviceController,
	allocationController framework.WorkerAllocationController,
	opts ...BackendOption,
) *SingleNodeBackend {
	b := &SingleNodeBackend{
		ctx:                  ctx,
		deviceController:     deviceController,
		allocationController: allocationController,
		workers:              make(map[string]*api.WorkerInfo),
		stopCh:               make(chan struct{}),
		subscribers:          make(map[string]*workerSubscriber),
		processes:            make(map[string]*processState),
	}

	// Apply options first to allow stateDir override
	for _, opt := range opts {
		opt(b)
	}

	// Determine state directory
	stateDir := b.stateDir
	if stateDir == "" {
		stateDir = os.Getenv("TENSOR_FUSION_STATE_DIR")
	}
	if stateDir == "" {
		stateDir = "/tmp/tensor-fusion-state"
	}
	b.stateDir = stateDir

	// Determine log directory
	logDir := os.Getenv(constants.TFLogPathEnv)
	if logDir == "" {
		logDir = filepath.Join(stateDir, "logs")
	}
	b.logDir = logDir

	// Ensure log directory exists
	_ = os.MkdirAll(logDir, 0755)

	// Initialize file state manager
	b.fileState = NewFileStateManager(stateDir)

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

	return b
}

func (b *SingleNodeBackend) Start() error {
	b.lifecycleMu.Lock()
	defer b.lifecycleMu.Unlock()
	if err := b.checkStopped(); err != nil {
		return err
	}
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
	b.lifecycleMu.Lock()
	defer b.lifecycleMu.Unlock()
	// Use sync.Once to ensure stopCh is only closed once
	b.stopOnce.Do(func() {
		close(b.stopCh)
	})

	// Close all subscriber channels
	b.subscribersMu.Lock()
	for id, subscriber := range b.subscribers {
		close(subscriber.changes)
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
	b.processWG.Wait()

	return nil
}

func (b *SingleNodeBackend) checkStopped() error {
	select {
	case <-b.stopCh:
		return fmt.Errorf("single node backend is stopped")
	case <-b.ctx.Done():
		return b.ctx.Err()
	default:
		return nil
	}
}

// loadState loads workers and devices from file state
func (b *SingleNodeBackend) loadState() error {
	b.mu.Lock()
	workers, err := b.fileState.LoadWorkers()
	if err != nil {
		b.mu.Unlock()
		return err
	}

	b.workers = workers
	b.mu.Unlock()
	b.notifySubscribers(nil)

	return nil
}

// discoverWorkers discovers workers from file state and notifies subscribers of changes
func (b *SingleNodeBackend) discoverWorkers() {
	b.lifecycleMu.Lock()
	defer b.lifecycleMu.Unlock()
	if b.checkStopped() != nil {
		return
	}
	b.mu.Lock()
	workers, err := b.fileState.LoadWorkers()
	if err != nil {
		b.mu.Unlock()
		klog.Errorf("Failed to load workers from file state: %v", err)
		return
	}

	changed := false
	for uid, worker := range workers {
		if worker.DeletedAt > 0 || worker.Status == api.WorkerStatusTerminated {
			continue
		}
		oldWorker, exists := b.workers[uid]
		if !exists || !reflect.DeepEqual(oldWorker, worker) {
			b.workers[uid] = worker
			changed = true
		}
	}
	var removed []string
	for uid := range b.workers {
		worker := workers[uid]
		if worker == nil || worker.DeletedAt > 0 || worker.Status == api.WorkerStatusTerminated {
			removed = append(removed, uid)
		}
	}
	b.mu.Unlock()
	// An external file deletion is also a stop request. Keep the cache entry
	// until the managed process has exited, so OnRemove cannot free a live GPU.
	for _, uid := range removed {
		if err := b.stopWorker(uid); err != nil {
			klog.Errorf("Failed to stop file-removed worker %s: %v", uid, err)
		}
	}

	if changed {
		b.notifySubscribers(nil)
	}
}

// Notifications only wake subscribers. Their snapshots determine the lifecycle
// changes, so a slow consumer cannot lose a deletion in a burst.
func (b *SingleNodeBackend) notifySubscribers(_ *api.WorkerInfo) {
	b.subscribersMu.RLock()
	defer b.subscribersMu.RUnlock()

	for _, subscriber := range b.subscribers {
		select {
		case subscriber.changes <- struct{}{}:
		default:
		}
	}
}

func (b *SingleNodeBackend) syncSubscriber(subscriber *workerSubscriber) {
	subscriber.mu.Lock()
	defer subscriber.mu.Unlock()
	current := make(map[string]*api.WorkerInfo)
	for _, info := range b.ListWorkers() {
		if info.DeletedAt == 0 {
			current[info.WorkerUID] = info
		}
	}
	for uid, old := range subscriber.workers {
		if current[uid] == nil && subscriber.handler.OnRemove != nil {
			subscriber.handler.OnRemove(old)
		}
	}
	for uid, info := range current {
		old := subscriber.workers[uid]
		if old == nil && subscriber.handler.OnAdd != nil {
			subscriber.handler.OnAdd(info)
		} else if old != nil && !reflect.DeepEqual(old, info) && subscriber.handler.OnUpdate != nil {
			subscriber.handler.OnUpdate(old, info)
		}
	}
	subscriber.workers = current
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
	subscriberID := uuid.NewString()
	subscriber := &workerSubscriber{
		handler: handler, changes: make(chan struct{}, 1), workers: make(map[string]*api.WorkerInfo),
	}

	// Register subscriber
	b.subscribersMu.Lock()
	select {
	case <-b.stopCh:
		b.subscribersMu.Unlock()
		return fmt.Errorf("single node backend is stopped")
	default:
	}
	if handler.OnPrepare != nil {
		b.workerHandler = subscriber
	}
	b.subscribers[subscriberID] = subscriber
	b.subscribersMu.Unlock()

	// Replay existing workers as well as future changes.
	go func() {
		defer func() {
			b.subscribersMu.Lock()
			delete(b.subscribers, subscriberID)
			b.subscribersMu.Unlock()
		}()

		b.syncSubscriber(subscriber)
		for {
			select {
			case <-b.ctx.Done():
				return
			case <-b.stopCh:
				return
			case _, ok := <-subscriber.changes:
				if !ok {
					return
				}
				b.syncSubscriber(subscriber)
			}
		}
	}()
	return nil
}

func (b *SingleNodeBackend) StartWorker(worker *api.WorkerInfo) (err error) {
	b.lifecycleMu.Lock()
	defer b.lifecycleMu.Unlock()
	if err := b.checkStopped(); err != nil {
		return err
	}
	b.mu.RLock()
	_, exists := b.workers[worker.WorkerUID]
	b.mu.RUnlock()
	if exists {
		return fmt.Errorf("worker %s already exists", worker.WorkerUID)
	}
	worker = worker.DeepCopy()
	isProcess := worker.WorkerRunningInfo != nil && worker.WorkerRunningInfo.Type == api.WorkerRuntimeTypeProcess
	if isProcess {
		worker.WorkerRunningInfo.PID = 0
		worker.WorkerRunningInfo.IsRunning = false
	}

	b.mu.Lock()
	if err := b.fileState.AddWorker(worker); err != nil {
		b.mu.Unlock()
		return err
	}

	// Register worker after persistence to avoid periodic file-discovery replacing
	// freshly added in-memory workers with stale snapshots.
	b.workers[worker.WorkerUID] = worker.DeepCopy()
	b.mu.Unlock()
	defer func() {
		if err != nil {
			err = errors.Join(err, b.stopWorker(worker.WorkerUID))
		}
	}()

	if isProcess {
		if err := b.prepareProcessWorker(worker); err != nil {
			return err
		}
		if err := b.checkStopped(); err != nil {
			return err
		}
		// Publish the PID before the exit callback can update this worker. This
		// also handles processes that exit immediately after exec.
		b.mu.Lock()
		err = b.startProcess(worker)
		if err == nil {
			b.workers[worker.WorkerUID] = worker.DeepCopy()
			err = b.fileState.AddWorker(worker)
		}
		b.mu.Unlock()
		if err != nil {
			return err
		}
	}

	b.notifySubscribers(worker)
	klog.Infof("Worker started: %s", worker.WorkerUID)
	return nil
}

func (b *SingleNodeBackend) prepareProcessWorker(worker *api.WorkerInfo) error {
	if len(worker.AllocatedDevices) == 0 ||
		(worker.IsolationMode != tfv1.IsolationModeSoft && worker.IsolationMode != tfv1.IsolationModeHard) {
		return nil
	}
	b.subscribersMu.RLock()
	subscriber := b.workerHandler
	b.subscribersMu.RUnlock()
	if subscriber == nil {
		return fmt.Errorf("worker %s has no preparation handler", worker.WorkerUID)
	}
	// Deliver OnAdd before asking WorkerController to prepare its mapping.
	b.syncSubscriber(subscriber)
	return subscriber.handler.OnPrepare(worker.WorkerUID)
}

// buildCmd creates exec.Cmd with proper environment and log redirection
func (b *SingleNodeBackend) buildCmd(ps *processState) (*exec.Cmd, io.Closer, error) {
	if ps.executable == "" {
		return nil, nil, fmt.Errorf("executable is empty")
	}

	cmd := exec.Command(ps.executable, ps.args...)

	// Build environment: start with current environment, then override with ps.env
	envMap := make(map[string]string)
	for _, env := range os.Environ() {
		parts := strings.SplitN(env, "=", 2)
		if len(parts) == 2 {
			envMap[parts[0]] = parts[1]
		}
	}
	// Override with custom environment variables
	for k, v := range ps.env {
		envMap[k] = v
	}
	// Convert back to []string format
	cmd.Env = make([]string, 0, len(envMap))
	for k, v := range envMap {
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
		done:       make(chan struct{}),
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
	done := ps.done
	b.processWG.Add(1)
	go func() {
		defer b.processWG.Done()
		b.waitForProcess(worker.WorkerUID, cmd, logFile, done)
	}()

	klog.Infof("✓ Process started for worker %s: PID=%d, executable=%s", worker.WorkerUID, pid, ps.executable)
	return nil
}

func (b *SingleNodeBackend) waitForProcess(workerUID string, cmd *exec.Cmd, logFile io.Closer, done chan struct{}) {
	defer close(done)
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

	// Update process state - this is the source of truth for process status
	b.processesMu.Lock()
	ps, exists := b.processes[workerUID]
	if !exists || ps.cmd != cmd {
		b.processesMu.Unlock()
		klog.Warningf("⚠ Process exited but processState not found for worker %s (already cleaned up)", workerUID)
		return
	}

	ps.cmd = nil
	ps.isRunning = false
	ps.lastExitCode = exitCode
	ps.lastExitError = exitError
	ps.retryCount++
	ps.lastRetry = time.Now()
	retryCount := ps.retryCount
	b.processesMu.Unlock()

	// Log process exit prominently
	klog.Warningf("═══════════════════════════════════════════════════════════════")
	klog.Warningf("⛔ PROCESS EXITED: worker=%s exitCode=%d retryCount=%d", workerUID, exitCode, retryCount)
	if exitError != "" {
		klog.Warningf("   Exit reason: %s", exitError)
	}
	klog.Warningf("═══════════════════════════════════════════════════════════════")

	// Update worker info in workers map (best effort, may have been removed)
	b.mu.Lock()
	worker, workerExists := b.workers[workerUID]
	if workerExists && worker.WorkerRunningInfo != nil && worker.WorkerRunningInfo.PID == uint32(cmd.Process.Pid) {
		worker = worker.DeepCopy()
		worker.WorkerRunningInfo.IsRunning = false
		worker.WorkerRunningInfo.ExitCode = exitCode
		worker.WorkerRunningInfo.PID = 0
		b.workers[workerUID] = worker
		_ = b.fileState.AddWorker(worker)
	}
	b.mu.Unlock()

	// Update file state and notify if worker still exists
	if workerExists {
		b.notifySubscribers(worker)
	}
}

func (b *SingleNodeBackend) StopWorker(workerUID string) error {
	b.lifecycleMu.Lock()
	defer b.lifecycleMu.Unlock()
	return b.stopWorker(workerUID)
}

// stopWorker requires lifecycleMu to exclude starts and restarts.
func (b *SingleNodeBackend) stopWorker(workerUID string) error {
	klog.Infof("Stopping worker: %s", workerUID)

	// Wait for the actual process exit before releasing its GPU. Lifecycle
	// serialization prevents a restart from slipping between signal and removal.
	b.processesMu.RLock()
	ps, exists := b.processes[workerUID]
	var cmd *exec.Cmd
	var done chan struct{}
	if exists {
		cmd, done = ps.cmd, ps.done
	}
	b.processesMu.RUnlock()
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Signal(syscall.SIGTERM)
	}
	if done != nil {
		select {
		case <-done:
		case <-time.After(100 * time.Millisecond):
			if cmd != nil && cmd.Process != nil {
				_ = cmd.Process.Kill()
			}
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				return fmt.Errorf("worker %s has not exited; keeping its allocation", workerUID)
			}
		}
	}
	b.processesMu.Lock()
	delete(b.processes, workerUID)
	b.processesMu.Unlock()

	// Serialize persistence with discovery so its snapshot cannot restore a
	// stopped worker between cache removal and file removal.
	b.mu.Lock()
	if err := b.fileState.RemoveWorker(workerUID); err != nil {
		b.mu.Unlock()
		klog.Errorf("Failed to remove worker %s from file state: %v", workerUID, err)
		return err
	}
	delete(b.workers, workerUID)
	b.mu.Unlock()
	b.notifySubscribers(nil)
	// Add and Remove can be coalesced before any subscriber sees the worker.
	// Explicit stop still owns the responsibility to release its allocation.
	if b.allocationController != nil {
		if err := b.allocationController.DeallocateWorker(workerUID); err != nil {
			return err
		}
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
	workers := make([]*api.WorkerInfo, 0, len(b.workers))
	for _, worker := range b.workers {
		workers = append(workers, worker.DeepCopy())
	}
	return workers
}

// UpdateWorkerEnv updates environment variables for a worker without restarting its process.
// The new env vars will take effect on next process restart (crash recovery).
func (b *SingleNodeBackend) UpdateWorkerEnv(workerUID string, env map[string]string) error {
	b.mu.Lock()
	worker, exists := b.workers[workerUID]
	if !exists {
		b.mu.Unlock()
		return fmt.Errorf("worker %s not found", workerUID)
	}
	if worker.WorkerRunningInfo == nil {
		b.mu.Unlock()
		return fmt.Errorf("worker %s has no running info", workerUID)
	}
	worker = worker.DeepCopy()
	worker.WorkerRunningInfo.Env = maps.Clone(env)
	if err := b.fileState.AddWorker(worker); err != nil {
		b.mu.Unlock()
		return fmt.Errorf("failed to persist worker env update: %w", err)
	}
	b.workers[workerUID] = worker
	b.mu.Unlock()
	b.notifySubscribers(worker)

	klog.Infof("Updated env vars for worker %s (will apply on next process restart)", workerUID)
	return nil
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
		// Skip if already running
		if ps.isRunning && ps.cmd != nil {
			continue
		}

		// Skip if no executable configured
		if ps.executable == "" {
			continue
		}

		// Calculate backoff delay
		if time.Since(ps.lastRetry) >= calculateBackoffDelay(ps.retryCount) {
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

// restartProcess restarts a process for the given worker.
//
// lifecycleMu serializes process creation with StopWorker and Stop. Worker and
// process state locks are taken separately so exit callbacks can finish.
func (b *SingleNodeBackend) restartProcess(workerUID string) error {
	b.lifecycleMu.Lock()
	defer b.lifecycleMu.Unlock()
	if err := b.checkStopped(); err != nil {
		return err
	}
	// Step 1: Read the current worker before preparing any restart.
	var envUpdate map[string]string
	b.mu.RLock()
	info := b.workers[workerUID].DeepCopy()
	if info != nil && info.WorkerRunningInfo != nil && info.WorkerRunningInfo.Env != nil {
		envUpdate = make(map[string]string, len(info.WorkerRunningInfo.Env))
		maps.Copy(envUpdate, info.WorkerRunningInfo.Env)
	}
	b.mu.RUnlock()
	if info == nil {
		return fmt.Errorf("worker %s no longer exists", workerUID)
	}
	if err := b.prepareProcessWorker(info); err != nil {
		return err
	}

	// Step 2: Under processesMu, verify state, merge env, and build cmd
	b.processesMu.Lock()
	ps, exists := b.processes[workerUID]
	if !exists {
		b.processesMu.Unlock()
		return fmt.Errorf("process state not found for worker %s", workerUID)
	}
	if ps.isRunning && ps.cmd != nil {
		b.processesMu.Unlock()
		return nil
	}

	if envUpdate != nil {
		maps.Copy(ps.env, envUpdate)
	}

	cmd, logFile, err := b.buildCmd(ps)
	if err != nil {
		b.processesMu.Unlock()
		return fmt.Errorf("failed to build cmd: %w", err)
	}

	retryCount := ps.retryCount
	executable := ps.executable
	args := ps.args
	b.processesMu.Unlock()

	// Step 3: Start process outside lock (may block briefly on exec)
	klog.Infof("Restarting process for worker %s: %s %v (retry #%d)", workerUID, executable, args, retryCount)

	if err := cmd.Start(); err != nil {
		if logFile != nil {
			_ = logFile.Close()
		}
		b.processesMu.Lock()
		if ps, ok := b.processes[workerUID]; ok {
			ps.lastRetry = time.Now()
		}
		b.processesMu.Unlock()
		return fmt.Errorf("failed to start process: %w", err)
	}

	pid := uint32(cmd.Process.Pid)

	// Step 4: Update process state — re-check existence in case StopWorker ran concurrently
	b.processesMu.Lock()
	if _, stillExists := b.processes[workerUID]; !stillExists {
		b.processesMu.Unlock()
		// Worker was removed while we were starting — kill the orphaned process
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		if logFile != nil {
			_ = logFile.Close()
		}
		return nil
	}
	ps.cmd = cmd
	ps.done = make(chan struct{})
	done := ps.done
	ps.isRunning = true
	ps.lastRetry = time.Now()
	restartCount := ps.retryCount
	b.processesMu.Unlock()

	// Step 5: Update worker running info
	b.mu.Lock()
	worker, workerExists := b.workers[workerUID]
	if workerExists && worker.WorkerRunningInfo != nil {
		worker = worker.DeepCopy()
		worker.WorkerRunningInfo.PID = pid
		worker.WorkerRunningInfo.IsRunning = true
		worker.WorkerRunningInfo.Restarts = int(restartCount)
		worker.WorkerRunningInfo.ExitCode = 0
		b.workers[workerUID] = worker
		_ = b.fileState.AddWorker(worker)
	}
	b.mu.Unlock()

	// Start goroutine to wait for process exit
	b.processWG.Add(1)
	go func() {
		defer b.processWG.Done()
		b.waitForProcess(workerUID, cmd, logFile, done)
	}()

	// Update file state
	if workerExists {
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
