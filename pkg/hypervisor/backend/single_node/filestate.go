package single_node

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
)

const (
	defaultStateDir = "/tmp/tensor-fusion-state"
	workersFile     = "workers.json"
	devicesFile     = "devices.json"
)

// FileStateManager manages file-based state persistence
type FileStateManager struct {
	stateDir string
	mu       sync.RWMutex
}

// NewFileStateManager creates a new file state manager
func NewFileStateManager(stateDir string) *FileStateManager {
	if stateDir == "" {
		stateDir = defaultStateDir
	}
	return &FileStateManager{
		stateDir: stateDir,
	}
}

// ensureStateDir ensures the state directory exists
func (fsm *FileStateManager) ensureStateDir() error {
	return os.MkdirAll(fsm.stateDir, 0755)
}

// SaveWorkers saves workers to JSON file
func (fsm *FileStateManager) SaveWorkers(workers map[string]*api.WorkerInfo) error {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	if err := fsm.ensureStateDir(); err != nil {
		return err
	}

	// Convert map to slice for JSON
	workersList := make([]*api.WorkerInfo, 0, len(workers))
	for _, worker := range workers {
		workersList = append(workersList, worker)
	}

	data, err := json.MarshalIndent(workersList, "", "  ")
	if err != nil {
		return err
	}

	filePath := filepath.Join(fsm.stateDir, workersFile)
	tmpPath := filePath + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return err
	}

	return os.Rename(tmpPath, filePath)
}

// LoadWorkers loads workers from JSON file
func (fsm *FileStateManager) LoadWorkers() (map[string]*api.WorkerInfo, error) {
	fsm.mu.RLock()
	defer fsm.mu.RUnlock()

	filePath := filepath.Join(fsm.stateDir, workersFile)
	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return make(map[string]*api.WorkerInfo), nil
		}
		return nil, err
	}

	var workersList []*api.WorkerInfo
	if err := json.Unmarshal(data, &workersList); err != nil {
		return nil, err
	}

	workers := make(map[string]*api.WorkerInfo, len(workersList))
	for _, worker := range workersList {
		if worker != nil {
			workers[worker.WorkerUID] = worker
		}
	}

	return workers, nil
}

// SaveDevices saves devices to JSON file
func (fsm *FileStateManager) SaveDevices(devices map[string]*api.DeviceInfo) error {
	fsm.mu.Lock()
	defer fsm.mu.Unlock()

	if err := fsm.ensureStateDir(); err != nil {
		return err
	}

	// Convert map to slice for JSON
	devicesList := make([]*api.DeviceInfo, 0, len(devices))
	for _, device := range devices {
		devicesList = append(devicesList, device)
	}

	data, err := json.MarshalIndent(devicesList, "", "  ")
	if err != nil {
		return err
	}

	filePath := filepath.Join(fsm.stateDir, devicesFile)
	tmpPath := filePath + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return err
	}

	return os.Rename(tmpPath, filePath)
}

// LoadDevices loads devices from JSON file
func (fsm *FileStateManager) LoadDevices() (map[string]*api.DeviceInfo, error) {
	fsm.mu.RLock()
	defer fsm.mu.RUnlock()

	filePath := filepath.Join(fsm.stateDir, devicesFile)
	data, err := os.ReadFile(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return make(map[string]*api.DeviceInfo), nil
		}
		return nil, err
	}

	var devicesList []*api.DeviceInfo
	if err := json.Unmarshal(data, &devicesList); err != nil {
		return nil, err
	}

	devices := make(map[string]*api.DeviceInfo, len(devicesList))
	for _, device := range devicesList {
		if device != nil {
			devices[device.UUID] = device
		}
	}

	return devices, nil
}

// AddWorker adds a worker to the state
func (fsm *FileStateManager) AddWorker(worker *api.WorkerInfo) error {
	workers, err := fsm.LoadWorkers()
	if err != nil {
		return err
	}
	workers[worker.WorkerUID] = worker
	return fsm.SaveWorkers(workers)
}

// RemoveWorker removes a worker from the state
func (fsm *FileStateManager) RemoveWorker(workerUID string) error {
	workers, err := fsm.LoadWorkers()
	if err != nil {
		return err
	}
	delete(workers, workerUID)
	return fsm.SaveWorkers(workers)
}

// AddDevice adds a device to the state
func (fsm *FileStateManager) AddDevice(device *api.DeviceInfo) error {
	devices, err := fsm.LoadDevices()
	if err != nil {
		return err
	}
	devices[device.UUID] = device
	return fsm.SaveDevices(devices)
}

// RemoveDevice removes a device from the state
func (fsm *FileStateManager) RemoveDevice(deviceUUID string) error {
	devices, err := fsm.LoadDevices()
	if err != nil {
		return err
	}
	delete(devices, deviceUUID)
	return fsm.SaveDevices(devices)
}

// UpdateDevice updates a device in the state
func (fsm *FileStateManager) UpdateDevice(device *api.DeviceInfo) error {
	return fsm.AddDevice(device)
}
