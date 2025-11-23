package device

import (
	"context"
	"fmt"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"k8s.io/klog/v2"
)

// Controller manages GPU device discovery, allocation, and lifecycle
type Controller struct {
	ctx               context.Context
	mu                sync.RWMutex
	devices           map[string]*api.DeviceInfo // key: device UUID
	allocations       map[string]*api.WorkerInfo // key: worker UID
	deviceToAlloc     map[string][]string        // device UUID -> []worker UID
	accelerator       *AcceleratorInterface
	discoveryInterval time.Duration
}

var _ framework.DeviceController = &Controller{}

// NewController creates a new device manager
func NewController(ctx context.Context, acceleratorLibPath string, discoveryInterval time.Duration) (framework.DeviceController, error) {
	accel, err := NewAcceleratorInterface(acceleratorLibPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create accelerator interface: %w", err)
	}
	return &Controller{
		ctx:               ctx,
		devices:           make(map[string]*api.DeviceInfo),
		allocations:       make(map[string]*api.WorkerInfo),
		deviceToAlloc:     make(map[string][]string),
		accelerator:       accel,
		discoveryInterval: discoveryInterval,
	}, nil
}

// DiscoverDevices discovers all available GPU devices
func (m *Controller) StartDiscoverDevices() error {
	// Initial device discovery
	if err := m.discoverDevices(); err != nil {
		return fmt.Errorf("initial device discovery failed: %w", err)
	}

	go m.periodicDiscovery()
	return nil
}

// discoverDevices discovers all available GPU devices
func (m *Controller) discoverDevices() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Get all devices at once
	devices, err := m.accelerator.GetAllDevices()
	if err != nil {
		return fmt.Errorf("failed to get all devices: %w", err)
	}

	// Update device map
	for _, device := range devices {
		m.devices[device.UUID] = device
	}

	return nil
}

// periodicDiscovery periodically discovers devices
func (m *Controller) periodicDiscovery() {
	ticker := time.NewTicker(m.discoveryInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			if err := m.discoverDevices(); err != nil {
				// Log error but continue
				continue
			}
		}
	}
}

// GetDevices returns all discovered devices
func (m *Controller) GetDevices() []*api.DeviceInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	devices := make([]*api.DeviceInfo, 0, len(m.devices))
	for _, device := range m.devices {
		devices = append(devices, device)
	}
	return devices
}

// getDevice returns a device by UUID (internal method)
func (m *Controller) getDevice(uuid string) (*api.DeviceInfo, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	device, exists := m.devices[uuid]
	return device, exists
}

// Allocate allocates devices for a worker request
func (m *Controller) Allocate(req *api.WorkerInfo) (*api.DeviceAllocateResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Validate devices exist
	for _, deviceUUID := range req.AllocatedDevices {
		if _, exists := m.devices[deviceUUID]; !exists {
			return &api.DeviceAllocateResponse{
				Success: false,
				ErrMsg:  fmt.Sprintf("device not found: %s", deviceUUID),
			}, nil
		}
	}

	// Handle partitioned mode
	if req.IsolationMode == tfv1.IsolationModePartitioned && req.TemplateID != "" {
		partitionUUID, overhead, err := m.accelerator.AssignPartition(req.TemplateID, req.AllocatedDevices[0])
		if err != nil {
			return &api.DeviceAllocateResponse{
				Success: false,
				ErrMsg:  fmt.Sprintf("failed to assign partition: %v", err),
			}, nil
		}
		req.PartitionUUID = partitionUUID
		// Adjust memory limit if needed
		if req.MemoryLimitBytes > 0 && overhead > 0 {
			req.MemoryLimitBytes -= overhead
		}
	}

	// Store allocation
	m.allocations[req.WorkerUID] = &api.WorkerInfo{
		WorkerUID:        req.WorkerUID,
		AllocatedDevices: req.AllocatedDevices,
		IsolationMode:    req.IsolationMode,
		TemplateID:       req.TemplateID,
		MemoryLimit:      req.MemoryLimitBytes,
		ComputeLimit:     req.ComputeLimitUnits,
	}

	// Update device to allocation mapping
	for _, deviceUUID := range req.AllocatedDevices {
		m.deviceToAlloc[deviceUUID] = append(m.deviceToAlloc[deviceUUID], req.WorkerUID)
	}

	return &api.DeviceAllocateResponse{
		DeviceNodes: req.AllocatedDevices,
		Annotations: make(map[string]string),
		Mounts:      make(map[string]string),
		EnvVars:     make(map[string]string),
		Success:     true,
	}, nil
}

// Deallocate de-allocates devices for a pod
func (m *Controller) Deallocate(workerUID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	allocation, exists := m.allocations[workerUID]
	if !exists {
		return fmt.Errorf("allocation not found for pod %s", workerUID)
	}

	// Handle partitioned mode cleanup
	if allocation.IsolationMode == tfv1.IsolationModePartitioned && allocation.TemplateID != "" {
		if err := m.accelerator.RemovePartition(allocation.TemplateID, allocation.AllocatedDevices[0]); err != nil {
			// Log error but continue
			klog.Errorf("failed to remove partition: %v", err)
		}
	}

	// Remove from allocations
	delete(m.allocations, workerUID)

	// Remove from device mapping
	if workerUIDs, exists := m.deviceToAlloc[allocation.DeviceUUID]; exists {
		for i, uid := range workerUIDs {
			if uid == workerUID {
				m.deviceToAlloc[allocation.DeviceUUID] = append(workerUIDs[:i], workerUIDs[i+1:]...)
				break
			}
		}
	}

	return nil
}

// GetAllocation returns allocation for a pod
func (m *Controller) GetAllocation(workerUID string) (*api.WorkerInfo, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	allocation, exists := m.allocations[workerUID]
	return allocation, exists
}

// Start implements framework.DeviceController
func (m *Controller) Start() error {
	// Start device discovery
	return m.StartDiscoverDevices()
}

// DiscoverDevices implements framework.DeviceController
func (m *Controller) DiscoverDevices() error {
	return m.discoverDevices()
}

// AllocateDevice implements framework.DeviceController
func (m *Controller) AllocateDevice(request *api.WorkerInfo) (*api.DeviceAllocateResponse, error) {
	return m.Allocate(request)
}

// ListDevices implements framework.DeviceController
func (m *Controller) ListDevices() ([]*api.DeviceInfo, error) {
	return m.GetDevices(), nil
}

// DevicesUpdates implements framework.DeviceController
func (m *Controller) DevicesUpdates() (<-chan []*api.DeviceInfo, error) {
	ch := make(chan []*api.DeviceInfo, 1)
	// Send initial device list
	go func() {
		devices := m.GetDevices()
		select {
		case ch <- devices:
		default:
		}
		// TODO: Implement proper device updates channel with periodic updates
		// Channel will be closed when controller is stopped
	}()
	return ch, nil
}

// GetDevice implements framework.DeviceController
func (m *Controller) GetDevice(deviceUUID string) (*api.DeviceInfo, error) {
	device, exists := m.getDevice(deviceUUID)
	if !exists {
		return nil, fmt.Errorf("device not found: %s", deviceUUID)
	}
	return device, nil
}

// GetDeviceAllocations implements framework.DeviceController
func (m *Controller) GetDeviceAllocations(deviceUUID string) ([]*api.DeviceAllocation, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if deviceUUID == "" {
		// Return all allocations
		allocations := make([]*api.DeviceAllocation, 0, len(m.allocations))
		for _, allocation := range m.allocations {
			allocations = append(allocations, allocation)
		}
		return allocations, nil
	}

	// Return allocations for specific device
	workerUIDs := m.deviceToAlloc[deviceUUID]
	allocations := make([]*api.DeviceAllocation, 0, len(workerUIDs))
	for _, workerUID := range workerUIDs {
		if allocation, exists := m.allocations[workerUID]; exists {
			allocations = append(allocations, allocation)
		}
	}
	return allocations, nil
}

// GetDeviceAllocationUpdates implements framework.DeviceController
func (m *Controller) GetDeviceAllocationUpdates(deviceUUID string, allocationID string) (<-chan []*api.DeviceAllocation, error) {
	ch := make(chan []*api.DeviceAllocation, 1)
	// Send initial allocation list
	go func() {
		allocations, err := m.GetDeviceAllocations(deviceUUID)
		if err == nil {
			select {
			case ch <- allocations:
			default:
			}
		}
		// TODO: Implement proper allocation updates channel with periodic updates
		// Channel will be closed when controller is stopped
	}()
	return ch, nil
}

// GetGPUMetrics implements framework.DeviceController
func (m *Controller) GetGPUMetrics() (map[string]*api.GPUUsageMetrics, error) {
	m.mu.RLock()
	devices := make([]*api.DeviceInfo, 0, len(m.devices))
	for _, device := range m.devices {
		devices = append(devices, device)
	}
	m.mu.RUnlock()

	// Get device metrics from accelerator interface
	// Note: This requires GetDeviceMetrics from accelerator.h which needs to be implemented
	// For now, we'll use process-level metrics to aggregate
	result := make(map[string]*api.GPUUsageMetrics)

	// Get memory utilization from processes
	memUtils, err := m.accelerator.GetProcessMemoryUtilization()
	if err != nil {
		// If we can't get metrics, return empty metrics for each device
		for _, device := range devices {
			result[device.UUID] = &api.GPUUsageMetrics{
				DeviceUUID: device.UUID,
			}
		}
		return result, nil
	}

	// Aggregate memory usage per device
	deviceMemoryUsed := make(map[string]uint64)
	for _, memUtil := range memUtils {
		deviceMemoryUsed[memUtil.DeviceUUID] += memUtil.UsedBytes
	}

	// Get compute utilization
	computeUtils, err := m.accelerator.GetProcessComputeUtilization()
	if err != nil {
		// Continue with memory metrics only
		computeUtils = []api.ComputeUtilization{}
	}

	// Aggregate compute usage per device
	deviceComputePercent := make(map[string]float64)
	deviceComputeTflops := make(map[string]float64)
	for _, computeUtil := range computeUtils {
		deviceComputePercent[computeUtil.DeviceUUID] += computeUtil.UtilizationPercent
		deviceComputeTflops[computeUtil.DeviceUUID] += computeUtil.TFLOPsUsed
	}

	// Build metrics for each device
	for _, device := range devices {
		memoryUsed := deviceMemoryUsed[device.UUID]
		memoryPercent := 0.0
		if device.Bytes > 0 {
			memoryPercent = float64(memoryUsed) / float64(device.Bytes) * 100.0
		}

		result[device.UUID] = &api.GPUUsageMetrics{
			DeviceUUID:        device.UUID,
			MemoryBytes:       memoryUsed,
			MemoryPercentage:  memoryPercent,
			ComputePercentage: deviceComputePercent[device.UUID],
			ComputeTflops:     deviceComputeTflops[device.UUID],
		}
	}

	return result, nil
}

// GetProcessComputeUtilization exposes accelerator interface method
func (m *Controller) GetProcessComputeUtilization() ([]api.ComputeUtilization, error) {
	return m.accelerator.GetProcessComputeUtilization()
}

// GetProcessMemoryUtilization exposes accelerator interface method
func (m *Controller) GetProcessMemoryUtilization() ([]api.MemoryUtilization, error) {
	return m.accelerator.GetProcessMemoryUtilization()
}

// Close closes the device controller and unloads the accelerator library
func (m *Controller) Close() error {
	return m.accelerator.Close()
}
