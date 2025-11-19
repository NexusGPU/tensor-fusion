package device

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"k8s.io/klog/v2"
)

// Controller manages GPU device discovery, allocation, and lifecycle
type Controller struct {
	ctx               context.Context
	mu                sync.RWMutex
	devices           map[string]*api.DeviceInfo       // key: device UUID
	allocations       map[string]*api.DeviceAllocation // key: worker UID
	deviceToAlloc     map[string][]string              // device UUID -> []worker UID
	accelerator       *AcceleratorInterface
	discoveryInterval time.Duration
}

// NewController creates a new device manager
func NewController(ctx context.Context, acceleratorLibPath string, discoveryInterval time.Duration) (framework.DeviceController, error) {
	accel, err := NewAcceleratorInterface(acceleratorLibPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create accelerator interface: %w", err)
	}

	return &Controller{
		ctx:               ctx,
		devices:           make(map[string]*api.DeviceInfo),
		allocations:       make(map[string]*api.DeviceAllocation),
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

// Allocate allocates devices for a pod request
func (m *Controller) Allocate(req *api.DeviceAllocateRequest) (*api.DeviceAllocateResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return &api.DeviceAllocateResponse{
		DeviceNodes: req.DeviceUUIDs,
		Annotations: make(map[string]string),
		Mounts:      make(map[string]string),
		EnvVars:     make(map[string]string),
		Success:     true,
	}, nil
}

// Deallocate de-allocates devices for a pod
func (m *Controller) Deallocate(podUID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	allocation, exists := m.allocations[podUID]
	if !exists {
		return fmt.Errorf("allocation not found for pod %s", podUID)
	}

	// Handle partitioned mode cleanup
	if allocation.IsolationMode == api.IsolationModePartitioned && allocation.TemplateID != "" {
		if err := m.accelerator.RemovePartition(allocation.TemplateID, allocation.DeviceUUID); err != nil {
			// Log error but continue
			klog.Errorf("failed to remove partition: %v", err)
		}
	}

	// Remove from allocations
	delete(m.allocations, podUID)

	// Remove from device mapping
	if podUIDs, exists := m.deviceToAlloc[allocation.DeviceUUID]; exists {
		for i, uid := range podUIDs {
			if uid == podUID {
				m.deviceToAlloc[allocation.DeviceUUID] = append(podUIDs[:i], podUIDs[i+1:]...)
				break
			}
		}
	}

	return nil
}

// GetAllocation returns allocation for a pod
func (m *Controller) GetAllocation(workerUID string) (*api.DeviceAllocation, bool) {
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
func (m *Controller) AllocateDevice(request *api.DeviceAllocateRequest) (*api.DeviceAllocateResponse, error) {
	return m.Allocate(request)
}

// ListDevices implements framework.DeviceController
func (m *Controller) ListDevices(ctx context.Context) ([]*api.DeviceInfo, error) {
	return m.GetDevices(), nil
}

// DevicesUpdates implements framework.DeviceController
func (m *Controller) DevicesUpdates(ctx context.Context) (<-chan []*api.DeviceInfo, error) {
	ch := make(chan []*api.DeviceInfo)
	// TODO: Implement proper device updates channel
	return ch, nil
}

// GetDevice implements framework.DeviceController
func (m *Controller) GetDevice(ctx context.Context, deviceUUID string) (*api.DeviceInfo, error) {
	device, exists := m.getDevice(deviceUUID)
	if !exists {
		return nil, fmt.Errorf("device not found: %s", deviceUUID)
	}
	return device, nil
}

// GetDeviceAllocations implements framework.DeviceController
func (m *Controller) GetDeviceAllocations(ctx context.Context, deviceUUID string) ([]*api.DeviceAllocation, error) {
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
func (m *Controller) GetDeviceAllocationUpdates(ctx context.Context, deviceUUID string, allocationID string) (<-chan []*api.DeviceAllocation, error) {
	ch := make(chan []*api.DeviceAllocation)
	// TODO: Implement proper allocation updates channel
	return ch, nil
}

// GetGPUMetrics implements framework.DeviceController
func (m *Controller) GetGPUMetrics(ctx context.Context) (map[string]*api.GPUUsageMetrics, error) {
	m.mu.RLock()
	devices := make([]*api.DeviceInfo, 0, len(m.devices))
	for _, device := range m.devices {
		devices = append(devices, device)
	}
	m.mu.RUnlock()

	// TODO: Get actual GPU metrics from accelerator interface
	// For now, return empty metrics
	result := make(map[string]*api.GPUUsageMetrics)
	for _, device := range devices {
		result[device.UUID] = &api.GPUUsageMetrics{
			DeviceUUID: device.UUID,
			// TODO: Populate with actual metrics from accelerator
		}
	}
	return result, nil
}
