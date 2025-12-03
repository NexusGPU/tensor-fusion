package device

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/samber/lo"
	"k8s.io/klog/v2"
)

// Controller manages GPU device discovery, allocation, and lifecycle
type Controller struct {
	ctx               context.Context
	mu                sync.RWMutex
	devices           map[string]*api.DeviceInfo // key: device UUID
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

	// TODO: check health status of device, handle not existing device and not existing partitions
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

// Start implements framework.DeviceController
func (m *Controller) Start() error {
	// Start device discovery
	return m.StartDiscoverDevices()
}

func (m *Controller) Stop() error {
	return m.accelerator.Close()
}

// DiscoverDevices implements framework.DeviceController
func (m *Controller) DiscoverDevices() error {
	return m.discoverDevices()
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
func (m *Controller) GetDevice(deviceUUID string) (*api.DeviceInfo, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	device, exists := m.devices[deviceUUID]
	return device, exists
}

// GetDeviceMetrics implements framework.DeviceController
func (m *Controller) GetDeviceMetrics() (map[string]*api.GPUUsageMetrics, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string]*api.GPUUsageMetrics, len(m.devices))
	metrics, err := m.accelerator.GetDeviceMetrics(lo.Keys(m.devices))
	if err != nil {
		return nil, fmt.Errorf("failed to get device metrics: %w", err)
	}
	for _, metric := range metrics {
		result[metric.DeviceUUID] = metric
	}
	return result, nil
}

func (m *Controller) GetVendorMountLibs() ([]*api.Mount, error) {
	return m.accelerator.GetVendorMountLibs()
}

func (m *Controller) SplitDevice(partitionTemplateID string, deviceUUID string) (*api.DeviceInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	existingDevice, exists := m.devices[deviceUUID]
	newPartitionedDevice := *existingDevice
	if !exists {
		return nil, fmt.Errorf("device %s not found, can not partition", deviceUUID)
	}
	partitionUUID, err := m.accelerator.AssignPartition(partitionTemplateID, deviceUUID)
	if err != nil {
		return nil, err
	}
	newPartitionedDevice.ParentUUID = newPartitionedDevice.UUID
	newPartitionedDevice.UUID = partitionUUID
	m.devices[partitionUUID] = &newPartitionedDevice
	return &newPartitionedDevice, nil
}

func (m *Controller) RemovePartitionedDevice(partitionUUID, deviceUUID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	_, exists := m.devices[partitionUUID]
	if !exists {
		return fmt.Errorf("partition %s not found, can not remove", partitionUUID)
	}

	err := m.accelerator.RemovePartition(partitionUUID, deviceUUID)
	if err != nil {
		return err
	}
	klog.Infof("removed partition %s from device %s", partitionUUID, deviceUUID)
	delete(m.devices, partitionUUID)
	return nil
}
