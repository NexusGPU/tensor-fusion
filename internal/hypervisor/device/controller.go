package device

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/metrics"
	"github.com/samber/lo"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/klog/v2"
)

var tmpDir = os.TempDir()

// Controller manages GPU device discovery and lifecycle
type Controller struct {
	ctx context.Context
	mu  sync.RWMutex

	devices map[string]*api.DeviceInfo // key: device UUID

	accelerator          *AcceleratorInterface
	acceleratorVendor    string
	discoveryInterval    time.Duration
	deviceUpdateHandlers []framework.DeviceChangeHandler
	isolationMode        api.IsolationMode

	// allocationController is set after creation to provide allocation data for telemetry
	allocationController framework.WorkerAllocationController
}

var _ framework.DeviceController = &Controller{}

// NewController creates a new device manager
func NewController(ctx context.Context, acceleratorLibPath string, acceleratorVendor string, discoveryInterval time.Duration, isolationMode string) (*Controller, error) {
	accel, err := NewAcceleratorInterface(acceleratorLibPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create accelerator interface: %w", err)
	}
	return &Controller{
		ctx:                  ctx,
		devices:              make(map[string]*api.DeviceInfo),
		accelerator:          accel,
		acceleratorVendor:    acceleratorVendor,
		discoveryInterval:    discoveryInterval,
		deviceUpdateHandlers: make([]framework.DeviceChangeHandler, 2),
		isolationMode:        api.IsolationMode(isolationMode),
	}, nil
}

// SetAllocationController sets the allocation controller for telemetry purposes
func (m *Controller) SetAllocationController(allocationController framework.WorkerAllocationController) {
	m.allocationController = allocationController
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
	klog.Infof("Start discovering devices using provider lib")
	devices, err := m.accelerator.GetAllDevices()
	if err != nil {
		return fmt.Errorf("failed to get all devices: %w", err)
	}

	// Build a map of newly fetched devices by UUID
	newDevicesMap := make(map[string]*api.DeviceInfo, len(devices))
	for _, device := range devices {
		// Convert UUID to lowercase for case-insensitive comparison
		// Kubernetes resource name has to be lowercase
		device.UUID = strings.ToLower(device.UUID)
		device.IsolationMode = m.isolationMode
		newDevicesMap[device.UUID] = device
	}

	// Diff logic: compare new devices with existing devices (K8s reconcile pattern)
	// First, identify all changes without modifying state
	var addedDevices []*api.DeviceInfo
	var removedDevices []*api.DeviceInfo
	var updatedDevices []struct {
		old *api.DeviceInfo
		new *api.DeviceInfo
	}

	// Find added devices (in new but not in old)
	for uuid, newDevice := range newDevicesMap {
		if _, exists := m.devices[uuid]; !exists {
			addedDevices = append(addedDevices, newDevice)
		}
	}

	// Find removed devices (in old but not in new)
	for uuid, oldDevice := range m.devices {
		if _, exists := newDevicesMap[uuid]; !exists {
			removedDevices = append(removedDevices, oldDevice)
		}
	}

	// Find updated devices (in both but changed)
	for uuid, newDevice := range newDevicesMap {
		if oldDevice, exists := m.devices[uuid]; exists {
			// Check if device has changed
			if !equality.Semantic.DeepEqual(oldDevice, newDevice) {
				updatedDevices = append(updatedDevices, struct {
					old *api.DeviceInfo
					new *api.DeviceInfo
				}{old: oldDevice, new: newDevice})
			}
		}
	}

	// Notify handlers for all changes (similar to K8s reconcile)
	for _, device := range addedDevices {
		m.notifyHandlers(func(handler framework.DeviceChangeHandler) {
			if handler.OnAdd != nil {
				handler.OnAdd(device)
			}
		})
		klog.V(4).Infof("Device added: %s (UUID: %s)", device.Model, device.UUID)
	}

	for _, device := range removedDevices {
		m.notifyHandlers(func(handler framework.DeviceChangeHandler) {
			if handler.OnRemove != nil {
				handler.OnRemove(device)
			}
		})
		klog.V(4).Infof("Device removed: %s (UUID: %s)", device.Model, device.UUID)
	}

	for _, update := range updatedDevices {
		m.notifyHandlers(func(handler framework.DeviceChangeHandler) {
			if handler.OnUpdate != nil {
				handler.OnUpdate(update.old, update.new)
			}
		})
		klog.V(4).Infof("Device updated: %s (UUID: %s)", update.new.Model, update.new.UUID)
	}

	// Update state after notifying handlers
	for _, device := range addedDevices {
		m.devices[device.UUID] = device
	}
	for _, device := range removedDevices {
		delete(m.devices, device.UUID)
	}
	for _, update := range updatedDevices {
		m.devices[update.new.UUID] = update.new
	}

	nodeInfo := m.AggregateNodeInfo()

	if metrics.ShouldSendTelemetry() {
		sampleGPUModel := ""
		if len(m.devices) > 0 {
			for _, device := range m.devices {
				if device.Model != "" {
					sampleGPUModel = device.Model
					break
				}
			}
		}
		workersCount := 0
		if m.allocationController != nil {
			for _, allocations := range m.allocationController.GetDeviceAllocations() {
				workersCount += len(allocations)
			}
		}

		go metrics.SendAnonymousTelemetry(
			nodeInfo, m.acceleratorVendor, sampleGPUModel, workersCount, m.isolationMode,
		)
	}
	m.notifyHandlers(func(handler framework.DeviceChangeHandler) {
		if handler.OnDiscoveryComplete != nil {
			handler.OnDiscoveryComplete(nodeInfo)
		}
	})
	return nil
}

// notifyHandlers calls the provided function for each registered handler
func (m *Controller) notifyHandlers(fn func(framework.DeviceChangeHandler)) {
	for _, handler := range m.deviceUpdateHandlers {
		fn(handler)
	}
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

// GetProcessInformation implements framework.DeviceController
// Returns process-level GPU metrics for all processes on all devices
func (m *Controller) GetProcessInformation() ([]api.ProcessInformation, error) {
	return m.accelerator.GetProcessInformation()
}

func (m *Controller) GetVendorMountLibs() ([]*api.Mount, error) {
	return m.accelerator.GetVendorMountLibs()
}

func (m *Controller) SplitDevice(deviceUUID string, partitionTemplateID string) (*api.DeviceInfo, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	existingDevice, exists := m.devices[deviceUUID]
	if !exists {
		return nil, fmt.Errorf("device %s not found, can not partition", deviceUUID)
	}
	if existingDevice == nil {
		return nil, fmt.Errorf("device %s is nil, can not partition", deviceUUID)
	}
	newPartitionedDevice := *existingDevice
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

func (m *Controller) RegisterDeviceUpdateHandler(handler framework.DeviceChangeHandler) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.deviceUpdateHandlers = append(m.deviceUpdateHandlers, handler)

	// Notify the newly registered handler about existing devices without triggering a new discovery
	// This ensures handlers get notified of devices that were discovered before they were registered
	if len(m.devices) > 0 {
		for _, device := range m.devices {
			if handler.OnAdd != nil {
				handler.OnAdd(device)
			}
		}
		// Also notify about discovery completion if there are existing devices
		if handler.OnDiscoveryComplete != nil {
			nodeInfo := m.AggregateNodeInfo()
			handler.OnDiscoveryComplete(nodeInfo)
		}
	}
}

func (m *Controller) GetAcceleratorVendor() string {
	return m.acceleratorVendor
}

func (m *Controller) AggregateNodeInfo() *api.NodeInfo {
	info := &api.NodeInfo{
		RAMSizeBytes:  GetTotalHostRAMBytes(),
		DataDiskBytes: GetDiskInfo(tmpDir),
	}
	for _, device := range m.devices {
		info.TotalTFlops += device.MaxTflops
		info.TotalVRAMBytes += int64(device.TotalMemoryBytes)
		info.DeviceIDs = append(info.DeviceIDs, device.UUID)
	}
	return info
}
