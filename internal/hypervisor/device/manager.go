/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package device

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/klog/v2"
)

// Manager manages GPU device discovery, allocation, and lifecycle
type Manager struct {
	mu                sync.RWMutex
	devices           map[string]*DeviceInfo       // key: device UUID
	allocations       map[string]*DeviceAllocation // key: pod UID
	deviceToAlloc     map[string][]string          // device UUID -> []pod UID
	pools             map[string]*DevicePool
	accelerator       *AcceleratorInterface
	discoveryInterval time.Duration
	stopCh            chan struct{}
}

// NewManager creates a new device manager
func NewManager(acceleratorLibPath string, discoveryInterval time.Duration) (*Manager, error) {
	accel := NewAcceleratorInterface(acceleratorLibPath)

	mgr := &Manager{
		devices:           make(map[string]*DeviceInfo),
		allocations:       make(map[string]*DeviceAllocation),
		deviceToAlloc:     make(map[string][]string),
		pools:             make(map[string]*DevicePool),
		accelerator:       accel,
		discoveryInterval: discoveryInterval,
		stopCh:            make(chan struct{}),
	}

	return mgr, nil
}

// Start starts the device manager (device discovery, etc.)
func (m *Manager) Start() error {
	// Initial device discovery
	if err := m.discoverDevices(); err != nil {
		return fmt.Errorf("initial device discovery failed: %w", err)
	}

	// Start periodic discovery
	go m.periodicDiscovery()

	return nil
}

// Stop stops the device manager
func (m *Manager) Stop() {
	close(m.stopCh)
}

// discoverDevices discovers all available GPU devices
func (m *Manager) discoverDevices() error {
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
func (m *Manager) periodicDiscovery() {
	ticker := time.NewTicker(m.discoveryInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopCh:
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
func (m *Manager) GetDevices() []*DeviceInfo {
	m.mu.RLock()
	defer m.mu.RUnlock()

	devices := make([]*DeviceInfo, 0, len(m.devices))
	for _, device := range m.devices {
		devices = append(devices, device)
	}
	return devices
}

// GetDevice returns a device by UUID
func (m *Manager) GetDevice(uuid string) (*DeviceInfo, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	device, exists := m.devices[uuid]
	return device, exists
}

// RegisterPool registers a device pool
func (m *Manager) RegisterPool(pool *DevicePool) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Validate pool devices exist
	for _, uuid := range pool.DeviceUUIDs {
		if _, exists := m.devices[uuid]; !exists {
			return fmt.Errorf("device %s not found", uuid)
		}
	}

	m.pools[pool.Name] = pool
	return nil
}

// Allocate allocates devices for a pod request
func (m *Manager) Allocate(req *AllocateRequest) (*AllocateResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Get pool
	pool, exists := m.pools[req.PoolName]
	if !exists {
		return &AllocateResponse{
			Success: false,
			Error:   fmt.Sprintf("pool %s not found", req.PoolName),
		}, nil
	}

	// Find available devices in pool
	availableDevices := m.findAvailableDevices(pool, req.DeviceCount)
	if len(availableDevices) < req.DeviceCount {
		return &AllocateResponse{
			Success: false,
			Error:   fmt.Sprintf("not enough available devices: need %d, found %d", req.DeviceCount, len(availableDevices)),
		}, nil
	}

	// Allocate devices
	allocations := make([]DeviceAllocation, 0, req.DeviceCount)
	for i := 0; i < req.DeviceCount; i++ {
		device := availableDevices[i]
		allocation := &DeviceAllocation{
			DeviceUUID:    device.UUID,
			PodUID:        req.PodUID,
			PodName:       req.PodName,
			Namespace:     req.Namespace,
			IsolationMode: req.IsolationMode,
			WorkerID:      fmt.Sprintf("%s-%s-%d", req.PodUID, device.UUID, i),
			AllocatedAt:   time.Now(),
		}

		// Handle different isolation modes
		switch req.IsolationMode {
		case IsolationModePartitioned:
			if req.TemplateID == "" {
				return &AllocateResponse{
					Success: false,
					Error:   "templateID required for partitioned mode",
				}, nil
			}
			partitionUUID, _, err := m.accelerator.AssignPartition(req.TemplateID, device.UUID)
			if err != nil {
				return &AllocateResponse{
					Success: false,
					Error:   fmt.Sprintf("failed to assign partition: %v", err),
				}, nil
			}
			allocation.PartitionUUID = partitionUUID
			allocation.TemplateID = req.TemplateID
			// Note: partition overhead could be used to adjust available memory

		case IsolationModeHard:
			if req.MemoryBytes > 0 {
				if err := m.accelerator.SetMemHardLimit(allocation.WorkerID, device.UUID, req.MemoryBytes); err != nil {
					return &AllocateResponse{
						Success: false,
						Error:   fmt.Sprintf("failed to set memory limit: %v", err),
					}, nil
				}
				allocation.MemoryLimit = req.MemoryBytes
			}
			if req.ComputeUnits > 0 {
				if err := m.accelerator.SetComputeUnitHardLimit(allocation.WorkerID, device.UUID, req.ComputeUnits); err != nil {
					return &AllocateResponse{
						Success: false,
						Error:   fmt.Sprintf("failed to set compute limit: %v", err),
					}, nil
				}
				allocation.ComputeLimit = req.ComputeUnits
			}

		case IsolationModeSoft, IsolationModeShared:
			// No immediate action needed, handled by limiter.so at runtime
		}

		allocations = append(allocations, *allocation)
		m.allocations[req.PodUID] = allocation
		if m.deviceToAlloc[device.UUID] == nil {
			m.deviceToAlloc[device.UUID] = make([]string, 0)
		}
		m.deviceToAlloc[device.UUID] = append(m.deviceToAlloc[device.UUID], req.PodUID)
	}

	return &AllocateResponse{
		Allocations: allocations,
		Success:     true,
	}, nil
}

// Deallocate deallocates devices for a pod
func (m *Manager) Deallocate(podUID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	allocation, exists := m.allocations[podUID]
	if !exists {
		return fmt.Errorf("allocation not found for pod %s", podUID)
	}

	// Handle partitioned mode cleanup
	if allocation.IsolationMode == IsolationModePartitioned && allocation.TemplateID != "" {
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

// findAvailableDevices finds available devices in a pool
func (m *Manager) findAvailableDevices(pool *DevicePool, count int) []*DeviceInfo {
	available := make([]*DeviceInfo, 0)

	for _, uuid := range pool.DeviceUUIDs {
		device, exists := m.devices[uuid]
		if !exists {
			continue
		}

		// Check if device has capacity (simple check: not too many allocations)
		allocCount := len(m.deviceToAlloc[uuid])
		if uint32(allocCount) < device.Capabilities.MaxWorkersPerDevice {
			available = append(available, device)
			if len(available) >= count {
				break
			}
		}
	}

	return available
}

// GetAllocation returns allocation for a pod
func (m *Manager) GetAllocation(podUID string) (*DeviceAllocation, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	allocation, exists := m.allocations[podUID]
	return allocation, exists
}
