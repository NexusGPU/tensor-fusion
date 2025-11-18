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
	allocations       map[string]*DeviceAllocation // key: worker UID
	deviceToAlloc     map[string][]string          // device UUID -> []worker UID
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

	// TODO new framework

	// TODO new backend
	// TODO start backend

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

// Allocate allocates devices for a pod request
func (m *Manager) Allocate(req *DeviceAllocateRequest) (*DeviceAllocateResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return &DeviceAllocateResponse{
		DeviceNodes: req.DeviceUUIDs,
		Annotations: make(map[string]string),
		Mounts:      make(map[string]string),
		EnvVars:     make(map[string]string),
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

// GetAllocation returns allocation for a pod
func (m *Manager) GetAllocation(workerUID string) (*DeviceAllocation, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	allocation, exists := m.allocations[workerUID]
	return allocation, exists
}
