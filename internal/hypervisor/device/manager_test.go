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
	"testing"
	"time"
)

func TestDeviceManager_Discovery(t *testing.T) {
	// Build accelerator library first
	// In real scenario, this would be done by Makefile
	mgr, err := NewManager("../../../provider/build/libaccelerator_stub.so", 5*time.Second)
	if err != nil {
		t.Skipf("Skipping test: failed to create manager (accelerator lib may not be built): %v", err)
		return
	}

	if err := mgr.Start(); err != nil {
		t.Fatalf("Failed to start manager: %v", err)
	}
	defer mgr.Stop()

	// Wait a bit for discovery
	time.Sleep(100 * time.Millisecond)

	devices := mgr.GetDevices()
	if len(devices) == 0 {
		t.Error("Expected at least one device, got 0")
		return
	}

	// Verify device properties
	device := devices[0]
	if device.UUID == "" {
		t.Error("Device UUID should not be empty")
	}
	if device.Vendor == "" {
		t.Error("Device vendor should not be empty")
	}
	if device.TotalMemory == 0 {
		t.Error("Device total memory should be > 0")
	}
}

func TestDeviceManager_Allocate_Shared(t *testing.T) {
	mgr, err := NewManager("../../../provider/build/libaccelerator_stub.so", 5*time.Second)
	if err != nil {
		t.Skipf("Skipping test: failed to create manager: %v", err)
		return
	}

	if err := mgr.Start(); err != nil {
		t.Fatalf("Failed to start manager: %v", err)
	}
	defer mgr.Stop()

	time.Sleep(100 * time.Millisecond)

	devices := mgr.GetDevices()
	if len(devices) == 0 {
		t.Skip("No devices available for testing")
		return
	}

	// Register a pool
	pool := &DevicePool{
		Name:          "test-pool",
		Vendor:        "STUB",
		IsolationMode: IsolationModeShared,
		DeviceUUIDs:   []string{devices[0].UUID},
	}
	if err := mgr.RegisterPool(pool); err != nil {
		t.Fatalf("Failed to register pool: %v", err)
	}

	// Allocate device
	req := &AllocateRequest{
		PodUID:        "test-pod-1",
		PodName:       "test-pod",
		Namespace:     "default",
		PoolName:      "test-pool",
		DeviceCount:   1,
		IsolationMode: IsolationModeShared,
	}

	resp, err := mgr.Allocate(req)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	if !resp.Success {
		t.Fatalf("Allocation failed: %s", resp.Error)
	}

	if len(resp.Allocations) != 1 {
		t.Fatalf("Expected 1 allocation, got %d", len(resp.Allocations))
	}

	allocation := resp.Allocations[0]
	if allocation.DeviceUUID != devices[0].UUID {
		t.Errorf("Expected device UUID %s, got %s", devices[0].UUID, allocation.DeviceUUID)
	}
	if allocation.IsolationMode != IsolationModeShared {
		t.Errorf("Expected isolation mode %s, got %s", IsolationModeShared, allocation.IsolationMode)
	}

	// Deallocate
	if err := mgr.Deallocate("test-pod-1"); err != nil {
		t.Fatalf("Failed to deallocate: %v", err)
	}
}

func TestDeviceManager_Allocate_Hard(t *testing.T) {
	mgr, err := NewManager("../../../provider/build/libaccelerator_stub.so", 5*time.Second)
	if err != nil {
		t.Skipf("Skipping test: failed to create manager: %v", err)
		return
	}

	if err := mgr.Start(); err != nil {
		t.Fatalf("Failed to start manager: %v", err)
	}
	defer mgr.Stop()

	time.Sleep(100 * time.Millisecond)

	devices := mgr.GetDevices()
	if len(devices) == 0 {
		t.Skip("No devices available for testing")
		return
	}

	// Register a pool
	pool := &DevicePool{
		Name:          "test-pool-hard",
		Vendor:        "STUB",
		IsolationMode: IsolationModeHard,
		DeviceUUIDs:   []string{devices[0].UUID},
	}
	if err := mgr.RegisterPool(pool); err != nil {
		t.Fatalf("Failed to register pool: %v", err)
	}

	// Allocate device with hard limits
	req := &AllocateRequest{
		PodUID:        "test-pod-hard",
		PodName:       "test-pod",
		Namespace:     "default",
		PoolName:      "test-pool-hard",
		DeviceCount:   1,
		IsolationMode: IsolationModeHard,
		MemoryBytes:   4 * 1024 * 1024 * 1024, // 4GB
		ComputeUnits:  50,                     // 50%
	}

	resp, err := mgr.Allocate(req)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	if !resp.Success {
		t.Fatalf("Allocation failed: %s", resp.Error)
	}

	allocation := resp.Allocations[0]
	if allocation.MemoryLimit != req.MemoryBytes {
		t.Errorf("Expected memory limit %d, got %d", req.MemoryBytes, allocation.MemoryLimit)
	}
	if allocation.ComputeLimit != req.ComputeUnits {
		t.Errorf("Expected compute limit %d, got %d", req.ComputeUnits, allocation.ComputeLimit)
	}

	// Deallocate
	if err := mgr.Deallocate("test-pod-hard"); err != nil {
		t.Fatalf("Failed to deallocate: %v", err)
	}
}

func TestDeviceManager_Allocate_Partitioned(t *testing.T) {
	mgr, err := NewManager("../../../provider/build/libaccelerator_stub.so", 5*time.Second)
	if err != nil {
		t.Skipf("Skipping test: failed to create manager: %v", err)
		return
	}

	if err := mgr.Start(); err != nil {
		t.Fatalf("Failed to start manager: %v", err)
	}
	defer mgr.Stop()

	time.Sleep(100 * time.Millisecond)

	devices := mgr.GetDevices()
	if len(devices) == 0 {
		t.Skip("No devices available for testing")
		return
	}

	// Get partition templates
	templates, err := mgr.accelerator.GetPartitionTemplates(0)
	if err != nil {
		t.Skipf("Skipping test: failed to get partition templates: %v", err)
		return
	}

	if len(templates) == 0 {
		t.Skip("No partition templates available (device may not support partitioning)")
		return
	}

	// Register a pool
	pool := &DevicePool{
		Name:          "test-pool-partitioned",
		Vendor:        "STUB",
		IsolationMode: IsolationModePartitioned,
		DeviceUUIDs:   []string{devices[0].UUID},
	}
	if err := mgr.RegisterPool(pool); err != nil {
		t.Fatalf("Failed to register pool: %v", err)
	}

	// Allocate device with partition
	req := &AllocateRequest{
		PodUID:        "test-pod-partitioned",
		PodName:       "test-pod",
		Namespace:     "default",
		PoolName:      "test-pool-partitioned",
		DeviceCount:   1,
		IsolationMode: IsolationModePartitioned,
		TemplateID:    templates[0].TemplateID,
	}

	resp, err := mgr.Allocate(req)
	if err != nil {
		t.Fatalf("Failed to allocate: %v", err)
	}

	if !resp.Success {
		t.Fatalf("Allocation failed: %s", resp.Error)
	}

	allocation := resp.Allocations[0]
	if allocation.PartitionUUID == "" {
		t.Error("Partition UUID should not be empty")
	}
	if allocation.TemplateID != templates[0].TemplateID {
		t.Errorf("Expected template ID %s, got %s", templates[0].TemplateID, allocation.TemplateID)
	}

	// Deallocate
	if err := mgr.Deallocate("test-pod-partitioned"); err != nil {
		t.Fatalf("Failed to deallocate: %v", err)
	}
}
