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
	"time"
)

// IsolationMode represents the isolation mode for GPU resources
type IsolationMode string

const (
	IsolationModeShared      IsolationMode = "shared"      // Timeslicing, no resource control
	IsolationModeSoft        IsolationMode = "soft"        // Hook-based, token-based limiting
	IsolationModeHard        IsolationMode = "hard"        // One-time resource limits
	IsolationModePartitioned IsolationMode = "partitioned" // Hardware/driver-level partitioning (MIG)
)

// DeviceInfo represents discovered GPU device information
type DeviceInfo struct {
	UUID            string
	Vendor          string
	Model           string
	Index           int32
	NUMANode        int32
	TotalMemory     uint64 // bytes
	TotalCompute    uint64 // compute units
	MaxTflops       float64
	PCIEGen         uint32
	PCIEWidth       uint32
	DriverVersion   string
	FirmwareVersion string
	Capabilities    DeviceCapabilities
	Properties      DeviceProperties
}

// DeviceCapabilities represents device capabilities
type DeviceCapabilities struct {
	SupportsPartitioning  bool
	SupportsSoftIsolation bool
	SupportsHardIsolation bool
	SupportsSnapshot      bool
	SupportsMetrics       bool
	MaxPartitions         uint32
	MaxWorkersPerDevice   uint32
}

// DeviceProperties represents device properties
type DeviceProperties struct {
	ClockGraphics          uint32
	ClockSM                uint32
	ClockMem               uint32
	ClockAI                uint32
	PowerLimit             uint32
	TemperatureThreshold   uint32
	ECCEnabled             bool
	PersistenceModeEnabled bool
	ComputeCapability      string
	ChipType               string
}

// PartitionTemplate represents a hardware partition template
type PartitionTemplate struct {
	TemplateID   string
	Name         string
	MemoryBytes  uint64
	ComputeUnits uint64
	Tflops       float64
	SliceCount   uint32
	IsDefault    bool
	Description  string
}

// DeviceAllocation represents an allocated device for a pod
type DeviceAllocation struct {
	DeviceUUID    string
	PodUID        string
	PodName       string
	Namespace     string
	IsolationMode IsolationMode
	PartitionUUID string // For partitioned mode
	TemplateID    string // For partitioned mode
	MemoryLimit   uint64 // For hard isolation
	ComputeLimit  uint32 // For hard isolation (percentage)
	WorkerID      string
	AllocatedAt   time.Time
}

// DevicePool represents a pool of devices with configuration
type DevicePool struct {
	Name           string
	Vendor         string // "NVIDIA", "Ascend", etc.
	IsolationMode  IsolationMode
	DeviceUUIDs    []string
	AcceleratorLib string // Path to accelerator.so library
}

// AllocateRequest represents a request to allocate devices
type AllocateRequest struct {
	PodUID        string
	PodName       string
	Namespace     string
	PoolName      string
	DeviceCount   int
	IsolationMode IsolationMode
	MemoryBytes   uint64
	ComputeUnits  uint32
	TemplateID    string // For partitioned mode
}

// AllocateResponse represents the response from device allocation
type AllocateResponse struct {
	Allocations []DeviceAllocation
	Success     bool
	Error       string
}

// ComputeUtilization represents compute utilization for a process on a device
type ComputeUtilization struct {
	ProcessID         string
	DeviceUUID        string
	UtilizationPercent float64
	ActiveSMs         uint64
	TotalSMs          uint64
	TflopsUsed        float64
}

// MemoryUtilization represents memory utilization for a process on a device
type MemoryUtilization struct {
	ProcessID         string
	DeviceUUID        string
	UsedBytes         uint64
	ReservedBytes     uint64
	UtilizationPercent float64
}
