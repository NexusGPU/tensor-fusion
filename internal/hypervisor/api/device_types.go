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

package api

// DeviceInfo represents discovered GPU device information
// +k8s:deepcopy-gen=true
type DeviceInfo struct {
	UUID                       string
	Vendor                     string
	Model                      string
	Index                      int32
	NUMANode                   int32
	TotalMemoryBytes           uint64
	MaxTflops                  float64
	VirtualizationCapabilities VirtualizationCapabilities
	Properties                 map[string]string
	Healthy                    bool

	ParentUUID string

	// Host - Guest device node mapping, eg /dev/nvidia0 -> /dev/nvidia0
	// When multiple device allocated, deduplicated by device node
	DeviceNode map[string]string

	// Env to inject to guest
	DeviceEnv map[string]string
}

type NodeInfo struct {
	// Extra metadata for centralized management
	RAMSizeBytes  int64
	DataDiskBytes int64

	// Aggregated info of whole Node
	TotalTFlops    float64
	TotalVRAMBytes int64
	DeviceIDs      []string

	// TODO: discover and merge extra devices and topology info like:
	// Nvlink/IB NICs, etc.
	// CXL available or not, PCIe generation etc.
}

// VirtualizationCapabilities represents virtualization capabilities
// +k8s:deepcopy-gen=true
type VirtualizationCapabilities struct {
	SupportsPartitioning  bool
	SupportsSoftIsolation bool
	SupportsHardIsolation bool
	SupportsSnapshot      bool
	SupportsMetrics       bool
	MaxPartitions         uint32
	MaxWorkersPerDevice   uint32
}

// ComputeUtilization represents compute utilization for a process on a device
type ComputeUtilization struct {
	ProcessID          string
	DeviceUUID         string
	UtilizationPercent float64
}

// MemoryUtilization represents memory utilization for a process on a device
type MemoryUtilization struct {
	ProcessID     string
	DeviceUUID    string
	UsedBytes     uint64
	ReservedBytes uint64
}

// GPUUsageMetrics represents GPU device metrics
// +k8s:deepcopy-gen=true
type GPUUsageMetrics struct {
	DeviceUUID        string
	MemoryBytes       uint64
	MemoryPercentage  float64
	ComputePercentage float64
	ComputeTflops     float64
	Rx                float64 // PCIe RX in KB
	Tx                float64 // PCIe TX in KB
	Temperature       float64
	PowerUsage        int64 // in watts
	ExtraMetrics      map[string]float64
}

// WorkerMetrics represents worker process metrics on a device
// +k8s:deepcopy-gen=true
type WorkerMetrics struct {
	DeviceUUID        string
	WorkerUID         string
	ProcessID         string
	MemoryBytes       uint64
	MemoryPercentage  float64
	ComputeTflops     float64
	ComputePercentage float64
	ExtraMetrics      map[string]float64
}
