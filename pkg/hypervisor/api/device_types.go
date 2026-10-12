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

import (
	"fmt"
	"math"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
)

// TflopsCapacitySource says where a device's TflopsCapacity comes from. The
// values are the ones published in constants.GPUTflopsCapacitySourceAnnotation.
type TflopsCapacitySource string

const (
	// TflopsCapacitySourceProviderConfig: the vendor-published FP16 figure of the
	// device model, from the ProviderConfig hardware table.
	TflopsCapacitySourceProviderConfig TflopsCapacitySource = constants.GPUTflopsCapacitySourceProviderConfig
	// TflopsCapacitySourceProviderEstimate: no table entry exists for the device
	// model, so the capacity is the provider library's own estimate. That
	// estimate is not in FP16 units (see DeviceInfo.ProviderEstimatedTflops).
	TflopsCapacitySourceProviderEstimate TflopsCapacitySource = constants.GPUTflopsCapacitySourceEstimate
)

// DeviceInfo represents discovered GPU device information
// +k8s:deepcopy-gen=true
type DeviceInfo struct {
	UUID             string
	Vendor           string
	Model            string
	Index            int32
	NUMANode         int32
	Topology         *DeviceTopology
	TotalMemoryBytes uint64

	// TflopsCapacity is the compute capacity of the device in TensorFusion
	// TFLOPS: the unit of resource requests, limits and of the GPU resource's
	// status.capacity.tflops, which is vendor-published FP16 TFLOPS. It is the
	// only TFLOPS figure of a device that may be published, summed or used to
	// turn a TFLOPS quantity into a share of the device.
	//
	// The device controller resolves it when the device is discovered or
	// refreshed, and again when the hardware table changes: the table entry of
	// the device model when there is one, ProviderEstimatedTflops otherwise.
	// TflopsCapacitySource tells the two apart. Zero means unknown. Consumers
	// read it through EffectiveTflopsCapacity, which also covers a DeviceInfo
	// whose capacity was never resolved.
	TflopsCapacity       float64
	TflopsCapacitySource TflopsCapacitySource

	// ProviderEstimatedTflops is the raw figure reported by the provider
	// library, kept for diagnostics. It is a theoretical estimate in the
	// provider's own terms (NVIDIA: FP32, CUDA cores x max SM clock x 2), not
	// vendor-published FP16: 44.08 for an RTX 3090 (FP16: 71), about 19.5 for an
	// A100 (FP16: 312). It is the capacity only as a fallback, through
	// TflopsCapacity; nothing else may treat it as one.
	ProviderEstimatedTflops float64

	// MaxTflops mirrors ProviderEstimatedTflops under its original name.
	//
	// Deprecated: the name suggests a capacity, but this is the provider's raw
	// estimate. Read TflopsCapacity for the capacity and ProviderEstimatedTflops
	// for the estimate. A DeviceInfo that only sets MaxTflops is still
	// understood as carrying that estimate.
	MaxTflops float64

	VirtualizationCapabilities VirtualizationCapabilities
	Properties                 map[string]string
	Healthy                    bool

	ParentUUID string

	// Host - Guest device node mapping, eg /dev/nvidia0 -> /dev/nvidia0
	// When multiple device allocated, deduplicated by device node
	DeviceNode map[string]string

	// Env to inject to guest
	DeviceEnv map[string]string

	IsolationMode   IsolationMode
	IsolationPolicy tfv1.IsolationModePolicyType
}

// ResolveTflopsCapacity sets TflopsCapacity and TflopsCapacitySource, and
// reports whether either changed. tableTflops is the hardware table's FP16
// figure for the device model; listed is false when the table has no usable
// entry for it, in which case the provider's estimate becomes the capacity.
//
// This is the one place the two figures are weighed against each other. It is a
// method so that every implementation of a device controller resolves capacity
// the same way; consumers only ever read the result.
func (d *DeviceInfo) ResolveTflopsCapacity(tableTflops float64, listed bool) bool {
	capacity, source := NormalizeTflops(d.providerEstimate()), TflopsCapacitySourceProviderEstimate
	if listed && tableTflops > 0 {
		capacity, source = NormalizeTflops(tableTflops), TflopsCapacitySourceProviderConfig
	}
	changed := d.TflopsCapacity != capacity || d.TflopsCapacitySource != source
	d.TflopsCapacity, d.TflopsCapacitySource = capacity, source
	return changed
}

// TflopsCapacityEstimated reports whether the capacity had to fall back to the
// provider's estimate, i.e. is not in vendor-published FP16 TFLOPS.
func (d *DeviceInfo) TflopsCapacityEstimated() bool {
	return d.TflopsCapacitySource == TflopsCapacitySourceProviderEstimate
}

// unresolvedCapacityLogged remembers the devices for which
// EffectiveTflopsCapacity already said that it had to fall back.
var unresolvedCapacityLogged sync.Map

// EffectiveTflopsCapacity is the capacity every consumer computes with:
// TflopsCapacity once it was resolved. A DeviceInfo whose capacity was never
// resolved (built by code that predates TflopsCapacity and only sets the
// provider's figure, or by a device controller that does not call
// ResolveTflopsCapacity) falls back to that figure instead of reading as "no
// capacity", which would leave its workers unlimited and its metrics empty;
// this is said once per device. Zero means unknown.
func (d *DeviceInfo) EffectiveTflopsCapacity() float64 {
	if d == nil {
		return 0
	}
	if d.TflopsCapacity > 0 || d.TflopsCapacitySource != "" {
		return d.TflopsCapacity
	}
	estimate := NormalizeTflops(d.providerEstimate())
	if estimate > 0 {
		if _, logged := unresolvedCapacityLogged.LoadOrStore(d.UUID, struct{}{}); !logged {
			klog.Warningf("TFLOPS capacity of device %s (%s) was never resolved: using the provider's estimate of %.2f",
				d.UUID, d.Model, estimate)
		}
	}
	return estimate
}

func (d *DeviceInfo) providerEstimate() float64 {
	if d.ProviderEstimatedTflops != 0 {
		return d.ProviderEstimatedTflops
	}
	return d.MaxTflops //nolint:staticcheck // input alias for callers that predate ProviderEstimatedTflops
}

// TflopsQuantity is how a TFLOPS figure is written to a Kubernetes resource:
// with six decimals. Negative and non-finite figures are written as zero.
func TflopsQuantity(tflops float64) resource.Quantity {
	if math.IsNaN(tflops) || math.IsInf(tflops, 0) || tflops < 0 {
		tflops = 0
	}
	return resource.MustParse(fmt.Sprintf("%f", tflops))
}

// NormalizeTflops returns the figure exactly as a reader of TflopsQuantity gets
// it back. Capacities are stored normalized, so the hypervisor computes with
// the very number the scheduler reads from the GPU resource.
func NormalizeTflops(tflops float64) float64 {
	quantity := TflopsQuantity(tflops)
	return quantity.AsApproximateFloat64()
}

// DeviceTopology represents normalized topology metadata for a device.
// This structure is vendor-agnostic and is intended to be persisted to the GPU CR status.
// +k8s:deepcopy-gen=true
type DeviceTopology struct {
	Peers []DevicePeerLink
}

// DevicePeerLink represents the topology relationship between two devices.
// +k8s:deepcopy-gen=true
type DevicePeerLink struct {
	PeerUUID  string
	Tier      int32
	LinkType  string
	Bandwidth int64
}

type NodeInfo struct {
	// Extra metadata for centralized management
	RAMSizeBytes  int64
	DataDiskBytes int64

	// Aggregated info of whole Node. TotalTFlops is the sum of the devices'
	// TflopsCapacity, in the same unit.
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
	SupportsRemoting      bool
	MaxPartitions         uint32
	MaxWorkersPerDevice   uint32
}

// ProcessInformation represents combined compute and memory utilization for a process on a device
// This follows AMD SMI style API design, combining both metrics in a single structure
// +k8s:deepcopy-gen=true
type ProcessInformation struct {
	ProcessID  string
	DeviceUUID string
	// ComputeUtilizationPercent is the process's share of the device's compute
	// time (0-100). Zero is a measured idle process, unless
	// ComputeUtilizationUnavailable is set.
	ComputeUtilizationPercent float64
	// ComputeUtilizationUnavailable is set when the provider did not supply
	// per-process compute utilization for this device in this sample: a
	// provider library that does not measure it at all, an unsupported driver,
	// a MIG or virtualized device, a failed query. ComputeUtilizationPercent is
	// then zero and must not be read as a measurement, summed or exported. The
	// memory fields stay valid.
	ComputeUtilizationUnavailable bool
	// ProcessListUnavailable marks a row that is not a process: the provider
	// could not read the process list of DeviceUUID in this sample. ProcessID
	// is empty and every other field is zero. Nothing is known about that
	// device's processes; it is not "no processes".
	ProcessListUnavailable   bool
	ActiveSMs                uint64
	TotalSMs                 uint64
	MemoryUsedBytes          uint64
	MemoryReservedBytes      uint64
	MemoryUtilizationPercent float64
}

// ComputeUtilization represents compute utilization for a process on a device
// DEPRECATED: Use ProcessInformation instead
type ComputeUtilization struct {
	ProcessID          string
	DeviceUUID         string
	UtilizationPercent float64
}

// MemoryUtilization represents memory utilization for a process on a device
// DEPRECATED: Use ProcessInformation instead
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
	// ComputeUnavailable is set when the device's utilization could not be
	// read: ComputePercentage and ComputeTflops are then zero and are not
	// measurements. MemoryUnavailable says the same of MemoryBytes and
	// MemoryPercentage. The remaining fields hold what was read. A field that
	// is marked must not be exported or summed.
	ComputeUnavailable bool
	MemoryUnavailable  bool
	Rx                 float64 // PCIe RX in KB
	Tx                 float64 // PCIe TX in KB
	Temperature        float64
	PowerUsage         int64 // in watts
	ExtraMetrics       map[string]float64
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
	// ComputeUnavailable is set when the provider did not supply compute
	// utilization for this process. ComputePercentage and ComputeTflops are
	// then zero and are not measurements.
	ComputeUnavailable bool
	ExtraMetrics       map[string]float64
}
