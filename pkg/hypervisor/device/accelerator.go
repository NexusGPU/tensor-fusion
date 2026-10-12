/*
 * Copyright 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package device

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"k8s.io/klog/v2"
)

// C structure definitions matching accelerator.h
// These must match the C struct definitions exactly

type Result int32

const (
	ResultSuccess                Result = 0
	ResultErrorInvalidParam      Result = 1
	ResultErrorNotFound          Result = 2
	ResultErrorNotSupported      Result = 3
	ResultErrorResourceExhausted Result = 4
	ResultErrorOperationFailed   Result = 5
	ResultErrorInternal          Result = 6
)

// AcceleratorError wraps a Result code as an error for type checking
type AcceleratorError struct {
	Code    Result
	Message string
}

func (e *AcceleratorError) Error() string {
	return e.Message
}

// IsNonFatalAcceleratorError returns true if the error is a non-fatal accelerator error
// that should not block the main flow (e.g., NOT_SUPPORTED, NOT_FOUND)
func IsNonFatalAcceleratorError(err error) bool {
	if err == nil {
		return true
	}
	accelErr, ok := err.(*AcceleratorError)
	if !ok {
		return false
	}
	switch accelErr.Code {
	case ResultErrorNotSupported, ResultErrorNotFound:
		return true
	default:
		return false
	}
}

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

// DeviceBasicInfo matches the C struct DeviceBasicInfo in provider/accelerator.h
// Field names in Go are capitalized for export, but memory layout must match C struct exactly
// C struct fields: uuid, vendor, model, driverVersion, firmwareVersion, deviceNode, index, numaNode,
//
//	totalMemoryBytes, totalComputeUnits, maxTflops, pcieGen, pcieWidth
type DeviceBasicInfo struct {
	UUID              [64]byte  // C: char uuid[64]
	Vendor            [32]byte  // C: char vendor[32]
	Model             [128]byte // C: char model[128]
	DriverVersion     [80]byte  // C: char driverVersion[80]
	FirmwareVersion   [64]byte  // C: char firmwareVersion[64]
	DeviceNode        [64]byte  // C: char deviceNode[64]
	Index             int32     // C: int32_t index
	NUMANode          int32     // C: int32_t numaNode
	TotalMemoryBytes  uint64    // C: uint64_t totalMemoryBytes
	TotalComputeUnits uint64    // C: uint64_t totalComputeUnits
	MaxTflops         float64   // C: double maxTflops
	PCIeGen           uint32    // C: uint32_t pcieGen
	PCIeWidth         uint32    // C: uint32_t pcieWidth
}

type DevicePropertyKV struct {
	Key   [64]byte
	Value [256]byte
}

const MaxDeviceProperties = 64

type DeviceProperties struct {
	Properties [MaxDeviceProperties]DevicePropertyKV
	Count      uintptr
}

type ExtendedDeviceInfo struct {
	Basic        DeviceBasicInfo
	Props        DeviceProperties
	Capabilities VirtualizationCapabilities
}

type ExtraMetric struct {
	Key   [64]byte
	Value float64
}

const MaxExtraMetrics = 64

// ProcessInformation combines compute and memory utilization (AMD SMI style)
type ProcessInformation struct {
	ProcessID                 [32]byte
	DeviceUUID                [64]byte
	ComputeUtilizationPercent float64
	ActiveSMs                 uint64
	TotalSMs                  uint64
	MemoryUsedBytes           uint64
	MemoryReservedBytes       uint64
	MemoryUtilizationPercent  float64
}

// Optional features of provider/accelerator.h (ACCEL_FEATURE_*), selected with
// AccelSetFeatures. This is the one place where the features this caller
// understands are declared.
const (
	// C: ACCEL_FEATURE_TELEMETRY_MARKERS. Without it the library does not
	// measure per-process utilization (the field is always 0) and never
	// returns one of the markers below.
	featureTelemetryMarkers uint64 = 1 << 0

	// requestedFeatures is what is asked for after every AccelInit.
	requestedFeatures = featureTelemetryMarkers
)

// In-band "not supplied" markers of provider/accelerator.h, returned by a
// library that agreed to featureTelemetryMarkers. They never leave this file:
// each is turned into an explicit flag of the api type it was found in.
const (
	// C: ACCEL_COMPUTE_UTILIZATION_UNAVAILABLE, in ProcessInformation.computeUtilizationPercent.
	computeUtilizationUnavailable = -1.0
	// C: ACCEL_DEVICE_UTILIZATION_UNAVAILABLE, in DeviceMetrics.utilizationPercent.
	deviceUtilizationUnavailable = math.MaxUint32
	// C: ACCEL_DEVICE_MEMORY_UNAVAILABLE, in DeviceMetrics.memoryUsedBytes.
	deviceMemoryUnavailable = math.MaxUint64
)

// The process list is read into a buffer that is kept between calls. The
// library returns nothing at all when its rows do not fit, so the buffer grows
// until they do, up to maxProcessRows (about 2.4 MB).
const (
	initialProcessRows = 1024
	maxProcessRows     = 16384
)

type DeviceMetrics struct {
	DeviceUUID         [64]byte
	PowerUsageWatts    float64
	TemperatureCelsius float64
	PCIeRxBytes        uint64
	PCIeTxBytes        uint64
	UtilizationPercent uint32
	MemoryUsedBytes    uint64
	ExtraMetrics       [MaxExtraMetrics]ExtraMetric
	ExtraMetricsCount  uintptr
}

const (
	MaxTopologyDevices = 64
)

// TopoLevelType represents GPU-to-GPU connection type
type TopoLevelType int32

const (
	TopoLevelInternal     TopoLevelType = 0 // e.g. Tesla K80 (same board)
	TopoLevelSingleSwitch TopoLevelType = 1 // single PCIe switch
	TopoLevelMultiSwitch  TopoLevelType = 2 // multiple PCIe switches (no host bridge traversal)
	TopoLevelHostBridge   TopoLevelType = 3 // same host bridge
	TopoLevelNUMANode     TopoLevelType = 4 // same NUMA node
	TopoLevelSystem       TopoLevelType = 5 // cross NUMA (system level)
	TopoLevelSelf         TopoLevelType = 6 // same device
	TopoLevelUnknown      TopoLevelType = 7 // unknown or error
)

// DeviceTopoNode represents connection to another device
type DeviceTopoNode struct {
	PeerUUID             [64]byte      // Peer device UUID
	PeerIndex            int32         // Peer device index
	TopoLevel            TopoLevelType // Topology level to this peer
	BandwidthBytesPerSec int64         // Estimated link bandwidth in bytes/sec, 0 if unknown
}

// DeviceTopologyInfo represents a device and its topology to all other devices
type DeviceTopologyInfo struct {
	DeviceUUID  [64]byte                           // This device's UUID
	DeviceIndex int32                              // This device's index
	NUMANode    int32                              // This device's NUMA node
	Peers       [MaxTopologyDevices]DeviceTopoNode // Topology to all other devices
	PeerCount   uintptr                            // Number of peers
}

// ExtendedDeviceTopology contains topology for all devices
type ExtendedDeviceTopology struct {
	Devices     [MaxTopologyDevices]DeviceTopologyInfo // Array of device topology rows
	DeviceCount uintptr                                // Number of devices
}

const MaxMountPath = 512

type Mount struct {
	HostPath  [MaxMountPath]byte
	GuestPath [MaxMountPath]byte
}

const (
	MaxPartitionEnvs             = 16
	MaxEnvValueLength            = 256
	MaxPartitionDeviceNodes      = 16
	MaxPartitionDeviceNodeLength = MaxMountPath*2 + 2
)

const MaxProcesses = 1024

// PartitionResultType represents the type of partition result
type PartitionResultType int32

const (
	PartitionTypeEnvironmentVariable PartitionResultType = 0
	PartitionTypeDeviceNode          PartitionResultType = 1
)

// PartitionResult matches the C struct PartitionResult in provider/accelerator.h
// Field names in Go are capitalized for export, but memory layout must match C struct exactly
// C struct fields: type, deviceUUID, envVars, deviceNodes
type PartitionResult struct {
	Type        PartitionResultType                       // C: PartitionResultType type
	DeviceUUID  [64]byte                                  // C: char deviceUUID[64]
	EnvVars     [MaxPartitionEnvs][MaxEnvValueLength]byte // C: char envVars[16][256], key-value pairs like "A=B"
	DeviceNodes [MaxPartitionDeviceNodes][MaxPartitionDeviceNodeLength]byte
	// C: char deviceNodes[16][MAX_MOUNT_PATH*2+2], key-value pairs like "host=guest"
}

// SnapshotContext for snapshot/resume operations
// Supports both process-level (CUDA) and device-level (other vendors) snapshots
type SnapshotContext struct {
	ProcessIDs   *int32  // Pointer to array of process IDs (for process-level snapshot, NULL for device-level)
	ProcessCount uintptr // Number of processes (0 for device-level snapshot)
	DeviceUUID   *byte   // Device UUID (for device-level snapshot, NULL for process-level)
}

// Function pointer types for purego
var (
	libHandle uintptr
	// DeviceInfo APIs
	accelInit             func() Result
	accelShutdown         func() Result
	getDeviceCount        func(*uintptr) Result
	getAllDevices         func(*ExtendedDeviceInfo, uintptr, *uintptr) Result
	getAllDevicesTopology func(*ExtendedDeviceTopology) Result
	// Virtualization APIs - signatures match C header exactly:
	// AccelResult AssignPartition(const char* templateId, const char* deviceUUID, PartitionResult* partitionResult);
	assignPartition func(*byte, *byte, *PartitionResult) Result
	// AccelResult RemovePartition(const char* templateId, const char* deviceUUID);
	removePartition func(*byte, *byte) Result
	// AccelResult SetMemHardLimit(const char* deviceUUID, uint64_t memoryLimitBytes);
	setMemHardLimit func(*byte, uint64) Result
	// AccelResult SetComputeUnitHardLimit(const char* deviceUUID, uint32_t computeUnitLimit);
	setComputeUnitHardLimit func(*byte, uint32) Result
	// AccelResult Snapshot(SnapshotContext* context);
	snapshot func(*SnapshotContext) Result
	// AccelResult Resume(SnapshotContext* context);
	resume func(*SnapshotContext) Result
	// Metrics APIs
	getProcessInformation func(*ProcessInformation, uintptr, *uintptr) Result
	getDeviceMetrics      func(**byte, uintptr, *DeviceMetrics) Result
	getVendorMountLibs    func(*Mount, uintptr, *uintptr) Result
	// Utility APIs (Unix only, set by accelerator_unix.go)
	registerLogCallback func(uintptr) Result
	// AccelResult AccelSetFeatures(uint64_t requested, uint64_t* enabled);
	// Added after the first release of the ABI: nil for a library that does
	// not export it.
	setFeatures func(uint64, *uint64) Result
)

// AcceleratorInterface provides Go bindings for the C accelerator library using purego
type AcceleratorInterface struct {
	libPath         string
	deviceProcesses map[string][]string
	mu              sync.RWMutex
	loaded          bool

	// telemetryMarkers says that the loaded library agreed to
	// featureTelemetryMarkers after its last initialization. See
	// TelemetryMarkers.
	telemetryMarkers atomic.Bool

	// processRows is the buffer the library writes the process list into, kept
	// between calls and guarded by processMu.
	processMu   sync.Mutex
	processRows []ProcessInformation
}

// initialize starts the library and selects the optional features this caller
// understands. Every path that initializes the library goes through it: the
// library forgets the selection when it is shut down.
func (a *AcceleratorInterface) initialize() error {
	a.telemetryMarkers.Store(false)
	if result := accelInit(); result != ResultSuccess {
		return fmt.Errorf("failed to initialize accelerator: %d", result)
	}
	a.selectFeatures()
	return nil
}

// selectFeatures asks the library for requestedFeatures and records what it
// agreed to. A library without AccelSetFeatures, one that refuses, and one that
// does not know a feature all leave that feature off.
func (a *AcceleratorInterface) selectFeatures() {
	if setFeatures == nil {
		klog.Infof("Accelerator library %s has no optional features: per-process utilization is not supplied", a.libPath)
		return
	}
	var enabled uint64
	if result := setFeatures(requestedFeatures, &enabled); result != ResultSuccess {
		klog.Warningf("Accelerator library %s refused the feature selection (%d): per-process utilization is not supplied",
			a.libPath, result)
		return
	}
	a.telemetryMarkers.Store(enabled&featureTelemetryMarkers != 0)
	klog.Infof("Accelerator library %s features: requested 0x%x, enabled 0x%x", a.libPath, requestedFeatures, enabled)
}

// TelemetryMarkers reports whether the library measures per-process
// utilization and marks what it could not read (ACCEL_FEATURE_TELEMETRY_MARKERS).
// When it does not, per-process utilization is not supplied for any device:
// every api.ProcessInformation has ComputeUtilizationUnavailable set.
func (a *AcceleratorInterface) TelemetryMarkers() bool {
	return a.telemetryMarkers.Load()
}

// NewAcceleratorInterface creates a new accelerator interface and loads the library
func NewAcceleratorInterface(libPath string) (*AcceleratorInterface, error) {
	accel := &AcceleratorInterface{
		libPath:         libPath,
		deviceProcesses: make(map[string][]string),
		loaded:          false,
	}

	// Load the library
	if err := accel.Load(); err != nil {
		return nil, fmt.Errorf("failed to load accelerator library from %s: %w", libPath, err)
	}

	return accel, nil
}

// Close unloads the accelerator library
func (a *AcceleratorInterface) Close() error {
	if a.loaded && libHandle != 0 {
		// Unregister log callback if it was registered (Unix only)
		func() {
			defer func() {
				if r := recover(); r != nil {
					// registerLogCallback not available or already unregistered
					_ = r // ignore recovery value
				}
			}()
			accelShutdown()
			// The library forgets the feature selection with it.
			a.telemetryMarkers.Store(false)
			if registerLogCallback != nil {
				registerLogCallback(0)
			}
		}()
		// Note: purego doesn't provide DlClose, but the library will be unloaded when process exits
		a.loaded = false
	}
	return nil
}

// byteArrayToString converts a fixed-size byte array to Go string
func byteArrayToString(arr []byte) string {
	// Find null terminator
	for i, b := range arr {
		if b == 0 {
			return string(arr[:i])
		}
	}
	return string(arr)
}

func topoLevelToTier(level TopoLevelType) int32 {
	switch level {
	case TopoLevelInternal:
		return 0 // High-speed interconnect: NVLink, HCCS, or same board
	case TopoLevelSingleSwitch, TopoLevelMultiSwitch, TopoLevelHostBridge, TopoLevelNUMANode:
		return 1 // PCIe local or same NUMA node
	case TopoLevelSystem:
		return 2 // Cross NUMA
	default:
		return 3 // Unknown
	}
}

func topoLevelToLinkType(level TopoLevelType) string {
	switch level {
	case TopoLevelInternal:
		return "Internal"
	case TopoLevelSingleSwitch:
		return "SingleSwitch"
	case TopoLevelMultiSwitch:
		return "MultiSwitch"
	case TopoLevelHostBridge:
		return "HostBridge"
	case TopoLevelNUMANode:
		return "NUMA"
	case TopoLevelSystem:
		return "System"
	case TopoLevelSelf:
		return "Self"
	default:
		return "Unknown"
	}
}

func (a *AcceleratorInterface) getTopologyMap() (map[string]*api.DeviceTopology, error) {
	var cTopology ExtendedDeviceTopology
	result := getAllDevicesTopology(&cTopology)
	if result != ResultSuccess {
		return nil, &AcceleratorError{
			Code:    result,
			Message: fmt.Sprintf("failed to get all devices topology: %d", result),
		}
	}

	topologyMap := make(map[string]*api.DeviceTopology, int(cTopology.DeviceCount))
	for i := 0; i < int(cTopology.DeviceCount); i++ {
		row := &cTopology.Devices[i]
		deviceUUID := strings.ToLower(byteArrayToString(row.DeviceUUID[:]))
		if deviceUUID == "" {
			continue
		}
		peers := make([]api.DevicePeerLink, 0, int(row.PeerCount))
		for j := 0; j < int(row.PeerCount) && j < MaxTopologyDevices; j++ {
			peer := &row.Peers[j]
			peerUUID := strings.ToLower(byteArrayToString(peer.PeerUUID[:]))
			if peerUUID == "" || peerUUID == deviceUUID || peer.TopoLevel == TopoLevelSelf {
				continue
			}
			peers = append(peers, api.DevicePeerLink{
				PeerUUID:  peerUUID,
				Tier:      topoLevelToTier(peer.TopoLevel),
				LinkType:  topoLevelToLinkType(peer.TopoLevel),
				Bandwidth: peer.BandwidthBytesPerSec,
			})
		}
		topologyMap[deviceUUID] = &api.DeviceTopology{
			Peers: peers,
		}
	}
	return topologyMap, nil
}

// GetTotalProcessCount returns the total number of processes across all devices
func (a *AcceleratorInterface) GetTotalProcessCount() int {
	a.mu.RLock()
	defer a.mu.RUnlock()

	total := 0
	for _, processes := range a.deviceProcesses {
		total += len(processes)
	}
	return total
}

// GetDeviceMetrics retrieves device metrics for the specified device UUIDs
func (a *AcceleratorInterface) GetDeviceMetrics(deviceUUIDs []string) ([]*api.GPUUsageMetrics, error) {
	if len(deviceUUIDs) == 0 {
		return []*api.GPUUsageMetrics{}, nil
	}

	const maxStackDevices = 64
	deviceCount := min(len(deviceUUIDs), maxStackDevices)

	// Convert Go strings to C string pointers array
	// Allocate C strings with null terminators
	cStrings := make([]*byte, deviceCount)
	cStringData := make([][]byte, deviceCount)
	for i := range deviceCount {
		// Convert Go string to null-terminated C string
		cStringData[i] = []byte(deviceUUIDs[i])
		cStringData[i] = append(cStringData[i], 0) // null terminator
		cStrings[i] = &cStringData[i][0]
	}

	// Allocate stack buffer for metrics
	var cMetrics [maxStackDevices]DeviceMetrics

	result := getDeviceMetrics(&cStrings[0], uintptr(deviceCount), &cMetrics[0])
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to get device metrics: %d", result)
	}

	// Convert C metrics to Go metrics
	metrics := make([]*api.GPUUsageMetrics, 0, deviceCount)
	for i := range deviceCount {
		cm := &cMetrics[i]
		// Utilization and memory are marked one by one when they could not be
		// read; the other fields of the entry still hold what was read. A value
		// above 100 that is not the marker is outside the contract and no
		// measurement either.
		computePercentage, computeUnavailable := float64(cm.UtilizationPercent), false
		if cm.UtilizationPercent > 100 {
			if cm.UtilizationPercent != deviceUtilizationUnavailable {
				klog.V(4).Infof("Ignoring utilization of device %s: %d%% is out of range",
					byteArrayToString(cm.DeviceUUID[:]), cm.UtilizationPercent)
			}
			computePercentage, computeUnavailable = 0, true
		}
		// Markers only mean something when the library agreed to return them;
		// from a library that did not, the value is whatever it reported.
		markers := a.telemetryMarkers.Load()
		memoryUsed, memoryUnavailable := cm.MemoryUsedBytes, false
		if markers && cm.MemoryUsedBytes == deviceMemoryUnavailable {
			memoryUsed, memoryUnavailable = 0, true
		}
		// With both marked the device could not be opened at all. A library
		// that returns no markers has no way to say which fields it read, so
		// an unusable utilization leaves nothing to trust in its entry. Either
		// way the device is left out: "no sample" to every consumer instead of
		// an idle, cold, powerless device.
		if computeUnavailable && (memoryUnavailable || !markers) {
			continue
		}
		var memoryPercentage float64 = 0

		// Convert extra metrics from C array to Go map
		extraMetrics := make(map[string]float64, int(cm.ExtraMetricsCount))
		// Add extra metrics from C array
		for j := 0; j < int(cm.ExtraMetricsCount); j++ {
			em := &cm.ExtraMetrics[j]
			key := byteArrayToString(em.Key[:])
			if key != "" {
				extraMetrics[key] = em.Value
			}
		}

		metrics = append(metrics, &api.GPUUsageMetrics{
			DeviceUUID:         byteArrayToString(cm.DeviceUUID[:]),
			MemoryBytes:        memoryUsed,
			MemoryPercentage:   memoryPercentage,
			ComputePercentage:  computePercentage,
			ComputeTflops:      0,
			ComputeUnavailable: computeUnavailable,
			MemoryUnavailable:  memoryUnavailable,
			Rx:                 float64(cm.PCIeRxBytes),
			Tx:                 float64(cm.PCIeTxBytes),
			Temperature:        float64(cm.TemperatureCelsius),
			PowerUsage:         int64(cm.PowerUsageWatts),
			ExtraMetrics:       extraMetrics,
		})
	}

	return metrics, nil
}

// GetAllDevices retrieves all available devices from the accelerator library
func (a *AcceleratorInterface) GetAllDevices() ([]*api.DeviceInfo, error) {
	// First, get the device count
	var cDeviceCount uintptr
	result := getDeviceCount(&cDeviceCount)
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to get device count: %d", result)
	}

	if cDeviceCount == 0 {
		klog.Infof("GetAllDevices: device count is 0")
		return []*api.DeviceInfo{}, nil
	}

	// Allocate stack buffer (max 64 devices to avoid stack overflow)
	const maxStackDevices = 64
	var stackDevices [maxStackDevices]ExtendedDeviceInfo
	maxDevices := min(int(cDeviceCount), maxStackDevices)

	var cCount uintptr
	klog.Infof("Getting all devices, max devices count: %d", maxDevices)
	result = getAllDevices(&stackDevices[0], uintptr(maxDevices), &cCount)
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to get all devices: %d", result)
	}

	if cCount == 0 {
		klog.Infof("GetAllDevices: returned 0 devices")
		return []*api.DeviceInfo{}, nil
	}

	devices := make([]*api.DeviceInfo, int(cCount))
	topologyMap, topoErr := a.getTopologyMap()
	if topoErr != nil {
		if IsNonFatalAcceleratorError(topoErr) {
			klog.Warningf("GetAllDevices: topology unavailable, continuing without topology metadata: %v", topoErr)
		} else {
			klog.Warningf("GetAllDevices: failed to retrieve topology, continuing without topology metadata: %v", topoErr)
		}
		topologyMap = nil
	}
	for i := 0; i < int(cCount); i++ {
		cInfo := &stackDevices[i]
		deviceUUID := byteArrayToString(cInfo.Basic.UUID[:])
		vendor := byteArrayToString(cInfo.Basic.Vendor[:])

		// Convert DeviceProperties KV array to map
		properties := make(map[string]string, int(cInfo.Props.Count))
		for j := 0; j < int(cInfo.Props.Count) && j < MaxDeviceProperties; j++ {
			key := byteArrayToString(cInfo.Props.Properties[j].Key[:])
			value := byteArrayToString(cInfo.Props.Properties[j].Value[:])
			if key != "" {
				properties[key] = value
			}
		}
		if driver := byteArrayToString(cInfo.Basic.DriverVersion[:]); driver != "" {
			properties["driverVersion"] = driver
		}
		if fw := byteArrayToString(cInfo.Basic.FirmwareVersion[:]); fw != "" {
			properties["firmwareVersion"] = fw
		}
		if cInfo.Basic.TotalComputeUnits != 0 {
			properties["totalComputeUnits"] = fmt.Sprintf("%d", cInfo.Basic.TotalComputeUnits)
		}
		if cInfo.Basic.PCIeGen != 0 {
			properties["pcieGen"] = fmt.Sprintf("%d", cInfo.Basic.PCIeGen)
		}
		if cInfo.Basic.PCIeWidth != 0 {
			properties["pcieWidth"] = fmt.Sprintf("%d", cInfo.Basic.PCIeWidth)
		}

		model := byteArrayToString(cInfo.Basic.Model[:])
		var deviceNodes map[string]string
		if hostPath := byteArrayToString(cInfo.Basic.DeviceNode[:]); hostPath != "" {
			// Legacy ABI exposes only a single device node, use host==guest by default.
			deviceNodes = map[string]string{hostPath: hostPath}
		}
		if nodes, applied := applyProviderDeviceMountPolicy(deviceNodes, vendor, model, false, nil); applied {
			deviceNodes = nodes
		}

		devices[i] = &api.DeviceInfo{
			UUID:             deviceUUID,
			Vendor:           vendor,
			Model:            model,
			Index:            cInfo.Basic.Index,
			NUMANode:         cInfo.Basic.NUMANode,
			Topology:         topologyMap[strings.ToLower(deviceUUID)],
			TotalMemoryBytes: cInfo.Basic.TotalMemoryBytes,
			// The provider's own estimate. The device controller decides what the
			// device's TFLOPS capacity is.
			ProviderEstimatedTflops: cInfo.Basic.MaxTflops,
			MaxTflops:               cInfo.Basic.MaxTflops, //nolint:staticcheck // deprecated mirror, see api.DeviceInfo
			VirtualizationCapabilities: api.VirtualizationCapabilities{
				SupportsPartitioning:  cInfo.Capabilities.SupportsPartitioning,
				SupportsSoftIsolation: cInfo.Capabilities.SupportsSoftIsolation,
				SupportsHardIsolation: cInfo.Capabilities.SupportsHardIsolation,
				SupportsSnapshot:      cInfo.Capabilities.SupportsSnapshot,
				SupportsMetrics:       cInfo.Capabilities.SupportsMetrics,
				SupportsRemoting:      cInfo.Capabilities.SupportsRemoting,
				MaxPartitions:         cInfo.Capabilities.MaxPartitions,
				MaxWorkersPerDevice:   cInfo.Capabilities.MaxWorkersPerDevice,
			},
			Properties: properties,
			DeviceNode: deviceNodes,
		}

		klog.Infof(
			"GetAllDevices[%d]: uuid=%s vendor=%s model=%s index=%d numa=%d vram=%d providerEstimatedTflops=%.2f props=%v",
			i,
			devices[i].UUID,
			devices[i].Vendor,
			devices[i].Model,
			devices[i].Index,
			devices[i].NUMANode,
			devices[i].TotalMemoryBytes,
			devices[i].ProviderEstimatedTflops,
			devices[i].Properties,
		)
	}
	return devices, nil
}

// AssignPartitionResult represents the result of assigning a partition
type AssignPartitionResult struct {
	PartitionUUID string
	EnvVars       map[string]string
	DeviceNodes   map[string]string
	Type          PartitionResultType
}

// AssignPartition assigns a partition to a device using a template (e.g., create MIG instance)
// Returns the assigned partition result including UUID and environment variables
func (a *AcceleratorInterface) AssignPartition(templateID, deviceUUID string) (*AssignPartitionResult, error) {
	// Validate input lengths
	const maxIDLength = 64
	if len(templateID) >= maxIDLength {
		return nil, fmt.Errorf("template ID is too long (max %d bytes)", maxIDLength-1)
	}
	if len(deviceUUID) >= maxIDLength {
		return nil, fmt.Errorf("device UUID is too long (max %d bytes)", maxIDLength-1)
	}

	// Create null-terminated C strings
	var templateArr [64]byte
	var deviceArr [64]byte

	copy(templateArr[:], templateID)
	copy(deviceArr[:], deviceUUID)
	if len(templateID) < len(templateArr) {
		templateArr[len(templateID)] = 0
	}
	if len(deviceUUID) < len(deviceArr) {
		deviceArr[len(deviceUUID)] = 0
	}

	// Allocate PartitionResult struct for C function to fill
	var cResult PartitionResult

	result := assignPartition(&templateArr[0], &deviceArr[0], &cResult)
	if result != ResultSuccess {
		return nil, fmt.Errorf("failed to assign partition: %d", result)
	}

	// Extract partition UUID from deviceUUID field
	partitionUUID := byteArrayToString(cResult.DeviceUUID[:])

	// Parse optional env vars returned by vendor
	envVars := make(map[string]string)
	for i := range MaxPartitionEnvs {
		envVarStr := byteArrayToString(cResult.EnvVars[i][:])
		if envVarStr == "" {
			continue
		}
		// Parse key-value pair in format "A=B".
		if key, value, ok := splitKeyValue(envVarStr); ok {
			envVars[key] = value
		}
	}

	// Parse optional device node mapping pairs returned by vendor in format "hostPath=guestPath".
	deviceNodes := make(map[string]string)
	for i := range MaxPartitionDeviceNodes {
		deviceNodePair := byteArrayToString(cResult.DeviceNodes[i][:])
		if deviceNodePair == "" {
			continue
		}
		if hostPath, guestPath, ok := splitKeyValue(deviceNodePair); ok {
			deviceNodes[hostPath] = guestPath
		}
	}
	if len(deviceNodes) == 0 {
		deviceNodes = nil
	}

	return &AssignPartitionResult{
		PartitionUUID: partitionUUID,
		EnvVars:       envVars,
		DeviceNodes:   deviceNodes,
		Type:          cResult.Type,
	}, nil
}

func splitKeyValue(input string) (string, string, bool) {
	separatorIndex := -1
	for i := 0; i < len(input); i++ {
		if input[i] == '=' {
			separatorIndex = i
			break
		}
	}
	if separatorIndex <= 0 || separatorIndex >= len(input)-1 {
		return "", "", false
	}
	return input[:separatorIndex], input[separatorIndex+1:], true
}

// RemovePartition removes a partition from a device
// templateID is the template ID used to create the partition
func (a *AcceleratorInterface) RemovePartition(templateID, deviceUUID string) error {
	// Create null-terminated C strings
	var templateArr [64]byte
	var deviceArr [64]byte

	copy(templateArr[:], templateID)
	copy(deviceArr[:], deviceUUID)
	if len(templateID) < len(templateArr) {
		templateArr[len(templateID)] = 0
	}
	if len(deviceUUID) < len(deviceArr) {
		deviceArr[len(deviceUUID)] = 0
	}

	result := removePartition(&templateArr[0], &deviceArr[0])
	if result != ResultSuccess {
		return fmt.Errorf("failed to remove partition: %d", result)
	}

	return nil
}

// SetMemHardLimit sets hard memory limit for a device (one-time, called at worker start by limiter.so)
func (a *AcceleratorInterface) SetMemHardLimit(deviceUUID string, memoryLimitBytes uint64) error {
	var deviceArr [64]byte
	copy(deviceArr[:], deviceUUID)
	if len(deviceUUID) < len(deviceArr) {
		deviceArr[len(deviceUUID)] = 0
	}

	result := setMemHardLimit(&deviceArr[0], memoryLimitBytes)
	if result != ResultSuccess {
		return fmt.Errorf("failed to set memory hard limit: %d", result)
	}

	return nil
}

// SetComputeUnitHardLimit sets hard compute unit limit for a device (one-time, called at worker start)
func (a *AcceleratorInterface) SetComputeUnitHardLimit(deviceUUID string, computeUnitLimit uint32) error {
	var deviceArr [64]byte
	copy(deviceArr[:], deviceUUID)
	if len(deviceUUID) < len(deviceArr) {
		deviceArr[len(deviceUUID)] = 0
	}

	result := setComputeUnitHardLimit(&deviceArr[0], computeUnitLimit)
	if result != ResultSuccess {
		return fmt.Errorf("failed to set compute unit hard limit: %d", result)
	}

	return nil
}

// GetProcessInformation retrieves process information (compute and memory utilization) for all processes
// on all devices. This combines the functionality of GetProcessComputeUtilization and GetProcessMemoryUtilization
// following AMD SMI style API design.
// Note: This directly calls the C API which returns all GPU processes, regardless of what Go tracks internally.
//
// With TelemetryMarkers the result carries, besides the processes, one row
// with ProcessListUnavailable for every device whose process list could not be
// read. Without, no process has a compute utilization.
//
// The call reads the driver's utilization samples and takes milliseconds per
// device: it is meant to be made once per sampling interval, by one caller.
func (a *AcceleratorInterface) GetProcessInformation() ([]api.ProcessInformation, error) {
	a.processMu.Lock()
	defer a.processMu.Unlock()

	count, err := a.readProcessRowsLocked()
	if err != nil {
		return nil, err
	}
	if count == 0 {
		return []api.ProcessInformation{}, nil
	}

	markers := a.telemetryMarkers.Load()
	processInfos := make([]api.ProcessInformation, count)
	for i := range count {
		pi := &a.processRows[i]
		processID := byteArrayToString(pi.ProcessID[:])
		if markers && processID == "" {
			// A row without a process is the marker of a device whose process
			// list could not be read. Only a library that agreed to return
			// markers means that by it; from any other, the row is passed on
			// as it always was and ignored by whoever cannot parse its id.
			processInfos[i] = api.ProcessInformation{
				DeviceUUID:                    byteArrayToString(pi.DeviceUUID[:]),
				ComputeUtilizationUnavailable: true,
				ProcessListUnavailable:        true,
			}
			continue
		}
		computePercent, computeUnavailable := convertProcessComputeUtilization(pi.ComputeUtilizationPercent)
		if !markers {
			// The library does not measure it: the field is always zero and
			// carries no information.
			computePercent, computeUnavailable = 0, true
		}
		processInfos[i] = api.ProcessInformation{
			ProcessID:                     processID,
			DeviceUUID:                    byteArrayToString(pi.DeviceUUID[:]),
			ComputeUtilizationPercent:     computePercent,
			ComputeUtilizationUnavailable: computeUnavailable,
			ActiveSMs:                     pi.ActiveSMs,
			TotalSMs:                      pi.TotalSMs,
			MemoryUsedBytes:               pi.MemoryUsedBytes,
			MemoryReservedBytes:           pi.MemoryReservedBytes,
			MemoryUtilizationPercent:      pi.MemoryUtilizationPercent,
		}
	}

	return processInfos, nil
}

// readProcessRowsLocked has the library fill a.processRows and returns the
// number of rows. A library whose rows do not fit returns none of them (a
// truncated list would under-account memory), so the buffer is doubled and the
// call repeated, up to maxProcessRows. The buffer keeps its size afterwards:
// in the steady state nothing is allocated for it. The caller holds processMu.
func (a *AcceleratorInterface) readProcessRowsLocked() (int, error) {
	if len(a.processRows) == 0 {
		a.processRows = make([]ProcessInformation, initialProcessRows)
	}
	for {
		var cCount uintptr
		result := getProcessInformation(&a.processRows[0], uintptr(len(a.processRows)), &cCount)
		switch {
		case result == ResultSuccess:
			return min(int(cCount), len(a.processRows)), nil
		case result != ResultErrorResourceExhausted:
			return 0, fmt.Errorf("failed to get process information: %d", result)
		case len(a.processRows) >= maxProcessRows:
			return 0, fmt.Errorf("failed to get process information: more than %d rows", maxProcessRows)
		}
		rows := min(2*len(a.processRows), maxProcessRows)
		klog.Infof("Process list does not fit in %d rows, growing the buffer to %d", len(a.processRows), rows)
		a.processRows = make([]ProcessInformation, rows)
	}
}

// convertProcessComputeUtilization separates a supplied per-process compute
// utilization from "not supplied". The provider marks the latter with exactly
// computeUtilizationUnavailable; zero is a measured idle process. Any other
// negative or non-finite value is outside the contract and is not a
// measurement either.
func convertProcessComputeUtilization(percent float64) (value float64, unavailable bool) {
	if percent == computeUtilizationUnavailable || math.IsNaN(percent) || math.IsInf(percent, 0) || percent < 0 {
		return 0, true
	}
	return percent, false
}

// GetVendorMountLibs retrieves vendor mount libs
func (a *AcceleratorInterface) GetVendorMountLibs() ([]*api.Mount, error) {
	const maxStackMounts = 64
	var stackMounts [maxStackMounts]Mount
	var cCount uintptr

	result := getVendorMountLibs(&stackMounts[0], uintptr(maxStackMounts), &cCount)
	if result != ResultSuccess {
		return nil, &AcceleratorError{
			Code:    result,
			Message: fmt.Sprintf("failed to get vendor mount libs: %d", result),
		}
	}

	if cCount == 0 {
		return []*api.Mount{}, nil
	}

	mounts := make([]*api.Mount, int(cCount))
	for i := 0; i < int(cCount); i++ {
		cm := &stackMounts[i]
		mounts[i] = &api.Mount{
			HostPath:  byteArrayToString(cm.HostPath[:]),
			GuestPath: byteArrayToString(cm.GuestPath[:]),
		}
	}

	return mounts, nil
}
