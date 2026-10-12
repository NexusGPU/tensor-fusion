package device

import (
	"bytes"
	"encoding/json"
	"io"
	"math"
	"os"
	"strings"
	"testing"
	"unsafe"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
)

// FP16 figures as listed by the default NVIDIA ProviderConfig
// (charts/tensor-fusion/values.yaml).
const testHardwareMetadataJSON = `[
	{"model": "A100_SXM_80G", "fullModelName": "NVIDIA A100-SXM4-80GB", "fp16TFlops": "312"},
	{"model": "H100_SXM", "fullModelName": "NVIDIA H100 80GB HBM3", "fp16TFlops": "989"},
	{"model": "L4", "fullModelName": "NVIDIA L4", "fp16TFlops": "121"},
	{"model": "T4", "fullModelName": "Tesla T4", "fp16TFlops": "65"},
	{"model": "RTX3090", "fullModelName": "NVIDIA GeForce RTX 3090", "fp16TFlops": "71.2"},
	{"model": "RTX4090", "fullModelName": "NVIDIA GeForce RTX 4090", "fp16TFlops": "165.16"},
	{"model": "NoFigure", "fullModelName": "NVIDIA No Figure", "fp16TFlops": "0"}
]`

func testHardwareMetadata(t *testing.T) []tfv1.HardwareModelInfo {
	t.Helper()
	var metadata []tfv1.HardwareModelInfo
	require.NoError(t, json.Unmarshal([]byte(testHardwareMetadataJSON), &metadata))
	return metadata
}

// providerDevice is what the provider library reports for one GPU: its
// maxTflops is the provider's own FP32 estimate (CUDA cores x max SM clock x 2).
type providerDevice struct {
	uuid, model string
	index       int32
	maxTflops   float64
	memoryBytes uint64
}

// stubProviderDevices makes the provider library report the given devices
// until the test ends.
func stubProviderDevices(t *testing.T, devices ...providerDevice) {
	t.Helper()
	oldCount, oldAll, oldTopology := getDeviceCount, getAllDevices, getAllDevicesTopology
	t.Cleanup(func() { getDeviceCount, getAllDevices, getAllDevicesTopology = oldCount, oldAll, oldTopology })
	// The anonymous telemetry call is not part of what is tested here.
	t.Setenv("DISABLE_TENSOR_FUSION_TELEMETRY", "1")

	getDeviceCount = func(count *uintptr) Result {
		*count = uintptr(len(devices))
		return ResultSuccess
	}
	getAllDevices = func(out *ExtendedDeviceInfo, maxCount uintptr, count *uintptr) Result {
		infos := (*[64]ExtendedDeviceInfo)(unsafe.Pointer(out))
		n := min(len(devices), int(maxCount))
		for i := range n {
			info := &infos[i]
			*info = ExtendedDeviceInfo{}
			copy(info.Basic.UUID[:], devices[i].uuid)
			copy(info.Basic.Vendor[:], constants.AcceleratorVendorNvidia)
			copy(info.Basic.Model[:], devices[i].model)
			info.Basic.Index = devices[i].index
			info.Basic.MaxTflops = devices[i].maxTflops
			info.Basic.TotalMemoryBytes = devices[i].memoryBytes
		}
		*count = uintptr(n)
		return ResultSuccess
	}
	getAllDevicesTopology = func(*ExtendedDeviceTopology) Result { return ResultErrorNotSupported }
}

func newCapacityTestController() *Controller {
	return &Controller{
		devices:     map[string]*api.DeviceInfo{},
		nativeUUIDs: map[string]string{},
		accelerator: &AcceleratorInterface{},
	}
}

// captureWarnings collects what klog writes until the test ends and returns a
// function counting the lines that contain every given fragment.
func captureWarnings(t *testing.T) func(fragments ...string) int {
	t.Helper()
	var buffer bytes.Buffer
	klog.LogToStderr(false)
	// klog copies a message to the output of every lower severity as well.
	klog.SetOutput(io.Discard)
	klog.SetOutputBySeverity("WARNING", &buffer)
	t.Cleanup(func() {
		klog.Flush()
		klog.SetOutput(os.Stderr)
		klog.LogToStderr(true)
	})
	return func(fragments ...string) int {
		klog.Flush()
		count := 0
	lines:
		for line := range strings.SplitSeq(buffer.String(), "\n") {
			for _, fragment := range fragments {
				if !strings.Contains(line, fragment) {
					continue lines
				}
			}
			count++
		}
		return count
	}
}

const notListedWarning = "has no fp16TFlops entry in the ProviderConfig hardware table"

// A device has one TFLOPS capacity. It is the vendor-published FP16 figure of
// its model when the hardware table lists one, and the provider's estimate
// otherwise, and the device says which of the two it is.
func TestDiscoveryResolvesDeviceTflopsCapacity(t *testing.T) {
	const (
		fromTable    = api.TflopsCapacitySourceProviderConfig
		fromEstimate = api.TflopsCapacitySourceProviderEstimate
	)
	tests := []struct {
		name string
		// model and estimate are what the provider library reports.
		model    string
		estimate float64
		// table is nil when no hardware table is configured at all.
		table        *string
		wantCapacity float64
		wantSource   api.TflopsCapacitySource
		wantWarning  bool
	}{
		{"RTX 3090 listed", "NVIDIA GeForce RTX 3090", 44.08, new(testHardwareMetadataJSON), 71.2, fromTable, false},
		{"RTX 4090 listed", "NVIDIA GeForce RTX 4090", 82.58, new(testHardwareMetadataJSON), 165.16, fromTable, false},
		{"A100 80GB listed", "NVIDIA A100-SXM4-80GB", 19.49184, new(testHardwareMetadataJSON), 312, fromTable, false},
		{"H100 80GB listed", "NVIDIA H100 80GB HBM3", 66.90816, new(testHardwareMetadataJSON), 989, fromTable, false},
		{"L4 listed", "NVIDIA L4", 30.28992, new(testHardwareMetadataJSON), 121, fromTable, false},
		{"T4 listed", "Tesla T4", 8.1408, new(testHardwareMetadataJSON), 65, fromTable, false},
		{"listed by short alias", "RTX3090", 44.08, new(testHardwareMetadataJSON), 71.2, fromTable, false},
		{"listed without vendor prefix", "NVIDIA T4", 8.1408, new(testHardwareMetadataJSON), 65, fromTable, false},
		{"listed in another case", "nvidia a100-sxm4-80gb", 19.49184, new(testHardwareMetadataJSON), 312, fromTable, false},
		{"MIG parent without an estimate", "NVIDIA A100-SXM4-80GB", 0, new(testHardwareMetadataJSON), 312, fromTable, false},
		{"RTX A6000 not listed", "NVIDIA RTX A6000", 38.7072, new(testHardwareMetadataJSON), 38.7072, fromEstimate, true},
		{"entry without a figure", "NVIDIA No Figure", 12.5, new(testHardwareMetadataJSON), 12.5, fromEstimate, true},
		{"not listed and no estimate", "NVIDIA RTX A6000", 0, new(testHardwareMetadataJSON), 0, fromEstimate, true},
		{"empty table", "NVIDIA A100-SXM4-80GB", 19.49184, new("[]"), 19.49184, fromEstimate, true},
		{"RTX 3090 without a table", "NVIDIA GeForce RTX 3090", 44.08, nil, 44.08, fromEstimate, false},
		{"A100 80GB without a table", "NVIDIA A100-SXM4-80GB", 19.49184, nil, 19.49184, fromEstimate, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			warnings := captureWarnings(t)
			stubProviderDevices(t, providerDevice{uuid: "GPU-0", model: tt.model, maxTflops: tt.estimate})
			controller := newCapacityTestController()
			if tt.table != nil {
				t.Setenv(constants.TFProviderHardwareMetadataEnv, *tt.table)
				metadata, err := LoadHardwareMetadataFromEnv()
				require.NoError(t, err)
				controller.SetHardwareMetadata(metadata)
			}

			require.NoError(t, controller.discoverDevices())
			device, exists := controller.GetDevice("gpu-0")
			require.True(t, exists)
			require.InDelta(t, tt.wantCapacity, device.TflopsCapacity, 1e-9)
			require.Equal(t, tt.wantSource, device.TflopsCapacitySource)
			require.Equal(t, tt.wantSource == fromEstimate, device.TflopsCapacityEstimated())
			// The provider's figure stays available, under a name that says what it is.
			require.Equal(t, tt.estimate, device.ProviderEstimatedTflops)
			require.Equal(t, tt.estimate, device.MaxTflops) //nolint:staticcheck // the deprecated mirror is part of the API

			// Every reader sees the same number: the node total is the sum of capacities.
			require.InDelta(t, tt.wantCapacity, controller.AggregateNodeInfo().TotalTFlops, 1e-9)

			wantWarnings := 0
			if tt.wantWarning {
				wantWarnings = 1
			}
			require.Equal(t, wantWarnings, warnings(notListedWarning, "gpu-0"))
		})
	}
}

func TestLoadHardwareMetadataFromEnv(t *testing.T) {
	t.Setenv(constants.TFProviderHardwareMetadataEnv, "")
	_, err := LoadHardwareMetadataFromEnv()
	require.Error(t, err, "an unset table is reported, not taken for an empty one")

	t.Setenv(constants.TFProviderHardwareMetadataEnv, "{not json")
	_, err = LoadHardwareMetadataFromEnv()
	require.Error(t, err)

	t.Setenv(constants.TFProviderHardwareMetadataEnv, "[]")
	metadata, err := LoadHardwareMetadataFromEnv()
	require.NoError(t, err)
	require.Empty(t, metadata)
}

// A node can carry different GPU models; each is resolved on its own, and the
// node total adds up their capacities, never the provider's estimates.
func TestDiscoveryResolvesEachModelOfAMixedNode(t *testing.T) {
	warnings := captureWarnings(t)
	stubProviderDevices(t,
		providerDevice{"GPU-3090", "NVIDIA GeForce RTX 3090", 0, 44.08, 24 << 30},
		providerDevice{"GPU-A100", "NVIDIA A100-SXM4-80GB", 1, 19.49184, 80 << 30},
		providerDevice{"GPU-A6000", "NVIDIA RTX A6000", 2, 38.7072, 48 << 30},
	)
	controller := newCapacityTestController()
	controller.SetHardwareMetadata(testHardwareMetadata(t))
	var completed []*api.NodeInfo
	controller.RegisterDeviceUpdateHandler(framework.DeviceChangeHandler{
		OnDiscoveryComplete: func(nodeInfo *api.NodeInfo) { completed = append(completed, nodeInfo) },
	})

	require.NoError(t, controller.discoverDevices())
	capacities := map[string]float64{}
	for _, device := range controller.GetDevices() {
		capacities[device.UUID] = device.TflopsCapacity
	}
	require.Equal(t, map[string]float64{"gpu-3090": 71.2, "gpu-a100": 312, "gpu-a6000": 38.7072}, capacities)
	require.Len(t, completed, 1)
	require.InDelta(t, 71.2+312+38.7072, completed[0].TotalTFlops, 1e-9)
	require.Equal(t, int64(24+80+48)<<30, completed[0].TotalVRAMBytes)

	// A partition is a slice of a device that is counted already.
	parent, _ := controller.GetDevice("gpu-a100")
	partition := parent.DeepCopy()
	partition.UUID, partition.ParentUUID = "MIG-1", parent.UUID
	controller.mu.Lock()
	controller.devices[partition.UUID] = partition
	nodeInfo := controller.AggregateNodeInfo()
	controller.mu.Unlock()
	require.InDelta(t, 71.2+312+38.7072, nodeInfo.TotalTFlops, 1e-9)
	require.Len(t, nodeInfo.DeviceIDs, 3)

	// Only the device missing from the table is reported, once, also when
	// discovery runs again.
	controller.mu.Lock()
	delete(controller.devices, partition.UUID)
	controller.mu.Unlock()
	require.NoError(t, controller.discoverDevices())
	require.NoError(t, controller.discoverDevices())
	require.Equal(t, 1, warnings(notListedWarning))
	require.Equal(t, 1, warnings(notListedWarning, "gpu-a6000"))
}

// The hardware table can change while the hypervisor runs. Devices are
// re-resolved against the new table at once and everything that reads their
// capacity is told.
func TestHardwareTableChangeReResolvesKnownDevices(t *testing.T) {
	warnings := captureWarnings(t)
	stubProviderDevices(t,
		providerDevice{uuid: "GPU-3090", model: "NVIDIA GeForce RTX 3090", index: 0, maxTflops: 44.08},
		providerDevice{uuid: "GPU-T4", model: "Tesla T4", index: 1, maxTflops: 8.1408},
	)
	controller := newCapacityTestController()
	// The table starts without the RTX 3090.
	withoutRTX3090 := []tfv1.HardwareModelInfo{}
	for _, hw := range testHardwareMetadata(t) {
		if hw.Model != "RTX3090" {
			withoutRTX3090 = append(withoutRTX3090, hw)
		}
	}
	controller.SetHardwareMetadata(withoutRTX3090)

	type update struct{ old, new *api.DeviceInfo }
	var updates []update
	var totals []float64
	controller.RegisterDeviceUpdateHandler(framework.DeviceChangeHandler{
		OnUpdate: func(oldDevice, newDevice *api.DeviceInfo) { updates = append(updates, update{oldDevice, newDevice}) },
		OnDiscoveryComplete: func(nodeInfo *api.NodeInfo) {
			totals = append(totals, nodeInfo.TotalTFlops)
		},
	})
	require.NoError(t, controller.discoverDevices())
	estimated, _ := controller.GetDevice("gpu-3090")
	require.Equal(t, 44.08, estimated.TflopsCapacity)
	require.True(t, estimated.TflopsCapacityEstimated())
	require.Equal(t, 1, warnings(notListedWarning, "gpu-3090"))
	require.Equal(t, []float64{44.08 + 65}, totals)
	// A partition created from the device carries the same capacity.
	partition := estimated.DeepCopy()
	partition.UUID, partition.ParentUUID = "MIG-3090", estimated.UUID
	controller.mu.Lock()
	controller.devices[partition.UUID] = partition
	controller.mu.Unlock()

	// The ProviderConfig gains the model.
	controller.SetHardwareMetadata(testHardwareMetadata(t))
	listed, _ := controller.GetDevice("gpu-3090")
	require.Equal(t, 71.2, listed.TflopsCapacity)
	require.Equal(t, api.TflopsCapacitySourceProviderConfig, listed.TflopsCapacitySource)
	require.Equal(t, 44.08, listed.ProviderEstimatedTflops)
	// Only the device that changed is announced, and the device handed out
	// earlier is replaced rather than modified under its readers.
	require.Len(t, updates, 1)
	require.Same(t, estimated, updates[0].old)
	require.Same(t, listed, updates[0].new)
	require.Equal(t, 44.08, estimated.TflopsCapacity)
	require.Equal(t, []float64{44.08 + 65, 71.2 + 65}, totals)
	resolvedPartition, _ := controller.GetDevice("MIG-3090")
	require.Equal(t, 71.2, resolvedPartition.TflopsCapacity, "a partition follows its parent's model")
	require.NotSame(t, partition, resolvedPartition)

	// Installing the same table again changes nothing and announces nothing.
	controller.SetHardwareMetadata(testHardwareMetadata(t))
	require.Len(t, updates, 1)
	require.Len(t, totals, 2)

	// The figure of a listed model is corrected.
	corrected := testHardwareMetadata(t)
	for i := range corrected {
		if corrected[i].Model == "T4" {
			require.NoError(t, json.Unmarshal([]byte(`"64.5"`), &corrected[i].Fp16TFlops))
		}
	}
	controller.SetHardwareMetadata(corrected)
	t4, _ := controller.GetDevice("gpu-t4")
	require.Equal(t, 64.5, t4.TflopsCapacity)
	require.Len(t, updates, 2)
	require.Equal(t, 71.2+64.5, totals[len(totals)-1])

	// The model is dropped from the table again: the fallback is visible and
	// reported again, once.
	controller.SetHardwareMetadata(withoutRTX3090)
	fallenBack, _ := controller.GetDevice("gpu-3090")
	require.Equal(t, 44.08, fallenBack.TflopsCapacity)
	require.True(t, fallenBack.TflopsCapacityEstimated())
	require.Equal(t, 2, warnings(notListedWarning, "gpu-3090"))
	require.NoError(t, controller.discoverDevices())
	require.Equal(t, 2, warnings(notListedWarning, "gpu-3090"))
	require.Equal(t, 0, warnings(notListedWarning, "MIG-3090"), "a partition is not reported on its own")
}

// What a GPU resource carries is the capacity with six decimals. The stored
// capacity is exactly what a reader of that resource gets back.
func TestTflopsCapacityIsStoredAsPublished(t *testing.T) {
	for _, tflops := range []float64{0, 8.1408, 19.49184, 44.08, 65, 71.2, 165.16, 312, 989, 19.4918399999, 1.0 / 3} {
		device := &api.DeviceInfo{ProviderEstimatedTflops: tflops}
		device.ResolveTflopsCapacity(0, false)
		published := api.TflopsQuantity(device.TflopsCapacity)
		raw, err := json.Marshal(published)
		require.NoError(t, err)
		var read resource.Quantity
		require.NoError(t, json.Unmarshal(raw, &read))
		require.Equal(t, device.TflopsCapacity, read.AsApproximateFloat64(), "capacity %v", tflops)
		require.InDelta(t, tflops, device.TflopsCapacity, 5e-7)
	}
	for _, invalid := range []float64{-1, math.NaN(), math.Inf(1)} {
		require.Zero(t, api.NormalizeTflops(invalid))
	}
	// A DeviceInfo built by a caller that predates ProviderEstimatedTflops.
	legacy := &api.DeviceInfo{MaxTflops: 40} //nolint:staticcheck // deprecated input alias
	require.True(t, legacy.ResolveTflopsCapacity(0, false))
	require.Equal(t, 40.0, legacy.TflopsCapacity)
	require.False(t, legacy.ResolveTflopsCapacity(0, false), "resolving again changes nothing")
	require.True(t, legacy.ResolveTflopsCapacity(71.2, true))
	require.Equal(t, 71.2, legacy.TflopsCapacity)
}
