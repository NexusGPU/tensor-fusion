package worker

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/server/handlers"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
)

// Byte offsets of the limit fields in the soft limiter mapping. They are part
// of the ABI shared with the in-process limiter: an 8-byte enum header, then
// 136-byte device entries holding a 64-byte UUID, up_limit (u32) and, after
// 4 bytes of padding, mem_limit (u64).
const (
	shmABIHeaderSize      = 8
	shmABIDeviceEntrySize = 136
	shmABIUpLimitOffset   = 64
	shmABIMemLimitOffset  = 72
)

const gib = uint64(1) << 30

// limitTestBackend is a worker backend that authenticates a Pod token.
type limitTestBackend struct {
	fakeWorkerBackend
	podUID string
}

// discoveredDevice builds a device the way the device controller hands it out:
// with the provider's estimate and with its TFLOPS capacity resolved. fp16Tflops
// is the hardware table's figure for the model, zero when it lists none.
func discoveredDevice(uuid, model string, index int32, estimate, fp16Tflops float64) *api.DeviceInfo {
	device := &api.DeviceInfo{
		UUID: uuid, Vendor: constants.AcceleratorVendorNvidia, Model: model, Index: index,
		TotalMemoryBytes: 80 * gib, ProviderEstimatedTflops: estimate,
	}
	device.ResolveTflopsCapacity(fp16Tflops, fp16Tflops > 0)
	return device
}

func (b *limitTestBackend) AuthenticatePodToken(context.Context, string) (string, error) {
	return b.podUID, nil
}

func newLimitTestController(
	t *testing.T, backend framework.Backend, devices ...*api.DeviceInfo,
) (*WorkerController, *AllocationController) {
	t.Helper()
	deviceController := &fakeDeviceController{devices: map[string]*api.DeviceInfo{}}
	for _, device := range devices {
		deviceController.devices[device.UUID] = device
	}
	allocations := NewAllocationController(deviceController)
	controller := NewWorkerControllerWithPolicy(
		deviceController, allocations, tfv1.IsolationModeSoft, tfv1.IsolationModePolicyStatic, backend,
	).(*WorkerController)
	controller.shmBasePath = t.TempDir()
	t.Cleanup(func() {
		controller.mu.Lock()
		defer controller.mu.Unlock()
		for uid := range controller.shmHandles {
			controller.closeSharedMemoryLocked(uid)
		}
	})
	return controller, allocations
}

func newLimitTestWorker(mode tfv1.IsolationModeType, limits tfv1.Resource, devices ...string) *api.WorkerInfo {
	return &api.WorkerInfo{
		WorkerUID: "worker-uid", Namespace: "default", WorkerName: "worker",
		Status: api.WorkerStatusRunning, IsolationMode: mode, QoS: tfv1.QoSMedium,
		AllocatedDevices: devices, Limits: limits,
	}
}

// readMappedLimits reads the limit fields from the published file by ABI
// offset, independently of the hypervisor's own mapping and struct layout.
func readMappedLimits(t *testing.T, w *WorkerController, worker *api.WorkerInfo, deviceIdx int) (uint32, uint64) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join(w.shmBasePath, worker.Namespace, worker.WorkerName, workerstate.ShmPathSuffix))
	require.NoError(t, err)
	entry := shmABIHeaderSize + deviceIdx*shmABIDeviceEntrySize
	require.Greater(t, len(raw), entry+shmABIDeviceEntrySize)
	return binary.LittleEndian.Uint32(raw[entry+shmABIUpLimitOffset:]),
		binary.LittleEndian.Uint64(raw[entry+shmABIMemLimitOffset:])
}

// openLimiterView maps the published file a second time, the way a limiter
// process inside the worker does.
func openLimiterView(t *testing.T, w *WorkerController, worker *api.WorkerInfo) *workerstate.SharedMemoryHandle {
	t.Helper()
	view, err := workerstate.OpenSharedMemoryHandle(
		w.shmBasePath, workerstate.NewPodIdentifier(worker.Namespace, worker.WorkerName),
	)
	require.NoError(t, err)
	t.Cleanup(func() { _ = view.Close() })
	return view
}

func loadLimiterView(view *workerstate.SharedMemoryHandle, deviceIdx int) (upLimit uint32, memLimit uint64) {
	view.WithState(func(state *workerstate.SharedDeviceState) {
		info := &state.V2.Devices[deviceIdx].DeviceInfo
		upLimit, memLimit = atomic.LoadUint32(&info.UpLimit), atomic.LoadUint64(&info.MemLimit)
	})
	return upLimit, memLimit
}

func snapshotUpLimit(t *testing.T, w *WorkerController, worker *api.WorkerInfo, deviceIdx int) uint32 {
	t.Helper()
	snapshot := w.buildWorkerInfoSnapshots()[worker.WorkerUID]
	require.NotNil(t, snapshot)
	for _, device := range snapshot.Devices {
		if device.DeviceIdx == deviceIdx {
			return device.UpLimit
		}
	}
	t.Fatalf("device %d is missing from the quota snapshot", deviceIdx)
	return 0
}

// A limit expressed in TFLOPS is a share of the device's TFLOPS capacity, which
// is the vendor-published FP16 figure when the hardware table lists the model.
// The provider library reports an FP32 estimate (CUDA cores x max SM clock x 2)
// that differs from it by up to an order of magnitude; it only decides when the
// model is not listed.
func TestSoftComputeLimitUsesDeviceTflopsCapacity(t *testing.T) {
	tests := []struct {
		name string
		// model and estimate are what the provider library reports.
		model    string
		estimate float64
		// fp16Tflops is the hardware table's figure; zero means not listed.
		fp16Tflops  float64
		limitTflops string
		want        uint32
	}{
		{"RTX 3090 listed", "NVIDIA GeForce RTX 3090", 44.08, 71.2, "40", 57},
		{"RTX 3090 not listed", "NVIDIA GeForce RTX 3090", 44.08, 0, "40", 91},
		{"A100 80GB listed", "NVIDIA A100-SXM4-80GB", 19.49184, 312, "100", 33},
		{"A100 80GB listed, small limit", "NVIDIA A100-SXM4-80GB", 19.49184, 312, "10", 4},
		{"A100 80GB listed, whole card", "NVIDIA A100-SXM4-80GB", 19.49184, 312, "312", 100},
		{"A100 80GB listed, exact quarter", "NVIDIA A100-SXM4-80GB", 19.49184, 312, "78", 25},
		{"A100 80GB listed, tiny limit", "NVIDIA A100-SXM4-80GB", 19.49184, 312, "0.5", 1},
		{"A100 80GB not listed", "NVIDIA A100-SXM4-80GB", 19.49184, 0, "10", 52},
		{"T4 listed", "Tesla T4", 8.1408, 65, "20", 31},
		{"T4 not listed", "Tesla T4", 8.1408, 0, "4", 50},
		{"T4 not listed, limit above the estimate", "Tesla T4", 8.1408, 0, "100", 100},
		{"H100 80GB listed", "NVIDIA H100 80GB HBM3", 66.90816, 989, "100", 11},
		{"L4 listed", "NVIDIA L4", 30.28992, 121, "40", 34},
		{"RTX 4090 listed", "NVIDIA GeForce RTX 4090", 82.58, 165.16, "82.58", 50},
		{"MIG parent listed, no estimate", "NVIDIA A100-SXM4-80GB", 0, 312, "100", 33},
		{"no capacity known at all", "NVIDIA RTX A6000", 0, 0, "10", 100},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			device := discoveredDevice("gpu-0", tt.model, 0, tt.estimate, tt.fp16Tflops)
			w, _ := newLimitTestController(t, &fakeWorkerBackend{}, device)
			worker := newLimitTestWorker(
				tfv1.IsolationModeSoft, tfv1.Resource{Tflops: resource.MustParse(tt.limitTflops)}, device.UUID,
			)
			w.workerChangeHandler().OnAdd(worker)

			upLimit, _ := readMappedLimits(t, w, worker, 0)
			require.Equal(t, tt.want, upLimit, "up_limit published to the limiter")
			require.Equal(t, tt.want, snapshotUpLimit(t, w, worker, 0), "quota controller target")
		})
	}
}

// A worker can hold GPUs of different models. Its TFLOPS limit is the same on
// each of them, so the share differs per device.
func TestSoftComputeLimitIsConvertedPerDeviceModel(t *testing.T) {
	rtx3090 := discoveredDevice("gpu-0", "NVIDIA GeForce RTX 3090", 0, 44.08, 71.2)
	a100 := discoveredDevice("gpu-1", "NVIDIA A100-SXM4-80GB", 1, 19.49184, 312)
	unlisted := discoveredDevice("gpu-2", "NVIDIA RTX A6000", 2, 38.7072, 0)
	w, _ := newLimitTestController(t, &fakeWorkerBackend{}, rtx3090, a100, unlisted)
	worker := newLimitTestWorker(
		tfv1.IsolationModeSoft, vramAndTflops("8Gi", "40"), rtx3090.UUID, a100.UUID, unlisted.UUID,
	)
	handler := w.workerChangeHandler()
	handler.OnAdd(worker)

	view := openLimiterView(t, w, worker)
	for deviceIdx, want := range map[int]uint32{0: 57, 1: 13, 2: 100} {
		upLimit, memLimit := readMappedLimits(t, w, worker, deviceIdx)
		require.Equal(t, want, upLimit, "device %d", deviceIdx)
		require.Equal(t, 8*gib, memLimit, "device %d", deviceIdx)
		require.Equal(t, want, snapshotUpLimit(t, w, worker, deviceIdx), "quota controller target of device %d", deviceIdx)
	}

	// An in-place change is converted per device as well.
	updated := *worker
	updated.Limits = vramAndTflops("8Gi", "20")
	handler.OnUpdate(worker, &updated)
	for deviceIdx, want := range map[int]uint32{0: 29, 1: 7, 2: 52} {
		upLimit, _ := readMappedLimits(t, w, &updated, deviceIdx)
		require.Equal(t, want, upLimit, "device %d after the update", deviceIdx)
		upLimit, _ = loadLimiterView(view, deviceIdx)
		require.Equal(t, want, upLimit, "device %d as seen by the running limiter", deviceIdx)
	}
}

// When the hardware table changes, the device controller re-resolves the
// device and announces it. Workers already running on the device get the limit
// that follows from the new capacity, in place.
func TestDeviceCapacityChangeReachesRunningWorkers(t *testing.T) {
	// The model is not listed yet: the capacity is the provider's estimate.
	estimated := discoveredDevice("gpu-0", "NVIDIA GeForce RTX 3090", 0, 44.08, 0)
	other := discoveredDevice("gpu-1", "NVIDIA A100-SXM4-80GB", 1, 19.49184, 312)
	w, _ := newLimitTestController(t, &fakeWorkerBackend{}, estimated, other)
	devices := w.deviceController.(*fakeDeviceController)
	handler := w.workerChangeHandler()

	onDevice := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("8Gi", "40"), estimated.UUID)
	elsewhere := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("8Gi", "40"), other.UUID)
	elsewhere.WorkerUID, elsewhere.WorkerName = "elsewhere-uid", "elsewhere"
	percentLimited := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndPercent("8Gi", "40"), estimated.UUID)
	percentLimited.WorkerUID, percentLimited.WorkerName = "percent-uid", "percent"
	for _, worker := range []*api.WorkerInfo{onDevice, elsewhere, percentLimited} {
		handler.OnAdd(worker)
	}
	upLimit, _ := readMappedLimits(t, w, onDevice, 0)
	require.Equal(t, uint32(91), upLimit)
	view := openLimiterView(t, w, onDevice)

	// ProviderConfig gains the model: the device is replaced by a re-resolved one.
	listed := discoveredDevice("gpu-0", "NVIDIA GeForce RTX 3090", 0, 44.08, 71.2)
	devices.devices[listed.UUID] = listed
	w.onDeviceUpdate(estimated, listed)

	upLimit, memLimit := loadLimiterView(view, 0)
	require.Equal(t, uint32(57), upLimit, "up_limit seen by the running limiter")
	require.Equal(t, 8*gib, memLimit)
	require.Equal(t, uint32(57), snapshotUpLimit(t, w, onDevice, 0), "quota controller target")
	upLimit, _ = readMappedLimits(t, w, elsewhere, 1)
	require.Equal(t, uint32(13), upLimit, "a worker on another device is not affected")
	upLimit, _ = readMappedLimits(t, w, percentLimited, 0)
	require.Equal(t, uint32(40), upLimit, "a percentage limit does not depend on the capacity")

	// A mapping opened after the change starts from the new capacity too, even
	// though the allocation still holds the device as it was when admitted.
	late := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("8Gi", "20"), listed.UUID)
	late.WorkerUID, late.WorkerName = "late-uid", "late"
	handler.OnAdd(late)
	upLimit, _ = readMappedLimits(t, w, late, 0)
	require.Equal(t, uint32(29), upLimit)

	// An update that does not change the capacity leaves the limits alone.
	relabeled := *listed
	relabeled.Healthy = true
	devices.devices[listed.UUID] = &relabeled
	w.onDeviceUpdate(listed, &relabeled)
	upLimit, _ = loadLimiterView(view, 0)
	require.Equal(t, uint32(57), upLimit)
}

func TestHardComputeLimitUsesDeviceTflopsCapacity(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		limits      tfv1.Resource
		want        string
	}{
		{
			name:   "scheduler without effective percent annotation",
			limits: tfv1.Resource{Tflops: resource.MustParse("100")}, want: "33",
		},
		{
			name:        "scheduler annotation is preferred",
			annotations: map[string]string{constants.EffectiveHardSMPercentAnnotation: "40"},
			limits:      tfv1.Resource{Tflops: resource.MustParse("100")}, want: "40",
		},
		{
			name:        "invalid scheduler annotation is ignored",
			annotations: map[string]string{constants.EffectiveHardSMPercentAnnotation: "400"},
			limits:      tfv1.Resource{Tflops: resource.MustParse("100")}, want: "33",
		},
		{
			name:   "percentage limit",
			limits: tfv1.Resource{ComputePercent: resource.MustParse("25")}, want: "25",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, allocations := newLimitTestController(t, &fakeWorkerBackend{}, testA100())
			worker := newLimitTestWorker(tfv1.IsolationModeHard, tt.limits, "gpu-0")
			worker.Annotations = tt.annotations
			allocation, err := allocations.AllocateWorkerDevices(worker)
			require.NoError(t, err)
			require.Equal(t, tt.want, allocation.Envs[constants.HardSMLimiterEnv])

			// The legacy client's mapping carries the same percentage as its env.
			w.workerChangeHandler().OnAdd(worker)
			upLimit, _ := readMappedLimits(t, w, worker, 0)
			require.Equal(t, tt.want, strconv.FormatUint(uint64(upLimit), 10))
		})
	}
}

// The scheduler's percentage only applies to hard isolation: a soft worker's
// limit can change in place, after the scheduler wrote it.
func TestSoftComputeLimitIgnoresHardPercentAnnotation(t *testing.T) {
	w, _ := newLimitTestController(t, &fakeWorkerBackend{}, testA100())
	worker := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("8Gi", "100"), "gpu-0")
	worker.Annotations = map[string]string{constants.EffectiveHardSMPercentAnnotation: "40"}
	w.workerChangeHandler().OnAdd(worker)
	upLimit, _ := readMappedLimits(t, w, worker, 0)
	require.Equal(t, uint32(33), upLimit)
}

// A soft worker's share is derived in the hypervisor, a hard worker's by the
// scheduler from the GPU resource. Both use one conversion on one capacity, so
// the same limit on the same GPU is the same percentage in either mode.
func TestSoftPercentMatchesTheSchedulersConversion(t *testing.T) {
	for _, device := range []*api.DeviceInfo{
		discoveredDevice("gpu-0", "NVIDIA GeForce RTX 3090", 0, 44.08, 71.2),
		discoveredDevice("gpu-0", "NVIDIA GeForce RTX 3090", 0, 44.08, 0),
		discoveredDevice("gpu-0", "NVIDIA A100-SXM4-80GB", 0, 19.49184, 312),
		discoveredDevice("gpu-0", "NVIDIA RTX A6000", 0, 38.7072, 0),
		discoveredDevice("gpu-0", "NVIDIA GeForce RTX 4090", 0, 82.58, 165.16),
	} {
		// What the scheduler reads from the GPU resource.
		published := api.TflopsQuantity(device.TflopsCapacity)
		for step := 1; step <= 240; step++ {
			limit := resource.MustParse(strconv.FormatFloat(device.TflopsCapacity*float64(step)/200, 'f', 6, 64))
			scheduled, ok := utils.TflopsToComputePercent(limit.AsApproximateFloat64(), published.AsApproximateFloat64())
			require.True(t, ok)
			soft := newLimitTestWorker(tfv1.IsolationModeSoft, tfv1.Resource{Tflops: limit}, device.UUID)
			require.Equal(t, scheduled, computeUpLimit(soft, device), "%s TFLOPS of %v", limit.String(), device.TflopsCapacity)
			// A hard worker placed by a scheduler that recorded the percentage
			// gets that one, and without it the same result.
			hard := newLimitTestWorker(tfv1.IsolationModeHard, tfv1.Resource{Tflops: limit}, device.UUID)
			require.Equal(t, scheduled, computeUpLimit(hard, device))
			hard.Annotations = map[string]string{
				constants.EffectiveHardSMPercentAnnotation: strconv.FormatUint(uint64(scheduled), 10),
			}
			require.Equal(t, scheduled, computeUpLimit(hard, device))
		}
	}
}

func testA100() *api.DeviceInfo {
	return discoveredDevice("gpu-0", "NVIDIA A100-SXM4-80GB", 0, 19.49184, 312)
}

func testA100Backend() *limitTestBackend {
	return &limitTestBackend{podUID: "worker-uid"}
}

func vramAndPercent(vram string, percent string) tfv1.Resource {
	return tfv1.Resource{Vram: resource.MustParse(vram), ComputePercent: resource.MustParse(percent)}
}

func vramAndTflops(vram string, tflops string) tfv1.Resource {
	return tfv1.Resource{Vram: resource.MustParse(vram), Tflops: resource.MustParse(tflops)}
}

// Limits of a running worker can be adjusted in place. The backend reports the
// change through OnUpdate; it must reach the mapping the limiter already holds.
func TestWorkerLimitUpdateReachesSharedMemory(t *testing.T) {
	tests := []struct {
		name           string
		from, to       tfv1.Resource
		fromUp, toUp   uint32
		fromMem, toMem uint64
	}{
		{
			name: "vram scale down", from: vramAndPercent("20Gi", "50"), to: vramAndPercent("8Gi", "50"),
			fromUp: 50, toUp: 50, fromMem: 20 * gib, toMem: 8 * gib,
		},
		{
			name: "vram scale up", from: vramAndPercent("8Gi", "50"), to: vramAndPercent("20Gi", "50"),
			fromUp: 50, toUp: 50, fromMem: 8 * gib, toMem: 20 * gib,
		},
		{
			name: "compute percent scale down", from: vramAndPercent("8Gi", "80"), to: vramAndPercent("8Gi", "30"),
			fromUp: 80, toUp: 30, fromMem: 8 * gib, toMem: 8 * gib,
		},
		{
			name: "compute percent scale up", from: vramAndPercent("8Gi", "30"), to: vramAndPercent("8Gi", "80"),
			fromUp: 30, toUp: 80, fromMem: 8 * gib, toMem: 8 * gib,
		},
		{
			name: "tflops scale down", from: vramAndTflops("8Gi", "156"), to: vramAndTflops("8Gi", "78"),
			fromUp: 50, toUp: 25, fromMem: 8 * gib, toMem: 8 * gib,
		},
		{
			name: "tflops and vram scale up", from: vramAndTflops("8Gi", "78"), to: vramAndTflops("16Gi", "156"),
			fromUp: 25, toUp: 50, fromMem: 8 * gib, toMem: 16 * gib,
		},
		{
			name: "unlimited worker becomes limited", from: tfv1.Resource{}, to: vramAndPercent("10Gi", "40"),
			fromUp: 100, toUp: 40, fromMem: 80 * gib, toMem: 10 * gib,
		},
		{
			name: "limited worker becomes unlimited", from: vramAndPercent("10Gi", "40"), to: tfv1.Resource{},
			fromUp: 40, toUp: 100, fromMem: 10 * gib, toMem: 80 * gib,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, allocations := newLimitTestController(t, testA100Backend(), testA100())
			handler := w.workerChangeHandler()
			worker := newLimitTestWorker(tfv1.IsolationModeSoft, tt.from, "gpu-0")
			handler.OnAdd(worker)

			upLimit, memLimit := readMappedLimits(t, w, worker, 0)
			require.Equal(t, tt.fromUp, upLimit)
			require.Equal(t, tt.fromMem, memLimit)

			view := openLimiterView(t, w, worker)
			shmPath := filepath.Join(w.shmBasePath, worker.Namespace, worker.WorkerName, workerstate.ShmPathSuffix)
			before, err := os.Stat(shmPath)
			require.NoError(t, err)
			require.NoError(t, w.WithWorkerSharedMemory(worker.WorkerUID, func(state *workerstate.SharedDeviceState) {
				state.SetPodMemoryUsed(0, 3*gib)
				state.AddPID(4242)
				state.V2.Devices[0].DeviceInfo.SetERLCurrentTokens(17)
			}))

			updated := *worker
			updated.Limits = tt.to
			handler.OnUpdate(worker, &updated)

			upLimit, memLimit = readMappedLimits(t, w, &updated, 0)
			require.Equal(t, tt.toUp, upLimit, "up_limit bytes in the published file")
			require.Equal(t, tt.toMem, memLimit, "mem_limit bytes in the published file")
			upLimit, memLimit = loadLimiterView(view, 0)
			require.Equal(t, tt.toUp, upLimit, "up_limit seen by an existing limiter mapping")
			require.Equal(t, tt.toMem, memLimit, "mem_limit seen by an existing limiter mapping")
			require.Equal(t, tt.toUp, snapshotUpLimit(t, w, &updated, 0), "quota controller target")

			// The update is in place: same inode, and no other limiter state is reset.
			after, err := os.Stat(shmPath)
			require.NoError(t, err)
			require.True(t, os.SameFile(before, after), "the mapping must not be recreated")
			view.WithState(func(state *workerstate.SharedDeviceState) {
				require.Equal(t, 3*gib, atomic.LoadUint64(&state.V2.Devices[0].DeviceInfo.PodMemoryUsed))
				require.Equal(t, []int{4242}, state.GetAllPIDs())
				require.Equal(t, float64(17), state.V2.Devices[0].DeviceInfo.GetERLCurrentTokens())
				require.True(t, state.HasDevice(0))
			})

			allocation, exists := allocations.GetWorkerAllocation(worker.WorkerUID)
			require.True(t, exists)
			require.True(t, allocation.WorkerInfo.Limits.Vram.Equal(tt.to.Vram), "allocation vram limit")
			require.True(t, allocation.WorkerInfo.Limits.Tflops.Equal(tt.to.Tflops), "allocation tflops limit")
			require.True(t, allocation.WorkerInfo.Limits.ComputePercent.Equal(tt.to.ComputePercent),
				"allocation compute percent limit")
		})
	}
}

func TestWorkerLimitUpdateCoversEveryAllocatedDevice(t *testing.T) {
	first, second := testA100(), testA100()
	second.UUID, second.Index = "gpu-3", 3
	w, _ := newLimitTestController(t, testA100Backend(), first, second)
	handler := w.workerChangeHandler()
	worker := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("20Gi", "156"), first.UUID, second.UUID)
	handler.OnAdd(worker)

	updated := *worker
	updated.Limits = vramAndTflops("6Gi", "78")
	handler.OnUpdate(worker, &updated)

	for _, deviceIdx := range []int{0, 3} {
		upLimit, memLimit := readMappedLimits(t, w, &updated, deviceIdx)
		require.Equal(t, uint32(25), upLimit, "device %d", deviceIdx)
		require.Equal(t, 6*gib, memLimit, "device %d", deviceIdx)
	}
	// Slots the worker was not allocated stay inactive and untouched.
	upLimit, memLimit := readMappedLimits(t, w, &updated, 1)
	require.Zero(t, upLimit)
	require.Zero(t, memLimit)
}

func getJSON(t *testing.T, handle gin.HandlerFunc, target string, authorized bool, out any) {
	t.Helper()
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	ctx.Request = httptest.NewRequest(http.MethodGet, target, nil)
	if authorized {
		ctx.Request.Header.Set(constants.AuthorizationHeader, "Bearer token")
	}
	handle(ctx)
	require.Equal(t, http.StatusOK, recorder.Code)
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), out))
}

func TestWorkerLimitUpdateIsServedOverHTTP(t *testing.T) {
	gin.SetMode(gin.TestMode)
	backend := testA100Backend()
	w, allocations := newLimitTestController(t, backend, testA100())
	handler := w.workerChangeHandler()
	worker := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("20Gi", "156"), "gpu-0")
	handler.OnAdd(worker)
	legacy := handlers.NewLegacyHandler(w, allocations, backend, nil)

	assertServed := func(vram uint64, tflops float64) {
		t.Helper()
		var list api.ListPodsResponse
		getJSON(t, legacy.HandleGetPods, "/api/v1/pod", false, &list)
		require.Len(t, list.Pods, 1)
		require.NotNil(t, list.Pods[0].VramLimit)
		require.NotNil(t, list.Pods[0].TflopsLimit)
		require.Equal(t, vram, *list.Pods[0].VramLimit, "GET /api/v1/pod vram_limit")
		require.Equal(t, tflops, *list.Pods[0].TflopsLimit, "GET /api/v1/pod tflops_limit")

		var pod api.PodInfoResponse
		getJSON(t, legacy.HandleGetPods, "/api/v1/pod?container_name=main", true, &pod)
		require.True(t, pod.Success, pod.Message)
		require.NotNil(t, pod.Data.VramLimit)
		require.NotNil(t, pod.Data.TflopsLimit)
		require.Equal(t, vram, *pod.Data.VramLimit, "authenticated GET /api/v1/pod vram_limit")
		require.Equal(t, tflops, *pod.Data.TflopsLimit, "authenticated GET /api/v1/pod tflops_limit")

		var limiters api.ListLimitersResponse
		getJSON(t, legacy.HandleGetLimiter, "/api/v1/limiter", false, &limiters)
		require.Len(t, limiters.Limiters, 1)
		require.Equal(t, int64(vram), limiters.Limiters[0].Limits.Vram.Value(), "GET /api/v1/limiter vram")
		require.Equal(t, tflops, limiters.Limiters[0].Limits.Tflops.AsApproximateFloat64(),
			"GET /api/v1/limiter tflops")
	}
	assertServed(20*gib, 156)

	scaledDown := *worker
	scaledDown.Limits = vramAndTflops("8Gi", "78")
	handler.OnUpdate(worker, &scaledDown)
	assertServed(8*gib, 78)

	scaledUp := scaledDown
	scaledUp.Limits = vramAndTflops("32Gi", "234")
	handler.OnUpdate(&scaledDown, &scaledUp)
	assertServed(32*gib, 234)
}

// A mapping outlives the hypervisor process. After a restart or an upgrade the
// existing file is reopened, not recreated, so its limits are brought in line
// with the worker's current ones at the moment it is opened.
func TestReopenedMappingGetsCurrentLimits(t *testing.T) {
	tests := []struct {
		name    string
		stale   workerstate.DeviceConfig
		limits  tfv1.Resource
		wantUp  uint32
		wantMem uint64
	}{
		{
			name:  "limits changed while the hypervisor was down",
			stale: workerstate.DeviceConfig{UpLimit: 50, MemLimit: 20 * gib}, limits: vramAndPercent("8Gi", "30"),
			wantUp: 30, wantMem: 8 * gib,
		},
		{
			name:  "mapping created by a release that divided by the provider's estimate",
			stale: workerstate.DeviceConfig{UpLimit: 100, MemLimit: 8 * gib}, limits: vramAndTflops("8Gi", "100"),
			wantUp: 33, wantMem: 8 * gib,
		},
		{
			name:  "limits already current",
			stale: workerstate.DeviceConfig{UpLimit: 33, MemLimit: 8 * gib}, limits: vramAndTflops("8Gi", "100"),
			wantUp: 33, wantMem: 8 * gib,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w, _ := newLimitTestController(t, testA100Backend(), testA100())
			worker := newLimitTestWorker(tfv1.IsolationModeSoft, tt.limits, "gpu-0")
			tt.stale.DeviceIdx, tt.stale.DeviceUUID = 0, "0"
			previous, err := workerstate.PrepareWorkerSharedMemory(
				w.shmBasePath, workerstate.NewPodIdentifier(worker.Namespace, worker.WorkerName),
				worker.WorkerUID, []workerstate.DeviceConfig{tt.stale}, true,
			)
			require.NoError(t, err)
			previous.WithState(func(state *workerstate.SharedDeviceState) {
				state.AddPID(4242)
				state.V2.Devices[0].DeviceInfo.SetERLCurrentTokens(17)
			})
			shmPath := filepath.Join(w.shmBasePath, worker.Namespace, worker.WorkerName, workerstate.ShmPathSuffix)
			before, err := os.Stat(shmPath)
			require.NoError(t, err)
			require.NoError(t, previous.Close())

			// No sync pass is involved: opening the mapping is enough.
			w.workerChangeHandler().OnAdd(worker)

			upLimit, memLimit := readMappedLimits(t, w, worker, 0)
			require.Equal(t, tt.wantUp, upLimit)
			require.Equal(t, tt.wantMem, memLimit)
			after, err := os.Stat(shmPath)
			require.NoError(t, err)
			require.True(t, os.SameFile(before, after), "the mapping must be reopened, not recreated")
			require.NoError(t, w.WithWorkerSharedMemory(worker.WorkerUID, func(state *workerstate.SharedDeviceState) {
				require.Equal(t, []int{4242}, state.GetAllPIDs())
				require.Equal(t, float64(17), state.V2.Devices[0].DeviceInfo.GetERLCurrentTokens())
			}))
		})
	}
}

// The refill rate a control loop left in a mapping was published for the limit
// the mapping carried then. The limit found when the mapping is opened, before
// the current one is written over it, is handed to the quota controller with
// the worker's snapshot, and stays what was found however often limits are
// applied afterwards.
func TestLimitFoundInAReopenedMappingReachesTheQuotaSnapshot(t *testing.T) {
	found := func(t *testing.T, w *WorkerController, worker *api.WorkerInfo) (foundUpLimit, upLimit uint32) {
		t.Helper()
		snapshot := w.buildWorkerInfoSnapshots()[worker.WorkerUID]
		require.NotNil(t, snapshot)
		require.Len(t, snapshot.Devices, 1)
		return snapshot.Devices[0].FoundUpLimit, snapshot.Devices[0].UpLimit
	}
	for _, tc := range []struct {
		name      string
		stale     uint32
		wantFound uint32
	}{
		{name: "limit lowered while the hypervisor was down", stale: 50, wantFound: 50},
		{name: "limit raised while the hypervisor was down", stale: 20, wantFound: 20},
		{name: "limit unchanged", stale: 30, wantFound: 30},
		{name: "a value that is not a percentage is not known", stale: 250, wantFound: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			w, _ := newLimitTestController(t, testA100Backend(), testA100())
			worker := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndPercent("8Gi", "30"), "gpu-0")
			previous, err := workerstate.PrepareWorkerSharedMemory(
				w.shmBasePath, workerstate.NewPodIdentifier(worker.Namespace, worker.WorkerName), worker.WorkerUID,
				[]workerstate.DeviceConfig{{DeviceIdx: 0, DeviceUUID: "0", UpLimit: tc.stale, MemLimit: 8 * gib}}, true,
			)
			require.NoError(t, err)
			require.NoError(t, previous.Close())

			w.workerChangeHandler().OnAdd(worker)
			foundUpLimit, upLimit := found(t, w, worker)
			require.Equal(t, tc.wantFound, foundUpLimit)
			require.Equal(t, uint32(30), upLimit)
			mapped, _ := readMappedLimits(t, w, worker, 0)
			require.Equal(t, uint32(30), mapped, "the current limit is applied to the mapping all the same")

			// Later applications find the current limit in place; what was
			// found when the mapping was opened is kept.
			w.syncSharedMemoryState()
			foundUpLimit, _ = found(t, w, worker)
			require.Equal(t, tc.wantFound, foundUpLimit)
		})
	}

	// A mapping that is created for the worker carries the limit it was
	// created with, and nothing is kept for a worker that is gone.
	w, _ := newLimitTestController(t, testA100Backend(), testA100())
	worker := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndPercent("8Gi", "30"), "gpu-0")
	w.workerChangeHandler().OnAdd(worker)
	foundUpLimit, upLimit := found(t, w, worker)
	require.Equal(t, uint32(30), foundUpLimit)
	require.Equal(t, uint32(30), upLimit)
	w.workerChangeHandler().OnRemove(worker)
	require.Empty(t, w.foundUpLimits)
}

// Limits are applied when a mapping is opened and on every change, never
// periodically. A change that arrives while the worker has no mapping yet must
// therefore not be lost: the mapping is created from the worker as it is then.
func TestLimitChangeBeforeMappingExistsIsAppliedWhenOpened(t *testing.T) {
	w, allocations := newLimitTestController(t, testA100Backend(), testA100())
	handler := w.workerChangeHandler()
	// Not admitted yet: no allocation, so no mapping can be prepared.
	pending := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("20Gi", "156"), "gpu-0")
	pending.Status = api.WorkerStatusPending
	handler.OnAdd(pending)
	require.Nil(t, w.getShmHandle(pending.WorkerUID))

	resized := *pending
	resized.Limits = vramAndTflops("8Gi", "78")
	handler.OnUpdate(pending, &resized)
	require.Nil(t, w.getShmHandle(pending.WorkerUID))

	// The device plugin allocates with the Pod as it was when it asked, then
	// prepares the mapping.
	_, err := allocations.AllocateWorkerDevices(pending)
	require.NoError(t, err)
	require.NoError(t, handler.OnPrepare(pending.WorkerUID))

	upLimit, memLimit := readMappedLimits(t, w, &resized, 0)
	require.Equal(t, uint32(25), upLimit)
	require.Equal(t, 8*gib, memLimit)

	// A second change before the worker runs, and one after, both arrive.
	running := resized
	running.Status, running.Limits = api.WorkerStatusRunning, vramAndTflops("16Gi", "234")
	handler.OnUpdate(&resized, &running)
	upLimit, memLimit = readMappedLimits(t, w, &running, 0)
	require.Equal(t, uint32(75), upLimit)
	require.Equal(t, 16*gib, memLimit)
}

// Hard and shared workers receive their limits when the container starts. The
// mapping of their legacy client keeps the values it was created with.
func TestWorkerLimitUpdateLeavesNonSoftMappingUntouched(t *testing.T) {
	for _, mode := range []tfv1.IsolationModeType{tfv1.IsolationModeHard, tfv1.IsolationModeShared} {
		t.Run(string(mode), func(t *testing.T) {
			w, allocations := newLimitTestController(t, testA100Backend(), testA100())
			handler := w.workerChangeHandler()
			worker := newLimitTestWorker(mode, vramAndPercent("20Gi", "50"), "gpu-0")
			handler.OnAdd(worker)
			require.NoError(t, w.WithWorkerSharedMemory(worker.WorkerUID, nil))

			updated := *worker
			updated.Limits = vramAndPercent("8Gi", "30")
			handler.OnUpdate(worker, &updated)
			w.syncSharedMemoryState()

			upLimit, memLimit := readMappedLimits(t, w, &updated, 0)
			require.Equal(t, uint32(50), upLimit)
			require.Equal(t, 20*gib, memLimit)
			allocation, exists := allocations.GetWorkerAllocation(worker.WorkerUID)
			require.True(t, exists)
			if mode == tfv1.IsolationModeHard {
				require.Equal(t, "50", allocation.Envs[constants.HardSMLimiterEnv], "admission-time env is kept")
			}
		})
	}
}

// A limiter reads each limit field with an independent atomic load while the
// hypervisor updates them. It must only ever observe a published value.
func TestWorkerLimitUpdateIsSafeForConcurrentReaders(t *testing.T) {
	w, allocations := newLimitTestController(t, testA100Backend(), testA100())
	handler := w.workerChangeHandler()
	low, high := vramAndPercent("8Gi", "30"), vramAndPercent("20Gi", "80")
	current := newLimitTestWorker(tfv1.IsolationModeSoft, low, "gpu-0")
	handler.OnAdd(current)
	view := openLimiterView(t, w, current)

	ctx, cancel := context.WithCancel(t.Context())
	w.startSharedMemorySyncLoop(ctx)
	stop := make(chan struct{})
	var readers sync.WaitGroup
	var unexpected atomic.Int64
	for range 4 {
		readers.Go(func() {
			for {
				select {
				case <-stop:
					return
				default:
				}
				upLimit, memLimit := loadLimiterView(view, 0)
				if (upLimit != 30 && upLimit != 80) || (memLimit != 8*gib && memLimit != 20*gib) {
					unexpected.Add(1)
				}
				if allocation, ok := allocations.GetWorkerAllocation("worker-uid"); ok {
					_ = allocation.WorkerInfo.Limits.Vram.Value()
				}
				_ = w.buildWorkerInfoSnapshots()
			}
		})
	}
	for i := range 400 {
		next := *current
		next.Limits = high
		if i%2 == 1 {
			next.Limits = low
		}
		handler.OnUpdate(current, &next)
		current = &next
	}
	close(stop)
	readers.Wait()
	cancel()
	w.syncWG.Wait()

	require.Zero(t, unexpected.Load(), "a reader observed a limit that was never published")
	upLimit, memLimit := readMappedLimits(t, w, current, 0)
	require.Equal(t, uint32(30), upLimit)
	require.Equal(t, 8*gib, memLimit)
}

func TestRefreshWorkerInfoReplacesAllocationWithoutMutatingIt(t *testing.T) {
	devices := &fakeDeviceController{devices: map[string]*api.DeviceInfo{"gpu-0": testA100()}}
	allocations := NewAllocationController(devices)
	worker := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("20Gi", "156"), "gpu-0")
	neighbor := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("4Gi", "30"), "gpu-0")
	neighbor.WorkerUID, neighbor.WorkerName = "neighbor-uid", "neighbor"
	original, err := allocations.AllocateWorkerDevices(worker)
	require.NoError(t, err)
	neighborAllocation, err := allocations.AllocateWorkerDevices(neighbor)
	require.NoError(t, err)
	// Callers keep allocations and per-device lists they were handed earlier.
	heldByDevice := allocations.GetDeviceAllocations()["gpu-0"]

	updated := *worker
	updated.Limits = vramAndTflops("8Gi", "78")
	refreshed, ok := allocations.RefreshWorkerInfo(&updated)
	require.True(t, ok)
	require.NotSame(t, original, refreshed)
	require.Same(t, &updated, refreshed.WorkerInfo)
	require.Same(t, worker, original.WorkerInfo, "an allocation already handed out must not change")
	require.Equal(t, original.DeviceInfos, refreshed.DeviceInfos)
	require.Equal(t, original.Envs, refreshed.Envs)
	require.Equal(t, original.Devices, refreshed.Devices)

	current, exists := allocations.GetWorkerAllocation(worker.WorkerUID)
	require.True(t, exists)
	require.Same(t, refreshed, current)
	byDevice := allocations.GetDeviceAllocations()["gpu-0"]
	require.Len(t, heldByDevice, 2)
	require.Same(t, original, heldByDevice[0])
	require.Len(t, byDevice, 2)
	require.Same(t, refreshed, byDevice[0])
	require.Same(t, neighborAllocation, byDevice[1])

	// Refreshing with the worker already recorded is a no-op.
	again, ok := allocations.RefreshWorkerInfo(&updated)
	require.True(t, ok)
	require.Same(t, refreshed, again)

	require.NoError(t, allocations.DeallocateWorker(worker.WorkerUID))
	byDevice = allocations.GetDeviceAllocations()["gpu-0"]
	require.Len(t, byDevice, 1)
	require.Same(t, neighborAllocation, byDevice[0])
}

func TestRefreshWorkerInfoKeepsDeviceAssignment(t *testing.T) {
	devices := &fakeDeviceController{devices: map[string]*api.DeviceInfo{"gpu-0": testA100()}}
	allocations := NewAllocationController(devices)
	worker := newLimitTestWorker(tfv1.IsolationModeSoft, vramAndTflops("20Gi", "156"), "gpu-0")
	original, err := allocations.AllocateWorkerDevices(worker)
	require.NoError(t, err)

	otherDevice := *worker
	otherDevice.AllocatedDevices = []string{"gpu-1"}
	otherMode := *worker
	otherMode.IsolationMode = tfv1.IsolationModeHard
	otherTemplate := *worker
	otherTemplate.PartitionTemplateID = "template"
	unknown := *worker
	unknown.WorkerUID = "unknown-uid"
	for name, update := range map[string]*api.WorkerInfo{
		"different devices": &otherDevice, "different isolation mode": &otherMode,
		"different partition template": &otherTemplate, "worker without allocation": &unknown, "nil worker": nil,
	} {
		_, ok := allocations.RefreshWorkerInfo(update)
		require.False(t, ok, name)
	}
	current, exists := allocations.GetWorkerAllocation(worker.WorkerUID)
	require.True(t, exists)
	require.Same(t, original, current)
	byDevice := allocations.GetDeviceAllocations()["gpu-0"]
	require.Len(t, byDevice, 1)
	require.Same(t, original, byDevice[0])
}
