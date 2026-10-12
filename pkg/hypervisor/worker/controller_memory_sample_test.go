package worker

import (
	"errors"
	"math"
	"sync/atomic"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/stretchr/testify/require"
)

// slowProcessQuery is a device controller whose process query takes time: the
// clock is moved on while it runs.
type slowProcessQuery struct {
	*fakeDeviceController
	during func()
}

func (d *slowProcessQuery) GetProcessInformation() ([]api.ProcessInformation, error) {
	d.during()
	return d.fakeDeviceController.GetProcessInformation()
}

// memorySampleFixture is a worker with two devices behind a provider whose
// process query takes 40 ms of the controller's clock.
type memorySampleFixture struct {
	controller *WorkerController
	devices    *fakeDeviceController
	handle     *workerstate.SharedMemoryHandle
	now        time.Time
}

const memorySampleQueryTime = 40 * time.Millisecond

func newMemorySampleFixture(t *testing.T) *memorySampleFixture {
	t.Helper()
	const namespace = "ns"
	shmBasePath := t.TempDir()
	handle, err := workerstate.CreateSharedMemoryHandle(shmBasePath, workerstate.NewPodIdentifier(namespace, "pod-a"),
		[]workerstate.DeviceConfig{
			{DeviceIdx: 0, DeviceUUID: "GPU-0", UpLimit: 50, MemLimit: 10 << 30},
			{DeviceIdx: 1, DeviceUUID: "GPU-1", UpLimit: 50, MemLimit: 10 << 30},
		})
	require.NoError(t, err)
	t.Cleanup(func() { _ = handle.Close() })

	worker := &api.WorkerInfo{
		WorkerUID: "uid-a", Namespace: namespace, WorkerName: "pod-a", IsolationMode: tfv1.IsolationModeHard,
	}
	// A clock with a fraction of a second, which the published time keeps.
	f := &memorySampleFixture{devices: &fakeDeviceController{}, handle: handle, now: time.Unix(1_710_000_000, 250_000_000)}
	f.controller = &WorkerController{
		backend: &fakeWorkerBackend{mappings: map[uint32]*framework.ProcessMappingInfo{
			101: {Namespace: namespace, PodName: "pod-a"},
			102: {Namespace: namespace, PodName: "pod-a"},
		}},
		deviceController: &slowProcessQuery{
			fakeDeviceController: f.devices,
			during:               func() { f.now = f.now.Add(memorySampleQueryTime) },
		},
		allocationController: &fakeWorkerAllocationController{allocations: map[string]*api.WorkerAllocation{
			"uid-a": {WorkerInfo: worker, DeviceInfos: []*api.DeviceInfo{{UUID: "gpu-0", Index: 0}, {UUID: "gpu-1", Index: 1}}},
		}},
		workers:     map[string]*api.WorkerInfo{"uid-a": worker},
		shmBasePath: shmBasePath,
		shmHandles:  map[string]*workerstate.SharedMemoryHandle{"uid-a": handle},
		nowFunc:     func() time.Time { return f.now },
	}
	return f
}

// sample takes one process sample the way the sync loop does and returns what
// it hands to publishPodMemoryUsed for the worker, by device slot, and the
// clock reading before the provider was queried.
func (f *memorySampleFixture) sample(t *testing.T, rows ...api.ProcessInformation) (map[int]podMemorySample, float64) {
	t.Helper()
	f.devices.processInfos = rows
	f.now = f.now.Add(sharedMemorySyncInterval)
	before := float64(f.now.UnixMicro()) / 1e6

	memory, unreadable, sampledAtSecs := f.controller.collectWorkerMemoryUsage(f.controller.buildWorkerLookupMap())
	require.NotNil(t, memory)
	allocation, _ := f.controller.allocationController.GetWorkerAllocation("uid-a")
	bySlot := make(map[int]podMemorySample)
	for _, sample := range podMemorySamples(allocation.DeviceInfos, memory["uid-a"], unreadable, sampledAtSecs) {
		bySlot[sample.index] = sample
		f.handle.WithState(func(state *workerstate.SharedDeviceState) {
			require.True(t, publishPodMemoryUsed(state, sample.index, sample.used, sample.sampledAtSecs))
		})
	}
	return bySlot, before
}

func (f *memorySampleFixture) published(slot int) (used uint64) {
	f.handle.WithState(func(state *workerstate.SharedDeviceState) {
		used = atomic.LoadUint64(&state.V2.Devices[slot].DeviceInfo.PodMemoryUsed)
	})
	return used
}

// publishedSample reads a slot the way a limiter does: the figure, and the time
// published with it (zero when there is none).
func (f *memorySampleFixture) publishedSample(t *testing.T, slot int) (used uint64, sampledAtSecs float64) {
	t.Helper()
	require.True(t, f.handle.WithState(func(state *workerstate.SharedDeviceState) {
		var ok bool
		used, sampledAtSecs, ok = state.GetPodMemorySample(slot)
		require.True(t, ok)
	}))
	return used, sampledAtSecs
}

// sync runs one pass of the sync loop half a second later and returns the
// clock reading from before the provider was queried.
func (f *memorySampleFixture) sync(rows ...api.ProcessInformation) float64 {
	f.devices.processInfos = rows
	f.now = f.now.Add(sharedMemorySyncInterval)
	before := float64(f.now.UnixMicro()) / 1e6
	f.controller.syncSharedMemoryState()
	return before
}

func (f *memorySampleFixture) requirePublished(t *testing.T, slot int, wantUsed uint64, wantAt float64, msg string) {
	t.Helper()
	used, sampledAtSecs := f.publishedSample(t, slot)
	require.Equal(t, wantUsed, used, msg)
	require.Equal(t, wantAt, sampledAtSecs, msg)
}

func memoryProcess(pid, device string, memory uint64) api.ProcessInformation {
	return api.ProcessInformation{ProcessID: pid, DeviceUUID: device, MemoryUsedBytes: memory}
}

// The limiter admits allocations against the usage figure the hypervisor
// publishes, and has to know how old that figure is: what the worker allocated
// after the sample was taken is not in it. The time that goes with a figure is
// therefore the clock reading from before the provider was queried, never one
// from after: a later one would declare allocations covered that the query may
// not have seen. It is in fractional seconds since the Unix epoch, like the
// refill timestamps of the same mapping.
func TestMemorySampleTimeIsTakenBeforeTheProviderIsQueried(t *testing.T) {
	f := newMemorySampleFixture(t)

	samples, before := f.sample(t, memoryProcess("101", "GPU-0", 1<<30), memoryProcess("102", "GPU-1", 2<<30))
	after := float64(f.now.UnixMicro()) / 1e6
	require.InDelta(t, memorySampleQueryTime.Seconds(), after-before, 1e-6, "the query took time on the clock")
	require.Equal(t, 1_710_000_000.75, before, "fractional Unix seconds")

	require.Len(t, samples, 2)
	for slot, want := range map[int]uint64{0: 1 << 30, 1: 2 << 30} {
		require.Equal(t, want, samples[slot].used)
		require.Equal(t, before, samples[slot].sampledAtSecs,
			"slot %d is published with the time read before the provider was queried", slot)
		require.Equal(t, want, f.published(slot))
		f.requirePublished(t, slot, want, before, "the limiter reads the figure with that time")
	}
	// The sample other readers get carries the same time.
	require.Equal(t, before, f.controller.processSample.Load().takenAtSecs())

	// Every sample has its own time.
	samples, next := f.sample(t, memoryProcess("101", "GPU-0", 3<<30))
	require.Greater(t, next, after)
	require.Equal(t, next, samples[0].sampledAtSecs)
	require.Equal(t, next, samples[1].sampledAtSecs, "a readable device without processes is a figure of this sample")
	require.Zero(t, f.published(1))
	f.requirePublished(t, 0, 3<<30, next, "every figure is published with the time of its own sample")
	f.requirePublished(t, 1, 0, next, "a measured zero has the time of its sample too")
}

// A figure that is kept because nothing newer is known keeps the time of the
// sample it came from: for a device whose process list could not be read, and
// when no sample could be taken at all, nothing is handed to the publishing
// helper, so neither the figure nor its time changes. Publishing the kept figure
// with the time of the current pass would tell the limiter that it is current.
func TestKeptMemoryFigureKeepsTheTimeOfItsSample(t *testing.T) {
	f := newMemorySampleFixture(t)
	unreadable := api.ProcessInformation{
		DeviceUUID: "GPU-1", ComputeUtilizationUnavailable: true, ProcessListUnavailable: true,
	}

	samples, first := f.sample(t, memoryProcess("101", "GPU-0", 1<<30), memoryProcess("102", "GPU-1", 2<<30))
	require.Equal(t, first, samples[1].sampledAtSecs)

	// GPU-1 cannot be read: its figure stays, and is not published again.
	samples, second := f.sample(t, memoryProcess("101", "GPU-0", 3<<30), unreadable)
	require.Greater(t, second, first)
	require.Equal(t, podMemorySample{index: 0, used: 3 << 30, sampledAtSecs: second}, samples[0])
	require.NotContains(t, samples, 1, "nothing is published for the slot of a device that could not be read")
	require.Equal(t, uint64(2<<30), f.published(1), "the kept figure is the one of the first sample")
	f.requirePublished(t, 0, 3<<30, second, "a readable device follows the sample")
	f.requirePublished(t, 1, 2<<30, first, "the kept figure is still published with the time of the first sample")

	// No sample at all: the sync pass publishes nothing for any slot.
	f.devices.processErr = errors.New("driver unavailable")
	f.now = f.now.Add(sharedMemorySyncInterval)
	memory, _, sampledAtSecs := f.controller.collectWorkerMemoryUsage(f.controller.buildWorkerLookupMap())
	require.Nil(t, memory)
	require.Zero(t, sampledAtSecs)
	f.controller.syncSharedMemoryState()
	require.Equal(t, uint64(3<<30), f.published(0))
	require.Equal(t, uint64(2<<30), f.published(1))
	f.requirePublished(t, 0, 3<<30, second, "a sample that could not be taken leaves the figure and its time")
	f.requirePublished(t, 1, 2<<30, first, "a sample that could not be taken leaves the figure and its time")
	f.devices.processErr = nil

	// Readable again: the slot gets a figure of this sample, with its time.
	samples, fourth := f.sample(t, memoryProcess("101", "GPU-0", 3<<30), memoryProcess("102", "GPU-1", 1<<30))
	require.Equal(t, podMemorySample{index: 1, used: 1 << 30, sampledAtSecs: fourth}, samples[1])
	f.requirePublished(t, 1, 1<<30, fourth, "a device that can be read again")

	// A device that stays unreadable for longer than the accounting is kept is
	// accounted as holding nothing. No query produced that zero: it is a figure
	// without a time, and the limiter checks against it alone, as it did before
	// times were published.
	f.sample(t, memoryProcess("101", "GPU-0", 3<<30), unreadable)
	f.requirePublished(t, 1, 1<<30, fourth, "kept, with its time, while within the bound")
	f.now = f.now.Add(unreadableMemoryMaxAge)
	samples, expired := f.sample(t, memoryProcess("101", "GPU-0", 3<<30), unreadable)
	require.Equal(t, podMemorySample{index: 1, used: 0, sampledAtSecs: 0}, samples[1])
	require.Equal(t, podMemorySample{index: 0, used: 3 << 30, sampledAtSecs: expired}, samples[0])
	f.requirePublished(t, 1, 0, 0, "past the bound: nothing is accounted, and no time stands next to it")
	f.requirePublished(t, 0, 3<<30, expired, "the readable device of the same worker keeps being sampled")
}

// The sync loop is the writer of both fields. Through it: the figure of every
// pass is published with the clock reading from before that pass's query; a
// device past the bound of unreadableMemoryMaxAge gets its zero without a time
// for as long as it stays unreadable; and a pass without a sample writes
// neither field.
func TestSyncLoopPublishesEachFigureWithTheTimeOfItsSample(t *testing.T) {
	f := newMemorySampleFixture(t)
	unreadable := api.ProcessInformation{
		DeviceUUID: "GPU-1", ComputeUtilizationUnavailable: true, ProcessListUnavailable: true,
	}
	f.requirePublished(t, 0, 0, 0, "a new segment publishes no time")

	first := f.sync(memoryProcess("101", "GPU-0", 1<<30), memoryProcess("102", "GPU-1", 2<<30))
	f.requirePublished(t, 0, 1<<30, first, "read before the provider was queried")
	f.requirePublished(t, 1, 2<<30, first, "read before the provider was queried")

	// The same figure in the next sample is a new sample.
	second := f.sync(memoryProcess("101", "GPU-0", 1<<30), unreadable)
	require.InDelta(t, (sharedMemorySyncInterval + memorySampleQueryTime).Seconds(), second-first, 1e-6)
	f.requirePublished(t, 0, 1<<30, second, "an unchanged figure of a new sample has the new time")
	f.requirePublished(t, 1, 2<<30, first, "kept with the time of the sample it came from")

	f.devices.processErr = errors.New("driver unavailable")
	f.sync()
	f.requirePublished(t, 0, 1<<30, second, "no sample: both fields stay")
	f.requirePublished(t, 1, 2<<30, first, "no sample: both fields stay")
	f.devices.processErr = nil

	f.now = f.now.Add(unreadableMemoryMaxAge)
	expired := f.sync(memoryProcess("101", "GPU-0", 1<<30), unreadable)
	f.requirePublished(t, 0, 1<<30, expired, "the readable device")
	f.requirePublished(t, 1, 0, 0, "past the bound")
	later := f.sync(memoryProcess("101", "GPU-0", 1<<30), unreadable)
	f.requirePublished(t, 0, 1<<30, later, "the readable device")
	f.requirePublished(t, 1, 0, 0, "still unreadable")

	readable := f.sync(memoryProcess("101", "GPU-0", 1<<30), memoryProcess("102", "GPU-1", 4<<30))
	f.requirePublished(t, 1, 4<<30, readable, "read again: a figure of this sample")
}

// A limiter tells a new sample by its time, so the time of a sample is never
// the one published before it in the slot, nor below it: when the clock stood
// still or was set back between two samples, the next representable time above
// the previous one is published. The previous one is what the segment holds,
// which a restarted hypervisor finds there.
func TestMemorySampleTimeDiffersFromThePreviousOneWhateverTheClockDoes(t *testing.T) {
	f := newMemorySampleFixture(t)
	above := func(secs float64) float64 { return math.Nextafter(secs, math.Inf(1)) }

	first := f.sync(memoryProcess("101", "GPU-0", 1<<30))
	f.requirePublished(t, 0, 1<<30, first, "the clock's reading")

	// The next pass reads the same time: the clock was set back by exactly what
	// had passed.
	f.now = f.now.Add(-memorySampleQueryTime - sharedMemorySyncInterval)
	require.Equal(t, first, f.sync(memoryProcess("101", "GPU-0", 2<<30)))
	stalled := above(first)
	f.requirePublished(t, 0, 2<<30, stalled, "a new sample at the same clock reading")
	// A slot for which this sample says what is published already, figure and
	// clock reading alike, is told nothing new and is not written.
	f.requirePublished(t, 1, 0, first, "the same statement as the one published")

	// The clock is set back by an hour.
	f.now = f.now.Add(-time.Hour)
	require.Less(t, f.sync(memoryProcess("101", "GPU-0", 3<<30)), first)
	back := above(stalled)
	f.requirePublished(t, 0, 3<<30, back, "a new sample at an earlier clock reading")

	// Publishing the pair that is published already changes nothing: it is the
	// same sample, not a new one.
	f.handle.WithState(func(state *workerstate.SharedDeviceState) {
		require.True(t, publishPodMemoryUsed(state, 0, 3<<30, back))
	})
	f.requirePublished(t, 0, 3<<30, back, "the same pair again")

	// The hypervisor restarts: the mapping is opened again, and the time before
	// is the one found in it.
	require.NoError(t, f.handle.Close())
	reopened, err := workerstate.OpenSharedMemoryHandle(
		f.controller.shmBasePath, workerstate.NewPodIdentifier("ns", "pod-a"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = reopened.Close() })
	f.handle = reopened
	f.controller.shmHandles["uid-a"] = reopened
	f.controller.processSample.Store(nil)
	f.requirePublished(t, 0, 3<<30, back, "the published pair outlives the process")
	require.Less(t, f.sync(memoryProcess("101", "GPU-0", 4<<30)), first)
	f.requirePublished(t, 0, 4<<30, above(back), "above what the segment held")

	// Once the clock is ahead of what is published, its reading is used again.
	f.now = f.now.Add(2 * time.Hour)
	ahead := f.sync(memoryProcess("101", "GPU-0", 4<<30))
	require.Greater(t, ahead, above(back))
	f.requirePublished(t, 0, 4<<30, ahead, "the clock's reading")
}
