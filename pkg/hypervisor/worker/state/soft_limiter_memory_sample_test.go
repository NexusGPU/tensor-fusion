//go:build darwin || linux || freebsd || netbsd

package worker

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

// Where the limiter finds the two fields of a memory sample, from the start of
// the segment, as the layout shared by both sides defines them.
const (
	limiterSegmentSize             = 35504
	limiterPodMemoryUsedOffset     = 96
	limiterDeviceEntrySize         = 136
	limiterPodMemorySampledAtStart = 34992
	limiterRemainingPaddingStart   = 35120
)

func allDeviceConfigs() []DeviceConfig {
	configs := make([]DeviceConfig, MaxDevices)
	for i := range configs {
		configs[i] = DeviceConfig{
			DeviceIdx: uint32(i), DeviceUUID: "GPU-" + string(rune('a'+i)), UpLimit: 50, MemLimit: 8 << 30,
		}
	}
	return configs
}

// The time of a memory sample lives in the first 128 bytes of what used to be
// 512 bytes of trailing padding: one 8-byte word per device slot, at
// 34992 + 8*i from the start of the segment. Nothing else moves and the
// segment keeps its size, which is what lets either side be upgraded alone.
// The limiter's test limiter_memory_layout asserts the same numbers against its
// own structs.
func TestPodMemorySampleFieldsKeepTheLimiterLayout(t *testing.T) {
	var raw rustSharedDeviceStateV2Layout
	require.Equal(t, uintptr(limiterSegmentSize), unsafe.Sizeof(raw))

	payload := unsafe.Offsetof(raw.V2)
	sampledAt := payload + unsafe.Offsetof(raw.V2.PodMemorySampledAt)
	require.Equal(t, uintptr(limiterPodMemorySampledAtStart), sampledAt)
	require.Equal(t, uintptr(8*MaxDevices), unsafe.Sizeof(raw.V2.PodMemorySampledAt))
	require.Equal(t, uintptr(35112), sampledAt+8*(MaxDevices-1), "the last slot")
	require.Equal(t, uintptr(limiterRemainingPaddingStart), payload+unsafe.Offsetof(raw.V2._padding))
	require.Equal(t, uintptr(limiterSegmentSize-limiterRemainingPaddingStart), unsafe.Sizeof(raw.V2._padding))
	// The field starts where the padding started: right behind the process set.
	require.Equal(t, sampledAt, payload+unsafe.Offsetof(raw.V2.PIDs)+unsafe.Sizeof(raw.V2.PIDs))

	entries := payload + unsafe.Offsetof(raw.V2.Devices)
	usedInEntry := unsafe.Offsetof(raw.V2.Devices[0].DeviceInfo) +
		unsafe.Offsetof(raw.V2.Devices[0].DeviceInfo.PodMemoryUsed)
	require.Equal(t, uintptr(limiterDeviceEntrySize), unsafe.Sizeof(raw.V2.Devices[0]))
	for i := range uintptr(MaxDevices) {
		used := entries + i*unsafe.Sizeof(raw.V2.Devices[0]) + usedInEntry
		require.Equal(t, limiterPodMemoryUsedOffset+limiterDeviceEntrySize*i, used, "pod_memory_used of slot %d", i)
		// Both are naturally aligned words, as atomic access from either side
		// of the mapping requires.
		require.Zero(t, used%8)
		require.Zero(t, (sampledAt+8*i)%8)
	}

	// A V1 state has no such field and keeps all of its padding.
	require.Equal(t, uintptr(512), unsafe.Sizeof(SharedDeviceStateV1{}._padding))
}

// The same offsets, by the bytes of a published segment: what a limiter that
// maps the file reads at 96 + 136*i and at 34992 + 8*i is the figure and the
// bits of the time, and publishing a sample changes no other byte.
func TestPodMemorySampleIsWrittenWhereTheLimiterReadsIt(t *testing.T) {
	base := t.TempDir()
	pod := NewPodIdentifier("namespace", "worker")
	handle, err := CreateSharedMemoryHandle(base, pod, allDeviceConfigs())
	require.NoError(t, err)
	defer func() { _ = handle.Close() }()
	segment := func() []byte {
		image, err := os.ReadFile(filepath.Join(pod.ToPath(base), ShmPathSuffix))
		require.NoError(t, err)
		require.Len(t, image, limiterSegmentSize)
		return image
	}

	created := segment()
	require.Equal(t, make([]byte, limiterSegmentSize-limiterPodMemorySampledAtStart),
		created[limiterPodMemorySampledAtStart:], "a new segment publishes no time; its padding is zero as it always was")

	sample := func(slot int) (used uint64, sampledAtSecs float64) {
		return uint64(slot+1) << 20, 1_800_000_000.25 + float64(slot)
	}
	for slot := range MaxDevices {
		used, sampledAtSecs := sample(slot)
		require.True(t, handle.GetState().SetPodMemorySample(slot, used, sampledAtSecs))
	}
	published := segment()
	for slot := range MaxDevices {
		used, sampledAtSecs := sample(slot)
		usedAt := limiterPodMemoryUsedOffset + limiterDeviceEntrySize*slot
		sampledAtAt := limiterPodMemorySampledAtStart + 8*slot
		require.Equal(t, used, binary.LittleEndian.Uint64(published[usedAt:]), "pod_memory_used of slot %d", slot)
		require.Equal(t, math.Float64bits(sampledAtSecs), binary.LittleEndian.Uint64(published[sampledAtAt:]),
			"pod_memory_sampled_at of slot %d", slot)
		// Both were zero in the new segment.
		clear(published[usedAt : usedAt+8])
		clear(published[sampledAtAt : sampledAtAt+8])
	}
	require.True(t, bytes.Equal(created, published), "nothing but the two fields of each slot was written")
	require.Equal(t, make([]byte, limiterSegmentSize-limiterRemainingPaddingStart),
		published[limiterRemainingPaddingStart:], "the rest of the padding stays zero")
}

type podMemoryStore struct {
	field string
	value uint64
}

const (
	usedFieldName      = "pod_memory_used"
	sampledAtFieldName = "pod_memory_sampled_at"
)

// recordingStore performs every store of the writer and notes it.
func recordingStore(sampledAtField *uint64, stores *[]podMemoryStore) func(*uint64, uint64) {
	return func(field *uint64, value uint64) {
		name := usedFieldName
		if field == sampledAtField {
			name = sampledAtFieldName
		}
		*stores = append(*stores, podMemoryStore{field: name, value: value})
		atomic.StoreUint64(field, value)
	}
}

// The order of the stores is the protocol: the time is cleared, the figure is
// replaced, the time of the new figure is set. A figure without a time stops
// after the second store.
func TestPodMemorySampleIsStoredInProtocolOrder(t *testing.T) {
	const at = 1_800_000_000.5
	var used, sampledAt uint64
	publish := func(figure uint64, sampledAtSecs float64) []podMemoryStore {
		var stores []podMemoryStore
		storePodMemorySample(recordingStore(&sampledAt, &stores), &used, &sampledAt, figure, sampledAtSecs)
		return stores
	}

	require.Equal(t, []podMemoryStore{
		{sampledAtFieldName, 0}, {usedFieldName, 1 << 30}, {sampledAtFieldName, math.Float64bits(at)},
	}, publish(1<<30, at), "the first sample of a slot")
	require.Equal(t, []podMemoryStore{
		{sampledAtFieldName, 0}, {usedFieldName, 2 << 30}, {sampledAtFieldName, math.Float64bits(at + 0.5)},
	}, publish(2<<30, at+0.5), "a sample that replaces another")
	require.Equal(t, []podMemoryStore{
		{sampledAtFieldName, 0}, {usedFieldName, 2 << 30}, {sampledAtFieldName, math.Float64bits(at + 1)},
	}, publish(2<<30, at+1), "a new sample with the same figure is published like any other")

	for name, notATime := range map[string]float64{
		"zero": 0, "negative": -1, "not a number": math.NaN(), "infinite": math.Inf(1),
	} {
		require.Equal(t, []podMemoryStore{{sampledAtFieldName, 0}, {usedFieldName, 3 << 30}},
			publish(3<<30, notATime), "a figure without a time (%s)", name)
		require.Zero(t, sampledAt)
		publish(2<<30, at+2)
	}
}

// interleavePodMemoryReads runs write, which performs exactly the given number
// of stores, once for every way the three loads of a reader can fall between
// those stores, and hands each reading to visit.
func interleavePodMemoryReads(
	t *testing.T, stores int, write func(store func(*uint64, uint64), used, sampledAt *uint64),
	visit func(before, used, after uint64),
) {
	t.Helper()
	for first := 0; first <= stores; first++ {
		for second := first; second <= stores; second++ {
			for third := second; third <= stores; third++ {
				var usedField, sampledAtField uint64
				var reading [3]uint64
				positions := [3]int{first, second, third}
				fields := [3]*uint64{&sampledAtField, &usedField, &sampledAtField}
				next, done := 0, 0
				// The loads due before the store that is about to happen.
				load := func() {
					for next < len(positions) && positions[next] == done {
						reading[next] = atomic.LoadUint64(fields[next])
						next++
					}
				}
				write(func(field *uint64, value uint64) {
					load()
					atomic.StoreUint64(field, value)
					done++
				}, &usedField, &sampledAtField)
				require.Equal(t, stores, done)
				load()
				visit(reading[0], reading[1], reading[2])
			}
		}
	}
}

// A reader loads the time, the figure and the time again, wherever between the
// writer's stores those three loads fall, and accepts the pair when both
// readings of the time are the same and not zero. Whatever it accepts is a
// figure and the time it was published with: never the figure of one sample
// with the time of another, and never a figure that was published without a
// time.
func TestPodMemorySampleReaderNeverPairsAFigureWithTheTimeOfAnotherSample(t *testing.T) {
	const at = 1_800_000_000.5
	type pair struct{ used, sampledAt uint64 }
	published := map[pair]bool{
		{1 << 30, math.Float64bits(at)}:     true,
		{2 << 30, math.Float64bits(at + 1)}: true,
		{4 << 30, math.Float64bits(at + 3)}: true,
	}
	accepted := make(map[pair]int)
	rejected := 0
	// Three samples, and a figure without a time between the last two: eleven stores.
	interleavePodMemoryReads(t, 11, func(store func(*uint64, uint64), used, sampledAt *uint64) {
		storePodMemorySample(store, used, sampledAt, 1<<30, at)
		storePodMemorySample(store, used, sampledAt, 2<<30, at+1)
		storePodMemorySample(store, used, sampledAt, 3<<30, 0)
		storePodMemorySample(store, used, sampledAt, 4<<30, at+3)
	}, func(before, used, after uint64) {
		if before != after || before == 0 {
			rejected++
			return
		}
		require.True(t, published[pair{used, before}], "accepted figure %d with time %v", used, math.Float64frombits(before))
		accepted[pair{used, before}]++
	})
	require.Len(t, accepted, len(published), "every published pair can be read")
	require.Positive(t, rejected)

	// The same readings do tell a writer that replaces the figure under a
	// time that is still published: the order is what the reader relies on.
	mismatched := 0
	interleavePodMemoryReads(t, 4, func(store func(*uint64, uint64), used, sampledAt *uint64) {
		store(used, 1<<30)
		store(sampledAt, math.Float64bits(at))
		store(used, 2<<30)
		store(sampledAt, math.Float64bits(at+1))
	}, func(before, used, after uint64) {
		if before == after && before != 0 && !published[pair{used, before}] {
			mismatched++
		}
	})
	require.Positive(t, mismatched)
}

// The same guarantee with a writer and readers that really run at the same
// time, on a mapped segment (run with -race). Every sample is self-describing:
// its figure is derived from the bits of its time, so a reader can tell a
// pair that was published together from one that was not. Figures without a
// time are written in between, and must never be accepted with one.
func TestPodMemorySampleConcurrentReadersAndWriter(t *testing.T) {
	handle, err := CreateSharedMemoryHandle(t.TempDir(), NewPodIdentifier("namespace", "worker"), createTestConfigs())
	require.NoError(t, err)
	defer func() { _ = handle.Close() }()
	state := handle.GetState()
	const (
		slot          = int(testDeviceIdx)
		samples       = 200_000
		start         = 1_800_000_000.0
		withoutATime  = uint64(1)
		readerThreads = 3
	)
	figureOf := func(sampledAt uint64) uint64 { return ^sampledAt }
	timeOf := func(i int) float64 { return start + float64(i)/1024 }

	var written atomic.Bool
	var accepted, mismatched atomic.Int64
	var wg sync.WaitGroup
	wg.Go(func() {
		defer written.Store(true)
		for i := range samples {
			if i%5 == 4 {
				state.SetPodMemoryUsed(slot, withoutATime)
			}
			at := timeOf(i)
			state.SetPodMemorySample(slot, figureOf(math.Float64bits(at)), at)
		}
	})
	usedField, sampledAtField := &state.V2.Devices[slot].DeviceInfo.PodMemoryUsed, &state.V2.PodMemorySampledAt[slot]
	check := func(used, sampledAt uint64) {
		if used != figureOf(sampledAt) {
			mismatched.Add(1)
		}
		accepted.Add(1)
	}
	for reader := range readerThreads {
		wg.Go(func() {
			for last := false; ; {
				if reader == 0 {
					// The reader the package offers.
					if used, at, ok := state.GetPodMemorySample(slot); ok && at != 0 {
						check(used, math.Float64bits(at))
					}
				} else {
					// The limiter's algorithm, spelled out.
					before := atomic.LoadUint64(sampledAtField)
					used := atomic.LoadUint64(usedField)
					after := atomic.LoadUint64(sampledAtField)
					if before == after && before != 0 {
						check(used, before)
					}
				}
				// One more reading once the writer is done: that one is accepted.
				if last {
					return
				}
				last = written.Load()
			}
		})
	}
	wg.Wait()

	t.Logf("%d samples published, %d readings accepted", samples, accepted.Load())
	require.Zero(t, mismatched.Load(), "a reader accepted a figure with the time of another sample")
	require.GreaterOrEqual(t, accepted.Load(), int64(readerThreads))
	used, at, ok := state.GetPodMemorySample(slot)
	require.True(t, ok)
	require.Equal(t, timeOf(samples-1), at)
	require.Equal(t, figureOf(math.Float64bits(at)), used)
}

// The time of a sample differs from the one published before it in the slot,
// and never lies below it, whatever the clock did: a reader tells a new sample
// by its time. The time before is the one in the segment, so the rule holds
// across a restart of the hypervisor.
func TestPodMemorySampleTimeAlwaysMovesForward(t *testing.T) {
	const (
		slot = int(testDeviceIdx)
		at   = 1_800_000_000.5
	)
	base := t.TempDir()
	pod := NewPodIdentifier("namespace", "worker")
	handle, err := CreateSharedMemoryHandle(base, pod, createTestConfigs())
	require.NoError(t, err)
	defer func() { _ = handle.Close() }()
	state := handle.GetState()
	publish := func(used uint64, sampledAtSecs float64) float64 {
		t.Helper()
		require.True(t, state.SetPodMemorySample(slot, used, sampledAtSecs))
		gotUsed, gotAt, ok := state.GetPodMemorySample(slot)
		require.True(t, ok)
		require.Equal(t, used, gotUsed)
		return gotAt
	}

	require.Equal(t, at, publish(1<<30, at), "the first time is the clock's")
	require.Equal(t, at+0.5, publish(2<<30, at+0.5), "and so is every later one of a clock that advances")

	// The clock did not advance between two samples.
	previous := at + 0.5
	stalled := publish(3<<30, previous)
	require.Equal(t, math.Nextafter(previous, math.Inf(1)), stalled)
	require.Greater(t, stalled, previous)
	// And still has not: every sample gets a time of its own.
	for figure := uint64(4); figure < 8; figure++ {
		next := publish(figure<<30, at+0.5)
		require.Equal(t, math.Nextafter(stalled, math.Inf(1)), next)
		stalled = next
	}

	// The clock went back by an hour.
	back := publish(1<<30, at-3600)
	require.Equal(t, math.Nextafter(stalled, math.Inf(1)), back)
	require.Greater(t, back, stalled)

	// The pair that is in the segment already is that sample: it is not
	// written again, and its time is not moved.
	usedField, sampledAtField := &state.V2.Devices[slot].DeviceInfo.PodMemoryUsed, &state.V2.PodMemorySampledAt[slot]
	var stores []podMemoryStore
	storePodMemorySample(recordingStore(sampledAtField, &stores), usedField, sampledAtField, 1<<30, back)
	require.Empty(t, stores)
	require.Equal(t, back, publish(1<<30, back))

	// A restarted hypervisor has nothing but the segment to go by.
	require.NoError(t, handle.Close())
	handle, err = OpenSharedMemoryHandle(base, pod)
	require.NoError(t, err)
	defer func() { _ = handle.Close() }()
	state = handle.GetState()
	used, found, ok := state.GetPodMemorySample(slot)
	require.True(t, ok)
	require.Equal(t, uint64(1<<30), used)
	require.Equal(t, back, found, "the published pair outlives the handle")
	restarted := publish(2<<30, at-7200)
	require.Equal(t, math.Nextafter(back, math.Inf(1)), restarted)
	require.Equal(t, restarted+10, publish(3<<30, restarted+10), "a clock that is ahead again is taken as it is")

	// The segment is writable by the workload's processes; values read from it
	// are validated. A value found in place of the time that is not a time is
	// not one to stay above.
	sampledAtField = &state.V2.PodMemorySampledAt[slot]
	for name, bits := range map[string]uint64{
		"not a number": math.Float64bits(math.NaN()), "infinite": math.Float64bits(math.Inf(1)),
		"negative": math.Float64bits(-at), "all ones": ^uint64(0),
	} {
		atomic.StoreUint64(sampledAtField, bits)
		_, unusable, _ := state.GetPodMemorySample(slot)
		require.Zero(t, unusable, "%s is read as no time", name)
		require.Equal(t, at, publish(5<<30, at), "after %s", name)
	}
}

// Whatever else writes pod_memory_used leaves no time next to it: a time
// stands next to the figure of the query that started at that time, or is zero.
func TestWritersOfPodMemoryUsedWithoutASampleLeaveNoTime(t *testing.T) {
	const (
		slot = int(testDeviceIdx)
		at   = 1_800_000_000.5
	)
	requireSample := func(t *testing.T, state *SharedDeviceState, wantUsed uint64, wantAt float64) {
		t.Helper()
		used, sampledAtSecs, ok := state.GetPodMemorySample(slot)
		require.True(t, ok)
		require.Equal(t, wantUsed, used)
		require.Equal(t, wantAt, sampledAtSecs)
	}

	t.Run("a new segment", func(t *testing.T) {
		handle, err := PrepareWorkerSharedMemory(
			t.TempDir(), NewPodIdentifier("namespace", "worker"), testWorkerUID, allDeviceConfigs(), false)
		require.NoError(t, err)
		defer func() { _ = handle.Close() }()
		for i := range MaxDevices {
			used, sampledAtSecs, ok := handle.GetState().GetPodMemorySample(i)
			require.True(t, ok)
			require.Zero(t, used)
			require.Zero(t, sampledAtSecs)
			require.Zero(t, handle.GetState().V2.PodMemorySampledAt[i])
		}
	})
	t.Run("a figure set without a sample", func(t *testing.T) {
		handle, err := CreateSharedMemoryHandle(t.TempDir(), NewPodIdentifier("namespace", "worker"), createTestConfigs())
		require.NoError(t, err)
		defer func() { _ = handle.Close() }()
		state := handle.GetState()
		require.True(t, state.SetPodMemorySample(slot, 1<<30, at))
		requireSample(t, state, 1<<30, at)

		require.True(t, state.SetPodMemoryUsed(slot, 2<<30))
		requireSample(t, state, 2<<30, 0)
		require.Zero(t, state.V2.PodMemorySampledAt[slot])
		// The same figure again is still a figure without a time.
		require.True(t, state.SetPodMemorySample(slot, 1<<30, at+1))
		require.True(t, state.SetPodMemoryUsed(slot, 1<<30))
		requireSample(t, state, 1<<30, 0)
	})
	t.Run("a segment republished because it became unusable", func(t *testing.T) {
		base := t.TempDir()
		pod := NewPodIdentifier("namespace", "worker")
		first, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, createTestConfigs(), false)
		require.NoError(t, err)
		require.True(t, first.GetState().SetPodMemorySample(slot, 1<<30, at))
		require.NoError(t, os.Truncate(testGenerationPath(base, pod, testWorkerUID), 0))
		require.NoError(t, first.Close())

		recovered, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, createTestConfigs(), true)
		require.NoError(t, err)
		defer func() { _ = recovered.Close() }()
		require.True(t, recovered.WithState(func(state *SharedDeviceState) { requireSample(t, state, 0, 0) }))
	})
	// Opening a segment again writes neither field: the pair stays as it was
	// published, for the processes that still map it.
	t.Run("a segment that is reopened or adopted", func(t *testing.T) {
		base := t.TempDir()
		pod := NewPodIdentifier("namespace", "worker")
		original, err := CreateSharedMemoryHandle(base, pod, createTestConfigs())
		require.NoError(t, err)
		require.True(t, original.GetState().SetPodMemorySample(slot, 1<<30, at))
		require.NoError(t, original.Close())

		adopted, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, createTestConfigs(), true)
		require.NoError(t, err)
		requireSample(t, adopted.GetState(), 1<<30, at)
		require.NoError(t, adopted.Close())

		reopened, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, createTestConfigs(), true)
		require.NoError(t, err)
		defer func() { _ = reopened.Close() }()
		requireSample(t, reopened.GetState(), 1<<30, at)
	})
	t.Run("a device that does not exist", func(t *testing.T) {
		handle, err := CreateSharedMemoryHandle(t.TempDir(), NewPodIdentifier("namespace", "worker"), createTestConfigs())
		require.NoError(t, err)
		defer func() { _ = handle.Close() }()
		state := handle.GetState()
		for _, index := range []int{-1, slot + 1, MaxDevices, 999} {
			require.False(t, state.SetPodMemorySample(index, 1<<30, at), "slot %d", index)
			require.False(t, state.SetPodMemoryUsed(index, 1<<30), "slot %d", index)
			_, _, ok := state.GetPodMemorySample(index)
			require.False(t, ok, "slot %d", index)
		}
		require.Equal(t, [MaxDevices]uint64{}, state.V2.PodMemorySampledAt)
	})
}

// A V1 state has no place for the time: it gets the figure, as before, and
// nothing else in it changes.
func TestPodMemorySampleLeavesAV1StateAsItWas(t *testing.T) {
	const slot = int(testDeviceIdx)
	v1, err := NewSharedDeviceStateV1(createTestConfigs())
	require.NoError(t, err)
	state := &SharedDeviceState{V1: v1}
	image := func() []byte {
		return bytes.Clone(unsafe.Slice((*byte)(unsafe.Pointer(v1)), int(unsafe.Sizeof(*v1))))
	}
	usedAt := int(unsafe.Offsetof(v1.Devices) + uintptr(slot)*unsafe.Sizeof(v1.Devices[0]) +
		unsafe.Offsetof(v1.Devices[0].DeviceInfo) + unsafe.Offsetof(v1.Devices[0].DeviceInfo.PodMemoryUsed))

	want := image()
	require.True(t, state.SetPodMemorySample(slot, 3<<30, 1_800_000_000.5))
	binary.LittleEndian.PutUint64(want[usedAt:], 3<<30)
	require.True(t, bytes.Equal(want, image()), "only pod_memory_used was written")
	used, sampledAtSecs, ok := state.GetPodMemorySample(slot)
	require.True(t, ok)
	require.Equal(t, uint64(3<<30), used)
	require.Zero(t, sampledAtSecs)

	require.True(t, state.SetPodMemoryUsed(slot, 1<<30))
	binary.LittleEndian.PutUint64(want[usedAt:], 1<<30)
	require.True(t, bytes.Equal(want, image()))
	require.False(t, state.SetPodMemoryUsed(MaxDevices, 1))
}
