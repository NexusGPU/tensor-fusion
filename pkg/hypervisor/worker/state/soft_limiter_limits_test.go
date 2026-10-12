//go:build darwin || linux || freebsd || netbsd

package worker

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

// The limiter addresses the limit fields by offset, as the layout shared by
// both sides defines them: an 8-byte enum header, 136-byte device entries, and
// in each entry up_limit at +64 and mem_limit at +72.
func TestDeviceLimitFieldsKeepLimiterLayout(t *testing.T) {
	var raw rustSharedDeviceStateV2Layout
	entry := unsafe.Offsetof(raw.V2) + unsafe.Offsetof(raw.V2.Devices)
	require.Equal(t, uintptr(8), entry)
	require.Equal(t, uintptr(136), unsafe.Sizeof(raw.V2.Devices[0]))
	info := unsafe.Offsetof(raw.V2.Devices[0].DeviceInfo)
	require.Equal(t, uintptr(64), info+unsafe.Offsetof(raw.V2.Devices[0].DeviceInfo.UpLimit))
	require.Equal(t, uintptr(72), info+unsafe.Offsetof(raw.V2.Devices[0].DeviceInfo.MemLimit))
	// Each field is a naturally aligned word, as required for atomic access
	// from both sides of the mapping.
	require.Zero(t, (entry+info+unsafe.Offsetof(raw.V2.Devices[0].DeviceInfo.UpLimit))%4)
	require.Zero(t, (entry+info+unsafe.Offsetof(raw.V2.Devices[0].DeviceInfo.MemLimit))%8)
}

func TestSetDeviceLimitsUpdatesLiveMapping(t *testing.T) {
	base := t.TempDir()
	pod := NewPodIdentifier("namespace", "pod")
	handle, err := CreateSharedMemoryHandle(base, pod, []DeviceConfig{
		{DeviceIdx: 0, DeviceUUID: "GPU-0", UpLimit: 100, MemLimit: 16 << 30, TotalCudaCores: 1024},
		{DeviceIdx: 2, DeviceUUID: "GPU-2", UpLimit: 50, MemLimit: 8 << 30, TotalCudaCores: 2048},
	})
	require.NoError(t, err)
	defer func() { _ = handle.Close() }()
	// A second mapping of the same file, as held by a limiter process.
	reader, err := OpenSharedMemoryHandle(base, pod)
	require.NoError(t, err)
	defer func() { _ = reader.Close() }()

	shmPath := filepath.Join(pod.ToPath(base), ShmPathSuffix)
	before, err := os.Stat(shmPath)
	require.NoError(t, err)
	state := handle.GetState()
	state.UpdateHeartbeat(1234)
	state.SetPodMemoryUsed(2, 3<<30)
	state.AddPID(77)
	state.V2.Devices[2].DeviceInfo.SetERLCurrentTokens(42)

	readFile := func(deviceIdx int) (uint32, uint64) {
		t.Helper()
		raw, err := os.ReadFile(shmPath)
		require.NoError(t, err)
		entry := 8 + deviceIdx*136
		return binary.LittleEndian.Uint32(raw[entry+64:]), binary.LittleEndian.Uint64(raw[entry+72:])
	}
	readMapping := func(deviceIdx int) (uint32, uint64) {
		info := &reader.GetState().V2.Devices[deviceIdx].DeviceInfo
		return atomic.LoadUint32(&info.UpLimit), atomic.LoadUint64(&info.MemLimit)
	}
	assertLimits := func(deviceIdx int, upLimit uint32, memLimit uint64) {
		t.Helper()
		gotUp, gotMem := readFile(deviceIdx)
		require.Equal(t, upLimit, gotUp, "up_limit bytes of device %d", deviceIdx)
		require.Equal(t, memLimit, gotMem, "mem_limit bytes of device %d", deviceIdx)
		gotUp, gotMem = readMapping(deviceIdx)
		require.Equal(t, upLimit, gotUp, "up_limit of device %d in a second mapping", deviceIdx)
		require.Equal(t, memLimit, gotMem, "mem_limit of device %d in a second mapping", deviceIdx)
	}
	assertLimits(0, 100, 16<<30)
	assertLimits(2, 50, 8<<30)

	// Decrease both limits of one device; the other device is unaffected.
	require.True(t, state.SetDeviceLimits(2, 20, 2<<30))
	assertLimits(2, 20, 2<<30)
	assertLimits(0, 100, 16<<30)
	// Increase them again, one field at a time.
	require.True(t, state.SetDeviceLimits(2, 20, 12<<30))
	assertLimits(2, 20, 12<<30)
	require.True(t, state.SetDeviceLimits(2, 90, 12<<30))
	assertLimits(2, 90, 12<<30)
	// An unlimited device becomes limited.
	require.True(t, state.SetDeviceLimits(0, 35, 16<<30))
	assertLimits(0, 35, 16<<30)
	// Republishing the same limits reports no change.
	require.False(t, state.SetDeviceLimits(0, 35, 16<<30))

	// Inactive and out-of-range slots are rejected and left untouched.
	require.False(t, state.SetDeviceLimits(1, 10, 1<<30))
	require.False(t, state.SetDeviceLimits(-1, 10, 1<<30))
	require.False(t, state.SetDeviceLimits(MaxDevices, 10, 1<<30))
	assertLimits(1, 0, 0)

	// The update is in place and leaves the rest of the limiter state alone.
	after, err := os.Stat(shmPath)
	require.NoError(t, err)
	require.True(t, os.SameFile(before, after))
	require.Equal(t, before.Size(), after.Size())
	view := reader.GetState()
	require.Equal(t, uint64(1234), view.GetLastHeartbeat())
	require.Equal(t, uint64(3<<30), atomic.LoadUint64(&view.V2.Devices[2].DeviceInfo.PodMemoryUsed))
	require.Equal(t, uint32(2048), view.V2.Devices[2].DeviceInfo.TotalCudaCores)
	require.Equal(t, float64(42), view.V2.Devices[2].DeviceInfo.GetERLCurrentTokens())
	require.Equal(t, "GPU-2", view.V2.Devices[2].GetUUID())
	require.Equal(t, []int{77}, view.GetAllPIDs())
	require.Equal(t, 2, view.DeviceCount())
}

func TestSetDeviceLimitsSupportsV1State(t *testing.T) {
	v1, err := NewSharedDeviceStateV1([]DeviceConfig{{DeviceIdx: 1, UpLimit: 60, MemLimit: 4 << 30}})
	require.NoError(t, err)
	state := &SharedDeviceState{V1: v1}
	require.True(t, state.SetDeviceLimits(1, 25, 1<<30))
	require.Equal(t, uint32(25), v1.Devices[1].DeviceInfo.UpLimit)
	require.Equal(t, uint64(1<<30), v1.Devices[1].DeviceInfo.MemLimit)
	require.False(t, state.SetDeviceLimits(0, 25, 1<<30))
}
