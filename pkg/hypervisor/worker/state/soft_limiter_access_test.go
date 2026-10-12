//go:build darwin || linux || freebsd || netbsd

package worker

import (
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/require"
)

const testWorkerUID = "worker-uid"

func testGenerationPath(base string, pod *PodIdentifier, uid string) string {
	return filepath.Join(pod.ToPath(base), ".shm-uid-"+uid)
}

func withUmask(t *testing.T, mask int) {
	t.Helper()
	previous := syscall.Umask(mask)
	t.Cleanup(func() { syscall.Umask(previous) })
}

// The workload container runs with its own UID and maps the published file
// read-write, so neither the file nor its directory may depend on the umask the
// hypervisor happens to run with.
func requireWorkloadAccessModes(t *testing.T, podPath string) {
	t.Helper()
	dir, err := os.Stat(podPath)
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0o005), dir.Mode().Perm()&0o005, "pod directory must stay traversable: %v", dir.Mode())
	file, err := os.Stat(filepath.Join(podPath, ShmPathSuffix))
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0o666), file.Mode().Perm(), "shared memory must be read-write for the workload user")
}

func TestSharedMemoryModeDoesNotDependOnProcessUmask(t *testing.T) {
	for _, mask := range []int{0o022, 0o077} {
		t.Run(fmt.Sprintf("prepared/umask-%03o", mask), func(t *testing.T) {
			withUmask(t, mask)
			base := t.TempDir()
			pod := NewPodIdentifier("namespace", "worker")
			handle, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, createTestConfigs(), false)
			require.NoError(t, err)
			defer func() { _ = handle.Close() }()
			requireWorkloadAccessModes(t, pod.ToPath(base))
		})
		t.Run(fmt.Sprintf("created/umask-%03o", mask), func(t *testing.T) {
			withUmask(t, mask)
			base := t.TempDir()
			pod := NewPodIdentifier("namespace", "worker")
			handle, err := CreateSharedMemoryHandle(base, pod, createTestConfigs())
			require.NoError(t, err)
			defer func() { _ = handle.Close() }()
			requireWorkloadAccessModes(t, pod.ToPath(base))
		})
	}
}

func TestSharedMemoryModeIsRepairedWhenReopened(t *testing.T) {
	t.Run("current generation", func(t *testing.T) {
		base := t.TempDir()
		pod := NewPodIdentifier("namespace", "worker")
		first, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, createTestConfigs(), false)
		require.NoError(t, err)
		first.GetState().SetPodMemoryUsed(0, 512)
		require.NoError(t, first.Close())
		require.NoError(t, os.Chmod(testGenerationPath(base, pod, testWorkerUID), 0o644))
		require.NoError(t, os.Chmod(pod.ToPath(base), 0o700))

		reopened, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, createTestConfigs(), true)
		require.NoError(t, err)
		defer func() { _ = reopened.Close() }()
		requireWorkloadAccessModes(t, pod.ToPath(base))
		require.Equal(t, uint64(512), reopened.GetState().GetPodMemoryUsed(0), "repairing the mode must keep the live state")
	})
	t.Run("file adopted from an earlier release", func(t *testing.T) {
		base := t.TempDir()
		pod := NewPodIdentifier("namespace", "worker")
		original, err := CreateSharedMemoryHandle(base, pod, createTestConfigs())
		require.NoError(t, err)
		original.GetState().SetPodMemoryUsed(0, 512)
		require.NoError(t, original.Close())
		shmPath := filepath.Join(pod.ToPath(base), ShmPathSuffix)
		require.NoError(t, os.Chmod(shmPath, 0o644))
		before, err := os.Stat(shmPath)
		require.NoError(t, err)

		adopted, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, createTestConfigs(), true)
		require.NoError(t, err)
		defer func() { _ = adopted.Close() }()
		requireWorkloadAccessModes(t, pod.ToPath(base))
		after, err := os.Stat(shmPath)
		require.NoError(t, err)
		require.True(t, os.SameFile(before, after), "running workloads keep using the adopted inode")
		require.Equal(t, uint64(512), adopted.GetState().GetPodMemoryUsed(0))
	})
}

// The backing file is writable by the workload's processes, so its length is
// not relied on. A file that no longer covers a live mapping must surface as a
// failed access on this one handle and never as a process-wide fault.
func TestSharedMemoryAccessSurvivesShrunkBackingFile(t *testing.T) {
	accesses := map[string]func(*SharedDeviceState){
		"heartbeat write": func(state *SharedDeviceState) { state.UpdateHeartbeat(42) },
		"usage write":     func(state *SharedDeviceState) { state.SetPodMemoryUsed(0, 1) },
		"limit read":      func(state *SharedDeviceState) { _ = state.V2.Devices[0].DeviceInfo.MemLimit },
		"token read":      func(state *SharedDeviceState) { _ = state.V2.Devices[0].DeviceInfo.GetERLCurrentTokens() },
		"process set":     func(state *SharedDeviceState) { _ = state.TryAddPID(42) },
	}
	for name, access := range accesses {
		t.Run(name, func(t *testing.T) {
			base := t.TempDir()
			pod := NewPodIdentifier("namespace", "worker")
			handle, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, createTestConfigs(), false)
			require.NoError(t, err)
			defer func() { _ = handle.Close() }()
			require.True(t, handle.WithState(access), "access works while the file is intact")
			require.False(t, handle.Faulted())

			require.NoError(t, os.Truncate(testGenerationPath(base, pod, testWorkerUID), 0))

			require.False(t, handle.WithState(access), "a faulting access is reported, not fatal")
			require.True(t, handle.Faulted())
			require.Nil(t, handle.GetState(), "a faulted mapping is no longer handed out")
			touched := false
			require.False(t, handle.WithState(func(*SharedDeviceState) { touched = true }))
			require.False(t, touched, "a faulted mapping is not accessed again")
			require.NoError(t, handle.Close())
		})
	}
}

func TestSharedMemoryAccessSurvivesPartiallyShrunkBackingFile(t *testing.T) {
	base := t.TempDir()
	pod := NewPodIdentifier("namespace", "worker")
	handle, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, createTestConfigs(), false)
	require.NoError(t, err)
	defer func() { _ = handle.Close() }()
	require.NoError(t, os.Truncate(testGenerationPath(base, pod, testWorkerUID), 1))

	// The process set spans the tail of the mapping, past the surviving page.
	require.False(t, handle.WithState(func(state *SharedDeviceState) { _ = state.TryAddPID(42) }))
	require.True(t, handle.Faulted())
}

// A panic that is not a fault inside the mapping is a programming error and
// must keep propagating.
func TestSharedMemoryAccessDoesNotSwallowUnrelatedPanics(t *testing.T) {
	handle, err := PrepareWorkerSharedMemory(
		t.TempDir(), NewPodIdentifier("namespace", "worker"), testWorkerUID, createTestConfigs(), false,
	)
	require.NoError(t, err)
	defer func() { _ = handle.Close() }()
	require.PanicsWithValue(t, "unrelated", func() {
		handle.WithState(func(*SharedDeviceState) { panic("unrelated") })
	})
	var missing *SharedDeviceState
	require.Panics(t, func() {
		handle.WithState(func(*SharedDeviceState) { _ = missing.V2 })
	})
	require.False(t, handle.Faulted())
	require.True(t, handle.WithState(func(state *SharedDeviceState) { state.UpdateHeartbeat(1) }))
}

func requireOnlyPublishedGeneration(t *testing.T, podPath, uid string) {
	t.Helper()
	entries, err := os.ReadDir(podPath)
	require.NoError(t, err)
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}
	require.ElementsMatch(t, []string{ShmPathSuffix, ".shm-uid-" + uid}, names)
	target, err := os.Readlink(filepath.Join(podPath, ShmPathSuffix))
	require.NoError(t, err)
	require.Equal(t, ".shm-uid-"+uid, target)
}

// Once a worker's own segment stops being a valid mapping, the worker gets a
// freshly initialised segment at the same published path.
func TestWorkerSharedMemoryIsRepublishedWhenItsSegmentBecomesUnusable(t *testing.T) {
	stateSize := int64(unsafe.Sizeof(rustSharedDeviceStateV2Layout{}))
	damage := map[string]func(t *testing.T, generationPath string){
		"shrunk to nothing": func(t *testing.T, path string) { require.NoError(t, os.Truncate(path, 0)) },
		"shrunk partially":  func(t *testing.T, path string) { require.NoError(t, os.Truncate(path, stateSize/2)) },
		"extended":          func(t *testing.T, path string) { require.NoError(t, os.Truncate(path, stateSize+4096)) },
		"shrunk and restored": func(t *testing.T, path string) {
			require.NoError(t, os.Truncate(path, 0))
			require.NoError(t, os.Truncate(path, stateSize))
		},
		"removed": func(t *testing.T, path string) { require.NoError(t, os.Remove(path)) },
	}
	for name, apply := range damage {
		t.Run(name, func(t *testing.T) {
			base := t.TempDir()
			pod := NewPodIdentifier("namespace", "worker")
			configs := createTestConfigs()
			first, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, configs, false)
			require.NoError(t, err)
			first.GetState().SetPodMemoryUsed(0, 512)
			apply(t, testGenerationPath(base, pod, testWorkerUID))
			require.NoError(t, first.Close())

			recovered, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, configs, true)
			require.NoError(t, err)
			defer func() { _ = recovered.Close() }()
			require.True(t, recovered.WithState(func(state *SharedDeviceState) {
				require.Equal(t, uint64(0), state.GetPodMemoryUsed(0))
				require.Equal(t, configs[0].MemLimit, state.V2.Devices[0].DeviceInfo.MemLimit)
				state.UpdateHeartbeat(7)
			}))
			requireOnlyPublishedGeneration(t, pod.ToPath(base), testWorkerUID)
			requireWorkloadAccessModes(t, pod.ToPath(base))

			// A workload process opening the published path sees the new segment.
			client, err := OpenSharedMemoryHandle(base, pod)
			require.NoError(t, err)
			defer func() { _ = client.Close() }()
			require.Equal(t, uint64(7), client.GetState().GetLastHeartbeat())
		})
	}
}

// A segment left unusable by an earlier pod must not block the next pod that
// reuses its namespace and name.
func TestUnusableSegmentOfPreviousPodDoesNotBlockItsSuccessor(t *testing.T) {
	base := t.TempDir()
	pod := NewPodIdentifier("namespace", "worker")
	first, err := PrepareWorkerSharedMemory(base, pod, "first-uid", createTestConfigs(), false)
	require.NoError(t, err)
	require.NoError(t, os.Truncate(testGenerationPath(base, pod, "first-uid"), 0))
	require.NoError(t, first.Close())

	successor, err := PrepareWorkerSharedMemory(base, pod, "successor-uid", createTestConfigs(), false)
	require.NoError(t, err)
	defer func() { _ = successor.Close() }()
	require.True(t, successor.WithState(func(state *SharedDeviceState) { state.UpdateHeartbeat(7) }))
	requireOnlyPublishedGeneration(t, pod.ToPath(base), "successor-uid")
}

func TestProcessSetListingIsBoundedByItsCapacity(t *testing.T) {
	set := NewPIDSet()
	require.True(t, set.InsertIfAbsent(7))
	// The stored length is read from the mapping and validated.
	set.len = ^uintptr(0)
	require.Equal(t, []int{7}, set.Values())
}
