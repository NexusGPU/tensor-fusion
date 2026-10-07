//go:build darwin || linux || freebsd || netbsd

package worker

import (
	"bytes"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCreateSharedMemoryPreservesExistingMapping(t *testing.T) {
	base := t.TempDir()
	pod := NewPodIdentifier("namespace", "worker")
	first, err := CreateSharedMemoryHandle(base, pod, createTestConfigs())
	require.NoError(t, err)
	t.Cleanup(func() { _ = first.Close() })
	first.GetState().SetPodMemoryUsed(0, 512)
	first.GetState().V2.Devices[0].DeviceInfo.SetERLCurrentTokens(3)
	before, err := os.Stat(filepath.Join(pod.ToPath(base), ShmPathSuffix))
	require.NoError(t, err)

	second, err := CreateSharedMemoryHandle(base, pod, createTestConfigs())
	require.NoError(t, err)
	t.Cleanup(func() { _ = second.Close() })
	after, err := os.Stat(filepath.Join(pod.ToPath(base), ShmPathSuffix))
	require.NoError(t, err)
	require.True(t, os.SameFile(before, after), "retries must use the inode mapped by the worker")
	require.Equal(t, uint64(512), second.GetState().GetPodMemoryUsed(0))
	require.Equal(t, float64(3), second.GetState().V2.Devices[0].DeviceInfo.GetERLCurrentTokens())
}

func TestConcurrentSharedMemoryCreatorsConverge(t *testing.T) {
	base := t.TempDir()
	pod := NewPodIdentifier("namespace", "worker")
	const count = 16
	handles := make([]*SharedMemoryHandle, count)
	errs := make([]error, count)
	var wg sync.WaitGroup
	for i := range count {
		wg.Go(func() { handles[i], errs[i] = CreateSharedMemoryHandle(base, pod, createTestConfigs()) })
	}
	wg.Wait()
	for i, handle := range handles {
		require.NoError(t, errs[i])
		t.Cleanup(func() { _ = handle.Close() })
	}
	handles[0].GetState().SetPodMemoryUsed(0, 512)
	for _, handle := range handles {
		require.Equal(t, uint64(512), handle.GetState().GetPodMemoryUsed(0))
	}
}

func TestSharedMemoryFollowsPodUIDAcrossRestartsAndNameReuse(t *testing.T) {
	base := t.TempDir()
	pod := NewPodIdentifier("namespace", "worker")
	configs := createTestConfigs()
	first, err := PrepareWorkerSharedMemory(base, pod, "first-uid", configs, false)
	require.NoError(t, err)
	t.Cleanup(func() { _ = first.Close() })
	first.GetState().SetPodMemoryUsed(0, 512)
	before, err := os.Stat(filepath.Join(pod.ToPath(base), ShmPathSuffix))
	require.NoError(t, err)

	// The worker need not have reached Running before hypervisor restarts.
	reopened, err := PrepareWorkerSharedMemory(base, pod, "first-uid", configs, false)
	require.NoError(t, err)
	t.Cleanup(func() { _ = reopened.Close() })
	require.Equal(t, uint64(512), reopened.GetState().GetPodMemoryUsed(0))
	after, err := os.Stat(filepath.Join(pod.ToPath(base), ShmPathSuffix))
	require.NoError(t, err)
	require.True(t, os.SameFile(before, after))

	configs[0].MemLimit *= 2
	replacement, err := PrepareWorkerSharedMemory(base, pod, "replacement-uid", configs, true)
	require.NoError(t, err)
	t.Cleanup(func() { _ = replacement.Close() })
	after, err = os.Stat(filepath.Join(pod.ToPath(base), ShmPathSuffix))
	require.NoError(t, err)
	require.False(t, os.SameFile(before, after))
	require.Equal(t, uint64(0), replacement.GetState().GetPodMemoryUsed(0))
	require.Equal(t, configs[0].MemLimit, replacement.GetState().V2.Devices[0].DeviceInfo.MemLimit)
	require.Equal(t, uint64(512), first.GetState().GetPodMemoryUsed(0), "never truncate a live old mapping")
	legacyClient, err := OpenSharedMemoryHandle(base, pod)
	require.NoError(t, err)
	defer func() { _ = legacyClient.Close() }()
	replacement.GetState().SetPodMemoryUsed(0, 42)
	require.Equal(t, uint64(42), legacyClient.GetState().GetPodMemoryUsed(0))
}

func TestSharedMemoryAdoptsLegacyRunningWorkerWithoutReplacingInode(t *testing.T) {
	base := t.TempDir()
	pod := NewPodIdentifier("namespace", "worker")
	original, err := CreateSharedMemoryHandle(base, pod, createTestConfigs())
	require.NoError(t, err)
	defer func() { _ = original.Close() }()
	original.GetState().SetPodMemoryUsed(0, 512)
	adopted, err := PrepareWorkerSharedMemory(base, pod, "running-uid", createTestConfigs(), true)
	require.NoError(t, err)
	defer func() { _ = adopted.Close() }()
	require.Equal(t, uint64(512), adopted.GetState().GetPodMemoryUsed(0))
	adopted.GetState().SetPodMemoryUsed(0, 256)
	require.Equal(t, uint64(256), original.GetState().GetPodMemoryUsed(0))
}

func TestSharedMemoryDoesNotOverwriteInvalidExistingState(t *testing.T) {
	for _, invalid := range []string{"size", "discriminant"} {
		t.Run(invalid, func(t *testing.T) {
			base := t.TempDir()
			pod := NewPodIdentifier("namespace", "worker")
			created, err := CreateSharedMemoryHandle(base, pod, createTestConfigs())
			require.NoError(t, err)
			require.NoError(t, created.Close())
			path := filepath.Join(pod.ToPath(base), ShmPathSuffix)
			corrupt, err := os.ReadFile(path)
			require.NoError(t, err)
			if invalid == "size" {
				corrupt = corrupt[:16]
			} else {
				copy(corrupt[:4], bytes.Repeat([]byte{0xff}, 4))
			}
			require.NoError(t, os.WriteFile(path, corrupt, 0600))
			_, err = PrepareWorkerSharedMemory(base, pod, "running-uid", createTestConfigs(), true)
			require.Error(t, err)
			after, err := os.ReadFile(path)
			require.NoError(t, err)
			require.Equal(t, corrupt, after)
		})
	}
}

func TestSharedMemoryRejectsLinksOutsidePodDirectory(t *testing.T) {
	for _, name := range []string{ShmPathSuffix, ".shm-uid-attacker"} {
		t.Run(name, func(t *testing.T) {
			base := t.TempDir()
			victim := NewPodIdentifier("namespace", "victim")
			original, err := CreateSharedMemoryHandle(base, victim, createTestConfigs())
			require.NoError(t, err)
			defer func() { _ = original.Close() }()
			original.GetState().SetPodMemoryUsed(0, 512)
			attacker := NewPodIdentifier("namespace", "attacker")
			require.NoError(t, os.MkdirAll(attacker.ToPath(base), 0755))
			require.NoError(t, os.Symlink(
				filepath.Join(victim.ToPath(base), ShmPathSuffix), filepath.Join(attacker.ToPath(base), name),
			))
			_, err = PrepareWorkerSharedMemory(base, attacker, "attacker", createTestConfigs(), false)
			require.Error(t, err, "a writable Pod directory must not redirect the hypervisor to another Pod")
			require.Equal(t, uint64(512), original.GetState().GetPodMemoryUsed(0))
		})
	}
}

func TestProcessRegistrationRetriesBusySharedMemoryLock(t *testing.T) {
	state, err := NewSharedDeviceState(createTestConfigs())
	require.NoError(t, err)
	state.V2.PIDs.Lock()
	require.False(t, state.TryAddPID(42), "registration must not spin on a held lock")
	state.V2.PIDs.Unlock()
	require.True(t, state.TryAddPID(42))
	require.True(t, state.TryAddPID(42), "registration retries remain idempotent")
	require.Equal(t, []int{42}, state.GetAllPIDs())
}
