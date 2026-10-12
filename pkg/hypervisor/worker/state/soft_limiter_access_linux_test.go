//go:build linux

package worker

import (
	"os"
	"runtime"
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/sys/unix"
)

// A workload container sees only the pod directory (it is the root of its
// mount) and opens the published file read-write as an arbitrary UID.
func TestUnprivilegedWorkloadUserCanOpenSharedMemoryReadWrite(t *testing.T) {
	if os.Geteuid() != 0 {
		t.Skip("switching the filesystem UID requires root")
	}
	const workloadID = 65534
	withUmask(t, 0o022)
	base := t.TempDir()
	pod := NewPodIdentifier("namespace", "worker")
	handle, err := PrepareWorkerSharedMemory(base, pod, testWorkerUID, createTestConfigs(), false)
	require.NoError(t, err)
	defer func() { _ = handle.Close() }()

	// The temporary parents are private to root; the container never walks them.
	dir, err := unix.Open(pod.ToPath(base), unix.O_PATH|unix.O_DIRECTORY, 0)
	require.NoError(t, err)
	defer func() { _ = unix.Close(dir) }()

	// The filesystem identity is per thread, so stay on this one until restored.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	groups, err := unix.Getgroups()
	require.NoError(t, err)
	// Without root's groups only the "other" permission bits apply.
	require.NoError(t, unix.Setgroups([]int{workloadID}))
	require.NoError(t, unix.Setfsgid(workloadID))
	require.NoError(t, unix.Setfsuid(workloadID))
	fd, openErr := unix.Openat(dir, ShmPathSuffix, unix.O_RDWR, 0)
	require.NoError(t, unix.Setfsuid(0))
	require.NoError(t, unix.Setfsgid(0))
	require.NoError(t, unix.Setgroups(groups))

	require.NoError(t, openErr, "the limiter opens the published file read-write as the workload user")
	require.NoError(t, unix.Close(fd))
}
