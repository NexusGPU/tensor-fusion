//go:build linux

package worker

import (
	"context"
	"os"
	"os/exec"
	"sync/atomic"
	"testing"
	"time"
)

func TestShmMutexRecoversAfterHypervisorRestart(t *testing.T) {
	if os.Getenv("TF_TEST_STALE_SHM_MUTEX") == "1" {
		// The mapped PID field survives the hypervisor that initialized it.
		// Use a PID outside the Linux PID range so the old holder is absent.
		const oldPID = uintptr(1 << 30)
		mutex := ShmMutex[int]{PID: oldPID, LockField: oldPID}
		mutex.Lock()
		if got := atomic.LoadUintptr(&mutex.LockField); got != uintptr(os.Getpid()) {
			t.Fatalf("lock must be owned by the current process, got %d", got)
		}
		mutex.Unlock()
		if atomic.LoadUintptr(&mutex.LockField) != 0 {
			t.Fatal("unlock retained a stale owner")
		}
		return
	}
	// Bound the regression: the old implementation spins forever.
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestShmMutexRecoversAfterHypervisorRestart$")
	cmd.Env = append(os.Environ(), "TF_TEST_STALE_SHM_MUTEX=1")
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("shared-memory mutex did not recover after restart: %v\n%s", err, output)
	}
}
