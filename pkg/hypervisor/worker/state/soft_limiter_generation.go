//go:build darwin || linux || freebsd || netbsd

package worker

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// PrepareWorkerSharedMemory binds the legacy shm path to a Pod UID. The UID is
// kept in the backing filename, leaving the shared-memory ABI unchanged. The
// caller must serialize this operation with worker removal and name reuse.
func PrepareWorkerSharedMemory(
	basePath string, pod *PodIdentifier, uid string, configs []DeviceConfig, recoverLegacy bool,
) (*SharedMemoryHandle, error) {
	if uid == "" || strings.ContainsAny(uid, "/\\") || strings.Contains(uid, "..") {
		return nil, fmt.Errorf("invalid shared memory worker UID: %q", uid)
	}
	// Reuse the public path validation before constructing generation paths.
	existing, openErr := OpenSharedMemoryHandle(basePath, pod)
	if existing != nil {
		defer func() { _ = existing.Close() }()
	}
	if openErr != nil && !errors.Is(openErr, os.ErrNotExist) {
		return nil, openErr
	}
	podPath := pod.ToPath(basePath)
	if err := os.MkdirAll(podPath, 0755); err != nil {
		return nil, err
	}
	shmPath := filepath.Join(podPath, ShmPathSuffix)
	generation := ".shm-uid-" + uid
	generationPath := filepath.Join(podPath, generation)
	previous, _ := os.Readlink(shmPath)
	if previous == generation {
		return openSharedMemoryHandle(generationPath)
	}
	// Upgrade running workers created before UID ownership was persisted by
	// linking their existing inode. Their CUDA processes keep using that inode.
	if existing != nil && previous == "" && recoverLegacy {
		if err := os.Link(shmPath, generationPath); err != nil && !os.IsExist(err) {
			return nil, err
		}
	}
	handle, err := createSharedMemoryHandle(generationPath, configs)
	if err != nil {
		return nil, err
	}
	if err := publishSharedMemoryGeneration(shmPath, generation); err != nil {
		_ = handle.Close()
		return nil, err
	}
	if previous != "" && previous != generation && filepath.Base(previous) == previous &&
		strings.HasPrefix(previous, ".shm-uid-") {
		_ = os.Remove(filepath.Join(podPath, previous))
	}
	return handle, nil
}

func publishSharedMemoryGeneration(shmPath, generation string) error {
	file, err := os.CreateTemp(filepath.Dir(shmPath), ".shm-link-*")
	if err != nil {
		return err
	}
	temporary := file.Name()
	_ = file.Close()
	defer func() { _ = os.Remove(temporary) }()
	if err := os.Remove(temporary); err != nil {
		return err
	}
	if err := os.Symlink(generation, temporary); err != nil {
		return err
	}
	return os.Rename(temporary, shmPath)
}
