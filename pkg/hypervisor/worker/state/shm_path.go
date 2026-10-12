//go:build darwin || linux || freebsd || netbsd

package worker

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

// PodIdentifier contains namespace and name
type PodIdentifier struct {
	Namespace string
	Name      string
}

// NewPodIdentifier creates a new PodIdentifier
func NewPodIdentifier(namespace, name string) *PodIdentifier {
	return &PodIdentifier{
		Namespace: namespace,
		Name:      name,
	}
}

// ToPath returns the path for this pod identifier
func (p *PodIdentifier) ToPath(basePath string) string {
	return filepath.Join(basePath, p.Namespace, p.Name)
}

// FromShmFilePath parses a PodIdentifier from a full shared memory path
// Path format: {base_path}/{namespace}/{name}/shm
func FromShmFilePath(path string) (*PodIdentifier, error) {
	path = filepath.Clean(path)
	components := strings.Split(path, string(filepath.Separator))

	// Filter out empty components (from leading/trailing separators)
	var filtered []string
	for _, comp := range components {
		if comp != "" {
			filtered = append(filtered, comp)
		}
	}
	components = filtered

	// Need at least: namespace, name, and "shm" (3 components minimum)
	if len(components) < 3 {
		return nil, fmt.Errorf("invalid path format: %s (need at least namespace/name/shm)", path)
	}

	// Extract the last 3 components: {namespace}/{name}/shm
	compLen := len(components)

	// Verify the last component is "shm"
	if components[compLen-1] != ShmPathSuffix {
		return nil, fmt.Errorf("invalid path format: %s (last component must be 'shm')", path)
	}

	namespace := components[compLen-3]
	name := components[compLen-2]

	// Validate namespace and name are not empty
	if namespace == "" || name == "" {
		return nil, fmt.Errorf("invalid path format: %s (namespace and name must be non-empty)", path)
	}

	return NewPodIdentifier(namespace, name), nil
}

// String returns the string representation
func (p *PodIdentifier) String() string {
	return fmt.Sprintf("%s/%s", p.Namespace, p.Name)
}

// CleanupEmptyParentDirectories removes empty parent directories after removing a file
func CleanupEmptyParentDirectories(filePath string, stopAtPath *string) error {
	parentDir := filepath.Dir(filePath)

	// Skip if we've reached the stop path
	if stopAtPath != nil && parentDir == *stopAtPath {
		return nil
	}

	// Try to remove the immediate parent directory if it's empty
	entries, err := os.ReadDir(parentDir)
	if err != nil {
		return err
	}

	if len(entries) == 0 {
		if err := os.Remove(parentDir); err != nil {
			return err
		}

		// Recursively try to remove parent directories if they're also empty
		return CleanupEmptyParentDirectories(parentDir, stopAtPath)
	}

	return nil
}

// The limiter runs with the workload's own UID and updates token counters in
// place, and the pod directory is the root of its mount. Both modes are set
// explicitly on the opened file descriptor so they do not depend on the umask
// of whichever process creates them.
const (
	sharedMemoryFileMode os.FileMode = 0o666
	sharedMemoryDirMode  os.FileMode = 0o755
)

// ensureSharedMemoryDir creates the pod directory and makes sure the workload
// user can traverse it, also for directories created by an earlier release or
// by the kubelet before the hypervisor saw the pod.
func ensureSharedMemoryDir(podPath string) error {
	if err := os.MkdirAll(podPath, sharedMemoryDirMode); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}
	dir, err := os.OpenFile(podPath, os.O_RDONLY|syscall.O_DIRECTORY|syscall.O_NOFOLLOW, 0)
	if err != nil {
		return fmt.Errorf("failed to open directory: %w", err)
	}
	defer func() { _ = dir.Close() }()
	return ensureMode(dir, sharedMemoryDirMode)
}

// ensureMode adds any missing permission bits through the open descriptor.
func ensureMode(file *os.File, required os.FileMode) error {
	stat, err := file.Stat()
	if err != nil {
		return fmt.Errorf("failed to stat %s: %w", file.Name(), err)
	}
	if current := stat.Mode().Perm(); current&required != required {
		if err := file.Chmod(current | required); err != nil {
			return fmt.Errorf("failed to set mode of %s: %w", file.Name(), err)
		}
	}
	return nil
}

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
	unusable := errors.Is(openErr, errSharedMemoryUnusable)
	if openErr != nil && !errors.Is(openErr, os.ErrNotExist) && !unusable {
		return nil, openErr
	}
	podPath := pod.ToPath(basePath)
	if err := ensureSharedMemoryDir(podPath); err != nil {
		return nil, err
	}
	shmPath := filepath.Join(podPath, ShmPathSuffix)
	generation := ".shm-uid-" + uid
	generationPath := filepath.Join(podPath, generation)
	previous, _ := os.Readlink(shmPath)
	if previous == generation {
		// The file is writable by the workload's processes; its length and
		// header are validated when it is opened. Nothing can be salvaged from
		// a segment that does not pass; give the worker a fresh one under the
		// same name. The swap is atomic, so openers never observe a missing or
		// partial file.
		if unusable {
			return initSharedMemoryHandle(generationPath, configs, true)
		}
		return createSharedMemoryHandle(generationPath, configs)
	}
	// State that predates UID ownership has an unknown writer and layout. Keep
	// refusing it instead of overwriting what a running workload may still map.
	if unusable && previous == "" {
		return nil, openErr
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
