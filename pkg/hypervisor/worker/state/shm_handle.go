//go:build darwin || linux || freebsd || netbsd

package worker

import (
	"crypto/rand"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"
)

// errSharedMemoryUnusable marks an existing file that cannot be mapped as the
// published layout: its length or header no longer match.
var errSharedMemoryUnusable = errors.New("unusable shared memory")

// errSharedMemoryFault reports an access that faulted because the backing file
// no longer covers the mapping.
var errSharedMemoryFault = fmt.Errorf("%w: backing file no longer covers the mapping", errSharedMemoryUnusable)

// accessMapping runs fn, which may read or write the file-backed mapping data.
// The mapping is writable by the workload's processes; values read from it are
// validated, and so is the access itself: the length of the backing file is not
// relied on, and an access that faults because the file no longer covers the
// mapping is turned into errSharedMemoryFault for this one mapping. Fault
// handling is per goroutine, so every access to a mapping has to go through
// here on the goroutine that performs it.
func accessMapping(data []byte, fn func()) (err error) {
	previous := debug.SetPanicOnFault(true)
	defer func() {
		debug.SetPanicOnFault(previous)
		if recovered := recover(); recovered != nil {
			if !isMappingFault(recovered, data) {
				panic(recovered)
			}
			err = errSharedMemoryFault
		}
	}()
	fn()
	return nil
}

// isMappingFault reports whether a recovered panic is a memory fault at an
// address inside data. Anything else is a programming error.
func isMappingFault(recovered any, data []byte) bool {
	fault, ok := recovered.(interface {
		error
		Addr() uintptr
	})
	if !ok || len(data) == 0 {
		return false
	}
	start := uintptr(unsafe.Pointer(unsafe.SliceData(data)))
	return fault.Addr() >= start && fault.Addr()-start < uintptr(len(data))
}

// SharedMemoryHandle manages a shared memory mapping
type SharedMemoryHandle struct {
	mu       sync.RWMutex
	path     string
	data     []byte
	state    *SharedDeviceState
	file     *os.File
	fileSize int64
	// faulted is set once an access faulted; the mapping is not touched again.
	faulted atomic.Bool
}

// CreateSharedMemoryHandle creates a fully initialized mapping, or opens the
// existing mapping without resetting counters or replacing its inode.
// Namespace and pod name are expected to be single path components.
func CreateSharedMemoryHandle(
	basePath string, pod *PodIdentifier, configs []DeviceConfig,
) (*SharedMemoryHandle, error) {
	if pod == nil {
		return nil, fmt.Errorf("pod identifier is required")
	}
	if pod.Namespace == "" || pod.Name == "" {
		return nil, fmt.Errorf("pod identifier must include namespace and name")
	}
	if strings.Contains(pod.Namespace, "/") ||
		strings.Contains(pod.Namespace, "\\") ||
		strings.Contains(pod.Namespace, "..") {
		return nil, fmt.Errorf("invalid namespace path component: %q", pod.Namespace)
	}
	if strings.Contains(pod.Name, "/") || strings.Contains(pod.Name, "\\") || strings.Contains(pod.Name, "..") {
		return nil, fmt.Errorf("invalid pod name path component: %q", pod.Name)
	}

	podPath := filepath.Join(basePath, pod.Namespace, pod.Name)
	shmPath := filepath.Join(podPath, ShmPathSuffix)
	if handle, err := OpenSharedMemoryHandle(basePath, pod); err == nil {
		return handle, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}

	// Create directory if it doesn't exist
	if err := ensureSharedMemoryDir(podPath); err != nil {
		return nil, err
	}
	return createSharedMemoryHandle(shmPath, configs)
}

func createSharedMemoryHandle(shmPath string, configs []DeviceConfig) (*SharedMemoryHandle, error) {
	if handle, err := openSharedMemoryHandle(shmPath); err == nil {
		return handle, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	return initSharedMemoryHandle(shmPath, configs, false)
}

// initSharedMemoryHandle publishes a fully initialized mapping at shmPath. With
// replace it atomically takes the place of whatever is published there, which
// is only correct for a file the caller has found to be unusable.
func initSharedMemoryHandle(shmPath string, configs []DeviceConfig, replace bool) (*SharedMemoryHandle, error) {
	// Rust maps this file as `#[repr(C)] enum SharedDeviceState::V2`.
	stateSize := int(unsafe.Sizeof(rustSharedDeviceStateV2Layout{}))

	// Publish only after initialization. O_TRUNC would reset live mappings and
	// can SIGBUS another process reading the file while its length is zero.
	temporary := filepath.Join(filepath.Dir(shmPath), ".shm-init-"+rand.Text())
	file, err := os.OpenFile(temporary, os.O_RDWR|os.O_CREATE|os.O_EXCL, sharedMemoryFileMode)
	if err != nil {
		return nil, fmt.Errorf("failed to create file: %w", err)
	}
	defer func() { _ = os.Remove(file.Name()) }()

	// The mode passed to open is filtered by the process umask.
	if err := file.Chmod(sharedMemoryFileMode); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("failed to set file mode: %w", err)
	}

	// Truncate to the required size
	if err := file.Truncate(int64(stateSize)); err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("failed to truncate file: %w", err)
	}

	// Memory map the file
	data, err := syscall.Mmap(int(file.Fd()), 0, stateSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("failed to mmap: %w", err)
	}

	// Initialize the state
	state, err := NewSharedDeviceStateV2(configs)
	if err != nil {
		_ = syscall.Munmap(data)
		_ = file.Close()
		return nil, err
	}

	// Get a pointer to the mapped state
	mappedState := (*rustSharedDeviceStateV2Layout)(unsafe.Pointer(&data[0]))

	// Copy the V2 payload bytes into the mapped enum payload without
	// assigning the lock-containing struct by value.
	statePayloadSize := int(unsafe.Sizeof(*state))
	stateBytes := (*[1 << 30]byte)(unsafe.Pointer(state))[:statePayloadSize:statePayloadSize]
	v2Offset := int(unsafe.Offsetof(rustSharedDeviceStateV2Layout{}.V2))
	// The temporary file is already visible in the workload's directory.
	if err := accessMapping(data, func() {
		copy(data[v2Offset:v2Offset+statePayloadSize], stateBytes)
		mappedState.Discriminant = rustSharedDeviceStateV2Discriminant
	}); err != nil {
		_ = syscall.Munmap(data)
		_ = file.Close()
		return nil, fmt.Errorf("failed to initialize shared memory: %w", err)
	}

	// Link is an atomic create-if-absent operation. Concurrent initializers
	// converge on one inode, including its cross-process memory reservations.
	// Replacing an unusable file is an atomic swap of the name instead.
	publish := os.Link
	if replace {
		publish = os.Rename
	}
	if err := publish(file.Name(), shmPath); err != nil {
		_ = syscall.Munmap(data)
		_ = file.Close()
		if os.IsExist(err) {
			return openSharedMemoryHandle(shmPath)
		}
		return nil, fmt.Errorf("failed to publish shared memory: %w", err)
	}

	return &SharedMemoryHandle{
		path:     shmPath,
		data:     data,
		state:    &SharedDeviceState{V2: &mappedState.V2},
		file:     file,
		fileSize: int64(stateSize),
	}, nil
}

// OpenSharedMemoryHandle opens an existing shared memory handle.
// Namespace and pod name are expected to be single path components.
func OpenSharedMemoryHandle(basePath string, pod *PodIdentifier) (*SharedMemoryHandle, error) {
	if pod == nil {
		return nil, fmt.Errorf("pod identifier is required")
	}
	if pod.Namespace == "" || pod.Name == "" {
		return nil, fmt.Errorf("pod identifier must include namespace and name")
	}
	if strings.Contains(pod.Namespace, "/") ||
		strings.Contains(pod.Namespace, "\\") ||
		strings.Contains(pod.Namespace, "..") {
		return nil, fmt.Errorf("invalid namespace path component: %q", pod.Namespace)
	}
	if strings.Contains(pod.Name, "/") || strings.Contains(pod.Name, "\\") || strings.Contains(pod.Name, "..") {
		return nil, fmt.Errorf("invalid pod name path component: %q", pod.Name)
	}

	podPath := filepath.Join(basePath, pod.Namespace, pod.Name)
	shmPath := filepath.Join(podPath, ShmPathSuffix)
	if target, err := os.Readlink(shmPath); err == nil {
		// The directory is writable by the workload's processes; a link read
		// from it is validated: only a generation file in the same directory
		// is followed.
		if filepath.Base(target) != target || !strings.HasPrefix(target, ".shm-uid-") {
			return nil, fmt.Errorf("invalid shared memory generation link: %q", target)
		}
		shmPath = filepath.Join(podPath, target)
	}
	return openSharedMemoryHandle(shmPath)
}

func openSharedMemoryHandle(shmPath string) (*SharedMemoryHandle, error) {
	// Open the file
	file, err := os.OpenFile(shmPath, os.O_RDWR|syscall.O_NOFOLLOW, 0666)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}

	// Get file size
	stat, err := file.Stat()
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("failed to stat file: %w", err)
	}

	fileSize := stat.Size()
	expectedSize := int64(unsafe.Sizeof(rustSharedDeviceStateV2Layout{}))
	legacySize := int64(unsafe.Sizeof(SharedDeviceStateV2{}))
	if fileSize != expectedSize {
		_ = file.Close()
		if fileSize == legacySize {
			return nil, fmt.Errorf("%w: legacy shared memory layout detected: size=%d expected=%d",
				errSharedMemoryUnusable, fileSize, expectedSize)
		}
		return nil, fmt.Errorf("%w: unexpected shared memory size: got=%d expected=%d",
			errSharedMemoryUnusable, fileSize, expectedSize)
	}

	// Files created by an earlier release kept the umask-filtered mode. Only
	// the owning hypervisor can repair it; other openers are plain readers.
	_ = ensureMode(file, sharedMemoryFileMode)

	// Memory map the file
	data, err := syscall.Mmap(int(file.Fd()), 0, int(fileSize), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		_ = file.Close()
		return nil, fmt.Errorf("failed to mmap: %w", err)
	}

	// Get a pointer to the mapped state and verify the Rust discriminant.
	mappedState := (*rustSharedDeviceStateV2Layout)(unsafe.Pointer(&data[0]))
	var discriminant uint32
	// The length was checked above, but it can change again before this read.
	if err := accessMapping(data, func() { discriminant = mappedState.Discriminant }); err != nil {
		_ = syscall.Munmap(data)
		_ = file.Close()
		return nil, err
	}
	if discriminant != rustSharedDeviceStateV2Discriminant {
		_ = syscall.Munmap(data)
		_ = file.Close()
		return nil, fmt.Errorf("%w: unsupported shared memory discriminant: %d", errSharedMemoryUnusable, discriminant)
	}

	return &SharedMemoryHandle{
		path:     shmPath,
		data:     data,
		state:    &SharedDeviceState{V2: &mappedState.V2},
		file:     file,
		fileSize: fileSize,
	}, nil
}

// GetState returns the shared device state. Accesses through the returned
// pointer are protected neither from Close nor from a fault in the mapping;
// anything that outlives a single test or one-shot tool must use WithState.
func (h *SharedMemoryHandle) GetState() *SharedDeviceState {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.data == nil || h.faulted.Load() {
		return nil
	}
	return h.state
}

// WithState runs fn while the shared-memory mapping is protected from Close.
// Callers that retain the returned state pointer after this callback must not
// do so; the pointer is only valid for the duration of fn.
//
// It returns false when the mapping is closed or when an access to it faulted,
// in which case fn may have run only partially. A faulted handle stays
// unusable; its owner closes it and prepares the worker's mapping again.
func (h *SharedMemoryHandle) WithState(fn func(*SharedDeviceState)) bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.data == nil || h.state == nil || h.faulted.Load() {
		return false
	}
	if err := accessMapping(h.data, func() { fn(h.state) }); err != nil {
		h.faulted.Store(true)
		return false
	}
	return true
}

// Faulted reports whether an access to the mapping faulted.
func (h *SharedMemoryHandle) Faulted() bool {
	return h.faulted.Load()
}

// Close closes the shared memory handle
func (h *SharedMemoryHandle) Close() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.data != nil {
		_ = syscall.Munmap(h.data)
		h.data = nil
	}
	h.state = nil
	if h.file != nil {
		_ = h.file.Close()
		h.file = nil
	}
	return nil
}

// Cleanup removes the shared memory file and cleans up empty directories
func (h *SharedMemoryHandle) Cleanup(stopAtPath *string) error {
	if err := h.Close(); err != nil {
		return err
	}

	if err := os.Remove(h.path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove file: %w", err)
	}

	if stopAtPath != nil {
		return CleanupEmptyParentDirectories(h.path, stopAtPath)
	}
	return CleanupEmptyParentDirectories(h.path, nil)
}
