//go:build darwin || linux || freebsd || netbsd

package worker

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sync/atomic"
	"syscall"
	"time"
)

// RefCountError represents errors in reference count operations
type RefCountError struct {
	Type string
}

func (e *RefCountError) Error() string {
	return fmt.Sprintf("ref count error: %s", e.Type)
}

var (
	ErrRefCountUnderflow = &RefCountError{Type: "underflow"}
)

func isProcessAlive(pid uintptr) bool {
	if pid == 0 {
		return false
	}
	// EPERM means the process exists but belongs to another user.
	return syscall.Kill(int(pid), 0) != syscall.ESRCH
}

// HasDevice checks if a device exists at the given index
func (s *SharedDeviceStateV1) HasDevice(index int) bool {
	return index < MaxDevices && s.Devices[index].IsActive()
}

// DeviceCount returns the number of devices
func (s *SharedDeviceStateV1) DeviceCount() int {
	return int(atomic.LoadUint32(&s.DeviceCountField))
}

// UpdateHeartbeat updates the heartbeat timestamp
func (s *SharedDeviceStateV1) UpdateHeartbeat(timestamp uint64) {
	atomic.StoreUint64(&s.LastHeartbeat, timestamp)
}

// GetLastHeartbeat returns the last heartbeat timestamp
func (s *SharedDeviceStateV1) GetLastHeartbeat() uint64 {
	return atomic.LoadUint64(&s.LastHeartbeat)
}

// IsHealthy checks if the shared memory is healthy based on heartbeat
func (s *SharedDeviceStateV1) IsHealthy(timeout time.Duration) bool {
	now := uint64(time.Now().Unix())
	lastHeartbeat := s.GetLastHeartbeat()

	if lastHeartbeat == 0 {
		return false
	}

	if lastHeartbeat > now {
		return false
	}

	return now-lastHeartbeat <= uint64(timeout.Seconds())
}

// AddPID adds a PID to the set
func (s *SharedDeviceStateV1) AddPID(pid int) {
	s.PIDs.Lock()
	defer s.PIDs.Unlock()
	s.PIDs.Value.InsertIfAbsent(pid)
}

// RemovePID removes a PID from the set
func (s *SharedDeviceStateV1) RemovePID(pid int) {
	s.PIDs.Lock()
	defer s.PIDs.Unlock()
	s.PIDs.Value.RemoveValue(pid)
}

// GetAllPIDs returns all PIDs currently stored
func (s *SharedDeviceStateV1) GetAllPIDs() []int {
	s.PIDs.Lock()
	defer s.PIDs.Unlock()
	return s.PIDs.Value.Values()
}

// CleanupOrphanedLocks cleans up any orphaned locks
func (s *SharedDeviceStateV1) CleanupOrphanedLocks() {
	s.PIDs.CleanupOrphanedLock()
}

// HasDevice checks if a device exists at the given index
func (s *SharedDeviceStateV2) HasDevice(index int) bool {
	return index < MaxDevices && s.Devices[index].IsActive()
}

// DeviceCount returns the number of devices
func (s *SharedDeviceStateV2) DeviceCount() int {
	return int(atomic.LoadUint32(&s.DeviceCountField))
}

// UpdateHeartbeat updates the heartbeat timestamp
func (s *SharedDeviceStateV2) UpdateHeartbeat(timestamp uint64) {
	atomic.StoreUint64(&s.LastHeartbeat, timestamp)
}

// GetLastHeartbeat returns the last heartbeat timestamp
func (s *SharedDeviceStateV2) GetLastHeartbeat() uint64 {
	return atomic.LoadUint64(&s.LastHeartbeat)
}

// IsHealthy checks if the shared memory is healthy based on heartbeat
func (s *SharedDeviceStateV2) IsHealthy(timeout time.Duration) bool {
	now := uint64(time.Now().Unix())
	lastHeartbeat := s.GetLastHeartbeat()

	if lastHeartbeat == 0 {
		return false
	}

	if lastHeartbeat > now {
		return false
	}

	return now-lastHeartbeat <= uint64(timeout.Seconds())
}

// AddPID adds a PID to the set
func (s *SharedDeviceStateV2) AddPID(pid int) {
	s.PIDs.Lock()
	defer s.PIDs.Unlock()
	s.PIDs.Value.InsertIfAbsent(pid)
}

// RemovePID removes a PID from the set
func (s *SharedDeviceStateV2) RemovePID(pid int) {
	s.PIDs.Lock()
	defer s.PIDs.Unlock()
	s.PIDs.Value.RemoveValue(pid)
}

// GetAllPIDs returns all PIDs currently stored
func (s *SharedDeviceStateV2) GetAllPIDs() []int {
	s.PIDs.Lock()
	defer s.PIDs.Unlock()
	return s.PIDs.Value.Values()
}

// CleanupOrphanedLocks cleans up any orphaned locks
func (s *SharedDeviceStateV2) CleanupOrphanedLocks() {
	s.PIDs.CleanupOrphanedLock()
}

// Helper methods for SharedDeviceState that delegate to the appropriate version

// HasDevice checks if a device exists
func (s *SharedDeviceState) HasDevice(index int) bool {
	if s.V1 != nil {
		return s.V1.HasDevice(index)
	}
	return s.V2.HasDevice(index)
}

// DeviceCount returns the number of devices
func (s *SharedDeviceState) DeviceCount() int {
	if s.V1 != nil {
		return s.V1.DeviceCount()
	}
	return s.V2.DeviceCount()
}

// UpdateHeartbeat updates the heartbeat
func (s *SharedDeviceState) UpdateHeartbeat(timestamp uint64) {
	if s.V1 != nil {
		s.V1.UpdateHeartbeat(timestamp)
	} else {
		s.V2.UpdateHeartbeat(timestamp)
	}
}

// GetLastHeartbeat returns the last heartbeat
func (s *SharedDeviceState) GetLastHeartbeat() uint64 {
	if s.V1 != nil {
		return s.V1.GetLastHeartbeat()
	}
	return s.V2.GetLastHeartbeat()
}

// IsHealthy checks if healthy
func (s *SharedDeviceState) IsHealthy(timeout time.Duration) bool {
	if s.V1 != nil {
		return s.V1.IsHealthy(timeout)
	}
	return s.V2.IsHealthy(timeout)
}

// AddPID adds a PID
func (s *SharedDeviceState) AddPID(pid int) {
	if s.V1 != nil {
		s.V1.AddPID(pid)
	} else {
		s.V2.AddPID(pid)
	}
}

// TryAddPID lets process initialization retry without blocking the worker
// controller on a lock held by a container (possibly in another PID namespace).
func (s *SharedDeviceState) TryAddPID(pid int) bool {
	var mutex *ShmMutex[PIDSet]
	if s.V1 != nil {
		mutex = &s.V1.PIDs
	} else {
		mutex = &s.V2.PIDs
	}
	mutex.CleanupOrphanedLock()
	if !atomic.CompareAndSwapUintptr(&mutex.LockField, 0, uintptr(os.Getpid())) {
		return false
	}
	defer mutex.Unlock()
	mutex.Value.InsertIfAbsent(pid)
	return true
}

// RemovePID removes a PID
func (s *SharedDeviceState) RemovePID(pid int) {
	if s.V1 != nil {
		s.V1.RemovePID(pid)
	} else {
		s.V2.RemovePID(pid)
	}
}

// GetAllPIDs returns all PIDs
func (s *SharedDeviceState) GetAllPIDs() []int {
	if s.V1 != nil {
		return s.V1.GetAllPIDs()
	}
	return s.V2.GetAllPIDs()
}

// CleanupOrphanedLocks cleans up orphaned locks
func (s *SharedDeviceState) CleanupOrphanedLocks() {
	if s.V1 != nil {
		s.V1.CleanupOrphanedLocks()
	} else {
		s.V2.CleanupOrphanedLocks()
	}
}

// SetPodMemoryUsed sets pod memory used for a device, as a figure without the
// time of a sample: in a V2 state the time published for the slot is cleared
// before the figure is replaced, so that a reader never takes the time of an
// earlier sample for the time of this figure. A figure that comes from a
// process sample is published with SetPodMemorySample.
func (s *SharedDeviceState) SetPodMemoryUsed(index int, memory uint64) bool {
	return s.SetPodMemorySample(index, memory, 0)
}

// SetPodMemorySample publishes what a process sample says about the memory a
// pod holds on a device: used bytes, from a provider query that started at
// sampledAtSecs (the wall clock read immediately before it, in seconds since
// the Unix epoch). It reports whether the device exists.
//
// A V2 state has one writer of these two fields, the hypervisor, and the
// order of its stores is the whole protocol; there is no sequence word:
//
//  1. store 0 to pod_memory_sampled_at
//  2. store the figure to pod_memory_used
//  3. store the bits of the time to pod_memory_sampled_at
//
// A reader loads the time, the figure and the time again, and takes the two
// for one sample only when both readings of the time are the same and not
// zero (GetPodMemorySample). The time is also how it tells a new sample from
// the one it has seen, so the time of a sample always differs from the one
// published before it in the slot, and never lies below it: when the clock did
// not advance, or went back, the next representable double above the previous
// time is published instead. The previous time is the one in the mapping,
// which outlives the hypervisor process. The pair that is already in the
// mapping is not written again.
//
// A sampledAtSecs that is not a finite positive number publishes the figure
// without a time (steps 1 and 2): the reader then checks against the figure
// alone, as it did before the time existed. A V1 state has no such field and
// gets the figure only.
func (s *SharedDeviceState) SetPodMemorySample(index int, used uint64, sampledAtSecs float64) bool {
	usedField, sampledAtField := s.podMemoryFields(index)
	if usedField == nil {
		return false
	}
	if sampledAtField == nil {
		atomic.StoreUint64(usedField, used)
		return true
	}
	storePodMemorySample(atomic.StoreUint64, usedField, sampledAtField, used, sampledAtSecs)
	return true
}

// podMemorySampleReadAttempts is how often a reader looks for a figure and a
// time that belong together before it settles for the figure alone.
const podMemorySampleReadAttempts = 4

// GetPodMemorySample reads the memory a pod holds on a device the way a
// limiter does, and reports whether the device exists. sampledAtSecs is zero
// when no time is published for the figure, or when the writer kept replacing
// the pair while it was read: used is then a figure without a time.
func (s *SharedDeviceState) GetPodMemorySample(index int) (used uint64, sampledAtSecs float64, ok bool) {
	usedField, sampledAtField := s.podMemoryFields(index)
	if usedField == nil {
		return 0, 0, false
	}
	if sampledAtField == nil {
		return atomic.LoadUint64(usedField), 0, true
	}
	for range podMemorySampleReadAttempts {
		before := atomic.LoadUint64(sampledAtField)
		used = atomic.LoadUint64(usedField)
		if before == 0 {
			return used, 0, true
		}
		if atomic.LoadUint64(sampledAtField) == before {
			if at := math.Float64frombits(before); isPodMemorySampleTime(at) {
				return used, at, true
			}
			return used, 0, true
		}
	}
	return used, 0, true
}

// podMemoryFields returns the words of an active device slot that hold the
// memory figure and, in a V2 state, the time of its sample. usedField is nil
// when there is no such device.
func (s *SharedDeviceState) podMemoryFields(index int) (usedField, sampledAtField *uint64) {
	if index < 0 || index >= MaxDevices {
		return nil, nil
	}
	if s.V1 != nil {
		if !s.V1.Devices[index].IsActive() {
			return nil, nil
		}
		return &s.V1.Devices[index].DeviceInfo.PodMemoryUsed, nil
	}
	if !s.V2.Devices[index].IsActive() {
		return nil, nil
	}
	return &s.V2.Devices[index].DeviceInfo.PodMemoryUsed, &s.V2.PodMemorySampledAt[index]
}

// isPodMemorySampleTime reports whether a value read from, or meant for,
// pod_memory_sampled_at is a time: a finite positive number of seconds.
// Anything else stands for "no time".
func isPodMemorySampleTime(secs float64) bool {
	return secs > 0 && !math.IsInf(secs, 0)
}

// storePodMemorySample is the writer's side of the protocol described at
// SetPodMemorySample. Every store to the two words goes through store, in the
// order in which a reader may observe them.
func storePodMemorySample(
	store func(field *uint64, value uint64), usedField, sampledAtField *uint64, used uint64, sampledAtSecs float64,
) {
	var sampledAt uint64
	if isPodMemorySampleTime(sampledAtSecs) {
		// The mapping is writable by the workload's processes; values read from
		// it are validated: a previous value that is not a time counts as none.
		if previous := math.Float64frombits(atomic.LoadUint64(sampledAtField)); isPodMemorySampleTime(previous) {
			if previous == sampledAtSecs && atomic.LoadUint64(usedField) == used {
				return
			}
			if sampledAtSecs <= previous {
				sampledAtSecs = math.Nextafter(previous, math.Inf(1))
			}
		}
		if isPodMemorySampleTime(sampledAtSecs) {
			sampledAt = math.Float64bits(sampledAtSecs)
		}
	}
	store(sampledAtField, 0)
	store(usedField, used)
	if sampledAt != 0 {
		store(sampledAtField, sampledAt)
	}
}

// SetDeviceLimits publishes the compute limit (percent) and the memory limit
// (bytes) of an active device to a live mapping, and reports whether either
// changed. Each limit is one aligned word that readers load atomically and use
// on its own, so one atomic store per field is all a concurrent reader needs:
// it sees the previous or the new value of a field, never a mix. A field that
// already holds the value is not written.
func (s *SharedDeviceState) SetDeviceLimits(index int, upLimit uint32, memLimit uint64) bool {
	if index < 0 || index >= MaxDevices {
		return false
	}
	var upLimitField *uint32
	var memLimitField *uint64
	if s.V1 != nil {
		if !s.V1.Devices[index].IsActive() {
			return false
		}
		upLimitField = &s.V1.Devices[index].DeviceInfo.UpLimit
		memLimitField = &s.V1.Devices[index].DeviceInfo.MemLimit
	} else {
		if !s.V2.Devices[index].IsActive() {
			return false
		}
		upLimitField = &s.V2.Devices[index].DeviceInfo.UpLimit
		memLimitField = &s.V2.Devices[index].DeviceInfo.MemLimit
	}
	changed := false
	if atomic.LoadUint32(upLimitField) != upLimit {
		atomic.StoreUint32(upLimitField, upLimit)
		changed = true
	}
	if atomic.LoadUint64(memLimitField) != memLimit {
		atomic.StoreUint64(memLimitField, memLimit)
		changed = true
	}
	return changed
}

// GetDeviceUpLimit returns the compute limit (percent) an active device slot
// of the mapping carries, loaded the way a limiter loads it. The mapping is
// writable by the workload's processes; values read from it are validated: a
// value that is not a percentage from 1 to 100 is reported as not known.
func (s *SharedDeviceState) GetDeviceUpLimit(index int) (upLimit uint32, known bool) {
	if index < 0 || index >= MaxDevices {
		return 0, false
	}
	var upLimitField *uint32
	if s.V1 != nil {
		if !s.V1.Devices[index].IsActive() {
			return 0, false
		}
		upLimitField = &s.V1.Devices[index].DeviceInfo.UpLimit
	} else {
		if !s.V2.Devices[index].IsActive() {
			return 0, false
		}
		upLimitField = &s.V2.Devices[index].DeviceInfo.UpLimit
	}
	if upLimit = atomic.LoadUint32(upLimitField); upLimit < 1 || upLimit > 100 {
		return 0, false
	}
	return upLimit, true
}

// ShmMutex is a shared memory mutex wrapper
type ShmMutex[T any] struct {
	LockField uintptr
	Value     T
	PID       uintptr
}

// NewShmMutex creates a new shared memory mutex
func NewShmMutex[T any](value T) ShmMutex[T] {
	return ShmMutex[T]{
		Value: value,
		PID:   uintptr(os.Getpid()),
	}
}

// Lock locks the mutex
func (m *ShmMutex[T]) Lock() {
	// PID is part of the persisted ABI and may name a previous hypervisor.
	// The owner of a new lock must always be the process acquiring it now.
	pid := uintptr(os.Getpid())
	for {
		if atomic.CompareAndSwapUintptr(&m.LockField, 0, pid) {
			return
		}
		holder := atomic.LoadUintptr(&m.LockField)
		if holder != 0 && holder != pid && !isProcessAlive(holder) {
			atomic.CompareAndSwapUintptr(&m.LockField, holder, 0)
			continue
		}
		runtime.Gosched()
	}
}

// Unlock unlocks the mutex
func (m *ShmMutex[T]) Unlock() {
	_ = atomic.CompareAndSwapUintptr(&m.LockField, uintptr(os.Getpid()), 0)
}

// CleanupOrphanedLock cleans up orphaned locks (placeholder for now)
func (m *ShmMutex[T]) CleanupOrphanedLock() {
	holder := atomic.LoadUintptr(&m.LockField)
	if holder != 0 && !isProcessAlive(holder) {
		atomic.CompareAndSwapUintptr(&m.LockField, holder, 0)
	}
}
