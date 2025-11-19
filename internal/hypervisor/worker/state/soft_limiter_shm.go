package worker

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

// Constants
const (
	MaxProcesses  = 2048
	MaxDevices    = 16
	MaxUUIDLen    = 64
	ShmPathSuffix = "shm"
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

// SharedDeviceInfoV1 is the legacy device state (without ERL)
type SharedDeviceInfoV1 struct {
	AvailableCudaCores int32
	UpLimit            uint32
	MemLimit           uint64
	TotalCudaCores     uint32
	PodMemoryUsed      uint64
}

// SharedDeviceInfoV2 is the V2 device state with ERL support
type SharedDeviceInfoV2 struct {
	UpLimit        uint32
	MemLimit       uint64
	TotalCudaCores uint32
	PodMemoryUsed  uint64

	// ERL (Elastic Rate Limiting) - PID-controlled token bucket
	ERLTokenRefillRate uint64 // f64 stored as bits
	ERLTokenCapacity   uint64 // f64 stored as bits
	ERLCurrentTokens   uint64 // f64 stored as bits
	ERLLastTokenUpdate uint64 // f64 stored as bits
}

// SharedDeviceInfo is a type alias for backward compatibility
type SharedDeviceInfo = SharedDeviceInfoV2

// NewSharedDeviceInfoV1 creates a new V1 device info
func NewSharedDeviceInfoV1(totalCudaCores, upLimit uint32, memLimit uint64) *SharedDeviceInfoV1 {
	return &SharedDeviceInfoV1{
		AvailableCudaCores: 0,
		UpLimit:            upLimit,
		MemLimit:           memLimit,
		TotalCudaCores:     totalCudaCores,
		PodMemoryUsed:      0,
	}
}

// NewSharedDeviceInfoV2 creates a new V2 device info
func NewSharedDeviceInfoV2(totalCudaCores, upLimit uint32, memLimit uint64) *SharedDeviceInfoV2 {
	return &SharedDeviceInfoV2{
		UpLimit:            upLimit,
		MemLimit:           memLimit,
		TotalCudaCores:     totalCudaCores,
		PodMemoryUsed:      0,
		ERLTokenRefillRate: math.Float64bits(10.0), // Default 10 tokens/sec
		ERLTokenCapacity:   math.Float64bits(100.0),
		ERLCurrentTokens:   math.Float64bits(100.0),
		ERLLastTokenUpdate: math.Float64bits(0.0),
	}
}

// DeviceEntryV1 is the legacy device entry
type DeviceEntryV1 struct {
	UUID          [MaxUUIDLen]byte
	DeviceInfo    SharedDeviceInfoV1
	IsActiveField uint32
	_padding      [4]byte // padding for alignment
}

// DeviceEntryV2 is the V2 device entry with ERL
type DeviceEntryV2 struct {
	UUID          [MaxUUIDLen]byte
	DeviceInfo    SharedDeviceInfoV2
	IsActiveField uint32
	_padding      [4]byte // padding for alignment
}

// DeviceEntry is a type alias for backward compatibility
type DeviceEntry = DeviceEntryV2

// NewDeviceEntryV1 creates a new V1 device entry
func NewDeviceEntryV1() *DeviceEntryV1 {
	return &DeviceEntryV1{
		DeviceInfo: *NewSharedDeviceInfoV1(0, 0, 0),
	}
}

// NewDeviceEntryV2 creates a new V2 device entry
func NewDeviceEntryV2() *DeviceEntryV2 {
	return &DeviceEntryV2{
		DeviceInfo: *NewSharedDeviceInfoV2(0, 0, 0),
	}
}

// SetUUID sets the device UUID
func (d *DeviceEntryV1) SetUUID(uuid string) {
	copyLen := len(uuid)
	if copyLen > MaxUUIDLen-1 {
		copyLen = MaxUUIDLen - 1
	}

	// Clear the UUID array
	for i := range d.UUID {
		d.UUID[i] = 0
	}

	// Copy the new UUID
	copy(d.UUID[:], uuid[:copyLen])
}

// GetUUID gets the device UUID as a string
func (d *DeviceEntryV1) GetUUID() string {
	nullPos := MaxUUIDLen - 1
	for i, b := range d.UUID {
		if b == 0 {
			nullPos = i
			break
		}
	}
	return string(d.UUID[:nullPos])
}

// IsActive checks if this entry is active
func (d *DeviceEntryV1) IsActive() bool {
	return atomic.LoadUint32(&d.IsActiveField) != 0
}

// SetActive sets the active status
func (d *DeviceEntryV1) SetActive(active bool) {
	var val uint32
	if active {
		val = 1
	}
	atomic.StoreUint32(&d.IsActiveField, val)
}

// SetUUID sets the device UUID
func (d *DeviceEntryV2) SetUUID(uuid string) {
	copyLen := len(uuid)
	if copyLen > MaxUUIDLen-1 {
		copyLen = MaxUUIDLen - 1
	}

	// Clear the UUID array
	for i := range d.UUID {
		d.UUID[i] = 0
	}

	// Copy the new UUID
	copy(d.UUID[:], uuid[:copyLen])
}

// GetUUID gets the device UUID as a string
func (d *DeviceEntryV2) GetUUID() string {
	nullPos := MaxUUIDLen - 1
	for i, b := range d.UUID {
		if b == 0 {
			nullPos = i
			break
		}
	}
	return string(d.UUID[:nullPos])
}

// IsActive checks if this entry is active
func (d *DeviceEntryV2) IsActive() bool {
	return atomic.LoadUint32(&d.IsActiveField) != 0
}

// SetActive sets the active status
func (d *DeviceEntryV2) SetActive(active bool) {
	var val uint32
	if active {
		val = 1
	}
	atomic.StoreUint32(&d.IsActiveField, val)
}

// DeviceConfig contains device configuration information
type DeviceConfig struct {
	DeviceIdx      uint32
	DeviceUUID     string
	UpLimit        uint32
	MemLimit       uint64
	SMCount        uint32
	MaxThreadPerSM uint32
	TotalCudaCores uint32
}

// SharedDeviceStateV1 is the V1 shared device state
type SharedDeviceStateV1 struct {
	Devices          [MaxDevices]DeviceEntryV1
	DeviceCountField uint32
	LastHeartbeat    uint64
	PIDs             *ShmMutex[*PIDSet]
	_padding         [512]byte
}

// SharedDeviceStateV2 is the V2 shared device state with ERL
type SharedDeviceStateV2 struct {
	Devices          [MaxDevices]DeviceEntryV2
	DeviceCountField uint32
	LastHeartbeat    uint64
	PIDs             *ShmMutex[*PIDSet]
	_padding         [512]byte
}

// SharedDeviceState is a versioned enum for compatibility
type SharedDeviceState struct {
	V1 *SharedDeviceStateV1
	V2 *SharedDeviceStateV2
}

// Version returns the version number
func (s *SharedDeviceState) Version() uint32 {
	if s.V1 != nil {
		return 1
	}
	return 2
}

// HasERL checks if this state uses ERL features
func (s *SharedDeviceState) HasERL() bool {
	return s.V2 != nil
}

// NewSharedDeviceStateV1 creates a new V1 state
func NewSharedDeviceStateV1(configs []DeviceConfig) (*SharedDeviceStateV1, error) {
	now := uint64(time.Now().Unix())

	state := &SharedDeviceStateV1{
		DeviceCountField: uint32(len(configs)),
		LastHeartbeat:    now,
		PIDs:             NewShmMutex(NewPIDSet()),
	}

	for _, config := range configs {
		deviceIdx := int(config.DeviceIdx)
		if deviceIdx >= MaxDevices {
			return nil, fmt.Errorf("device index %d exceeds maximum devices %d", deviceIdx, MaxDevices)
		}

		entry := &state.Devices[deviceIdx]
		entry.SetUUID(config.DeviceUUID)
		entry.DeviceInfo.TotalCudaCores = config.TotalCudaCores
		entry.DeviceInfo.AvailableCudaCores = int32(config.TotalCudaCores)
		entry.DeviceInfo.UpLimit = config.UpLimit
		entry.DeviceInfo.MemLimit = config.MemLimit
		entry.SetActive(true)
	}

	return state, nil
}

// NewSharedDeviceStateV2 creates a new V2 state
func NewSharedDeviceStateV2(configs []DeviceConfig) (*SharedDeviceStateV2, error) {
	now := uint64(time.Now().Unix())

	state := &SharedDeviceStateV2{
		DeviceCountField: uint32(len(configs)),
		LastHeartbeat:    now,
		PIDs:             NewShmMutex(NewPIDSet()),
	}

	for _, config := range configs {
		deviceIdx := int(config.DeviceIdx)
		if deviceIdx >= MaxDevices {
			return nil, fmt.Errorf("device index %d exceeds maximum devices %d", deviceIdx, MaxDevices)
		}

		entry := &state.Devices[deviceIdx]
		entry.SetUUID(config.DeviceUUID)
		entry.DeviceInfo.TotalCudaCores = config.TotalCudaCores
		entry.DeviceInfo.UpLimit = config.UpLimit
		entry.DeviceInfo.MemLimit = config.MemLimit

		// Initialize ERL fields with defaults
		entry.DeviceInfo.ERLTokenCapacity = math.Float64bits(100.0)
		entry.DeviceInfo.ERLTokenRefillRate = math.Float64bits(10.0)
		entry.DeviceInfo.ERLCurrentTokens = math.Float64bits(100.0)
		entry.DeviceInfo.ERLLastTokenUpdate = math.Float64bits(float64(now))

		entry.SetActive(true)
	}

	return state, nil
}

// NewSharedDeviceState creates a new SharedDeviceState (defaults to V2)
func NewSharedDeviceState(configs []DeviceConfig) (*SharedDeviceState, error) {
	v2, err := NewSharedDeviceStateV2(configs)
	if err != nil {
		return nil, err
	}
	return &SharedDeviceState{V2: v2}, nil
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

// SetPodMemoryUsed sets pod memory used for a device
func (s *SharedDeviceState) SetPodMemoryUsed(index int, memory uint64) bool {
	if s.V1 != nil {
		if index >= MaxDevices || !s.V1.Devices[index].IsActive() {
			return false
		}
		atomic.StoreUint64(&s.V1.Devices[index].DeviceInfo.PodMemoryUsed, memory)
		return true
	}
	if index >= MaxDevices || !s.V2.Devices[index].IsActive() {
		return false
	}
	atomic.StoreUint64(&s.V2.Devices[index].DeviceInfo.PodMemoryUsed, memory)
	return true
}

// ERL token bucket operations for SharedDeviceInfoV2

// GetERLTokenCapacity returns the token capacity
func (d *SharedDeviceInfoV2) GetERLTokenCapacity() float64 {
	return math.Float64frombits(atomic.LoadUint64(&d.ERLTokenCapacity))
}

// SetERLTokenCapacity sets the token capacity
func (d *SharedDeviceInfoV2) SetERLTokenCapacity(capacity float64) {
	atomic.StoreUint64(&d.ERLTokenCapacity, math.Float64bits(capacity))
}

// GetERLTokenRefillRate returns the refill rate
func (d *SharedDeviceInfoV2) GetERLTokenRefillRate() float64 {
	return math.Float64frombits(atomic.LoadUint64(&d.ERLTokenRefillRate))
}

// SetERLTokenRefillRate sets the refill rate
func (d *SharedDeviceInfoV2) SetERLTokenRefillRate(rate float64) {
	atomic.StoreUint64(&d.ERLTokenRefillRate, math.Float64bits(rate))
}

// GetERLCurrentTokens returns the current tokens
func (d *SharedDeviceInfoV2) GetERLCurrentTokens() float64 {
	return math.Float64frombits(atomic.LoadUint64(&d.ERLCurrentTokens))
}

// SetERLCurrentTokens sets the current tokens
func (d *SharedDeviceInfoV2) SetERLCurrentTokens(tokens float64) {
	atomic.StoreUint64(&d.ERLCurrentTokens, math.Float64bits(tokens))
}

// GetERLLastTokenUpdate returns the last token update timestamp
func (d *SharedDeviceInfoV2) GetERLLastTokenUpdate() float64 {
	return math.Float64frombits(atomic.LoadUint64(&d.ERLLastTokenUpdate))
}

// SetERLLastTokenUpdate sets the last token update timestamp
func (d *SharedDeviceInfoV2) SetERLLastTokenUpdate(timestamp float64) {
	atomic.StoreUint64(&d.ERLLastTokenUpdate, math.Float64bits(timestamp))
}

// LoadERLTokenState loads the token state atomically
func (d *SharedDeviceInfoV2) LoadERLTokenState() (float64, float64) {
	return d.GetERLCurrentTokens(), d.GetERLLastTokenUpdate()
}

// StoreERLTokenState stores the token state atomically
func (d *SharedDeviceInfoV2) StoreERLTokenState(tokens, timestamp float64) {
	d.SetERLCurrentTokens(tokens)
	d.SetERLLastTokenUpdate(timestamp)
}

// LoadERLQuota loads the quota configuration
func (d *SharedDeviceInfoV2) LoadERLQuota() (float64, float64) {
	return d.GetERLTokenCapacity(), d.GetERLTokenRefillRate()
}

// FetchSubERLTokens atomically subtracts tokens and returns the value before subtraction
func (d *SharedDeviceInfoV2) FetchSubERLTokens(cost float64) float64 {
	for {
		currentBits := atomic.LoadUint64(&d.ERLCurrentTokens)
		current := math.Float64frombits(currentBits)

		if current < cost {
			return current
		}

		newValue := math.Max(0.0, current-cost)
		newBits := math.Float64bits(newValue)

		if atomic.CompareAndSwapUint64(&d.ERLCurrentTokens, currentBits, newBits) {
			return current
		}
	}
}

// FetchAddERLTokens atomically adds tokens (capped at capacity) and returns the value before addition
func (d *SharedDeviceInfoV2) FetchAddERLTokens(amount float64) float64 {
	capacity := d.GetERLTokenCapacity()

	for {
		currentBits := atomic.LoadUint64(&d.ERLCurrentTokens)
		current := math.Float64frombits(currentBits)

		newValue := math.Max(0.0, math.Min(capacity, current+amount))
		newBits := math.Float64bits(newValue)

		if atomic.CompareAndSwapUint64(&d.ERLCurrentTokens, currentBits, newBits) {
			return current
		}
	}
}

// PIDSet is a set of process IDs with a fixed capacity
type PIDSet struct {
	values []int
	mu     sync.Mutex
}

// NewPIDSet creates a new PID set
func NewPIDSet() *PIDSet {
	return &PIDSet{
		values: make([]int, 0, MaxProcesses),
	}
}

// InsertIfAbsent inserts a value if it's not already present
func (s *PIDSet) InsertIfAbsent(pid int) bool {
	for _, v := range s.values {
		if v == pid {
			return false
		}
	}
	if len(s.values) >= MaxProcesses {
		return false
	}
	s.values = append(s.values, pid)
	return true
}

// RemoveValue removes a value from the set
func (s *PIDSet) RemoveValue(pid int) bool {
	for i, v := range s.values {
		if v == pid {
			s.values = append(s.values[:i], s.values[i+1:]...)
			return true
		}
	}
	return false
}

// Values returns all values in the set
func (s *PIDSet) Values() []int {
	result := make([]int, len(s.values))
	copy(result, s.values)
	return result
}

// ShmMutex is a shared memory mutex wrapper
type ShmMutex[T any] struct {
	mu    sync.Mutex
	Value T
}

// NewShmMutex creates a new shared memory mutex
func NewShmMutex[T any](value T) *ShmMutex[T] {
	return &ShmMutex[T]{
		Value: value,
	}
}

// Lock locks the mutex
func (m *ShmMutex[T]) Lock() {
	m.mu.Lock()
}

// Unlock unlocks the mutex
func (m *ShmMutex[T]) Unlock() {
	m.mu.Unlock()
}

// CleanupOrphanedLock cleans up orphaned locks (placeholder for now)
func (m *ShmMutex[T]) CleanupOrphanedLock() {
	// In a real implementation, this would check for dead processes
	// and release their locks. For now, it's a no-op.
}

// SharedMemoryHandle manages a shared memory mapping
type SharedMemoryHandle struct {
	path     string
	data     []byte
	state    *SharedDeviceState
	file     *os.File
	fileSize int64
}

// CreateSharedMemoryHandle creates a new shared memory handle
func CreateSharedMemoryHandle(podPath string, configs []DeviceConfig) (*SharedMemoryHandle, error) {
	shmPath := filepath.Join(podPath, ShmPathSuffix)

	// Create directory if it doesn't exist
	if err := os.MkdirAll(podPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory: %w", err)
	}

	// Calculate size needed for SharedDeviceStateV2
	stateSize := int(unsafe.Sizeof(SharedDeviceStateV2{}))

	// Create or open the file
	file, err := os.OpenFile(shmPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		return nil, fmt.Errorf("failed to create file: %w", err)
	}

	// Truncate to the required size
	if err := file.Truncate(int64(stateSize)); err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to truncate file: %w", err)
	}

	// Memory map the file
	data, err := syscall.Mmap(int(file.Fd()), 0, stateSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to mmap: %w", err)
	}

	// Initialize the state
	state, err := NewSharedDeviceStateV2(configs)
	if err != nil {
		syscall.Munmap(data)
		file.Close()
		return nil, err
	}

	// Copy the state to the mapped memory
	stateBytes := (*[1 << 30]byte)(unsafe.Pointer(state))[:stateSize:stateSize]
	copy(data, stateBytes)

	// Get a pointer to the mapped state
	mappedState := (*SharedDeviceStateV2)(unsafe.Pointer(&data[0]))

	// Initialize the PIDs mutex in the mapped memory
	// Note: This is a simplified version - in a real implementation,
	// you'd need to properly initialize the mutex for shared memory
	mappedState.PIDs = NewShmMutex(NewPIDSet())

	return &SharedMemoryHandle{
		path:     shmPath,
		data:     data,
		state:    &SharedDeviceState{V2: mappedState},
		file:     file,
		fileSize: int64(stateSize),
	}, nil
}

// OpenSharedMemoryHandle opens an existing shared memory handle
func OpenSharedMemoryHandle(podPath string) (*SharedMemoryHandle, error) {
	shmPath := filepath.Join(podPath, ShmPathSuffix)

	// Open the file
	file, err := os.OpenFile(shmPath, os.O_RDWR, 0666)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}

	// Get file size
	stat, err := file.Stat()
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to stat file: %w", err)
	}

	fileSize := stat.Size()

	// Memory map the file
	data, err := syscall.Mmap(int(file.Fd()), 0, int(fileSize), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		file.Close()
		return nil, fmt.Errorf("failed to mmap: %w", err)
	}

	// Get a pointer to the mapped state (assume V2 for now)
	mappedState := (*SharedDeviceStateV2)(unsafe.Pointer(&data[0]))

	return &SharedMemoryHandle{
		path:     shmPath,
		data:     data,
		state:    &SharedDeviceState{V2: mappedState},
		file:     file,
		fileSize: fileSize,
	}, nil
}

// GetState returns the shared device state
func (h *SharedMemoryHandle) GetState() *SharedDeviceState {
	return h.state
}

// Close closes the shared memory handle
func (h *SharedMemoryHandle) Close() error {
	if h.data != nil {
		syscall.Munmap(h.data)
		h.data = nil
	}
	if h.file != nil {
		h.file.Close()
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
