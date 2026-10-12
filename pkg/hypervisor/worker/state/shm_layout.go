//go:build darwin || linux || freebsd || netbsd

package worker

import (
	"fmt"
	"math"
	"sync/atomic"
	"time"
)

// Constants
const (
	MaxProcesses  = 2048
	MaxDevices    = 16
	MaxUUIDLen    = 64
	ShmPathSuffix = "shm"

	// Rust `#[repr(C)] enum SharedDeviceState` discriminants.
	rustSharedDeviceStateV1Discriminant = uint32(0)
	rustSharedDeviceStateV2Discriminant = uint32(1)
)

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
	//nolint:unused // Padding field for memory alignment in shared memory structures
	_padding [4]byte
}

// DeviceEntryV2 is the V2 device entry with ERL
type DeviceEntryV2 struct {
	UUID          [MaxUUIDLen]byte
	DeviceInfo    SharedDeviceInfoV2
	IsActiveField uint32
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

// SharedDeviceStateV1 is the Rust-compatible V1 shared device state.
type SharedDeviceStateV1 struct {
	Devices          [MaxDevices]DeviceEntryV1
	DeviceCountField uint32
	LastHeartbeat    uint64
	PIDs             ShmMutex[PIDSet]
	//nolint:unused // Padding field for Rust shared memory compatibility.
	_padding [512]byte
}

// SharedDeviceStateV2 is the Rust-compatible V2 shared device state with ERL.
type SharedDeviceStateV2 struct {
	Devices          [MaxDevices]DeviceEntryV2
	DeviceCountField uint32
	LastHeartbeat    uint64
	PIDs             ShmMutex[PIDSet]
	// PodMemorySampledAt[i] (pod_memory_sampled_at[i] in the limiter's layout)
	// is when the process sample behind Devices[i].DeviceInfo.PodMemoryUsed was
	// taken: the wall clock read immediately before the provider's process
	// query started, in seconds since the Unix epoch as the bits of an IEEE-754
	// double, the clock and encoding of ERLLastTokenUpdate. Zero means that no
	// time is published for the figure. It is written through SetPodMemorySample
	// and SetPodMemoryUsed only, which keep the order of stores that lets a
	// reader tell a figure and a time that belong together.
	//
	// The 16 words are the first 128 of the 512 bytes of trailing padding, at
	// 34992 + 8*i from the start of the segment; the size of the segment and
	// every other offset are unchanged, and there is still no version field.
	// Every implementation has always written these bytes as zero, so both
	// mixed pairings behave as before the field existed: a limiter that does
	// not know it never reads it, and a limiter that does finds zero in a
	// segment written by a hypervisor that does not, and then checks a request
	// against the figure alone.
	PodMemorySampledAt [MaxDevices]uint64
	//nolint:unused // Padding field for Rust shared memory compatibility.
	_padding [512 - 8*MaxDevices]byte
}

// SharedDeviceState is the Go wrapper used by the hypervisor code.
type SharedDeviceState struct {
	V1 *SharedDeviceStateV1
	V2 *SharedDeviceStateV2
}

// rustSharedDeviceStateV2Layout matches Rust's `#[repr(C)] enum SharedDeviceState::V2`.
// The V2 payload begins at offset 8, which is required by `libcuda_limiter.so`.
type rustSharedDeviceStateV2Layout struct {
	Discriminant uint32
	//nolint:unused // Required to align the payload to 8 bytes like Rust's repr(C) enum.
	_padding [4]byte
	V2       SharedDeviceStateV2
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
