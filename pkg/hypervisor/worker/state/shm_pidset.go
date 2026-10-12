//go:build darwin || linux || freebsd || netbsd

package worker

// PIDBitmap matches Rust's oversized `Bitmap<MAX_PROCESSES>` layout.
type PIDBitmap [MaxProcesses]uint64

func (b *PIDBitmap) test(offset uint32) bool {
	idx := offset / 64
	if idx >= uint32(len(b)) {
		return false
	}
	bit := (uint64(1) << 63) >> (offset & 63)
	return (b[idx] & bit) == bit
}

func (b *PIDBitmap) testAndSet(offset uint32, value bool) {
	idx := offset / 64
	if idx >= uint32(len(b)) {
		return
	}
	bit := (uint64(1) << 63) >> (offset & 63)
	if value {
		b[idx] |= bit
	} else {
		b[idx] &^= bit
	}
}

// PIDSet is a Rust-compatible fixed-capacity set of process IDs.
type PIDSet struct {
	values [MaxProcesses]uintptr
	bitmap PIDBitmap
	len    uintptr
}

// NewPIDSet creates a new PID set
func NewPIDSet() PIDSet {
	return PIDSet{}
}

// InsertIfAbsent inserts a value if it's not already present
func (s *PIDSet) InsertIfAbsent(pid int) bool {
	pidValue := uintptr(pid)
	for i := 0; i < MaxProcesses; i++ {
		if s.bitmap.test(uint32(i)) && s.values[i] == pidValue {
			return false
		}
	}
	if s.len >= uintptr(MaxProcesses) {
		return false
	}
	for i := 0; i < MaxProcesses; i++ {
		if s.bitmap.test(uint32(i)) {
			continue
		}
		s.values[i] = pidValue
		s.bitmap.testAndSet(uint32(i), true)
		s.len++
		return true
	}
	return false
}

// RemoveValue removes a value from the set
func (s *PIDSet) RemoveValue(pid int) bool {
	pidValue := uintptr(pid)
	for i := 0; i < MaxProcesses; i++ {
		if s.bitmap.test(uint32(i)) && s.values[i] == pidValue {
			s.bitmap.testAndSet(uint32(i), false)
			s.values[i] = 0
			if s.len > 0 {
				s.len--
			}
			return true
		}
	}
	return false
}

// Values returns all values in the set
func (s *PIDSet) Values() []int {
	// The stored length is shared with the workload; never size from it alone.
	result := make([]int, 0, min(s.len, uintptr(MaxProcesses)))
	for i := 0; i < MaxProcesses; i++ {
		if s.bitmap.test(uint32(i)) {
			result = append(result, int(s.values[i]))
		}
	}
	return result
}
