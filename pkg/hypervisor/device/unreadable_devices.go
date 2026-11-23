package device

import (
	"sync"
	"time"
)

// defaultUnreadableGrace is how long a device may go without a metrics sample
// before the devices are discovered again. It spans a few metrics intervals,
// so one failed query or a driver hiccup does not count.
const defaultUnreadableGrace = 2 * time.Minute

// unreadableDevices follows the devices the metrics queries cannot read.
//
// The device list changes only when the devices are discovered, which happens
// at start and then rarely, while the metrics are read from the driver all the
// time. A GPU that fails at runtime (fallen off the bus: nvidia-smi reports
// "Unable to determine the device handle") shows up there first, as a device
// without a sample. When that lasts, the devices are discovered again, and a
// device the provider no longer lists is taken out of scheduling.
type unreadableDevices struct {
	mu    sync.Mutex
	grace time.Duration
	// since is when each device was first found unreadable in its current run
	// of failed queries.
	since map[string]time.Time
	// asked holds the devices a rediscovery was already requested for in that
	// run. A device the provider still lists although its metrics cannot be
	// read is not discovered over and over.
	asked map[string]struct{}
}

// observe records one metrics query over the known devices, read telling which
// of them returned a sample. It returns the devices that have now been
// unreadable for the grace period and were not reported before: the caller
// discovers the devices again when there is any.
func (u *unreadableDevices) observe(now time.Time, known []string, read func(uuid string) bool) []string {
	u.mu.Lock()
	defer u.mu.Unlock()
	if u.since == nil {
		u.since = make(map[string]time.Time)
		u.asked = make(map[string]struct{})
	}

	var overdue []string
	stillKnown := make(map[string]struct{}, len(known))
	for _, uuid := range known {
		stillKnown[uuid] = struct{}{}
		if read(uuid) {
			delete(u.since, uuid)
			delete(u.asked, uuid)
			continue
		}
		first, failing := u.since[uuid]
		if !failing {
			u.since[uuid] = now
			continue
		}
		if _, asked := u.asked[uuid]; asked || now.Sub(first) < u.grace {
			continue
		}
		u.asked[uuid] = struct{}{}
		overdue = append(overdue, uuid)
	}
	// A device that is no longer known was removed by a discovery.
	for uuid := range u.since {
		if _, ok := stillKnown[uuid]; !ok {
			delete(u.since, uuid)
			delete(u.asked, uuid)
		}
	}
	return overdue
}
