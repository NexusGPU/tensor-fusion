package computing

import (
	"strings"
	"time"
)

// attributedUtilization is the utilization the control loop attributed to one
// worker on one device in its latest interval.
type attributedUtilization struct {
	value float64 // share of the device, 0-1
	at    time.Time
}

// recordAttributedUtilization keeps the control loop's latest attribution for a
// bucket, for consumers outside the loop (usage metrics).
func (c *Controller) recordAttributedUtilization(key string, value float64, at time.Time) {
	c.erlStatesMu.Lock()
	defer c.erlStatesMu.Unlock()
	if c.attributed == nil {
		c.attributed = make(map[string]attributedUtilization)
	}
	c.attributed[key] = attributedUtilization{value: value, at: at}
}

// AttributedUtilization returns the share of a device (0-1) the control loop
// attributed to a worker in its latest interval: the worker's own utilization,
// not the device's. It reports false when the loop has no signal for the worker
// or its latest attribution is older than maxAge.
func (c *Controller) AttributedUtilization(
	workerUID, deviceUUID string, now time.Time, maxAge time.Duration,
) (float64, bool) {
	c.erlStatesMu.Lock()
	defer c.erlStatesMu.Unlock()
	latest, ok := c.attributed[workerUID+":"+strings.ToLower(deviceUUID)]
	if !ok {
		return 0, false
	}
	if age := now.Sub(latest.at); age > maxAge || age < -maxAge {
		return 0, false
	}
	return latest.value, true
}
