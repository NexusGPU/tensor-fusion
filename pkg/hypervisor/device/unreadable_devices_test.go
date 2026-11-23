package device

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestUnreadableDevices(t *testing.T) {
	start := time.Unix(1700000000, 0)
	known := []string{"gpu-0", "gpu-1", "gpu-2"}
	all := func(string) bool { return true }
	without := func(lost ...string) func(string) bool {
		return func(uuid string) bool {
			for _, l := range lost {
				if l == uuid {
					return false
				}
			}
			return true
		}
	}

	t.Run("a device unreadable for the grace period is reported once", func(t *testing.T) {
		u := &unreadableDevices{grace: 2 * time.Minute}
		assert.Empty(t, u.observe(start, known, all))
		assert.Empty(t, u.observe(start.Add(time.Minute), known, without("gpu-1")))
		assert.Empty(t, u.observe(start.Add(2*time.Minute), known, without("gpu-1")),
			"one minute after the first failed query is within the grace period")
		assert.Equal(t, []string{"gpu-1"}, u.observe(start.Add(3*time.Minute), known, without("gpu-1")))
		assert.Empty(t, u.observe(start.Add(4*time.Minute), known, without("gpu-1")),
			"a device the provider still lists is not discovered again and again")
	})

	t.Run("a sample ends the run of failures", func(t *testing.T) {
		u := &unreadableDevices{grace: 2 * time.Minute}
		assert.Empty(t, u.observe(start, known, without("gpu-1")))
		assert.Empty(t, u.observe(start.Add(time.Minute), known, all))
		assert.Empty(t, u.observe(start.Add(3*time.Minute), known, without("gpu-1")),
			"the grace period starts over")
		assert.Equal(t, []string{"gpu-1"}, u.observe(start.Add(5*time.Minute), known, without("gpu-1")))

		// Readable again after it was reported, then lost again: reported again.
		assert.Empty(t, u.observe(start.Add(6*time.Minute), known, all))
		assert.Empty(t, u.observe(start.Add(7*time.Minute), known, without("gpu-1")))
		assert.Equal(t, []string{"gpu-1"}, u.observe(start.Add(9*time.Minute), known, without("gpu-1")))
	})

	t.Run("a query that reads nothing counts for every device", func(t *testing.T) {
		u := &unreadableDevices{grace: 2 * time.Minute}
		none := func(string) bool { return false }
		assert.Empty(t, u.observe(start, known, none))
		assert.ElementsMatch(t, known, u.observe(start.Add(2*time.Minute), known, none))
	})

	t.Run("a removed device is forgotten", func(t *testing.T) {
		u := &unreadableDevices{grace: 2 * time.Minute}
		assert.Empty(t, u.observe(start, known, without("gpu-1")))
		assert.Equal(t, []string{"gpu-1"}, u.observe(start.Add(2*time.Minute), known, without("gpu-1")))
		// The discovery removed it; it comes back later and fails again.
		assert.Empty(t, u.observe(start.Add(3*time.Minute), []string{"gpu-0", "gpu-2"}, all))
		assert.Empty(t, u.observe(start.Add(10*time.Minute), known, without("gpu-1")))
		assert.Equal(t, []string{"gpu-1"}, u.observe(start.Add(12*time.Minute), known, without("gpu-1")))
	})
}
