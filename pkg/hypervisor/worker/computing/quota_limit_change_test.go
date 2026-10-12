package computing

import (
	"fmt"
	"math"
	"testing"
	"time"
)

// Tests of a change of a worker's limit while the worker runs, and while the
// hypervisor is away.

const (
	// A worker counts as following its limit while every interval is within
	// limitBand of it (as a part of the limit), for limitBandFor intervals.
	limitBand    = 0.20
	limitBandFor = 20
)

// limitChange is a change of a worker's limit in the coarse simulation.
type limitChange struct {
	from, to uint32
	// neighbor is the limit of a busy worker on the same device, 0 for none.
	neighbor uint32
	// inDebt starts from a balance at the debt bound instead of a full bucket.
	inDebt bool
	// charged makes the limiter charge tokens also while the worker has no
	// limit. The one measured on a device does not.
	charged bool
	// restart makes the change while the hypervisor is away for four seconds
	// instead of while it runs.
	restart bool
}

func (lc limitChange) String() string {
	name := fmt.Sprintf("%d%% to %d%%", lc.from, lc.to)
	if lc.neighbor > 0 {
		name += fmt.Sprintf(" next to a %d%% worker", lc.neighbor)
	}
	if lc.inDebt {
		name += ", in debt"
	}
	if lc.charged {
		name += ", limiter that charges without a limit"
	}
	return name
}

// run returns what the worker and its neighbor used in every interval from
// the change on (from the first interval of the new control loop for a
// restart), and the controller's state for the worker at the end.
func (lc limitChange) run(t *testing.T, seed uint64) (worker, neighbor []float64, es *erlState) {
	t.Helper()
	rng := seedRNG(seed)
	// A worker that ends up without a limit started with one, so that its
	// neighbor has been under control before.
	w := &simWorker{uid: "worker", upLimit: lc.from, demand: 1, freeWithoutLimit: !lc.charged}
	if lc.from >= 100 {
		w.upLimit = 60
	}
	workers := []*simWorker{w}
	var n *simWorker
	if lc.neighbor > 0 {
		n = &simWorker{uid: "neighbor", upLimit: lc.neighbor, demand: 1}
		workers = append(workers, n)
	}
	sim := newClosedLoopSim(t, workers...)
	sim.selfRefill = true
	realisticTelemetry(sim, rng)
	sim.run(240 + int(seed%7))
	if lc.from >= 100 {
		w.upLimit = lc.from
		sim.run(60)
	}
	if lc.inDebt {
		// The meter is told, as if it had seen the work the debt is for.
		bound := sim.controller.erlStates["worker:gpu-sim"].debtBound
		w.deviceInfo(t).SetERLCurrentTokens(-bound)
		sim.controller.tokenMeters["worker:gpu-sim"].level, sim.controller.buckets["worker:gpu-sim"].added = -bound, 0
		if lc.from >= 100 {
			// Debt of a worker without a limit is from before it had none.
			sim.run(1)
		}
	} else {
		w.deviceInfo(t).SetERLCurrentTokens(w.deviceInfo(t).GetERLTokenCapacity())
	}

	if lc.restart {
		sim.hypervisorAway = true
		w.upLimit = lc.to
		sim.run(8)
		sim.hypervisorAway = false
		sim.startController()
	} else {
		w.upLimit = lc.to
	}
	from := len(w.history)
	sim.run(120)
	if n != nil {
		neighbor = n.history[from:]
	}
	return w.history[from:], neighbor, sim.controller.erlStates["worker:gpu-sim"]
}

// inBandAfter is the number of intervals after which a worker with the given
// limit (0-1) stays within limitBand of it for limitBandFor intervals, or the
// number of intervals there are when it never does. A worker without a limit
// is in its band while it uses nine tenths of what it can get (most).
func inBandAfter(used []float64, limit, most float64) int {
	inBand := func(v float64) bool {
		if limit >= 1 {
			return v >= 0.9*most
		}
		return math.Abs(v-limit) <= limitBand*limit
	}
	for i := 0; i+limitBandFor <= len(used); i++ {
		ok := true
		for _, v := range used[i : i+limitBandFor] {
			ok = ok && inBand(v)
		}
		if ok {
			return i
		}
	}
	return len(used)
}

// A change of a worker's limit while it runs must be followed at once, in both
// directions and from any state: without a limit (unthrottled, its limiter
// spending no tokens), with a full bucket, and in debt.
//
// Measured on a device, a worker whose limit went from 100% to 30% took 30 s
// to get to its new limit and ran at a few percent of it for the first ten:
// the rate it had was cut to the tokens it had spent in the fraction of a
// second since its limiter began to charge (100 -> 8.4 tokens/s in one
// interval, where 1500 were needed), lowered further while the average of its
// earlier utilization was still above the new limit, and then found again by
// a calibration that had started from those intervals.
//
// Asserted on the worst of twenty seeds, in control intervals after the change
// (the simulation makes it at the start of one, so the control loop sees it
// one interval later, which is the longest that takes):
//
//   - from one limit to another: within its band (every interval within 20% of
//     the limit for ten seconds) after one interval, the one in which the
//     change is seen. A worker in debt is not held up by it beyond that
//     interval.
//   - from no limit to a limit: after two intervals, the one in which the
//     change is seen and one more in which what it spends unthrottled is
//     measured; it runs unthrottled in both, as it did before the change.
//   - to no limit: at once.
//
// A neighbor stays within its own band throughout.
func TestLiveLimitChangeIsFollowedWithinTwoIntervals(t *testing.T) {
	cases := make([]limitChange, 0, 19)
	for _, neighbor := range []uint32{0, 20} {
		for _, inDebt := range []bool{false, true} {
			for _, limits := range [][2]uint32{{100, 30}, {30, 100}, {60, 20}, {20, 60}} {
				cases = append(cases, limitChange{from: limits[0], to: limits[1], neighbor: neighbor, inDebt: inDebt})
			}
		}
	}
	cases = append(cases,
		limitChange{from: 100, to: 30, charged: true}, limitChange{from: 30, to: 100, charged: true},
		limitChange{from: 100, to: 30, neighbor: 20, charged: true})
	for _, lc := range cases {
		t.Run(lc.String(), func(t *testing.T) {
			// What a worker without a limit can get next to a busy neighbor.
			most := 1 - float64(lc.neighbor)/100
			within := 1
			switch {
			case lc.to >= 100 && lc.charged:
				// Such a limiter holds the worker at its old limit until the
				// control loop has seen the change.
			case lc.to >= 100:
				within = 0
			case lc.from >= 100:
				within = 2
			}
			got := worst(func(seed uint64) []float64 {
				worker, neighbor, es := lc.run(t, seed)
				limit := float64(lc.to) / 100
				figures := []float64{float64(inBandAfter(worker, limit, most)), 0, 0, float64(es.alternating)}
				// Past the band's start the worker is at its limit on average.
				if lc.to < 100 {
					figures[1] = math.Abs(meanOf(worker[within:within+limitBandFor]) - limit)
				}
				if neighbor != nil {
					figures[2] = math.Abs(meanOf(neighbor[2:limitBandFor]) - float64(lc.neighbor)/100)
				}
				return figures
			})
			t.Logf("worst seeds: within the band after %.0f intervals (seed %d), then %.1f points from the limit on "+
				"average; the neighbor %.1f points from its own", got[0].value, got[0].seed, got[1].value*100, got[2].value*100)
			if got[0].value > float64(within) {
				t.Errorf("seed %d: the worker was within %.0f%% of its new limit after %.0f intervals, want %d",
					got[0].seed, limitBand*100, got[0].value, within)
			}
			if got[1].value > 0.03 {
				t.Errorf("seed %d: the worker was %.3f from its new limit on average", got[1].seed, got[1].value)
			}
			if got[2].value > 0.04 {
				t.Errorf("seed %d: the neighbor was %.3f from its limit after the change", got[2].seed, got[2].value)
			}
			if got[3].value > 0 {
				t.Errorf("seed %d: the rate was held for alternating work after a change of the limit", got[3].seed)
			}
		})
	}
}

func meanOf(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// A limit that was changed while the hypervisor was away is found in the
// mapping next to a rate that was published for the old one. That rate is not
// held and nothing is inherited from it: the change is followed the way it is
// while the control loop runs. Before, the rate was held for at least ten
// seconds, and with per-process readings the calibration then started from it
// as fully trusted.
//
// Asserted, on the worst of twenty seeds: counted from the new control loop's
// first interval, the worker is within its band no later than it is after the
// same change made while the control loop runs, plus the one interval in which
// the limiters refill on their own before the new loop's first pass.
func TestLimitChangedAcrossARestartIsFollowedAsFastAsALiveChange(t *testing.T) {
	for _, limits := range [][2]uint32{{50, 25}, {25, 50}, {100, 30}, {60, 20}} {
		for _, neighbor := range []uint32{0, 20} {
			live := limitChange{from: limits[0], to: limits[1], neighbor: neighbor}
			restarted := live
			restarted.restart = true
			t.Run(live.String(), func(t *testing.T) {
				most, limit := 1-float64(neighbor)/100, float64(limits[1])/100
				figures := func(lc limitChange) []worstFigure {
					return worst(func(seed uint64) []float64 {
						worker, _, _ := lc.run(t, seed)
						after := min(inBandAfter(worker, limit, most), len(worker)-40)
						return []float64{float64(after), math.Abs(meanOf(worker[after:after+40]) - limit)}
					})
				}
				a, b := figures(live), figures(restarted)
				t.Logf("worst seeds: within the band %.0f intervals after a live change (seed %d) and %.0f intervals "+
					"after the restart (seed %d); then %.1f and %.1f points from the limit over twenty seconds",
					a[0].value, a[0].seed, b[0].value, b[0].seed, a[1].value*100, b[1].value*100)
				if b[0].value > a[0].value+1 {
					t.Errorf("seed %d: within the band %.0f intervals after the restart, %.0f after a live change",
						b[0].seed, b[0].value, a[0].value)
				}
				if limit < 1 && b[1].value > 0.04 {
					t.Errorf("seed %d: %.3f from the limit over the twenty seconds after reaching it across a restart",
						b[1].seed, b[1].value)
				}
			})
		}
	}
}

// The same changes at the resolution of single launches: the worker is never
// kept waiting for longer than a throttled worker waits between two launches,
// except a worker in debt, which waits until the control loop has seen the
// change (one control interval at most); and a worker that gets a limit after
// having none is at it within two control intervals of the change.
func TestLiveLimitChangeLeavesNoAdmissionGap(t *testing.T) {
	for _, tc := range []struct {
		from, to uint32
		inDebt   bool
	}{
		{from: 100, to: 30}, {from: 30, to: 100}, {from: 60, to: 20}, {from: 20, to: 60},
		{from: 100, to: 30, inDebt: true}, {from: 30, to: 100, inDebt: true},
		{from: 60, to: 20, inDebt: true}, {from: 20, to: 60, inDebt: true},
	} {
		name := fmt.Sprintf("%d%% to %d%%", tc.from, tc.to)
		if tc.inDebt {
			name += ", in debt"
		}
		t.Run(name, func(t *testing.T) {
			worker := smallLaunchWorker("worker", tc.from)
			worker.freeWithoutLimit = true
			sim := newPacingSim(t, worker)
			// The change comes 130 ms into a control interval.
			sim.run(60*time.Second + 130*time.Millisecond)
			info := worker.info(t)
			if tc.inDebt {
				// The meter is told, as if it had seen the work the debt is for.
				bound := sim.controller.erlStates["worker:gpu-sim"].debtBound
				info.SetERLCurrentTokens(-bound)
				sim.controller.tokenMeters["worker:gpu-sim"].level, sim.controller.buckets["worker:gpu-sim"].added = -bound, 0
			}
			worker.upLimit = tc.to
			changedAt, intervals := sim.now, len(worker.history)
			sim.run(15 * time.Second)

			bound := 3 * erlRefillInterval
			if tc.inDebt {
				bound += erlUpdateInterval
			}
			// From before the change, so that the wait across it counts.
			gap := worker.maxGap(changedAt.Add(-100 * time.Millisecond))
			gap = max(gap, sim.now.Sub(worker.admissions[len(worker.admissions)-1]))
			t.Logf("longest wait between two launches across and after the change: %v", gap)
			if gap > bound {
				t.Errorf("longest wait between two launches after the change was %v, want at most %v", gap, bound)
			}
			// history[intervals] is the interval the change fell into; the two
			// after it are the ones the control loop needs at most.
			after := worker.history[intervals+3:]
			limit := float64(tc.to) / 100
			for i, used := range after[:limitBandFor] {
				if tc.to < 100 && math.Abs(used-limit) > limitBand*limit || tc.to >= 100 && used < 0.9 {
					t.Errorf("%.1f s after the change the worker used %.3f of the device with a limit of %.2f",
						float64(i+3)*erlUpdateInterval.Seconds(), used, limit)
				}
			}
		})
	}
}
