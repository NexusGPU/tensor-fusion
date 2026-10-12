package computing

import (
	"fmt"
	"math"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
)

// Tests of the burst window: what a worker may save up while it has nothing to
// run, and spend at full speed when it resumes. The rule is the comment at
// burstWindow.

var (
	// The windows the default is going to be chosen from, and the shares they
	// are tested at.
	burstTestWindows = []float64{0.1, 0.5, 1, 2}
	burstTestShares  = []uint32{25, 50}
)

// ownBurstWindows gives workers of a simulation a burst window of their own
// (seconds, by UID), the way the annotation does. The map is read in every
// control interval, so a test changes a window in place by changing the map.
func ownBurstWindows(controller *Controller, windows map[string]float64) {
	inner := controller.workerInfoFn
	controller.workerInfoFn = func() map[string]*WorkerInfoSnapshot {
		snapshots := inner()
		for uid, window := range windows {
			if snapshot := snapshots[uid]; snapshot != nil {
				for i := range snapshot.Devices {
					snapshot.Devices[i].BurstWindow = &window
				}
			}
		}
		return snapshots
	}
}

// fullSpeedFor is how long the worker ran without waiting for tokens from
// `from` on: up to the end of the last launch that was admitted the moment the
// one before it ended.
func (w *pacedWorker) fullSpeedFor(from time.Time) time.Duration {
	end := from
	for _, at := range w.admissions {
		if at.Before(from) {
			continue
		}
		if at.Sub(end) >= pacedResolution {
			break
		}
		end = at.Add(w.launchTime)
	}
	return end.Sub(from)
}

// burstFormula is how long a worker with the given share runs at full speed on
// a bucket of `window` seconds of its refill: share x W / (1 - share).
func burstFormula(share uint32, window float64) time.Duration {
	s := float64(share) / 100
	return time.Duration(s * window / (1 - s) * float64(time.Second))
}

// burstTolerance is how far a measured burst may be from the formula: a tenth,
// because the rate is held within a few percent of the one that gives the
// share and the burst's length follows from the difference between spending
// and earning, and two refill intervals, because the bucket is spent and
// refilled in portions.
func burstTolerance(want time.Duration) time.Duration {
	return want/10 + 2*erlRefillInterval
}

// A worker that wakes up with a full bucket runs at full speed for
// share x W / (1 - share) and is paced at its share from then on, whether the
// window is the pool's or the worker's own. However long it was idle before,
// the burst is no longer than that: its rate did not climb while it had
// nothing to run, and the bucket holds W seconds of refill and no more.
func TestBurstAfterIdleLastsItsWindowAndNoLonger(t *testing.T) {
	for _, source := range []string{"pool", "workload"} {
		for _, share := range burstTestShares {
			for _, window := range burstTestWindows {
				t.Run(fmt.Sprintf("%s/%d%%/W=%gs", source, share, window), func(t *testing.T) {
					worker := smallLaunchWorker("worker", share)
					sim := newPacingSim(t, worker)
					if source == "pool" {
						sim.controller.config.burstWindow = window
					} else {
						ownBurstWindows(sim.controller, map[string]float64{"worker": window})
					}
					sim.run(60 * time.Second)
					info, limit := worker.info(t), float64(share)/100
					rate := info.GetERLTokenRefillRate()
					want := burstFormula(share, window)

					idles := []time.Duration{time.Duration((window + 1) * float64(time.Second)), 2 * time.Minute}
					for _, idle := range idles {
						worker.paused = true
						sim.run(idle)
						if got := info.GetERLTokenRefillRate(); math.Abs(got-rate) > 0.05*rate {
							t.Errorf("the refill rate moved from %.0f to %.0f tokens/s in %v without work", rate, got, idle)
						}
						capacity, banked := info.GetERLTokenCapacity(), info.GetERLCurrentTokens()
						if math.Abs(capacity-info.GetERLTokenRefillRate()*window) > 1e-6*capacity {
							t.Errorf("capacity = %.1f tokens at %.1f tokens/s, want %g s of refill", capacity, rate, window)
						}
						if banked != capacity {
							t.Errorf("after %v without work the bucket holds %.1f of %.1f tokens", idle, banked, capacity)
						}

						worker.paused = false
						wake, intervals := sim.now, len(worker.history)
						const observed = 20 * time.Second
						sim.run(observed)
						burst := worker.fullSpeedFor(wake)
						if diff := (burst - want).Abs(); diff > burstTolerance(want) {
							t.Errorf("after %v without work the worker ran at full speed for %v, want %v ± %v",
								idle, burst, want, burstTolerance(want))
						}
						// From the control interval after the one the burst ended in,
						// the worker is at its share.
						resumed := worker.history[intervals:]
						settled := int(burst/erlUpdateInterval) + 2
						for i := settled; i < len(resumed); i++ {
							if resumed[i] > limit+0.08 {
								t.Errorf("%.1f s after waking up the worker used %.2f of the device",
									float64(i)*erlUpdateInterval.Seconds(), resumed[i])
							}
						}
						assertNear(t, "utilization after the burst", worker.meanUtil(16), limit, simUtilTolerance)
						// Over the whole stretch it got its share of that time and of
						// one window, and no more.
						used := 0.0
						for _, interval := range resumed {
							used += interval * erlUpdateInterval.Seconds()
						}
						if bound := limit * (observed.Seconds() + window) * 1.05; used > bound {
							t.Errorf("in the %v after waking up the worker used the device for %.2f s, bound %.2f s",
								observed, used, bound)
						}
						t.Logf("BURST %s share=%d%% W=%gs idle=%v: full speed for %v (formula %v), "+
							"device time in %v: %.2f s (share alone %.2f s)",
							source, share, window, idle, burst, want, observed, used, limit*observed.Seconds())
						rate = info.GetERLTokenRefillRate()
					}
					if sim.maxBalance > 1+1e-9 {
						t.Errorf("a bucket held %.3f times its capacity", sim.maxBalance)
					}
				})
			}
		}
	}
}

// A worker that alternates between work and pauses uses what it saved in the
// pauses, and that is all: over many cycles its utilization is its share of the
// time plus one window, never more, and its rate is what it was. Each cycle is
// W without work, which fills the bucket, and 1.5 W of work, so without saving
// up the worker would get six tenths of its share.
func TestBurstyWorkerStaysWithinItsShareInTheLongRun(t *testing.T) {
	for _, share := range burstTestShares {
		for _, window := range burstTestWindows {
			t.Run(fmt.Sprintf("%d%%/W=%gs", share, window), func(t *testing.T) {
				worker := smallLaunchWorker("worker", share)
				sim := newPacingSim(t, worker)
				ownBurstWindows(sim.controller, map[string]float64{"worker": window})
				sim.run(60 * time.Second)
				info, limit := worker.info(t), float64(share)/100
				rate := info.GetERLTokenRefillRate()

				off := time.Duration(window * float64(time.Second))
				on := 3 * off / 2
				cycles := int(math.Ceil(float64(2*time.Minute) / float64(on+off)))
				from, highest := len(worker.history), 0.0
				for range cycles {
					worker.paused = true
					sim.run(off)
					worker.paused = false
					sim.run(on)
					highest = math.Max(highest, info.GetERLTokenRefillRate())
				}
				total := time.Duration(cycles) * (on + off)
				used := 0.0
				for _, interval := range worker.history[from:] {
					used += interval * erlUpdateInterval.Seconds()
				}
				utilization := used / total.Seconds()
				// Two points for the rate being held within a few percent of the
				// one that gives the share.
				if bound := limit*(1+window/total.Seconds()) + 0.02; utilization > bound {
					t.Errorf("long-run utilization of the bursty worker = %.3f, bound %.3f", utilization, bound)
				}
				// And it does get what it saved: its whole share, not the six
				// tenths of it that work at its share during 1.5 W of 2.5 W gives.
				if utilization < limit-simUtilTolerance {
					t.Errorf("long-run utilization of the bursty worker = %.3f, it did not get its share of %.2f",
						utilization, limit)
				}
				if highest > 1.1*rate {
					t.Errorf("the refill rate rose from %.0f to %.0f tokens/s under bursty work", rate, highest)
				}
				t.Logf("BURSTY share=%d%% W=%gs: %d cycles of %v without work and %v of work: long-run utilization %.3f "+
					"(limit %.2f, without saving up %.3f), highest rate %.0f of %.0f tokens/s",
					share, window, cycles, off, on, utilization, limit, limit*0.6, highest, rate)
			})
		}
	}
}

// A burst takes device time from a busy neighbor only while it lasts. Two
// workloads that both run get half of the device each, so a burst reaches a
// neighbor whose share is above half: on a device the two fill between them,
// the neighbor of a 25% worker is down from 75% to that half for as long as the
// burst lasts, which is W there (the burster spends at half speed), and is back
// within five points of its share in the control interval after the burst
// ended. A 50% worker next to a 50% neighbor cannot take more than its share
// while the neighbor runs, whatever it saved, and the neighbor does not notice.
// Across seeds of the telemetry's timing.
func TestNeighborDipIsBoundedByTheBurst(t *testing.T) {
	period := erlRefillInterval.Seconds()
	// lowestMean is the lowest mean of `window` consecutive refill periods and
	// the last period at which such a mean was below `band`, -1 if none was.
	lowestMean := func(periods []float64, window int, band float64) (lowest float64, lastBelow int) {
		lowest, lastBelow = 1, -1
		sum := 0.0
		for k, used := range periods {
			sum += used
			if k >= window {
				sum -= periods[k-window]
			}
			if k < window-1 {
				continue
			}
			mean := sum / float64(window)
			lowest = math.Min(lowest, mean)
			if mean < band {
				lastBelow = k
			}
		}
		return lowest, lastBelow
	}
	for _, share := range burstTestShares {
		for _, own := range burstTestWindows {
			t.Run(fmt.Sprintf("%d%%/W=%gs", share, own), func(t *testing.T) {
				neighborShare := float64(100-share) / 100
				got := worst(func(seed uint64) []float64 {
					burster := &simWorker{uid: "burster", upLimit: share, demand: 1}
					neighbor := &simWorker{uid: "neighbor", upLimit: 100 - share, demand: 1}
					sim := newClosedLoopSim(t, burster, neighbor)
					ownBurstWindows(sim.controller, map[string]float64{"burster": own})
					realisticTelemetry(sim, seedRNG(seed))
					sim.run(240)
					burster.demand = 0
					sim.run(int(own/erlUpdateInterval.Seconds()) + 20)
					before := neighbor.meanOver(len(neighbor.history)-10, len(neighbor.history))

					sim.recordPeriods = true
					burster.demand = 1
					sim.run(40)

					// The burst ends with the first refill period in which the
					// burster has to wait for tokens.
					burstEnd := len(burster.periods)
					for k, used := range burster.periods {
						if used == 0 {
							burstEnd = k
							break
						}
					}
					// The neighbor's use over 100 ms, and over a control interval,
					// which is what its band is about.
					brief, _ := lowestMean(neighbor.periods, 10, 0)
					lowest, lastBelow := lowestMean(neighbor.periods, simRefillsPerInterval, neighborShare-0.05)
					recovered := 0.0
					if end := burstEnd + simRefillsPerInterval; end <= len(neighbor.periods) {
						mean := 0.0
						for _, used := range neighbor.periods[burstEnd:end] {
							mean += used / float64(simRefillsPerInterval)
						}
						recovered = math.Abs(mean - neighborShare)
					}
					burst := float64(burstEnd) * period
					return []float64{
						neighborShare - lowest, float64(lastBelow+1-burstEnd) * period, recovered, burst, -burst,
						math.Abs(before - neighborShare), neighborShare - brief,
					}
				})
				depth, overrun, recovered, longest, shortest := got[0], got[1], got[2], got[3], got[4]
				before, brief := got[5], got[6]
				t.Logf("DIP share=%d%% W=%gs neighbor=%.0f%%: burst %.2f to %.2f s; neighbor down by at most %.0f points "+
					"over 100 ms (seed %d) and %.0f points over a control interval (seed %d), below its band until %.2f s "+
					"after the burst ended at the latest (seed %d), %.1f points from its share in the control interval "+
					"after the burst (seed %d)",
					share, own, neighborShare*100, -shortest.value, longest.value, brief.value*100, brief.seed,
					depth.value*100, depth.seed, overrun.value, overrun.seed, recovered.value*100, recovered.seed)
				if before.value > 0.06 {
					t.Fatalf("seed %d: the neighbor was %.3f from its share before the burst", before.seed, before.value)
				}
				// Two workloads that both run get half of the device each.
				if bound := math.Max(neighborShare-0.5, 0) + 0.06; depth.value > bound {
					t.Errorf("seed %d: the neighbor was down by %.3f of the device over a control interval, bound %.3f",
						depth.seed, depth.value, bound)
				}
				if share < 50 {
					// Where the burster can take more than its share, it does so
					// for W: it spends at half speed and earns at a quarter.
					if -shortest.value < 0.75*own || longest.value > 1.15*own+2*period {
						t.Errorf("bursts lasted %.2f to %.2f s next to the neighbor, want about %g s",
							-shortest.value, longest.value, own)
					}
				}
				// Below its band no later than one control interval after the
				// burst ended.
				if overrun.value > erlUpdateInterval.Seconds() {
					t.Errorf("seed %d: the neighbor was below its band until %.2f s after the burst ended",
						overrun.seed, overrun.value)
				}
				if recovered.value > 0.06 {
					t.Errorf("seed %d: in the control interval after the burst the neighbor was %.3f from its share",
						recovered.seed, recovered.value)
				}
			})
		}
	}
}

// The burst window of a running worker is changed in place: the control loop
// reads it in every interval, and nothing is restarted. A longer window is
// filled at the worker's rate, so nothing is handed out for the change. A
// shorter one takes what the bucket holds above the new capacity, within one
// refill interval, and nothing below it. Without a window of its own the
// worker has the pool's again.
func TestBurstWindowChangesInPlace(t *testing.T) {
	const share = 25
	worker := smallLaunchWorker("worker", share)
	sim := newPacingSim(t, worker)
	windows := map[string]float64{"worker": 0.1}
	ownBurstWindows(sim.controller, windows)
	sim.run(60 * time.Second)
	info := worker.info(t)
	state := sim.controller.erlStates["worker:gpu-sim"]
	rate := info.GetERLTokenRefillRate()
	// The change is seen by the next control interval and applied by the
	// refill pass after it.
	applied := erlUpdateInterval + 2*erlRefillInterval

	// Longer: 0.1 s to 2 s, on a bucket that is full.
	worker.paused = true
	sim.run(time.Second)
	before := info.GetERLCurrentTokens()
	windows["worker"] = 2
	sim.run(applied)
	if got := info.GetERLTokenCapacity(); math.Abs(got-2*rate) > 1e-6*got {
		t.Fatalf("capacity = %.1f tokens after the window became 2 s at %.1f tokens/s", got, rate)
	}
	if got := info.GetERLCurrentTokens(); got > before+rate*applied.Seconds()+1e-6 {
		t.Errorf("the balance went from %.1f to %.1f tokens in %v: more than the worker earned at %.1f tokens/s",
			before, got, applied, rate)
	}
	sim.run(2500 * time.Millisecond)
	worker.paused = false
	wake := sim.now
	sim.run(5 * time.Second)
	want := burstFormula(share, 2)
	if burst := worker.fullSpeedFor(wake); (burst - want).Abs() > burstTolerance(want) {
		t.Errorf("with a window of 2 s set in place the worker ran at full speed for %v, want %v ± %v",
			burst, want, burstTolerance(want))
	} else {
		t.Logf("INPLACE 0.1 s to 2 s: full speed for %v (formula %v)", burst, want)
	}

	// Shorter: 2 s to 0.5 s, again on a full bucket.
	worker.paused = true
	sim.run(3 * time.Second)
	rate = info.GetERLTokenRefillRate()
	if got := info.GetERLCurrentTokens(); math.Abs(got-2*rate) > 1e-6*got {
		t.Fatalf("the bucket holds %.1f tokens before the window is shortened, want %.1f", got, 2*rate)
	}
	windows["worker"] = 0.5
	sim.run(applied)
	if got, capacity := info.GetERLCurrentTokens(), info.GetERLTokenCapacity(); got != capacity ||
		math.Abs(capacity-0.5*rate) > 1e-6*capacity {
		t.Errorf("after the window became 0.5 s the bucket holds %.1f of %.1f tokens, want %.1f of %.1f: "+
			"what was above the new capacity is taken, nothing else", got, capacity, 0.5*rate, 0.5*rate)
	}
	worker.paused = false
	wake = sim.now
	sim.run(5 * time.Second)
	want = burstFormula(share, 0.5)
	if burst := worker.fullSpeedFor(wake); (burst - want).Abs() > burstTolerance(want) {
		t.Errorf("with a window of 0.5 s set in place the worker ran at full speed for %v, want %v ± %v",
			burst, want, burstTolerance(want))
	} else {
		t.Logf("INPLACE 2 s to 0.5 s: full speed for %v (formula %v)", burst, want)
	}

	// A bucket that is not full keeps what it holds when the window shrinks to
	// something that still holds it.
	sim.run(5 * time.Second)
	windows["worker"] = 2
	sim.run(applied)
	held := info.GetERLCurrentTokens()
	windows["worker"] = 1
	sim.run(erlUpdateInterval)
	if got := info.GetERLCurrentTokens(); got < held-worker.launchCost*float64(erlUpdateInterval/worker.launchTime) {
		t.Errorf("a busy worker's balance went from %.1f to %.1f tokens when its window shrank above it", held, got)
	}

	// No window of its own any more: the pool's.
	worker.paused = true
	delete(windows, "worker")
	sim.run(time.Second)
	rate = info.GetERLTokenRefillRate()
	if got := info.GetERLTokenCapacity(); math.Abs(got-burstWindow*rate) > 1e-6*got {
		t.Errorf("capacity = %.1f tokens without a window of its own at %.1f tokens/s, want the pool's %g s",
			got, rate, burstWindow)
	}
	if sim.controller.erlStates["worker:gpu-sim"] != state {
		t.Error("the control state of the worker was replaced by a change of its window")
	}
	worker.paused = false
	sim.run(20 * time.Second)
	assertNear(t, "utilization after the changes", worker.meanUtil(20), share/100.0, simUtilTolerance)
}

// The bounds of a window, whoever asks for it: 0 to burstWindowMax, and a
// bucket never holds less than two refill intervals, so the refill loop can
// hand out everything a worker earns. A worker with a window of 0 is held at
// its share and paced as evenly as any other; one with the longest window is
// held at its share too.
func TestBurstWindowBounds(t *testing.T) {
	seconds := func(v float64) *float64 { return &v }
	controller := NewQuotaController(&quotaMetricsDeviceController{}, nil).(*Controller)
	for _, tc := range []struct {
		name string
		pool float64
		own  *float64
		want float64
	}{
		{name: "the pool's", pool: 0.1, want: 0.1},
		{name: "its own", pool: 0.1, own: seconds(2), want: 2},
		{name: "its own, shorter than the pool's", pool: 2, own: seconds(0.5), want: 0.5},
		{name: "zero", pool: 0.1, own: seconds(0), want: burstWindowFloor},
		{name: "below two refill intervals", pool: 0.1, own: seconds(0.005), want: burstWindowFloor},
		{name: "a pool's zero", pool: 0, want: burstWindowFloor},
		{name: "negative", pool: 0.1, own: seconds(-1), want: burstWindowFloor},
		{name: "the longest", pool: 0.1, own: seconds(10), want: 10},
		{name: "above the longest", pool: 0.1, own: seconds(3600), want: burstWindowMax},
		{name: "not a number", pool: 0.5, own: seconds(math.NaN()), want: 0.5},
	} {
		controller.config.burstWindow = tc.pool
		if got := controller.burstWindowOf(DeviceSnapshot{BurstWindow: tc.own}); got != tc.want {
			t.Errorf("%s: burst window = %v s, want %v s", tc.name, got, tc.want)
		}
	}
	if burstWindowMax != constants.ComputeBurstWindowMax.Seconds() || burstWindowFloor != 2*erlRefillInterval.Seconds() {
		t.Errorf("bounds = [%v, %v] s", burstWindowFloor, burstWindowMax)
	}
	// The compiled configuration clips no window at any rate.
	cfg := defaultERLConfig()
	if cfg.capacityMax < cfg.rateMax*burstWindowMax || cfg.burstWindow != burstWindow {
		t.Errorf("compiled configuration: %+v", cfg)
	}

	for _, window := range []float64{0, 10} {
		t.Run(fmt.Sprintf("W=%gs", window), func(t *testing.T) {
			worker := smallLaunchWorker("worker", 25)
			sim := newPacingSim(t, worker)
			ownBurstWindows(sim.controller, map[string]float64{"worker": window})
			sim.run(60 * time.Second)
			measureFrom := sim.now
			sim.run(30 * time.Second)
			info := worker.info(t)
			rate, capacity := info.GetERLTokenRefillRate(), info.GetERLTokenCapacity()
			if want := rate * math.Max(window, burstWindowFloor); math.Abs(capacity-want) > 1e-6*want {
				t.Errorf("capacity = %.1f tokens at %.1f tokens/s, want %.1f", capacity, rate, want)
			}
			assertNear(t, "mean utilization", worker.meanUtil(60), 0.25, simUtilTolerance)
			if gap := worker.maxGap(measureFrom); gap > 4*erlRefillInterval {
				t.Errorf("longest wait between two launches was %v", gap)
			}
			worker.paused = true
			sim.run(15 * time.Second)
			worker.paused = false
			wake := sim.now
			sim.run(10 * time.Second)
			want := burstFormula(25, math.Max(window, burstWindowFloor))
			if burst := worker.fullSpeedFor(wake); (burst - want).Abs() > burstTolerance(want) {
				t.Errorf("the worker ran at full speed for %v after 15 s without work, want %v ± %v",
					burst, want, burstTolerance(want))
			}
		})
	}
}

// Where a capacity bound decides instead of the window, the window in effect is
// capacity / rate, and the control loop knows which bound it was (that is what
// it logs): the floor for work so cheap that a window of its rate is less than
// a token, the ceiling when a pool set one below what a workload asks for.
func TestCapacityBoundsDecideTheEffectiveWindow(t *testing.T) {
	t.Run("capacity floor", func(t *testing.T) {
		worker := &pacedWorker{uid: "worker", upLimit: 30, launchCost: 0.1, launchTime: 50 * time.Millisecond}
		sim := newPacingSim(t, worker)
		sim.resolution = 10 * time.Millisecond
		sim.run(90 * time.Second)
		info, state := worker.info(t), sim.controller.erlStates["worker:gpu-sim"]
		if info.GetERLTokenCapacity() != capacityMin || state.windowHeld != "capacityMin" || state.window != burstWindow {
			t.Errorf("capacity = %v tokens at %.2f tokens/s, held by %q with a window of %v s",
				info.GetERLTokenCapacity(), info.GetERLTokenRefillRate(), state.windowHeld, state.window)
		}
	})
	t.Run("capacity ceiling set by the pool", func(t *testing.T) {
		const share = 25
		worker := smallLaunchWorker("worker", share)
		sim := newPacingSim(t, worker)
		// About 0.4 s of this worker's refill.
		sim.controller.config.capacityMax = 1000
		ownBurstWindows(sim.controller, map[string]float64{"worker": 2})
		sim.run(60 * time.Second)
		info, state := worker.info(t), sim.controller.erlStates["worker:gpu-sim"]
		if info.GetERLTokenCapacity() != 1000 || state.windowHeld != "capacityMax" || state.window != 2 {
			t.Fatalf("capacity = %v tokens, held by %q with a window of %v s",
				info.GetERLTokenCapacity(), state.windowHeld, state.window)
		}
		effective := info.GetERLTokenCapacity() / info.GetERLTokenRefillRate()
		worker.paused = true
		sim.run(5 * time.Second)
		worker.paused = false
		wake := sim.now
		sim.run(5 * time.Second)
		want := burstFormula(share, effective)
		if burst := worker.fullSpeedFor(wake); (burst - want).Abs() > burstTolerance(want) {
			t.Errorf("the worker ran at full speed for %v, want %v ± %v for the %.2f s the ceiling leaves of its window",
				burst, want, burstTolerance(want), effective)
		}
	})
}

// A hypervisor that is restarted finds buckets whose capacity was derived with
// the worker's own window, and keeps their rate. A mapping no control loop has
// published to is not mistaken for one, although its initial values are ten
// seconds of its initial rate: with a window of 10 s it was adopted at that
// rate and held there for ten seconds.
func TestRestartKeepsTheRateOfAWorkerWithItsOwnWindow(t *testing.T) {
	for _, window := range []float64{0, 0.5, 2, 10} {
		t.Run(fmt.Sprintf("W=%gs", window), func(t *testing.T) {
			worker := smallLaunchWorker("worker", 25)
			sim := newPacingSim(t, worker)
			windows := map[string]float64{"worker": window}
			ownBurstWindows(sim.controller, windows)
			sim.run(60 * time.Second)
			rate := worker.info(t).GetERLTokenRefillRate()

			sim.startController()
			ownBurstWindows(sim.controller, windows)
			sim.run(erlUpdateInterval + 2*pacedResolution)
			state := sim.controller.erlStates["worker:gpu-sim"]
			if state == nil || state.holdIntervals == 0 || state.currentRate != rate {
				t.Fatalf("after the restart the rate of %.0f tokens/s was not kept: %+v", rate, state)
			}
			sim.run(20 * time.Second)
			assertNear(t, "utilization after the restart", worker.meanUtil(20), 0.25, simUtilTolerance)
		})
	}

	t.Run("new mapping", func(t *testing.T) {
		worker := smallLaunchWorker("worker", 25)
		worker.paused = true
		sim := newPacingSim(t, worker)
		info := worker.info(t)
		fresh := workerstate.NewSharedDeviceInfoV2(0, 25, 0)
		if info.GetERLTokenRefillRate() != newMappingRate || info.GetERLTokenCapacity() != newMappingCapacity ||
			fresh.GetERLTokenRefillRate() != newMappingRate || fresh.GetERLTokenCapacity() != newMappingCapacity {
			t.Fatalf("a new mapping carries %v tokens/s and %v tokens", info.GetERLTokenRefillRate(), info.GetERLTokenCapacity())
		}
		ownBurstWindows(sim.controller, map[string]float64{"worker": 10})
		sim.run(erlUpdateInterval + 2*pacedResolution)
		state := sim.controller.erlStates["worker:gpu-sim"]
		if state == nil || state.holdIntervals != 0 || state.currentRate < initialRate {
			t.Errorf("a new mapping was taken for one a control loop published: %+v", state)
		}
	})
}

// The pool's window reaches the control loop from the scheduling configuration
// the operator hands to the hypervisor, including the 0.5 s of the earlier
// charts when it is the only parameter set.
func TestPoolBurstWindowIsLoadedFromTheSchedulingConfiguration(t *testing.T) {
	for raw, want := range map[string]float64{"0.5": 0.5, "2": 2, "0": 0, "10": 10, "11": burstWindow, "-1": burstWindow} {
		t.Setenv(constants.HypervisorSchedulingConfigEnv,
			`{"elasticRateLimitParameters":{"burstWindow":"`+raw+`"}}`)
		got, expected := loadERLConfigFromEnv(), defaultERLConfig()
		expected.burstWindow = want
		if got != expected {
			t.Errorf("burstWindow %q: configuration = %+v, want %+v", raw, got, expected)
		}
	}
	// All ten values of the earlier charts, should they reach a hypervisor after
	// all: the bucket bounds among them are not ones that work, and the window
	// that came with them is not applied on its own.
	cfg, _, rejected := applyERLOverrides(defaultERLConfig(), tfv1.ElasticRateLimitParameters{
		Kp: "0.12", Ki: "0.4", Kd: "1.0", FilterAlpha: "0.30", BurstWindow: "0.5", IntegralDecayFactor: "0.85",
		CapacityMin: "150", CapacityMax: "6000", MinRefillRate: "80", MaxRefillRate: "12000",
	})
	if cfg.burstWindow != burstWindow || cfg.capacityMax != capacityMax || len(rejected) == 0 {
		t.Errorf("the earlier charts' block gave %+v (rejected: %v)", cfg, rejected)
	}
}

// A worker that runs above its target while its bucket is not empty is spending
// what it saved. The bleed takes no more from it than it takes from a bucket of
// the default window; before, it took a bucket of any size down to 35% of its
// capacity, 40% of the capacity per control interval, which cut a burst that
// was still running after two or three intervals.
func TestBleedLeavesTheBurstBudgetOfALongerWindow(t *testing.T) {
	const rate, target, used = 1000.0, 0.5, 1.0
	bleedable := rate * burstWindow
	for _, tc := range []struct {
		name                       string
		capacity, tokens, wantLeft float64
	}{
		{name: "default window, full", capacity: bleedable, tokens: bleedable,
			wantLeft: bleedable * (1 - tokenDrainRatio*0.5)},
		{name: "default window, half spent", capacity: bleedable, tokens: 0.5 * bleedable,
			wantLeft: bleedable * tokenReserveRatio},
		{name: "2 s, full", capacity: 2 * rate, tokens: 2 * rate, wantLeft: 2*rate - bleedable*(1-tokenReserveRatio)},
		{name: "2 s, half spent", capacity: 2 * rate, tokens: rate, wantLeft: rate},
		{name: "2 s, above a capacity that shrank", capacity: 2 * rate, tokens: 5 * rate,
			wantLeft: 5*rate - 0.5*tokenDrainRatio*2*rate},
	} {
		info := workerstate.NewSharedDeviceInfoV2(1024, 50, 1<<30)
		info.SetERLTokenCapacity(tc.capacity)
		info.SetERLCurrentTokens(tc.tokens)
		left, removed := bleedTokenBucket(info, tc.capacity, bleedable, target, used)
		if math.Abs(left-tc.wantLeft) > 1e-9 || math.Abs(removed-(tc.tokens-left)) > 1e-9 {
			t.Errorf("%s: %.1f tokens became %.1f (removed %.1f), want %.1f", tc.name, tc.tokens, left, removed, tc.wantLeft)
		}
	}
}
