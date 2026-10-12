package computing

import (
	"context"
	"math"
	"math/rand/v2"
	"strconv"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
)

// Tests of what one worker can do to another through the limiter: a worker's
// limit must hold, and its allocation must not move, whatever happens to its
// neighbor's bucket, and whatever the platform does to the hypervisor.

// A balance that no limiter produces is not a measurement, in the interval in
// which it shows up and for a while after, so that a balance written up and
// down in turns never yields a figure.
// simWriterUID names the simulated worker that writes its own balance.
const simWriterUID = "writer"

func TestTokenMeterReportsOnlyWhatALimiterCanProduce(t *testing.T) {
	look := func(m *tokenMeter, tokens, added float64) {
		m.observe(tokens, added, m.updatedAt+0.5)
	}
	fresh := func() *tokenMeter {
		return &tokenMeter{level: 100, updatedAt: 10, measured: true, consumption: 80, debtFloor: 5000}
	}

	m := fresh()
	look(m, 60, 20)
	if !m.measured || m.last != 120 {
		t.Fatalf("an honest interval: spent %v tokens/s (measured=%v), want 120", m.last, m.measured)
	}

	// The balance rose by more than the hypervisor added.
	m = fresh()
	look(m, 400, 20)
	if m.measured || m.consumption != 0 {
		t.Fatalf("a balance that rose on its own was measured: %v tokens/s", m.consumption)
	}
	// Writing it down again afterwards is not spending.
	for i := range meterQuarantineIntervals - 1 {
		look(m, -4000-float64(i), 0)
		if m.measured {
			t.Fatalf("look %d after an implausible balance was measured (%v tokens/s)", i, m.consumption)
		}
	}
	// Left alone, the bucket is measured again.
	level := m.level
	look(m, level+10, 50)
	if !m.measured || m.last != 80 {
		t.Fatalf("after the quarantine: spent %v tokens/s (measured=%v), want 80", m.last, m.measured)
	}

	// More owed than the overdraft bound allows.
	m = fresh()
	look(m, -5001, 0)
	if m.measured {
		t.Fatalf("a balance below the overdraft bound was measured: %v tokens/s", m.consumption)
	}
	// Exactly at the bound is what a limiter can charge.
	m = fresh()
	look(m, -5000, 0)
	if !m.measured || m.last != 10200 {
		t.Fatalf("an overdraft to the bound: spent %v tokens/s (measured=%v), want 10200", m.last, m.measured)
	}

	for name, tokens := range map[string]float64{"NaN": math.NaN(), "+Inf": math.Inf(1), "-Inf": math.Inf(-1)} {
		m = fresh()
		look(m, tokens, 0)
		look(m, 50, 0)
		look(m, 40, 0)
		if m.measured {
			t.Fatalf("%s: measured %v tokens/s right after a balance that is not a number", name, m.consumption)
		}
	}

	// A limiter that found nobody refilling added tokens itself, at the
	// published rate: no figure for that interval, and none is held against it.
	m = fresh()
	m.foreignRate = 400
	look(m, 250, 0)
	if m.measured {
		t.Fatal("an interval a limiter refilled in was measured")
	}
	look(m, 200, 0)
	if !m.measured || m.last != 100 {
		t.Fatalf("after a limiter's own refill: spent %v tokens/s (measured=%v), want 100", m.last, m.measured)
	}
	// More than that rate can have added is somebody writing the balance.
	m = fresh()
	m.foreignRate = 400
	look(m, 5000, 0)
	look(m, 4900, 0)
	if m.measured {
		t.Fatal("a balance far above what a limiter's own refill adds was taken for one")
	}
}

// The overdraft bound is the cost of a minute of work with the whole device,
// and never less than a bucket.
func TestDebtBound(t *testing.T) {
	for _, tc := range []struct {
		capacity, rate float64
		upLimit        uint32
		want           float64
	}{
		{capacity: 300, rate: 3000, upLimit: 30, want: 600000},
		{capacity: 100, rate: 1000, upLimit: 10, want: 600000},
		{capacity: 500, rate: 5000, upLimit: 100, want: 300000},
		{capacity: 20, rate: 0.01, upLimit: 50, want: 20},
		{capacity: 20, rate: 100, upLimit: 0, want: 600000}, // a limit of 0 counts as 1%
	} {
		if got := debtBound(tc.capacity, tc.rate, tc.upLimit); math.Abs(got-tc.want) > 1e-6*tc.want {
			t.Errorf("debtBound(%v, %v, %d) = %v, want %v", tc.capacity, tc.rate, tc.upLimit, got, tc.want)
		}
	}
}

// The hypervisor is held up for 150 ms every second (a long garbage collection,
// a starved node). A limiter that finds nobody refilling refills itself, so the
// worker's spending in that interval is not known. That must not be billed to
// its neighbor: before, the neighbor was taken for the only one running, and a
// 50% worker was held at 25%. Here the neighbor's limiter is one that waits for
// the hypervisor instead, so that its own spending stays known throughout.
func TestHypervisorStallsAreNotBilledToNeighbors(t *testing.T) {
	got := worst(func(seed uint64) []float64 {
		large, small := smallLaunchWorker("large", 50), smallLaunchWorker("small", 25)
		small.selfRefill = true
		sim := newPacingSim(t, large, small)
		sim.resolution = 2 * time.Millisecond
		sim.run(30 * time.Second)
		before := len(large.history)
		// The stall begins at another moment of the second for every seed, so
		// that it covers a control interval's end for some and not for others.
		offset := time.Duration(seed*47%850) * time.Millisecond
		for range 30 {
			sim.run(offset)
			sim.hypervisorDown = true
			sim.run(150 * time.Millisecond)
			sim.hypervisorDown = false
			sim.run(850*time.Millisecond - offset)
		}
		intervals := len(large.history) - before
		return []float64{math.Abs(large.meanUtil(intervals) - 0.50), math.Abs(small.meanUtil(intervals) - 0.25)}
	})
	t.Logf("worst seeds: the 50%% worker was %.1f points from its limit (seed %d), the 25%% worker %.1f (seed %d)",
		got[0].value*100, got[0].seed, got[1].value*100, got[1].seed)
	// A limiter that waits loses what its bucket cannot hold of a stall: the
	// bucket holds a tenth of a second of refill, the stall lasts 150 ms.
	for i, w := range []string{"50%", "25%"} {
		if got[i].value > []float64{0.07, 0.05}[i] {
			t.Errorf("seed %d: the %s worker was %.3f from its limit while the hypervisor kept stalling",
				got[i].seed, w, got[i].value)
		}
	}
}

// A workload that uses the device without going through its limiter spends no
// tokens. Its bucket looks like that of an idle worker, and what it runs was
// billed to the worker next to it that does pay: an honest 30% worker fell to
// about 1% of the device. Lowering the honest worker's rate does not lower
// that load, which is how it is told apart; the rate goes back to where it
// was. The honest worker keeps its allocation except for the seconds it takes
// to tell, less often each time.
func TestWorkerThatDoesNotPayIsNotBilledToItsNeighbor(t *testing.T) {
	for _, unpaid := range []float64{0.30, 0.60} {
		t.Run("unpaid load of "+strconv.Itoa(int(unpaid*100))+"%", func(t *testing.T) {
			const settle, during = 160, 440
			got := worst(func(seed uint64) []float64 {
				honest := &simWorker{uid: "honest", upLimit: 30, demand: 1}
				rider := &simWorker{uid: "rider", upLimit: 60, unpaid: true}
				sim := newClosedLoopSim(t, honest, rider)
				realisticTelemetry(sim, seedRNG(seed))
				sim.run(settle)
				rate := sim.refillRate(honest)
				rider.demand = unpaid
				sim.run(during)
				lowest := 1.0
				for i := settle; i+40 <= settle+during; i++ {
					lowest = math.Min(lowest, honest.meanOver(i, i+40))
				}
				return []float64{
					0.30 - honest.meanOver(settle, settle+during),
					0.30 - lowest,
					rate / sim.refillRate(honest),
				}
			})
			mean, lowest, rate := got[0], got[1], got[2]
			t.Logf("worst seeds: the honest worker's mean over 220 s was %.1f%% (seed %d), its lowest twenty-second mean "+
				"%.1f%% (seed %d), and its rate ended at 1/%.2f of what it was (seed %d)",
				(0.30-mean.value)*100, mean.seed, (0.30-lowest.value)*100, lowest.seed, rate.value, rate.seed)
			if mean.value > 0.04 {
				t.Errorf("seed %d: the honest worker got %.3f of the device on average, its share is 0.30",
					mean.seed, 0.30-mean.value)
			}
			if lowest.value > 0.15 {
				t.Errorf("seed %d: the honest worker was down to %.3f of the device for twenty seconds",
					lowest.seed, 0.30-lowest.value)
			}
		})
	}
}

// What a workload writes into its own bucket must not move its neighbor's
// rate. A 40% worker runs next to one whose balance is overwritten in every
// control interval; before, the neighbor fell to 0.1% of the device for most of
// these, and ran unthrottled for a balance of -1e300.
func TestWorkerWritingItsOwnBalanceDoesNotMoveItsNeighbor(t *testing.T) {
	type balance func(interval int, capacity, bound float64) float64
	constant := func(v float64) balance { return func(int, float64, float64) float64 { return v } }
	for _, tc := range []struct {
		name  string
		write balance
	}{
		{name: "not a number", write: constant(math.NaN())},
		{name: "positive infinity", write: constant(math.Inf(1))},
		{name: "negative infinity", write: constant(math.Inf(-1))},
		{name: "huge positive", write: constant(1e12)},
		{name: "minus 1e300", write: constant(-1e300)},
		{name: "topped up to the capacity", write: func(_ int, capacity, _ float64) float64 { return capacity }},
		{name: "flapping between huge values", write: func(i int, _, _ float64) float64 {
			return []float64{1e9, -1e9}[i%2]
		}},
		{name: "flapping within what a limiter can produce", write: func(i int, capacity, bound float64) float64 {
			return []float64{capacity, -bound / 2}[i%2]
		}},
		{name: "flapping slowly within what a limiter can produce", write: func(i int, capacity, bound float64) float64 {
			if i%30 >= 2 {
				return math.NaN() // left alone
			}
			return []float64{-bound / 2, capacity}[i%30]
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const settle, during = 100, 140
			got := worst(func(seed uint64) []float64 {
				neighbor := &simWorker{uid: "neighbor", upLimit: 40, demand: 1}
				// Unthrottled, the writer would use half of the device: what the
				// neighbor loses is then the control loop's doing, not a full device.
				writer := &simWorker{uid: simWriterUID, upLimit: 30, demand: 0.5}
				sim := newClosedLoopSim(t, neighbor, writer)
				realisticTelemetry(sim, seedRNG(seed))
				sim.run(settle)
				rate := sim.refillRate(neighbor)
				lowest, highest := rate, rate
				for i := range during {
					info := writer.deviceInfo(t)
					es := sim.controller.erlStates["writer:gpu-sim"]
					value := tc.write(i, info.GetERLTokenCapacity(), es.debtBound)
					if !(math.IsNaN(value) && tc.name != "not a number") {
						info.SetERLCurrentTokens(value)
					}
					sim.step()
					lowest, highest = math.Min(lowest, sim.refillRate(neighbor)), math.Max(highest, sim.refillRate(neighbor))
				}
				return []float64{
					math.Abs(neighbor.meanOver(settle, settle+during) - 0.40),
					math.Max(rate/lowest, highest/rate),
				}
			})
			off, moved := got[0], got[1]
			t.Logf("worst seeds: the neighbor's mean was %.1f points from its limit (seed %d), "+
				"its rate moved by a factor of %.2f (seed %d)",
				off.value*100, off.seed, moved.value, moved.seed)
			if off.value > 0.04 {
				t.Errorf("seed %d: the neighbor was %.3f from its limit", off.seed, off.value)
			}
			if moved.value > 1.15 {
				t.Errorf("seed %d: the neighbor's rate moved by a factor of %.2f", moved.seed, moved.value)
			}
		})
	}
}

// longLaunchWorker is a workload whose every launch keeps the device busy for
// launchTime and costs the tokens for that much work; the seed varies the
// length by a few percent.
func longLaunchWorker(share uint32, launchTime time.Duration, seed uint64) *pacedWorker {
	launchTime = time.Duration(float64(launchTime) * (0.95 + 0.005*float64(seed)))
	return &pacedWorker{
		uid: "graph", upLimit: share, launchCost: launchTime.Seconds() / simUtilPerToken, launchTime: launchTime,
	}
}

// A launch that keeps the device busy for many seconds is charged in full, and
// its worker waits until refills have repaid it. The debt must not be forgiven
// because it is large, nor because the control loop lowers the rate while it is
// outstanding: before, the bound on debt was a minute of the current rate, and
// launches of 30 s ran back to back with a limit of 10%.
func TestDebtIsNotForgivenWhileItIsOutstanding(t *testing.T) {
	for _, tc := range []struct {
		share      uint32
		launchTime time.Duration
	}{
		{share: 10, launchTime: 30 * time.Second},
		{share: 30, launchTime: 30 * time.Second},
		{share: 5, launchTime: 10 * time.Second},
	} {
		t.Run(strconv.Itoa(int(tc.share))+"% share, launches of "+tc.launchTime.String(), func(t *testing.T) {
			limit := float64(tc.share) / 100
			// Two launches with their waits to settle, three to measure.
			cycle := time.Duration(float64(tc.launchTime) / limit)
			got := worst(func(seed uint64) []float64 {
				worker := longLaunchWorker(tc.share, tc.launchTime, seed)
				sim := newPacingSim(t, worker)
				sim.resolution = 100 * time.Millisecond
				sim.run(2 * cycle)
				before := len(worker.history)
				sim.run(3 * cycle)
				return []float64{worker.meanUtil(len(worker.history) - before)}
			})[0]
			t.Logf("worst seed %d: long-run utilization %.1f%% with a limit of %d%%", got.seed, got.value*100, tc.share)
			if got.value > limit*1.15+0.01 {
				t.Errorf("seed %d: the worker ran at %.3f of the device with a limit of %.2f", got.seed, got.value, limit)
			}
		})
	}
}

// The bound on debt never shrinks while debt is outstanding, and a balance
// beyond it is brought up to it, no further.
func TestDebtBoundDoesNotShrinkWhileDebtIsOutstanding(t *testing.T) {
	worker := smallLaunchWorker("worker", 25)
	sim := newPacingSim(t, worker)
	sim.run(60 * time.Second)
	info := worker.info(t)
	es := sim.controller.erlStates["worker:gpu-sim"]
	bound := debtBound(info.GetERLTokenCapacity(), info.GetERLTokenRefillRate(), 25)
	if math.Abs(es.debtBound-bound) > 1e-6*bound {
		t.Fatalf("bound in force = %.0f, want %.0f", es.debtBound, bound)
	}

	// The worker runs up debt within the bound and stops; the control loop then
	// has the rate much lower, as it does when it takes the debt's work for
	// more than the worker's share.
	worker.paused = true
	info.SetERLCurrentTokens(-0.9 * bound)
	es.currentRate /= 50
	es.holdIntervals = 10
	sim.telemetryDown = true
	sim.run(2 * time.Second)
	if got := info.GetERLCurrentTokens(); got > -0.89*bound || got < -0.9*bound {
		t.Errorf("debt of %.0f became %.0f after the rate was lowered: it was forgiven", -0.9*bound, got)
	}
	if es.debtBound < bound*(1-1e-9) {
		t.Errorf("the bound shrank from %.0f to %.0f while debt was outstanding", bound, es.debtBound)
	}

	// A balance beyond the bound is not debt a limiter charged.
	info.SetERLCurrentTokens(-1e300)
	sim.run(erlUpdateInterval + 2*pacedResolution)
	if got := info.GetERLCurrentTokens(); got < -es.debtBound || got > -0.99*es.debtBound {
		t.Errorf("a balance of -1e300 was brought to %.0f, want the bound of %.0f", got, -es.debtBound)
	}
}

// The limiter of the previous release is what every workload runs while a node
// is being upgraded: the hypervisor is replaced first, the limiter library
// follows as pods restart. It does not carry debt in the bucket: it admits a
// launch that has waited a second, whatever it costs. Work that costs less than
// a bucket is held at its share as before. Work in launches that cost more is
// not, by that limiter's design: it runs at launch time / (launch time + 1 s),
// whatever the rate. The control loop must not answer that by lowering the rate
// to the floor, where everything else the worker does would be starved for as
// long as it takes to climb back. (Launches that keep the device full on
// average, of about five seconds and more, leave nothing to tell that by: a
// full device shows no less for a lower rate either. They are reported here
// without a bound on the rate.)
func TestControlLoopWithTheLimiterOfThePreviousRelease(t *testing.T) {
	legacyWorker := func(share uint32) *pacedWorker {
		return &pacedWorker{uid: "worker", upLimit: share, legacy: true, launchCost: 100, launchTime: 10 * time.Millisecond}
	}

	t.Run("work that costs less than a bucket is held at its share", func(t *testing.T) {
		got := worst(func(seed uint64) []float64 {
			share := uint32(15 + 3*seed)
			worker := legacyWorker(share)
			sim := newPacingSim(t, worker)
			sim.resolution = 10 * time.Millisecond
			sim.run(90 * time.Second)
			return []float64{math.Abs(worker.meanUtil(60) - float64(share)/100)}
		})[0]
		t.Logf("worst seed %d: %.1f points from its limit", got.seed, got.value*100)
		if got.value > 0.05 {
			t.Errorf("seed %d: a worker with the previous limiter was %.3f from its limit", got.seed, got.value)
		}
	})

	for _, launchTime := range []time.Duration{500 * time.Millisecond, 2 * time.Second, 5 * time.Second} {
		t.Run("launches of "+launchTime.String(), func(t *testing.T) {
			got := worst(func(seed uint64) []float64 {
				worker := legacyWorker(25)
				sim := newPacingSim(t, worker)
				sim.resolution = 10 * time.Millisecond
				sim.run(60 * time.Second)
				rate := worker.info(t).GetERLTokenRefillRate()

				// The work turns into long launches for two minutes.
				long := longLaunchWorker(25, launchTime, seed)
				worker.launchCost, worker.launchTime = long.launchCost, long.launchTime
				before := len(worker.history)
				lowest := rate
				for range 240 {
					sim.run(erlUpdateInterval)
					lowest = math.Min(lowest, worker.info(t).GetERLTokenRefillRate())
				}
				ran := worker.meanUtil(len(worker.history) - before)
				expected := launchTime.Seconds() / (launchTime.Seconds() + legacyAdmitAfter.Seconds())

				// And back: how long until the worker is at its share again.
				worker.launchCost, worker.launchTime, worker.legacyDebt = 100, 10*time.Millisecond, 0
				back := math.Inf(1)
				for elapsed := 0.0; elapsed < 180; elapsed += 2 {
					sim.run(2 * time.Second)
					if math.Abs(worker.meanUtil(4)-0.25) <= 0.05 {
						back = elapsed
						break
					}
				}
				return []float64{rate / lowest, math.Abs(ran - expected), back}
			})
			lowest, ran, after := got[0], got[1], got[2]
			t.Logf("worst seeds: the rate went down to 1/%.1f of what it was (seed %d); the long launches ran %.1f points "+
				"from launch time / (launch time + 1 s) (seed %d); the worker was back at its share %.0f s after them (seed %d)",
				lowest.value, lowest.seed, ran.value*100, ran.seed, after.value, after.seed)
			if ran.value > 0.05 {
				t.Errorf("seed %d: the long launches ran %.3f from what the previous limiter admits", ran.seed, ran.value)
			}
			if launchTime >= 5*time.Second {
				return
			}
			if lowest.value > 1/futilityDepth*1.01 {
				t.Errorf("seed %d: the rate was lowered to 1/%.0f of what it was", lowest.seed, lowest.value)
			}
			if after.value > 10 {
				t.Errorf("seed %d: the worker was back at its share %.0f s after the long launches", after.seed, after.value)
			}
		})
	}
}

// A bucket that keeps yielding no figure is left out of the split of its
// device after meterExclusionIntervals looks, whether the balance is written
// at every look or only once in a while, and is taken back once it has been
// readable for meterReadmissionIntervals looks in a row.
func TestTokenMeterLeavesOutABucketThatKeepsYieldingNoFigure(t *testing.T) {
	fresh := func() *tokenMeter {
		return &tokenMeter{level: 100, updatedAt: 10, measured: true, consumption: 80, debtFloor: 5000}
	}
	// honest is a look at a bucket that spent 40 tokens/s; written one at a
	// balance that rose by more than was added.
	honest := func(m *tokenMeter) { m.observe(m.level-20+50, 50, m.updatedAt+0.5) }
	written := func(m *tokenMeter) { m.observe(m.level+1000, 0, m.updatedAt+0.5) }

	m := fresh()
	for look := 1; look <= meterExclusionIntervals; look++ {
		if m.excluded {
			t.Fatalf("left out after %d looks, want %d", look-1, meterExclusionIntervals)
		}
		written(m)
	}
	if !m.excluded {
		t.Fatalf("not left out after %d looks at a balance no limiter produces", meterExclusionIntervals)
	}
	// Left alone, it is measured again after the quarantine, and taken back
	// after meterReadmissionIntervals such looks, not before.
	for look := 1; look < meterQuarantineIntervals+meterReadmissionIntervals-1; look++ {
		honest(m)
		if !m.excluded {
			t.Fatalf("taken back %d looks after the last written balance", look)
		}
	}
	honest(m)
	if m.excluded || m.unaccounted != 0 || !m.measured {
		t.Fatalf("not taken back after the quarantine and %d readable looks: %+v", meterReadmissionIntervals, *m)
	}

	// One stray balance costs a quarantine and is forgotten.
	m = fresh()
	written(m)
	for range meterQuarantineIntervals + meterReadmissionIntervals {
		honest(m)
	}
	if m.excluded || m.unaccounted != 0 {
		t.Fatalf("one stray balance left %d looks on the count (excluded=%v)", m.unaccounted, m.excluded)
	}

	// A balance written once per quarantine and a bit, so that a few looks in
	// between are measured, adds up all the same.
	m = fresh()
	looks := 0
	for ; !m.excluded && looks < 4*meterExclusionIntervals; looks++ {
		if looks%(meterQuarantineIntervals+meterReadmissionIntervals/2) == 0 {
			written(m)
		} else {
			honest(m)
		}
	}
	if !m.excluded || looks > 2*meterExclusionIntervals {
		t.Fatalf("a balance written every %d looks: left out=%v after %d looks, want within %d",
			meterQuarantineIntervals+meterReadmissionIntervals/2, m.excluded, looks, 2*meterExclusionIntervals)
	}

	// A bucket somebody else keeps refilling yields no figure either.
	m = fresh()
	for range meterExclusionIntervals {
		m.foreignRate = 400
		m.observe(m.level, 0, m.updatedAt+0.5)
	}
	if !m.excluded {
		t.Fatal("a bucket refilled by somebody else at every look was not left out")
	}
	// A hypervisor that is held up now and then makes a limiter refill itself
	// now and then, for as long as the node is busy: with readable runs in
	// between that is not held against the bucket for ever.
	m = fresh()
	for look := range 10 * meterExclusionIntervals {
		if look%(4*meterReadmissionIntervals) == 0 {
			m.foreignRate = 400
			m.observe(m.level, 0, m.updatedAt+0.5)
		} else {
			honest(m)
		}
	}
	if m.excluded {
		t.Fatal("a bucket its limiter refilled once in forty seconds was left out")
	}

	// A first look and a look after a gap say nothing either way.
	m = &tokenMeter{}
	m.observe(100, 0, 10)
	m.observe(80, 0, 10+2*maxTokenUpdateGap)
	if m.unaccounted != 0 || m.accounted != 0 {
		t.Fatalf("a first look and a gap were counted: %+v", *m)
	}
}

// Next to a worker that is left out of the split, nobody is billed for what
// the device runs: the others are measured by what they spent themselves, the
// one left out and one without a calibration are held.
func TestAttributionNextToAWorkerThatIsLeftOut(t *testing.T) {
	calibrated := func() *utilizationCalibration {
		// A token per second buys 1e-4 of the device.
		return &utilizationCalibration{utilization: 0.3, consumption: 3000, samples: 50, evidence: 15}
	}
	build := func() (honest, leftOutWorker, newcomer *erlMember) {
		honest = &erlMember{workerUID: "honest", fallback: calibrated(),
			meter: &tokenMeter{consumption: 2000, window: 2000, measured: true}}
		leftOutWorker = &erlMember{workerUID: "left-out", fallback: calibrated(),
			meter: &tokenMeter{consumption: 500, window: 500, measured: true, excluded: true}}
		newcomer = &erlMember{workerUID: "newcomer", fallback: &utilizationCalibration{},
			meter: &tokenMeter{consumption: 1000, window: 1000, measured: true}}
		return honest, leftOutWorker, newcomer
	}

	for _, deviceUtil := range []float64{0.2, 0.6, 1.0} {
		honest, leftOutWorker, newcomer := build()
		attributeDeviceUtilization([]*erlMember{honest, leftOutWorker, newcomer}, deviceUtil, true, nil, false, 0)
		// 2000 tokens/s at 1e-4 each, whatever the device shows.
		if honest.source != utilizationFromSpending || !honest.calibrated || math.Abs(honest.utilization-0.2) > 1e-9 ||
			math.Abs(honest.gain-1e-4) > 1e-12 || honest.deviceRead {
			t.Errorf("device at %.0f%%: the honest worker got %.3f from %q (calibrated=%v, gain=%g, deviceRead=%v), "+
				"want 0.200 from %q", deviceUtil*100, honest.utilization, honest.source, honest.calibrated, honest.gain,
				honest.deviceRead, utilizationFromSpending)
		}
		for _, held := range []*erlMember{leftOutWorker, newcomer} {
			if held.source != utilizationUnknown {
				t.Errorf("device at %.0f%%: %s got %.3f from %q, want its rate held",
					deviceUtil*100, held.workerUID, held.utilization, held.source)
			}
		}
	}

	// Before the worker is left out, the split is unknown for everybody.
	honest, writer, _ := build()
	writer.meter.excluded, writer.meter.measured = false, false
	attributeDeviceUtilization([]*erlMember{honest, writer}, 0.6, true, nil, false, 0)
	if honest.source != utilizationUnknown {
		t.Errorf("next to a bucket that is not measured the honest worker got %.3f from %q",
			honest.utilization, honest.source)
	}
	// Other load no bucket accounts for, a missing device sample or another
	// bucket that is not measured in this interval: held, as without anybody
	// left out.
	for name, attribute := range map[string]func(members []*erlMember){
		"unmetered load": func(members []*erlMember) { attributeDeviceUtilization(members, 0.6, true, nil, true, 0) },
		"no sample":      func(members []*erlMember) { attributeDeviceUtilization(members, 0, false, nil, false, 0) },
		"new bucket": func(members []*erlMember) {
			members = append(members, &erlMember{workerUID: "new", meter: &tokenMeter{}})
			attributeDeviceUtilization(members, 0.6, true, nil, false, 0)
		},
	} {
		honest, leftOutWorker, _ := build()
		attribute([]*erlMember{honest, leftOutWorker})
		if honest.source != utilizationUnknown {
			t.Errorf("%s: the honest worker got %.3f from %q", name, honest.utilization, honest.source)
		}
	}
	// A worker that has the device to itself is read from the device as before.
	_, alone, _ := build()
	attributeDeviceUtilization([]*erlMember{alone}, 0.6, true, nil, false, 0)
	if alone.source != utilizationFromDevice {
		t.Errorf("a worker alone on its device got its utilization from %q", alone.source)
	}
}

type recordedPodEvent struct {
	namespace, podName, podUID, eventType, reason, message string
}

// eventBackend is a backend that can record Events on Pods.
type eventBackend struct {
	framework.Backend
	events chan recordedPodEvent
}

func (b *eventBackend) EmitPodEvent(
	_ context.Context, namespace, podName, podUID, eventType, reason, message string,
) error {
	b.events <- recordedPodEvent{namespace, podName, podUID, eventType, reason, message}
	return nil
}

// The Event about a bucket that is left out goes to the Pod through the
// backend the controller was created with, as a Warning with a fixed text, and
// once per Pod in meterExclusionEventInterval however long it lasts.
func TestLeftOutBucketRaisesOneEventPerPodAndInterval(t *testing.T) {
	backend := &eventBackend{events: make(chan recordedPodEvent, 8)}
	controller := NewQuotaController(&quotaMetricsDeviceController{}, backend).(*Controller)
	now := time.Unix(1_800_000_000, 0)
	controller.nowFunc = func() time.Time { return now }
	pod := &WorkerInfoSnapshot{Namespace: "team-a", WorkerName: "pod-0"}
	member := func(uid string, upLimit uint32, excluded bool) *erlMember {
		return &erlMember{workerUID: uid, stateKey: uid + ":gpu", device: DeviceSnapshot{UpLimit: upLimit},
			meter: &tokenMeter{excluded: excluded}}
	}
	next := func() recordedPodEvent {
		t.Helper()
		select {
		case event := <-backend.events:
			return event
		case <-time.After(10 * time.Second):
			t.Fatal("no Event was recorded")
			return recordedPodEvent{}
		}
	}

	controller.reportMeterExclusion(member("uid-0", 30, true), pod, false)
	want := recordedPodEvent{"team-a", "pod-0", "uid-0", "Warning", "LimiterBalanceUnaccounted",
		"The token balance of the GPU limiter of this Pod could not be accounted for during 60 s: the tokens it " +
			"spends are not used to measure its GPU usage, its refill rate is held, and its compute limit may not be " +
			"fully enforced. This ends once the balance has been consistent for 10 s."}
	if got := next(); got != want {
		t.Fatalf("Event = %+v, want %+v", got, want)
	}

	// Every interval it stays left out, on any of its devices: no further Event.
	for range 3 * meterExclusionIntervals {
		now = now.Add(erlUpdateInterval)
		controller.reportMeterExclusion(member("uid-0", 30, true), pod, true)
		controller.reportMeterExclusion(&erlMember{workerUID: "uid-0", stateKey: "uid-0:gpu-1",
			device: DeviceSnapshot{UpLimit: 30}, meter: &tokenMeter{excluded: true}}, pod, true)
	}
	// Workers that are not left out, or have no limit, get none at all.
	controller.reportMeterExclusion(member("uid-1", 30, false), pod, true)
	controller.reportMeterExclusion(member("uid-2", 100, true), pod, false)
	// Another Pod gets its own, and the first one again after the interval.
	other := &WorkerInfoSnapshot{Namespace: "team-b", WorkerName: "pod-3"}
	controller.reportMeterExclusion(member("uid-3", 30, true), other, false)
	if got := next(); got.podUID != "uid-3" || got.namespace != "team-b" || got.podName != "pod-3" {
		t.Fatalf("Event = %+v, want one for uid-3", got)
	}
	now = now.Add(meterExclusionEventInterval)
	controller.reportMeterExclusion(member("uid-0", 30, true), pod, true)
	if got := next(); got != want {
		t.Fatalf("Event after the interval = %+v, want %+v", got, want)
	}
	// A worker that is gone is not told, and nothing is kept for it.
	controller.CleanupWorker("uid-0")
	controller.reportMeterExclusion(member("uid-0", 30, true), pod, true)
	if _, kept := controller.exclusionEvents["uid-0"]; kept {
		t.Error("the Event time of a removed worker is kept")
	}
	select {
	case event := <-backend.events:
		t.Fatalf("an Event nobody should get was recorded: %+v", event)
	case <-time.After(200 * time.Millisecond):
	}

	// Without a backend that records Events the controller carries on without.
	plain := NewQuotaController(&quotaMetricsDeviceController{}, nil).(*Controller)
	if plain.podEventFn != nil {
		t.Fatal("a controller without a backend has somewhere to record Events")
	}
	plain.reportMeterExclusion(member("uid-0", 30, true), pod, false)
}

// A worker that keeps writing its own balance must not hold its neighbor's
// rate for as long as it likes. A 40% worker runs next to one whose balance is
// overwritten for minutes. For the first minute (two when the balance is only
// written once in a while) the neighbor's rate is held, as in
// TestWorkerWritingItsOwnBalanceDoesNotMoveItsNeighbor. Then the writer is left
// out of the split, its Pod is told once, and the neighbor is under control
// again: it stays at its limit, and when its limit is halved it is at the new
// one within five seconds, where a held rate would have left it at twice its
// limit. The writer's own rate is held throughout and its bucket refilled.
// When it stops writing, it is back in the split within twenty-one seconds.
func TestWorkerThatKeepsWritingItsBalanceIsLeftOutOfTheSplit(t *testing.T) {
	const (
		settle = 100 // 50 s
		// How long the neighbor is watched once the writer is left out, before
		// and after its limit changes, and how long it has to follow the change.
		watch  = 60 // 30 s
		follow = 10 // 5 s
		// Once the writer has stopped: the quarantine, the readable looks, and a
		// look or two of slack for where in its pattern it stopped.
		readmitted = meterQuarantineIntervals + meterReadmissionIntervals + 2
		normal     = 60
	)
	for _, tc := range []struct {
		name string
		// write returns the balance to store before interval i, and whether to.
		write func(i int) (float64, bool)
		// excludedWithin is the number of intervals after which the writer has
		// to be left out.
		excludedWithin int
	}{
		{name: "a huge balance in every interval", excludedWithin: meterExclusionIntervals,
			write: func(int) (float64, bool) { return 1e12, true }},
		{name: "a huge balance once in thirty intervals", excludedWithin: 2 * meterExclusionIntervals,
			write: func(i int) (float64, bool) { return 1e12, i%30 == 0 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := worst(func(seed uint64) []float64 {
				neighbor := &simWorker{uid: "neighbor", upLimit: 40, demand: 1}
				writer := &simWorker{uid: simWriterUID, upLimit: 30, demand: 0.5}
				sim := newClosedLoopSim(t, neighbor, writer)
				realisticTelemetry(sim, seedRNG(seed))
				var events []recordedPodEvent
				sim.controller.podEventFn = func(namespace, podName, podUID, reason, message string) {
					events = append(events, recordedPodEvent{namespace, podName, podUID, "", reason, message})
				}
				sim.run(settle)
				meter := sim.controller.tokenMeters["writer:gpu-sim"]
				interval := 0
				step := func(writing bool) {
					if value, write := tc.write(interval); writing && write {
						writer.deviceInfo(t).SetERLCurrentTokens(value)
					}
					interval++
					sim.step()
				}

				// Until the writer is left out, the neighbor's rate is held.
				rate := sim.refillRate(neighbor)
				lowest, highest := rate, rate
				for !meter.excluded && interval < 4*meterExclusionIntervals {
					step(true)
					if !meter.excluded {
						lowest, highest = math.Min(lowest, sim.refillRate(neighbor)), math.Max(highest, sim.refillRate(neighbor))
					}
				}
				excludedAfter := interval
				heldWithin := math.Max(rate/lowest, highest/rate)

				// Left out: the neighbor stays at its limit, under control.
				writerRate := sim.refillRate(writer)
				from := len(neighbor.history)
				uncontrolled := 0.0
				for range watch {
					step(true)
					if sim.controller.utilSources["neighbor:gpu-sim"] == utilizationUnknown {
						uncontrolled++
					}
				}
				atLimit := math.Abs(neighbor.meanOver(from, from+watch) - 0.40)

				// The neighbor's limit is halved, and it follows.
				neighbor.upLimit = 20
				from = len(neighbor.history)
				for range watch {
					step(true)
				}
				atNewLimit := math.Abs(neighbor.meanOver(from+follow, from+watch) - 0.20)
				writerMoved := math.Abs(sim.refillRate(writer)/writerRate - 1)
				writerRan := writer.meanOver(from, from+watch)

				// The writer stops: it is taken back, and both are accounted for
				// as any two workers are.
				stopped := interval
				for meter.excluded && interval < stopped+4*readmitted {
					step(false)
				}
				readmittedAfter := interval - stopped
				for range normal {
					step(false)
				}
				end := len(neighbor.history)
				notSplit := 0.0
				for _, key := range []string{"neighbor:gpu-sim", "writer:gpu-sim"} {
					if sim.controller.utilSources[key] != utilizationFromTokenShare {
						notSplit++
					}
				}

				wrongEvents := math.Abs(float64(len(events) - 1))
				for _, event := range events {
					if event.podUID != simWriterUID || event.namespace != "sim" || event.podName != simWriterUID ||
						event.reason != meterExclusionEventReason || event.message != meterExclusionEventMessage {
						wrongEvents++
					}
				}
				return []float64{
					float64(excludedAfter), heldWithin, uncontrolled, atLimit, atNewLimit, writerMoved, -writerRan,
					float64(readmittedAfter), notSplit,
					math.Abs(neighbor.meanOver(end-normal/2, end) - 0.20), math.Abs(writer.meanOver(end-normal/2, end) - 0.30),
					wrongEvents,
				}
			})
			t.Logf("worst seeds: left out after %.0f intervals; until then the neighbor's rate moved by a factor of %.2f; "+
				"then it was %.1f points from its limit, and %.1f points from the new one from %d intervals after the "+
				"change; the writer was taken back %.0f intervals after it stopped; at the end the neighbor was %.1f and "+
				"the writer %.1f points from their limits",
				got[0].value, got[1].value, got[3].value*100, got[4].value*100, follow, got[7].value,
				got[9].value*100, got[10].value*100)
			for i, check := range []struct {
				what string
				most float64
			}{
				{"intervals until the writer was left out", float64(tc.excludedWithin)},
				{"factor by which the neighbor's rate moved before that", 1.15},
				{"intervals in which the neighbor's rate was held after that", 0},
				{"distance of the neighbor from its limit after that", 0.04},
				{"distance of the neighbor from its new limit five seconds after the change", 0.04},
				{"relative change of the writer's rate while it was left out", 1e-9},
				{"(negated) share of the device the writer ran on its held rate", -0.15},
				{"intervals until the writer was taken back", readmitted},
				{"workers not measured by the token split at the end", 0},
				{"distance of the neighbor from its limit at the end", 0.04},
				{"distance of the writer from its limit at the end", 0.05},
				{"Events other than the one for the writer's Pod", 0},
			} {
				if got[i].value > check.most {
					t.Errorf("seed %d: %s: %.4g, want at most %.4g", got[i].seed, check.what, got[i].value, check.most)
				}
			}
		})
	}
}

// starvedHypervisor holds the hypervisor up the way a node without CPU to
// spare does: once every two to five seconds for 150 to 300 ms, either the
// whole process or the refill loop alone.
type starvedHypervisor struct {
	rng         *rand.Rand
	from, until time.Time
	process     bool
	stalls      int
}

// run advances the simulation, calling each before every control interval.
func (s *starvedHypervisor) run(sim *pacingSim, duration time.Duration, each func()) {
	for end, next := sim.now.Add(duration), sim.now; sim.now.Before(end); {
		if !sim.now.Before(next) {
			each()
			next = next.Add(erlUpdateInterval)
		}
		if !sim.now.Before(s.until) {
			s.from = sim.now.Add(2*time.Second + time.Duration(s.rng.Int64N(int64(3*time.Second))))
			s.until = s.from.Add(150*time.Millisecond + time.Duration(s.rng.Int64N(int64(150*time.Millisecond))))
			s.process = s.rng.IntN(2) == 0
			s.stalls++
		}
		held := !sim.now.Before(s.from)
		sim.hypervisorDown, sim.refillStalled = held && s.process, held && !s.process
		sim.run(erlRefillInterval)
	}
	sim.hypervisorDown, sim.refillStalled = false, false
}

// On a node without CPU to spare the refill loop is held up for longer than a
// limiter waits before it refills its bucket itself, every few seconds, for as
// long as the node stays busy. Those intervals yield no figure, for every
// worker at once, and that is all: before, each of them counted towards
// leaving the worker out of the split, the count only started over after ten
// seconds without one, and after 120 of them every worker of the node was left
// out, with an Event on its Pod, its rate held and no way back while the
// stalls went on.
//
// Thirty minutes of such stalls: nobody is left out, no Event is raised, and a
// limit that is changed after twenty minutes is followed. A worker that
// rewrites its own balance in every interval of the same thirty minutes is
// left out as fast as without the stalls, stays out while it goes on, and is
// taken back once it stops although the stalls continue.
func TestStarvedRefillLoopDoesNotLeaveLimitersOutOfTheSplit(t *testing.T) {
	newSim := func(t *testing.T, workers ...*pacedWorker) (*pacingSim, *[]recordedPodEvent) {
		for _, w := range workers {
			w.selfRefill = true
		}
		sim := newPacingSim(t, workers...)
		sim.resolution = 2 * time.Millisecond
		events := &[]recordedPodEvent{}
		wire := func() {
			sim.controller.podEventFn = func(namespace, podName, podUID, reason, message string) {
				*events = append(*events, recordedPodEvent{namespace, podName, podUID, "", reason, message})
			}
		}
		wire()
		sim.run(60 * time.Second)
		return sim, events
	}

	t.Run("limiters that refill themselves", func(t *testing.T) {
		large, small := smallLaunchWorker("large", 40), smallLaunchWorker("small", 25)
		sim, events := newSim(t, large, small)
		starved := &starvedHypervisor{rng: seedRNG(1)}
		leftOut, unaccounted := 0, 0
		watch := func() {
			for _, meter := range sim.controller.tokenMeters {
				unaccounted = max(unaccounted, meter.unaccounted)
				if meter.excluded {
					leftOut++
				}
			}
		}
		starved.run(sim, 20*time.Minute, watch)
		assertNear(t, "large worker after twenty minutes", large.meanUtil(240), 0.40, simUtilTolerance)
		assertNear(t, "small worker after twenty minutes", small.meanUtil(240), 0.25, simUtilTolerance)

		large.upLimit = 20
		starved.run(sim, 10*time.Minute, watch)
		t.Logf("%d stalls in thirty minutes; the highest count of looks held against a worker was %d of %d",
			starved.stalls, unaccounted, meterExclusionIntervals)
		if leftOut > 0 || len(*events) > 0 {
			t.Errorf("a worker was left out of the split in %d intervals, and %d Events were raised", leftOut, len(*events))
		}
		// Ten seconds after the change, for the rest of the ten minutes.
		history := large.history[len(large.history)-1200+20:]
		assertNear(t, "large worker after its limit was halved", meanOf(history), 0.20, 0.03)
		assertNear(t, "small worker at the end", small.meanUtil(240), 0.25, simUtilTolerance)
	})

	t.Run("and a worker that rewrites its balance", func(t *testing.T) {
		neighbor, writer := smallLaunchWorker("neighbor", 40), smallLaunchWorker(simWriterUID, 30)
		sim, events := newSim(t, neighbor, writer)
		starved := &starvedHypervisor{rng: seedRNG(1)}
		meter := sim.controller.tokenMeters["writer:gpu-sim"]
		intervals, leftOutAfter, takenBack, neighborLeftOut := 0, 0, 0, 0
		starved.run(sim, 30*time.Minute, func() {
			intervals++
			switch {
			case meter.excluded && leftOutAfter == 0:
				leftOutAfter = intervals
			case !meter.excluded && leftOutAfter > 0:
				takenBack++
			}
			if sim.controller.tokenMeters["neighbor:gpu-sim"].excluded {
				neighborLeftOut++
			}
			writer.info(t).SetERLCurrentTokens(1e12)
		})
		// A control pass that falls into a stall of the whole process is skipped.
		if leftOutAfter == 0 || leftOutAfter > meterExclusionIntervals+meterExclusionIntervals/8 {
			t.Errorf("the writer was left out after %d intervals, want within %d", leftOutAfter, meterExclusionIntervals)
		}
		if takenBack > 0 || neighborLeftOut > 0 {
			t.Errorf("while it kept writing the writer was back in the split in %d intervals; its neighbor was left "+
				"out in %d", takenBack, neighborLeftOut)
		}
		for _, event := range *events {
			if event.podUID != simWriterUID {
				t.Errorf("an Event was raised for %s", event.podUID)
			}
		}
		if len(*events) == 0 {
			t.Error("no Event was raised for the writer")
		}

		// It stops; the stalls go on.
		back := 0
		starved.run(sim, 2*time.Minute, func() {
			if meter.excluded {
				back++
			}
		})
		t.Logf("left out %d intervals after it began to write, taken back %d intervals after it stopped", leftOutAfter, back)
		if back > 2*(meterQuarantineIntervals+meterReadmissionIntervals) {
			t.Errorf("the writer was taken back %d intervals after it stopped, want within %d",
				back, 2*(meterQuarantineIntervals+meterReadmissionIntervals))
		}
	})
}
