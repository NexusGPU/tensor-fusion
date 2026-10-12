package computing

import (
	"math"
	"testing"
	"time"

	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
)

// The limiter's side of the contract (see debtBound) as the simulated
// workloads (simWorker, pacedWorker) follow it, in one place: admit while the
// balance is above zero, charge in full down to the debt bound, refill the
// bucket when its refill timestamp is stale, and let one launch through when
// the balance has not moved for thirty seconds.

const (
	// limiterStalledBucketAfter is how long the balance of a bucket has to stay
	// exactly as it is, with a launch waiting on it, before the limiter takes
	// the bucket for one nobody refills (no hypervisor, and no refilling of its
	// own) and admits that launch.
	limiterStalledBucketAfter = 30 * time.Second
	// A launch that is not admitted sleeps between limiterMinWait and
	// limiterMaxWait before it looks at the balance again.
	limiterMinWait = time.Millisecond
	limiterMaxWait = 20 * time.Millisecond
)

// limiterCharge takes the cost of admitted work from the balance the way the
// limiter does: in full, except that a charge never takes the balance below
// -bound, with bound as debtBound defines it from the capacity, the rate and
// the limit in the mapping at the time of the charge. Work admitted from a
// stalled bucket is charged down to -bound whatever it costs. A balance that is
// already below -bound is left as it is.
func limiterCharge(info *workerstate.SharedDeviceInfoV2, upLimit uint32, cost float64, stalled bool) {
	balance := info.GetERLCurrentTokens()
	floor := -debtBound(info.GetERLTokenCapacity(), info.GetERLTokenRefillRate(), upLimit)
	if stalled && !isFinite(floor) {
		// No bound can be told from a rate or capacity that is not a number.
		return
	}
	charged := balance - cost
	if stalled || charged < floor {
		charged = floor
	}
	if balance < floor {
		charged = balance
	}
	info.SetERLCurrentTokens(charged)
}

// limiterWait is how long a launch that was not admitted waits in the
// simulations before it looks again: the time the debt takes to repay at the
// published rate, and never less than limiterMinWait or more than
// limiterMaxWait. The contract does not fix that time; the simulations use one
// value, which keeps them deterministic.
func limiterWait(balance, rate float64) time.Duration {
	if !isFinite(balance) || !isFinite(rate) || rate <= 0 {
		return limiterMaxWait
	}
	seconds := clampFloat64(-balance/rate, limiterMinWait.Seconds(), limiterMaxWait.Seconds())
	return time.Duration(seconds * float64(time.Second))
}

func TestLimiterModelCharge(t *testing.T) {
	worker := smallLaunchWorker("worker", 25)
	worker.paused = true
	newPacingSim(t, worker)
	info := worker.info(t)
	info.SetERLTokenRefillRate(1000)
	info.SetERLTokenCapacity(100)
	bound := debtBound(100, 1000, 25) // a minute of work with the whole device: 240000 tokens

	for _, tc := range []struct {
		name          string
		balance, cost float64
		stalled       bool
		want          float64
	}{
		{name: "charged in full", balance: 40, cost: 10, want: 30},
		{name: "charged in full into debt", balance: 5, cost: 1000, want: -995},
		{name: "not below the bound", balance: 5, cost: 10 * bound, want: -bound},
		{name: "already below the bound", balance: -2 * bound, cost: 10, want: -2 * bound},
		{name: "stalled bucket, small launch", balance: -3, cost: 10, stalled: true, want: -bound},
		{name: "stalled bucket, nothing owed", balance: 0, cost: 10, stalled: true, want: -bound},
		{name: "stalled bucket already below the bound", balance: -2 * bound, cost: 10, stalled: true, want: -2 * bound},
		{name: "stalled bucket whose balance is not a number", balance: math.NaN(), cost: 10, stalled: true, want: -bound},
	} {
		info.SetERLCurrentTokens(tc.balance)
		limiterCharge(info, worker.upLimit, tc.cost, tc.stalled)
		if got := info.GetERLCurrentTokens(); got != tc.want {
			t.Errorf("%s: balance %v charged %v became %v, want %v", tc.name, tc.balance, tc.cost, got, tc.want)
		}
	}

	for _, tc := range []struct {
		balance, rate float64
		want          time.Duration
	}{
		{balance: 0, rate: 1000, want: limiterMinWait},
		{balance: 3, rate: 1000, want: limiterMinWait},
		{balance: -0.2, rate: 1000, want: limiterMinWait},
		{balance: -7, rate: 1000, want: 7 * time.Millisecond},
		{balance: -500, rate: 1000, want: limiterMaxWait},
		{balance: -bound, rate: 1000, want: limiterMaxWait},
		{balance: -7, rate: 0, want: limiterMaxWait},
		{balance: -7, rate: math.NaN(), want: limiterMaxWait},
		{balance: math.NaN(), rate: 1000, want: limiterMaxWait},
		{balance: math.Inf(1), rate: 1000, want: limiterMaxWait},
	} {
		if got := limiterWait(tc.balance, tc.rate); got != tc.want {
			t.Errorf("a launch facing a balance of %v at %v tokens/s sleeps %v, want %v", tc.balance, tc.rate, got, tc.want)
		}
	}
}

// With the hypervisor gone and a limiter that does not refill its own bucket,
// nothing repays what the worker owes. Its launches wait: there is no admission
// after a time. Only when the balance has not moved for thirty seconds is the
// waiting launch let through, and it is charged down to the debt bound, so the
// worker gets one launch per thirty seconds for as long as the bucket stays
// stalled.
func TestStalledBucketAdmitsOneLaunchPerThirtySeconds(t *testing.T) {
	worker := smallLaunchWorker("worker", 25)
	sim := newPacingSim(t, worker)
	sim.run(30 * time.Second)
	info := worker.info(t)
	bound := debtBound(info.GetERLTokenCapacity(), info.GetERLTokenRefillRate(), worker.upLimit)

	sim.hypervisorDown = true
	// What the bucket still holds is spent within a few launches.
	sim.run(time.Second)
	stalledFrom, before := sim.now, len(worker.admissions)
	if balance := info.GetERLCurrentTokens(); balance > 0 || balance < -worker.launchCost {
		t.Fatalf("balance = %.1f a second after the hypervisor went away", balance)
	}

	sim.run(100 * time.Second)
	escapes := worker.admissions[before:]
	if len(escapes) != 3 || worker.escapes != 3 {
		t.Fatalf("%d launches were admitted in 100 s on a stalled bucket (%d of them as stalled), want 3: %v",
			len(escapes), worker.escapes, escapes)
	}
	// Thirty seconds, one sleep and one launch apart. The first one waited
	// since the balance last moved: in the second before stalledFrom, or in the
	// refill period before the hypervisor went away.
	lastMoved := time.Second + erlRefillInterval
	previous, slack := stalledFrom.Add(-lastMoved), lastMoved+limiterMaxWait+worker.launchTime
	for i, at := range escapes {
		if waited := at.Sub(previous); waited < limiterStalledBucketAfter || waited > limiterStalledBucketAfter+slack {
			t.Errorf("launch %d was admitted %v after the balance last moved, want %v",
				i, waited, limiterStalledBucketAfter)
		}
		previous, slack = at, limiterMaxWait+worker.launchTime
	}
	if got := info.GetERLCurrentTokens(); got != -bound {
		t.Errorf("balance = %.1f after a launch from a stalled bucket, want the debt bound of %.1f", got, -bound)
	}

	// A limiter that refills its own bucket never finds it stalled: it keeps
	// the worker at its share for as long as the hypervisor is away.
	refilling := smallLaunchWorker("refilling", 25)
	refilling.selfRefill = true
	sim = newPacingSim(t, refilling)
	sim.run(60 * time.Second)
	sim.hypervisorDown = true
	sim.run(70 * time.Second)
	if refilling.escapes != 0 {
		t.Errorf("%d launches were admitted as from a stalled bucket while the limiter refilled it", refilling.escapes)
	}
	assertNear(t, "utilization with the limiter refilling", refilling.meanUtil(120), 0.25, simUtilTolerance)
}

// The same rule in the coarse simulation, where the limiter is asked once per
// refill period.
func TestStalledBucketInTheCoarseSimulation(t *testing.T) {
	worker := &simWorker{uid: "worker", upLimit: 25, demand: 1}
	sim := newClosedLoopSim(t, worker)
	sim.run(simConvergeIntervals)
	info := worker.deviceInfo(t)
	bound := debtBound(info.GetERLTokenCapacity(), info.GetERLTokenRefillRate(), worker.upLimit)

	// Nobody refills from here on: only the limiter is asked.
	info.SetERLCurrentTokens(-5)
	periods := int(limiterStalledBucketAfter / erlRefillInterval)
	for round := range 3 {
		for period := range periods {
			if admitted, _ := worker.limiterAdmits(info); admitted {
				t.Fatalf("round %d: work was admitted after %d refill periods without tokens", round, period)
			}
		}
		admitted, stalled := worker.limiterAdmits(info)
		if !admitted || !stalled {
			t.Fatalf("round %d: after thirty seconds without a change admitted = %v, stalled = %v", round, admitted, stalled)
		}
		limiterCharge(info, worker.upLimit, 100, stalled)
		if got := info.GetERLCurrentTokens(); got != -bound {
			t.Fatalf("round %d: balance = %.1f, want the debt bound of %.1f", round, got, -bound)
		}
	}
	// A balance that moves is not stalled, however long the worker waits.
	for period := range 2 * periods {
		info.SetERLCurrentTokens(-bound + float64(period%7))
		if admitted, _ := worker.limiterAdmits(info); admitted {
			t.Fatalf("work was admitted on a balance that keeps changing, in period %d", period)
		}
	}
	// A workload with nothing to run is not waiting, so it is never let through
	// as stalled and its balance is not charged.
	worker.demand = 0
	info.SetERLCurrentTokens(-5)
	for range 2 * periods {
		if _, stalled := worker.limiterAdmits(info); stalled {
			t.Fatal("an idle workload was admitted from a stalled bucket")
		}
	}
	if worker.escapes != 3 {
		t.Errorf("escapes = %d, want 3", worker.escapes)
	}
}

// A balance that is not a number is not tokens. The limiter does not let work
// through on it, and does not time out either: the launch waits for the refill
// that restarts the bucket, which is the hypervisor's when the limiter does not
// refill itself. Before, the simulated limiter ran unthrottled on a balance of
// +Inf.
func TestNonFiniteBalanceBlocksUntilTheBucketIsRestarted(t *testing.T) {
	for name, value := range map[string]float64{
		"not a number": math.NaN(), "positive infinity": math.Inf(1), "negative infinity": math.Inf(-1),
	} {
		t.Run(name, func(t *testing.T) {
			worker := smallLaunchWorker("worker", 50)
			sim := newPacingSim(t, worker)
			sim.run(60 * time.Second)
			assertNear(t, "utilization before", worker.meanUtil(20), 0.50, simUtilTolerance)
			info := worker.info(t)

			sim.hypervisorDown = true
			info.SetERLCurrentTokens(value)
			sim.run(10 * pacedResolution) // a launch that was running ends
			before := len(worker.admissions)
			sim.run(10 * time.Second)
			if got := len(worker.admissions) - before; got != 0 {
				t.Fatalf("%d launches were admitted on a balance of %v", got, value)
			}
			if got := info.GetERLCurrentTokens(); math.Float64bits(got) != math.Float64bits(value) {
				t.Fatalf("the balance became %v with nobody refilling", got)
			}

			// The hypervisor comes back and restarts the bucket from empty.
			sim.hypervisorDown = false
			sim.startController()
			restartedAt := sim.now
			sim.run(2 * time.Second)
			if len(worker.admissions) == before {
				t.Fatalf("nothing was admitted in the two seconds after the hypervisor came back (balance %v)",
					info.GetERLCurrentTokens())
			}
			waited := worker.admissions[before].Sub(restartedAt)
			if waited > erlUpdateInterval+3*limiterMaxWait {
				t.Errorf("the first launch was admitted %v after the hypervisor came back", waited)
			}
			t.Logf("first launch %v after the hypervisor came back", waited)
			if got := info.GetERLCurrentTokens(); !isFinite(got) || got > info.GetERLTokenCapacity() {
				t.Errorf("balance = %v after the restart", got)
			}
			sim.run(40 * time.Second)
			assertNear(t, "utilization after", worker.meanUtil(20), 0.50, simUtilTolerance)
		})
	}

	// In the coarse simulation the refill at the end of a period restarts the
	// bucket, and the workload runs from the next one.
	t.Run("coarse simulation", func(t *testing.T) {
		worker := &simWorker{uid: "worker", upLimit: 50, demand: 1}
		sim := newClosedLoopSim(t, worker)
		sim.run(simConvergeIntervals)
		info := worker.deviceInfo(t)
		for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			info.SetERLCurrentTokens(value)
			if admitted, _ := worker.limiterAdmits(info); admitted {
				t.Errorf("work was admitted on a balance of %v", value)
			}
			sim.run(simMeasureIntervals)
			if got := info.GetERLCurrentTokens(); !isFinite(got) {
				t.Errorf("the balance is still %v after the refill loop ran", got)
			}
			assertNear(t, "utilization", worker.meanUtil(simMeasureIntervals/2), 0.50, simUtilTolerance)
		}
	})
}
