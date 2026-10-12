package computing

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strconv"
	"strings"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
)

// reportProcessUtilizationInRuns makes the simulation report per-process
// utilization the way it was measured on a device that two busy pods share: the
// driver's attribution of its samples to processes is not independent from one
// sample to the next. For seconds on end it books nearly everything one process
// does under the other, then splits evenly again, then errs the other way.
//
// Measured at one reading per second over 190 s, with tenants limited to 30% and
// 50% of the device: the 30% tenant read exactly zero in 13% of the readings, in
// runs of three to eight seconds, and two to three times its share in others
// while its neighbor read a third of its own; the deviation of one reading from
// the tenant's share correlated with the next by 0.66 and with the one three
// seconds later by 0.27. Over minutes the readings average to the shares.
//
// The model: the attribution is in one state for two to eight seconds. In half
// of the states it is even. In the others part of what one worker does is
// booked under another: half or 95% of the smaller of the two shares, for a
// pair of workers picked at random, so that every worker gains as often as it
// loses and the readings stay unbiased. On top of that every reading is the
// part of a handful of driver samples that fell to the worker, as in
// reportSampledProcessUtilization.
func reportProcessUtilizationInRuns(sim *closedLoopSim, rng *rand.Rand) {
	const samplesPerWindow, samplesPerInterval = 8, 4
	window := make([]int, 0, samplesPerWindow)
	left, from, to, moved := 0, 0, 0, 0.0
	shares := make([]float64, len(sim.workers))

	sim.controller.SetProcessUtilizationProvider(func() *ProcessUtilizationSnapshot {
		if left == 0 {
			left, moved = 4+rng.IntN(13), 0
			if len(sim.workers) > 1 && rng.IntN(2) == 0 {
				from = rng.IntN(len(sim.workers))
				to = (from + 1 + rng.IntN(len(sim.workers)-1)) % len(sim.workers)
				moved = []float64{0.5, 0.95}[rng.IntN(2)]
			}
		}
		left--
		for i, w := range sim.workers {
			shares[i] = 0
			if len(w.history) > 0 {
				shares[i] = w.history[len(w.history)-1]
			}
		}
		if moved > 0 {
			amount := moved * math.Min(shares[from], shares[to])
			shares[from], shares[to] = shares[from]-amount, shares[to]+amount
		}
		for range samplesPerInterval {
			owner, draw := -1, rng.Float64()
			for i, share := range shares {
				if draw < share {
					owner = i
					break
				}
				draw -= share
			}
			window = append(window, owner)
		}
		if len(window) > samplesPerWindow {
			window = window[len(window)-samplesPerWindow:]
		}

		processes := make([]api.ProcessInformation, len(sim.workers))
		for i := range sim.workers {
			attributed := 0
			for _, owner := range window {
				if owner == i {
					attributed++
				}
			}
			processes[i] = api.ProcessInformation{
				ProcessID: strconv.Itoa(1000 + i), DeviceUUID: simDeviceUUID,
				ComputeUtilizationPercent: 100 * float64(attributed) / float64(len(window)),
			}
		}
		snapshot := NewProcessUtilizationSnapshot(sim.now, processes)
		for i, w := range sim.workers {
			snapshot.AddWorkerProcess(w.uid, simDeviceUUID, processes[i].ComputeUtilizationPercent)
		}
		return snapshot
	})
}

// restartScenario is a hypervisor restart under two busy tenants limited to 30%
// and 50% of one device.
type restartScenario struct {
	name string
	// before and after say how the provider reports per-process utilization
	// before and after the restart: not at all (""), as independent samples
	// ("sampled"), or in runs ("runs").
	before, after string
	// costRatio is what a token buys the 50% tenant relative to the 30% one.
	costRatio float64
	// away is for how many control intervals there is no hypervisor.
	away int
	// continued leaves the control loop in place instead: the same readings
	// without a restart, for comparison.
	continued bool
	// What is asserted of every five-second window of the thirty seconds after
	// the restart, each as a part of what it is compared with (0: nothing).
	// rise is how far a window may be above the first one, in which the adopted
	// rate is in force, drift how far from it either way, and over how far
	// above the tenant's limit.
	rise, drift, over float64
}

const (
	restartSettle  = 600 // control intervals before the restart (5 min)
	restartWindow  = 10  // control intervals per window (5 s)
	restartWindows = 6   // windows after the restart (30 s)
)

// restartRun is what the two tenants got after a restart, for one seed: every
// five-second window of the thirty seconds after the new control loop started,
// as shares of the device, and their rates at its start and at the end.
type restartRun struct {
	after [2][restartWindows]float64
	rates [2][2]float64
}

func (sc restartScenario) run(t *testing.T, seed uint64) restartRun {
	t.Helper()
	rng := seedRNG(seed)
	workers := []*simWorker{
		{uid: "small", upLimit: 30, demand: 1},
		{uid: "large", upLimit: 50, demand: 1, utilPerToken: sc.costRatio * simUtilPerToken},
	}
	sim := newClosedLoopSim(t, workers...)
	sim.selfRefill = true
	realisticTelemetry(sim, rng)
	report := func(how string) {
		switch how {
		case "sampled":
			reportSampledProcessUtilization(sim, rng)
		case "runs":
			reportProcessUtilizationInRuns(sim, rng)
		}
	}
	report(sc.before)
	sim.run(restartSettle)

	// The hypervisor goes away, the limiters keep their buckets filled at the
	// published rates, and a new control loop starts without any state.
	if !sc.continued {
		sim.hypervisorAway = true
		sim.run(sc.away)
		sim.hypervisorAway = false
		sim.startController()
	}
	report(sc.after)
	var got restartRun
	from := len(workers[0].history)
	sim.step()
	for i, w := range workers {
		got.rates[i][0] = sim.refillRate(w)
	}
	sim.run(restartWindows*restartWindow - 1)
	for i, w := range workers {
		got.rates[i][1] = sim.refillRate(w)
		for k := range restartWindows {
			got.after[i][k] = w.meanOver(from+k*restartWindow, from+(k+1)*restartWindow)
		}
	}
	return got
}

// worst is the highest value of figure over the windows of both tenants, where
// figure gets a window, the tenant's first window and the tenant's limit.
func (r restartRun) worst(figure func(used, first, limit float64) float64) float64 {
	worst := math.Inf(-1)
	for i, limit := range []float64{0.30, 0.50} {
		for _, used := range r.after[i] {
			worst = math.Max(worst, figure(used, r.after[i][0], limit))
		}
	}
	return worst
}

func (r restartRun) String() string {
	var b strings.Builder
	for i, name := range []string{"30%", "50%"} {
		fmt.Fprintf(&b, "%s tenant", name)
		for _, used := range r.after[i] {
			fmt.Fprintf(&b, " %.3f", used)
		}
		fmt.Fprintf(&b, " (rate %.0f -> %.0f tokens/s); ", r.rates[i][0], r.rates[i][1])
	}
	return b.String()
}

// A hypervisor restart must not show in what the tenants of a shared device
// get: the new control loop holds the rates it adopts until it has a signal of
// its own for a worker, and goes on from them.
//
// Measured on a device with tenants at 30% and 50%: the new loop adopted both
// rates, held them for its ten seconds, and then took the 30% tenant's rate to
// 3.3 times the adopted one within two seconds. The tenant ran at 1.9 times its
// limit for 25 s and its neighbor at 0.56 of its own. The ten seconds of
// per-process readings the new loop had by then came in runs (see
// reportProcessUtilizationInRuns) and put what a token bought the tenant at a
// third of the truth.
//
// Asserted, for both tenants, on every five-second window of the thirty seconds
// after the new loop started, on the worst of twenty seeds:
//
//   - With readings in runs no window is more than 30% above the first one, in
//     which the adopted rate is in force, and, where the tenants were at their
//     limits before, none is more than 30% above the limit. The worst seeds
//     are at 22% and 24%; with the hold of ten seconds alone they were at 163%
//     and 98%. The band is what such readings leave of a limit in any case:
//     the loop that is not restarted, on the same readings, has a window 74%
//     above the limit (the scenario that is only logged).
//   - With independent readings windows stay within 15% of the first one
//     either way and 25% above the limit (9% and 19%; before, 31% from the
//     first one).
//   - Without per-process readings windows stay within 5% of the first one,
//     as they did before: this is the case the hold of ten seconds was enough
//     for, whatever the tenants' work costs in tokens and however long the
//     hypervisor is away.
//
// What is not asserted is how far a tenant may fall below its share once the
// rates are adapted again with readings in runs (46% on the worst seed): the
// loop that was not restarted does the same on them, to the same extent, and
// that is not what a restart adds.
func TestRestartDoesNotShowInWhatTenantsGet(t *testing.T) {
	for _, sc := range []restartScenario{
		// As measured: per-process readings in runs, the hypervisor away for
		// four seconds. With such readings the tenants are not at their limits
		// before the restart either.
		{name: "per-process readings in runs", before: "runs", after: "runs", costRatio: 1, away: 8, rise: 0.30},
		// The same from tenants that are at their limits when the hypervisor
		// goes away.
		{
			name:   "per-process readings in runs, tenants at their limits",
			before: "sampled", after: "runs", costRatio: 1, away: 8, rise: 0.30, over: 0.30,
		},
		{
			name:   "per-process readings in runs, tenants at their limits, no restart",
			before: "sampled", after: "runs", costRatio: 1, continued: true,
		},
		{
			name:   "independent per-process readings",
			before: "sampled", after: "sampled", costRatio: 1, away: 8, drift: 0.15, over: 0.25,
		},
		// Without per-process readings, tenants whose work costs a different
		// number of tokens, and a hypervisor that is away for twenty seconds.
		{name: "device utilization only, unequal token cost", costRatio: 1.5, away: 40, drift: 0.05},
		{name: "device utilization only, equal token cost", costRatio: 1, away: 40, drift: 0.05, over: 0.25},
	} {
		t.Run(sc.name, func(t *testing.T) {
			figures := []struct {
				what   string
				bound  float64
				figure func(used, first, limit float64) float64
			}{
				{"above the tenant's first window", sc.rise, func(used, first, _ float64) float64 { return used/first - 1 }},
				{"from the tenant's first window", sc.drift, func(used, first, _ float64) float64 {
					return math.Abs(used/first - 1)
				}},
				{"above the tenant's limit", sc.over, func(used, _, limit float64) float64 { return used/limit - 1 }},
				{"below the tenant's limit", 0, func(used, _, limit float64) float64 { return 1 - used/limit }},
			}
			runs := make([]restartRun, acceptanceSeeds)
			for seed := range runs {
				runs[seed] = sc.run(t, uint64(seed+1))
			}
			for _, f := range figures {
				worstSeed := 0
				for seed := range runs {
					if runs[seed].worst(f.figure) > runs[worstSeed].worst(f.figure) {
						worstSeed = seed
					}
				}
				off := runs[worstSeed].worst(f.figure)
				t.Logf("worst seed %d: a window %.0f%% %s: %s", worstSeed+1, off*100, f.what, runs[worstSeed])
				if f.bound > 0 && off > f.bound {
					t.Errorf("seed %d: a five-second window was %.0f%% %s, want at most %.0f%%",
						worstSeed+1, off*100, f.what, f.bound*100)
				}
			}
		})
	}
}

// The rule of the hold, on its own: the count of intervals that re-establish a
// signal starts over with an interval in which the worker's spending was not
// measured or its figure was not the calibrated one; the hold has an upper
// bound; a bucket at the rate every bucket starts from is not held as one a
// control loop learned; and an adopted rate is not doubled.
func TestAdoptedRateIsHeldUntilTheSignalIsReestablished(t *testing.T) {
	measured := &erlMember{meter: &tokenMeter{measured: true}, calibrated: true}
	for name, without := range map[string]*erlMember{
		"an interval somebody else refilled in": {meter: &tokenMeter{}, calibrated: true},
		"a figure that is not calibrated":       {meter: &tokenMeter{measured: true}},
	} {
		es := &erlState{adopted: true}
		for range adoptionHoldIntervals - 1 {
			if es.signalReestablished(measured) {
				t.Fatalf("%s: re-established before %d intervals in a row", name, adoptionHoldIntervals)
			}
		}
		if es.signalReestablished(without) || es.settled != 0 {
			t.Fatalf("%s: counted as one of a run (%d in a row)", name, es.settled)
		}
		for i := range adoptionHoldIntervals {
			if got, want := es.signalReestablished(measured), i == adoptionHoldIntervals-1; got != want {
				t.Fatalf("%s: re-established = %v after %d intervals in a row", name, got, i+1)
			}
		}
	}

	for _, tc := range []struct {
		rate        float64
		wantAdopted bool
		wantHold    int
	}{
		{rate: 1478, wantAdopted: true, wantHold: adoptionHoldMaxIntervals},
		{rate: initialRate, wantHold: adoptionHoldIntervals},
	} {
		worker := smallLaunchWorker("worker", 30)
		worker.paused = true
		sim := newPacingSim(t, worker)
		info := worker.info(t)
		info.SetERLTokenRefillRate(tc.rate)
		info.SetERLTokenCapacity(tc.rate * burstWindow)
		es := &erlState{currentRate: initialRate}
		sim.controller.adoptPublishedRate(es, "worker:gpu-sim", info, burstWindow)
		if es.currentRate != tc.rate || es.adopted != tc.wantAdopted || es.holdIntervals != tc.wantHold {
			t.Errorf("a published rate of %.0f tokens/s: rate %.0f, adopted %v, held for %d intervals; want adopted %v, %d",
				tc.rate, es.currentRate, es.adopted, es.holdIntervals, tc.wantAdopted, tc.wantHold)
		}
	}

	cfg := defaultERLConfig()
	adopted := &erlState{adopted: true}
	if got, want := computeDesiredRate(1000, 0.3, 0, 0.5, adopted, cfg), 1000*(1+maxRateIncreaseRatio); got != want {
		t.Errorf("an adopted rate of 1000 tokens/s went to %.0f in one interval without a reading, want %.0f", got, want)
	}
	if got, want := computeDesiredRate(1000, 0.3, 0, 0.5, &erlState{}, cfg), 1000*blindRampFactor; got != want {
		t.Errorf("the rate of a new worker went from 1000 to %.0f tokens/s in one interval, want %.0f", got, want)
	}
}
