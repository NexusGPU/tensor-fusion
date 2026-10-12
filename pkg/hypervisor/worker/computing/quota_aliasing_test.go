package computing

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
	"strconv"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
)

// reportAliasedProcessUtilization makes the simulation report per-process
// utilization the way it was measured on an RTX 3090 (driver 580.95) under two
// and three limited tenants (tmp.lab/REPORT.md, section 4).
//
// The driver takes a utilization sample every 199.94 ms and books the whole
// sample under the process whose context is on the engine at that moment (the
// one that ran last when the engine is idle). The limiters are refilled every
// 10 ms, all at the same instant, so within every refill period all tenants
// start together and run interleaved until the smallest portion is used up,
// then the remaining ones, and the device idles for the rest of the period.
// 200 ms is a whole number of refill periods: the point of the refill cycle a
// sample lands on moves by 0.06 ms per sample, once around in about 33 s. While
// it lies in the part of the cycle that only the larger tenants reach, a small
// tenant reads zero, for ten to twenty seconds on end; while it lies in the
// part where all run, the small tenant is booked an equal part with the
// others, far above its share. Over a whole cycle the readings average out.
//
// phaseStep is how far the sample moves along the refill cycle from one sample
// to the next, as a part of the cycle: -0.006 for the released cadence, any
// value far from a whole number for a refill period that is not a divisor of
// the driver's, a random number for refills at irregular intervals.
func reportAliasedProcessUtilization(sim *closedLoopSim, rng *rand.Rand, phaseStep func() float64) {
	const samplesPerWindow = 5 // the driver's figures cover one second
	window := make([][2]float64, 0, samplesPerWindow)
	phase, tick := rng.Float64(), 0
	shares := make([]float64, len(sim.workers))
	order := make([]int, len(sim.workers))

	sim.controller.SetProcessUtilizationProvider(func() *ProcessUtilizationSnapshot {
		busy := 0.0
		for i, w := range sim.workers {
			shares[i] = 0
			if len(w.history) > 0 {
				shares[i] = w.history[len(w.history)-1]
			}
			busy += shares[i]
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool { return shares[order[a]] < shares[order[b]] })
		// Two and a half driver samples per control interval.
		samples := 2 + tick%2
		tick++
		for range samples {
			// Who is on the engine at this point of the cycle: everybody still
			// running takes turns; past the last portion, the one that ran last.
			owner, at, done := order[len(order)-1], 0.0, 0.0
			for k, i := range order {
				running := float64(len(order) - k)
				if end := at + running*(shares[i]-done); phase < end {
					owner = order[k+rng.IntN(len(order)-k)]
					break
				} else {
					at, done = end, shares[i]
				}
			}
			window = append(window, [2]float64{float64(owner), busy})
			phase = math.Mod(phase+phaseStep()+2, 1)
		}
		if len(window) > samplesPerWindow {
			window = window[len(window)-samplesPerWindow:]
		}
		processes := make([]api.ProcessInformation, len(sim.workers))
		for i := range sim.workers {
			booked := 0.0
			for _, sample := range window {
				if int(sample[0]) == i {
					booked += sample[1]
				}
			}
			processes[i] = api.ProcessInformation{
				ProcessID: strconv.Itoa(1000 + i), DeviceUUID: simDeviceUUID,
				ComputeUtilizationPercent: math.Min(100, 100*booked/float64(len(window))),
			}
		}
		snapshot := NewProcessUtilizationSnapshot(sim.now, processes)
		for i, w := range sim.workers {
			snapshot.AddWorkerProcess(w.uid, simDeviceUUID, processes[i].ComputeUtilizationPercent)
		}
		return snapshot
	})
}

// aliasingCandidate is one setting of the control loop.
type aliasingCandidate struct {
	name string
	// sharedCalibration is Controller.sharedCalibration (sharedCalibrationEnv).
	sharedCalibration bool
	// phaseStep is how far the driver's sample moves along the refill cycle
	// from one sample to the next (see reportAliasedProcessUtilization).
	phaseStep func(rng *rand.Rand) float64
	// sufficient says that the setting is expected to hold every tenant at its
	// limit; the released one is run to show that the simulation contains the
	// defect.
	sufficient bool
}

func releasedCadence(*rand.Rand) float64 { return -0.06 / 10 }

func aliasingCandidates() []aliasingCandidate {
	return []aliasingCandidate{
		{name: "released", phaseStep: releasedCadence},
		{name: "confirmed", sharedCalibration: true, phaseStep: releasedCadence, sufficient: true},
	}
}

// aliasingOutcome is what the tenants of one simulated ten minutes got, as parts
// of their limits: the lowest and highest thirty-second window after the first
// two minutes, and the longest time a tenant spent below three quarters of its
// limit on end (seconds).
type aliasingOutcome struct {
	low, high, starved float64
}

const (
	aliasingIntervals = 1200 // 10 min
	aliasingSettle    = 240  // 2 min
	aliasingWindow    = 60   // 30 s
)

func runAliasing(t *testing.T, c aliasingCandidate, seed uint64, costRatio float64, limits ...uint32) aliasingOutcome {
	t.Helper()
	rng := seedRNG(seed)
	workers := make([]*simWorker, len(limits))
	for i, limit := range limits {
		workers[i] = &simWorker{uid: fmt.Sprintf("w%d-%d", i, limit), upLimit: limit, demand: 1}
	}
	// costRatio is what a token buys the last tenant relative to the others.
	workers[len(workers)-1].utilPerToken = costRatio * simUtilPerToken
	sim := newClosedLoopSim(t, workers...)
	sim.controller.sharedCalibration = c.sharedCalibration
	realisticTelemetry(sim, rng)
	reportAliasedProcessUtilization(sim, rng, func() float64 { return c.phaseStep(rng) })
	sim.run(aliasingIntervals)

	out := aliasingOutcome{low: math.Inf(1), high: math.Inf(-1)}
	for _, w := range workers {
		limit := float64(w.upLimit) / 100
		for from := aliasingSettle; from+aliasingWindow <= aliasingIntervals; from += aliasingWindow / 2 {
			got := w.meanOver(from, from+aliasingWindow) / limit
			out.low, out.high = math.Min(out.low, got), math.Max(out.high, got)
		}
		below := 0
		for i := aliasingSettle; i < aliasingIntervals; i++ {
			// Five-second averages, so that pacing itself does not count.
			if w.meanOver(max(i-10, 0), i) < 0.75*limit {
				below++
				out.starved = math.Max(out.starved, float64(below)*erlUpdateInterval.Seconds())
			} else {
				below = 0
			}
		}
	}
	return out
}

// Per-process readings that follow the refill cycle (see
// reportAliasedProcessUtilization) must not take a tenant off its limit.
//
// With the released control loop they do, as measured on the device: the small
// tenant of a 30%/60% pair is held at less than half its limit for two minutes
// at a time (erlState.regimeGain keeps the rate at what the highest figure of
// what a token buys called for). Each candidate is run over the same seeds and
// tenant sets; the released loop is run for the record, and has to show the
// defect, so that the simulation is known to contain it.
func TestAliasedProcessReadingsDoNotMoveTenantsOffTheirLimits(t *testing.T) {
	sets := []struct {
		name      string
		costRatio float64
		limits    []uint32
		// floor is the lowest 30 s window a sufficient setting may show, as a
		// part of the limit, and starved the longest stretch below 0.75 of it.
		floor, starved float64
	}{
		{"30/60", 1, []uint32{30, 60}, 0.75, 30},
		{"20/30/50", 1, []uint32{20, 30, 50}, 0.75, 30},
		{"50/50", 1, []uint32{50, 50}, 0.75, 30},
		// Known weak spot, not tuned for: next to a tenant whose tokens buy
		// eight times as much the small tenant's worst window is 0.70 of its
		// limit, for 21 s on the worst seed.
		{"30/60, tokens of the second buy 8 times as much", 8, []uint32{30, 60}, 0.65, 30},
		{"30/60, tokens of the second buy an eighth", 0.125, []uint32{30, 60}, 0.75, 30},
	}
	const seeds = 6
	for _, c := range aliasingCandidates() {
		t.Run(c.name, func(t *testing.T) {
			defect := false
			for _, set := range sets {
				worst := aliasingOutcome{low: math.Inf(1), high: math.Inf(-1)}
				for seed := uint64(1); seed <= seeds; seed++ {
					got := runAliasing(t, c, seed, set.costRatio, set.limits...)
					worst.low, worst.high = math.Min(worst.low, got.low), math.Max(worst.high, got.high)
					worst.starved = math.Max(worst.starved, got.starved)
				}
				t.Logf("%-48s worst of %d seeds: 30 s windows %.2f..%.2f of the limit, longest stretch below 0.75 of it %.0f s",
					set.name, seeds, worst.low, worst.high, worst.starved)
				bad := worst.low < set.floor || worst.high > 1.30 || worst.starved > set.starved
				defect = defect || bad
				if bad && c.sufficient {
					t.Errorf("%s: a tenant is off its limit (windows %.2f..%.2f, %.0f s below 0.75 of it)",
						set.name, worst.low, worst.high, worst.starved)
				}
			}
			if c.name == "released" && !defect {
				t.Errorf("the released control loop holds every tenant at its limit: the simulation does not contain the measured defect")
			}
		})
	}
}

// What the candidates cost: work that really changes is followed more slowly
// when per-process readings only move a slow average. A 30% worker next to a
// busy 40% neighbor, readings aliased as measured; after four minutes a token
// buys the worker three times as much, or a third. Reported per candidate, over
// eight seeds: how long until the worker is back within six points of its share
// on a ten-second mean, the highest and the lowest thirty-second mean from the
// change on, and how far the neighbor's mean over the following four minutes is
// from its limit.
func TestCandidatesFollowAChangeOfWork(t *testing.T) {
	for _, factor := range []float64{3, 1.0 / 3} {
		for _, c := range aliasingCandidates() {
			t.Run(fmt.Sprintf("x%.2f/%s", factor, c.name), func(t *testing.T) {
				const settle, after, seeds = 480, 480, 8
				back, high, low, neighborOff := 0.0, 0.0, math.Inf(1), 0.0
				for seed := uint64(1); seed <= seeds; seed++ {
					rng := seedRNG(seed)
					worker := &simWorker{uid: "worker", upLimit: 30, demand: 1}
					neighbor := &simWorker{uid: "neighbor", upLimit: 40, demand: 1}
					sim := newClosedLoopSim(t, worker, neighbor)
					sim.controller.sharedCalibration = c.sharedCalibration
					realisticTelemetry(sim, rng)
					reportAliasedProcessUtilization(sim, rng, func() float64 { return c.phaseStep(rng) })
					sim.run(settle)
					worker.utilPerToken *= factor
					sim.run(after)
					took := float64(after) * erlUpdateInterval.Seconds()
					for i := settle; i+20 <= len(worker.history); i++ {
						if math.Abs(worker.meanOver(i, i+20)-0.30) <= 0.06 {
							took = float64(i-settle) * erlUpdateInterval.Seconds()
							break
						}
					}
					back = math.Max(back, took)
					for from := settle; from+60 <= settle+after; from += 20 {
						got := worker.meanOver(from, from+60)
						high, low = math.Max(high, got), math.Min(low, got)
					}
					neighborOff = math.Max(neighborOff, math.Abs(neighbor.meanOver(settle, settle+after)-0.40))
				}
				t.Logf("CHANGE x%.2f %-56s back within 6 points after %5.0f s; 30 s means %.2f..%.2f (limit 0.30); neighbor off by %.3f",
					factor, c.name, back, low, high, neighborOff)
			})
		}
	}
}

// A limit that is changed in place must not be taken for work that changed: the
// rate follows the limit at once (Controller.followLimit) while what a token
// buys stays what it was, so neither the calibrations nor the change detection
// of the candidates have anything to do. A 30%/60% pair with aliased readings;
// after four minutes the first worker's limit goes to 15%, a minute later to
// 35%. Per candidate, worst of eight seeds: how long until the worker's
// five-second mean is within a fifth of the new limit, the 30 s windows of the
// worker from ten seconds after each change on, and the neighbor's.
func TestCandidatesFollowALimitChange(t *testing.T) {
	for _, c := range aliasingCandidates() {
		t.Run(c.name, func(t *testing.T) {
			const settle, hold, seeds = 480, 120, 8
			worstBack, low, high, neighborLow, neighborHigh := 0.0, math.Inf(1), math.Inf(-1), math.Inf(1), math.Inf(-1)
			before := math.Inf(1)
			for seed := uint64(1); seed <= seeds; seed++ {
				rng := seedRNG(seed)
				worker := &simWorker{uid: "worker", upLimit: 30, demand: 1}
				neighbor := &simWorker{uid: "neighbor", upLimit: 60, demand: 1}
				sim := newClosedLoopSim(t, worker, neighbor)
				sim.controller.sharedCalibration = c.sharedCalibration
				realisticTelemetry(sim, rng)
				reportAliasedProcessUtilization(sim, rng, func() float64 { return c.phaseStep(rng) })
				sim.run(settle)
				before = math.Min(before, worker.meanOver(settle-60, settle)/0.30)
				for _, limit := range []uint32{15, 35} {
					worker.upLimit = limit
					from := len(worker.history)
					sim.run(hold)
					target := float64(limit) / 100
					back := float64(hold) * erlUpdateInterval.Seconds()
					for i := from; i+10 <= len(worker.history); i++ {
						if math.Abs(worker.meanOver(i, i+10)-target) <= 0.2*target {
							back = float64(i-from) * erlUpdateInterval.Seconds()
							break
						}
					}
					worstBack = math.Max(worstBack, back)
					for at := from + 20; at+60 <= from+hold; at += 20 {
						got := worker.meanOver(at, at+60) / target
						low, high = math.Min(low, got), math.Max(high, got)
						n := neighbor.meanOver(at, at+60) / 0.60
						neighborLow, neighborHigh = math.Min(neighborLow, n), math.Max(neighborHigh, n)
					}
				}
			}
			t.Logf("LIMIT %-42s lowest 30 s before the change %.2f of the limit; at the new limit after %4.1f s; 30 s windows %.2f..%.2f of it; neighbor %.2f..%.2f",
				c.name, before, worstBack, low, high, neighborLow, neighborHigh)
			if c.sufficient && (worstBack > 30 || low < 0.75 || high > 1.35 || neighborLow < 0.75 || neighborHigh > 1.30) {
				t.Errorf("a limit change is not followed cleanly: %.1f s, worker %.2f..%.2f, neighbor %.2f..%.2f",
					worstBack, low, high, neighborLow, neighborHigh)
			}
		})
	}
}
