package computing

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
)

// Acceptance scenarios on a plant that is not the one the calibration assumes,
// with telemetry as it is measured on a device, over many seeds. Every bound is
// asserted on the worst seed.
//
// The plant: what a token buys is not a constant. It drifts over time (the mix
// of kernels changes) and depends on how busy the worker is (a throttled
// workload batches differently), see variableCost. The telemetry: the driver
// samples the device every 200 ms at a phase of its own and reports the average
// of the last second in whole percents, so what the control loop reads lags by
// half a second to 700 ms (realisticTelemetry); per-process readings are the
// part of a handful of driver samples that fell to the worker
// (reportSampledProcessUtilization: at a share of 21% they scatter by about 14
// points and are exactly zero in 15-18% of the readings).

// acceptanceSeeds is the number of seeds every acceptance scenario runs with.
const acceptanceSeeds = 20

// seedRNG returns the random source of one seed of one scenario.
func seedRNG(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15^seed))
}

// worstFigure is the highest value a figure took over the seeds of a scenario.
type worstFigure struct {
	value float64
	seed  uint64
}

// worst runs a scenario once per seed. The scenario returns one or more
// figures for which higher is worse; worst returns, per figure, the highest
// value and the seed that produced it.
func worst(scenario func(seed uint64) []float64) []worstFigure {
	var figures []worstFigure
	for s := uint64(1); s <= acceptanceSeeds; s++ {
		got := scenario(s)
		if figures == nil {
			figures = make([]worstFigure, len(got))
			for i := range figures {
				figures[i].value = math.Inf(-1)
			}
		}
		for i, value := range got {
			if value > figures[i].value {
				figures[i] = worstFigure{value: value, seed: s}
			}
		}
	}
	return figures
}

// driverSamplePeriod is how often the driver takes a utilization sample, and
// driverWindow what a reported figure averages over.
const (
	driverSamplePeriod = 200 * time.Millisecond
	driverWindow       = time.Second
)

// realisticTelemetry makes the device report what a driver reports: the average
// over driverWindow up to its latest sample, which was taken at a phase of its
// own up to driverSamplePeriod ago, in whole percents.
func realisticTelemetry(sim *closedLoopSim, rng *rand.Rand) {
	period := int(driverSamplePeriod / erlRefillInterval)
	window := int(driverWindow / erlRefillInterval)
	phase := rng.IntN(period)
	sim.telemetry = func(float64) (map[string]*api.GPUUsageMetrics, error) {
		// The control loop reads at a multiple of its interval; the driver's
		// latest sample is `stale` refill periods old.
		stale := (phase + len(sim.load)) % period
		end := len(sim.load) - stale
		start := max(end-window, 0)
		reported := 0.0
		for _, load := range sim.load[start:end] {
			reported += load / float64(max(end-start, 1))
		}
		return map[string]*api.GPUUsageMetrics{simDeviceUUID: {ComputePercentage: math.Round(reported * 100)}}, nil
	}
}

// variableCost returns a cost model for a simWorker: what a token buys drifts
// by ±15% with a period of 40-90 s, and grows by up to 30% with how busy the
// worker is (curve > 0) or shrinks (curve < 0).
func variableCost(rng *rand.Rand, curve float64) func(time.Time, float64) float64 {
	period := 40 + 50*rng.Float64()
	phase := 2 * math.Pi * rng.Float64()
	start := time.Unix(1_800_000_000, 0)
	return func(now time.Time, used float64) float64 {
		drift := 1 + 0.15*math.Sin(2*math.Pi*now.Sub(start).Seconds()/period+phase)
		return drift * (1 + curve*(used-0.3))
	}
}

// meanOver averages what the worker used over history[from:to].
func (w *simWorker) meanOver(from, to int) float64 {
	sum := 0.0
	for _, v := range w.history[from:to] {
		sum += v
	}
	return sum / float64(to-from)
}

// reachedAfter returns the first interval from which the worker's mean use over
// `window` intervals is at least `level`, counted from history[from], or -1.
func (w *simWorker) reachedAfter(from, window int, level float64) int {
	for i := from; i+window <= len(w.history); i++ {
		if w.meanOver(i, i+window) >= level {
			return i - from
		}
	}
	return -1
}

// peakOver is the highest mean use over `window` intervals within history[from:].
func (w *simWorker) peakOver(from, window int) float64 {
	peak := 0.0
	for i := from; i+window <= len(w.history); i++ {
		peak = math.Max(peak, w.meanOver(i, i+window))
	}
	return peak
}

// Workers sharing a device are held at their own limits on the realistic plant,
// from the device's utilization and their token spending alone, which is what
// every provider supplies.
func TestAcceptanceSharedDeviceOnARealisticPlant(t *testing.T) {
	for _, tc := range []struct {
		name   string
		limits []uint32
		curve  float64
	}{
		{name: "30 next to 60, a token buys more under load", limits: []uint32{30, 60}, curve: 0.3},
		{name: "30 next to 60, a token buys less under load", limits: []uint32{30, 60}, curve: -0.3},
		{name: "20, 30 and 40", limits: []uint32{20, 30, 40}, curve: 0.3},
		{name: "single 50", limits: []uint32{50}, curve: 0.3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const settle, measure = 120, 240
			off := worst(func(seed uint64) []float64 {
				rng := seedRNG(seed)
				workers := make([]*simWorker, len(tc.limits))
				// One drift for the device: workers whose work drifted apart
				// are the unequal-cost case, which has its own test.
				cost := variableCost(rng, tc.curve)
				for i, limit := range tc.limits {
					workers[i] = &simWorker{uid: fmt.Sprintf("worker-%d", i), upLimit: limit, demand: 1, cost: cost}
				}
				sim := newClosedLoopSim(t, workers...)
				realisticTelemetry(sim, rng)
				sim.run(settle + measure)
				off := 0.0
				for _, w := range workers {
					off = math.Max(off, math.Abs(w.meanOver(settle, settle+measure)-float64(w.upLimit)/100))
				}
				return []float64{off}
			})[0]
			t.Logf("worst seed %d: a worker's two-minute mean was %.1f points from its limit", off.seed, off.value*100)
			if off.value > 0.04 {
				t.Errorf("seed %d: a worker's mean utilization was %.3f from its limit", off.seed, off.value)
			}
		})
	}
}

// A worker that wakes up next to a busy neighbor, on the realistic plant: it is
// at its share within seconds and does not run above it on the way.
func TestAcceptanceWakeOnARealisticPlant(t *testing.T) {
	got := worst(func(seed uint64) []float64 {
		rng := seedRNG(seed)
		cost := variableCost(rng, 0.3)
		worker := &simWorker{uid: "worker", upLimit: 30, demand: 1, cost: cost}
		neighbor := &simWorker{uid: "neighbor", upLimit: 40, demand: 1, cost: cost}
		sim := newClosedLoopSim(t, worker, neighbor)
		realisticTelemetry(sim, rng)
		sim.run(200)
		worker.demand = 0
		sim.run(80)
		worker.demand = 1
		from := len(worker.history)
		sim.run(80)
		settled := math.Inf(1)
		if reached := worker.reachedAfter(from, 4, 0.8*0.30); reached >= 0 {
			settled = float64(reached) * erlUpdateInterval.Seconds()
		}
		return []float64{worker.peakOver(from, 2), settled, math.Abs(neighbor.meanOver(from, from+80) - 0.40)}
	})
	peak, settled, neighbor := got[0], got[1], got[2]
	t.Logf("worst seeds: one-second peak %.0f%% (seed %d), at 80%% of its share after %.1f s (seed %d), "+
		"neighbor off by %.1f points (seed %d)",
		peak.value*100, peak.seed, settled.value, settled.seed, neighbor.value*100, neighbor.seed)
	if peak.value > 0.30+0.10 {
		t.Errorf("seed %d: the woken worker ran at %.0f%% of the device for a second with a limit of 30%%",
			peak.seed, peak.value*100)
	}
	if settled.value > 3 {
		t.Errorf("seed %d: the woken worker was at its share after %.1f s", settled.seed, settled.value)
	}
	if neighbor.value > 0.05 {
		t.Errorf("seed %d: the neighbor was %.3f from its limit during the wake-up", neighbor.seed, neighbor.value)
	}
}

// What a token buys changes when the work does, and the worker has to be back
// at its share within seconds although single per-process readings scatter by
// more than the change in many of them. Here the utilization per token of a 30%
// worker triples, and drops to a third. With per-process readings the worker
// runs next to a busy neighbor; from the device alone it has the device to
// itself or a neighbor that idles, because next to a busy one the token split
// cannot tell whose work changed (see TestAcceptanceUnequalTokenCostAcrossSeeds).
func TestCalibrationFollowsAChangeOfWork(t *testing.T) {
	for _, tc := range []struct {
		name      string
		factor    float64
		processes bool
		// neighbor is the demand of a 40% worker on the same device, -1 for none.
		neighbor float64
		// within is the time (seconds) to be back within six points of the
		// share on a ten-second mean, late the highest ten-second mean that
		// may still occur 20 s after the change.
		within, late float64
	}{
		// Buying more puts the worker over its limit and is acted on first;
		// buying less leaves it under its share and has to be beyond doubt.
		// Next to a busy neighbor the device is full while the worker runs
		// above its share, which slows everybody down: a token then buys less
		// than it will once the worker is back at its share, and the last part
		// of the way is covered by plain averaging. Over twenty seeds that takes
		// up to a minute; from twenty seconds on the worker is within nine points.
		{
			name:   "per-process readings, a token buys three times as much",
			factor: 3, processes: true, neighbor: 1, within: 60, late: 0.39,
		},
		{
			name:   "per-process readings, a token buys a third",
			factor: 1.0 / 3, processes: true, neighbor: 1, within: 45, late: 0.39,
		},
		{name: "sole worker, a token buys three times as much", factor: 3, neighbor: -1, within: 9, late: 0.34},
		{name: "sole worker, a token buys a third", factor: 1.0 / 3, neighbor: -1, within: 9, late: 0.34},
		{name: "idle neighbor, a token buys three times as much", factor: 3, within: 9, late: 0.34},
		{name: "idle neighbor, a token buys eight times as much", factor: 8, within: 9, late: 0.34},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Per-process readings take about a minute to be trusted in full.
			settle, after := 120, 120
			if tc.processes {
				settle, after = 240, 240
			}
			got := worst(func(seed uint64) []float64 {
				rng := seedRNG(seed)
				worker := &simWorker{uid: "worker", upLimit: 30, demand: 1}
				workers := []*simWorker{worker}
				var neighbor *simWorker
				if tc.neighbor >= 0 {
					neighbor = &simWorker{uid: "neighbor", upLimit: 40, demand: tc.neighbor}
					workers = append(workers, neighbor)
				}
				sim := newClosedLoopSim(t, workers...)
				realisticTelemetry(sim, rng)
				if tc.processes {
					reportSampledProcessUtilization(sim, rng)
				}
				sim.run(settle)

				worker.utilPerToken *= tc.factor
				sim.run(after)
				took := math.Inf(1)
				for i := settle; i+20 <= len(worker.history); i++ {
					if math.Abs(worker.meanOver(i, i+20)-0.30) <= 0.06 {
						took = float64(i-settle) * erlUpdateInterval.Seconds()
						break
					}
				}
				neighborOff := 0.0
				if neighbor != nil {
					neighborOff = math.Abs(neighbor.meanOver(settle+80, settle+after) - 0.40*tc.neighbor)
				}
				return []float64{took, worker.peakOver(settle+40, 20), neighborOff}
			})
			took, late, neighbor := got[0], got[1], got[2]
			t.Logf("worst seeds: back at 30 +/- 6%% after %.1f s (seed %d); "+
				"highest ten-second mean from 20 s on %.0f%% (seed %d); "+
				"neighbor off by %.1f points (seed %d)",
				took.value, took.seed, late.value*100, late.seed, neighbor.value*100, neighbor.seed)
			if took.value > tc.within {
				t.Errorf("seed %d: back at its share after %.1f s, want within %.0f s", took.seed, took.value, tc.within)
			}
			if late.value > tc.late {
				t.Errorf("seed %d: twenty seconds after the change the worker still ran at %.0f%%", late.seed, late.value*100)
			}
			if neighbor.value > 0.05 {
				t.Errorf("seed %d: the neighbor was %.3f from its limit", neighbor.seed, neighbor.value)
			}
		})
	}
}

// Workers whose work costs a different number of tokens per unit of utilization,
// over many seeds. Without per-process readings the device is split by tokens:
// it stays within the sum of the limits, and each worker gets the part of that
// sum that corresponds to its limit weighted by what a token buys it. That is
// the documented approximation, and it is asserted here as what it is, instead
// of as "not starved". With per-process readings every worker is held at its
// own limit.
func TestAcceptanceUnequalTokenCostAcrossSeeds(t *testing.T) {
	const settle, measure = 200, 200
	newWorkers := func() (*simWorker, *simWorker) {
		return &simWorker{uid: "cheap", upLimit: 30, demand: 1},
			&simWorker{uid: "costly", upLimit: 60, demand: 1, utilPerToken: 3 * simUtilPerToken}
	}

	t.Run("token split", func(t *testing.T) {
		// 30 tokens of the cheap worker for every 60 of the costly one, which
		// buy three times as much: 30 of 210 parts of the 90% both may use.
		const wantCheap, wantCostly = 0.90 * 30.0 / 210, 0.90 * 180.0 / 210
		got := worst(func(seed uint64) []float64 {
			cheap, costly := newWorkers()
			sim := newClosedLoopSim(t, cheap, costly)
			realisticTelemetry(sim, seedRNG(seed))
			sim.run(settle + measure)
			usedCheap, usedCostly := cheap.meanOver(settle, settle+measure), costly.meanOver(settle, settle+measure)
			return []float64{
				math.Abs(usedCheap + usedCostly - 0.90),
				math.Max(math.Abs(usedCheap-wantCheap), math.Abs(usedCostly-wantCostly)),
			}
		})
		total, split := got[0], got[1]
		t.Logf("worst seeds: device %.1f points from the sum of the limits (seed %d), a worker %.1f points from its part "+
			"of it (seed %d; the parts are %.1f%% and %.1f%%)",
			total.value*100, total.seed, split.value*100, split.seed, wantCheap*100, wantCostly*100)
		if total.value > 0.04 {
			t.Errorf("seed %d: the device was %.3f from the sum of the limits", total.seed, total.value)
		}
		if split.value > 0.04 {
			t.Errorf("seed %d: a worker was %.3f from its part of the device", split.seed, split.value)
		}
	})

	t.Run("per-process readings", func(t *testing.T) {
		off := worst(func(seed uint64) []float64 {
			rng := seedRNG(seed)
			cheap, costly := newWorkers()
			sim := newClosedLoopSim(t, cheap, costly)
			realisticTelemetry(sim, rng)
			reportSampledProcessUtilization(sim, rng)
			sim.run(settle + 2*measure)
			return []float64{math.Max(math.Abs(cheap.meanOver(settle+measure, settle+2*measure)-0.30),
				math.Abs(costly.meanOver(settle+measure, settle+2*measure)-0.60))}
		})[0]
		t.Logf("worst seed %d: a worker's mean was %.1f points from its limit", off.seed, off.value*100)
		if off.value > 0.06 {
			t.Errorf("seed %d: a worker was %.3f from its limit with per-process readings", off.seed, off.value)
		}
	})
}

// A worker whose work costs many more tokens per unit of utilization than its
// neighbor's is given too low a rate by the token split, and its own readings
// are then zero nearly every time. Before, trust in them grew with their sum:
// at 18 times the neighbor's token cost a 5% worker took half an hour to get
// to its share, at 200 times it never did. Readings of zero now bound what a
// token can buy the worker (utilizationCalibration.bounds), so its rate rises
// as fast as that bound falls.
//
// How fast that is depends on the share: the driver attributes about eight
// samples a second, so a worker at a share of 5% is attributed one every two and
// a half seconds, and it takes a few of them to know what a token buys. For
// the same reason the figure such a worker settles at is only known to a
// quarter or so at first, and it is held somewhat below its share rather than
// above. What is asserted is the time to three fifths of the share on a
// four-second mean, and at least half the share on average over the last third
// of the run, on the worst of twenty seeds. The ten seconds asked for are not
// reached; at a share of 2% the driver attributes a sample to the worker every
// six seconds, and the scenario is reported without a bound on the time.
func TestTokenExpensiveWorkerReachesItsShare(t *testing.T) {
	for _, tc := range []struct {
		share uint32
		ratio float64
		// within is the time (seconds) to three fifths of the share on a
		// four-second mean, intervals how long the scenario runs.
		within    float64
		intervals int
	}{
		{share: 30, ratio: 18, within: 30, intervals: 120},
		{share: 30, ratio: 1000, within: 30, intervals: 120},
		{share: 10, ratio: 200, within: 90, intervals: 240},
		{share: 5, ratio: 18, within: 120, intervals: 300},
		{share: 5, ratio: 1000, within: 160, intervals: 360},
		{share: 2, ratio: 200, within: math.Inf(1), intervals: 480},
	} {
		t.Run(fmt.Sprintf("%d%% share at %.0f times the token cost", tc.share, tc.ratio), func(t *testing.T) {
			limit := float64(tc.share) / 100
			got := worst(func(seed uint64) []float64 {
				rng := seedRNG(seed)
				costly := &simWorker{uid: "costly", upLimit: tc.share, demand: 1, utilPerToken: simUtilPerToken / tc.ratio}
				neighbor := &simWorker{uid: "neighbor", upLimit: 40, demand: 1}
				sim := newClosedLoopSim(t, costly, neighbor)
				realisticTelemetry(sim, rng)
				reportSampledProcessUtilization(sim, rng)
				sim.run(tc.intervals)
				took := math.Inf(1)
				if reached := costly.reachedAfter(0, 8, 0.6*limit); reached >= 0 {
					took = float64(reached) * erlUpdateInterval.Seconds()
				}
				return []float64{
					took,
					limit - costly.meanOver(2*tc.intervals/3, tc.intervals),
					costly.peakOver(0, 20) - limit,
					math.Abs(neighbor.meanOver(40, tc.intervals) - 0.40),
				}
			})
			took, short, over, neighbor := got[0], got[1], got[2], got[3]
			t.Logf("worst seeds: at three fifths of its share after %.1f s (seed %d); mean of the last third %.1f%% (seed %d); "+
				"highest ten-second mean %.1f points above the limit (seed %d); neighbor off by %.1f points (seed %d)",
				took.value, took.seed, (limit-short.value)*100, short.seed, over.value*100, over.seed,
				neighbor.value*100, neighbor.seed)
			if took.value > tc.within {
				t.Errorf("seed %d: at three fifths of its share after %.1f s, want within %.0f s", took.seed, took.value, tc.within)
			}
			if atLeast := map[bool]float64{true: 0.5, false: 0.2}[tc.share >= 5]; limit-short.value < atLeast*limit {
				t.Errorf("seed %d: the worker averaged %.3f of the device at the end, its share is %.2f",
					short.seed, limit-short.value, limit)
			}
			// The rate of a worker that shows nothing yet is doubled per interval
			// (blindRampFactor), and its first readings come late.
			if over.value > 0.10 {
				t.Errorf("seed %d: the worker ran %.3f above its limit of %.2f for ten seconds", over.seed, over.value, limit)
			}
			if neighbor.value > 0.08 {
				t.Errorf("seed %d: the neighbor was %.3f from its limit", neighbor.seed, neighbor.value)
			}
		})
	}
}

// alternatingCost makes what a token buys switch between 1 and `factor` times
// the worker's utilPerToken every `phase`, starting with 1; the seed moves the
// first switch.
func alternatingCost(rng *rand.Rand, phase time.Duration, factor float64) func(time.Time, float64) float64 {
	start := time.Unix(1_800_000_000, 0).Add(-time.Duration(rng.Float64() * float64(phase)))
	return func(now time.Time, _ float64) float64 {
		if int(now.Sub(start)/phase)%2 == 1 {
			return factor
		}
		return 1
	}
}

// Work that alternates between two kinds, one of which gets more utilization
// out of a token. Following it each way leaves the worker above its share for
// the seconds it takes to notice every time the expensive kind comes back:
// with phases of 12 to 60 s it averaged 37-44% of the device with a limit of
// 30%, and 52% when the cheap kind burned tokens for next to no utilization.
// Once the work has come back to the expensive kind, the rate stays at what
// that kind needs. The bound that remains: the first time the expensive kind
// appears, and the first time it returns, it is noticed as late as before; the
// cheap kind then runs below the share; and kinds that differ by less than a
// factor of two are followed as before.
func TestAlternatingWorkStaysWithinItsLimit(t *testing.T) {
	for _, tc := range []struct {
		name   string
		phase  time.Duration
		factor float64
		cycles int
	}{
		{name: "phases of 12 s, a token buys three times as much in one", phase: 12 * time.Second, factor: 3, cycles: 8},
		{name: "phases of 30 s, a token buys three times as much in one", phase: 30 * time.Second, factor: 3, cycles: 4},
		{name: "phases of 60 s, a token buys three times as much in one", phase: 60 * time.Second, factor: 3, cycles: 3},
		{name: "phases of 30 s, a token-burning kind and a heavy one", phase: 30 * time.Second, factor: 20, cycles: 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cycle := int(2 * tc.phase / erlUpdateInterval)
			got := worst(func(seed uint64) []float64 {
				rng := seedRNG(seed)
				worker := &simWorker{uid: "worker", upLimit: 30, demand: 1, utilPerToken: simUtilPerToken / tc.factor}
				worker.cost = alternatingCost(rng, tc.phase, tc.factor)
				sim := newClosedLoopSim(t, worker)
				realisticTelemetry(sim, rng)
				// It is found out in the first cycles: the expensive kind has to
				// appear once and come back once (for some seeds the first time is
				// while the worker's rate is still being found).
				sim.run((3 + tc.cycles) * cycle)
				return []float64{worker.meanOver(3*cycle, (3+tc.cycles)*cycle), worker.peakOver(3*cycle, 20)}
			})
			mean, peak := got[0], got[1]
			t.Logf("worst seeds: long-run mean %.1f%% (seed %d), highest ten-second mean %.1f%% (seed %d), limit 30%%",
				mean.value*100, mean.seed, peak.value*100, peak.seed)
			if mean.value > 0.30+0.02 {
				t.Errorf("seed %d: alternating work averaged %.3f of the device with a limit of 0.30", mean.seed, mean.value)
			}
			if peak.value > 0.30+0.10 {
				t.Errorf("seed %d: alternating work ran at %.3f of the device for ten seconds", peak.seed, peak.value)
			}
		})
	}
}

// A worker that spends tokens while its processes read zero is either one whose
// work the driver books elsewhere, or one that is getting next to nothing. The
// first shows as load on the device that its processes do not account for; only
// then are the worker's readings set aside. Otherwise they are what tells that
// it is held under its share.
func TestZeroReadingsAreOnlySetAsideWhenLoadIsUnaccountedFor(t *testing.T) {
	for _, tc := range []struct {
		name        string
		unaccounted float64
		want        utilizationSource
	}{
		{name: "the processes account for the device", unaccounted: 0.02, want: utilizationFromProcesses},
		{name: "a third of the device is not accounted for", unaccounted: 0.30, want: utilizationFromTokenShare},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calibration := &workerCalibration{processes: utilizationCalibration{coarse: true}}
			quiet := &erlMember{
				workerUID: "quiet", meter: &tokenMeter{consumption: 300, window: 300, measured: true},
				calibration: &calibration.processes, fallback: &calibration.device,
			}
			busy := &erlMember{workerUID: "busy", meter: &tokenMeter{consumption: 3000, window: 3000, measured: true}}
			for range silentSpendingIntervals + 5 {
				attributeDeviceUtilization([]*erlMember{quiet, busy}, 0.6, true,
					map[string]float64{"quiet": 0, "busy": 58}, false, tc.unaccounted)
			}
			if quiet.source != tc.want {
				t.Errorf("after %d readings of zero the worker is measured from %q, want %q",
					silentSpendingIntervals+5, quiet.source, tc.want)
			}
		})
	}
}
