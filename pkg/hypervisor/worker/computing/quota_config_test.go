package computing

import (
	"math"
	"strings"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// The compiled defaults are what the limiter was measured and tested with.
// Overrides remain possible, within the range that is known to work; anything
// else is rejected, reported, and replaced by the default.
func TestERLOverridesAreValidated(t *testing.T) {
	defaults := defaultERLConfig()
	for _, tc := range []struct {
		name string
		erl  tfv1.ElasticRateLimitParameters
		// want is the configuration in effect, rejected a part of every reason
		// that has to be reported.
		want     func(cfg *erlConfig)
		rejected []string
	}{
		{name: "nothing overridden", want: func(*erlConfig) {}},
		{
			name: "values within the range are applied",
			erl:  tfv1.ElasticRateLimitParameters{BurstWindow: "0.2", MinRefillRate: " 0.5 ", Kp: "1.5"},
			want: func(cfg *erlConfig) { cfg.burstWindow, cfg.rateMin, cfg.kp = 0.2, 0.5, 1.5 },
		},
		{
			name: "what the previous chart shipped",
			erl: tfv1.ElasticRateLimitParameters{
				MaxRefillRate: "12000", MinRefillRate: "80", BurstWindow: "0.5", CapacityMin: "150", CapacityMax: "6000",
				FilterAlpha: "0.3", Kp: "0.6", Ki: "0.1", Kd: "0.05", IntegralDecayFactor: "0.95",
			},
			// The control gains are in range; everything that bounds the bucket
			// is what held workloads away from their share.
			want: func(cfg *erlConfig) {
				cfg.utilAlpha, cfg.kp, cfg.ki, cfg.kd, cfg.integralDecayFactor = 0.3, 0.6, 0.1, 0.05, 0.95
			},
			rejected: []string{
				"maxRefillRate=12000", "minRefillRate=80", "burstWindow=0.5", "capacityMin=150", "capacityMax=6000",
			},
		},
		{
			name: "not numbers",
			erl: tfv1.ElasticRateLimitParameters{
				MaxRefillRate: "NaN", MinRefillRate: "Inf", Kp: "fast", Ki: "-Inf", Kd: "1e999",
			},
			want: func(*erlConfig) {},
			rejected: []string{
				`maxRefillRate="NaN"`, `minRefillRate="Inf"`, `kp="fast"`, `ki="-Inf"`, `kd="1e999"`,
			},
		},
		{
			name:     "not positive",
			erl:      tfv1.ElasticRateLimitParameters{BurstWindow: "-0.1", CapacityMin: "-5", Ki: "0", FilterAlpha: "-0.2"},
			want:     func(*erlConfig) {},
			rejected: []string{"burstWindow=-0.1 is negative", "capacityMin=-5", "ki=0", "filterAlpha=-0.2"},
		},
		{
			// No saving up beyond what the refill loop needs: see burstWindow.
			name: "a burst window of zero is a setting",
			erl:  tfv1.ElasticRateLimitParameters{BurstWindow: "0"},
			want: func(cfg *erlConfig) { cfg.burstWindow = 0 },
		},
		{
			// The earlier charts' value, set on its own by a user who wants it:
			// it reaches the hypervisor (see HypervisorSchedulingForHypervisor)
			// and is applied; everything else stays as compiled.
			name: "a burst window of half a second and nothing else",
			erl:  tfv1.ElasticRateLimitParameters{BurstWindow: "0.5"},
			want: func(cfg *erlConfig) { cfg.burstWindow = 0.5 },
		},
		{
			name: "the longest burst window",
			erl:  tfv1.ElasticRateLimitParameters{BurstWindow: "10"},
			want: func(cfg *erlConfig) { cfg.burstWindow = 10 },
		},
		{
			name: "outside the range that works",
			erl: tfv1.ElasticRateLimitParameters{
				BurstWindow: "10.5", FilterAlpha: "0.99", Kp: "50", IntegralDecayFactor: "0.1", MaxRefillRate: "500000",
			},
			want: func(*erlConfig) {},
			rejected: []string{
				"burstWindow=10.5 is above", "filterAlpha=0.99 is above", "kp=50 is above",
				"integralDecayFactor=0.1 is below", "maxRefillRate=500000 is below",
			},
		},
		{
			name: "a capacity ceiling that cannot hold a burst window of the rate ceiling",
			erl: tfv1.ElasticRateLimitParameters{
				MaxRefillRate: "50000000", BurstWindow: "0.2", CapacityMax: "2000000", Kd: "0.2",
			},
			// Each of the three is in range; together they are not. All three
			// keep their defaults, the unrelated override stays.
			want:     func(cfg *erlConfig) { cfg.kd = 0.2 },
			rejected: []string{"capacityMax=2e+06 holds less than burstWindow=0.2 s of maxRefillRate=5e+07"},
		},
		{
			// Before, the compiled capacity ceiling stayed and the rate ceiling
			// was rejected for not fitting under it.
			name: "a capacity ceiling that is not set follows the rate ceiling",
			erl:  tfv1.ElasticRateLimitParameters{MaxRefillRate: "1000000000"},
			want: func(cfg *erlConfig) { cfg.rateMax, cfg.capacityMax = 1e9, 1e9*burstWindowMax },
		},
		{
			// Two seconds of the compiled rate ceiling is more than the compiled
			// capacity ceiling used to hold; the window is applied all the same.
			name: "a burst window of two seconds and nothing else",
			erl:  tfv1.ElasticRateLimitParameters{BurstWindow: "2"},
			want: func(cfg *erlConfig) { cfg.burstWindow = 2 },
		},
		{
			name: "a capacity ceiling the pool sets is kept when it holds the pool's burst window",
			erl:  tfv1.ElasticRateLimitParameters{BurstWindow: "2", CapacityMax: "20000000"},
			want: func(cfg *erlConfig) { cfg.burstWindow, cfg.capacityMax = 2, 2e7 },
		},
		{
			name:     "a capacity ceiling the pool sets that cannot hold the pool's burst window",
			erl:      tfv1.ElasticRateLimitParameters{BurstWindow: "2", CapacityMax: "10000000"},
			want:     func(*erlConfig) {},
			rejected: []string{"capacityMax=1e+07 holds less than burstWindow=2 s of maxRefillRate=1e+07"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			want := defaults
			tc.want(&want)
			got, _, rejected := applyERLOverrides(defaults, tc.erl)
			if got != want {
				t.Errorf("configuration in effect = %+v, want %+v", got, want)
			}
			report := strings.Join(rejected, "; ")
			for _, reason := range tc.rejected {
				if !strings.Contains(report, reason) {
					t.Errorf("the report %q does not mention %q", report, reason)
				}
			}
			if len(tc.rejected) == 0 && len(rejected) != 0 {
				t.Errorf("nothing should be rejected, got %q", report)
			}
			// Whatever comes in, the configuration in effect is one that works.
			if got.rateMin <= 0 || got.rateMin > got.rateMax || got.capacityMin <= 0 || got.capacityMin > got.capacityMax ||
				got.capacityMax < got.rateMax*got.burstWindow || got.rateMin > initialRate || got.rateMax < rateMaxLowest {
				t.Errorf("configuration in effect cannot work: %+v", got)
			}
		})
	}
}

// oddOverrides are configurations at the edges of what validation accepts.
var oddOverrides = []struct {
	name string
	erl  tfv1.ElasticRateLimitParameters
}{
	{name: "longest burst window, highest floors, quick filter, strong gains", erl: tfv1.ElasticRateLimitParameters{
		BurstWindow: "0.25", MinRefillRate: "10", CapacityMin: "20", FilterAlpha: "0.6",
		Kp: "2", Ki: "0.1", Kd: "0.3", IntegralDecayFactor: "0.5",
	}},
	{name: "shortest burst window, lowest floors, slow filter, weak gains", erl: tfv1.ElasticRateLimitParameters{
		BurstWindow: "0.02", MinRefillRate: "0.001", CapacityMin: "0.5", MaxRefillRate: "1000000000",
		CapacityMax: "1000000000", FilterAlpha: "0.05", Kp: "0.3", Ki: "1", Kd: "0.02", IntegralDecayFactor: "0.99",
	}},
}

// The acceptance scenarios hold under overrides at the edges of what is
// accepted, with bounds a little wider than under the compiled defaults (a
// point or two on the steady state, and up to nine points on a 60% neighbor in
// the half minute after a worker next to it woke up), and one bound that
// depends on the override: what a worker can save up while idle is burstWindow
// seconds of its rate, so a woken worker's first second is at most its share
// plus burstWindow of the rest of the device.
func TestAcceptanceUnderOddButValidOverrides(t *testing.T) {
	for _, override := range oddOverrides {
		cfg, _, rejected := applyERLOverrides(defaultERLConfig(), override.erl)
		if len(rejected) != 0 {
			t.Fatalf("%s: overrides meant to be valid were rejected: %v", override.name, rejected)
		}
		t.Run(override.name+"/workers sharing a device, and one of them waking up", func(t *testing.T) {
			got := worst(func(seed uint64) []float64 {
				rng := seedRNG(seed)
				cost := variableCost(rng, 0.3)
				small := &simWorker{uid: "small", upLimit: 30, demand: 1, cost: cost}
				large := &simWorker{uid: "large", upLimit: 60, demand: 1, cost: cost}
				sim := newClosedLoopSim(t, small, large)
				sim.controller.config = cfg
				realisticTelemetry(sim, rng)
				sim.run(240)
				steady := math.Max(math.Abs(small.meanOver(120, 240)-0.30), math.Abs(large.meanOver(120, 240)-0.60))
				small.demand = 0
				sim.run(60)
				small.demand = 1
				sim.run(60)
				return []float64{
					steady, small.peakOver(300, 2), math.Abs(small.meanOver(310, 360) - 0.30),
					math.Abs(large.meanOver(300, 360) - 0.60),
				}
			})
			steady, peak, woken, neighbor := got[0], got[1], got[2], got[3]
			t.Logf("worst seeds: steady %.1f points from a limit (seed %d); woken worker's first-second peak %.0f%% (seed %d), "+
				"then %.1f points from its limit (seed %d); its neighbor %.1f points (seed %d)",
				steady.value*100, steady.seed, peak.value*100, peak.seed, woken.value*100, woken.seed,
				neighbor.value*100, neighbor.seed)
			if steady.value > 0.06 {
				t.Errorf("seed %d: a worker was %.3f from its limit", steady.seed, steady.value)
			}
			if bound := 0.30 + cfg.burstWindow*0.70 + 0.10; peak.value > bound {
				t.Errorf("seed %d: the woken worker's first second was at %.3f of the device, bound %.3f",
					peak.seed, peak.value, bound)
			}
			if woken.value > 0.06 || neighbor.value > 0.09 {
				t.Errorf("after the wake-up the worker was %.3f and its neighbor %.3f from their limits",
					woken.value, neighbor.value)
			}
		})
		t.Run(override.name+"/a saturated worker is paced evenly", func(t *testing.T) {
			worker := smallLaunchWorker("worker", 25)
			sim := newPacingSim(t, worker)
			sim.controller.config = cfg
			sim.run(40 * time.Second)
			measureFrom := sim.now
			sim.run(20 * time.Second)
			assertNear(t, "mean utilization", worker.meanUtil(40), 0.25, simUtilTolerance)
			if gap := worker.maxGap(measureFrom); gap > 4*erlRefillInterval {
				t.Errorf("longest wait between two launches was %v", gap)
			}
		})
		t.Run(override.name+"/a workload that spends 2,000,000 tokens/s gets a share of 90%", func(t *testing.T) {
			worker := &pacedWorker{uid: "worker", upLimit: 90, launchCost: 20000, launchTime: 10 * time.Millisecond}
			sim := newPacingSim(t, worker)
			sim.controller.config = cfg
			sim.resolution = 10 * time.Millisecond
			sim.run(60 * time.Second)
			assertNear(t, "mean utilization", worker.meanUtil(40), 0.90, simUtilTolerance)
		})
		t.Run(override.name+"/a workload that spends 500 tokens/s is held at 5%", func(t *testing.T) {
			worker := &pacedWorker{uid: "worker", upLimit: 5, launchCost: 5, launchTime: 10 * time.Millisecond}
			sim := newPacingSim(t, worker)
			sim.controller.config = cfg
			sim.resolution = 10 * time.Millisecond
			sim.run(60 * time.Second)
			assertNear(t, "mean utilization", worker.meanUtil(40), 0.05, 0.02)
		})
	}
}

// Work that is cheap in tokens needs a rate below the previous floor of 10
// tokens/s to be held at its share: a launch that costs a tenth of a token and
// runs for 50 ms spends 2 tokens/s with the whole device, and ran unlimited
// with a limit of 30%.
func TestWorkCheaperThanTheOldRateFloorIsHeldAtItsShare(t *testing.T) {
	got := worst(func(seed uint64) []float64 {
		cost := 0.05 + 0.01*float64(seed)
		worker := &pacedWorker{uid: "worker", upLimit: 30, launchCost: cost, launchTime: 50 * time.Millisecond}
		sim := newPacingSim(t, worker)
		sim.resolution = 10 * time.Millisecond
		sim.run(90 * time.Second)
		return []float64{math.Abs(worker.meanUtil(80) - 0.30), worker.info(t).GetERLTokenRefillRate()}
	})
	off, rate := got[0], got[1]
	t.Logf("worst seeds: %.1f points from the limit (seed %d); highest rate %.2f tokens/s (seed %d)",
		off.value*100, off.seed, rate.value, rate.seed)
	if off.value > 0.05 {
		t.Errorf("seed %d: cheap work was %.3f from its limit of 0.30", off.seed, off.value)
	}
	if rate.value >= 10 {
		t.Errorf("seed %d: the rate ended at %.2f tokens/s, not below the previous floor", rate.seed, rate.value)
	}
}

// An upgrade replaces the hypervisor under workloads that are paced correctly.
// The rate their buckets carry was published by the previous release, with the
// capacity that release derived from it; the new control loop adopts it. Before,
// it only recognised its own derivation and started every worker over from the
// initial rate: a dip of several seconds.
func TestUpgradeFromThePreviousReleaseKeepsTheRate(t *testing.T) {
	for _, tc := range []struct {
		name                                  string
		burstWindow, capacityMin, capacityMax float64
	}{
		{name: "compiled defaults of the previous release", burstWindow: 0.5, capacityMin: 200, capacityMax: 200000},
		{name: "values the previous chart shipped", burstWindow: 0.5, capacityMin: 150, capacityMax: 6000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			worker := smallLaunchWorker("worker", 25)
			worker.selfRefill = true
			sim := newPacingSim(t, worker)
			sim.resolution = 2 * time.Millisecond
			sim.run(40 * time.Second)
			info := worker.info(t)
			rate := info.GetERLTokenRefillRate()

			// The bucket as the previous release's control loop left it, and a
			// hypervisor that is away for two seconds while it is replaced.
			info.SetERLTokenCapacity(clampFloat64(rate*tc.burstWindow, tc.capacityMin, tc.capacityMax))
			sim.hypervisorDown = true
			sim.run(2 * time.Second)
			sim.hypervisorDown = false
			sim.startController()
			sim.run(erlUpdateInterval + 2*sim.resolution)
			if got := info.GetERLTokenRefillRate(); math.Abs(got-rate) > 0.01*rate {
				t.Fatalf("after the upgrade the refill rate is %.0f tokens/s, it was %.0f", got, rate)
			}
			before := len(worker.history)
			sim.run(15 * time.Second)
			lowest := 1.0
			for i := before; i+4 <= len(worker.history); i++ {
				window := (worker.history[i] + worker.history[i+1] + worker.history[i+2] + worker.history[i+3]) / 4
				lowest = math.Min(lowest, window)
			}
			if lowest < 0.25-0.05 {
				t.Errorf("after the upgrade the worker was down to %.3f of the device for two seconds", lowest)
			}
		})
	}
}

// Setting the wall clock back leaves every refill timestamp ahead of it, and
// nobody refills a bucket whose timestamp is ahead: before, a worker got no
// tokens for as long as the step (up to five seconds). The refill loop now
// notices within a few passes and starts over from the clock, crediting nothing
// for it.
func TestBackwardClockStepDoesNotStopRefilling(t *testing.T) {
	for _, step := range []time.Duration{200 * time.Millisecond, 3 * time.Second, 4900 * time.Millisecond} {
		t.Run(step.String(), func(t *testing.T) {
			worker := smallLaunchWorker("worker", 25)
			sim := newPacingSim(t, worker)
			sim.run(40 * time.Second)
			info := worker.info(t)
			rate := info.GetERLTokenRefillRate()

			// The wall clock goes back; the tickers of the two loops and the
			// work on the device are not affected by that.
			sim.now = sim.now.Add(-step)
			sim.nextRefill, sim.nextControl = sim.nextRefill.Add(-step), sim.nextControl.Add(-step)
			worker.busyUntil = worker.busyUntil.Add(-step)
			stepped, intervals := sim.now, len(worker.history)
			sim.run(10 * time.Second)

			if gap := worker.maxGap(stepped); gap > 100*time.Millisecond {
				t.Errorf("after the clock went back by %v the worker waited %v for tokens", step, gap)
			}
			// Nothing was credited for the step: not the time up to the old
			// timestamp again, and not more than a bucket at any moment.
			used := worker.history[intervals:]
			for i := 0; i+4 <= len(used); i += 4 {
				if window := (used[i] + used[i+1] + used[i+2] + used[i+3]) / 4; math.Abs(window-0.25) > 0.04 {
					t.Errorf("%.0f s after the step the worker used %.3f of the device",
						float64(i)*erlUpdateInterval.Seconds(), window)
				}
			}
			if sim.maxBalance > 1+1e-9 {
				t.Errorf("a bucket held %.3f times its capacity", sim.maxBalance)
			}
			if got := info.GetERLTokenRefillRate(); math.Abs(got-rate) > 0.15*rate {
				t.Errorf("the refill rate moved from %.0f to %.0f tokens/s", rate, got)
			}
		})
	}
}
