package computing

import (
	"math"
	"math/rand/v2"
	"strconv"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
)

// Tests of how the quota controller reads what the provider and the limiter
// hand it: per-process utilization with its "not supplied" marker and its
// sampling noise, and token balances that can be in debt.

func TestProcessUtilizationIsCheckedAgainstTheDevice(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	const unsupplied = -1.0
	sampleOf := func(device string, percents ...float64) *ProcessUtilizationSnapshot {
		processes := make([]api.ProcessInformation, 0, len(percents))
		for i, percent := range percents {
			process := api.ProcessInformation{ProcessID: strconv.Itoa(100 + i), DeviceUUID: device}
			if percent == unsupplied {
				process.ComputeUtilizationUnavailable = true
			} else {
				process.ComputeUtilizationPercent = percent
			}
			processes = append(processes, process)
		}
		return NewProcessUtilizationSnapshot(now, processes)
	}
	sample := func(percents ...float64) *ProcessUtilizationSnapshot { return sampleOf("GPU-0", percents...) }
	type step struct {
		name          string
		percents      []float64
		deviceUtil    float64
		deviceMissing bool
		repeat        int
		want          bool
	}
	for _, tc := range []struct {
		name  string
		steps []step
	}{
		{
			name: "a provider that marks what it cannot supply",
			steps: []step{
				{name: "not supplied", percents: []float64{unsupplied, unsupplied}, deviceUtil: 0.6},
				{name: "supplied", percents: []float64{40, 20}, deviceUtil: 0.6, want: true},
				{name: "not supplied again", percents: []float64{unsupplied, unsupplied}, deviceUtil: 0.6},
				{name: "back", percents: []float64{40, 20}, deviceUtil: 0.6, want: true},
			},
		},
		{
			name: "a measuring provider: single zero readings change nothing",
			steps: []step{
				{name: "idle device", percents: []float64{0, 0}, deviceUtil: 0, want: true},
				{name: "activity", percents: []float64{20, 0}, deviceUtil: 0.21, repeat: 20, want: true},
				{name: "all processes read zero, device reads busy", percents: []float64{0, 0}, deviceUtil: 0.21, want: true},
				{name: "again", percents: []float64{0, 0}, deviceUtil: 0.21, want: true},
				{name: "activity", percents: []float64{40, 0}, deviceUtil: 0.2, want: true},
				{name: "without a device sample", percents: []float64{0, 0}, deviceMissing: true, want: true},
			},
		},
		{
			name: "figures that never account for a busy device",
			steps: []step{
				{name: "idle device", percents: []float64{0, 0}, deviceUtil: 0.02, want: true},
				{name: "device busy, nothing reported", percents: []float64{0, 0}, deviceUtil: 0.6, repeat: 3},
				{name: "no device sample: the verdict stands", percents: []float64{0, 0}, deviceMissing: true},
				{name: "still busy", percents: []float64{0, 0}, deviceUtil: 0.9},
			},
		},
		{
			name: "work booked under something that is not on the device",
			steps: []step{
				{name: "plausible", percents: []float64{30, 30}, deviceUtil: 0.6, repeat: 10, want: true},
				// The first samples after the figures stop adding up still pass.
				{
					name: "processes stop accounting for the device", percents: []float64{0, 1},
					deviceUtil: 0.6, repeat: 6, want: true,
				},
				{name: "persistently", percents: []float64{0, 1}, deviceUtil: 0.6, repeat: 4},
				{name: "plausible again", percents: []float64{30, 30}, deviceUtil: 0.6, repeat: 8, want: true},
			},
		},
		{
			name: "a lightly used device is never held against its processes",
			steps: []step{
				{name: "below the busy mark", percents: []float64{0, 0}, deviceUtil: 0.08, repeat: 40, want: true},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			controller := NewQuotaController(&quotaMetricsDeviceController{}, nil).(*Controller)
			for _, s := range tc.steps {
				got := false
				for range max(s.repeat, 1) {
					got = controller.processUtilizationFor("gpu-0", sample(s.percents...), now, s.deviceUtil, !s.deviceMissing) != nil
				}
				if got != s.want {
					t.Fatalf("%s: per-process utilization used = %v, want %v", s.name, got, s.want)
				}
			}
			// The verdict is kept per device.
			if controller.processUtilizationFor("gpu-1", sampleOf("GPU-1", 30), now, 0.3, true) == nil {
				t.Error("another device must not inherit the verdict")
			}
		})
	}

	controller := NewQuotaController(&quotaMetricsDeviceController{}, nil).(*Controller)
	stale := sample(40, 20)
	if controller.processUtilizationFor("gpu-0", stale, now.Add(processUtilizationMaxAge+time.Second), 0.6, true) != nil {
		t.Error("a stale sample must not be used")
	}
	if controller.processUtilizationFor("gpu-0", nil, now, 0.6, true) != nil {
		t.Error("a missing sample must not be used")
	}
}

// A sample without per-process utilization for a device (the provider has no
// utilization sample of it this time, or does not measure it at all) is not a
// sample of zeros: the workers are measured by other means for that interval,
// and nothing that is remembered about the device changes.
func TestUnsuppliedProcessUtilizationChangesNothingRemembered(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	supplied := NewProcessUtilizationSnapshot(now, []api.ProcessInformation{
		{ProcessID: "100", DeviceUUID: "GPU-0", ComputeUtilizationPercent: 40},
		{ProcessID: "200", DeviceUUID: "GPU-0", ComputeUtilizationPercent: 20},
	})
	unsupplied := NewProcessUtilizationSnapshot(now, []api.ProcessInformation{
		{ProcessID: "100", DeviceUUID: "GPU-0", ComputeUtilizationUnavailable: true},
		{ProcessID: "200", DeviceUUID: "GPU-0", ComputeUtilizationUnavailable: true},
	})
	unreadable := NewProcessUtilizationSnapshot(now, nil)

	controller := NewQuotaController(&quotaMetricsDeviceController{}, nil).(*Controller)
	for range 10 {
		if controller.processUtilizationFor("gpu-0", supplied, now, 0.6, true) == nil {
			t.Fatal("plausible figures must be used")
		}
	}
	before := *controller.processChecks["gpu-0"]
	for _, snapshot := range []*ProcessUtilizationSnapshot{unsupplied, unreadable, nil} {
		for range 50 {
			if controller.processUtilizationFor("gpu-0", snapshot, now, 0.9, true) != nil {
				t.Fatal("figures that were not supplied must not be used")
			}
		}
	}
	if after := *controller.processChecks["gpu-0"]; after != before {
		t.Fatalf("samples without figures changed what is remembered: %+v -> %+v", before, after)
	}
	if controller.processUtilizationFor("gpu-0", supplied, now, 0.6, true) == nil {
		t.Fatal("the figures are used again as soon as they are supplied")
	}
}

// reportSampledProcessUtilization makes the simulation report per-process
// utilization the way the driver produces it: the window behind every reading
// consists of a few short samples, each attributed to a single process. A
// reading is therefore unbiased but coarse: a worker using a fifth of the device
// reads exactly zero in about one reading out of six, and deviates by about 14
// points on average. Successive readings overlap by half a window.
func reportSampledProcessUtilization(sim *closedLoopSim, rng *rand.Rand) (zeroReadings func(uid string) float64) {
	const samplesPerWindow, samplesPerInterval = 8, 4
	// window[i] holds, per short sample, the index of the worker it was
	// attributed to (-1: nobody).
	window := make([]int, 0, samplesPerWindow)
	readings, zeros := map[string]int{}, map[string]int{}

	sim.controller.SetProcessUtilizationProvider(func() *ProcessUtilizationSnapshot {
		for range samplesPerInterval {
			owner, draw := -1, rng.Float64()
			for i, w := range sim.workers {
				share := 0.0
				if len(w.history) > 0 {
					share = w.history[len(w.history)-1]
				}
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
			if len(w.history) > 0 && w.history[len(w.history)-1] > 0 {
				readings[w.uid]++
				if processes[i].ComputeUtilizationPercent == 0 {
					zeros[w.uid]++
				}
			}
		}
		return snapshot
	})
	return func(uid string) float64 {
		if readings[uid] == 0 {
			return 0
		}
		return float64(zeros[uid]) / float64(readings[uid])
	}
}

// Per-process utilization of a lightly used share regularly reads zero although
// the worker is running, and is noisy otherwise. Workers must still be
// controlled to their own limits, a zero reading must not be taken for an idle
// worker, and the device must not drop out of per-process attribution.
func TestERLClosedLoopToleratesZeroReadingsOfProcessUtilization(t *testing.T) {
	for _, tc := range []struct {
		name   string
		limits []uint32
	}{
		{name: "20 next to 50", limits: []uint32{20, 50}},
		{name: "20, 20 and 30", limits: []uint32{20, 20, 30}},
		{name: "single 20", limits: []uint32{20}},
		{name: "10 next to 70", limits: []uint32{10, 70}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			workers := make([]*simWorker, len(tc.limits))
			for i, limit := range tc.limits {
				workers[i] = &simWorker{uid: "worker-" + strconv.Itoa(i), upLimit: limit, demand: 1}
			}
			sim := newClosedLoopSim(t, workers...)
			zeroReadings := reportSampledProcessUtilization(sim, rand.New(rand.NewPCG(20260101, uint64(len(tc.limits)))))

			const settle, measure = 2 * simConvergeIntervals, 8 * simConvergeIntervals
			sim.run(settle)
			otherSources, unknown := 0, 0
			for range measure {
				sim.step()
				for _, w := range workers {
					switch sim.controller.utilSources[w.uid+":gpu-sim"] {
					case utilizationFromProcesses:
					case utilizationUnknown:
						unknown++
					default:
						otherSources++
					}
				}
			}
			// A long run of zero readings of a small share is taken for a worker
			// its processes do not describe, and its tokens are used instead for
			// as long as it lasts. That has to stay the exception, and a worker
			// must never be left without any signal.
			if limit := measure * len(workers) / 20; otherSources > limit {
				t.Errorf("per-process attribution was abandoned in %d worker intervals, more than %d", otherSources, limit)
			}
			if unknown != 0 {
				t.Errorf("%d worker intervals had no utilization signal", unknown)
			}

			for _, w := range workers {
				limit := float64(w.upLimit) / 100
				got := w.meanUtil(measure)
				// The signal is far noisier than whole-device utilization, so the
				// rate keeps moving; on average the worker has to sit at its limit.
				assertNear(t, w.uid+" (limit "+strconv.Itoa(int(w.upLimit))+"%) mean utilization", got, limit, 0.05)
				// A zero reading must not open the throttle.
				peak := 0.0
				for _, v := range w.history[len(w.history)-measure:] {
					peak = math.Max(peak, v)
				}
				if peak > limit+0.25 {
					t.Errorf("%s (limit %d%%) peaked at %.0f%% of the device", w.uid, w.upLimit, peak*100)
				}
				if rate := sim.refillRate(w); rate >= rateMax || rate < simStarvedRate {
					t.Errorf("%s ended at refill rate %.0f", w.uid, rate)
				}
			}
			// The scenario is the one it claims to be: small shares do read zero.
			if rate := zeroReadings(workers[0].uid); rate < 0.08 || rate > 0.55 {
				t.Errorf("%s read zero in %.0f%% of its samples, expected a noticeable minority", workers[0].uid, rate*100)
			}
		})
	}
}

// The limiter can admit work that costs more than the bucket holds and carry
// the difference as a negative balance. Refills repay it; the balance must
// neither be mistaken for corruption nor be reset to an empty bucket.
func TestRebalanceTokenBucketRepaysDebt(t *testing.T) {
	deviceInfo := workerstate.NewSharedDeviceInfoV2(1024, 50, 1<<30)
	deviceInfo.SetERLTokenCapacity(1000)
	deviceInfo.SetERLCurrentTokens(-700)
	deviceInfo.SetERLLastTokenUpdate(10)

	// 500 tokens/s for half a second repay 250.
	if got, added := rebalanceTokenBucket(deviceInfo, 10.5, 500, 1000, 0.5); got != -450 || added != 250 {
		t.Fatalf("balance after the first refill = %v, want -450", got)
	}
	if got, _ := rebalanceTokenBucket(deviceInfo, 11, 500, 1000, 0.5); got != -200 {
		t.Fatalf("balance after the second refill = %v, want -200", got)
	}
	// Bleeding surplus tokens of a worker above its target never touches debt.
	if got, _ := rebalanceTokenBucket(deviceInfo, 11.5, 500, 1000, 0.9); got != 50 {
		t.Fatalf("balance after the third refill = %v, want 50", got)
	}
	if got := deviceInfo.GetERLCurrentTokens(); got != 50 {
		t.Fatalf("stored balance = %v, want 50", got)
	}
}

// A balance that is not a number cannot be refilled, and the limiter stops
// limiting on it. It is the one state that is repaired.
func TestRebalanceTokenBucketRepairsInvalidBalance(t *testing.T) {
	for name, invalid := range map[string]float64{
		"NaN": math.NaN(), "infinite": math.Inf(1), "negative infinite": math.Inf(-1),
	} {
		t.Run(name, func(t *testing.T) {
			deviceInfo := workerstate.NewSharedDeviceInfoV2(1024, 50, 1<<30)
			deviceInfo.SetERLTokenCapacity(1000)
			deviceInfo.SetERLCurrentTokens(invalid)
			deviceInfo.SetERLLastTokenUpdate(10)

			got, _ := rebalanceTokenBucket(deviceInfo, 10.5, 500, 1000, 0.5)
			if got != 250 || deviceInfo.GetERLCurrentTokens() != 250 {
				t.Fatalf("balance = %v (stored %v), want the bucket restarted and refilled to 250",
					got, deviceInfo.GetERLCurrentTokens())
			}
		})
	}
}

// End to end with a limiter that carries debt in the bucket: a launch is
// admitted whenever the balance is positive and is charged in full, so the
// balance swings below zero. The long-run rate must still follow the limit, for
// the worker and for a neighbor accounted by token share.
func TestERLClosedLoopLimiterCarryingDebt(t *testing.T) {
	for _, tc := range []struct {
		name   string
		limits []uint32
	}{
		{name: "single worker at 50", limits: []uint32{50}},
		{name: "30 next to 60", limits: []uint32{30, 60}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			workers := make([]*simWorker, len(tc.limits))
			for i, limit := range tc.limits {
				workers[i] = &simWorker{uid: "worker-" + strconv.Itoa(i), upLimit: limit, demand: 1}
			}
			sim := newClosedLoopSim(t, workers...)
			sim.run(2 * simConvergeIntervals)

			sawDebt := false
			for range 4 * simConvergeIntervals {
				sim.step()
				for _, w := range workers {
					tokens := w.deviceInfo(t).GetERLCurrentTokens()
					if math.IsNaN(tokens) || math.IsInf(tokens, 0) {
						t.Fatalf("%s: balance became %v", w.uid, tokens)
					}
					sawDebt = sawDebt || tokens < 0
				}
			}
			if !sawDebt {
				t.Fatal("the scenario never took a bucket into debt")
			}
			for _, w := range workers {
				assertNear(t, w.uid+" mean utilization", w.meanUtil(4*simConvergeIntervals), float64(w.upLimit)/100, 0.06)
				if meter := sim.controller.tokenMeters[w.uid+":gpu-sim"]; meter == nil || !meter.measured {
					t.Errorf("%s: token consumption is not measured while in debt", w.uid)
				}
			}
		})
	}
}

// A restarted control loop knows nothing about the workers it takes over, and
// the per-process figures it gets are too noisy to judge a worker by a handful
// of them. It adopts the published rates and leaves them alone until its
// estimate has settled, so that the restart does not show in what the workers
// get.
func TestControlLoopRestartWithNoisyProcessUtilization(t *testing.T) {
	small := &simWorker{uid: "small", upLimit: 20, demand: 1}
	large := &simWorker{uid: "large", upLimit: 50, demand: 1}
	sim := newClosedLoopSim(t, small, large)
	reportSampledProcessUtilization(sim, rand.New(rand.NewPCG(7, 11)))
	sim.run(6 * simConvergeIntervals)
	rates := map[*simWorker]float64{small: sim.refillRate(small), large: sim.refillRate(large)}

	// What a worker gets in one interval is a whole number of launches, so the
	// figure of a single interval is good to a few points only.
	atRestart := map[*simWorker]float64{small: small.meanUtil(4), large: large.meanUtil(4)}
	sim.startController()
	reportSampledProcessUtilization(sim, rand.New(rand.NewPCG(13, 17)))
	sim.step()
	for _, w := range sim.workers {
		if got := sim.refillRate(w); math.Abs(got-rates[w]) > 0.01*rates[w] {
			t.Errorf("%s: refill rate after the restart is %.0f tokens/s, it was %.0f", w.uid, got, rates[w])
		}
	}

	// While the new loop's estimate settles, every worker gets exactly what it
	// got in the moment of the restart.
	// (The very first interval is the limiter's to bridge, which this
	// simulation does not model; see the pacing simulation for the hand-over.)
	sim.run(adoptionHoldIntervals - 2)
	for _, w := range sim.workers {
		for i, used := range w.history[len(w.history)-(adoptionHoldIntervals-3):] {
			if math.Abs(used-atRestart[w]) > 0.03 {
				t.Errorf("%s: %.1f s after the restart it used %.3f of the device, at the restart %.3f",
					w.uid, float64(i)*erlUpdateInterval.Seconds(), used, atRestart[w])
			}
		}
	}

	// Afterwards it is controlled as before the restart.
	sim.run(6 * simConvergeIntervals)
	for _, w := range sim.workers {
		assertNear(t, w.uid+" mean utilization after the restart",
			w.meanUtil(6*simConvergeIntervals), float64(w.upLimit)/100, 0.05)
	}
}
