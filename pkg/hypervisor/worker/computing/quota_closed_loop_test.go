package computing

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
)

const (
	simDeviceUUID = "GPU-sim"
	// Device share produced by spending one token per second: the simulated
	// device is fully busy at 10000 tokens/s.
	simUtilPerToken = 1e-4
)

// simWorker models one limited pod: a workload that spends tokens from its own
// shared-memory bucket, exactly like the in-process limiter does.
type simWorker struct {
	uid     string
	upLimit uint32
	// demand is the device share (0-1) the workload would use when unthrottled.
	demand float64
	// utilPerToken is the device share produced by spending one token per second.
	utilPerToken float64
	// cost, when set, makes what a token buys depend on the time and on how
	// busy the worker was in the previous interval (0-1): it returns the factor
	// by which utilPerToken is multiplied.
	cost func(now time.Time, used float64) float64
	// unpaid makes the workload use the device without going through a limiter:
	// it runs at its demand whatever the balance and spends no tokens.
	unpaid bool
	// freeWithoutLimit makes the worker's limiter leave the bucket alone while
	// the worker has no limit (100%), as measured on a device: the balance
	// stayed at the capacity while the worker used the whole device.
	freeWithoutLimit bool
	// foundUpLimit is the limit the worker's mapping carried when the current
	// controller was started; see closedLoopSim.publishLimits.
	foundUpLimit uint32

	handle *workerstate.SharedMemoryHandle
	// history holds the device share actually used in every interval.
	history []float64
	// periods holds it for every refill period, while the simulation is asked
	// to record that (closedLoopSim.recordPeriods).
	periods []float64

	// waiting is set while the limiter holds the workload back; stalledBits is
	// the balance it was last held back at and stalledFor how long that balance
	// has been seen unchanged since. escapes counts the refill periods of work
	// admitted from a stalled bucket.
	waiting     bool
	stalledBits uint64
	stalledFor  time.Duration
	escapes     int
	// ownRefill is the refill timestamp the worker's limiter stored last while
	// it refilled its own bucket (f64 bits); see closedLoopSim.limitersRefill.
	ownRefill uint64
}

// limiterAdmits reports whether the worker's limiter lets work through in the
// refill period that starts now, and whether it does so because the bucket is
// stalled (see limiterCharge for the rule). The simulation advances in refill
// periods, so a workload that is held back looks again one period later: that
// is its sleep, a fixed 10 ms out of the 1 to 20 ms the limiter sleeps, and one
// period of work is what "a launch" amounts to here.
func (w *simWorker) limiterAdmits(info *workerstate.SharedDeviceInfoV2) (admitted, stalled bool) {
	balance := info.GetERLCurrentTokens()
	if w.demand <= 0 || (isFinite(balance) && balance > 0) {
		// Nothing is waiting: there is nothing to launch, or it is let through.
		w.waiting = false
		return true, false
	}
	if bits := math.Float64bits(balance); !w.waiting || bits != w.stalledBits {
		w.waiting, w.stalledBits, w.stalledFor = true, bits, 0
		return false, false
	}
	if w.stalledFor += erlRefillInterval; w.stalledFor < limiterStalledBucketAfter {
		return false, false
	}
	w.waiting = false
	w.escapes++
	return true, true
}

// pays reports whether the worker's work goes through its token bucket.
func (w *simWorker) pays() bool {
	return !w.unpaid && (!w.freeWithoutLimit || w.upLimit < 100)
}

func (w *simWorker) deviceInfo(t *testing.T) *workerstate.SharedDeviceInfoV2 {
	t.Helper()
	state := w.handle.GetState()
	if state == nil || state.V2 == nil {
		t.Fatalf("worker %s has no shared memory state", w.uid)
	}
	return &state.V2.Devices[0].DeviceInfo
}

// meanUtil averages the device share used over the last n intervals.
func (w *simWorker) meanUtil(n int) float64 {
	if n > len(w.history) {
		n = len(w.history)
	}
	sum := 0.0
	for _, v := range w.history[len(w.history)-n:] {
		sum += v
	}
	return sum / float64(n)
}

// closedLoopSim drives the real controller against a fake device whose
// utilization is the sum of what every worker managed to run: each worker's
// demand capped by the tokens it was granted, and the device capped at 100%.
type closedLoopSim struct {
	t          *testing.T
	controller *Controller
	device     *quotaMetricsDeviceController
	workers    []*simWorker
	now        time.Time
	// externalUtil is the device share (0-1) used by something that is not a
	// limited worker and therefore spends no tokens.
	externalUtil float64
	// telemetry converts the true device share (0-1) into what the provider
	// reports. The default reports it exactly, rounded to a whole percent.
	telemetry func(actual float64) (map[string]*api.GPUUsageMetrics, error)
	// load is the device share used in every refill period of the recent past,
	// oldest first, for telemetry that samples more often than the control loop.
	load []float64
	// recordPeriods makes every worker keep what it used per refill period.
	recordPeriods bool
	// hypervisorAway suspends both loops of the controller: nobody refills the
	// buckets but the limiters themselves (limitersRefill), and nothing adapts.
	// selfRefill lets the limiters do that whenever they find nobody refilling,
	// as they do before a new control loop's first pass.
	hypervisorAway, selfRefill bool
}

// limiterTakeover is how old the refill timestamp of a bucket has to be before
// its limiter refills the bucket itself.
const limiterTakeover = 0.1

// limitersRefill is what the limiters do while nobody refills their buckets: one
// that finds the refill timestamp limiterTakeover old claims the time since
// then, by the same protocol as the hypervisor, and adds what was earned in it
// at the published rate; it goes on with every period for as long as the
// timestamp stays its own.
func (s *closedLoopSim) limitersRefill() {
	nowSecs := float64(s.now.UnixMicro()) / 1e6
	for _, w := range s.workers {
		if !w.pays() {
			continue
		}
		info := w.deviceInfo(s.t)
		rate, capacity, last := info.GetERLTokenRefillRate(), info.GetERLTokenCapacity(), info.GetERLLastTokenUpdate()
		refilled := math.Float64bits(last) != w.ownRefill && last > 0 && nowSecs-last < limiterTakeover
		if rate <= 0 || capacity <= 0 || refilled {
			continue
		}
		if elapsed, _, claimed := info.ClaimERLRefill(nowSecs, maxTokenUpdateGap); claimed {
			w.ownRefill = math.Float64bits(nowSecs)
			info.AddERLTokens(rate*elapsed, capacity)
		}
	}
}

// perToken is the device share one token per second buys the worker right now.
func (w *simWorker) perToken(now time.Time) float64 {
	if w.cost == nil {
		return w.utilPerToken
	}
	used := 0.0
	if len(w.history) > 0 {
		used = w.history[len(w.history)-1]
	}
	return w.utilPerToken * w.cost(now, used)
}

func newClosedLoopSim(t *testing.T, workers ...*simWorker) *closedLoopSim {
	t.Helper()
	sim := &closedLoopSim{
		t:       t,
		device:  &quotaMetricsDeviceController{},
		workers: workers,
		now:     time.Unix(1_800_000_000, 0),
	}
	sim.telemetry = func(actual float64) (map[string]*api.GPUUsageMetrics, error) {
		return map[string]*api.GPUUsageMetrics{
			simDeviceUUID: {ComputePercentage: math.Round(actual * 100)},
		}, nil
	}
	for _, w := range workers {
		if w.utilPerToken == 0 {
			w.utilPerToken = simUtilPerToken
		}
		handle, err := workerstate.CreateSharedMemoryHandle(t.TempDir(),
			&workerstate.PodIdentifier{Namespace: "sim", Name: w.uid},
			[]workerstate.DeviceConfig{{DeviceIdx: 0, DeviceUUID: simDeviceUUID, UpLimit: w.upLimit}})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := handle.Close(); err != nil {
				t.Error(err)
			}
		})
		w.handle = handle
	}

	sim.startController()
	return sim
}

// startController gives the simulation a controller without any state, the way
// a restarted hypervisor has one; the mappings are the ones that exist.
//
//nolint:dupl // the two simulations are kept independent of each other on purpose
func (sim *closedLoopSim) startController() {
	sim.controller = NewQuotaController(sim.device, nil).(*Controller)
	sim.controller.nowFunc = func() time.Time { return sim.now }
	sim.publishLimits(true)
	sim.controller.workerInfoFn = func() map[string]*WorkerInfoSnapshot {
		snapshots := make(map[string]*WorkerInfoSnapshot, len(sim.workers))
		for _, w := range sim.workers {
			snapshots[w.uid] = &WorkerInfoSnapshot{Namespace: "sim", WorkerName: w.uid, Devices: []DeviceSnapshot{
				{DeviceUUID: simDeviceUUID, DeviceIdx: 0, UpLimit: w.upLimit, FoundUpLimit: w.foundUpLimit},
			}}
		}
		return snapshots
	}
	sim.controller.shmHandleFn = func(uid string) *workerstate.SharedMemoryHandle {
		for _, w := range sim.workers {
			if w.uid == uid {
				return w.handle
			}
		}
		return nil
	}
}

// publishLimits is what the worker controller does with the limits: it writes
// every worker's current limit to its mapping, when it opens the mapping
// (opened), after noting the limit the mapping carried, and in every interval
// from then on. While there is no hypervisor nobody does.
func (sim *closedLoopSim) publishLimits(opened bool) {
	for _, w := range sim.workers {
		w.handle.WithState(func(state *workerstate.SharedDeviceState) {
			if opened {
				w.foundUpLimit, _ = state.GetDeviceUpLimit(0)
			}
			state.SetDeviceLimits(0, w.upLimit, 0)
		})
	}
}

// simRefillsPerInterval is how often the refill loop runs within one control
// interval of the simulation.
const simRefillsPerInterval = int(erlUpdateInterval / erlRefillInterval)

// step simulates one control interval: in every refill period each workload
// runs what its limiter admits and the refill loop hands out tokens; then the
// device reports the resulting utilization and the control loop runs once.
//
// The limiter is modelled as the shared account it is: work is admitted while
// the balance is above zero and charged in full, which can take the balance
// below zero but not below the debt bound; nothing is admitted while the
// balance is not above zero or is not a number, however long that lasts, unless
// the balance stops changing altogether (see limiterAdmits and limiterCharge).
func (s *closedLoopSim) step() {
	s.t.Helper()
	dt := erlRefillInterval.Seconds()

	used := make([]float64, len(s.workers))
	deviceTotal := 0.0
	spend := make([]float64, len(s.workers))
	perToken := make([]float64, len(s.workers))
	infos := make([]*workerstate.SharedDeviceInfoV2, len(s.workers))
	stalled := make([]bool, len(s.workers))
	for i, w := range s.workers {
		perToken[i], infos[i] = w.perToken(s.now), w.deviceInfo(s.t)
	}
	for range simRefillsPerInterval {
		total := 0.0
		for i, w := range s.workers {
			spend[i], stalled[i] = 0, false
			admitted := !w.pays()
			if w.pays() {
				admitted, stalled[i] = w.limiterAdmits(infos[i])
			}
			if admitted {
				spend[i] = w.demand / perToken[i] * dt
			}
			total += spend[i] * perToken[i] / dt
		}
		// A saturated device cannot run more than 100%: everyone is slowed down.
		scale := 1.0
		if total+s.externalUtil > 1 {
			scale = (1 - s.externalUtil) / total
			total = 1 - s.externalUtil
		}
		deviceTotal += (total + s.externalUtil) / float64(simRefillsPerInterval)
		s.load = append(s.load, total+s.externalUtil)
		for i, w := range s.workers {
			taken := spend[i] * scale
			if w.pays() && spend[i] > 0 {
				limiterCharge(infos[i], w.upLimit, taken, stalled[i])
			}
			used[i] += taken * perToken[i] / dt / float64(simRefillsPerInterval)
			if s.recordPeriods {
				w.periods = append(w.periods, taken*perToken[i]/dt)
			}
		}
		s.now = s.now.Add(erlRefillInterval)
		if !s.hypervisorAway {
			s.controller.refillBuckets()
		}
		if s.hypervisorAway || s.selfRefill {
			s.limitersRefill()
		}
	}
	for i, w := range s.workers {
		w.history = append(w.history, used[i])
	}
	if keep := 4 * simRefillsPerInterval; len(s.load) > keep {
		s.load = append(s.load[:0], s.load[len(s.load)-keep:]...)
	}

	s.device.metrics, s.device.err = s.telemetry(deviceTotal)
	if !s.hypervisorAway {
		s.publishLimits(false)
		s.controller.updateERLControllers()
	}
}

func (s *closedLoopSim) run(intervals int) {
	s.t.Helper()
	for range intervals {
		s.step()
	}
}

func (s *closedLoopSim) refillRate(w *simWorker) float64 {
	return w.deviceInfo(s.t).GetERLTokenRefillRate()
}

const (
	simConvergeIntervals = 160 // 80 s of simulated time
	simMeasureIntervals  = 40
	// The controller's own deadband is 3 points; allow a little on top of it.
	simUtilTolerance = 0.05
	// A busy worker of the simulation whose rate ends at or below the one it
	// started from was never let up to its share.
	simStarvedRate = initialRate
)

// assertNear fails unless got is within tolerance of want.
func assertNear(t *testing.T, what string, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Errorf("%s = %.3f, want %.3f ± %.3f", what, got, want, tolerance)
	}
}

// Every limited worker sharing a device must be controlled towards its own
// limit, whatever its neighbors do.
func TestERLClosedLoopSharedDeviceConvergesPerWorker(t *testing.T) {
	type workerCase struct {
		upLimit uint32
		demand  float64
		want    float64 // expected steady-state device share
	}
	for _, tc := range []struct {
		name    string
		workers []workerCase
	}{
		{name: "single busy worker", workers: []workerCase{{50, 1, 0.50}}},
		{name: "30 and 70 both busy", workers: []workerCase{{30, 1, 0.30}, {70, 1, 0.70}}},
		{name: "50 and 50 both busy", workers: []workerCase{{50, 1, 0.50}, {50, 1, 0.50}}},
		{name: "20, 30 and 40 all busy", workers: []workerCase{{20, 1, 0.20}, {30, 1, 0.30}, {40, 1, 0.40}}},
		{name: "idle 30 next to busy 70", workers: []workerCase{{30, 0, 0}, {70, 1, 0.70}}},
		{name: "busy 30 next to idle 70", workers: []workerCase{{30, 1, 0.30}, {70, 0, 0}}},
		{name: "light 70 next to busy 30", workers: []workerCase{{70, 0.2, 0.20}, {30, 1, 0.30}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			workers := make([]*simWorker, len(tc.workers))
			for i, w := range tc.workers {
				workers[i] = &simWorker{uid: fmt.Sprintf("worker-%d", i), upLimit: w.upLimit, demand: w.demand}
			}
			sim := newClosedLoopSim(t, workers...)
			sim.run(simConvergeIntervals)

			total, wantTotal := 0.0, 0.0
			for i, w := range workers {
				got := w.meanUtil(simMeasureIntervals)
				total += got
				wantTotal += tc.workers[i].want
				assertNear(t, fmt.Sprintf("%s (limit %d%%) utilization", w.uid, w.upLimit),
					got, tc.workers[i].want, simUtilTolerance)
				// Nobody may be starved because of what a neighbor uses.
				if rate := sim.refillRate(w); rate < simStarvedRate {
					t.Errorf("%s (limit %d%%) is pinned at refill rate %.1f", w.uid, w.upLimit, rate)
				}
			}
			// On a device that is full, nobody can use up what it is granted.
			assertNear(t, "device utilization", total, wantTotal, 1.5*simUtilTolerance)
		})
	}
}

// laggedTelemetry makes the device report the average of the last two control
// intervals, as a driver that averages over a second does, in whole percents.
func laggedTelemetry(sim *closedLoopSim) {
	previous := 0.0
	sim.telemetry = func(actual float64) (map[string]*api.GPUUsageMetrics, error) {
		reported := (actual + previous) / 2
		previous = actual
		return map[string]*api.GPUUsageMetrics{simDeviceUUID: {ComputePercentage: math.Round(reported * 100)}}, nil
	}
}

// transient describes what a worker did in the intervals after a change.
type transient struct {
	peak       float64 // highest share of the device in one interval
	aboveLimit int     // intervals more than three points above the limit
	settled    int     // first interval from which it stays within five points of the limit, -1 if never
}

func observeTransient(sim *closedLoopSim, w *simWorker, intervals int) transient {
	from := len(w.history)
	sim.run(intervals)
	limit := float64(w.upLimit) / 100
	result := transient{settled: -1}
	for i, used := range w.history[from:] {
		result.peak = math.Max(result.peak, used)
		if used > limit+0.03 {
			result.aboveLimit++
		}
		if math.Abs(used-limit) > 0.05 {
			result.settled = -1
		} else if result.settled < 0 {
			result.settled = i
		}
	}
	return result
}

// A worker that starts running, or starts running more, must be at its share
// within a second or two and must not get more than its share on the way. What
// it may have saved up while idle is a tenth of a second of work at its share.
// The same holds whichever signal its utilization is derived from, with
// telemetry that lags by a second, and for its neighbor throughout.
func TestERLTransientsStayWithinTheLimit(t *testing.T) {
	type source struct {
		name string
		// neighbor is the limit of a busy worker on the same device, 0 for none.
		neighbor uint32
		// processes makes the provider report per-process utilization, sampled
		// the way the driver does it.
		processes bool
		// The noise of sampled per-process utilization shows in what a worker
		// gets from one interval to the next.
		peakMargin float64
	}
	sources := []source{
		{name: "sole worker, device utilization", peakMargin: 0.08},
		{name: "token share next to a busy neighbor", neighbor: 40, peakMargin: 0.08},
		{name: "per-process utilization next to a busy neighbor", neighbor: 40, processes: true, peakMargin: 0.16},
	}
	for _, src := range sources {
		newSim := func(t *testing.T, demand float64) (*closedLoopSim, *simWorker, *simWorker) {
			worker := &simWorker{uid: "worker", upLimit: 30, demand: demand}
			workers := []*simWorker{worker}
			var neighbor *simWorker
			if src.neighbor > 0 {
				neighbor = &simWorker{uid: "neighbor", upLimit: src.neighbor, demand: 1}
				workers = append(workers, neighbor)
			}
			sim := newClosedLoopSim(t, workers...)
			laggedTelemetry(sim)
			if src.processes {
				reportSampledProcessUtilization(sim, rand.New(rand.NewPCG(41, 43)))
			}
			return sim, worker, neighbor
		}
		checkNeighbor := func(t *testing.T, neighbor *simWorker, intervals int) {
			if neighbor == nil {
				return
			}
			assertNear(t, "neighbor during the transient", neighbor.meanUtil(intervals), float64(neighbor.upLimit)/100, 0.06)
		}

		t.Run(src.name+"/wakes after idling", func(t *testing.T) {
			sim, worker, neighbor := newSim(t, 1)
			sim.run(2 * simConvergeIntervals)
			rate := sim.refillRate(worker)
			worker.demand = 0
			sim.run(simConvergeIntervals)
			if got := sim.refillRate(worker); got > 1.2*rate {
				t.Errorf("the rate climbed from %.0f to %.0f tokens/s while the worker idled", rate, got)
			}

			worker.demand = 1
			got := observeTransient(sim, worker, 40)
			if got.peak > 0.30+src.peakMargin {
				t.Errorf("peaked at %.0f%% of the device with a limit of 30%%", got.peak*100)
			}
			if got.settled < 0 || got.settled > 4 {
				t.Errorf("at its share after %d intervals, want within two seconds", got.settled)
			}
			if !src.processes && got.aboveLimit > 2 {
				t.Errorf("above its limit in %d intervals", got.aboveLimit)
			}
			checkNeighbor(t, neighbor, 40)
			t.Logf("wake: peak %.0f%%, above the limit in %d intervals, at its share after %d",
				got.peak*100, got.aboveLimit, got.settled)
		})

		t.Run(src.name+"/demand steps up", func(t *testing.T) {
			sim, worker, neighbor := newSim(t, 0.1)
			sim.run(3 * simConvergeIntervals)
			assertNear(t, "light load before the step", worker.meanUtil(simMeasureIntervals), 0.10, 0.03)

			worker.demand = 1
			got := observeTransient(sim, worker, 60)
			if got.peak > 0.30+src.peakMargin {
				t.Errorf("peaked at %.0f%% of the device with a limit of 30%%", got.peak*100)
			}
			if !src.processes && got.aboveLimit > 2 {
				t.Errorf("above its limit in %d intervals", got.aboveLimit)
			}
			if got.settled < 0 || got.settled > 20 {
				t.Errorf("at its share after %d intervals", got.settled)
			}
			checkNeighbor(t, neighbor, 60)
			t.Logf("step: peak %.0f%%, above the limit in %d intervals, at its share after %d",
				got.peak*100, got.aboveLimit, got.settled)
		})

		t.Run(src.name+"/first runs after never running", func(t *testing.T) {
			sim, worker, neighbor := newSim(t, 0)
			sim.run(simConvergeIntervals)
			if got := sim.refillRate(worker); got > 200 {
				t.Errorf("the rate climbed to %.0f tokens/s before the worker ran anything", got)
			}
			worker.demand = 1
			got := observeTransient(sim, worker, 80)
			if got.peak > 0.30+src.peakMargin {
				t.Errorf("peaked at %.0f%% of the device with a limit of 30%%", got.peak*100)
			}
			if got.settled < 0 || got.settled > 40 {
				t.Errorf("at its share after %d intervals", got.settled)
			}
			checkNeighbor(t, neighbor, 80)
			t.Logf("first run: peak %.0f%%, at its share after %d intervals", got.peak*100, got.settled)
		})
	}
}

// A worker that starts while its neighbor is already busy must reach its own
// limit without going over it on the way, and the neighbor must keep its own
// throughout.
func TestERLClosedLoopWorkerWakesNextToBusyNeighbor(t *testing.T) {
	sleeper := &simWorker{uid: "sleeper", upLimit: 30}
	busy := &simWorker{uid: "busy", upLimit: 70, demand: 1}
	sim := newClosedLoopSim(t, sleeper, busy)

	sim.run(simConvergeIntervals)
	assertNear(t, "busy worker alone", busy.meanUtil(simMeasureIntervals), 0.70, simUtilTolerance)

	sleeper.demand = 1
	got := observeTransient(sim, sleeper, simConvergeIntervals)
	if got.peak > 0.30+0.08 {
		t.Errorf("the woken worker peaked at %.0f%% of the device with a limit of 30%%", got.peak*100)
	}
	if got.aboveLimit > 2 {
		t.Errorf("the woken worker was above its limit in %d intervals", got.aboveLimit)
	}
	lowest := 1.0
	for _, used := range busy.history[len(busy.history)-simConvergeIntervals:] {
		lowest = math.Min(lowest, used)
	}
	if lowest < 0.70-0.08 {
		t.Errorf("the busy worker was pushed down to %.0f%% of the device", lowest*100)
	}
	assertNear(t, "woken worker", sleeper.meanUtil(simMeasureIntervals), 0.30, simUtilTolerance)
	assertNear(t, "busy worker", busy.meanUtil(simMeasureIntervals), 0.70, simUtilTolerance)
}

// When a neighbor stops, the remaining worker keeps its own limit instead of
// inheriting the freed share.
func TestERLClosedLoopNeighborStops(t *testing.T) {
	small := &simWorker{uid: "small", upLimit: 30, demand: 1}
	large := &simWorker{uid: "large", upLimit: 70, demand: 1}
	sim := newClosedLoopSim(t, small, large)
	sim.run(simConvergeIntervals)

	large.demand = 0
	sim.run(simConvergeIntervals)
	assertNear(t, "remaining worker", small.meanUtil(simMeasureIntervals), 0.30, simUtilTolerance)
	if got := large.meanUtil(simMeasureIntervals); got != 0 {
		t.Errorf("stopped worker still uses %.3f of the device", got)
	}
}

// Provider utilization is noisy, quantized to whole percents and describes the
// previous sampling window. The per-worker split must stay stable under that.
func TestERLClosedLoopToleratesNoisyDelayedTelemetry(t *testing.T) {
	small := &simWorker{uid: "small", upLimit: 30, demand: 1}
	large := &simWorker{uid: "large", upLimit: 60, demand: 1}
	sim := newClosedLoopSim(t, small, large)

	previous, tick := 0.0, 0
	sim.telemetry = func(actual float64) (map[string]*api.GPUUsageMetrics, error) {
		tick++
		// Deterministic ±3-point ripple on top of a one-interval delay.
		noise := 0.03 * math.Sin(float64(tick)*1.7)
		reported := math.Min(math.Max(previous+noise, 0), 1)
		previous = actual
		return map[string]*api.GPUUsageMetrics{
			simDeviceUUID: {ComputePercentage: math.Round(reported * 100)},
		}, nil
	}

	sim.run(2 * simConvergeIntervals)
	assertNear(t, "small worker", small.meanUtil(simMeasureIntervals), 0.30, 0.08)
	assertNear(t, "large worker", large.meanUtil(simMeasureIntervals), 0.60, 0.08)
}

// Losing telemetry must not starve running workloads: token buckets keep being
// refilled at the last known rate, and the rate is not adapted on stale data.
func TestERLKeepsRefillingTokensWithoutTelemetry(t *testing.T) {
	for _, tc := range []struct {
		name      string
		telemetry func(float64) (map[string]*api.GPUUsageMetrics, error)
	}{
		{name: "query failed", telemetry: func(float64) (map[string]*api.GPUUsageMetrics, error) {
			return nil, errors.New("driver unavailable")
		}},
		{name: "device absent", telemetry: func(float64) (map[string]*api.GPUUsageMetrics, error) {
			return map[string]*api.GPUUsageMetrics{"GPU-other": {ComputePercentage: 10}}, nil
		}},
		{name: "invalid sample", telemetry: func(float64) (map[string]*api.GPUUsageMetrics, error) {
			return map[string]*api.GPUUsageMetrics{simDeviceUUID: {ComputePercentage: math.NaN()}}, nil
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			worker := &simWorker{uid: "worker", upLimit: 50, demand: 1}
			sim := newClosedLoopSim(t, worker)
			sim.run(simConvergeIntervals)
			before := worker.meanUtil(simMeasureIntervals)
			assertNear(t, "utilization before the outage", before, 0.50, simUtilTolerance)
			rate, capacity := sim.refillRate(worker), worker.deviceInfo(t).GetERLTokenCapacity()

			sim.telemetry = tc.telemetry
			sim.run(simMeasureIntervals) // 20 s without usable telemetry

			assertNear(t, "utilization during the outage", worker.meanUtil(simMeasureIntervals), before, 0.02)
			info := worker.deviceInfo(t)
			if info.GetERLTokenRefillRate() != rate || info.GetERLTokenCapacity() != capacity {
				t.Errorf("limits changed without telemetry: rate %.1f -> %.1f, capacity %.1f -> %.1f",
					rate, info.GetERLTokenRefillRate(), capacity, info.GetERLTokenCapacity())
			}
		})
	}
}
