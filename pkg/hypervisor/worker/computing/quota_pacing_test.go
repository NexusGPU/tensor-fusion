package computing

import (
	"errors"
	"math"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
)

// Tests of how tokens are handed out in time. The control loop decides a rate
// twice a second; the refill loop hands the tokens out in small portions, so a
// throttled workload is slowed down evenly instead of running in bursts.

const pacedResolution = time.Millisecond

// pacedWorker is a workload that always has the next launch ready, behind a
// model of the limiter's side of the contract (see debtBound): all processes of
// a worker share one account, the balance in the mapping. A launch is admitted
// while the balance is above zero and is charged in full, which can take the
// balance below zero but not below the debt bound (limiterCharge). A launch
// that is not admitted waits for as long as it takes and looks again
// (limiterWait). There is no admission after a time, and none on a balance that
// is not a number, which waits for the refill that restarts the bucket. The one
// exception is a bucket whose balance has not changed for
// limiterStalledBucketAfter, which nobody is refilling: the launch waiting on
// it is admitted and charged down to the debt bound. Refills repay the
// overdraft. The limiter only refills itself when it finds the refill
// timestamp stale.
type pacedWorker struct {
	uid     string
	upLimit uint32
	// launchCost is what one launch is charged, launchTime how long it keeps
	// the device busy.
	launchCost float64
	launchTime time.Duration
	// paused stops the workload from launching anything.
	paused bool
	// freeWithoutLimit makes the limiter leave the bucket alone while the
	// worker has no limit (100%); see simWorker.freeWithoutLimit.
	freeWithoutLimit bool
	// selfRefill models a limiter that refills its own bucket when it finds
	// the refill timestamp stale (a tenth of a second old): a launch waiting on
	// an empty bucket claims the time since then by the refill protocol, and
	// goes on refilling for as long as nobody else does.
	selfRefill bool
	ownRefill  uint64
	// legacy models the limiter of the previous release instead, by what it
	// does to the shared balance: it never takes it below zero, and it admits a
	// launch that has waited a second for tokens whatever the balance.
	legacy       bool
	legacyDebt   float64
	waitingSince time.Time
	// selfRefilled counts the tokens the limiter added itself.
	selfRefilled float64
	// sleepUntil is when a launch that was not admitted looks at the balance
	// again. waiting is set while a launch is held back; stalledBits is the
	// balance it last found and stalledSince when it first found it. escapes
	// counts the launches admitted from a stalled bucket.
	sleepUntil   time.Time
	waiting      bool
	stalledBits  uint64
	stalledSince time.Time
	escapes      int

	handle    *workerstate.SharedMemoryHandle
	busyUntil time.Time

	admissions []time.Time
	// busy counts the simulation steps the device spent on this worker, in the
	// current control interval and per finished interval.
	busy    int
	history []float64
}

func (w *pacedWorker) info(t *testing.T) *workerstate.SharedDeviceInfoV2 {
	t.Helper()
	state := w.handle.GetState()
	if state == nil || state.V2 == nil {
		t.Fatalf("worker %s has no shared memory state", w.uid)
	}
	return &state.V2.Devices[0].DeviceInfo
}

// step advances the workload by one simulation step.
func (w *pacedWorker) step(t *testing.T, now time.Time) {
	if w.paused {
		// Nothing is waiting.
		w.waiting = false
		return
	}
	if now.Before(w.busyUntil) {
		w.busy++
		return
	}
	info := w.info(t)
	if w.legacy {
		w.stepLegacy(info, now)
		return
	}
	if w.freeWithoutLimit && w.upLimit >= 100 {
		w.admit(now)
		return
	}
	// A sleep is never longer than limiterMaxWait, whatever happens to the wall
	// clock in the meantime.
	if left := w.sleepUntil.Sub(now); left > 0 && left <= limiterMaxWait {
		return
	}
	if balance := info.GetERLCurrentTokens(); isFinite(balance) && balance > 0 {
		limiterCharge(info, w.upLimit, w.launchCost, false)
		w.admit(now)
		return
	}
	rate := info.GetERLTokenRefillRate()
	w.refillIfStalled(info, now, rate, info.GetERLTokenCapacity())
	balance := info.GetERLCurrentTokens()
	if bits := math.Float64bits(balance); !w.waiting || bits != w.stalledBits {
		w.waiting, w.stalledBits, w.stalledSince = true, bits, now
	} else if now.Sub(w.stalledSince) >= limiterStalledBucketAfter {
		limiterCharge(info, w.upLimit, w.launchCost, true)
		w.escapes++
		w.admit(now)
		return
	}
	w.sleepUntil = now.Add(limiterWait(balance, rate))
}

// admit starts a launch.
func (w *pacedWorker) admit(now time.Time) {
	w.waiting = false
	w.admissions = append(w.admissions, now)
	w.busyUntil = now.Add(w.launchTime)
	w.busy++
}

// legacyAdmitAfter is how long the limiter of the previous release makes a
// launch wait for tokens before admitting it anyway.
const legacyAdmitAfter = time.Second

func (w *pacedWorker) stepLegacy(info *workerstate.SharedDeviceInfoV2, now time.Time) {
	if w.waitingSince.IsZero() {
		w.waitingSince = now
	}
	owed, balance := w.launchCost+w.legacyDebt, info.GetERLCurrentTokens()
	if balance < owed && now.Sub(w.waitingSince) < legacyAdmitAfter {
		return
	}
	paid := math.Max(0, math.Min(balance, owed))
	info.SetERLCurrentTokens(balance - paid)
	w.legacyDebt, w.waitingSince = owed-paid, time.Time{}
	w.admissions = append(w.admissions, now)
	w.busyUntil = now.Add(w.launchTime)
	w.busy++
}

func (w *pacedWorker) refillIfStalled(info *workerstate.SharedDeviceInfoV2, now time.Time, rate, capacity float64) {
	if !w.selfRefill || rate <= 0 || capacity <= 0 {
		return
	}
	nowSecs := float64(now.UnixMicro()) / 1e6
	last := info.GetERLLastTokenUpdate()
	takeover := 0.1
	if math.Float64bits(last) == w.ownRefill {
		takeover = 0.001
	}
	if last > 0 && nowSecs-last < takeover {
		return
	}
	elapsed, _, claimed := info.ClaimERLRefill(nowSecs, 5)
	if !claimed {
		return
	}
	w.ownRefill = math.Float64bits(nowSecs)
	w.selfRefilled += info.AddERLTokens(rate*elapsed, capacity)
}

// maxGap is the longest time between two consecutive admissions since from.
func (w *pacedWorker) maxGap(from time.Time) time.Duration {
	longest := time.Duration(0)
	for i := 1; i < len(w.admissions); i++ {
		if w.admissions[i-1].Before(from) {
			continue
		}
		longest = max(longest, w.admissions[i].Sub(w.admissions[i-1]))
	}
	return longest
}

// meanUtil averages the device share used over the last n control intervals.
func (w *pacedWorker) meanUtil(n int) float64 {
	n = min(n, len(w.history))
	sum := 0.0
	for _, v := range w.history[len(w.history)-n:] {
		sum += v
	}
	return sum / float64(n)
}

// pacingSim runs the real controller, with both of its loops, against paced
// workers on one device in steps of pacedResolution.
type pacingSim struct {
	t          *testing.T
	controller *Controller
	device     *quotaMetricsDeviceController
	workers    []*pacedWorker
	now        time.Time

	nextRefill, nextControl time.Time
	// refillStalled suspends the refill loop, telemetryDown the device metrics,
	// hypervisorDown both loops of the controller.
	refillStalled, telemetryDown, hypervisorDown bool
	// maxBalance is the highest balance seen relative to the bucket's capacity.
	maxBalance float64
	// resolution is the simulation step, pacedResolution unless set. Scenarios
	// that only ask for long-run figures of long launches use a coarser one.
	resolution time.Duration
}

func newPacingSim(t *testing.T, workers ...*pacedWorker) *pacingSim {
	t.Helper()
	sim := &pacingSim{
		t: t, device: &quotaMetricsDeviceController{}, workers: workers, now: time.Unix(1_800_000_000, 0),
	}
	sim.nextRefill, sim.nextControl = sim.now.Add(erlRefillInterval), sim.now.Add(erlUpdateInterval)
	for _, w := range workers {
		handle, err := workerstate.CreateSharedMemoryHandle(t.TempDir(),
			&workerstate.PodIdentifier{Namespace: "sim", Name: w.uid},
			[]workerstate.DeviceConfig{{DeviceIdx: 0, DeviceUUID: simDeviceUUID, UpLimit: w.upLimit}})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = handle.Close() })
		w.handle = handle
	}
	sim.startController()
	return sim
}

// startController gives the simulation a controller without any state, the
// way a restarted hypervisor has one; the mappings are the ones that exist.
//
//nolint:dupl // the two simulations are kept independent of each other on purpose
func (sim *pacingSim) startController() {
	sim.controller = NewQuotaController(sim.device, nil).(*Controller)
	sim.controller.nowFunc = func() time.Time { return sim.now }
	sim.controller.workerInfoFn = func() map[string]*WorkerInfoSnapshot {
		snapshots := make(map[string]*WorkerInfoSnapshot, len(sim.workers))
		for _, w := range sim.workers {
			snapshots[w.uid] = &WorkerInfoSnapshot{Namespace: "sim", WorkerName: w.uid, Devices: []DeviceSnapshot{
				{DeviceUUID: simDeviceUUID, DeviceIdx: 0, UpLimit: w.upLimit},
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

func (s *pacingSim) run(duration time.Duration) {
	s.t.Helper()
	resolution := s.resolution
	if resolution == 0 {
		resolution = pacedResolution
	}
	stepsPerInterval := float64(erlUpdateInterval / resolution)
	for end := s.now.Add(duration); s.now.Before(end); s.now = s.now.Add(resolution) {
		// A ticker that fell behind fires once, not once per missed tick.
		if !s.refillStalled && !s.hypervisorDown && !s.now.Before(s.nextRefill) {
			s.controller.refillBuckets()
			s.nextRefill = s.now.Add(erlRefillInterval)
		}
		if !s.now.Before(s.nextControl) {
			total := 0.0
			for _, w := range s.workers {
				share := float64(w.busy) / stepsPerInterval
				w.history, w.busy = append(w.history, share), 0
				total += share
			}
			s.device.metrics, s.device.err = map[string]*api.GPUUsageMetrics{
				simDeviceUUID: {ComputePercentage: math.Round(math.Min(total, 1) * 100)},
			}, nil
			if s.telemetryDown {
				s.device.metrics, s.device.err = nil, errors.New("driver unavailable")
			}
			if !s.hypervisorDown {
				s.controller.updateERLControllers()
			}
			s.nextControl = s.nextControl.Add(erlUpdateInterval)
		}
		for _, w := range s.workers {
			w.step(s.t, s.now)
			info := w.info(s.t)
			if capacity := info.GetERLTokenCapacity(); capacity > 0 {
				s.maxBalance = math.Max(s.maxBalance, info.GetERLCurrentTokens()/capacity)
			}
		}
	}
}

// A launch of 10 tokens keeps the simulated device busy for 1 ms, so the device
// is fully busy at 10000 tokens/s as in the coarse simulation.
func smallLaunchWorker(uid string, upLimit uint32) *pacedWorker {
	return &pacedWorker{uid: uid, upLimit: upLimit, launchCost: 10, launchTime: time.Millisecond}
}

// A worker held at a share must be slowed down evenly. Its longest wait for
// tokens is of the order of the refill interval; with tokens handed out once
// per control interval it would be most of that interval, twice a second.
func TestRefillPacesASaturatedWorkerEvenly(t *testing.T) {
	for _, share := range []uint32{25, 50, 75} {
		t.Run(strconv.Itoa(int(share))+"%", func(t *testing.T) {
			worker := smallLaunchWorker("worker", share)
			sim := newPacingSim(t, worker)
			sim.run(60 * time.Second)
			measureFrom := sim.now
			sim.run(60 * time.Second)

			limit := float64(share) / 100
			assertNear(t, "mean utilization", worker.meanUtil(120), limit, simUtilTolerance)
			gap := worker.maxGap(measureFrom)
			if bound := 3 * erlRefillInterval; gap > bound {
				t.Errorf("longest wait between two launches was %v, want at most %v", gap, bound)
			}
			lumped := time.Duration((1 - limit) * float64(erlUpdateInterval))
			if gap > lumped/4 {
				t.Errorf("longest wait %v is not clearly below the %v stall of a refill per control interval", gap, lumped)
			}
			if sim.maxBalance > 1+1e-9 {
				t.Errorf("a bucket held %.3f times its capacity", sim.maxBalance)
			}
		})
	}
}

// Two workers on one device are each paced at their own share.
func TestRefillPacesWorkersSharingADevice(t *testing.T) {
	small, large := smallLaunchWorker("small", 25), smallLaunchWorker("large", 50)
	sim := newPacingSim(t, small, large)
	sim.run(90 * time.Second)
	measureFrom := sim.now
	sim.run(60 * time.Second)

	assertNear(t, "small worker", small.meanUtil(120), 0.25, simUtilTolerance)
	assertNear(t, "large worker", large.meanUtil(120), 0.50, simUtilTolerance)
	for _, w := range sim.workers {
		if gap := w.maxGap(measureFrom); gap > 3*erlRefillInterval {
			t.Errorf("%s: longest wait between two launches was %v", w.uid, gap)
		}
	}
}

// The refill loop needs no telemetry. During an outage the rate is held and
// tokens keep arriving at it, as evenly as before.
func TestRefillKeepsPacingWithoutTelemetry(t *testing.T) {
	worker := smallLaunchWorker("worker", 50)
	sim := newPacingSim(t, worker)
	sim.run(60 * time.Second)
	rate := worker.info(t).GetERLTokenRefillRate()
	before := worker.meanUtil(40)

	sim.telemetryDown = true
	outageFrom := sim.now
	sim.run(30 * time.Second)

	if got := worker.info(t).GetERLTokenRefillRate(); got != rate {
		t.Errorf("refill rate changed without telemetry: %.1f -> %.1f", rate, got)
	}
	assertNear(t, "utilization during the outage", worker.meanUtil(60), before, 0.02)
	if gap := worker.maxGap(outageFrom); gap > 3*erlRefillInterval {
		t.Errorf("longest wait between two launches during the outage was %v", gap)
	}
}

// A refill loop that stalls (a long garbage collection, a starved node) makes
// up for exactly the tokens earned in the meantime at its next pass, bounded by
// the capacity.
func TestRefillCatchesUpAfterAStall(t *testing.T) {
	worker := smallLaunchWorker("worker", 25)
	sim := newPacingSim(t, worker)
	sim.run(60 * time.Second)
	info := worker.info(t)
	rate, capacity := info.GetERLTokenRefillRate(), info.GetERLTokenCapacity()

	// The workload pauses too, so the bucket shows what the catch-up adds, and
	// the rate is held as it is without telemetry.
	worker.paused, sim.telemetryDown = true, true
	info.SetERLCurrentTokens(0)
	sim.controller.refillBuckets()
	stalledAt := info.GetERLCurrentTokens()

	sim.refillStalled = true
	sim.run(300 * time.Millisecond)
	if got := info.GetERLCurrentTokens(); got > stalledAt+rate*erlUpdateInterval.Seconds() {
		// Only the control loop may have refilled in between; it uses the same
		// protocol, so the total below still has to come out exactly.
		t.Fatalf("balance grew to %.1f while the refill loop was stalled", got)
	}
	sim.refillStalled = false
	sim.run(2 * pacedResolution)
	want := math.Min(capacity, stalledAt+rate*0.3)
	if got := info.GetERLCurrentTokens(); math.Abs(got-want) > rate*2*erlRefillInterval.Seconds() {
		t.Errorf("after a 300 ms stall the bucket holds %.1f tokens, want about %.1f (rate %.0f/s)", got, want, rate)
	}

	// However long the stall, the bucket never holds more than its capacity.
	sim.refillStalled = true
	sim.run(3 * time.Second)
	sim.refillStalled = false
	sim.run(100 * time.Millisecond)
	if got := info.GetERLCurrentTokens(); got > info.GetERLTokenCapacity() {
		t.Errorf("after a long stall the bucket holds %.1f tokens, capacity is %.1f", got, info.GetERLTokenCapacity())
	}
	if sim.maxBalance > 1+1e-9 {
		t.Errorf("a bucket held %.3f times its capacity", sim.maxBalance)
	}

	// The workload resumes and is paced as before.
	worker.paused, sim.telemetryDown = false, false
	sim.run(30 * time.Second)
	assertNear(t, "utilization after the stall", worker.meanUtil(40), 0.25, simUtilTolerance)
}

// A CUDA graph is charged the cost of all its kernels in one launch, which can
// be several times what the bucket holds. Such a launch is admitted on credit
// and repaid from the following refills. The worker must still be held at its
// share, and must not end up pinned at the smallest bucket or rate.
func TestERLClosedLoopLaunchCostAboveCapacity(t *testing.T) {
	for _, tc := range []struct {
		name string
		// launchTime is how long one launch keeps the device busy; its cost is
		// the tokens for that much work.
		share      uint32
		launchTime time.Duration
		tolerance  float64
	}{
		// At a 25% share the bucket holds about 125 ms of work, at 50% about 250 ms.
		{name: "25% share, launches of 200 ms", share: 25, launchTime: 200 * time.Millisecond, tolerance: 0.06},
		{name: "50% share, launches of 400 ms", share: 50, launchTime: 400 * time.Millisecond, tolerance: 0.06},
		{name: "25% share, launches of 150 ms", share: 25, launchTime: 150 * time.Millisecond, tolerance: 0.06},
		{name: "10% share, launches of 100 ms", share: 10, launchTime: 100 * time.Millisecond, tolerance: 0.04},
	} {
		t.Run(tc.name, func(t *testing.T) {
			worker := &pacedWorker{
				uid: "graph", upLimit: tc.share,
				launchCost: tc.launchTime.Seconds() / simUtilPerToken, launchTime: tc.launchTime,
			}
			sim := newPacingSim(t, worker)
			sim.run(120 * time.Second)
			sim.run(240 * time.Second)

			info := worker.info(t)
			rate, capacity := info.GetERLTokenRefillRate(), info.GetERLTokenCapacity()
			if worker.launchCost <= capacity {
				t.Fatalf("launch cost %.0f does not exceed the capacity %.0f: the scenario is not the one intended",
					worker.launchCost, capacity)
			}
			assertNear(t, "mean utilization", worker.meanUtil(480), float64(tc.share)/100, tc.tolerance)
			if rate <= 2*rateMin {
				t.Errorf("refill rate ended at %.1f tokens/s, pinned at the floor of %.1f", rate, rateMin)
			}
			if rate >= rateMax {
				t.Errorf("refill rate ended at the ceiling of %.1f tokens/s", rateMax)
			}
			if balance := info.GetERLCurrentTokens(); math.IsNaN(balance) || balance < -2*worker.launchCost {
				t.Errorf("balance = %v, more than a launch can overdraw", balance)
			}
		})
	}
}

// A launch that keeps the device busy for a long time is charged in full when
// it is admitted, and the next one waits until the refills have repaid it. A
// worker running such launches is held at its share like any other: there is no
// lowest share below which it cannot be held.
func TestERLLongLaunchesAreHeldAtTheirShare(t *testing.T) {
	for _, tc := range []struct {
		share      uint32
		launchTime time.Duration
	}{
		{share: 25, launchTime: 500 * time.Millisecond},
		{share: 10, launchTime: 500 * time.Millisecond},
		{share: 25, launchTime: 2 * time.Second},
	} {
		t.Run(strconv.Itoa(int(tc.share))+"% share, launches of "+tc.launchTime.String(), func(t *testing.T) {
			worker := &pacedWorker{
				uid: "graph", upLimit: tc.share,
				launchCost: tc.launchTime.Seconds() / simUtilPerToken, launchTime: tc.launchTime,
			}
			sim := newPacingSim(t, worker)
			sim.run(300 * time.Second)
			sim.run(600 * time.Second)
			assertNear(t, "mean utilization", worker.meanUtil(1200), float64(tc.share)/100, 0.03)
			if rate := worker.info(t).GetERLTokenRefillRate(); rate <= 2*rateMin {
				t.Errorf("refill rate ended at %.1f tokens/s, pinned at the floor", rate)
			}
		})
	}
}

// Tokens per unit of utilization differ by orders of magnitude between
// workloads (a graph launch is charged far more than a single kernel doing the
// same work). The floors on rate and capacity must not hold back a worker whose
// work is cheap in tokens, nor the ceilings one whose work is expensive, within
// the configured range.
func TestERLClosedLoopAcrossTokenCosts(t *testing.T) {
	for _, tc := range []struct {
		name string
		// tokensPerSecondAtFullUse is what the workload spends when it has the
		// whole device.
		tokensPerSecondAtFullUse float64
	}{
		{name: "cheap work, 500 tokens/s at full use", tokensPerSecondAtFullUse: 500},
		{name: "reference work, 10000 tokens/s at full use", tokensPerSecondAtFullUse: 10_000},
		{name: "graph-priced work, 370000 tokens/s at full use", tokensPerSecondAtFullUse: 370_000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			worker := &pacedWorker{
				uid: "worker", upLimit: 25,
				launchCost: tc.tokensPerSecondAtFullUse / 1000, launchTime: time.Millisecond,
			}
			sim := newPacingSim(t, worker)
			sim.run(90 * time.Second)
			measureFrom := sim.now
			sim.run(60 * time.Second)

			assertNear(t, "mean utilization", worker.meanUtil(120), 0.25, simUtilTolerance)
			if gap := worker.maxGap(measureFrom); gap > 5*erlRefillInterval {
				t.Errorf("longest wait between two launches was %v", gap)
			}
		})
	}
}

// One pass of the refill loop touches every bucket a hundred times a second, so
// it must not allocate and must stay cheap with hundreds of workers.
func newRefillLoadController(tb testing.TB, buckets int) (*Controller, *time.Time) {
	tb.Helper()
	now := time.Unix(1_800_000_000, 0)
	device := &quotaMetricsDeviceController{metrics: map[string]*api.GPUUsageMetrics{}}
	controller := NewQuotaController(device, nil).(*Controller)
	controller.nowFunc = func() time.Time { return now }

	handles := make(map[string]*workerstate.SharedMemoryHandle, buckets)
	snapshots := make(map[string]*WorkerInfoSnapshot, buckets)
	base := tb.TempDir()
	for i := range buckets {
		uid := "worker-" + strconv.Itoa(i)
		deviceUUID := "GPU-" + strconv.Itoa(i%8)
		handle, err := workerstate.CreateSharedMemoryHandle(base,
			&workerstate.PodIdentifier{Namespace: "load", Name: uid},
			[]workerstate.DeviceConfig{{DeviceIdx: 0, DeviceUUID: deviceUUID, UpLimit: 50}})
		if err != nil {
			tb.Fatal(err)
		}
		tb.Cleanup(func() { _ = handle.Close() })
		handles[uid] = handle
		snapshots[uid] = &WorkerInfoSnapshot{Devices: []DeviceSnapshot{{DeviceUUID: deviceUUID, DeviceIdx: 0, UpLimit: 50}}}
		device.metrics[deviceUUID] = &api.GPUUsageMetrics{ComputePercentage: 40}
	}
	controller.workerInfoFn = func() map[string]*WorkerInfoSnapshot { return snapshots }
	controller.shmHandleFn = func(uid string) *workerstate.SharedMemoryHandle { return handles[uid] }
	controller.updateERLControllers()
	if served := controller.refillSet.Load(); served == nil || len(*served) != buckets {
		tb.Fatalf("the refill loop does not serve all %d buckets", buckets)
	}
	return controller, &now
}

func TestRefillPassDoesNotAllocate(t *testing.T) {
	controller, now := newRefillLoadController(t, 64)
	allocations := testing.AllocsPerRun(200, func() {
		*now = now.Add(erlRefillInterval)
		controller.refillBuckets()
	})
	if allocations != 0 {
		t.Errorf("a refill pass allocates %.1f times, want 0", allocations)
	}
	// And it did refill: every bucket was touched at the latest instant.
	want := float64(now.UnixMicro()) / 1e6
	for _, bucket := range *controller.refillSet.Load() {
		bucket.handle.WithState(func(state *workerstate.SharedDeviceState) {
			if got := state.V2.Devices[0].DeviceInfo.GetERLLastTokenUpdate(); got != want {
				t.Fatalf("bucket last refilled at %.3f, want %.3f", got, want)
			}
		})
	}
}

// BenchmarkRefillPass measures one pass of the refill loop over 500 buckets,
// which runs every erlRefillInterval.
func BenchmarkRefillPass(b *testing.B) {
	const buckets = 500
	controller, now := newRefillLoadController(b, buckets)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		*now = now.Add(erlRefillInterval)
		controller.refillBuckets()
	}
	perPass := float64(b.Elapsed().Nanoseconds()) / float64(b.N)
	b.ReportMetric(perPass/buckets, "ns/bucket")
	// Share of one CPU the loop needs at its cadence.
	b.ReportMetric(100*perPass/float64(erlRefillInterval.Nanoseconds()), "%cpu")
}

// Workers come and go and the limiter stops while the refill loop runs. A
// bucket whose mapping was closed is skipped, never touched.
func TestRefillLoopSurvivesWorkerRemovalAndShutdown(t *testing.T) {
	workers := make([]*simWorker, 8)
	for i := range workers {
		workers[i] = &simWorker{uid: "worker-" + strconv.Itoa(i), upLimit: 50, demand: 1}
	}
	sim := newClosedLoopSim(t, workers...)
	// A real clock: the loops below run concurrently with the test.
	sim.controller.nowFunc = time.Now
	var mu sync.Mutex
	removed := map[string]bool{}
	snapshots := sim.controller.workerInfoFn
	sim.controller.workerInfoFn = func() map[string]*WorkerInfoSnapshot {
		mu.Lock()
		defer mu.Unlock()
		current := snapshots()
		for uid := range removed {
			delete(current, uid)
		}
		return current
	}
	sim.device.metrics = map[string]*api.GPUUsageMetrics{simDeviceUUID: {ComputePercentage: 40}}

	stop := make(chan struct{})
	sim.controller.loops.Add(2)
	go func() {
		defer sim.controller.loops.Done()
		for {
			select {
			case <-stop:
				return
			default:
				sim.controller.refillBuckets()
			}
		}
	}()
	go func() {
		defer sim.controller.loops.Done()
		for {
			select {
			case <-stop:
				return
			default:
				sim.controller.updateERLControllers()
			}
		}
	}()

	// Remove workers the way the worker controller does: forget the worker,
	// close its mapping, then drop its quota state.
	for _, w := range workers[:6] {
		time.Sleep(2 * time.Millisecond)
		mu.Lock()
		removed[w.uid] = true
		mu.Unlock()
		if err := w.handle.Close(); err != nil {
			t.Fatal(err)
		}
		sim.controller.CleanupWorker(w.uid)
	}
	time.Sleep(5 * time.Millisecond)
	close(stop)
	sim.controller.loops.Wait()

	sim.controller.erlStatesMu.Lock()
	remaining := len(sim.controller.buckets)
	sim.controller.erlStatesMu.Unlock()
	if remaining != 2 {
		t.Errorf("%d buckets left for 2 remaining workers", remaining)
	}
	for _, bucket := range *sim.controller.refillSet.Load() {
		for _, w := range workers[:6] {
			if bucket.handle == w.handle {
				t.Errorf("the refill loop still serves removed worker %s", w.uid)
			}
		}
	}
}

// A hypervisor restart or upgrade must be invisible to a workload that is paced
// correctly: while the hypervisor is away the limiter keeps the bucket filled
// at the published rate, and the new control loop adopts that rate instead of
// starting the worker over from the initial one.
func TestHypervisorRestartIsInvisibleToASteadyWorkload(t *testing.T) {
	for _, share := range []uint32{25, 75} {
		t.Run(strconv.Itoa(int(share))+"%", func(t *testing.T) {
			worker := smallLaunchWorker("worker", share)
			worker.selfRefill = true
			sim := newPacingSim(t, worker)
			sim.run(30 * time.Second)
			// Until the first control interval nobody refills a new bucket, so the
			// limiter does. Once the hypervisor refills, it is the only one.
			selfRefilled := worker.selfRefilled
			sim.run(30 * time.Second)
			limit := float64(share) / 100
			before := worker.meanUtil(40)
			assertNear(t, "utilization before the restart", before, limit, simUtilTolerance)
			rate := worker.info(t).GetERLTokenRefillRate()
			if worker.selfRefilled != selfRefilled {
				t.Fatalf("the limiter refilled %.0f tokens while the hypervisor was running", worker.selfRefilled-selfRefilled)
			}

			// The hypervisor is gone for three seconds, then comes back empty.
			restartAt := sim.now
			intervalsBefore := len(worker.history)
			sim.hypervisorDown = true
			sim.run(3 * time.Second)
			if worker.selfRefilled == selfRefilled {
				t.Fatal("the limiter did not take over refilling")
			}
			sim.hypervisorDown = false
			sim.startController()
			sim.run(erlUpdateInterval + 2*pacedResolution)
			if got := worker.info(t).GetERLTokenRefillRate(); math.Abs(got-rate) > 0.01*rate {
				t.Errorf("after the restart the refill rate is %.0f tokens/s, it was %.0f", got, rate)
			}
			sim.run(40 * time.Second)

			// Throughput in every two-second window across the outage and while
			// the new loop leaves the adopted rate alone is what it was before;
			// afterwards the worker is controlled to its share as always.
			history := worker.history[intervalsBefore:]
			for i := 0; i+4 <= len(history); i += 4 {
				window := (history[i] + history[i+1] + history[i+2] + history[i+3]) / 4
				want, tolerance := before, 0.03
				if i > 6+adoptionHoldIntervals {
					want, tolerance = limit, 0.045
				}
				if math.Abs(window-want) > tolerance {
					t.Errorf("%.0f s after the hypervisor went away the worker used %.3f of the device, before %.3f",
						float64(i)*erlUpdateInterval.Seconds(), window, before)
				}
			}
			// One hiccup of the limiter's take-over delay, no stall.
			if gap := worker.maxGap(restartAt); gap > 150*time.Millisecond {
				t.Errorf("longest wait between two launches across the restart was %v", gap)
			}
			if got := worker.info(t).GetERLTokenRefillRate(); math.Abs(got-rate) > 0.15*rate {
				t.Errorf("40 s after the restart the refill rate is %.0f tokens/s, it was %.0f", got, rate)
			}
		})
	}
}

// Without anything to adopt (a new worker, or a mapping no control loop has
// maintained) the loop starts from the initial rate as before.
func TestControlLoopAdoptsOnlyAPublishedRate(t *testing.T) {
	for _, tc := range []struct {
		name           string
		rate, capacity float64
		wantAdopted    bool
	}{
		{name: "pair published by a control loop", rate: 4000, capacity: 4000 * burstWindow, wantAdopted: true},
		{name: "pair at the capacity floor", rate: 5, capacity: capacityMin, wantAdopted: true},
		// An upgrade changes how the capacity follows from the rate, not the rate
		// that is right for the worker.
		{name: "pair published by the previous release", rate: 4000, capacity: 2000, wantAdopted: true},
		{name: "pair at the previous release's capacity floor", rate: 300, capacity: 200, wantAdopted: true},
		{name: "pair at the ceilings the previous chart shipped", rate: 12000, capacity: 6000, wantAdopted: true},
		{name: "pair at the capacity floor the previous chart shipped", rate: 100, capacity: 150, wantAdopted: true},
		{name: "pair of an earlier build at its capacity floor", rate: 50, capacity: 20, wantAdopted: true},
		{name: "freshly created mapping", rate: 10, capacity: 100},
		{name: "capacity that no control loop derives from the rate", rate: 4000, capacity: 300},
		{name: "rate above the ceiling", rate: 10 * rateMax, capacity: capacityMax},
		{name: "rate below the floor", rate: rateMin / 2, capacity: capacityMin},
		{name: "rate that is not a number", rate: math.NaN(), capacity: 400},
		{name: "infinite rate", rate: math.Inf(1), capacity: 400},
	} {
		t.Run(tc.name, func(t *testing.T) {
			worker := smallLaunchWorker("worker", 50)
			worker.paused = true
			sim := newPacingSim(t, worker)
			info := worker.info(t)
			info.SetERLTokenRefillRate(tc.rate)
			info.SetERLTokenCapacity(tc.capacity)
			sim.run(erlUpdateInterval + 2*pacedResolution)

			es := sim.controller.erlStates["worker:gpu-sim"]
			if es == nil {
				t.Fatal("no control state for the worker")
			}
			if adopted := es.holdIntervals > 0; adopted != tc.wantAdopted {
				t.Fatalf("adopted = %v, want %v (rate now %.1f)", adopted, tc.wantAdopted, es.currentRate)
			}
			if tc.wantAdopted && es.currentRate != tc.rate {
				t.Errorf("adopted rate = %.1f, want %.1f", es.currentRate, tc.rate)
			}
			if got := info.GetERLTokenRefillRate(); math.IsNaN(got) || got < rateMin || got > rateMax {
				t.Errorf("published rate = %v, outside the bounds", got)
			}
		})
	}
}

// A limiter that finds nobody refilling adds tokens itself. Those are not in
// the hypervisor's account, so the interval in which that happened yields no
// consumption figure at all, rather than one that is too low or negative.
func TestMeterDiscardsIntervalsRefilledBySomebodyElse(t *testing.T) {
	worker := smallLaunchWorker("worker", 50)
	worker.selfRefill = true
	sim := newPacingSim(t, worker)
	sim.run(60 * time.Second)
	meter := sim.controller.tokenMeters["worker:gpu-sim"]
	if !meter.measured || meter.consumption <= 0 {
		t.Fatalf("consumption = %v (measured=%v) in steady state", meter.consumption, meter.measured)
	}
	steady, selfRefilled := meter.consumption, worker.selfRefilled

	// The refill loop stalls for 300 ms inside one control interval: the
	// limiter takes over after 100 ms and hands back afterwards.
	sim.run(50 * time.Millisecond)
	sim.refillStalled = true
	sim.run(300 * time.Millisecond)
	sim.refillStalled = false
	if worker.selfRefilled == selfRefilled {
		t.Fatal("the limiter did not refill during the stall")
	}
	// The control interval that contains the stall ends with the next look.
	sim.run(erlUpdateInterval - 350*time.Millisecond + 2*pacedResolution)
	if meter.measured {
		t.Errorf("consumption %.0f tokens/s was measured over an interval somebody else refilled in", meter.consumption)
	}
	// It is measured again from the next clean interval, at the old figure.
	sim.run(3 * erlUpdateInterval)
	if !meter.measured || math.Abs(meter.consumption-steady) > 0.15*steady {
		t.Errorf("consumption = %.0f tokens/s (measured=%v) after the hand-back, before %.0f",
			meter.consumption, meter.measured, steady)
	}
	// Nothing was credited twice: the worker stayed at its share throughout.
	assertNear(t, "utilization across the hand-over", worker.meanUtil(8), 0.50, simUtilTolerance)
}

// A workload that spends far more tokens at full speed than the previous
// ceiling of 200000 tokens/s must still get its share, also a high one.
func TestERLClosedLoopWorkloadFarAboveTheOldCeiling(t *testing.T) {
	const tokensPerSecondAtFullUse = 2_000_000
	for _, share := range []uint32{25, 90} {
		t.Run(strconv.Itoa(int(share))+"%", func(t *testing.T) {
			worker := &pacedWorker{
				uid: "worker", upLimit: share,
				launchCost: tokensPerSecondAtFullUse / 1000, launchTime: time.Millisecond,
			}
			sim := newPacingSim(t, worker)
			limit := float64(share) / 100

			// From a cold start: how long until the worker gets nine tenths of its share.
			reached := time.Duration(-1)
			for elapsed := time.Duration(0); elapsed < 120*time.Second; elapsed += erlUpdateInterval {
				sim.run(erlUpdateInterval)
				if reached < 0 && worker.meanUtil(2) >= 0.9*limit {
					reached = elapsed + erlUpdateInterval
				}
			}
			if reached < 0 || reached > 20*time.Second {
				t.Errorf("the worker reached nine tenths of its share after %v", reached)
			}
			t.Logf("cold start to 90%% of a %d%% share at %d tokens/s full-speed: %v", share, tokensPerSecondAtFullUse, reached)

			measureFrom := sim.now
			sim.run(60 * time.Second)
			assertNear(t, "mean utilization", worker.meanUtil(120), limit, simUtilTolerance)
			if rate := worker.info(t).GetERLTokenRefillRate(); rate >= rateMax {
				t.Errorf("refill rate ended at the ceiling of %.0f tokens/s", rateMax)
			}
			if gap := worker.maxGap(measureFrom); gap > 5*erlRefillInterval {
				t.Errorf("longest wait between two launches was %v", gap)
			}
		})
	}
}

// A worker that goes idle keeps the rate that was right for it. When it
// resumes it runs at its share at once, instead of unthrottled from a rate that
// climbed to the ceiling while it had nothing to do.
func TestIdleWorkerResumesAtItsShare(t *testing.T) {
	worker := smallLaunchWorker("worker", 25)
	sim := newPacingSim(t, worker)
	sim.run(60 * time.Second)
	rate := worker.info(t).GetERLTokenRefillRate()

	worker.paused = true
	sim.run(120 * time.Second)
	if got := worker.info(t).GetERLTokenRefillRate(); got != rate {
		t.Errorf("refill rate moved from %.0f to %.0f tokens/s while the worker was idle", rate, got)
	}

	worker.paused = false
	intervals := len(worker.history)
	sim.run(10 * time.Second)
	// The bucket filled up while idle: half a second of work at full speed is
	// the burst the capacity allows. After that the worker is at its share.
	resumed := worker.history[intervals:]
	for i := 2; i < len(resumed); i++ {
		if resumed[i] > 0.25+0.08 {
			t.Errorf("%.1f s after resuming the worker used %.2f of the device",
				float64(i)*erlUpdateInterval.Seconds(), resumed[i])
		}
	}
	assertNear(t, "utilization after resuming", worker.meanUtil(16), 0.25, simUtilTolerance)
}

// A worker that never ran anything keeps its initial rate while it idles: the
// rate only rises for a worker it is holding back. Its first burst therefore
// starts throttled and climbs to its share like any cold start, without ever
// running unthrottled.
func TestFirstBurstAfterIdleStartNeverRunsUnthrottled(t *testing.T) {
	worker := smallLaunchWorker("worker", 25)
	worker.paused = true
	sim := newPacingSim(t, worker)
	sim.run(60 * time.Second)
	if idle := worker.info(t).GetERLTokenRefillRate(); idle > 200 {
		t.Fatalf("the rate climbed to %.0f tokens/s while the worker had nothing to run", idle)
	}

	worker.paused = false
	intervals := len(worker.history)
	sim.run(30 * time.Second)
	burst := worker.history[intervals:]
	peak, reached := 0.0, -1
	for i, used := range burst {
		peak = math.Max(peak, used)
		if reached < 0 && used >= 0.9*0.25 {
			reached = i
		}
	}
	if peak > 0.25+0.08 {
		t.Errorf("the first burst peaked at %.2f of the device", peak)
	}
	if reached < 0 || float64(reached)*erlUpdateInterval.Seconds() > 15 {
		t.Errorf("the first burst reached its share after %d intervals", reached)
	}
	t.Logf("first burst of a worker that never ran: at nine tenths of its share after %.1f s, peak %.2f",
		float64(reached)*erlUpdateInterval.Seconds(), peak)
	assertNear(t, "utilization after the first burst", worker.meanUtil(20), 0.25, simUtilTolerance)
}
