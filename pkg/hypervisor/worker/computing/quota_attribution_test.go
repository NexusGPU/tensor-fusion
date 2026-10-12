package computing

import (
	"errors"
	"math"
	"strconv"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
)

func TestTokenMeterMeasuresOnlyWhatTheWorkerSpent(t *testing.T) {
	meter := &tokenMeter{}

	// Nothing is known before the hypervisor has left a level behind.
	meter.observe(100, 0, 10)
	if meter.measured {
		t.Fatal("consumption must be unknown before the first full interval")
	}
	meter.level, meter.updatedAt = 100, 10

	meter.observe(40, 0, 10.5)
	if !meter.measured || meter.consumption != 120 {
		t.Fatalf("consumption = %v (measured=%v), want 120 tokens/s", meter.consumption, meter.measured)
	}

	// The refill is the hypervisor's own doing: the level is recorded after it.
	meter.level, meter.updatedAt = 90, 10.5
	meter.observe(90, 0, 11)
	if want := (1 - consumptionAlpha) * 120; meter.consumption != want {
		t.Fatalf("idle interval: consumption = %v, want smoothed %v", meter.consumption, want)
	}

	// Tokens are added many times between two looks. The meter is told how many,
	// and what is missing from the previous level plus those was spent.
	meter = &tokenMeter{level: 100, updatedAt: 20}
	meter.observe(50, 250, 20.5)
	if !meter.measured || meter.consumption != 600 {
		t.Fatalf("with refills in between: consumption = %v (measured=%v), want 600 tokens/s",
			meter.consumption, meter.measured)
	}
	if meter.level != 50 || meter.updatedAt != 20.5 {
		t.Fatalf("the look itself must be recorded, got level %v at %v", meter.level, meter.updatedAt)
	}
	// What the hypervisor removed (a drained surplus) was not spent.
	meter = &tokenMeter{level: 400, updatedAt: 20}
	meter.observe(300, -100, 20.5)
	if !meter.measured || meter.consumption != 0 {
		t.Fatalf("after a drain: consumption = %v (measured=%v), want a measured 0", meter.consumption, meter.measured)
	}
	meter = &tokenMeter{level: 100, updatedAt: 20}
	meter.observe(50, math.NaN(), 20.5)
	if meter.measured {
		t.Fatal("an invalid refill account must not produce a measurement")
	}
	meter = &tokenMeter{level: 90, updatedAt: 11, measured: true, consumption: 60}

	// A level above the recorded one is never negative consumption.
	meter.level, meter.updatedAt = 90, 11
	meter.observe(500, 0, 11.5)
	if meter.consumption < 0 {
		t.Fatalf("consumption = %v, must not be negative", meter.consumption)
	}

	// An idle worker eventually reads as exactly idle.
	for i := range 60 {
		meter.level, meter.updatedAt = 90, 12+float64(i)/2
		meter.observe(90, 0, 12.5+float64(i)/2)
	}
	if meter.consumption != 0 {
		t.Fatalf("consumption = %v after a long idle period, want 0", meter.consumption)
	}

	for name, tokens := range map[string]float64{
		"NaN": math.NaN(), "infinite": math.Inf(1), "negative infinite": math.Inf(-1),
	} {
		meter.level, meter.updatedAt, meter.measured, meter.consumption = 90, 50, true, 7
		meter.observe(tokens, 0, 50.5)
		if meter.measured || meter.consumption != 0 {
			t.Fatalf("%s token level must not produce a measurement", name)
		}
	}
	// A recorded level that is not a number says nothing either, and must not
	// be divided into a NaN that later comparisons silently accept.
	meter.level, meter.updatedAt, meter.measured, meter.consumption = math.NaN(), 50, true, 7
	meter.observe(10, 0, 50.5)
	if meter.measured || meter.consumption != 0 {
		t.Fatal("an invalid recorded level must not produce a measurement")
	}

	// A balance below zero is debt: the worker spent what the bucket held and
	// what it now owes.
	meter = &tokenMeter{level: 50, updatedAt: 60}
	meter.observe(-200, 0, 60.5)
	if !meter.measured || meter.consumption != 500 {
		t.Fatalf("spending into debt: consumption = %v (measured=%v), want 500 tokens/s", meter.consumption, meter.measured)
	}
	// Repaying debt is the hypervisor's refill, not consumption.
	meter = &tokenMeter{level: -200, updatedAt: 60.5}
	meter.observe(-200, 0, 61)
	if !meter.measured || meter.consumption != 0 {
		t.Fatalf("idle in debt: consumption = %v (measured=%v), want a measured 0", meter.consumption, meter.measured)
	}
	// Borrowing further while already in debt.
	meter = &tokenMeter{level: -150, updatedAt: 61}
	meter.observe(-250, 0, 61.5)
	if !meter.measured || meter.consumption != 200 {
		t.Fatalf("borrowing more: consumption = %v (measured=%v), want 200 tokens/s", meter.consumption, meter.measured)
	}

	// After a long gap the interval no longer describes current behavior.
	meter.level, meter.updatedAt, meter.measured, meter.consumption = 90, 50, true, 7
	meter.observe(0, 0, 50+maxTokenUpdateGap)
	if meter.measured || meter.consumption != 0 {
		t.Fatal("a stale interval must not produce a measurement")
	}
}

func TestProcessUtilizationSnapshotCoverage(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	// A negative percentage stands for a process the provider marked as "not
	// supplied" (api.ProcessInformation.ComputeUtilizationUnavailable).
	const unsupplied = -1000.0
	build := func(percents map[string][]float64) *ProcessUtilizationSnapshot {
		processes := make([]api.ProcessInformation, 0, 16)
		pid := 0
		for device, values := range percents {
			for _, v := range values {
				pid++
				process := api.ProcessInformation{ProcessID: strconv.Itoa(pid), DeviceUUID: device}
				if v == unsupplied {
					process.ComputeUtilizationUnavailable = true
				} else {
					process.ComputeUtilizationPercent = v
				}
				processes = append(processes, process)
			}
		}
		return NewProcessUtilizationSnapshot(now, processes)
	}

	snapshot := build(map[string][]float64{
		"GPU-Zero":       {0, 0},                   // every process measured idle
		"GPU-Busy":       {0, 35},                  // measured
		"GPU-Unsupplied": {unsupplied, unsupplied}, // the provider could not supply it
		"GPU-Partial":    {20, unsupplied},         // one missing value spoils the device sample
		"GPU-NaN":        {20, math.NaN()},         // implausible
		"GPU-Neg":        {20, -1},                 // implausible
		"GPU-Over":       {20, 100.5},              // implausible
		"GPU-Foreign":    {60},                     // only a process that belongs to no worker
		"GPU-Empty":      nil,                      // no process at all
		"GPU-Full":       {100, 0, 0.5, 12.5},      // boundaries are valid
	})
	for device, want := range map[string]bool{
		"gpu-zero": true, "gpu-busy": true, "gpu-unsupplied": false, "gpu-partial": false, "gpu-nan": false,
		"gpu-neg": false, "gpu-over": false, "gpu-foreign": true, "gpu-empty": false, "gpu-full": true,
		"gpu-unknown": false,
	} {
		if got := snapshot.device(device, now) != nil; got != want {
			t.Errorf("device %s covered = %v, want %v", device, got, want)
		}
	}
	// The total of a covered device is what all of its processes report
	// together; it is what the figures are checked against the device with.
	for device, want := range map[string]float64{
		"gpu-zero": 0, "gpu-busy": 35, "gpu-partial": 0, "gpu-nan": 0, "gpu-foreign": 60, "gpu-full": 113,
	} {
		if got := snapshot.Totals[device]; got != want {
			t.Errorf("device %s total = %v, want %v", device, got, want)
		}
	}

	// Attribution is summed per worker, case-insensitively, on covered devices only.
	snapshot.AddWorkerProcess("worker-a", "GPU-Busy", 0)
	snapshot.AddWorkerProcess("worker-b", "gpu-busy", 20)
	snapshot.AddWorkerProcess("worker-b", "GPU-BUSY", 15)
	snapshot.AddWorkerProcess("worker-a", "GPU-Unsupplied", 0)
	busy := snapshot.device("gpu-busy", now)
	if v, ok := busy["worker-a"]; !ok || v != 0 {
		t.Errorf("worker-a = %v (present=%v), want an attributed 0", v, ok)
	}
	if busy["worker-b"] != 35 {
		t.Errorf("worker-b = %v, want 35", busy["worker-b"])
	}
	if snapshot.device("gpu-unsupplied", now) != nil {
		t.Error("attributing a process must not make an unreported device covered")
	}
	// A foreign process keeps workers at zero instead of charging them for it.
	if foreign := snapshot.device("gpu-foreign", now); foreign == nil || len(foreign) != 0 {
		t.Errorf("gpu-foreign = %v, want covered and empty", foreign)
	}

	// Freshness.
	if snapshot.device("gpu-busy", now.Add(processUtilizationMaxAge)) == nil {
		t.Error("a snapshot at the age limit must still be used")
	}
	if snapshot.device("gpu-busy", now.Add(processUtilizationMaxAge+time.Millisecond)) != nil {
		t.Error("a stale snapshot must be ignored")
	}
	if snapshot.device("gpu-busy", now.Add(-100*time.Millisecond)) == nil {
		t.Error("a snapshot taken just after the tick started must still be used")
	}
	var missing *ProcessUtilizationSnapshot
	missing.AddWorkerProcess("worker-a", "GPU-Busy", 10)
	if missing.device("gpu-busy", now) != nil {
		t.Error("a missing snapshot covers nothing")
	}
}

func TestAttributeDeviceUtilization(t *testing.T) {
	type member struct {
		uid         string
		consumption float64
		unmeasured  bool
	}
	type want struct {
		util   float64
		source utilizationSource
	}
	for _, tc := range []struct {
		name          string
		members       []member
		deviceUtil    float64
		deviceMissing bool
		processUtil   map[string]float64
		want          []want
	}{
		{
			name:       "sole worker owns the device utilization",
			members:    []member{{uid: "a", unmeasured: true}},
			deviceUtil: 0.6,
			want:       []want{{0.6, utilizationFromDevice}},
		},
		{
			name:          "sole worker without telemetry holds",
			members:       []member{{uid: "a", consumption: 100}},
			deviceMissing: true,
			want:          []want{{0, utilizationUnknown}},
		},
		{
			name:       "device utilization is split by tokens spent",
			members:    []member{{uid: "a", consumption: 100}, {uid: "b", consumption: 300}},
			deviceUtil: 0.8,
			want:       []want{{0.2, utilizationFromTokenShare}, {0.6, utilizationFromTokenShare}},
		},
		{
			name:       "an idle worker is not charged for its neighbor",
			members:    []member{{uid: "a"}, {uid: "b", consumption: 300}},
			deviceUtil: 0.7,
			want:       []want{{0, utilizationFromTokenShare}, {0.7, utilizationFromTokenShare}},
		},
		{
			name:       "nobody spent tokens: the load is not theirs",
			members:    []member{{uid: "a"}, {uid: "b"}},
			deviceUtil: 0.9,
			want:       []want{{0, utilizationFromTokenShare}, {0, utilizationFromTokenShare}},
		},
		{
			// What the unmeasured worker ran would otherwise be billed to its
			// neighbor.
			name:       "a worker without a measured interval makes the split unknown for the device",
			members:    []member{{uid: "a", unmeasured: true}, {uid: "b", consumption: 300}},
			deviceUtil: 0.5,
			want:       []want{{0, utilizationUnknown}, {0, utilizationUnknown}},
		},
		{
			name:        "per-process utilization does not depend on the neighbor being measured",
			members:     []member{{uid: "a", unmeasured: true}, {uid: "b", consumption: 300}},
			deviceUtil:  0.5,
			processUtil: map[string]float64{"a": 10, "b": 40},
			want:        []want{{0.1, utilizationFromProcesses}, {0.4, utilizationFromProcesses}},
		},
		{
			name:          "shared device without telemetry holds everyone",
			members:       []member{{uid: "a", consumption: 100}, {uid: "b", consumption: 300}},
			deviceMissing: true,
			want:          []want{{0, utilizationUnknown}, {0, utilizationUnknown}},
		},
		{
			name:        "per-process utilization wins over the token split",
			members:     []member{{uid: "a", consumption: 100}, {uid: "b", consumption: 300}},
			deviceUtil:  0.8,
			processUtil: map[string]float64{"a": 55, "b": 25},
			want:        []want{{0.55, utilizationFromProcesses}, {0.25, utilizationFromProcesses}},
		},
		{
			name:          "per-process utilization does not need device telemetry",
			members:       []member{{uid: "a", consumption: 100}, {uid: "b", consumption: 300}},
			deviceMissing: true,
			processUtil:   map[string]float64{"a": 55, "b": 25},
			want:          []want{{0.55, utilizationFromProcesses}, {0.25, utilizationFromProcesses}},
		},
		{
			name:        "per-process: a worker without processes that spends nothing is idle",
			members:     []member{{uid: "a"}, {uid: "b", consumption: 300}},
			deviceUtil:  0.9,
			processUtil: map[string]float64{"b": 40},
			want:        []want{{0, utilizationFromProcesses}, {0.4, utilizationFromProcesses}},
		},
		{
			name:        "per-process: a spending worker without attributed processes uses its tokens",
			members:     []member{{uid: "a", consumption: 100}, {uid: "b", consumption: 300}},
			deviceUtil:  0.8,
			processUtil: map[string]float64{"b": 60},
			want:        []want{{0.2, utilizationFromTokenShare}, {0.6, utilizationFromProcesses}},
		},
		{
			name:        "per-process sums above 100% are clamped",
			members:     []member{{uid: "a", consumption: 100}},
			deviceUtil:  1,
			processUtil: map[string]float64{"a": 140},
			want:        []want{{1, utilizationFromProcesses}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			members := make([]*erlMember, len(tc.members))
			for i, m := range tc.members {
				members[i] = &erlMember{workerUID: m.uid, meter: &tokenMeter{
					consumption: m.consumption, measured: !m.unmeasured,
				}}
			}
			attributeDeviceUtilization(members, tc.deviceUtil, !tc.deviceMissing, tc.processUtil, false, 0)
			total := 0.0
			for i, m := range members {
				if m.source != tc.want[i].source || math.Abs(m.utilization-tc.want[i].util) > 1e-9 {
					t.Errorf("%s: got %.3f from %q, want %.3f from %q",
						m.workerUID, m.utilization, m.source, tc.want[i].util, tc.want[i].source)
				}
				total += m.utilization
			}
			// Without per-process data nobody can be charged more than the device ran.
			if tc.processUtil == nil && total > tc.deviceUtil+1e-9 {
				t.Errorf("attributed %.3f in total, device only ran %.3f", total, tc.deviceUtil)
			}
		})
	}
}

// reportProcessUtilization makes the simulation report every worker's true
// utilization per process, as a provider with per-process telemetry would.
// report can rewrite the value reported for a worker.
func reportProcessUtilization(sim *closedLoopSim, report func(w *simWorker, actual float64) float64) {
	sim.controller.SetProcessUtilizationProvider(func() *ProcessUtilizationSnapshot {
		processes := make([]api.ProcessInformation, 0, len(sim.workers)+1)
		for i, w := range sim.workers {
			actual := 0.0
			if len(w.history) > 0 {
				actual = w.history[len(w.history)-1]
			}
			if report != nil {
				actual = report(w, actual)
			}
			processes = append(processes, api.ProcessInformation{
				ProcessID: strconv.Itoa(1000 + i), DeviceUUID: simDeviceUUID,
				ComputeUtilizationPercent: math.Round(actual * 100),
			})
		}
		if sim.externalUtil > 0 {
			processes = append(processes, api.ProcessInformation{
				ProcessID: "1", DeviceUUID: simDeviceUUID,
				ComputeUtilizationPercent: math.Round(sim.externalUtil * 100),
			})
		}
		snapshot := NewProcessUtilizationSnapshot(sim.now, processes)
		for i, w := range sim.workers {
			snapshot.AddWorkerProcess(w.uid, simDeviceUUID, processes[i].ComputeUtilizationPercent)
		}
		return snapshot
	})
}

// A provider that reports zero for every process has not supplied per-process
// utilization. Treating those zeros as "idle" would lift every limit.
func TestERLClosedLoopIgnoresUnsuppliedProcessUtilization(t *testing.T) {
	small := &simWorker{uid: "small", upLimit: 30, demand: 1}
	large := &simWorker{uid: "large", upLimit: 70, demand: 1}
	sim := newClosedLoopSim(t, small, large)
	reportProcessUtilization(sim, func(*simWorker, float64) float64 { return 0 })

	sim.run(simConvergeIntervals)
	assertNear(t, "small worker", small.meanUtil(simMeasureIntervals), 0.30, simUtilTolerance)
	assertNear(t, "large worker", large.meanUtil(simMeasureIntervals), 0.70, simUtilTolerance)
}

// Workers whose kernels cost a different number of tokens per unit of
// utilization are where the token split is only an approximation. It must stay
// safe there (the device never exceeds the sum of the limits, nobody is
// starved), and per-process utilization must make every worker exact.
func TestERLClosedLoopUnequalTokenCost(t *testing.T) {
	newWorkers := func() (*simWorker, *simWorker) {
		return &simWorker{uid: "cheap", upLimit: 30, demand: 1},
			&simWorker{uid: "costly", upLimit: 60, demand: 1, utilPerToken: 3 * simUtilPerToken}
	}

	t.Run("token split keeps the device within the sum of limits", func(t *testing.T) {
		cheap, costly := newWorkers()
		sim := newClosedLoopSim(t, cheap, costly)
		sim.run(2 * simConvergeIntervals)

		total := cheap.meanUtil(simMeasureIntervals) + costly.meanUtil(simMeasureIntervals)
		assertNear(t, "device utilization", total, 0.90, 0.06)
		for _, w := range sim.workers {
			if rate := sim.refillRate(w); rate < simStarvedRate {
				t.Errorf("%s is pinned at refill rate %.1f", w.uid, rate)
			}
			if got := w.meanUtil(simMeasureIntervals); got < 0.05 {
				t.Errorf("%s is starved at %.3f of the device", w.uid, got)
			}
		}
	})

	t.Run("per-process utilization makes every worker exact", func(t *testing.T) {
		cheap, costly := newWorkers()
		sim := newClosedLoopSim(t, cheap, costly)
		reportProcessUtilization(sim, nil)
		sim.run(2 * simConvergeIntervals)

		assertNear(t, "cheap worker", cheap.meanUtil(simMeasureIntervals), 0.30, simUtilTolerance)
		assertNear(t, "costly worker", costly.meanUtil(simMeasureIntervals), 0.60, simUtilTolerance)
	})
}

// With per-process utilization a worker is not charged for load on its device
// that belongs to something else.
func TestERLClosedLoopProcessUtilizationExcludesForeignLoad(t *testing.T) {
	worker := &simWorker{uid: "worker", upLimit: 50, demand: 1}
	sim := newClosedLoopSim(t, worker)
	sim.externalUtil = 0.30
	reportProcessUtilization(sim, nil)

	sim.run(simConvergeIntervals)
	assertNear(t, "worker next to foreign load", worker.meanUtil(simMeasureIntervals), 0.50, simUtilTolerance)
}

// Per-process attribution can miss a worker's processes. As long as that worker
// spends tokens it must stay limited rather than be taken for idle.
func TestERLClosedLoopUnattributedWorkerStaysLimited(t *testing.T) {
	seen := &simWorker{uid: "seen", upLimit: 60, demand: 1}
	unseen := &simWorker{uid: "unseen", upLimit: 30, demand: 1}
	sim := newClosedLoopSim(t, seen, unseen)
	sim.controller.SetProcessUtilizationProvider(func() *ProcessUtilizationSnapshot {
		percent := math.Round(seen.history[len(seen.history)-1] * 100)
		snapshot := NewProcessUtilizationSnapshot(sim.now, []api.ProcessInformation{
			{ProcessID: "1000", DeviceUUID: simDeviceUUID, ComputeUtilizationPercent: percent},
		})
		snapshot.AddWorkerProcess(seen.uid, simDeviceUUID, percent)
		return snapshot
	})

	sim.run(2 * simConvergeIntervals)
	assertNear(t, "attributed worker", seen.meanUtil(simMeasureIntervals), 0.60, simUtilTolerance)
	if got := unseen.meanUtil(simMeasureIntervals); got > 0.30+simUtilTolerance {
		t.Errorf("unattributed worker uses %.3f of the device, limit is 0.30", got)
	}
	if rate := sim.refillRate(unseen); rate >= rateMax {
		t.Errorf("unattributed worker is unthrottled (refill rate %.0f)", rate)
	}
}

// Telemetry loss on a shared device: every worker keeps the share it had.
func TestERLSharedDeviceKeepsSharesWithoutTelemetry(t *testing.T) {
	small := &simWorker{uid: "small", upLimit: 30, demand: 1}
	large := &simWorker{uid: "large", upLimit: 60, demand: 1}
	sim := newClosedLoopSim(t, small, large)
	sim.run(simConvergeIntervals)
	smallBefore, largeBefore := small.meanUtil(simMeasureIntervals), large.meanUtil(simMeasureIntervals)
	smallRate, largeRate := sim.refillRate(small), sim.refillRate(large)

	sim.telemetry = func(float64) (map[string]*api.GPUUsageMetrics, error) {
		return nil, errors.New("driver unavailable")
	}
	sim.run(simMeasureIntervals)

	assertNear(t, "small worker during the outage", small.meanUtil(simMeasureIntervals), smallBefore, 0.02)
	assertNear(t, "large worker during the outage", large.meanUtil(simMeasureIntervals), largeBefore, 0.02)
	if sim.refillRate(small) != smallRate || sim.refillRate(large) != largeRate {
		t.Error("refill rates changed without telemetry")
	}
}

// One device without a valid sample must not stop rate control on the others,
// and its own workers must still be refilled.
func TestERLDeviceWithoutSampleDoesNotAffectOtherDevices(t *testing.T) {
	now := time.Unix(1_800_000_000, 0)
	device := &quotaMetricsDeviceController{}
	controller := NewQuotaController(device, nil).(*Controller)
	controller.nowFunc = func() time.Time { return now }

	handles := map[string]*workerstate.SharedMemoryHandle{}
	snapshots := map[string]*WorkerInfoSnapshot{}
	for uid, deviceUUID := range map[string]string{"on-failed": "GPU-failed", "on-healthy": "GPU-healthy"} {
		handle, err := workerstate.CreateSharedMemoryHandle(t.TempDir(),
			&workerstate.PodIdentifier{Namespace: "test", Name: uid},
			[]workerstate.DeviceConfig{{DeviceIdx: 0, DeviceUUID: deviceUUID, UpLimit: 50}})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = handle.Close() })
		handles[uid] = handle
		snapshots[uid] = &WorkerInfoSnapshot{Devices: []DeviceSnapshot{{DeviceUUID: deviceUUID, DeviceIdx: 0, UpLimit: 50}}}
	}
	controller.workerInfoFn = func() map[string]*WorkerInfoSnapshot { return snapshots }
	controller.shmHandleFn = func(uid string) *workerstate.SharedMemoryHandle { return handles[uid] }
	info := func(uid string) *workerstate.SharedDeviceInfoV2 {
		return &handles[uid].GetState().V2.Devices[0].DeviceInfo
	}

	device.metrics = map[string]*api.GPUUsageMetrics{
		"GPU-failed": {ComputePercentage: 50}, "GPU-healthy": {ComputePercentage: 50},
	}
	controller.updateERLControllers()
	failedRate, healthyRate := info("on-failed").GetERLTokenRefillRate(), info("on-healthy").GetERLTokenRefillRate()

	// Only the healthy device keeps reporting. Its worker spends every token it
	// gets and still shows no utilization: it is held back by its rate.
	device.metrics = map[string]*api.GPUUsageMetrics{
		"GPU-failed": {ComputePercentage: math.NaN()}, "GPU-healthy": {ComputePercentage: 0},
	}
	for range 4 {
		info("on-failed").SetERLCurrentTokens(0) // the workload keeps spending everything
		info("on-healthy").SetERLCurrentTokens(0)
		now = now.Add(erlUpdateInterval)
		controller.updateERLControllers()

		want := math.Min(info("on-failed").GetERLTokenCapacity(), failedRate*erlUpdateInterval.Seconds())
		if got := info("on-failed").GetERLCurrentTokens(); got != want {
			t.Fatalf("bucket on the failed device holds %.1f tokens, want it refilled to %.1f", got, want)
		}
	}
	if got := info("on-failed").GetERLTokenRefillRate(); got != failedRate {
		t.Errorf("rate on the failed device changed from %.1f to %.1f", failedRate, got)
	}
	if got := info("on-healthy").GetERLTokenRefillRate(); got <= healthyRate {
		t.Errorf("rate on the healthy device did not adapt (%.1f -> %.1f)", healthyRate, got)
	}
}

func TestCleanupWorkerDropsTokenAccounting(t *testing.T) {
	worker := &simWorker{uid: "worker", upLimit: 50, demand: 1}
	other := &simWorker{uid: "other", upLimit: 50, demand: 1}
	sim := newClosedLoopSim(t, worker, other)
	sim.run(4)
	if len(sim.controller.tokenMeters) != 2 || len(sim.controller.erlStates) != 2 {
		t.Fatalf("expected state for both workers, got %d meters and %d rate states",
			len(sim.controller.tokenMeters), len(sim.controller.erlStates))
	}

	sim.controller.CleanupWorker(worker.uid)
	for key := range sim.controller.tokenMeters {
		if key != "other:gpu-sim" {
			t.Errorf("token accounting for %s survived cleanup", key)
		}
	}
	if len(sim.controller.tokenMeters) != 1 || len(sim.controller.erlStates) != 1 {
		t.Errorf("cleanup must only drop the removed worker's state")
	}
}
