package computing

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"k8s.io/klog/v2"
)

// WorkerInfoSnapshot holds the worker info needed for ERL updates
type WorkerInfoSnapshot struct {
	Namespace  string
	WorkerName string
	Devices    []DeviceSnapshot
}

// DeviceSnapshot holds per-device info for a worker
type DeviceSnapshot struct {
	DeviceUUID string
	DeviceIdx  int
	UpLimit    uint32
	// FoundUpLimit is the compute limit the device's slot carried when the
	// worker's mapping was opened, before the current one was written to it:
	// the limit that whatever a control loop left in the mapping was published
	// for. Zero when it is not known.
	FoundUpLimit uint32
	// BurstWindow is the worker's own burst window in seconds, nil when it has
	// none and the pool's applies. It is read in every control interval, so a
	// change takes effect within one. See burstWindow.
	BurstWindow *float64
}

type Controller struct {
	deviceController framework.DeviceController
	backend          framework.Backend

	mu      sync.RWMutex
	running bool
	stopCh  chan struct{}

	workerInfoFn  func() map[string]*WorkerInfoSnapshot
	shmHandleFn   func(workerUID string) *workerstate.SharedMemoryHandle
	processUtilFn func() *ProcessUtilizationSnapshot

	// Per-worker per-device AIMD state: key = workerUID + ":" + deviceUUID.
	// Guarded by erlStatesMu; the ERL update loop and the worker-delete callback
	// can race otherwise.
	erlStatesMu sync.Mutex
	erlStates   map[string]*erlState
	// Token accounting per bucket, same key and lock as erlStates.
	tokenMeters map[string]*tokenMeter
	// Per device (lower-case UUID), how the per-process figures have compared
	// with the device's utilization; same lock as erlStates.
	processChecks map[string]*processCheck
	// Workers removed recently, by UID, and when; same lock as erlStates.
	removedWorkers map[string]time.Time
	// Per bucket, same key and lock as erlStates: the calibrations of
	// utilization readings against token spending, and the utilization signal
	// used in the previous interval.
	calibrations map[string]*workerCalibration
	utilSources  map[string]utilizationSource
	// Latest attribution per bucket, see quota_utilization.go.
	attributed map[string]attributedUtilization
	// Buckets by the same key and under the same lock, and the ones the refill
	// loop currently serves: those of the workers seen by the latest control
	// interval. The slice is replaced, never modified.
	buckets   map[string]*refillBucket
	refillSet atomic.Pointer[[]*refillBucket]
	// loops tracks the control and refill goroutines of a started limiter.
	loops  sync.WaitGroup
	config erlConfig

	nowFunc func() time.Time

	// podEventFn records a Warning Event on a Pod, nil when there is nowhere to
	// record one. It is called from the control loop and must not wait.
	podEventFn func(namespace, podName, podUID, reason, message string)
	// When the Event about a bucket that is left out of the split was last
	// raised, by worker UID; same lock as erlStates.
	exclusionEvents map[string]time.Time
}

func NewQuotaController(
	deviceController framework.DeviceController,
	backend framework.Backend,
) framework.QuotaController {
	c := &Controller{
		deviceController: deviceController,
		backend:          backend,
		stopCh:           make(chan struct{}),
		erlStates:        make(map[string]*erlState),
		tokenMeters:      make(map[string]*tokenMeter),
		config:           defaultERLConfig(),

		processChecks:  make(map[string]*processCheck),
		removedWorkers: make(map[string]time.Time),
		calibrations:   make(map[string]*workerCalibration),
		utilSources:    make(map[string]utilizationSource),
		buckets:        make(map[string]*refillBucket),
		nowFunc:        time.Now,

		exclusionEvents: make(map[string]time.Time),
	}
	if emitter, ok := backend.(PodEventEmitter); ok {
		c.SetPodEventEmitter(emitter)
	}
	return c
}

func (c *Controller) SetWorkerInfoProvider(fn func() map[string]*WorkerInfoSnapshot) {
	c.workerInfoFn = fn
}

func (c *Controller) SetShmHandleProvider(fn func(workerUID string) *workerstate.SharedMemoryHandle) {
	c.shmHandleFn = fn
}

// SetProcessUtilizationProvider wires the optional per-process utilization
// source. Without it, or while it has nothing usable for a device, utilization
// is derived from token consumption and device metrics.
func (c *Controller) SetProcessUtilizationProvider(fn func() *ProcessUtilizationSnapshot) {
	c.processUtilFn = fn
}

func (c *Controller) SetQuota(workerUID string) error {
	return nil
}

func (c *Controller) StartSoftQuotaLimiter() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.running {
		return nil
	}

	vendor := c.deviceController.GetAcceleratorVendor()
	if !constants.SupportsSoftIsolation(vendor) {
		return fmt.Errorf("soft isolation mode is not supported for vendor: %s", vendor)
	}
	c.config = loadERLConfigFromEnv()

	c.running = true
	c.stopCh = make(chan struct{})

	c.loops.Add(2)
	go c.runERLUpdateLoop(c.stopCh)
	go c.runERLRefillLoop(c.stopCh)
	klog.Infof(
		"Soft quota limiter started (feedback-controlled ERL), effective configuration: "+
			"rate=[%g, %g] burstWindow=%g cap=[%g, %g] "+
			"alpha=%g kp=%g ki=%g kd=%g decay=%g",
		c.config.rateMin, c.config.rateMax,
		c.config.burstWindow,
		c.config.capacityMin, c.config.capacityMax,
		c.config.utilAlpha,
		c.config.kp, c.config.ki, c.config.kd,
		c.config.integralDecayFactor,
	)
	return nil
}

func (c *Controller) StopSoftQuotaLimiter() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.running {
		return nil
	}
	close(c.stopCh)
	c.running = false
	// Both loops have left the mappings alone once this returns.
	c.loops.Wait()
	c.refillSet.Store(nil)
	klog.Info("Soft quota limiter stopped")
	return nil
}

func (c *Controller) GetWorkerQuotaStatus(workerUID string) error {
	return nil
}

func (c *Controller) runERLUpdateLoop(stopCh <-chan struct{}) {
	defer c.loops.Done()
	ticker := time.NewTicker(erlUpdateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-stopCh:
			return
		case <-ticker.C:
			c.updateERLControllers()
		}
	}
}

func (c *Controller) nowSecs() float64 {
	return float64(c.nowFunc().UnixMicro()) / 1e6
}

// dropBucketState forgets what the control loop keeps for a bucket it could not
// reach, unless the entry has been replaced in the meantime.
func (c *Controller) dropBucketState(key string, bucket *refillBucket) {
	c.erlStatesMu.Lock()
	defer c.erlStatesMu.Unlock()
	if c.buckets[key] != bucket {
		return
	}
	delete(c.buckets, key)
	delete(c.tokenMeters, key)
	delete(c.calibrations, key)
}

// getOrCreateBucket returns the bucket of a worker's device. The caller binds
// it to the worker's current mapping under the bucket's own lock.
func (c *Controller) getOrCreateBucket(
	key string, handle *workerstate.SharedMemoryHandle, deviceIdx int,
) *refillBucket {
	c.erlStatesMu.Lock()
	defer c.erlStatesMu.Unlock()
	bucket, ok := c.buckets[key]
	if !ok {
		bucket = newRefillBucket(handle, deviceIdx)
		c.buckets[key] = bucket
	}
	return bucket
}

// workerRemoved reports whether the worker was cleaned up recently. A control
// pass works on a list of workers taken at its start; one removed since must
// not get its state back.
func (c *Controller) workerRemoved(workerUID string) bool {
	c.erlStatesMu.Lock()
	defer c.erlStatesMu.Unlock()
	_, removed := c.removedWorkers[workerUID]
	return removed
}

func (c *Controller) getOrCreateERLState(key string) (state *erlState, created bool) {
	c.erlStatesMu.Lock()
	defer c.erlStatesMu.Unlock()
	st, ok := c.erlStates[key]
	created = !ok
	if !ok {
		st = &erlState{
			currentRate:  initialRate, // Start low to avoid immediate overshoot.
			smoothedUtil: 0,
			integralErr:  0,
			lastError:    0,
			initialized:  false,
		}
		c.erlStates[key] = st
	}
	return st, created
}

func (c *Controller) getOrCreateTokenMeter(key string) *tokenMeter {
	c.erlStatesMu.Lock()
	defer c.erlStatesMu.Unlock()
	meter, ok := c.tokenMeters[key]
	if !ok {
		meter = &tokenMeter{}
		c.tokenMeters[key] = meter
	}
	return meter
}

func (c *Controller) getOrCreateCalibration(key string) *workerCalibration {
	c.erlStatesMu.Lock()
	defer c.erlStatesMu.Unlock()
	calibration, ok := c.calibrations[key]
	if !ok {
		calibration = &workerCalibration{processes: utilizationCalibration{coarse: true}}
		c.calibrations[key] = calibration
	}
	return calibration
}

// swapUtilizationSource records the signal used for a bucket in this interval
// and returns the one used in the previous interval ("" if none).
func (c *Controller) swapUtilizationSource(key string, source utilizationSource) utilizationSource {
	c.erlStatesMu.Lock()
	defer c.erlStatesMu.Unlock()
	previous := c.utilSources[key]
	c.utilSources[key] = source
	return previous
}

// CleanupWorker drops everything the control loop keeps for the given worker,
// so that nothing of it outlives the worker. The worker's mapping is not
// touched: it is the worker's own and goes away with it. Safe to call with an
// empty workerUID (no-op) or before the soft limiter has started.
func (c *Controller) CleanupWorker(workerUID string) {
	if workerUID == "" {
		return
	}

	prefix := workerUID + ":"
	now := c.nowFunc()
	c.erlStatesMu.Lock()
	defer c.erlStatesMu.Unlock()
	for uid, removedAt := range c.removedWorkers {
		if now.Sub(removedAt) > removedWorkerMemory {
			delete(c.removedWorkers, uid)
		}
	}
	c.removedWorkers[workerUID] = now
	delete(c.exclusionEvents, workerUID)
	for key := range c.erlStates {
		if strings.HasPrefix(key, prefix) {
			delete(c.erlStates, key)
		}
	}
	for key := range c.tokenMeters {
		if strings.HasPrefix(key, prefix) {
			delete(c.tokenMeters, key)
		}
	}
	for key := range c.calibrations {
		if strings.HasPrefix(key, prefix) {
			delete(c.calibrations, key)
		}
	}
	for key := range c.utilSources {
		if strings.HasPrefix(key, prefix) {
			delete(c.utilSources, key)
		}
	}
	for key := range c.attributed {
		if strings.HasPrefix(key, prefix) {
			delete(c.attributed, key)
		}
	}
	removed := make(map[*refillBucket]bool)
	for key, bucket := range c.buckets {
		if strings.HasPrefix(key, prefix) {
			removed[bucket] = true
			delete(c.buckets, key)
		}
	}
	// Stop refilling the worker's buckets now rather than at the next control
	// interval. A pass that is already running is harmless: it only reaches the
	// mapping through its handle, which refuses once closed.
	if served := c.refillSet.Load(); served != nil && len(removed) > 0 {
		remaining := make([]*refillBucket, 0, len(*served))
		for _, bucket := range *served {
			if !removed[bucket] {
				remaining = append(remaining, bucket)
			}
		}
		c.refillSet.Store(&remaining)
	}
}

func (c *Controller) updateERLControllers() {
	if c.workerInfoFn == nil || c.shmHandleFn == nil {
		return
	}

	workers := c.workerInfoFn()
	if len(workers) == 0 {
		c.refillSet.Store(nil)
		return
	}

	now := c.nowFunc()
	dt := erlUpdateInterval.Seconds()

	// Missing telemetry is not an idle device: affected workers keep their
	// rate and keep receiving tokens at it.
	deviceUtilization := c.sampleDeviceUtilization()
	var processUtil *ProcessUtilizationSnapshot
	if c.processUtilFn != nil {
		processUtil = c.processUtilFn()
	}

	// Measure what every worker spent first, so each device can be arbitrated
	// across all of its workers.
	membersByDevice := make(map[string][]*erlMember)
	served := make([]*refillBucket, 0, len(workers))
	for workerUID, workerInfo := range workers {
		handle := c.shmHandleFn(workerUID)
		if handle == nil {
			continue
		}
		for _, dev := range workerInfo.Devices {
			if c.workerRemoved(workerUID) {
				break
			}
			deviceUUID := strings.ToLower(dev.DeviceUUID)
			member := &erlMember{
				workerUID: workerUID,
				stateKey:  workerUID + ":" + deviceUUID,
				device:    dev,
			}
			member.bucket = c.getOrCreateBucket(member.stateKey, handle, dev.DeviceIdx)
			member.meter = c.getOrCreateTokenMeter(member.stateKey)
			calibration := c.getOrCreateCalibration(member.stateKey)
			member.calibration, member.fallback = &calibration.processes, &calibration.device

			present, excludedBefore := false, member.meter.excluded
			member.bucket.mu.Lock()
			if member.bucket.handle != handle || member.bucket.deviceIdx != dev.DeviceIdx {
				// The mapping was republished: its bucket starts over, and nothing
				// measured on the previous one describes it.
				member.bucket.handle, member.bucket.deviceIdx, member.bucket.added = handle, dev.DeviceIdx, 0
				*member.meter = tokenMeter{}
			}
			handle.WithState(func(state *workerstate.SharedDeviceState) {
				if state.V2 == nil || !state.HasDevice(dev.DeviceIdx) {
					return
				}
				present = true
				tokens := state.V2.Devices[dev.DeviceIdx].DeviceInfo.GetERLCurrentTokens()
				// The clock is read here, not at the start of the pass: everything
				// before this point can take time.
				if member.bucket.foreign {
					// Somebody else added tokens during this interval, at the
					// published rate if it was a limiter.
					member.meter.foreignRate = math.Max(member.bucket.rate, consumptionFloor)
					// A foreign timestamp is only taken for one of a writer that
					// is refilling while it is no more than clockStepTolerance
					// ahead of the clock (see refillInfo), so a limiter's refill
					// is seen after a gap of at least the difference.
					member.meter.foreignExpected = member.bucket.ownGap >= limiterSelfRefillAfter-clockStepTolerance
					member.bucket.foreign = false
				}
				member.bucket.ownGap = 0
				member.meter.observe(tokens, member.bucket.added, c.nowSecs())
				member.bucket.added = 0
				member.tokenBound = member.bucket.capacity <= 0 || tokens < tokenBoundFraction*member.bucket.capacity ||
					member.bucket.drained
				member.bucket.drained = false
			})
			member.bucket.mu.Unlock()
			if !present {
				// Closed or not carrying this device: the worker is going away or
				// is not set up yet. Keep nothing for it.
				c.dropBucketState(member.stateKey, member.bucket)
				continue
			}
			c.reportMeterExclusion(member, workerInfo, excludedBefore)
			membersByDevice[deviceUUID] = append(membersByDevice[deviceUUID], member)
			served = append(served, member.bucket)
		}
	}
	// From here on the refill loop serves exactly the buckets of this interval.
	c.refillSet.Store(&served)

	for deviceUUID, members := range membersByDevice {
		deviceUtil, deviceSampled := deviceUtilization[deviceUUID]
		attributeDeviceUtilization(members, deviceUtil, deviceSampled,
			c.processUtilizationFor(deviceUUID, processUtil, now, deviceUtil, deviceSampled),
			processUtil.unmetered(deviceUUID, now), c.processShortfall(deviceUUID))

		for _, member := range members {
			member.bucket.mu.Lock()
			member.bucket.handle.WithState(func(state *workerstate.SharedDeviceState) {
				if state.V2 == nil || !state.HasDevice(member.device.DeviceIdx) {
					return
				}
				deviceInfo := &state.V2.Devices[member.device.DeviceIdx].DeviceInfo
				c.updateERLBucket(member, deviceInfo, deviceUtil, dt)
			})
			member.bucket.mu.Unlock()
		}
	}
}
