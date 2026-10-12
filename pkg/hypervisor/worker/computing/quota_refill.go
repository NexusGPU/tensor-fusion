package computing

import (
	"math"
	"sync"
	"time"

	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
)

// refillBucket is one worker's token bucket on one device as the refill loop
// sees it. The control loop decides rate and capacity; both loops refill
// through it, and it keeps the account of what the hypervisor added for the
// worker's tokenMeter.
type refillBucket struct {
	// mu serializes the two loops on this bucket and guards every field. It is
	// always taken before entering the handle, never inside it.
	mu        sync.Mutex
	handle    *workerstate.SharedMemoryHandle
	deviceIdx int
	// rate (tokens/s) and capacity as last decided by the control loop; zero
	// until it has run for this bucket.
	rate, capacity float64
	// added is the net amount put into the bucket since the meter last looked.
	added float64

	// ownStamp is the refill timestamp this controller stored last (f64 bits, 0
	// before its first claim). Finding another one in the mapping means that
	// somebody else refilled in between: a limiter does when it finds nobody
	// refilling. Those tokens are not in added, so foreign marks the current
	// metering interval as one whose consumption cannot be told.
	ownStamp uint64
	foreign  bool
	// ownGap is the longest this controller went without refilling the bucket
	// since the meter last looked (seconds). A limiter refills by itself once
	// it finds the timestamp limiterSelfRefillAfter old, so after a gap of
	// that length a refill by somebody else is what the contract calls for.
	ownGap float64
	// aheadStamp is a refill timestamp that was found ahead of the clock, and
	// aheadPasses for how many passes in a row; see refillInfo.
	aheadStamp  uint64
	aheadPasses int
	// drained says that a refill since the control loop's previous look found
	// the bucket less than tokenBoundFraction full. The control loop looks twice
	// a second; a worker that takes all it is given in one go (a launch that
	// costs more than the bucket holds, admitted by a limiter that waited for
	// it) shows a refilled bucket at most of those looks although its rate is
	// what holds it back.
	drained bool

	// refillFn is refillState bound once, and nowSecs its argument, so that a
	// refill pass allocates nothing.
	refillFn func(*workerstate.SharedDeviceState)
	nowSecs  float64
}

func newRefillBucket(handle *workerstate.SharedMemoryHandle, deviceIdx int) *refillBucket {
	bucket := &refillBucket{handle: handle, deviceIdx: deviceIdx}
	bucket.refillFn = bucket.refillState
	return bucket
}

// refill adds what the bucket earned up to nowSecs. The mapping is only touched
// inside the handle, which keeps it mapped for the duration and contains a
// fault in it; a closed or faulted handle is skipped.
func (b *refillBucket) refill(nowSecs float64) {
	b.mu.Lock()
	b.nowSecs = nowSecs
	b.handle.WithState(b.refillFn)
	b.mu.Unlock()
}

func (b *refillBucket) refillState(state *workerstate.SharedDeviceState) {
	if b.rate <= 0 || state.V2 == nil || !state.HasDevice(b.deviceIdx) {
		return
	}
	b.refillInfo(&state.V2.Devices[b.deviceIdx].DeviceInfo, b.nowSecs)
}

// refillInfo is the one way this controller adds tokens: it claims the time
// since the bucket's last refill, whoever did that, and adds what was earned in
// it at the current rate. Tokens are never added without a successful claim.
// The caller holds b.mu.
//
// A timestamp ahead of the clock is left alone by the claim: it is normally
// that of a writer whose clock reading is a little newer, and moving it back
// would credit the time up to it twice. After the wall clock was set back it is
// ahead for as long as the step, and nobody would refill until the clock has
// caught up. The two differ in what happens next: a writer that is refilling
// stores a new timestamp within milliseconds, and no clock reading is held up
// for several passes. A timestamp that is still the same and still ahead by
// more than clockStepTolerance on clockStepPasses passes in a row is therefore
// brought to the clock, and nothing is credited for doing so: refilling
// resumes after a few refill intervals instead of after the length of the step.
func (b *refillBucket) refillInfo(info *workerstate.SharedDeviceInfoV2, nowSecs float64) {
	if b.ownStamp != 0 {
		b.ownGap = math.Max(b.ownGap, nowSecs-math.Float64frombits(b.ownStamp))
	}
	elapsed, found, claimed := info.ClaimERLRefill(nowSecs, maxTokenUpdateGap)
	if ahead := math.Float64frombits(found) - nowSecs; !claimed && ahead > clockStepTolerance {
		if found != b.aheadStamp {
			b.aheadStamp, b.aheadPasses = found, 0
		}
		if b.aheadPasses++; b.aheadPasses >= clockStepPasses && info.RestartERLRefillClock(found, nowSecs) {
			b.ownStamp, b.aheadStamp, b.aheadPasses = math.Float64bits(nowSecs), 0, 0
		}
		return
	}
	b.aheadStamp, b.aheadPasses = 0, 0
	if b.ownStamp != 0 && found != b.ownStamp {
		b.foreign = true
		// A timestamp older than the one stored here last was put back by
		// somebody: time already credited is not credited again.
		if own := math.Float64frombits(b.ownStamp); math.Float64frombits(found) < own {
			elapsed = math.Min(elapsed, math.Max(0, nowSecs-own))
		}
	}
	if !claimed {
		return
	}
	b.ownStamp = math.Float64bits(nowSecs)
	if info.GetERLCurrentTokens() < tokenBoundFraction*b.capacity {
		b.drained = true
	}
	b.added += info.AddERLTokens(b.rate*elapsed, b.capacity)
}

// runERLRefillLoop hands out tokens between control intervals. A late or
// skipped tick loses nothing: every refill adds what was earned since the
// previous one.
func (c *Controller) runERLRefillLoop(stopCh <-chan struct{}) {
	defer c.loops.Done()
	ticker := time.NewTicker(erlRefillInterval)
	defer ticker.Stop()

	for {
		select {
		case <-stopCh:
			return
		case <-ticker.C:
			c.refillBuckets()
		}
	}
}

// refillBuckets is one pass of the refill loop over the buckets the control
// loop last published. It holds no lock across the pass, only each bucket's own
// for the few atomic operations on its mapping, and it needs no telemetry: a
// worker whose rate is being held keeps receiving tokens at that rate.
func (c *Controller) refillBuckets() {
	buckets := c.refillSet.Load()
	if buckets == nil {
		return
	}
	nowSecs := c.nowSecs()
	for _, bucket := range *buckets {
		bucket.refill(nowSecs)
	}
}
