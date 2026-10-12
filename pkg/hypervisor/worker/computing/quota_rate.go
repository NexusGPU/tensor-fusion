package computing

import (
	"math"

	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"k8s.io/klog/v2"
)

// erlState tracks per-device rate control state
type erlState struct {
	currentRate  float64
	smoothedUtil float64
	integralErr  float64
	lastError    float64
	initialized  bool
	// holdIntervals counts down the control intervals during which an adopted
	// rate is not adapted (see adoptionHoldIntervals).
	holdIntervals int
	// adopted says that the rate was adopted from one a control loop had
	// learned and that the worker's signal has not been re-established since;
	// settled counts the intervals in a row that qualify for that, and coarse
	// says that per-process readings have taken part in the worker's figure
	// since the adoption. See adoptionHoldIntervals and signalReestablished.
	adopted bool
	settled int
	coarse  bool

	// limit is the compute limit (percent) the rate was last decided for, 0
	// before the first interval. probe counts down the intervals in which what
	// the worker spends unthrottled is measured after it became limited, and
	// probeSpent says that it spent tokens in the previous one. See followLimit.
	limit      uint32
	probe      int
	probeSpent bool
	// settling counts down the intervals after the limit moved the rate in
	// which utilization readings still describe the time before, and are kept
	// out of the calibrations (limitSettleIntervals).
	settling int

	// debtBound is the most the worker could owe by the rates and capacities
	// published since its balance was last not negative; see debtBound.
	debtBound float64

	// window is the burst window (seconds) the published capacity was last
	// derived with, and windowHeld which capacity bound decided instead, if
	// one did; both only to report a change. See burstWindow.
	window     float64
	windowHeld string

	// Lowering the rate only makes sense while it lowers what the worker uses.
	// It does not when the utilization that is taken for the worker's is not
	// paid for with its tokens: load of a process on the same device that does
	// not go through a limiter, or work that a limiter of an earlier release
	// admits after a fixed wait whatever the balance. The rate would then be
	// lowered all the way to the floor, to no effect on that load and at the
	// expense of everything the worker does pay for.
	//
	// cutFrom is the rate before the first step of the current run of lowering
	// (0: none), cutSpend and cutRead the highest average spending on the device
	// and the highest average load of the device since.
	// See the futility constants for the test and what follows from it.
	//
	// What this does not cover: load that is not paid for and smaller than
	// about two thirds of what the paying workers use is not told apart, and
	// they lose that much; a device that such load fills leaves nothing to tell
	// it by; and a worker that starts next to such load has no rate of its own
	// to go back to, only the one it started from.
	spendAvg, readAvg          float64
	averaged                   bool
	cutFrom, cutSpend, cutRead float64
	cutQuiet, futile           int
	effective, futileHolds     int
	capped                     int
	// What the worker's calibrations had learned before the run began: what
	// they learn during a run that turns out futile is how much load a token
	// does not pay for.
	cutDevice, cutProcesses calibrationSnapshot
	// futileHold says that the current hold follows such a run.
	futileHold bool

	// heavyGain is the most a token bought the worker (utilization per token/s)
	// in the kinds of work seen recently, and heavyAge the intervals since it
	// did. When the work turns cheaper and then comes back, it alternates:
	// following it each way lets the worker run above its share every time the
	// expensive kind returns, for the seconds it takes to notice. For
	// alternating intervals the rate is therefore the one the expensive kind
	// needs. The first return is still followed late, the cheap kind then runs
	// below the share, and work that changes once is followed as before.
	heavyGain   float64
	heavyAge    int
	cheaper     bool
	alternating int
}

// publishedDerivation is how a control loop derives the capacity it publishes
// from the rate: rate x burstWindow within [capacityMin, capacityMax], with the
// rate within [rateMin, rateMax].
type publishedDerivation struct {
	burstWindow, capacityMin, capacityMax, rateMin, rateMax float64
}

// earlierDerivations are the ones of the control loops a hypervisor may be
// upgraded from: the compiled defaults before the capacity floor was lowered,
// the compiled defaults of the previous release, and the values its Helm chart
// shipped as overrides.
var earlierDerivations = []publishedDerivation{
	{burstWindow: 0.1, capacityMin: 20, capacityMax: 10000000, rateMin: 10, rateMax: 10000000},
	{burstWindow: 0.5, capacityMin: 200, capacityMax: 200000, rateMin: 10, rateMax: 200000},
	{burstWindow: 0.5, capacityMin: 150, capacityMax: 6000, rateMin: 80, rateMax: 12000},
}

// newMappingRate and newMappingCapacity are what a mapping carries before any
// control loop has published to it (workerstate.NewSharedDeviceInfoV2).
const (
	newMappingRate     = 10.0
	newMappingCapacity = 100.0
)

func (d publishedDerivation) published(rate, capacity float64) bool {
	if rate < d.rateMin || rate > d.rateMax {
		return false
	}
	derived := clampFloat64(rate*d.burstWindow, d.capacityMin, d.capacityMax)
	return math.Abs(capacity-derived) <= 1e-6*derived
}

// adoptPublishedRate seeds the state of a bucket this control loop sees for the
// first time from the mapping. A bucket that a control loop was already
// maintaining (before a hypervisor restart or upgrade) carries the rate that was
// right for its worker; starting over from the initial rate would starve a
// correctly paced workload and take many intervals to climb back. The pair in
// the mapping is taken for published by a control loop when the capacity is the
// one this configuration derives from the rate, or the one an earlier release
// derived (see earlierDerivations: an upgrade changes how the capacity follows
// from the rate, not the rate that is right for the worker); a freshly created
// mapping satisfies none of them and starts from the initial rate. The adopted
// rate is brought within this configuration's bounds. Everything else starts
// neutral.
//
// This configuration derives the capacity with the worker's burst window
// (window), or with the pool's when the worker's own was set or removed while
// no control loop was running. A new mapping is told by its values first: with
// a window of ten seconds they are a pair this configuration derives.
func (c *Controller) adoptPublishedRate(
	es *erlState, key string, deviceInfo *workerstate.SharedDeviceInfoV2, window float64,
) {
	rate, capacity := deviceInfo.GetERLTokenRefillRate(), deviceInfo.GetERLTokenCapacity()
	if !isFinite(rate) || !isFinite(capacity) || (rate == newMappingRate && capacity == newMappingCapacity) {
		return
	}
	current := publishedDerivation{
		burstWindow: window, capacityMin: c.config.capacityMin, capacityMax: c.config.capacityMax,
		rateMin: c.config.rateMin, rateMax: c.config.rateMax,
	}
	adopted := current.published(rate, capacity)
	current.burstWindow = math.Max(c.config.burstWindow, burstWindowFloor)
	adopted = adopted || current.published(rate, capacity)
	for _, earlier := range earlierDerivations {
		adopted = adopted || earlier.published(rate, capacity)
	}
	if !adopted {
		return
	}
	es.currentRate, es.holdIntervals = clampFloat64(rate, c.config.rateMin, c.config.rateMax), adoptionHoldIntervals
	if rate != initialRate {
		es.adopted, es.holdIntervals = true, adoptionHoldMaxIntervals
	}
	klog.Infof("ERL [%s] adopted the published refill rate of %.4g tokens/s", key, rate)
}

// adoptedForAnotherLimit handles a rate that was adopted from a mapping whose
// limit has changed since it was published (device.FoundUpLimit, the limit the
// mapping carried when it was opened, differs from the worker's limit now: it
// was changed while no hypervisor was running). Such a rate was right for the
// limit it was published for. It is not held, and nothing is inherited from it
// when the worker's signal is re-established: the change is followed as one
// made while the control loop runs, see followLimit. A limit that is not known
// (zero) is taken for unchanged.
func (es *erlState) adoptedForAnotherLimit(key string, device DeviceSnapshot) {
	found := device.FoundUpLimit
	if es.holdIntervals == 0 || found == 0 || found == device.UpLimit {
		return
	}
	es.holdIntervals = 0
	if !es.adopted {
		// The rate every bucket starts from was not learned for any limit.
		return
	}
	es.limit = found
	klog.Infof("ERL [%s] the adopted refill rate was published for a limit of %d%%, the limit is %d%% now: not held",
		key, found, device.UpLimit)
}

// balanceReset is what a change of a worker's limit does to the balance of its
// bucket; see followLimit.
type balanceReset int

const (
	// balanceKept leaves the balance as it is.
	balanceKept balanceReset = iota
	// balanceWithoutDebt sets a balance below zero to zero. Debt is waited out
	// at the refill rate, and the wait it stood for was that of the limit it
	// was charged under; it is not carried over to another limit, and a worker
	// without a limit owes nothing.
	balanceWithoutDebt
	// balanceEmptied sets the balance to zero either way: what the bucket holds
	// at the end of a measurement was not saved up at the worker's rate.
	balanceEmptied
)

// followLimit makes the rate follow the worker's compute limit where the limit
// itself decides it, and reports whether it did (pinned: the rate is not
// adapted in this interval) and what becomes of the balance.
//
// A worker without a limit (100%) gets the highest rate. Its limiter spends no
// tokens, so nothing is measured or learned about it, and the rate cannot hold
// it back whatever a limiter does with it.
//
// A change from one limit to another is followed in the interval in which it
// is seen: the rate that was in force is scaled by new / old. What a token
// buys the worker does not depend on its limit, so the scaled rate is as right
// for the new limit as the rate was for the old one, and the control goes on
// from there with everything it has learned.
//
// A worker that gets a limit after having none has spent no tokens so far:
// what its work costs is not known, and the utilization that was measured is
// that of a worker nobody held back. Measured on a device: the rate it had
// (the initial one) was cut to the few tokens it had spent in the first
// fraction of a second, lowered further for as long as the average of its
// earlier utilization stayed above the new limit, and the calibration that
// started from those intervals took half a minute to find the rate; the
// worker ran at a few percent of its limit meanwhile. Instead, the worker
// stays at the highest rate until two intervals in a row show what it spends:
// the second is one in which it paid for everything it ran, unthrottled. With
// the utilization the device shows for it over the same time (its part by
// tokens next to other workers) that is what a token buys it, and the rate is
// the one its limit calls for: limit x spent / utilization. Its calibrations
// start from that figure with little weight. If there is nothing to measure
// for limitProbeIntervals (an idle worker, no device reading), it starts from
// the initial rate like a new worker.
func (c *Controller) followLimit(
	es *erlState, member *erlMember, targetUtil float64,
) (pinned bool, balance balanceReset) {
	limit, previous := member.device.UpLimit, es.limit
	es.limit = limit
	changed := previous != 0 && previous != limit
	if limit >= 100 {
		es.currentRate, es.probe, es.cutFrom = c.config.rateMax, 0, 0
		return true, balanceWithoutDebt
	}
	spent := 0.0
	if member.meter.measured {
		spent = member.meter.last
	}
	spending := spent > consumptionFloor
	if changed {
		// A rate set by the limit is not a run of lowering to test for effect.
		es.cutFrom = 0
		if previous >= 100 {
			es.currentRate, es.probe, es.probeSpent = c.config.rateMax, limitProbeIntervals, spending
			klog.Infof("ERL [%s] limit changed from none to %d%%: measuring what the worker spends unthrottled",
				member.stateKey, limit)
			return true, balanceWithoutDebt
		}
		scale, before := float64(limit)/float64(previous), es.currentRate
		es.currentRate = clampFloat64(before*scale, c.config.rateMin, c.config.rateMax)
		es.smoothedUtil, es.settling = es.smoothedUtil*scale, limitSettleIntervals
		if es.adopted && member.fallback != nil && !member.fallback.calibrated() {
			// The limit was changed while no control loop was running, and this
			// one has learned nothing yet. The adopted rate was the one the
			// previous limit called for, which says what a token buys: the
			// calibration of the device's utilization starts from that, with
			// the weight of its first few readings, so that the rate is not
			// steered by readings of the time before in the meantime. Nothing
			// is passed on to the calibration of per-process readings.
			*member.fallback = utilizationCalibration{
				utilization: targetUtil, consumption: es.currentRate,
				samples: calibrationMinSamples, evidence: calibrationMinEvidence,
			}
			es.smoothedUtil, es.initialized = targetUtil, true
		}
		klog.Infof("ERL [%s] limit changed from %d%% to %d%%: refill rate %.4g -> %.4g tokens/s",
			member.stateKey, previous, limit, before, es.currentRate)
		return true, balanceWithoutDebt
	}
	if es.probe == 0 {
		return false, balanceKept
	}
	if spending && es.probeSpent && member.deviceRead {
		// The worker's part of what the device shows, by the tokens spent in
		// this interval; all of it for a worker that has the device to itself.
		utilization := 0.0
		if total := member.deviceSpending - member.meter.window + spent; total > 0 {
			utilization = clampFloat64(member.deviceLoad*spent/total, 0, 1)
		}
		if utilization >= limitProbeMinUtil {
			es.currentRate = clampFloat64(targetUtil*spent/utilization, c.config.rateMin, c.config.rateMax)
			es.smoothedUtil, es.initialized, es.probe, es.settling = targetUtil, true, 0, limitSettleIntervals
			// What the calibrations saw so far relates the utilization of a
			// worker that was not held back to tokens it did not have to spend.
			if member.fallback != nil {
				*member.fallback = utilizationCalibration{
					utilization: utilization, consumption: spent,
					samples: calibrationMinSamples, evidence: calibrationMinEvidence,
				}
			}
			if member.calibration != nil {
				*member.calibration = utilizationCalibration{coarse: true}
			}
			klog.Infof("ERL [%s] limit %d%%: the worker spent %.4g tokens/s at %.0f%% of the device unthrottled: "+
				"refill rate %.4g tokens/s", member.stateKey, limit, spent, utilization*100, es.currentRate)
			return true, balanceEmptied
		}
	}
	es.probeSpent = spending
	if es.probe--; es.probe == 0 {
		es.currentRate = clampFloat64(initialRate, c.config.rateMin, c.config.rateMax)
		es.smoothedUtil, es.initialized = 0, false
		klog.Infof("ERL [%s] limit %d%%: nothing to measure what the worker spends by: starting from %.4g tokens/s",
			member.stateKey, limit, es.currentRate)
		return true, balanceEmptied
	}
	return true, balanceKept
}

// signalReestablished follows the signal of a worker whose rate was adopted,
// once per control interval in which the worker has a utilization figure, and
// reports when the rate may be adapted again; see adoptionHoldIntervals for the
// rule. It also notes whether per-process readings take part in the figure.
func (es *erlState) signalReestablished(member *erlMember) bool {
	es.coarse = es.coarse || member.coarse
	if !member.meter.measured || !member.calibrated {
		es.settled = 0
		return false
	}
	es.settled++
	return es.settled >= adoptionHoldIntervals
}

// burstWindowOf is the burst window (seconds) of a worker's bucket on a device:
// the worker's own when it has one, otherwise the pool's, within the bounds
// and never below what the refill loop needs. See burstWindow.
func (c *Controller) burstWindowOf(device DeviceSnapshot) float64 {
	window := c.config.burstWindow
	if device.BurstWindow != nil && isFinite(*device.BurstWindow) {
		window = *device.BurstWindow
	}
	return clampFloat64(window, burstWindowFloor, burstWindowMax)
}

func clampFloat64(v, lo, hi float64) float64 {
	return math.Min(math.Max(v, lo), hi)
}

func slewRate(current, target, upRatio, downRatio float64) float64 {
	if target > current {
		return math.Min(target, current*(1.0+upRatio))
	}
	return math.Max(target, current*(1.0-downRatio))
}

func computeDesiredRate(currentRate, targetUtil, smoothedUtil, dt float64, es *erlState, cfg erlConfig) float64 {
	if smoothedUtil <= idleUtilization {
		// Nothing measurable comes of the current rate. It is raised steadily;
		// for a worker whose share is far above what "nothing measurable" can
		// hide, by doubling, which finds the scale of a fast workload (millions
		// of tokens per second) in seconds. The caller only lets the rate rise
		// for a worker it is holding back.
		if targetUtil >= blindRampMinTarget && !es.adopted {
			return math.Min(currentRate*blindRampFactor, cfg.rateMax)
		}
		return math.Min(currentRate*(1.0+maxRateIncreaseRatio), cfg.rateMax)
	}

	error := targetUtil - smoothedUtil
	if math.Abs(error) < utilDeadband {
		es.integralErr *= cfg.integralDecayFactor
		return currentRate
	}

	es.integralErr = clampFloat64(es.integralErr*cfg.integralDecayFactor+error*dt, -integralClamp, integralClamp)
	derivative := 0.0
	if dt > 0 {
		derivative = (error - es.lastError) / dt
	}
	es.lastError = error

	// Feed-forward: estimate the rate required to move current utilization to target.
	feedForwardRate := currentRate * (targetUtil / math.Max(smoothedUtil, 0.05))
	controlFactor := 1.0 + cfg.kp*error + cfg.ki*es.integralErr + cfg.kd*derivative
	controlFactor = clampFloat64(controlFactor, 0.5, 1.5)

	desiredRate := clampFloat64(feedForwardRate*controlFactor, cfg.rateMin, cfg.rateMax)
	return slewRate(currentRate, desiredRate, maxRateIncreaseRatio, maxRateDecreaseRatio)
}

// bleedTokenBucket takes surplus burst budget out of the bucket: what exceeds
// the capacity, and, for a worker running above its target, what exceeds the
// reserve. It returns the resulting balance and how many tokens it removed.
// Removing tokens credits no time and needs no claim.
//
// A worker that runs above its target while its bucket is not empty is spending
// what it saved, which is what a burst window is for. The reserve is therefore
// counted from the top of the bucket: this takes no more than it takes from a
// bucket of the default window (bleedable is the capacity of that one at the
// worker's rate), and a bucket of that size or smaller is treated as before.
func bleedTokenBucket(
	deviceInfo *workerstate.SharedDeviceInfoV2, capacity, bleedable, targetUtil, smoothedUtil float64,
) (currentTokens, removed float64) {
	currentTokens = deviceInfo.GetERLCurrentTokens()
	reserveTarget := clampFloat64(capacity*tokenReserveRatio, 0.0, capacity)
	if capacity > bleedable {
		reserveTarget = capacity - bleedable*(1-tokenReserveRatio)
	}
	drainAmount := math.Max(tokenDrainMin, capacity*tokenDrainRatio) * (erlUpdateInterval.Seconds())
	switch {
	case currentTokens > capacity:
		removed = deviceInfo.DrainERLTokens(drainAmount, capacity)
	case smoothedUtil > targetUtil+utilDeadband && currentTokens > reserveTarget:
		// Smoothly bleed extra burst budget when utilization is already above target.
		removed = deviceInfo.DrainERLTokens(drainAmount, reserveTarget)
	default:
		return currentTokens, 0
	}
	return deviceInfo.GetERLCurrentTokens(), removed
}

// debtBound is the overdraft bound of a token bucket: the most a worker can owe.
// This comment is the whole rule; the limiter inside the workload and the
// hypervisor both follow it.
//
//	bound = max(capacity, rate * 60 s * 100 / up_limit)
//
// capacity (erl_token_capacity, tokens), rate (erl_token_refill_rate, tokens
// per second) and up_limit (percent, 1-99) are the values in the shared device
// state. rate * 100 / up_limit is what the worker spends per second when it has
// the whole device, so the bound is the cost of 60 s of work at full speed, and
// never less than one bucket.
//
// Limiter: work is admitted while the balance is above zero and is charged in
// full, except that a charge never takes the balance below -bound, with the
// three values read at the time of the charge; a balance that is already below
// -bound is left as it is. Nothing is admitted while the balance is not above
// zero, so a worker waits out what it owes at its refill rate: at the rate of
// the time of the charge, at most 60 s * 100 / up_limit.
//
// Hypervisor: a balance below -bound is raised to -bound and no further, where
// bound is the largest value the formula gave for any capacity and rate the
// hypervisor published since it last saw the balance at or above zero. The
// bound therefore never shrinks while debt is outstanding: debt that was within
// the bound when the limiter charged it is not forgiven because the rate was
// lowered afterwards. A balance below that bound is not debt a limiter can have
// charged; it counts as written by somebody else. The one time debt is not
// carried on is a change of the worker's limit: in the interval in which the
// hypervisor sees the change, a balance below zero is set to zero, and so is
// that of a worker without a limit (see balanceReset).
//
// A single piece of work that keeps the device busy for longer than 60 s is
// charged for 60 s. Its worker is locked out for 60 s * 100 / up_limit and so
// runs above its share: at (its length) / (60 s * 100 / up_limit) of the device.
func debtBound(capacity, rate float64, upLimit uint32) float64 {
	share := clampFloat64(float64(upLimit), 1, 100) / 100
	return math.Max(capacity, rate*maxTokenDebtSeconds/share)
}

// regimeGain returns what a token buys the worker (utilization per token/s) as
// far as the rate is concerned: the gain itself, or the gain of the most
// expensive kind of work seen recently while the work alternates; see
// erlState.heavyGain.
func (es *erlState) regimeGain(gain float64) float64 {
	switch {
	case gain >= es.heavyGain:
		// As expensive as it has been, or more. Getting here from a cheaper
		// kind of work is what alternating work does.
		if es.cheaper {
			es.alternating = regimeHoldIntervals
		}
		es.heavyGain, es.heavyAge, es.cheaper = gain, 0, false
	case gain*regimeChangeFactor < es.heavyGain:
		es.cheaper = true
		es.heavyAge++
	case gain*calibrationChangeFactor >= es.heavyGain:
		// The same kind of work: back to it, or still at it, in which case
		// the figure follows what is measured, slowly.
		if es.cheaper {
			es.alternating, es.cheaper = regimeHoldIntervals, false
		} else {
			es.heavyGain += regimeAlpha * (gain - es.heavyGain)
		}
		es.heavyAge = 0
	case es.cheaper:
		// In between: on the way back, or another kind of work altogether.
		es.heavyAge++
	}
	if es.alternating > 0 {
		es.alternating--
		return es.heavyGain
	}
	// Work that turned cheaper and stayed so is another workload: the
	// expensive kind is forgotten after as long as it would have been held.
	if es.heavyAge > regimeHoldIntervals {
		es.heavyGain, es.heavyAge, es.cheaper = gain, 0, false
	}
	return gain
}

// loweringIsFutile follows a run of lowering of the worker's rate, from
// rateBefore to es.currentRate in this interval, and reports when it has shown
// no effect on what the device does. It then holds the rate; the caller puts it
// back to es.cutFrom. See erlState.cutFrom.
//
// The test is on the device as a whole: what all of its limited workers spend
// against what it shows. Load that tokens pay for comes down in proportion when
// they are spent at a lower rate, whoever spends them; load that does not come
// down is not paid for with them.
func (es *erlState) loweringIsFutile(member *erlMember, rateBefore float64) bool {
	if !member.deviceRead || !member.meter.measured {
		return false
	}
	// A worker that owes more than a bucket is running, or waiting out, work it
	// paid for in one piece. What the device shows then is paid for, and comes
	// long after the spending: nothing is concluded from comparing the two.
	if member.meter.level < -member.bucket.capacity {
		es.cutFrom = 0
		return false
	}
	if !es.averaged {
		es.spendAvg, es.readAvg, es.averaged = member.deviceSpending, member.deviceLoad, true
	} else {
		es.spendAvg += futilityAlpha * (member.deviceSpending - es.spendAvg)
		es.readAvg += futilityAlpha * (member.deviceLoad - es.readAvg)
	}
	lowered := es.currentRate < rateBefore
	if lowered && es.cutFrom == 0 {
		es.cutFrom, es.cutSpend, es.cutRead = rateBefore, es.spendAvg, es.readAvg
		es.cutQuiet, es.futile, es.effective, es.capped = 0, 0, 0, 0
		es.cutDevice, es.cutProcesses = calibrationSnapshot{}, calibrationSnapshot{}
		if member.fallback != nil {
			es.cutDevice = member.fallback.snapshot()
		}
		if member.calibration != nil {
			es.cutProcesses = member.calibration.snapshot()
		}
	}
	switch {
	case es.cutFrom == 0:
		return false
	case lowered:
		es.cutQuiet = 0
	case es.currentRate >= es.cutFrom || es.cutQuiet >= futilityQuietIntervals:
		// Back where the lowering began, or settled below it: the worker was
		// found at or below its share, so lowering did have an effect.
		es.cutFrom, es.futileHolds = 0, 0
		return false
	default:
		es.cutQuiet++
	}
	if es.readAvg >= futilitySaturated {
		// A device that is full shows no less when a little less is run on it.
		// The run starts where it stops being full. The average load still
		// holds intervals in which the device was full, and the spending that
		// goes with it is not the average spending, which was cut off by
		// nothing: it is what buys that load at the latest interval's price,
		// if the device was not full in that one.
		es.cutFrom, es.cutSpend, es.cutRead = es.currentRate, es.spendAvg, es.readAvg
		if member.deviceLoad > 0 && member.deviceLoad < futilitySaturated {
			es.cutSpend = es.readAvg * member.deviceSpending / member.deviceLoad
		}
		es.futile, es.effective, es.capped = 0, 0, 0
		return false
	}
	// Until it is known whether lowering has an effect, it goes no further
	// than this. When nothing can be told at that depth for as long as a run
	// may stay quiet, lowering goes on from there as a new run.
	if lowest := es.cutFrom * futilityDepth; es.currentRate < lowest {
		es.currentRate = lowest
		if es.capped++; es.capped >= futilityQuietIntervals {
			es.cutFrom = 0
			return false
		}
	}
	// What was spent and what the device showed before the lowering took
	// effect: the highest of each since it began.
	es.cutSpend, es.cutRead = math.Max(es.cutSpend, es.spendAvg), math.Max(es.cutRead, es.readAvg)
	if es.cutSpend <= consumptionFloor || es.cutRead < futilityMinReading {
		return false
	}
	spending, reading := es.spendAvg/es.cutSpend, es.readAvg/es.cutRead
	switch {
	case spending > futilitySpending:
		es.futile, es.effective = 0, 0
		return false
	case reading-spending < futilityShortfall*(1-spending):
		// The load came down with the spending. The run is over; lowering
		// further starts a new one from here.
		es.futile = 0
		if es.effective++; es.effective >= futilityConfirmIntervals {
			es.cutFrom, es.futileHolds = 0, 0
		}
		return false
	}
	es.effective = 0
	if es.futile++; es.futile < futilityConfirmIntervals {
		return false
	}
	es.holdIntervals = futilityHoldIntervals << min(es.futileHolds, futilityHoldDoublings)
	es.futileHolds++
	es.futile, es.futileHold = 0, true
	if member.fallback != nil {
		member.fallback.restore(es.cutDevice)
	}
	if member.calibration != nil {
		member.calibration.restore(es.cutProcesses)
	}
	return true
}

// updateERLBucket adapts one worker's refill rate to its own utilization, hands
// the new rate and capacity to the refill loop and brings the bucket up to
// date. Without a utilization signal the rate is held, and the bucket keeps
// being refilled at it: while the hypervisor is alive, it is the limiter's only
// source of tokens. The caller holds the member's bucket lock.
func (c *Controller) updateERLBucket(
	member *erlMember, deviceInfo *workerstate.SharedDeviceInfoV2, deviceUtil, dt float64,
) {
	if c.workerRemoved(member.workerUID) {
		return
	}
	es, created := c.getOrCreateERLState(member.stateKey)
	window := c.burstWindowOf(member.device)
	if created {
		c.adoptPublishedRate(es, member.stateKey, deviceInfo, window)
		es.adoptedForAnotherLimit(member.stateKey, member.device)
	}
	targetUtil := float64(member.device.UpLimit) / 100.0
	pinned, balance := c.followLimit(es, member, targetUtil)
	rateBefore := es.currentRate

	// Burst budget is only bled on a fresh measurement.
	bleedUtil := targetUtil
	if member.source != utilizationUnknown {
		// EMA smooth utilization
		if !es.initialized {
			es.smoothedUtil = member.utilization
			es.initialized = true
		} else {
			es.smoothedUtil = c.config.utilAlpha*member.utilization + (1-c.config.utilAlpha)*es.smoothedUtil
		}

		es.endAdoptionHold(member, targetUtil)

		switch {
		case pinned:
			// The limit decided the rate in this interval; see followLimit.
		case es.holdIntervals > 0:
			// An adopted rate is left alone while the calibration settles, and
			// so is one that lowering had no effect on.
			es.holdIntervals--
		case member.calibrated && member.gain > 0:
			// What a token buys this worker is known, and with it the rate
			// that holds it at its share: the share divided by that. The rate
			// goes there, in bounded steps, instead of being steered by the
			// utilization measured in the last interval, which lags behind
			// every change of the rate, is bursty for work that comes in long
			// launches, and says nothing about the rate while the worker has
			// little to run. The right rate for a worker is then the same
			// whether it is busy or idle, so it neither climbs while the worker
			// idles nor has to be found again when it resumes.
			right := clampFloat64(targetUtil/es.regimeGain(member.gain), c.config.rateMin, c.config.rateMax)
			if math.Abs(right-es.currentRate) > calibratedRateDeadband*es.currentRate {
				// A worker that its own readings show to be held under its share
				// by the figure derived from the device, and that drains its
				// bucket, is one whose rate is still being found: it rises as
				// fast as the bound on what a token buys it falls, up to a
				// doubling per interval.
				increase := maxRateIncreaseRatio
				if member.held && member.tokenBound && !es.adopted {
					increase = blindRampFactor - 1
				}
				es.currentRate = slewRate(es.currentRate, right, increase, maxRateDecreaseRatio)
			}
			bleedUtil = es.smoothedUtil
		default:
			// Not calibrated yet: the rate follows the measured utilization. A
			// worker above its share cannot be held by a rate above what it
			// spends; the control starts from what it has recently been spending.
			if es.smoothedUtil > targetUtil+utilDeadband && member.meter.measured && member.meter.peak > 0 {
				es.currentRate = math.Min(es.currentRate, math.Max(member.meter.peak, c.config.rateMin))
			}
			desired := computeDesiredRate(es.currentRate, targetUtil, es.smoothedUtil, dt, es, c.config)
			// The rate only goes up for a worker it is holding back. One that
			// leaves tokens in its bucket has less to run than it may, which no
			// rate can change; raising it anyway would let the rate climb while
			// the worker idles and leave it unthrottled when it resumes.
			if desired > es.currentRate && !member.tokenBound {
				desired = es.currentRate
			}
			es.currentRate = desired
			bleedUtil = es.smoothedUtil
		}
		if !pinned && es.loweringIsFutile(member, rateBefore) {
			klog.Infof("ERL [%s] lowering the refill rate from %.4g to %.4g tokens/s did not lower what the device "+
				"shows for the worker: the load is not paid for with its tokens; back to %.4g for %d intervals",
				member.stateKey, es.cutFrom, es.currentRate, es.cutFrom, es.holdIntervals)
			es.currentRate, es.cutFrom = es.cutFrom, 0
		}
		c.recordAttributedUtilization(member.stateKey, member.utilization, c.nowFunc())
	} else {
		// An interval without a figure is not one of a run that re-establishes
		// the signal of an adopted worker.
		es.settled = 0
	}

	es.freezeCalibrations(member)

	newCapacity := c.bucketCapacity(es, member, window)

	// Publish the decision: to the limiter through the mapping, and to the
	// refill loop.
	deviceInfo.SetERLTokenRefillRate(es.currentRate)
	deviceInfo.SetERLTokenCapacity(newCapacity)
	member.bucket.rate, member.bucket.capacity = es.currentRate, newCapacity

	// Bring the bucket up to date. The refill loop has been doing that all
	// along, so only the moment since its last pass is left to add here, by a
	// clock read now.
	member.bucket.refillInfo(deviceInfo, c.nowSecs())
	bleedable := clampFloat64(es.currentRate*burstWindow, c.config.capacityMin, c.config.capacityMax)
	currentTokens, removed := bleedTokenBucket(deviceInfo, newCapacity, bleedable, targetUtil, bleedUtil)
	member.bucket.added -= removed
	// A balance below zero is what the worker owes for work it was admitted
	// for, and refills repay it. A balance below the overdraft bound is not
	// debt a limiter charged; it is brought up to the bound, no further.
	bound := debtBound(newCapacity, es.currentRate, member.device.UpLimit)
	currentTokens = balance.apply(member, deviceInfo, currentTokens)
	if currentTokens < 0 {
		bound = math.Max(bound, es.debtBound)
	}
	es.debtBound, member.meter.debtFloor = bound, bound
	if currentTokens < -bound {
		// The balance is stored, not added to: next to a balance that is far
		// off, the difference to the bound is not a number that can be added.
		// Nothing is lost by that: a limiter charges nothing while the balance
		// is below zero, and the refill loop is kept out by the bucket's lock.
		// For the same reason the amount is not accounted as added; the meter
		// continues from the balance as it is now.
		deviceInfo.SetERLCurrentTokens(-bound)
		currentTokens = -bound
		member.meter.level, member.bucket.added = currentTokens, 0
	}

	// A worker gaining or losing its utilization signal is worth a line; the
	// per-interval trace is for debugging only.
	previous := c.swapUtilizationSource(member.stateKey, member.source)
	if previous != "" && (previous == utilizationUnknown) != (member.source == utilizationUnknown) {
		if member.source == utilizationUnknown {
			klog.Infof("ERL [%s] lost its utilization signal (was %s): holding refill rate %.0f tokens/s at target %.0f%%",
				member.stateKey, previous, es.currentRate, targetUtil*100)
		} else {
			klog.Infof("ERL [%s] utilization signal is back (%s): resuming rate control at target %.0f%%",
				member.stateKey, member.source, targetUtil*100)
		}
	}

	if klog.V(4).Enabled() {
		klog.V(4).Infof(
			"ERL [%s] util=%.0f%% src=%s device=%.0f%% smooth=%.0f%% target=%.0f%% err=%.1f%% "+
				"rate=%.0f cap=%.0f tokens=%.1f spent=%.0f/s int=%.3f",
			member.stateKey,
			member.utilization*100, member.source, deviceUtil*100, es.smoothedUtil*100, targetUtil*100,
			(targetUtil-es.smoothedUtil)*100,
			es.currentRate, newCapacity, currentTokens, member.meter.consumption, es.integralErr,
		)
	}
}

// apply carries out the reset on the worker's bucket and returns the balance
// after it. The balance is stored and the meter continues from it, as for a
// balance below the overdraft bound.
func (r balanceReset) apply(
	member *erlMember, deviceInfo *workerstate.SharedDeviceInfoV2, currentTokens float64,
) float64 {
	if (r == balanceWithoutDebt && currentTokens < 0) || (r == balanceEmptied && currentTokens != 0) {
		deviceInfo.SetERLCurrentTokens(0)
		member.meter.level, member.bucket.added = 0, 0
		return 0
	}
	return currentTokens
}

// freezeCalibrations keeps the interval's readings out of the worker's
// calibrations while they do not describe it.
func (es *erlState) freezeCalibrations(member *erlMember) {
	// While the rate is held after lowering it had no effect, the readings
	// still describe load that is not the worker's.
	es.futileHold = es.futileHold && es.holdIntervals > 0
	// Nor do they describe the worker under its limit right after the limit
	// moved its rate.
	frozen := es.futileHold || es.settling > 0
	es.settling = max(es.settling-1, 0)
	if member.fallback != nil {
		member.fallback.frozen = frozen
	}
	if member.calibration != nil {
		member.calibration.frozen = frozen
	}
}

// bucketCapacity is the capacity of a worker's bucket at its current rate, and
// says in the log when the worker's effective burst window changes.
func (c *Controller) bucketCapacity(es *erlState, member *erlMember, window float64) float64 {
	// Dynamic capacity: the worker's burst window of its rate (see burstWindow).
	derived := es.currentRate * window
	newCapacity := clampFloat64(derived, c.config.capacityMin, c.config.capacityMax)
	held := ""
	switch {
	case newCapacity > derived:
		held = "capacityMin"
	case newCapacity < derived:
		held = overrideCapacityMax
	}
	if window != es.window || held != es.windowHeld {
		es.window, es.windowHeld = window, held
		if held == "" {
			klog.V(4).Infof("ERL [%s] burst window %.4g s in effect (pool default %.4g s): "+
				"the bucket holds %.4g tokens at %.4g tokens/s",
				member.stateKey, window, c.config.burstWindow, newCapacity, es.currentRate)
		} else {
			klog.V(4).Infof("ERL [%s] burst window %.4g s in effect, not the %.4g s asked for (pool default %.4g s): "+
				"%s holds the bucket at %.4g tokens at %.4g tokens/s",
				member.stateKey, newCapacity/es.currentRate, window, c.config.burstWindow, held, newCapacity, es.currentRate)
		}
	}
	return newCapacity
}

// endAdoptionHold releases an adopted rate once the worker's signal is
// re-established (see signalReestablished).
func (es *erlState) endAdoptionHold(member *erlMember, targetUtil float64) {
	if !es.adopted || !es.signalReestablished(member) {
		return
	}
	// The adopted rate has been held for long enough: it is adapted
	// from where it is, starting with this interval. A hold that
	// follows futile lowering is not this one and runs its course.
	es.adopted = false
	if es.holdIntervals > 0 && !es.futileHold {
		es.holdIntervals = 0
		// The rate is still the adopted one, and what the previous loop
		// knew about the worker's own readings is in it.
		if es.coarse {
			member.calibration.inherit(targetUtil / es.currentRate)
		}
	}
	klog.Infof("ERL [%s] signal re-established (%s): adapting the refill rate from %.4g tokens/s",
		member.stateKey, member.source, es.currentRate)
}
