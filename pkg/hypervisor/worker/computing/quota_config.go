package computing

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"k8s.io/klog/v2"
)

const (
	// Two cadences. The control loop measures utilization and adapts each
	// worker's refill rate and bucket capacity; utilization telemetry does not
	// move faster than this. The refill loop only hands out tokens at the rate
	// the control loop decided, in small portions.
	//
	// Refilling at the control cadence would hand a throttled worker half a
	// second of tokens at once: at a 25% share it would run for about 125 ms and
	// then stall for 375 ms, twice a second. In portions of erlRefillInterval
	// the same share is a steady slowdown: a unit of work (a token of a served
	// stream, a training step) takes its unthrottled time divided by the share,
	// plus at most one refill interval of waiting for the next portion.
	//
	// Measured on an RTX 3090 with a streaming-shaped workload, the 99th
	// percentile of the gap between two units, relative to that ideal, was
	// 3.2-4.5x with a refill every 500 ms, 1.5x at 100 ms, 1.05-1.36x at 20 ms
	// and 1.01-1.06x at 5 ms; the mean rate matched the share at every cadence.
	erlUpdateInterval = 500 * time.Millisecond
	erlRefillInterval = 10 * time.Millisecond

	// Token bucket defaults.
	//
	// The ceilings are guards against a runaway rate, not part of the control:
	// they must stay well above what any workload spends at full speed, or a
	// fast workload with a high share is held below it. Measured on an RTX 3090,
	// a tight kernel loop launches about 25,000 kernels/s and a replayed
	// 50-kernel CUDA graph costs 358 tokens at 514 replays/s: about 184,000
	// tokens/s for one process using the whole device. Faster devices, larger
	// kernels and several processes per pod multiply that, so the ceiling leaves
	// a factor of about 50. Both remain configurable per pool (see
	// loadERLConfigFromEnv).
	//
	// Burst. A worker that has nothing to run saves up tokens, and spends them at
	// full speed when it resumes. burstWindow is how much it may save: W seconds
	// of its own refill, so its bucket holds rate x W tokens. This block is the
	// whole rule; the documentation and the annotation refer to it.
	//
	//   - Where W comes from: the workload's own, from the annotation
	//     constants.ComputeBurstWindowAnnotation, which can be changed while the
	//     Pod runs; otherwise the pool's, burstWindow of the rate limiter
	//     parameters of its SchedulingConfigTemplate; otherwise the value
	//     compiled in here, the one default everything else derives from.
	//   - Bounds: 0 to burstWindowMax (constants.ComputeBurstWindowMax, 10 s).
	//     A bucket always holds burstWindowFloor seconds of refill, two refill
	//     intervals, whatever W says: tokens arrive in portions of one refill
	//     interval, and a pass that comes late has to be able to hand out what
	//     was earned since the previous one. W = 0 therefore means a bucket of
	//     rate x burstWindowFloor and no burst to speak of.
	//   - The long-run share does not change with W. A burst is paid with tokens
	//     that were saved at the worker's own rate while it ran below its share;
	//     W never raises the rate, and an idle worker's rate does not climb (see
	//     tokenBoundFraction and updateERLBucket). Over any stretch of T seconds
	//     a worker gets at most its share x (T + W) of the device.
	//   - How long a burst lasts: a worker with a share s of the device that
	//     wakes up with a full bucket spends at full speed while it earns at s,
	//     so the bucket lasts about s x W / (1 - s) seconds, when the device is
	//     otherwise free: 33 ms at 25% and 100 ms at 50% for W = 0.1 s, 0.67 s
	//     and 2 s for W = 2 s. After that it is paced at its share as always.
	//     The bucket is full again after W seconds without work.
	//   - Debt is not affected: a launch that costs more than the bucket holds
	//     is admitted on credit as before, within debtBound, and W, at most 10 s
	//     of refill, stays below the 60 s of work at full speed of that bound.
	//   - A larger W takes effect as the bucket fills, at the worker's rate. A
	//     smaller one takes effect with the next refill, which brings a balance
	//     above the new capacity down to it and takes nothing else.
	//   - capacityMin and capacityMax still bound the bucket. Where one of them
	//     decides, the worker's effective window is capacity / rate, and that is
	//     what is logged. The compiled ceiling holds burstWindowMax of the
	//     highest rate, so it does not decide unless a pool lowers it.
	//
	// The floors only keep the numbers away from zero. A worker whose work is
	// cheap in tokens needs a low rate to be held at its share (a workload that
	// spends 2 tokens/s with the whole device needs 0.6 tokens/s at a 30% share),
	// and a floor above that rate leaves it unlimited.
	burstWindow      = 0.1 // default W: capacity = rate × burstWindow (seconds)
	burstWindowMax   = float64(constants.ComputeBurstWindowMax) / float64(time.Second)
	burstWindowFloor = 2 * float64(erlRefillInterval) / float64(time.Second)
	rateMin          = 0.01                     // minimum refill rate (tokens/s)
	rateMax          = 10000000.0               // maximum refill rate (tokens/s)
	capacityMin      = 1.0                      // minimum token capacity
	capacityMax      = rateMax * burstWindowMax // holds the longest burst window of the highest rate

	// Overrides (see loadERLConfigFromEnv) are only accepted within these
	// bounds; the reasons are the measurements quoted above and at
	// erlUpdateInterval. A ceiling below what one fast process spends holds a
	// worker under its share, and a floor above what cheap work needs leaves it
	// unlimited. The burst window is accepted from 0 to burstWindowMax (see
	// burstWindow).
	rateMaxLowest     = 1000000.0 // lowest accepted rate ceiling, about 5x the measured 184,000 tokens/s
	rateMinHighest    = 10.0      // highest accepted rate floor
	capacityMinHigh   = 20.0      // highest accepted capacity floor
	utilAlphaLow      = 0.05
	utilAlphaHigh     = 0.6
	pidGainHigh       = 3.0 // highest accepted kp, ki or kd
	integralDecayLow  = 0.5
	integralDecayHigh = 0.99
	// The rate a new bucket starts from.
	initialRate = 100.0

	// A worker is token-bound, held back by its rate rather than by having
	// nothing to run, when its bucket is found less than this full. Only then
	// can a higher rate change what it does.
	tokenBoundFraction = 0.5

	// A control loop that takes over a bucket adopts the rate it finds there
	// when a control loop published it (after a hypervisor restart or upgrade).
	// That rate is what the previous loop learned over minutes, and the new one
	// knows nothing yet: it leaves the rate alone until the worker's signal is
	// re-established, and adapts it from there.
	//
	// The signal is re-established when, in adoptionHoldIntervals control
	// intervals in a row, what the worker spent was measured and its utilization
	// was the calibrated figure. An interval without a figure starts the count
	// over: one in which somebody else refilled the bucket, as the limiters do
	// until the new loop's first pass, or one without telemetry.
	//
	// Where per-process readings take part in the figure, that many intervals
	// tell what the worker spends, but not what a token buys it. The readings
	// do not scatter independently: measured on a device that two busy tenants
	// shared, the driver booked nearly all of one tenant's work under the other
	// for three to eight seconds at a time, in 13% of the readings. Ten seconds
	// of such readings put what a token bought a 30% tenant at a third of the
	// truth; taken for a worker that is held under its share, its rate went to
	// 3.3 times the adopted one within two seconds, and it ran at 1.9 times its
	// limit for 25 s. The calibration of those readings therefore does not start
	// from what the hold has seen: it starts as one that has learned the adopted
	// rate (utilizationCalibration.inherit), and follows the readings from
	// there the way it does for a worker it has known for minutes.
	//
	// Without a signal that is re-established (a worker that idles, a bucket
	// that keeps being refilled by somebody else) the hold ends after
	// adoptionHoldMaxIntervals intervals with a figure: a minute, about what
	// the calibrations need in the first place. The rate is then adapted from
	// the adopted one by what there is, but not by doubling (blindRampFactor)
	// until the signal is re-established: an adopted rate is not that of a
	// worker whose scale is still unknown.
	//
	// The rate every bucket starts from is not one a control loop learned. A
	// bucket adopted at it is that of a worker that has not been controlled yet:
	// it is held for adoptionHoldIntervals and then treated as any new worker.
	adoptionHoldIntervals    = 20
	adoptionHoldMaxIntervals = 120

	// Utilization (0-1) below which a worker counts as idle.
	idleUtilization = 0.01

	// The highest recent consumption of a worker is remembered with this decay
	// per control interval (half after about three seconds), which outlasts the
	// delay of utilization telemetry behind a change in what the worker does.
	consumptionPeakDecay = 0.89

	// Feedback-control parameters.
	utilAlpha           = 0.25 // utilization smoothing (EMA)
	utilDeadband        = 0.03 // ignore tiny error to reduce oscillation
	kp                  = 0.9
	ki                  = 0.35
	kd                  = 0.10
	integralDecayFactor = 0.85
	integralClamp       = 1.5

	// Rate slew limiting avoids abrupt oscillation when NVML samples jump.
	maxRateIncreaseRatio = 0.35 // max +35% per control interval
	maxRateDecreaseRatio = 0.25 // max -25% per control interval
	// While a worker's utilization is still below idleUtilization, its rate is
	// doubled per interval instead, provided its share is at least
	// blindRampMinTarget: by the time telemetry shows the first percent, the
	// worker is at a few percent, well below such a share.
	blindRampFactor    = 2.0
	blindRampMinTarget = 0.10

	// Keep a modest token reserve for bursts while smoothing bucket release.
	tokenReserveRatio = 0.35
	tokenDrainRatio   = 0.80 // drain up to 80% of capacity/sec when oversupplied
	tokenDrainMin     = 25.0

	// Token consumption is what splits a shared device's utilization between
	// its workers. It is smoothed so that a worker launching kernels less often
	// than the control interval still counts as active, but only lightly: a
	// worker that stopped must release its share within a few intervals.
	consumptionAlpha = 0.5
	// Consumption below this (tokens/s) is treated as idle.
	consumptionFloor = 1e-3

	// A bucket that was not updated for this long (seconds) is not refilled
	// for the gap, and its consumption over the gap is not measured.
	maxTokenUpdateGap = 5.0
	// The most a worker can owe, in seconds of work with the whole device; see
	// debtBound.
	maxTokenDebtSeconds = 60.0

	// After a balance that no limiter produces, a bucket's spending counts as
	// not measured for this many looks (10 s); see tokenMeter.
	meterQuarantineIntervals = 20
	// A bucket whose spending could not be told at meterExclusionIntervals looks
	// (60 s) is left out of the token split of its device, until it could be
	// told at meterReadmissionIntervals looks in a row (10 s); see
	// tokenMeter.excluded. A minute is several times what leaves a bucket
	// unreadable in normal operation (the quarantine after one stray balance is
	// 10 s, a stall or restart of the hypervisor a few seconds), and it is the
	// longest the neighbors' rates are held for one worker's sake: about what
	// the calibrations need to learn what a token buys in the first place, so a
	// rate held for that long has missed little. The Pod is told with an Event,
	// at most once per meterExclusionEventInterval.
	meterExclusionIntervals     = 120
	meterReadmissionIntervals   = 20
	meterExclusionEventInterval = 5 * time.Minute
	// podEventTimeout bounds one attempt to record an Event.
	podEventTimeout = 5 * time.Second

	// Per-process utilization older than this is ignored.
	processUtilizationMaxAge = 4 * erlUpdateInterval

	// The per-process figures of a device are checked against the device's own
	// utilization, both averaged with processCheckAlpha (about 5 s). They are not
	// used while the device is busier than processCheckBusyUtil and they account
	// for less than processCheckMinShare of it. See Controller.processUtilizationFor.
	processCheckAlpha    = 0.1
	processCheckBusyUtil = 0.10
	processCheckMinShare = 0.5
	// A worker that keeps spending tokens while its processes read zero for this
	// many intervals in a row, on a busy device, is not described by them.
	silentSpendingIntervals = 20

	// Utilization readings calibrate how much utilization a worker gets out of a
	// token; see utilizationCalibration. The averages behind it move slowly
	// (about 17 s at the control interval) because single per-process readings
	// are very noisy, and are not used before a few readings went in.
	calibrationAlpha       = 0.03
	calibrationMinSamples  = 4
	calibrationMinEvidence = 0.5
	// Per-process readings are coarse: each is the part of a handful of driver
	// samples that fell to the worker. Their calibration counts for nothing
	// before the readings add up to calibrationCoarseEvidence (about eight such
	// samples), is trusted in full once they add up to calibrationTrustEvidence
	// (about sixty: what a token buys is then known to about an eighth), and
	// moves more slowly afterwards (about 50 s), which leaves it steady to a
	// few percent. Until it is trusted in full the calibration of the device's
	// own utilization, which is exact from the start, makes up the difference;
	// see attributeDeviceUtilization.
	calibrationCoarseAlpha    = 0.01
	calibrationCoarseEvidence = 2.0
	calibrationTrustEvidence  = 16.0
	// A calibrated rate is left alone while it is within this fraction of the
	// right one.
	calibratedRateDeadband = 0.03
	// The recent readings are only compared with the average while the worker
	// has been spending at least this share of what it spends on average.
	calibrationMinSpending = 0.2
	// What a token buys is also taken over the most recent calibrationRecent
	// readings alone (8 s). When that differs from the slow average by more than
	// the scatter expected of such readings explains (and at least by
	// calibrationChangeFactor) for calibrationChangeReadings readings in a row,
	// the work has changed and the calibration starts over.
	calibrationRecent         = 16
	calibrationChangeFactor   = 1.25
	calibrationChangeReadings = 4
	calibrationChangeSigmas   = 2.4
	// A reading is the part of calibrationDriverSamples samples of the driver
	// that was attributed to the worker; half of them are new in each reading.
	calibrationDriverSamples = 8.0
	// The two directions are not judged alike for per-process readings.
	// Finding that a token buys more than assumed lowers the rate: when wrong,
	// the worker gets less than its share for a few seconds. Finding that it
	// buys less raises the rate: when wrong, the worker exceeds its limit, and a
	// small share reads zero for seconds on end by chance alone. That finding
	// is acted on only when chance explains the recent readings with less than
	// calibrationLessChance, or with less than calibrationLessConfirmedChance
	// while the device's own utilization shows the same (it then bought less by
	// at least 1-calibrationLessConfirmation over the same readings).
	calibrationLessChance          = 1e-6
	calibrationLessConfirmedChance = 1e-2
	calibrationLessConfirmation    = 0.97

	// State of a removed worker is not re-created by a control pass that was
	// already under way for this long after the removal.
	removedWorkerMemory = time.Minute

	// A worker's own per-process readings bound what a token can buy it,
	// however few of them there are: so many readings with so few driver
	// samples attributed to the worker are not what a higher figure produces,
	// and the other way round. The bounds are the samples seen plus and minus
	// calibrationBoundSigmas standard deviations of such a count and one more;
	// see utilizationCalibration.bounds.
	calibrationBoundSigmas = 1.0
	// A worker that spends tokens while its processes read zero is only taken
	// for one its processes do not describe (see silentSpendingIntervals) while
	// the processes of the device account for this much less than the device
	// shows (0-1, both averaged, see processCheck): otherwise nothing is missing
	// from them, and the worker is simply getting very little.
	silentUnaccounted = 0.08

	// Lowering a worker's rate is checked for effect; see erlState.cutFrom and
	// loweringIsFutile. What the limited workers of the device spend and what
	// the device shows are averaged with futilityAlpha (about 2 s). Once the
	// spending is down to futilitySpending of what it was when the lowering
	// began, the load has to have come down with it: by no less than
	// 1-futilityShortfall of the same proportion. If it has, for
	// futilityConfirmIntervals in a row, or the rate has not been lowered for
	// futilityQuietIntervals, the run of lowering is over. If it has not, for
	// as many intervals, the rate goes back to where the lowering began and is
	// held for futilityHoldIntervals, twice as long each further time in a row,
	// up to futilityHoldDoublings times. Until one or the other is known the
	// rate is not lowered below futilityDepth of where the run began. A device
	// at futilitySaturated or more is full and shows nothing either way, and
	// readings below futilityMinReading are too coarse (whole percents).
	futilityAlpha            = 0.25
	futilitySpending         = 0.5
	futilityShortfall        = 0.4
	futilityConfirmIntervals = 4
	futilityQuietIntervals   = 20
	futilityHoldIntervals    = 40
	futilityHoldDoublings    = 4
	futilityDepth            = 0.125
	futilitySaturated        = 0.92
	futilityMinReading       = 0.05

	// Work that alternates between two kinds is not followed back and forth:
	// see erlState.heavyGain. A kind of work counts as a cheaper one when a token
	// buys regimeChangeFactor less, and the rate stays at what the more
	// expensive kind needs for regimeHoldIntervals (two minutes) after the
	// work last came back to it. Within one kind of work the figure follows
	// what is measured with regimeAlpha.
	regimeHoldIntervals = 240
	regimeChangeFactor  = 2.0
	regimeAlpha         = 0.02

	// A worker that gets a limit after having none is left at the highest rate
	// until what it spends unthrottled has been measured, for at most
	// limitProbeIntervals intervals, and the measurement only counts with a
	// utilization of at least limitProbeMinUtil (whole percents say little
	// below that); see Controller.followLimit.
	limitProbeIntervals = 4
	limitProbeMinUtil   = 0.05
	// After the limit moved a worker's rate, utilization readings describe the
	// time before for as long as the driver averages over (a second) and its
	// latest sample is old: that many intervals are kept out of the
	// calibrations.
	limitSettleIntervals = 3

	// A limiter refills its bucket itself once it finds the refill timestamp
	// this old (seconds): nobody has refilled for that long.
	limiterSelfRefillAfter = 0.1

	// A refill timestamp ahead of the clock by more than clockStepTolerance, and
	// found unchanged by clockStepPasses refill passes in a row, was stored
	// before the clock was set back; see refillBucket.refillInfo.
	clockStepTolerance = 0.05
	clockStepPasses    = 3
)

type erlConfig struct {
	burstWindow         float64
	rateMin             float64
	rateMax             float64
	capacityMin         float64
	capacityMax         float64
	utilAlpha           float64
	kp                  float64
	ki                  float64
	kd                  float64
	integralDecayFactor float64
}

func defaultERLConfig() erlConfig {
	return erlConfig{
		burstWindow:         burstWindow,
		rateMin:             rateMin,
		rateMax:             rateMax,
		capacityMin:         capacityMin,
		capacityMax:         capacityMax,
		utilAlpha:           utilAlpha,
		kp:                  kp,
		ki:                  ki,
		kd:                  kd,
		integralDecayFactor: integralDecayFactor,
	}
}

// erlOverride is one configurable parameter of the control loop: where its
// value lives, and the range an override has to be in. The compiled default is
// the value that was measured and tested; an override outside the range is one
// that is known not to work (see the constants).
type erlOverride struct {
	name     string
	raw      string
	value    *float64
	min, max float64
}

// overrideCapacityMax is the name of the bucket ceiling among the overrides.
const overrideCapacityMax = "capacityMax"

// applyERLOverrides validates the overrides against the compiled defaults in
// cfg and applies the ones that pass. It returns the names of the parameters
// that were overridden and, for each one that was not accepted, why; a
// rejected parameter keeps its compiled default.
func applyERLOverrides(cfg erlConfig, erl tfv1.ElasticRateLimitParameters) (erlConfig, []string, []string) {
	defaults := cfg
	overrides := []erlOverride{
		{"maxRefillRate", erl.MaxRefillRate, &cfg.rateMax, rateMaxLowest, math.MaxFloat64},
		{"minRefillRate", erl.MinRefillRate, &cfg.rateMin, math.SmallestNonzeroFloat64, rateMinHighest},
		{"filterAlpha", erl.FilterAlpha, &cfg.utilAlpha, utilAlphaLow, utilAlphaHigh},
		{"kp", erl.Kp, &cfg.kp, math.SmallestNonzeroFloat64, pidGainHigh},
		{"ki", erl.Ki, &cfg.ki, math.SmallestNonzeroFloat64, pidGainHigh},
		{"kd", erl.Kd, &cfg.kd, math.SmallestNonzeroFloat64, pidGainHigh},
		{"burstWindow", erl.BurstWindow, &cfg.burstWindow, 0, burstWindowMax},
		{"capacityMin", erl.CapacityMin, &cfg.capacityMin, math.SmallestNonzeroFloat64, capacityMinHigh},
		{overrideCapacityMax, erl.CapacityMax, &cfg.capacityMax, math.SmallestNonzeroFloat64, math.MaxFloat64},
		{"integralDecayFactor", erl.IntegralDecayFactor, &cfg.integralDecayFactor, integralDecayLow, integralDecayHigh},
	}
	var applied, rejected []string
	for _, o := range overrides {
		raw := strings.TrimSpace(o.raw)
		if raw == "" {
			continue
		}
		v, err := strconv.ParseFloat(raw, 64)
		switch {
		case err != nil || !isFinite(v):
			rejected = append(rejected, fmt.Sprintf("%s=%q is not a finite number", o.name, raw))
		case v < 0 && o.min == 0:
			rejected = append(rejected, fmt.Sprintf("%s=%s is negative", o.name, raw))
		case v <= 0 && o.min > 0:
			rejected = append(rejected, fmt.Sprintf("%s=%s is not positive", o.name, raw))
		case v < o.min:
			rejected = append(rejected, fmt.Sprintf("%s=%s is below the lowest value that works, %g", o.name, raw, o.min))
		case v > o.max:
			rejected = append(rejected, fmt.Sprintf("%s=%s is above the highest value that works, %g", o.name, raw, o.max))
		default:
			*o.value = v
			applied = append(applied, o.name)
		}
	}
	// The bucket has to hold a burst window of the highest rate: a lower
	// capacity ceiling caps what a refill can add, and with it the rate. A
	// ceiling the pool does not set follows from the rate ceiling, for the
	// longest window a workload can ask for. One the pool sets is kept, and has
	// to hold at least the pool's own window; a workload's longer window is then
	// held to it (see burstWindow).
	if !slices.Contains(applied, overrideCapacityMax) {
		cfg.capacityMax = math.Max(cfg.capacityMax, math.Min(cfg.rateMax*burstWindowMax, math.MaxFloat64))
	} else if cfg.capacityMax < cfg.rateMax*cfg.burstWindow {
		rejected = append(rejected, fmt.Sprintf(
			"capacityMax=%g holds less than burstWindow=%g s of maxRefillRate=%g; all three keep their defaults",
			cfg.capacityMax, cfg.burstWindow, cfg.rateMax))
		cfg.capacityMax, cfg.burstWindow, cfg.rateMax = defaults.capacityMax, defaults.burstWindow, defaults.rateMax
		kept := applied[:0]
		for _, name := range applied {
			if name != overrideCapacityMax && name != "burstWindow" && name != "maxRefillRate" {
				kept = append(kept, name)
			}
		}
		applied = kept
	}
	return cfg, applied, rejected
}

// loadERLConfigFromEnv returns the configuration of the control loop: the
// compiled defaults, which are what the limiter was measured and tested with,
// changed by the overrides of the pool that pass validation. Whatever is
// rejected is reported in one line and keeps its default.
func loadERLConfigFromEnv() erlConfig {
	cfg := defaultERLConfig()

	raw := os.Getenv(constants.HypervisorSchedulingConfigEnv)
	if raw == "" {
		return cfg
	}

	var scheduling tfv1.HypervisorScheduling
	if err := json.Unmarshal([]byte(raw), &scheduling); err != nil {
		klog.Warningf("Failed to parse %s, using the compiled rate limiter defaults: %v",
			constants.HypervisorSchedulingConfigEnv, err)
		return cfg
	}

	cfg, applied, rejected := applyERLOverrides(cfg, scheduling.ElasticRateLimitParameters)
	if len(rejected) > 0 {
		klog.Warningf("Rate limiter overrides rejected, the compiled defaults are used for them: %s",
			strings.Join(rejected, "; "))
	}
	if len(applied) > 0 {
		klog.Infof("Rate limiter overrides in effect: %s", strings.Join(applied, ", "))
	}
	return cfg
}
