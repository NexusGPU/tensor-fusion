package computing

import (
	"math"
)

// tokenMeter tracks one token bucket between control intervals. The hypervisor
// is the only writer that adds tokens, so whatever is missing since the level it
// left behind was spent by the worker.
//
// A balance below zero is debt: the limiter admitted work that cost more than
// the bucket held and the worker owes the difference. It is as valid a level as
// a positive one, and what was borrowed counts as spent.
//
// The meter looks at a bucket once per control interval, while tokens are added
// far more often. It is therefore told how many tokens the hypervisor added (net
// of what it removed) since its previous look, instead of inferring anything
// from two levels.
//
// The balance lives in the worker's own mapping, which is writable by the
// workload's processes; values read from it are validated. What the meter
// reports decides how the utilization of a shared device is split, so it only
// reports what the limiter's contract can produce. By that
// contract a balance only falls, except by what the hypervisor added and by
// what a limiter refilled itself at the published rate while it found nobody
// refilling (foreignRate), and it never falls below the overdraft bound
// (debtFloor). A balance that is not a number, rose by more, or lies below the
// bound was written by somebody else. It is not a measurement of anything: the
// interval counts as not measured, like one without a previous look, and so do
// the following meterQuarantineIntervals, so that a balance written up and down
// in turns never yields a figure either.
//
// A bucket that keeps yielding no figure is left out of the split of its
// device (excluded), see attributeDeviceUtilization. unaccounted counts the
// looks whose spending could not be told because of what was found in the
// mapping: a balance as above, or a refill by somebody else. It is not a count
// of looks in a row in the strict sense: it only starts over after
// meterReadmissionIntervals measured looks in a row (accounted), so that a
// balance written once every so often adds up as well, and a worker leaves its
// neighbors without a split for meterExclusionIntervals looks at most between
// two such runs. A first look and a look after a gap count for neither, and
// nor does a look at an interval in which a limiter refilled the bucket because
// the hypervisor itself was late with its refills (foreignExpected): on a node
// without CPU to spare that happens every few seconds for as long as it lasts,
// to every bucket, and is nothing the worker did. Such a look changes neither
// count; the interval still yields no figure.
type tokenMeter struct {
	level       float64 // balance at the previous look, negative when in debt
	updatedAt   float64 // when level was recorded (seconds), 0 if never
	consumption float64 // smoothed tokens/s spent by the worker
	measured    bool    // consumption covers at least one full interval
	// debtFloor is the most the worker can owe (tokens, positive), 0 while that
	// is not known yet.
	debtFloor float64
	// foreignRate is the rate (tokens/s) at which somebody else refilled the
	// bucket since the previous look, 0 if nobody did. It is set by the caller
	// and consumed by observe.
	foreignRate float64
	// foreignExpected says that this controller was itself late enough with
	// its refills in that time for a limiter to refill the bucket (see
	// refillBucket.ownGap). Set by the caller with foreignRate.
	foreignExpected bool
	// quarantine counts down the looks that are not measured after a balance
	// that no limiter produces.
	quarantine int
	// peak is the highest consumption of the recent past (tokens/s), decaying.
	peak float64
	// last is what was spent in the latest interval alone (tokens/s), unsmoothed,
	// and window the same over the latest two intervals: the second or so that
	// a utilization reading describes.
	last   float64
	window float64
	// See the type's comment.
	unaccounted, accounted int
	excluded               bool
}

// account records whether the latest look told what the worker spent, and
// decides on leaving the bucket out of the split of its device.
func (m *tokenMeter) account(told bool) {
	if !told {
		m.accounted = 0
		if m.unaccounted = min(m.unaccounted+1, meterExclusionIntervals); m.unaccounted >= meterExclusionIntervals {
			m.excluded = true
		}
		return
	}
	if m.accounted = min(m.accounted+1, meterReadmissionIntervals); m.accounted >= meterReadmissionIntervals {
		m.unaccounted, m.excluded = 0, false
	}
}

func isFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// observe records the current balance and folds what the worker spent since the
// previous look into consumption. added is the net amount the hypervisor put
// into the bucket in between.
func (m *tokenMeter) observe(tokens, added, nowSecs float64) {
	previous, elapsed, foreignRate, foreignExpected := m.level, nowSecs-m.updatedAt, m.foreignRate, m.foreignExpected
	measurable := m.updatedAt > 0
	m.level, m.updatedAt, m.foreignRate, m.foreignExpected = tokens, nowSecs, 0, false
	if !isFinite(tokens) {
		m.quarantine = meterQuarantineIntervals
	}
	// Without a previous look, a usable interval and numbers throughout there
	// is nothing to measure, and nothing may be divided.
	if !measurable || !isFinite(elapsed) || elapsed <= 0 || elapsed >= maxTokenUpdateGap ||
		!isFinite(tokens) || !isFinite(previous) || !isFinite(added) {
		m.consumption, m.measured, m.last, m.window = 0, false, 0, 0
		if m.quarantine > 0 {
			m.account(false)
		}
		return
	}
	// The slack covers rounding, and a limiter's own refill a little more time
	// than the interval, whose ends the two sides do not read at the same instant.
	slack := 1e-6 * (1 + math.Abs(previous) + math.Abs(added) + math.Abs(tokens))
	if tokens > previous+added+1.5*foreignRate*elapsed+slack || (m.debtFloor > 0 && tokens < -m.debtFloor-slack) {
		m.quarantine = meterQuarantineIntervals
	}
	// An interval somebody else refilled in yields no figure either: those
	// tokens are in no account, and anything computed from the levels would
	// undercount what the worker spent.
	if m.quarantine > 0 || foreignRate > 0 {
		// A limiter's refill that the hypervisor's own lateness explains is
		// nothing found wrong with the bucket: the look counts for neither
		// count, like a first look.
		counted := m.quarantine > 0 || !foreignExpected
		m.quarantine = max(m.quarantine-1, 0)
		m.consumption, m.measured, m.last, m.window = 0, false, 0, 0
		if counted {
			m.account(false)
		}
		return
	}
	spent := math.Max(0, (previous+added-tokens)/elapsed)
	m.window = spent
	if m.measured {
		m.window = (spent + m.last) / 2
	}
	m.last = spent
	if m.measured {
		spent = consumptionAlpha*spent + (1-consumptionAlpha)*m.consumption
	}
	if spent < consumptionFloor {
		spent = 0
	}
	m.consumption, m.measured = spent, true
	m.peak = math.Max(spent, m.peak*consumptionPeakDecay)
	m.account(true)
}

// utilizationCalibration turns the utilization readings of a worker into a
// figure the rate control can act on.
//
// What the driver reports per process is unbiased but coarse: its sampling
// attributes each short sample to a single process, so one reading of a worker
// using a fifth of the device is off by about 14 points and is exactly zero
// every sixth time. Every reading also describes the past: the driver averages
// over a second. Fed to the rate control directly, the noise is larger than the
// limit it should hold, and the delay makes it act on a rate it has already
// changed.
//
// The tokens a worker spends are known exactly and at once. Over many readings
// the utilization figures therefore pin down how much utilization this worker
// gets out of one token per second, and the utilization of a single interval is
// that ratio applied to what was spent in it. A worker that changes its pace is
// followed at once through its spending.
//
// What a token buys changes when the work does (other kernels, another model).
// The slow average alone would take most of a minute to follow. The same ratio
// is therefore also taken over the recent readings alone; when the two disagree
// by more than the scatter of the readings explains, several readings in a row,
// the calibration starts over between the two and lets the next readings
// decide. That takes a few seconds.
//
// With the ratio known, the rate that holds the worker at its share is known
// too: the share divided by what a token buys. The rate control aims at that,
// see Controller.updateERLBucket.
type utilizationCalibration struct {
	// coarse says that the readings are per-process figures; see
	// calibrationCoarseEvidence.
	coarse bool

	utilization float64 // average reading (0-1)
	consumption float64 // average tokens/s spent over the same readings
	samples     int
	// evidence is the sum of all readings so far. Utilization is reported in
	// whole percents, so the ratio means little until the readings add up to
	// something the rounding is small against.
	evidence float64

	// The most recent readings and what was spent with them, as a ring, and
	// for how many readings in a row they have disagreed with the averages.
	recentUtilization [calibrationRecent]float64
	recentConsumption [calibrationRecent]float64
	recentNext        int
	recentCount       int
	disagreeing       int

	// silent counts the readings of zero in a row while the worker was spending
	// tokens on a busy device.
	silent int

	// lessConfirmed is set for one reading of a coarse calibration by whoever
	// has exact readings to compare with: those show a token buying less too.
	lessConfirmed bool

	// frozen keeps readings out of the calibration: they are known to describe
	// load the worker's tokens do not pay for (see erlState.cutFrom).
	frozen bool

	// slow says that the readings are those of a worker that shares its device
	// with other limited workers (see calibrationSlowAlpha): they move the
	// average slowly, and restart the calibration only when the device's own
	// utilization shows the change as well (moreConfirmed, lessConfirmed).
	// matured says that the calibration has been trusted in full once; a
	// restart after that does not take the trust away.
	slow, moreConfirmed, lessShown, matured bool
}

// calibrationSnapshot is what a calibration had learned at some moment.
type calibrationSnapshot struct {
	utilization, consumption float64
	samples                  int
	taken                    bool
}

func (c *utilizationCalibration) snapshot() calibrationSnapshot {
	return calibrationSnapshot{utilization: c.utilization, consumption: c.consumption, samples: c.samples, taken: true}
}

// restore puts back what the calibration had learned when the snapshot was
// taken; the recent readings are forgotten with what was learned from them.
func (c *utilizationCalibration) restore(from calibrationSnapshot) {
	if !from.taken {
		return
	}
	c.utilization, c.consumption, c.samples = from.utilization, from.consumption, from.samples
	c.recentCount, c.recentNext, c.disagreeing = 0, 0, 0
}

// inherit makes a calibration of per-process readings one that has learned
// gain (utilization per token/s) the way a calibration does over minutes: it is
// trusted in full, moves as slowly as one that has seen that many readings, and
// compares the readings to come with it. It keeps what the worker was found to
// spend. See adoptionHoldIntervals for when a gain is inherited.
func (c *utilizationCalibration) inherit(gain float64) {
	if c == nil || !c.coarse || c.consumption <= consumptionFloor || !isFinite(gain) || gain <= 0 {
		return
	}
	c.utilization = gain * c.consumption
	c.samples = max(c.samples, int(1/calibrationCoarseAlpha))
	c.evidence = math.Max(c.evidence, calibrationTrustEvidence)
	c.recentCount, c.recentNext, c.disagreeing = 0, 0, 0
}

// recentRatio returns what a token bought over the recent readings as a share
// of what it buys on average, once there are enough of both.
func (c *utilizationCalibration) recentRatio() (float64, bool) {
	if !c.calibrated() || c.recentCount < calibrationRecent {
		return 0, false
	}
	utilization, consumption := 0.0, 0.0
	for i := range calibrationRecent {
		utilization += c.recentUtilization[i]
		consumption += c.recentConsumption[i]
	}
	if consumption < calibrationMinSpending*c.consumption*calibrationRecent || consumption <= 0 {
		return 0, false
	}
	return utilization / consumption / c.gain(), true
}

// poissonAtMost returns the chance of at most k events when mean are expected.
func poissonAtMost(k int, mean float64) float64 {
	term, sum := math.Exp(-mean), 0.0
	for i := 0; i <= k; i++ {
		sum += term
		term *= mean / float64(i+1)
	}
	return math.Min(sum, 1)
}

func (c *utilizationCalibration) calibrated() bool {
	return c != nil && c.samples >= calibrationMinSamples && c.consumption > consumptionFloor &&
		c.evidence >= c.minEvidence()
}

func (c *utilizationCalibration) minEvidence() float64 {
	if c.coarse {
		return calibrationCoarseEvidence
	}
	return calibrationMinEvidence
}

// trust is the weight (0-1) the calibration deserves next to one made from
// exact readings: all of it for exact readings, growing with the evidence for
// coarse ones.
func (c *utilizationCalibration) trust() float64 {
	if !c.calibrated() {
		return 0
	}
	if !c.coarse {
		return 1
	}
	trust := clampFloat64(c.evidence/calibrationTrustEvidence, 0, 1)
	if c.slow && !c.matured {
		if c.samples >= calibrationSlowSamples {
			c.matured = true
		}
		trust = math.Min(trust, clampFloat64(float64(c.samples)/calibrationSlowSamples, 0, 1))
	}
	return trust
}

// gain is the utilization (0-1) the worker gets out of one token per second.
// It is only meaningful once calibrated.
func (c *utilizationCalibration) gain() float64 {
	return c.utilization / c.consumption
}

// bounds are the least and the most a token can buy the worker (utilization per
// token/s) by its own per-process readings, and whether there are enough of
// them to say.
//
// Trust in what those readings average to grows with their sum, and a worker
// that is held far below its share has next to nothing to sum: its readings
// are zero nearly every time. That is information all the same. So many
// readings, so much spent in them, and so few driver samples attributed to the
// worker put a bound on what a token buys it, from the first readings on, and
// the bound tightens with every further one; likewise from below for a worker
// that is attributed many. A figure derived from the device that lies outside
// the bounds is not used beyond them: the token split is such a figure next to
// a neighbor whose work costs a different number of tokens, too high for the
// one that pays more tokens and too low for the other. How fast the bounds
// close in is set by how often the driver can attribute a sample to the worker
// at all: at a share of 5% that is about once every two and a half seconds.
func (c *utilizationCalibration) bounds() (least, most float64, known bool) {
	if c == nil || !c.coarse || c.samples < calibrationMinSamples || c.consumption <= consumptionFloor {
		return 0, 0, false
	}
	// The averages cover this many readings, each with samplesPerReading new
	// samples of the driver.
	const samplesPerReading = calibrationDriverSamples / 2
	readings := math.Min(float64(c.samples), 1/calibrationCoarseAlpha)
	seen := c.utilization * readings * samplesPerReading
	margin := calibrationBoundSigmas*math.Sqrt(seen+1) + 1
	perSample := 1 / (readings * samplesPerReading) / c.consumption
	return math.Max(seen-margin, 0) * perSample, (seen + margin) * perSample, true
}

// estimate folds one utilization reading (0-1) into the calibration and returns
// the worker's utilization for this interval, and whether it is the calibrated
// figure. Without token spending to relate it to, or before enough readings,
// the reading itself is the estimate.
func (c *utilizationCalibration) estimate(reading float64, meter *tokenMeter) (float64, bool) {
	if c == nil || meter == nil || !meter.measured {
		return reading, false
	}
	// The reading is related to what was spent in the time it describes. The
	// driver averages over about a second, two control intervals; the smoothed
	// consumption would trail further behind. While the rate is being raised
	// step by step, relating a reading to more recent spending than it covers
	// makes a token look as if it bought less than it does, and the rate would
	// be taken past the share. A provider that describes a shorter time errs
	// the other way, towards a rate that is reached later. An interval counts
	// when something was spent or something ran, also when only one of the two
	// happened: work that comes in long launches is paid for in one interval
	// and runs through the following ones, and only the sums over all of them
	// relate the two.
	spending := meter.window
	// An interval in which nothing was spent and nothing ran says nothing about
	// what a token buys. Counting it would only let what was learned fade while
	// the worker idles, and leave the first readings after it resumes, which
	// still trail behind what it spends, to decide the ratio alone.
	if c.frozen || (reading <= 0 && spending <= consumptionFloor) {
		if !c.calibrated() {
			return reading, false
		}
		return clampFloat64(c.gain()*meter.consumption, 0, 1), true
	}

	c.recentUtilization[c.recentNext], c.recentConsumption[c.recentNext] = reading, spending
	c.recentNext = (c.recentNext + 1) % calibrationRecent
	c.recentCount = min(c.recentCount+1, calibrationRecent)
	if c.calibrated() && c.recentCount == calibrationRecent {
		recentUtilization, recentConsumption := 0.0, 0.0
		for i := range calibrationRecent {
			recentUtilization += c.recentUtilization[i]
			recentConsumption += c.recentConsumption[i]
		}
		// Recent readings only say something while the worker has been spending
		// about as much as it does on average.
		comparable := recentConsumption >= calibrationMinSpending*c.consumption*calibrationRecent
		recent, average := recentUtilization/math.Max(recentConsumption, consumptionFloor), c.gain()
		// How far the recent readings may be off by chance depends on the share:
		// they hold calibrationRecent*calibrationDriverSamples/2 samples of the
		// driver, each attributed to the worker or not, so small shares scatter
		// more.
		const samplesPerReading = calibrationDriverSamples / 2
		share := clampFloat64(c.utilization, 0.01, 0.99)
		scatter := math.Sqrt((1 - share) / (share * calibrationRecent * samplesPerReading))
		factor := math.Max(calibrationChangeFactor, 1+calibrationChangeSigmas*scatter)
		buysMore := recent > average*factor
		buysLess := recent*factor < average
		if buysLess && c.coarse {
			// How likely so few samples fell to the worker if nothing changed.
			expected := average * recentConsumption * samplesPerReading
			seen := int(math.Ceil(recentUtilization * samplesPerReading))
			chance := poissonAtMost(seen, expected)
			buysLess = chance <= calibrationLessChance || (c.lessConfirmed && chance <= calibrationLessConfirmedChance)
		}
		confirmed := !c.slow || (buysMore && c.moreConfirmed) || (buysLess && c.lessShown)
		if comparable && (buysMore || buysLess) && confirmed {
			c.disagreeing++
		} else {
			c.disagreeing = 0
		}
		if c.disagreeing >= calibrationChangeReadings {
			// Start over half-way between the two (in proportion), with little
			// weight, so that the next readings decide: when the work has
			// changed they pull the ratio the rest of the way within seconds,
			// and when the recent readings were only an unlucky run, the half
			// step is taken back as quickly.
			c.consumption = recentConsumption / calibrationRecent
			c.utilization = math.Sqrt(recent*average) * c.consumption
			c.samples, c.disagreeing = calibrationMinSamples, 0
		}
	}

	// A plain average until enough readings are in, a slow moving one after.
	slowest := calibrationAlpha
	if c.coarse {
		slowest = calibrationCoarseAlpha
	}
	if c.slow {
		slowest = calibrationSlowAlpha
	}
	alpha := math.Max(slowest, 1/float64(c.samples+1))
	c.utilization += alpha * (reading - c.utilization)
	c.consumption += alpha * (spending - c.consumption)
	c.samples++
	c.evidence += reading
	if !c.calibrated() {
		return reading, false
	}
	return clampFloat64(c.gain()*meter.consumption, 0, 1), true
}

// workerCalibration holds the calibrations of one bucket: one per kind of
// utilization reading.
type workerCalibration struct {
	// processes is fed with the worker's per-process readings: unbiased, but
	// coarse and slow to tell anything.
	processes utilizationCalibration
	// device is fed with what is derived from the device's own utilization:
	// exact at once, but only as good as the attribution to the worker.
	device utilizationCalibration
}
