package computing

import (
	"math"
	"strings"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"k8s.io/klog/v2"
)

// utilizationSource says which signal a worker's utilization was derived from.
type utilizationSource string

const (
	// No usable signal: the rate is held and tokens keep being refilled.
	utilizationUnknown utilizationSource = "hold"
	// Sum of the worker's own processes as reported by the provider.
	utilizationFromProcesses utilizationSource = "process"
	// Device utilization split by the tokens each worker spent.
	utilizationFromTokenShare utilizationSource = "token-share"
	// Device utilization of a device used by a single worker.
	utilizationFromDevice utilizationSource = "device"
	// The worker's own token spending by what a token was found to buy it
	// earlier, next to a worker that is left out of the split of the device.
	utilizationFromSpending utilizationSource = "spending"
)

// erlMember is one worker's allocation on one device during a control interval.
type erlMember struct {
	workerUID string
	stateKey  string
	device    DeviceSnapshot
	bucket    *refillBucket
	meter     *tokenMeter
	// calibration relates the worker's per-process readings to what it spends,
	// fallback does the same for readings derived from the device's own
	// utilization. Both are optional; without them readings are used as they are.
	calibration *utilizationCalibration
	fallback    *utilizationCalibration

	// utilization is the worker's own share of the device (0-1); it is only
	// meaningful when source is not utilizationUnknown. calibrated says that it
	// is the calibrated figure rather than a raw reading, and gain is then the
	// utilization (0-1) the worker gets out of one token per second.
	utilization float64
	source      utilizationSource
	calibrated  bool
	gain        float64
	// tokenBound says that the worker's bucket was found (nearly) empty: its
	// rate is what holds it back.
	tokenBound bool
	// deviceLoad is the utilization of the device in this interval and
	// deviceSpending what all of its limited workers spent in the time it
	// describes (tokens/s), when the device's utilization could be split
	// between them (deviceRead).
	deviceLoad, deviceSpending float64
	deviceRead                 bool
	// held says that the worker's own readings put what a token buys it below
	// the figure derived from the device: it is being held under its share.
	held bool
	// coarse says that the worker's per-process readings took part in its
	// figure of this interval.
	coarse bool
}

// ProcessUtilizationSnapshot is the compute utilization the provider reported
// per process in one sample, summed per worker and device.
//
// A device is covered by the sample when the provider supplied a value for
// every one of its processes. The provider marks "not supplied" explicitly
// (api.ProcessInformation.ComputeUtilizationUnavailable), for all processes of
// a device at once; zero is a measurement like any other. A device with no
// process at all, or with a value outside 0-100, is not covered.
//
// A supplied zero is not proof of an idle process. The driver attributes each
// of its short samples to a single process, so a process using a small share of
// the device regularly reads zero for a whole sample although it is running.
// Consumers have to look at more than one sample; one sample only says what was
// reported in it.
type ProcessUtilizationSnapshot struct {
	SampledAt time.Time
	// Devices maps covered device (lower-case UUID) -> worker UID -> utilization
	// percent. A worker is present once any of its processes was seen on the
	// device.
	Devices map[string]map[string]float64
	// Totals holds, per covered device, the utilization percent of all of its
	// processes together, whether they belong to a worker or not.
	Totals map[string]float64
	// Unmetered holds the devices (covered or not) that run something whose
	// load no token bucket accounts for: a process outside every worker, or
	// one of a worker that is not limited through tokens.
	Unmetered map[string]bool
}

// NewProcessUtilizationSnapshot starts a snapshot from one provider sample.
// Every process counts towards coverage, whether or not it belongs to a worker.
func NewProcessUtilizationSnapshot(
	sampledAt time.Time, processes []api.ProcessInformation,
) *ProcessUtilizationSnapshot {
	snapshot := &ProcessUtilizationSnapshot{
		SampledAt: sampledAt,
		Devices:   make(map[string]map[string]float64),
		Totals:    make(map[string]float64),
		Unmetered: make(map[string]bool),
	}
	uncovered := make(map[string]bool)
	for i := range processes {
		deviceUUID := strings.ToLower(processes[i].DeviceUUID)
		percent := processes[i].ComputeUtilizationPercent
		if processes[i].ComputeUtilizationUnavailable || math.IsNaN(percent) || percent < 0 || percent > 100 {
			uncovered[deviceUUID] = true
			continue
		}
		if snapshot.Devices[deviceUUID] == nil {
			snapshot.Devices[deviceUUID] = make(map[string]float64)
		}
		snapshot.Totals[deviceUUID] += percent
	}
	// One missing or implausible value makes the whole device sample unusable:
	// the remaining processes would not add up to what the device does.
	for deviceUUID := range uncovered {
		delete(snapshot.Devices, deviceUUID)
		delete(snapshot.Totals, deviceUUID)
	}
	return snapshot
}

// MarkUnmetered records that a device runs a process whose load no token
// bucket accounts for.
func (s *ProcessUtilizationSnapshot) MarkUnmetered(deviceUUID string) {
	if s != nil {
		s.Unmetered[strings.ToLower(deviceUUID)] = true
	}
}

// unmetered reports whether a fresh snapshot saw unmetered load on the device.
func (s *ProcessUtilizationSnapshot) unmetered(deviceUUID string, now time.Time) bool {
	if s == nil {
		return false
	}
	if age := now.Sub(s.SampledAt); age > processUtilizationMaxAge || age < -processUtilizationMaxAge {
		return false
	}
	return s.Unmetered[deviceUUID]
}

// AddWorkerProcess attributes one process' utilization (percent) to a worker.
// It is a no-op for devices the provider did not cover.
func (s *ProcessUtilizationSnapshot) AddWorkerProcess(workerUID, deviceUUID string, percent float64) {
	if s == nil {
		return
	}
	if workers := s.Devices[strings.ToLower(deviceUUID)]; workers != nil {
		workers[workerUID] += percent
	}
}

// device returns the per-worker utilization of a covered device, or nil when
// the snapshot is missing, stale, or does not cover the device.
func (s *ProcessUtilizationSnapshot) device(deviceUUID string, now time.Time) map[string]float64 {
	if s == nil {
		return nil
	}
	// The sample is taken by another loop and may be marginally newer than now.
	if age := now.Sub(s.SampledAt); age > processUtilizationMaxAge || age < -processUtilizationMaxAge {
		return nil
	}
	return s.Devices[deviceUUID]
}

// sampleDeviceUtilization returns the valid whole-device utilization samples
// (0-1) keyed by lower-case device UUID. A device without a valid sample is
// absent, which is different from an idle device.
func (c *Controller) sampleDeviceUtilization() map[string]float64 {
	deviceUtilization := make(map[string]float64)
	deviceMetrics, err := c.deviceController.GetDeviceMetrics()
	if err != nil {
		klog.V(4).Infof("Device metrics unavailable, holding ERL rates: %v", err)
		return deviceUtilization
	}
	for uuid, m := range deviceMetrics {
		if m == nil || m.ComputeUnavailable || math.IsNaN(m.ComputePercentage) ||
			m.ComputePercentage < 0 || m.ComputePercentage > 100 {
			continue
		}
		deviceUtilization[strings.ToLower(uuid)] = m.ComputePercentage / 100.0
	}
	return deviceUtilization
}

// attributeDeviceUtilization derives, for every worker on one device, that
// worker's own share of the device. Feeding whole-device utilization to each
// worker would make it answer for what its neighbors use.
//
// Readings, per worker:
//  1. processUtil (non-nil only when the provider supplied it for this device
//     and it is plausible, see Controller.processUtilizationFor): the sum of the
//     worker's own processes. Unbiased, whatever the neighbors do, but coarse:
//     it takes many of them to say anything.
//  2. deviceUtil split by the tokens each worker spent. Exact at once when
//     workers pay a similar number of tokens per unit of utilization,
//     approximate otherwise; the shares always add up to deviceUtil.
//  3. deviceUtil itself, exact when the worker has the device to itself.
//
// No reading is used as read. Each kind goes through a calibration of its own
// (utilizationCalibration), and the worker's utilization is what its token
// spending amounts to by them: by the per-process calibration as far as its
// readings have earned trust, by the device's for the rest. A new worker is
// thus measured from the device from its first interval, and from its own
// processes after some tens of seconds; from then on its limit no longer
// depends on how its neighbors' work compares with its own. Before anything
// is calibrated the per-process reading is the estimate, or else the one
// derived from the device. A worker for which no reading applies is left as
// utilizationUnknown: its rate is held.
//
// A worker that is not limited (a limit of 100%) spends no tokens, so nothing
// can be told from them: its utilization is what its processes read, or the
// device's when it has the device to itself.
//
// Readings 2 and 3 take everything the device does for the work of the workers
// that spend tokens. unmetered says that the device also runs load no token
// bucket accounts for (a process outside every worker, a worker limited by
// other means). Those readings would then blame the metered workers for it and
// throttle them for what they do not do, down to the lowest rate; they are not
// used, and so is neither next to a worker that is not limited (a limit of
// 100%), whose limiter spends no tokens. Without per-process readings the
// limits of such workers are then only as good as the rate they had.
//
// The same holds for an interval in which what one of the limited workers
// spent is not known (see tokenMeter: a new bucket, an interval in which a
// limiter refilled itself because the hypervisor was held up, a balance its
// workload wrote). Splitting the device among the others would bill them for
// whatever that worker ran, and they would be throttled for it. The split is
// then unknown for everybody on the device for that interval: rates are held
// and tokens keep being refilled. What a worker does to its own balance thus
// cannot move a neighbor's rate.
//
// It could still keep the neighbors' rates where they are for as long as it
// goes on. A worker whose spending could not be told for a minute (see
// tokenMeter.excluded) is therefore left out of the split: it no longer makes
// the split unknown, and it counts as load no token bucket accounts for, like
// unmetered load above. Nothing can tell how much of the device it uses, so
// the device's utilization still cannot be divided among the others without
// billing them for some of it, and it is not: readings 2 and 3 stay unused for
// everybody on the device. What the others have left is their own. Their
// per-process readings, where the provider supplies them, are used as always.
// And a worker's own bucket tells exactly what it spent, which the excluded
// worker cannot touch; with what a token was found to buy that worker while
// the split was still known (its device-derived calibration, as it stood: it
// learns nothing meanwhile) that is its utilization, and the rate control
// works on it again. Such a worker is brought to the rate its share calls for,
// follows a change of its limit, and is reported with a utilization of its
// own. What it cannot follow without per-process readings is a change in what
// a token buys it, and a worker that has no calibration yet (it started next
// to the excluded one) keeps the rate it has: for both, nothing measures them.
// For the excluded worker itself nothing changes: its rate is held as in any
// interval in which its spending is not known, its bucket keeps being
// refilled at that rate, and it is taken back once its bucket has been
// readable for ten seconds.
//
// unaccounted is how much of the device's utilization (0-1) its processes have
// not accounted for lately, 0 when that is not known.
func attributeDeviceUtilization(
	members []*erlMember, deviceUtil float64, deviceSampled bool, processUtil map[string]float64, unmetered bool,
	unaccounted float64,
) {
	totalConsumption, totalWindow, unlimited, unmeasured, excluded := 0.0, 0.0, 0, 0, 0
	for _, m := range members {
		switch {
		case m.device.UpLimit >= 100:
			unlimited++
		case leftOut(m, len(members)):
			excluded++
		case m.meter.measured:
			totalConsumption += m.meter.consumption
			totalWindow += m.meter.window
		default:
			unmeasured++
		}
	}
	splitUnknown := len(members) > 1 && unmeasured > 0

	for _, m := range members {
		m.utilization, m.source, m.calibrated, m.gain = 0, utilizationUnknown, false, 0
		m.deviceLoad, m.deviceSpending, m.deviceRead, m.held, m.coarse = 0, 0, false, false, false

		if m.device.UpLimit >= 100 {
			attributeUnlimitedUtilization(m, len(members), deviceUtil, deviceSampled, processUtil, unmetered)
			continue
		}

		// The reading derived from the device, and its calibration.
		deviceSource, deviceReading, besideExcluded := utilizationUnknown, 0.0, false
		if othersUnmetered := unmetered || unlimited > 0; deviceSampled && !othersUnmetered && !splitUnknown {
			// Next to a worker that is left out the device is not read.
			if besideExcluded = excluded > 0; !besideExcluded {
				deviceReading, deviceSource = deviceDerivedReading(m, len(members), totalConsumption, deviceUtil)
			}
		}
		// A worker that has a calibration is measured through what it spent. In
		// an interval in which that is not known it has no figure at all; a raw
		// reading in its place would move its rate on noise.
		if !m.meter.measured && (m.calibration.calibrated() || m.fallback.calibrated()) {
			continue
		}
		deviceEstimate, deviceCalibrated := 0.0, false
		if deviceSource != utilizationUnknown {
			deviceEstimate, deviceCalibrated = m.fallback.estimate(deviceReading, m.meter)
			m.deviceLoad, m.deviceSpending, m.deviceRead = deviceUtil, totalWindow, true
		} else {
			deviceEstimate, deviceCalibrated, deviceSource = spendingEstimate(m, besideExcluded)
		}

		// The worker's own processes, and their calibration.
		processEstimate, processCalibrated := 0.0, false
		deviceBusy := deviceSampled && deviceUtil >= processCheckBusyUtil && unaccounted >= silentUnaccounted
		processReading, fromProcesses := processDerivedReading(m, processUtil, deviceBusy)
		m.coarse = fromProcesses
		if fromProcesses {
			if m.calibration != nil {
				ratio, known := m.fallback.recentRatio()
				m.calibration.lessConfirmed = known && ratio <= calibrationLessConfirmation
			}
			processEstimate, processCalibrated = m.calibration.estimate(processReading, m.meter)
		}

		deviceGain, bounded := boundedDeviceGain(m, deviceCalibrated, fromProcesses)

		switch {
		case processCalibrated:
			m.gain, m.source = m.calibration.gain(), utilizationFromProcesses
			if trust := m.calibration.trust(); deviceCalibrated && trust < 1 {
				m.gain = trust*m.gain + (1-trust)*deviceGain
			}
			m.utilization, m.calibrated = clampFloat64(m.gain*m.meter.consumption, 0, 1), true
		case deviceCalibrated && bounded:
			m.gain, m.source = deviceGain, utilizationFromProcesses
			m.utilization, m.calibrated = clampFloat64(m.gain*m.meter.consumption, 0, 1), true
		case deviceCalibrated:
			m.gain, m.source = deviceGain, deviceSource
			m.utilization, m.calibrated = deviceEstimate, true
		case fromProcesses:
			m.utilization, m.source = processEstimate, utilizationFromProcesses
		case deviceSource != utilizationUnknown:
			m.utilization, m.source = deviceEstimate, deviceSource
		}
	}
}

// leftOut reports whether a limited worker is left out of the token split of a
// device it shares with others; see tokenMeter.excluded.
func leftOut(m *erlMember, members int) bool {
	return members > 1 && m.meter.excluded
}

// spendingEstimate is the utilization of a limited worker next to one that is
// left out of the split (besideExcluded): what its own bucket shows it spent,
// by its device-derived calibration as it stands. It has none when its own
// spending is not known, when it is the one left out, or before it has a
// calibration.
func spendingEstimate(m *erlMember, besideExcluded bool) (estimate float64, ok bool, source utilizationSource) {
	if !besideExcluded || m.meter.excluded || !m.meter.measured || !m.fallback.calibrated() {
		return 0, false, utilizationUnknown
	}
	return clampFloat64(m.fallback.gain()*m.meter.consumption, 0, 1), true, utilizationFromSpending
}

// attributeUnlimitedUtilization sets the utilization of a worker without a
// compute limit: what its own processes read, or the device's utilization when
// it has the device to itself.
func attributeUnlimitedUtilization(
	m *erlMember, members int, deviceUtil float64, deviceSampled bool, processUtil map[string]float64, unmetered bool,
) {
	if percent, attributed := processUtil[m.workerUID]; attributed {
		m.utilization, m.source = clampFloat64(percent/100.0, 0, 1), utilizationFromProcesses
	} else if deviceSampled && !unmetered && members == 1 {
		m.utilization, m.source = deviceUtil, utilizationFromDevice
	}
}

// boundedDeviceGain is the gain of a limited worker's device-derived
// calibration, 0 when it has none, and whether the worker's own readings moved
// it; m.held is set when they lowered it.
func boundedDeviceGain(m *erlMember, deviceCalibrated, fromProcesses bool) (deviceGain float64, bounded bool) {
	// The device-derived figure counts as far as the worker's own readings
	// allow; see utilizationCalibration.bounds.
	if deviceCalibrated {
		deviceGain = m.fallback.gain()
		if least, most, known := m.calibration.bounds(); fromProcesses && known {
			bounded, m.held = deviceGain < least || deviceGain > most, deviceGain > most
			deviceGain = clampFloat64(deviceGain, least, most)
		}
	}
	return deviceGain, bounded
}

// deviceDerivedReading is a limited worker's share of the device as far as the
// device's own utilization tells it: all of it for a worker that has the device
// to itself, otherwise the part that corresponds to the tokens it spent. The
// caller has made sure that the device runs nothing but limited workers.
func deviceDerivedReading(
	m *erlMember, members int, totalConsumption, deviceUtil float64,
) (reading float64, source utilizationSource) {
	switch {
	case members == 1:
		return deviceUtil, utilizationFromDevice
	case !m.meter.measured:
		return 0, utilizationUnknown
	case totalConsumption <= 0:
		// Nothing was spent on this device: whatever keeps it busy is not
		// this worker.
		return 0, utilizationFromTokenShare
	default:
		return deviceUtil * m.meter.consumption / totalConsumption, utilizationFromTokenShare
	}
}

// processDerivedReading is a limited worker's utilization (0-1) as its own
// processes read, and whether those readings describe it. deviceBusy says that
// the device itself reports load that its processes do not account for.
func processDerivedReading(m *erlMember, processUtil map[string]float64, deviceBusy bool) (reading float64, ok bool) {
	if processUtil == nil {
		return 0, false
	}
	spending := m.meter.measured && m.meter.consumption > 0
	percent, attributed := processUtil[m.workerUID]
	// Spending tokens for a long time while its processes read zero on a busy
	// device, the worker is not what those readings describe (its work is
	// attributed to another process by the driver).
	if m.calibration != nil {
		if attributed && percent <= 0 && spending && deviceBusy {
			m.calibration.silent++
		} else {
			m.calibration.silent = 0
		}
		attributed = attributed && m.calibration.silent < silentSpendingIntervals
	}
	// A worker that spends tokens although none of its processes was attributed
	// to it is measured from its tokens instead of being taken for idle.
	if !attributed && spending {
		return 0, false
	}
	return clampFloat64(percent/100.0, 0, 1), true
}

// processCheck compares what the processes of a device report with what the
// device reports, both averaged over a few seconds.
type processCheck struct {
	process, device float64 // utilization, 0-1
	samples         int
}

// processUtilizationFor returns the per-worker utilization of a device from the
// provider's per-process figures, or nil when they cannot be used for it.
//
// The sample has to cover the device (see ProcessUtilizationSnapshot). Whether
// a provider measures per-process utilization at all is not guessed here: one
// that does not supplies none, and so does one that could not read it in this
// sample; the device is then not covered, and its workers are measured from
// the device's utilization for this interval. Nothing is remembered about it.
//
// Supplied figures still have to be plausible: the processes of a device add
// up to what the device does. They are averaged against the device's own
// utilization, and are not used while the device has been clearly busy and its
// processes account for less than half of that, as with a driver mode that
// books the work under something that is not a process of the device at all.
// Both averages move slowly, so a single sample in which a lightly used device
// reads zero for every process changes nothing, and the verdict follows the
// figures if they become plausible again. Without a device sample to check
// against, the figures are used as supplied.
func (c *Controller) processUtilizationFor(
	deviceUUID string, snapshot *ProcessUtilizationSnapshot, now time.Time, deviceUtil float64, deviceSampled bool,
) map[string]float64 {
	workers := snapshot.device(deviceUUID, now)
	if workers == nil {
		return nil
	}
	c.erlStatesMu.Lock()
	defer c.erlStatesMu.Unlock()
	check := c.processChecks[deviceUUID]
	if check == nil {
		check = &processCheck{}
		c.processChecks[deviceUUID] = check
	}
	if deviceSampled {
		total := clampFloat64(snapshot.Totals[deviceUUID]/100, 0, 1)
		if check.samples == 0 {
			check.process, check.device = total, deviceUtil
		} else {
			check.process += processCheckAlpha * (total - check.process)
			check.device += processCheckAlpha * (deviceUtil - check.device)
		}
		check.samples++
	}
	if check.samples > 0 && check.device >= processCheckBusyUtil && check.process < processCheckMinShare*check.device {
		return nil
	}
	return workers
}

// processShortfall is how much of a device's utilization (0-1) its processes
// have not accounted for lately, by the averages of processUtilizationFor.
func (c *Controller) processShortfall(deviceUUID string) float64 {
	c.erlStatesMu.Lock()
	defer c.erlStatesMu.Unlock()
	check := c.processChecks[deviceUUID]
	if check == nil || check.samples == 0 {
		return 0
	}
	return math.Max(check.device-check.process, 0)
}
