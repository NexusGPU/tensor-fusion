package api

// WorkerComputeUsage is a worker's compute utilization on one device,
// accumulated over time. It only ever grows, so any number of readers can each
// take the average over their own interval from two readings:
//
//	mean utilization (percent of the device) = ΔPercentSeconds / ΔSeconds
//
// Seconds only advances while a utilization signal exists. A reading whose
// Seconds did not advance since the previous one therefore means "not
// measured", which is not the same as an idle worker.
type WorkerComputeUsage struct {
	// PercentSeconds is the integral of the worker's utilization, in percent of
	// the device, over the observed time.
	PercentSeconds float64
	// Seconds is the time the utilization was observed for.
	Seconds float64
}

// MeanPercentSince returns the mean utilization (percent of the device) between
// an earlier reading and this one. It reports false when nothing was observed
// in between.
func (u WorkerComputeUsage) MeanPercentSince(previous WorkerComputeUsage) (float64, bool) {
	seconds := u.Seconds - previous.Seconds
	if seconds <= 0 {
		return 0, false
	}
	mean := (u.PercentSeconds - previous.PercentSeconds) / seconds
	if mean < 0 {
		mean = 0
	}
	return mean, true
}

// TflopsInUse converts a utilization, in percent of a device, into TFLOPS in
// use on that device: the share of the device's TFLOPS capacity. It is the one
// definition behind every "compute_tflops" figure. It reports false when the
// device's capacity is unknown.
func TflopsInUse(utilizationPercent float64, device *DeviceInfo) (float64, bool) {
	capacity := device.EffectiveTflopsCapacity()
	if capacity <= 0 || utilizationPercent < 0 {
		return 0, false
	}
	return utilizationPercent / 100 * capacity, true
}
