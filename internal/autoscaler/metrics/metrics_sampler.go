package metrics

import (
	"time"
)

type WorkerUsageSampler struct {
	LastTflopsSampleTime time.Time
	// TflopsMeasured is set once the worker reported a compute figure above
	// zero. Until then its zero figures are not sampled: releases before the
	// hypervisor computed the figure wrote zero into every usage row, and a
	// worker whose rows never held anything else may just have no real figure
	// yet. From the first real figure on, a zero is a measured idle interval.
	//
	// The flag lives with the sampler, in memory. After an operator restart it
	// is learned again from the usage history, which is replayed oldest first;
	// a worker that was idle for the whole history period contributes no
	// compute samples until it reports a figure above zero again.
	TflopsMeasured     bool
	VramPeak           uint64
	LastVramSampleTime time.Time
	VramWindowEnd      time.Time
}

func NewWorkerUsageSampler() *WorkerUsageSampler {
	return &WorkerUsageSampler{
		LastTflopsSampleTime: time.Time{},
		LastVramSampleTime:   time.Time{},
		VramWindowEnd:        time.Time{},
	}
}

func (w *WorkerUsageSampler) AddSample(aggregator *WorkerUsageAggregator, sample *WorkerUsage) bool {
	w.AddTflopsSample(aggregator, sample)
	w.AddVramSample(aggregator, sample)
	return true
}

func (w *WorkerUsageSampler) AddTflopsSample(aggregator *WorkerUsageAggregator, sample *WorkerUsage) bool {
	// An unmeasured utilization is no sample; counting it as zero would pull
	// every recommendation down.
	if sample.TflopsUnknown || sample.TflopsUsage < 0 || sample.Timestamp.Before(w.LastTflopsSampleTime) {
		return false
	}
	if !w.TflopsMeasured {
		if sample.TflopsUsage == 0 {
			return false
		}
		w.TflopsMeasured = true
	}
	aggregator.AddTflopsSample(sample)
	w.LastTflopsSampleTime = sample.Timestamp
	return true
}

func (w *WorkerUsageSampler) AddVramSample(aggregator *WorkerUsageAggregator, sample *WorkerUsage) bool {
	ts := sample.Timestamp
	if ts.Before(w.LastVramSampleTime) {
		return false
	}
	w.LastVramSampleTime = ts
	if w.VramWindowEnd.IsZero() {
		w.VramWindowEnd = ts
	}

	addNewPeak := false
	if ts.Before(w.VramWindowEnd) {
		if sample.VramUsage > w.VramPeak {
			aggregator.SubtractVramSample(float64(w.VramPeak), w.VramWindowEnd)
			addNewPeak = true
		}
	} else {
		aggregationInteval := DefaultAggregationInterval
		shift := ts.Sub(w.VramWindowEnd).Truncate(aggregationInteval) + aggregationInteval
		w.VramWindowEnd = w.VramWindowEnd.Add(shift)
		w.VramPeak = 0
		addNewPeak = true
	}

	if addNewPeak {
		aggregator.AddVramSample(sample)
		w.VramPeak = sample.VramUsage
	}

	return true
}
