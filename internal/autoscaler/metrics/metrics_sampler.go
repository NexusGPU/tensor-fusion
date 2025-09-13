package metrics

import (
	"time"
)

type WorkerUsageSampler struct {
	LastTflopsSampleTime time.Time
	VramPeak             uint64
	LastVramSampleTime   time.Time
	VramWindowEnd        time.Time
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
	if sample.TflopsUsage < 0 || sample.Timestamp.Before(w.LastTflopsSampleTime) {
		return false
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
		if w.VramPeak != 0 && sample.VramUsage > w.VramPeak {
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
