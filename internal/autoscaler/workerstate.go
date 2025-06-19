package autoscaler

import (
	"time"
)

type WorkerState struct {
	Name                 string
	Workload             string
	TflopsPeak           ResourceAmount
	LastTflopsSampleTime time.Time
	TflopsWindowEnd      time.Time

	VramPeak           ResourceAmount
	LastVramSampleTime time.Time
	VramWindowEnd      time.Time
}

func NewWorkerState(name string, workload string) *WorkerState {
	return &WorkerState{
		Name:                 name,
		Workload:             workload,
		LastTflopsSampleTime: time.Time{},
		TflopsWindowEnd:      time.Time{},
		LastVramSampleTime:   time.Time{},
		VramWindowEnd:        time.Time{},
	}
}

func (w *WorkerState) AddTflopsSample(workload *WorkloadState, metrics *WorkerMetrics) bool {
	ts := metrics.Timestamp
	if ts.Before(w.LastTflopsSampleTime) {
		return false
	}
	w.LastTflopsSampleTime = ts
	if w.TflopsWindowEnd.IsZero() {
		w.TflopsWindowEnd = ts
	}

	addNewPeak := false
	if ts.Before(w.TflopsWindowEnd) {
		if w.TflopsPeak != 0 && metrics.TflopsUsage > w.TflopsPeak {
			workload.TflopsHistogram.SubtractSample(float64(w.TflopsPeak), 1.0, w.TflopsWindowEnd)
			addNewPeak = true
		}
	} else {
		aggregationInteval := DefaultAggregationInterval
		shift := ts.Sub(w.TflopsWindowEnd).Truncate(aggregationInteval) + aggregationInteval
		w.TflopsWindowEnd = w.TflopsWindowEnd.Add(shift)
		w.TflopsPeak = 0
		addNewPeak = true
	}

	if addNewPeak {
		workload.TflopsHistogram.AddSample(float64(metrics.TflopsUsage), 1.0, metrics.Timestamp)
		w.TflopsPeak = metrics.TflopsUsage
	}

	return true
}

func (w *WorkerState) AddVramSample(workload *WorkloadState, metrics *WorkerMetrics) bool {
	ts := metrics.Timestamp
	if ts.Before(w.LastVramSampleTime) {
		return false
	}
	w.LastVramSampleTime = ts
	if w.VramWindowEnd.IsZero() {
		w.VramWindowEnd = ts
	}

	addNewPeak := false
	if ts.Before(w.VramWindowEnd) {
		if w.VramPeak != 0 && metrics.VramUsage > w.VramPeak {
			workload.VramHistogram.SubtractSample(float64(w.VramPeak), 1.0, w.VramWindowEnd)
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
		workload.VramHistogram.AddSample(float64(metrics.VramUsage), 1.0, metrics.Timestamp)
		w.VramPeak = metrics.VramUsage
	}

	return true
}
