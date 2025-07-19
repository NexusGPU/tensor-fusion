package workload

import (
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
)

type WorkerState struct {
	Name                 string
	WorkloadName         string
	LastTflopsSampleTime time.Time

	VramPeak           uint64
	LastVramSampleTime time.Time
	VramWindowEnd      time.Time
}

func NewWorkerState(name string, workloadName string) *WorkerState {
	return &WorkerState{
		Name:                 name,
		WorkloadName:         workloadName,
		LastTflopsSampleTime: time.Time{},
		LastVramSampleTime:   time.Time{},
		VramWindowEnd:        time.Time{},
	}
}

func (w *WorkerState) AddSample(aggregator *metrics.WorkerUsageAggregator, sample *metrics.WorkerUsage) bool {
	w.AddTflopsSample(aggregator, sample)
	w.AddVramSample(aggregator, sample)
	return true
}

func (w *WorkerState) AddTflopsSample(aggregator *metrics.WorkerUsageAggregator, sample *metrics.WorkerUsage) bool {
	if sample.Timestamp.Before(w.LastTflopsSampleTime) {
		return false
	}
	aggregator.AddTflopsSample(sample)
	w.LastTflopsSampleTime = sample.Timestamp
	return true
}

func (w *WorkerState) AddVramSample(aggregator *metrics.WorkerUsageAggregator, sample *metrics.WorkerUsage) bool {
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
		aggregationInteval := metrics.DefaultAggregationInterval
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
