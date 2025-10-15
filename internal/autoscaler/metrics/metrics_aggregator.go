package metrics

import (
	"time"

	vpa "k8s.io/autoscaler/vertical-pod-autoscaler/pkg/recommender/util"
)

const (
	// minSampleWeight is the minimal weight of any sample (prior to including decaying factor)
	minSampleWeight = 0.1
	// epsilon is the minimal weight kept in histograms, it should be small enough that old samples
	// (just inside AggregationWindowLength) added with minSampleWeight are still kept
	epsilon = 0.001 * minSampleWeight
	// DefaultAggregationInterval is the default value for AggregationInterval.
	DefaultAggregationInterval = time.Hour * 24
	// DefaultHistogramBucketSizeGrowth is the default value for HistogramBucketSizeGrowth.
	DefaultHistogramBucketSizeGrowth = 0.05 // Make each bucket 5% larger than the previous one.
	// DefaultHistogramDecayHalfLife is the default value for HistogramDecayHalfLife.
	DefaultHistogramDecayHalfLife = time.Hour * 24
)

type WorkerUsageAggregator struct {
	TflopsHistogram   vpa.Histogram
	VramHistogram     vpa.Histogram
	FirstSampleStart  time.Time
	LastSampleStart   time.Time
	TotalSamplesCount int
}

func NewWorkerUsageAggregator() *WorkerUsageAggregator {
	return &WorkerUsageAggregator{
		TflopsHistogram: vpa.NewDecayingHistogram(histogramOptions(10000.0, 0.1), DefaultHistogramDecayHalfLife),
		VramHistogram:   vpa.NewDecayingHistogram(histogramOptions(1e12, 1e7), DefaultHistogramDecayHalfLife),
	}
}

func (w *WorkerUsageAggregator) IsEmpty() bool {
	return w.TflopsHistogram.IsEmpty() && w.VramHistogram.IsEmpty()
}

func (w *WorkerUsageAggregator) AddTflopsSample(sample *WorkerUsage) bool {
	w.TflopsHistogram.AddSample(float64(sample.TflopsUsage), minSampleWeight, sample.Timestamp)
	if sample.Timestamp.After(w.LastSampleStart) {
		w.LastSampleStart = sample.Timestamp
	}
	if w.FirstSampleStart.IsZero() || sample.Timestamp.Before(w.FirstSampleStart) {
		w.FirstSampleStart = sample.Timestamp
	}
	w.TotalSamplesCount++
	return true
}

func (w *WorkerUsageAggregator) AddVramSample(sample *WorkerUsage) bool {
	w.VramHistogram.AddSample(float64(sample.VramUsage), 1.0, sample.Timestamp)
	return true
}

func (w *WorkerUsageAggregator) SubtractVramSample(usage float64, time time.Time) bool {
	w.VramHistogram.SubtractSample(usage, 1.0, time)
	return true
}

func histogramOptions(maxValue, firstBucketSize float64) vpa.HistogramOptions {
	options, err := vpa.NewExponentialHistogramOptions(maxValue, firstBucketSize, 1.+DefaultHistogramBucketSizeGrowth, epsilon)
	if err != nil {
		panic("Invalid histogram options") // Should not happen.
	}
	return options
}
