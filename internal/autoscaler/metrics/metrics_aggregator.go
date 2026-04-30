package metrics

import (
	"sync"
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
)

// WorkerUsageAggregator owns the per-workload tflops/vram histograms.
//
// It is shared between the metrics-loader goroutines (which call AddTflopsSample /
// AddVramSample / SubtractVramSample) and the autoscaler recommender path
// (which reads via TflopsPercentile / VramPercentile / IsEmpty). The internal
// mu serializes those, so callers no longer need to hold State.Mu while
// reading the histograms.
type WorkerUsageAggregator struct {
	mu                sync.RWMutex
	tflopsHistogram   vpa.Histogram
	vramHistogram     vpa.Histogram
	firstSampleStart  time.Time
	lastSampleStart   time.Time
	totalSamplesCount int
}

func NewWorkerUsageAggregator(decayHalfTime time.Duration) *WorkerUsageAggregator {
	return &WorkerUsageAggregator{
		tflopsHistogram: vpa.NewDecayingHistogram(histogramOptions(10000.0, 0.1), decayHalfTime),
		vramHistogram:   vpa.NewDecayingHistogram(histogramOptions(1e12, 1e7), decayHalfTime),
	}
}

func (w *WorkerUsageAggregator) IsEmpty() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.tflopsHistogram.IsEmpty() && w.vramHistogram.IsEmpty()
}

// TflopsIsEmpty reports whether the tflops histogram has no recorded samples.
func (w *WorkerUsageAggregator) TflopsIsEmpty() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.tflopsHistogram.IsEmpty()
}

// VramIsEmpty reports whether the vram histogram has no recorded samples.
func (w *WorkerUsageAggregator) VramIsEmpty() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.vramHistogram.IsEmpty()
}

// TflopsPercentile returns the requested percentile from the tflops histogram.
func (w *WorkerUsageAggregator) TflopsPercentile(percentile float64) float64 {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.tflopsHistogram.Percentile(percentile)
}

// VramPercentile returns the requested percentile from the vram histogram.
func (w *WorkerUsageAggregator) VramPercentile(percentile float64) float64 {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.vramHistogram.Percentile(percentile)
}

// FirstSampleStart returns the timestamp of the earliest seen sample.
func (w *WorkerUsageAggregator) FirstSampleStart() time.Time {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.firstSampleStart
}

// LastSampleStart returns the timestamp of the latest seen sample.
func (w *WorkerUsageAggregator) LastSampleStart() time.Time {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.lastSampleStart
}

// TotalSamplesCount returns the number of samples ingested so far.
func (w *WorkerUsageAggregator) TotalSamplesCount() int {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.totalSamplesCount
}

func (w *WorkerUsageAggregator) AddTflopsSample(sample *WorkerUsage) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.tflopsHistogram.AddSample(float64(sample.TflopsUsage), minSampleWeight, sample.Timestamp)
	if sample.Timestamp.After(w.lastSampleStart) {
		w.lastSampleStart = sample.Timestamp
	}
	if w.firstSampleStart.IsZero() || sample.Timestamp.Before(w.firstSampleStart) {
		w.firstSampleStart = sample.Timestamp
	}
	w.totalSamplesCount++
	return true
}

func (w *WorkerUsageAggregator) AddVramSample(sample *WorkerUsage) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.vramHistogram.AddSample(float64(sample.VramUsage), 1.0, sample.Timestamp)
	return true
}

func (w *WorkerUsageAggregator) SubtractVramSample(usage float64, time time.Time) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.vramHistogram.SubtractSample(usage, 1.0, time)
	return true
}

func histogramOptions(maxValue, firstBucketSize float64) vpa.HistogramOptions {
	options, err := vpa.NewExponentialHistogramOptions(maxValue, firstBucketSize, 1.+DefaultHistogramBucketSizeGrowth, epsilon)
	if err != nil {
		panic("Invalid histogram options") // Should not happen.
	}
	return options
}
