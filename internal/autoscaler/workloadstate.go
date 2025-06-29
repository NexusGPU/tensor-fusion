package autoscaler

import (
	"strconv"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
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

type WorkloadState struct {
	Namespace         string
	Name              string
	Resources         tfv1.Resources
	AutoScalingConfig tfv1.AutoScalingConfig

	TflopsHistogram vpa.Histogram
	VramHistogram   vpa.Histogram

	FirstSampleStart  time.Time
	LastSampleStart   time.Time
	TotalSamplesCount int
	CreationTime      time.Time
}

func NewWorkloadState(name string) *WorkloadState {
	return &WorkloadState{
		Name:            name,
		TflopsHistogram: vpa.NewDecayingHistogram(histogramOptions(10000.0, 0.1), DefaultHistogramDecayHalfLife),
		VramHistogram:   vpa.NewDecayingHistogram(histogramOptions(1e12, 1e7), DefaultHistogramDecayHalfLife),
		CreationTime:    time.Now(),
	}
}

func histogramOptions(maxValue, firstBucketSize float64) vpa.HistogramOptions {
	options, err := vpa.NewExponentialHistogramOptions(maxValue, firstBucketSize, 1.+DefaultHistogramBucketSizeGrowth, epsilon)
	if err != nil {
		panic("Invalid histogram options") // Should not happen.
	}
	return options
}

func (w *WorkloadState) UpdateSampleStats(metrics *WorkerMetrics) {
	if metrics.Timestamp.After(w.LastSampleStart) {
		w.LastSampleStart = metrics.Timestamp
	}
	if w.FirstSampleStart.IsZero() || metrics.Timestamp.Before(w.FirstSampleStart) {
		w.FirstSampleStart = metrics.Timestamp
	}
	w.TotalSamplesCount++
}

func (w *WorkloadState) GetResourceRecommenderConfig() *ResourceRecommenderConfig {
	cfg := DefaultResourceRecommenderConfig

	asr := w.AutoScalingConfig.AutoSetResources
	fields := []struct {
		val string
		dst *float64
	}{
		{asr.TargetTflopsPercentile, &cfg.TargetTflopsPercentile},
		{asr.LowerBoundTflopsPercentile, &cfg.LowerBoundTflopsPercentile},
		{asr.UpperBoundTflopsPercentile, &cfg.UpperBoundTflopsPercentile},
		{asr.TargetVramPercentile, &cfg.TargetVramPercentile},
		{asr.LowerBoundVramPercentile, &cfg.LowerBoundVramPercentile},
		{asr.UpperBoundVramPercentile, &cfg.UpperBoundVramPercentile},
		{asr.RequestMarginFraction, &cfg.RequestMarginFraction},
	}
	for _, f := range fields {
		if f.val == "" {
			continue
		}
		if v, err := strconv.ParseFloat(f.val, 64); err == nil {
			*f.dst = v
		}
	}

	if asr.ConfidenceInterval != "" {
		if d, err := time.ParseDuration(asr.ConfidenceInterval); err == nil {
			cfg.ConfidenceInterval = d
		}
	}

	return &cfg
}

func (w *WorkloadState) IsTargetResource(resourceName string) bool {
	target := w.AutoScalingConfig.AutoSetResources.TargetResource
	return target == "" || strings.EqualFold(target, "all") || strings.EqualFold(resourceName, target)
}

func (w *WorkloadState) IsAutoScalingEnabled() bool {
	return w.AutoScalingConfig.AutoSetResources.Enable
}
