package autoscaling

import (
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaling/metrics"
	"k8s.io/apimachinery/pkg/api/resource"
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

type RecommendedResources struct {
	LowerBoundTflops resource.Quantity
	TargetTflops     resource.Quantity
	UpperBoundTflops resource.Quantity
	LowerBoundVram   resource.Quantity
	TargetVram       resource.Quantity
	UpperBoundVram   resource.Quantity
}

type WorkloadState struct {
	Namespace         string
	Name              string
	Resources         tfv1.Resources
	AutoScalingConfig tfv1.AutoScalingConfig
	Recommendation    RecommendedResources

	TflopsHistogram   vpa.Histogram
	VramHistogram     vpa.Histogram
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

func (w *WorkloadState) UpdateSampleStats(sample *metrics.WorkerUsage) {
	if sample.Timestamp.After(w.LastSampleStart) {
		w.LastSampleStart = sample.Timestamp
	}
	if w.FirstSampleStart.IsZero() || sample.Timestamp.Before(w.FirstSampleStart) {
		w.FirstSampleStart = sample.Timestamp
	}
	w.TotalSamplesCount++
}

func (w *WorkloadState) IsTargetResource(name tfv1.ResourceName) bool {
	target := w.AutoScalingConfig.AutoSetResources.TargetResource
	return target == "" || strings.EqualFold(target, "all") || strings.EqualFold(string(name), target)
}

func (w *WorkloadState) IsAutoScalingEnabled() bool {
	return w.AutoScalingConfig.AutoSetResources.Enable
}
