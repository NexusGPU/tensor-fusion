package recommender

import (
	"math"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	// MaxResourceAmount is the maximum allowed value of resource amount.
	MaxResourceAmount = ResourceAmount(1e14)
)

type ResourceAmount int64

type VramEstimator interface {
	GetVramEstimation(w *metrics.WorkerUsageAggregator) ResourceAmount
}

type percentileVramEstimator struct {
	percentile float64
}

// NewPercentileVramEstimator returns a new percentileVramEstimator that uses provided percentile.
func NewPercentileVramEstimator(percentile float64) VramEstimator {
	return &percentileVramEstimator{percentile}
}

func (e *percentileVramEstimator) GetVramEstimation(w *metrics.WorkerUsageAggregator) ResourceAmount {
	return resourceAmountFromFloat(float64(w.VramHistogram.Percentile(e.percentile)))
}

type vramMarginEstimator struct {
	marginFraction float64
	baseEstimator  VramEstimator
}

// WithvramMargin returns a vramEstimator that adds a margin to the base estimator.
func WithVramMargin(marginFraction float64, baseEstimator VramEstimator) VramEstimator {
	return &vramMarginEstimator{marginFraction: marginFraction, baseEstimator: baseEstimator}
}

// GetvramEstimation returns the vram estimation for the given AggregateContainerState.
func (e *vramMarginEstimator) GetVramEstimation(w *metrics.WorkerUsageAggregator) ResourceAmount {
	base := e.baseEstimator.GetVramEstimation(w)
	margin := resourceAmountFromFloat(float64(base) * e.marginFraction)
	return base + margin
}

type vramConfidenceMultiplier struct {
	multiplier         float64
	exponent           float64
	baseEstimator      VramEstimator
	confidenceInterval time.Duration
}

// WithVramConfidenceMultiplier returns a VramEstimator that scales the
func WithVramConfidenceMultiplier(multiplier, exponent float64, baseEstimator VramEstimator, confidenceInterval time.Duration) VramEstimator {
	return &vramConfidenceMultiplier{
		multiplier:         multiplier,
		exponent:           exponent,
		baseEstimator:      baseEstimator,
		confidenceInterval: confidenceInterval,
	}
}

func (e *vramConfidenceMultiplier) GetVramEstimation(w *metrics.WorkerUsageAggregator) ResourceAmount {
	confidence := getConfidence(w, e.confidenceInterval)
	base := e.baseEstimator.GetVramEstimation(w)
	return resourceAmountFromFloat(float64(base) * math.Pow(1.+e.multiplier/confidence, e.exponent))
}

type TflopsEstimator interface {
	GetTflopsEstimation(w *metrics.WorkerUsageAggregator) ResourceAmount
}

type percentileTflopsEstimator struct {
	percentile float64
}

// NewPercentileTflopsEstimator returns a new percentileTflopsEstimator that uses provided percentile.
func NewPercentileTflopsEstimator(percentile float64) TflopsEstimator {
	return &percentileTflopsEstimator{percentile}
}

func (e *percentileTflopsEstimator) GetTflopsEstimation(w *metrics.WorkerUsageAggregator) ResourceAmount {
	return resourceAmountFromFloat(float64(w.TflopsHistogram.Percentile(e.percentile)))
}

type tflopsMarginEstimator struct {
	marginFraction float64
	baseEstimator  TflopsEstimator
}

// WithTflopsMargin returns a tflopsEstimator that adds a margin to the base estimator.
func WithTflopsMargin(marginFraction float64, baseEstimator TflopsEstimator) TflopsEstimator {
	return &tflopsMarginEstimator{marginFraction: marginFraction, baseEstimator: baseEstimator}
}

// GetTflopsEstimation returns the tflops estimation for the given AggregateContainerState.
func (e *tflopsMarginEstimator) GetTflopsEstimation(w *metrics.WorkerUsageAggregator) ResourceAmount {
	base := e.baseEstimator.GetTflopsEstimation(w)
	margin := resourceAmountFromFloat(float64(base) * e.marginFraction)
	return base + margin
}

type tflopsConfidenceMultiplier struct {
	multiplier         float64
	exponent           float64
	baseEstimator      TflopsEstimator
	confidenceInterval time.Duration
}

// WithTflopsConfidenceMultiplier returns a TflopsEstimator that scales the
func WithTflopsConfidenceMultiplier(multiplier, exponent float64, baseEstimator TflopsEstimator, confidenceInterval time.Duration) TflopsEstimator {
	return &tflopsConfidenceMultiplier{
		multiplier:         multiplier,
		exponent:           exponent,
		baseEstimator:      baseEstimator,
		confidenceInterval: confidenceInterval,
	}
}

func (e *tflopsConfidenceMultiplier) GetTflopsEstimation(w *metrics.WorkerUsageAggregator) ResourceAmount {
	confidence := getConfidence(w, e.confidenceInterval)
	base := e.baseEstimator.GetTflopsEstimation(w)
	return resourceAmountFromFloat(float64(base) * math.Pow(1.+e.multiplier/confidence, e.exponent))
}

// Returns a non-negative real number that heuristically measures how much
// confidence the history aggregated in the AggregateState provides.
// For a workload producing a steady stream of samples over N days at the rate
// of 1 sample per minute, this metric is equal to N.
// This implementation is a very simple heuristic which looks at the total count
// of samples and the time between the first and the last sample.
func getConfidence(w *metrics.WorkerUsageAggregator, confidenceInterval time.Duration) float64 {
	// Distance between the first and the last observed sample time, measured in days.
	lifespanInDays := float64(w.LastSampleStart.Sub(w.FirstSampleStart)) / float64(confidenceInterval)
	// Total count of samples normalized such that it equals the number of days for
	// frequency of 1 sample/minute.
	samplesAmount := float64(w.TotalSamplesCount) / confidenceInterval.Minutes()
	return math.Min(lifespanInDays, samplesAmount)
}

// ResourceAmountMax returns the larger of two resource amounts.
func ResourceAmountMax(amount1, amount2 ResourceAmount) ResourceAmount {
	if amount1 > amount2 {
		return amount1
	}
	return amount2
}

func QuantityFromAmount(amount ResourceAmount, format resource.Format) resource.Quantity {
	return *resource.NewQuantity(int64(amount), format)
}

func resourceAmountFromFloat(amount float64) ResourceAmount {
	if amount < 0 {
		return ResourceAmount(0)
	} else if amount > float64(MaxResourceAmount) {
		return MaxResourceAmount
	} else {
		return ResourceAmount(amount)
	}
}
