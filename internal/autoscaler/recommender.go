package autoscaler

import (
	"time"
)

const (
	// Fraction of usage added as the safety margin to the recommended request
	defaultRequestMarginFraction = 0.15
	// Vram usage percentile that will be used as a base for vram target recommendation. Doesn't affect vram lower bound nor vram upper bound.
	defaultTargetVramPercentile = 0.9
	// Vram usage percentile that will be used for the lower bound on vram recommendation.
	defaultLowerBoundVramPercentile = 0.5
	// Vram usage percentile that will be used for the upper bound on vram recommendation.
	defaultUpperBoundVramPercentile = 0.95
	// Tflops usage percentile that will be used as a base for tflops target recommendation. Doesn't affect tflops lower bound nor tflops upper bound.
	defaultTargetTflopsPercentile = 0.9
	// Tflops usage percentile that will be used for the lower bound on tflops recommendation.
	defaultLowerBoundTflopsPercentile = 0.5
	// Tflops usage percentile that will be used for the upper bound on tflops recommendation.
	defaultUpperBoundTflopsPercentile = 0.95
	// The time interval used for computing the confidence multiplier for the lower and upper bound. Default: 24h
	defaultConfidenceInterval = time.Hour * 24
)

var DefaultResourceRecommenderConfig = ResourceRecommenderConfig{
	TargetTflopsPercentile:     defaultTargetTflopsPercentile,
	LowerBoundTflopsPercentile: defaultLowerBoundTflopsPercentile,
	UpperBoundTflopsPercentile: defaultUpperBoundTflopsPercentile,
	TargetVramPercentile:       defaultTargetVramPercentile,
	LowerBoundVramPercentile:   defaultLowerBoundVramPercentile,
	UpperBoundVramPercentile:   defaultUpperBoundVramPercentile,
	RequestMarginFraction:      defaultRequestMarginFraction,
	ConfidenceInterval:         defaultConfidenceInterval,
}

type ResourceRecommender interface {
	GetRecommendedResources(*WorkloadState) *RecommendedResources
}

type RecommendedResources struct {
	LowerBoundTflops ResourceAmount
	TargetTflops     ResourceAmount
	UpperBoundTflops ResourceAmount
	LowerBoundVram   ResourceAmount
	TargetVram       ResourceAmount
	UpperBoundVram   ResourceAmount
}

type ResourceRecommenderConfig struct {
	TargetTflopsPercentile     float64
	LowerBoundTflopsPercentile float64
	UpperBoundTflopsPercentile float64
	TargetVramPercentile       float64
	LowerBoundVramPercentile   float64
	UpperBoundVramPercentile   float64
	RequestMarginFraction      float64
	ConfidenceInterval         time.Duration
}

func NewResourceRecommender() ResourceRecommender {
	return &resourceRecommender{}
}

type resourceRecommender struct {
	lowerBoundTflops TflopsEstimator
	targetTflops     TflopsEstimator
	upperBoundTflops TflopsEstimator
	lowerBoundVram   VramEstimator
	targetVram       VramEstimator
	upperBoundVram   VramEstimator
}

func (r *resourceRecommender) GetRecommendedResources(s *WorkloadState) *RecommendedResources {

	r.createEstimatorsFromConfig(s.GetResourceRecommenderConfig())

	return &RecommendedResources{
		LowerBoundTflops: r.lowerBoundTflops.GetTflopsEstimation(s),
		TargetTflops:     r.targetTflops.GetTflopsEstimation(s),
		UpperBoundTflops: r.upperBoundTflops.GetTflopsEstimation(s),
		LowerBoundVram:   r.lowerBoundVram.GetVramEstimation(s),
		TargetVram:       r.targetVram.GetVramEstimation(s),
		UpperBoundVram:   r.upperBoundVram.GetVramEstimation(s),
	}
}

func (r *resourceRecommender) createEstimatorsFromConfig(config *ResourceRecommenderConfig) {
	targetTflops := NewPercentileTflopsEstimator(config.TargetTflopsPercentile)
	lowerBoundTflops := NewPercentileTflopsEstimator(config.LowerBoundTflopsPercentile)
	upperBoundTflops := NewPercentileTflopsEstimator(config.UpperBoundTflopsPercentile)

	targetTflops = WithTflopsMargin(config.RequestMarginFraction, targetTflops)
	lowerBoundTflops = WithTflopsMargin(config.RequestMarginFraction, lowerBoundTflops)
	upperBoundTflops = WithTflopsMargin(config.RequestMarginFraction, upperBoundTflops)

	upperBoundTflops = WithTflopsConfidenceMultiplier(1.0, 1.0, upperBoundTflops, config.ConfidenceInterval)
	lowerBoundTflops = WithTflopsConfidenceMultiplier(0.001, -2.0, lowerBoundTflops, config.ConfidenceInterval)

	targetVram := NewPercentileVramEstimator(config.TargetVramPercentile)
	lowerBoundVram := NewPercentileVramEstimator(config.LowerBoundVramPercentile)
	upperBoundVram := NewPercentileVramEstimator(config.UpperBoundVramPercentile)

	targetVram = WithVramMargin(config.RequestMarginFraction, targetVram)
	lowerBoundVram = WithVramMargin(config.RequestMarginFraction, lowerBoundVram)
	upperBoundVram = WithVramMargin(config.RequestMarginFraction, upperBoundVram)

	upperBoundVram = WithVramConfidenceMultiplier(1.0, 1.0, upperBoundVram, config.ConfidenceInterval)
	lowerBoundVram = WithVramConfidenceMultiplier(0.001, -2.0, lowerBoundVram, config.ConfidenceInterval)

	*r = resourceRecommender{
		lowerBoundTflops: lowerBoundTflops,
		targetTflops:     targetTflops,
		upperBoundTflops: upperBoundTflops,
		lowerBoundVram:   lowerBoundVram,
		targetVram:       targetVram,
		upperBoundVram:   upperBoundVram,
	}
}
