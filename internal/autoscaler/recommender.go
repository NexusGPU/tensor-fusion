package autoscaler

import (
	"flag"
	"time"
)

var (
	safetyMarginFraction       = flag.Float64("recommendation-margin-fraction", 0.15, `Fraction of usage added as the safety margin to the recommended request`)
	targetVramPercentile       = flag.Float64("target-vram-percentile", 0.9, "Vram usage percentile that will be used as a base for vram target recommendation. Doesn't affect vram lower bound nor vram upper bound.")
	lowerBoundVramPercentile   = flag.Float64("recommendation-lower-bound-vram-percentile", 0.5, `Vram usage percentile that will be used for the lower bound on vram recommendation.`)
	upperBoundVramPercentile   = flag.Float64("recommendation-upper-bound-vram-percentile", 0.95, `Vram usage percentile that will be used for the upper bound on vram recommendation.`)
	targetTflopsPercentile     = flag.Float64("target-tflops-percentile", 0.9, "Tflops usage percentile that will be used as a base for tflops target recommendation. Doesn't affect tflops lower bound nor tflops upper bound.")
	lowerBoundTflopsPercentile = flag.Float64("recommendation-lower-bound-tflops-percentile", 0.5, `Tflops usage percentile that will be used for the lower bound on tflops recommendation.`)
	upperBoundTflopsPercentile = flag.Float64("recommendation-upper-bound-tflops-percentile", 0.95, `Tflops usage percentile that will be used for the upper bound on tflops recommendation.`)
	confidenceInterval         = flag.Duration("confidence-interval", time.Hour*24, "The time interval used for computing the confidence multiplier for the lower and upper bound. Default: 24h")
)

type Recommender interface {
	GetRecommendedResources(*WorkloadState) *RecommendedResources
}

type RecommendedResources struct {
	TargetTflops     ResourceAmount
	LowerBoundTflops ResourceAmount
	UpperBoundTflops ResourceAmount

	TargetVram     ResourceAmount
	LowerBoundVram ResourceAmount
	UpperBoundVram ResourceAmount
}

func NewRecommender() Recommender {
	targetTflops := NewPercentileTflopsEstimator(*targetTflopsPercentile)
	lowerBoundTflops := NewPercentileTflopsEstimator(*lowerBoundTflopsPercentile)
	upperBoundTflops := NewPercentileTflopsEstimator(*upperBoundTflopsPercentile)

	targetTflops = WithTflopsMargin(*safetyMarginFraction, targetTflops)
	lowerBoundTflops = WithTflopsMargin(*safetyMarginFraction, lowerBoundTflops)
	upperBoundTflops = WithTflopsMargin(*safetyMarginFraction, upperBoundTflops)

	upperBoundTflops = WithTflopsConfidenceMultiplier(1.0, 1.0, upperBoundTflops, *confidenceInterval)
	lowerBoundTflops = WithTflopsConfidenceMultiplier(0.001, -2.0, lowerBoundTflops, *confidenceInterval)

	targetVram := NewPercentileVramEstimator(*targetVramPercentile)
	lowerBoundVram := NewPercentileVramEstimator(*lowerBoundVramPercentile)
	upperBoundVram := NewPercentileVramEstimator(*upperBoundVramPercentile)

	targetVram = WithVramMargin(*safetyMarginFraction, targetVram)
	lowerBoundVram = WithVramMargin(*safetyMarginFraction, lowerBoundVram)
	upperBoundVram = WithVramMargin(*safetyMarginFraction, upperBoundVram)

	upperBoundVram = WithVramConfidenceMultiplier(1.0, 1.0, upperBoundVram, *confidenceInterval)
	lowerBoundVram = WithVramConfidenceMultiplier(0.001, -2.0, lowerBoundVram, *confidenceInterval)

	return &recommender{
		targetTflops:     targetTflops,
		lowerBoundTflops: lowerBoundTflops,
		upperBoundTflops: upperBoundTflops,
		targetVram:       targetVram,
		lowerBoundVram:   lowerBoundVram,
		upperBoundVram:   upperBoundVram,
	}
}

type recommender struct {
	targetTflops     TflopsEstimator
	lowerBoundTflops TflopsEstimator
	upperBoundTflops TflopsEstimator
	targetVram       VramEstimator
	lowerBoundVram   VramEstimator
	upperBoundVram   VramEstimator
}

func (r *recommender) GetRecommendedResources(s *WorkloadState) *RecommendedResources {
	return &RecommendedResources{
		TargetTflops:     r.targetTflops.GetTflopsEstimation(s),
		LowerBoundTflops: r.lowerBoundTflops.GetTflopsEstimation(s),
		UpperBoundTflops: r.upperBoundTflops.GetTflopsEstimation(s),
		TargetVram:       r.targetVram.GetVramEstimation(s),
		LowerBoundVram:   r.lowerBoundVram.GetVramEstimation(s),
		UpperBoundVram:   r.upperBoundVram.GetVramEstimation(s),
	}
}
