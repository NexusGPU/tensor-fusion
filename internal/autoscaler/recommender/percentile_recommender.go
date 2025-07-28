package recommender

import (
	"strconv"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
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

var defaultPercentileConfig = PercentileConfig{
	TargetTflopsPercentile:     defaultTargetTflopsPercentile,
	LowerBoundTflopsPercentile: defaultLowerBoundTflopsPercentile,
	UpperBoundTflopsPercentile: defaultUpperBoundTflopsPercentile,
	TargetVramPercentile:       defaultTargetVramPercentile,
	LowerBoundVramPercentile:   defaultLowerBoundVramPercentile,
	UpperBoundVramPercentile:   defaultUpperBoundVramPercentile,
	RequestMarginFraction:      defaultRequestMarginFraction,
	ConfidenceInterval:         defaultConfidenceInterval,
}

type PercentileConfig struct {
	TargetTflopsPercentile     float64
	LowerBoundTflopsPercentile float64
	UpperBoundTflopsPercentile float64
	TargetVramPercentile       float64
	LowerBoundVramPercentile   float64
	UpperBoundVramPercentile   float64
	RequestMarginFraction      float64
	ConfidenceInterval         time.Duration
}

type PercentileRecommender struct {
	lowerBoundTflops TflopsEstimator
	targetTflops     TflopsEstimator
	upperBoundTflops TflopsEstimator
	lowerBoundVram   VramEstimator
	targetVram       VramEstimator
	upperBoundVram   VramEstimator
}

func NewPercentileRecommender() *PercentileRecommender {
	return &PercentileRecommender{}
}

func (p *PercentileRecommender) Name() string {
	return "percentile"
}

func (p *PercentileRecommender) Recommend(workload *workload.State) (*tfv1.RecommendedResources, error) {
	// TODO: cache config
	aggregator := workload.WorkerUsageAggregator
	if aggregator.TflopsHistogram.IsEmpty() && aggregator.VramHistogram.IsEmpty() {
		return nil, nil
	}

	p.createEstimatorsFromConfig(p.getPercentileConfig(&workload.Spec.AutoScalingConfig.AutoSetResources))
	return &tfv1.RecommendedResources{
		LowerBoundTflops: QuantityFromAmount(p.lowerBoundTflops.GetTflopsEstimation(aggregator)),
		TargetTflops:     QuantityFromAmount(p.targetTflops.GetTflopsEstimation(aggregator)),
		UpperBoundTflops: QuantityFromAmount(p.upperBoundTflops.GetTflopsEstimation(aggregator)),
		LowerBoundVram:   QuantityFromAmount(p.lowerBoundVram.GetVramEstimation(aggregator)),
		TargetVram:       QuantityFromAmount(p.targetVram.GetVramEstimation(aggregator)),
		UpperBoundVram:   QuantityFromAmount(p.upperBoundVram.GetVramEstimation(aggregator)),
	}, nil
}

func (p *PercentileRecommender) getPercentileConfig(asr *tfv1.AutoSetResources) *PercentileConfig {
	cfg := defaultPercentileConfig

	if asr == nil {
		return &cfg
	}

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

func (p *PercentileRecommender) createEstimatorsFromConfig(config *PercentileConfig) {
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

	*p = PercentileRecommender{
		lowerBoundTflops: lowerBoundTflops,
		targetTflops:     targetTflops,
		upperBoundTflops: upperBoundTflops,
		lowerBoundVram:   lowerBoundVram,
		targetVram:       targetVram,
		upperBoundVram:   upperBoundVram,
	}
}
