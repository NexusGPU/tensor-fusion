package recommender

import (
	"context"
	"fmt"
	"math/big"
	"strconv"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
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

type ResourcesEstimator interface {
	GetResourcesEstimation(*workload.State) *EstimatedResources
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
	ResourcesEstimator
	RecommendationProcessor
}

func NewPercentileRecommender(recommendationProcessor RecommendationProcessor) *PercentileRecommender {
	return &PercentileRecommender{
		ResourcesEstimator:      &resourcesEstimator{},
		RecommendationProcessor: recommendationProcessor,
	}
}

func (p *PercentileRecommender) Name() string {
	return "percentile"
}

func (p *PercentileRecommender) Recommend(ctx context.Context, workload *workload.State) (*RecResult, error) {
	log := log.FromContext(ctx)

	estimations := p.GetResourcesEstimation(workload)
	if estimations == nil {
		return nil, nil
	}

	log.Info("estimated resources", "workload", workload.Name, "estimations", estimations)

	curRes := workload.GetCurrentResourcesSpec()
	recommendation := tfv1.Resources{}
	message := ""

	// Handle TFLOPS scaling
	if result := p.handleResourceScaling(
		"TFLOPS",
		&curRes.Requests.Tflops,
		&curRes.Limits.Tflops,
		&estimations.TargetTflops,
		&estimations.LowerBoundTflops,
		&estimations.UpperBoundTflops,
	); result != nil {
		message = result.message
		recommendation.Requests.Tflops = result.targetRequest
		recommendation.Limits.Tflops = result.targetLimit
	}

	// Handle VRAM scaling
	if result := p.handleResourceScaling(
		"VRAM",
		&curRes.Requests.Vram,
		&curRes.Limits.Vram,
		&estimations.TargetVram,
		&estimations.LowerBoundVram,
		&estimations.UpperBoundVram,
	); result != nil {
		if len(message) > 0 {
			message += fmt.Sprintf(", %s", result.message)
		} else {
			message = result.message
		}
		recommendation.Requests.Vram = result.targetRequest
		recommendation.Limits.Vram = result.targetLimit
	}

	if recommendation.IsZero() {
		return nil, nil
	}

	if p.RecommendationProcessor != nil {
		var err error
		var msg string
		recommendation, msg, err = p.RecommendationProcessor.Apply(ctx, workload, &recommendation)
		if err != nil {
			return nil, fmt.Errorf("failed to apply recommendation processor: %v", err)
		}
		if msg != "" {
			message += fmt.Sprintf(", %s", msg)
			log.Info("recommendation processor applied", "message", message)
		}
	}

	hasApplied := recommendation.Equal(curRes)
	if !hasApplied {
		meta.SetStatusCondition(&workload.Status.Conditions, metav1.Condition{
			Type:               constants.ConditionStatusTypeRecommendationProvided,
			Status:             metav1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
			Reason:             "OutOfEstimatedBound",
			Message:            message,
		})
	}

	return &RecResult{
		Resources:        recommendation,
		HasApplied:       hasApplied,
		ScaleDownLocking: false,
	}, nil
}

type scalingResult struct {
	message       string
	targetRequest resource.Quantity
	targetLimit   resource.Quantity
}

func (p *PercentileRecommender) handleResourceScaling(
	resourceName string,
	currentRequest, currentLimit, targetRequest, lowerBound, upperBound *resource.Quantity,
) *scalingResult {
	isScaleUp := currentRequest.Cmp(*lowerBound) < 0
	isScaleDown := currentRequest.Cmp(*upperBound) > 0

	if !isScaleUp && !isScaleDown {
		return nil
	}

	targetLimit := getProportionalLimit(currentLimit, currentRequest, targetRequest)
	if targetLimit == nil {
		return nil
	}

	var message string
	if isScaleUp {
		message = fmt.Sprintf("%s scaled up due to (%s) below lower bound (%s)",
			resourceName, currentRequest.String(), lowerBound.String())
	} else {
		message = fmt.Sprintf("%s scaled down due to (%s) above upper bound (%s)",
			resourceName, currentRequest.String(), upperBound.String())
	}

	return &scalingResult{
		message:       message,
		targetRequest: *targetRequest,
		targetLimit:   *targetLimit,
	}
}

func getProportionalLimit(originalLimit, originalRequest, recommendedRequest *resource.Quantity) *resource.Quantity {
	if originalLimit == nil || originalLimit.IsZero() ||
		originalRequest == nil || originalRequest.IsZero() ||
		recommendedRequest == nil || recommendedRequest.IsZero() {
		return nil
	}

	originalValue := big.NewInt(originalLimit.Value())
	scaleBaseValue := big.NewInt(originalRequest.Value())
	scaleResultValue := big.NewInt(recommendedRequest.Value())
	var scaledOriginal big.Int
	scaledOriginal.Mul(originalValue, scaleResultValue)
	scaledOriginal.Div(&scaledOriginal, scaleBaseValue)
	if scaledOriginal.IsInt64() {
		return resource.NewQuantity(scaledOriginal.Int64(), originalLimit.Format)
	}

	return nil
}

type EstimatedResources struct {
	LowerBoundTflops resource.Quantity
	TargetTflops     resource.Quantity
	UpperBoundTflops resource.Quantity
	LowerBoundVram   resource.Quantity
	TargetVram       resource.Quantity
	UpperBoundVram   resource.Quantity
}

type resourcesEstimator struct {
	lowerBoundTflops TflopsEstimator
	targetTflops     TflopsEstimator
	upperBoundTflops TflopsEstimator
	lowerBoundVram   VramEstimator
	targetVram       VramEstimator
	upperBoundVram   VramEstimator
}

// var percentileConfigToEstimatorsMap map[PercentileConfig]resourcesEstimator

func (r *resourcesEstimator) GetResourcesEstimation(workload *workload.State) *EstimatedResources {
	aggregator := workload.WorkerUsageAggregator
	if aggregator.IsEmpty() {
		return nil
	}
	// TODO: cache config
	r.createEstimatorsFromConfig(getPercentileConfig(&workload.Spec.AutoScalingConfig.AutoSetResources))
	return &EstimatedResources{
		LowerBoundTflops: QuantityFromAmount(r.lowerBoundTflops.GetTflopsEstimation(aggregator), resource.DecimalSI),
		TargetTflops:     QuantityFromAmount(r.targetTflops.GetTflopsEstimation(aggregator), resource.DecimalSI),
		UpperBoundTflops: QuantityFromAmount(r.upperBoundTflops.GetTflopsEstimation(aggregator), resource.DecimalSI),
		LowerBoundVram:   QuantityFromAmount(r.lowerBoundVram.GetVramEstimation(aggregator), resource.BinarySI),
		TargetVram:       QuantityFromAmount(r.targetVram.GetVramEstimation(aggregator), resource.BinarySI),
		UpperBoundVram:   QuantityFromAmount(r.upperBoundVram.GetVramEstimation(aggregator), resource.BinarySI),
	}
}

func (r *resourcesEstimator) createEstimatorsFromConfig(config *PercentileConfig) {
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

	*r = resourcesEstimator{
		lowerBoundTflops: lowerBoundTflops,
		targetTflops:     targetTflops,
		upperBoundTflops: upperBoundTflops,
		lowerBoundVram:   lowerBoundVram,
		targetVram:       targetVram,
		upperBoundVram:   upperBoundVram,
	}
}

func getPercentileConfig(asr *tfv1.AutoSetResources) *PercentileConfig {
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
