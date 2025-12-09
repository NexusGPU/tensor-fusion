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
	defaultTargetVramPercentile = 0.98
	// Vram usage percentile that will be used for the lower bound on vram recommendation.
	defaultLowerBoundVramPercentile = 0.5
	// Vram usage percentile that will be used for the upper bound on vram recommendation.
	defaultUpperBoundVramPercentile = 0.99
	// Tflops usage percentile that will be used as a base for tflops target recommendation. Doesn't affect tflops lower bound nor tflops upper bound.
	defaultTargetTflopsPercentile = 0.95
	// Tflops usage percentile that will be used for the lower bound on tflops recommendation.
	defaultLowerBoundTflopsPercentile = 0.5
	// Tflops usage percentile that will be used for the upper bound on tflops recommendation.
	defaultUpperBoundTflopsPercentile = 0.98
	// Default update threshold
	defaultUpdateThreshold = 0.1
	// Default min/max scaling ratios
	defaultMinVRAMResourcesRatio    = 0.2
	defaultMaxVRAMResourcesRatio    = 5.0
	defaultMinComputeResourcesRatio = 0.1
	defaultMaxComputeResourcesRatio = 10.0
	// Minimum resource values
	minComputeResource = 1.0  // 1 TFlops
	minVRAMResource    = 1024 // 1Gi in MiB
)

var defaultPercentileConfig = PercentileConfig{
	TargetTflopsPercentile:     defaultTargetTflopsPercentile,
	LowerBoundTflopsPercentile: defaultLowerBoundTflopsPercentile,
	UpperBoundTflopsPercentile: defaultUpperBoundTflopsPercentile,
	TargetVramPercentile:       defaultTargetVramPercentile,
	LowerBoundVramPercentile:   defaultLowerBoundVramPercentile,
	UpperBoundVramPercentile:   defaultUpperBoundVramPercentile,
	RequestMarginFraction:      defaultRequestMarginFraction,
	UpdateThreshold:            defaultUpdateThreshold,
	MinVRAMResourcesRatio:      defaultMinVRAMResourcesRatio,
	MaxVRAMResourcesRatio:      defaultMaxVRAMResourcesRatio,
	MinComputeResourcesRatio:   defaultMinComputeResourcesRatio,
	MaxComputeResourcesRatio:   defaultMaxComputeResourcesRatio,
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
	UpdateThreshold            float64
	MinVRAMResourcesRatio      float64
	MaxVRAMResourcesRatio      float64
	MinComputeResourcesRatio   float64
	MaxComputeResourcesRatio   float64
}

type PercentileRecommender struct {
	ResourcesEstimator
	recommendationProcessor RecommendationProcessor
}

func NewPercentileRecommender(recommendationProcessor RecommendationProcessor) *PercentileRecommender {
	return &PercentileRecommender{
		ResourcesEstimator:      &resourcesEstimator{},
		recommendationProcessor: recommendationProcessor,
	}
}

func (p *PercentileRecommender) Name() string {
	return "percentile"
}

func (p *PercentileRecommender) Recommend(ctx context.Context, workload *workload.State) (*RecResult, error) {
	log := log.FromContext(ctx)

	// Check InitialDelayPeriod
	asr := workload.Spec.AutoScalingConfig.AutoSetResources
	if asr == nil {
		return nil, nil
	}
	config := getPercentileConfig(asr)
	initialDelay, err := parseDurationOrDefault(asr.InitialDelayPeriod, 30*time.Minute)
	if err != nil {
		log.Error(err, "failed to parse initial delay period, using default")
		initialDelay = 30 * time.Minute
	}

	workloadCreationTime := workload.CreationTimestamp.Time
	if workloadCreationTime.IsZero() {
		// Fallback: use current time if creation timestamp is not set
		workloadCreationTime = time.Now()
	}

	timeSinceCreation := time.Since(workloadCreationTime)
	if timeSinceCreation < initialDelay {
		meta.SetStatusCondition(&workload.Status.Conditions, metav1.Condition{
			Type:               constants.ConditionStatusTypeResourceUpdate,
			Status:             metav1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
			Reason:             "LowConfidence",
			Message:            fmt.Sprintf("Workload created time less than InitialDelayPeriod %v, no update performed", initialDelay),
		})
		return &RecResult{
			Resources:        tfv1.Resources{},
			HasApplied:       true,
			ScaleDownLocking: false,
		}, nil
	}

	estimations := p.GetResourcesEstimation(workload)
	if estimations == nil {
		return nil, nil
	}

	log.Info("estimated resources", "workload", workload.Name, "estimations", estimations)

	curRes := workload.GetCurrentResourcesSpec()
	originalRes := workload.GetOriginalResourcesSpec()
	recommendation := tfv1.Resources{}
	message := ""

	// Apply min/max scaling ratio constraints - config already set above

	// Handle TFLOPS scaling
	if result := p.handleResourceScaling(
		"Compute",
		&curRes.Requests.Tflops,
		&curRes.Limits.Tflops,
		&estimations.TargetTflops,
		&estimations.LowerBoundTflops,
		&estimations.UpperBoundTflops,
		&originalRes.Requests.Tflops,
		config,
	); result != nil {
		message = result.message
		recommendation.Requests.Tflops = result.targetRequest
		recommendation.Limits.Tflops = result.targetLimit
	} else {
		recommendation.Requests.Tflops = curRes.Requests.Tflops
		recommendation.Limits.Tflops = curRes.Limits.Tflops
	}

	// Handle VRAM scaling
	if result := p.handleResourceScaling(
		"VRAM",
		&curRes.Requests.Vram,
		&curRes.Limits.Vram,
		&estimations.TargetVram,
		&estimations.LowerBoundVram,
		&estimations.UpperBoundVram,
		&originalRes.Requests.Vram,
		config,
	); result != nil {
		if len(message) > 0 {
			message += fmt.Sprintf(", %s", result.message)
		} else {
			message = result.message
		}
		recommendation.Requests.Vram = result.targetRequest
		recommendation.Limits.Vram = result.targetLimit
	} else {
		recommendation.Requests.Vram = curRes.Requests.Vram
		recommendation.Limits.Vram = curRes.Limits.Vram
	}

	// Check UpdateThreshold
	if !recommendation.IsZero() {
		updateThreshold := config.UpdateThreshold
		shouldUpdate := false
		thresholdMessage := ""

		// Check if change exceeds threshold
		if !curRes.Requests.Tflops.IsZero() && !recommendation.Requests.Tflops.IsZero() {
			diff := absDiff(curRes.Requests.Tflops, recommendation.Requests.Tflops)
			threshold := multiplyQuantity(curRes.Requests.Tflops, updateThreshold)
			if diff.Cmp(threshold) > 0 {
				shouldUpdate = true
			} else {
				thresholdMessage += fmt.Sprintf("Compute change (%s) within threshold (%s), ", diff.String(), threshold.String())
			}
		}

		if !curRes.Requests.Vram.IsZero() && !recommendation.Requests.Vram.IsZero() {
			diff := absDiff(curRes.Requests.Vram, recommendation.Requests.Vram)
			threshold := multiplyQuantity(curRes.Requests.Vram, updateThreshold)
			if diff.Cmp(threshold) > 0 {
				shouldUpdate = true
			} else {
				if thresholdMessage == "" {
					thresholdMessage = "VRAM change within threshold, "
				} else {
					thresholdMessage += "VRAM change within threshold, "
				}
			}
		}

		if !shouldUpdate && thresholdMessage != "" {
			meta.SetStatusCondition(&workload.Status.Conditions, metav1.Condition{
				Type:               constants.ConditionStatusTypeResourceUpdate,
				Status:             metav1.ConditionTrue,
				LastTransitionTime: metav1.Now(),
				Reason:             "InsideUpdateThreshold",
				Message:            thresholdMessage + "no update performed",
			})
			// Still update recommendation in status
			return &RecResult{
				Resources:        recommendation,
				HasApplied:       false,
				ScaleDownLocking: false,
			}, nil
		}
	}

	if recommendation.IsZero() {
		return nil, nil
	}

	if p.recommendationProcessor != nil {
		var err error
		var msg string
		recommendation, msg, err = p.recommendationProcessor.Apply(ctx, workload, &recommendation)
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
			Type:               constants.ConditionStatusTypeResourceUpdate,
			Status:             metav1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
			Reason:             "Updated",
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
	currentRequest, currentLimit, targetRequest, lowerBound, upperBound, originalRequest *resource.Quantity,
	config *PercentileConfig,
) *scalingResult {
	// UpperBound becomes limit, Target becomes request
	targetReq := *targetRequest
	targetLim := *upperBound

	// Apply min/max scaling ratio constraints
	var minRatio, maxRatio float64
	if resourceName == "Compute" {
		minRatio = config.MinComputeResourcesRatio
		maxRatio = config.MaxComputeResourcesRatio
	} else {
		minRatio = config.MinVRAMResourcesRatio
		maxRatio = config.MaxVRAMResourcesRatio
	}

	// Calculate min and max allowed values based on original request
	originalValue := originalRequest.Value()
	minAllowed := int64(float64(originalValue) * minRatio)
	maxAllowed := int64(float64(originalValue) * maxRatio)

	// Apply minimum resource constraints
	var minResource int64
	if resourceName == "Compute" {
		minResource = int64(minComputeResource * 1e12) // Convert TFlops to base units
	} else {
		minResource = int64(minVRAMResource * 1024 * 1024) // Convert GiB to bytes
	}

	// Use original value if it's smaller than minimum
	if originalValue < minResource {
		minResource = originalValue
	}

	// Clamp target request to min/max bounds
	if targetReq.Value() < minAllowed {
		targetReq = *resource.NewQuantity(minAllowed, targetReq.Format)
	}
	if targetReq.Value() > maxAllowed {
		targetReq = *resource.NewQuantity(maxAllowed, targetReq.Format)
	}
	if targetReq.Value() < minResource {
		targetReq = *resource.NewQuantity(minResource, targetReq.Format)
	}

	// Clamp target limit to min/max bounds
	if targetLim.Value() < minAllowed {
		targetLim = *resource.NewQuantity(minAllowed, targetLim.Format)
	}
	if targetLim.Value() > maxAllowed {
		targetLim = *resource.NewQuantity(maxAllowed, targetLim.Format)
	}
	if targetLim.Value() < minResource {
		targetLim = *resource.NewQuantity(minResource, targetLim.Format)
	}

	// Check if scaling is needed
	isScaleUp := currentRequest.Cmp(targetReq) < 0
	isScaleDown := currentRequest.Cmp(targetReq) > 0

	if !isScaleUp && !isScaleDown {
		return nil
	}

	var message string
	if isScaleUp {
		message = fmt.Sprintf("%s scaled up: request %s -> %s, limit %s -> %s",
			resourceName, currentRequest.String(), targetReq.String(), currentLimit.String(), targetLim.String())
	} else {
		message = fmt.Sprintf("%s scaled down: request %s -> %s, limit %s -> %s",
			resourceName, currentRequest.String(), targetReq.String(), currentLimit.String(), targetLim.String())
	}

	return &scalingResult{
		message:       message,
		targetRequest: targetReq,
		targetLimit:   targetLim,
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

func absDiff(a, b resource.Quantity) resource.Quantity {
	if a.Cmp(b) > 0 {
		return *resource.NewQuantity(a.Value()-b.Value(), a.Format)
	}
	return *resource.NewQuantity(b.Value()-a.Value(), a.Format)
}

func multiplyQuantity(q resource.Quantity, multiplier float64) resource.Quantity {
	value := float64(q.Value()) * multiplier
	return *resource.NewQuantity(int64(value), q.Format)
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

func (r *resourcesEstimator) GetResourcesEstimation(workload *workload.State) *EstimatedResources {
	aggregator := workload.WorkerUsageAggregator
	if aggregator.IsEmpty() {
		return nil
	}
	// TODO: cache config
	asr := workload.Spec.AutoScalingConfig.AutoSetResources
	if asr == nil {
		return nil
	}
	r.createEstimatorsFromConfig(getPercentileConfig(asr))
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
	// Simplified: no confidence multiplier, just percentile + margin
	targetTflops := NewPercentileTflopsEstimator(config.TargetTflopsPercentile)
	lowerBoundTflops := NewPercentileTflopsEstimator(config.LowerBoundTflopsPercentile)
	upperBoundTflops := NewPercentileTflopsEstimator(config.UpperBoundTflopsPercentile)

	targetTflops = WithTflopsMargin(config.RequestMarginFraction, targetTflops)
	lowerBoundTflops = WithTflopsMargin(config.RequestMarginFraction, lowerBoundTflops)
	upperBoundTflops = WithTflopsMargin(config.RequestMarginFraction, upperBoundTflops)

	targetVram := NewPercentileVramEstimator(config.TargetVramPercentile)
	lowerBoundVram := NewPercentileVramEstimator(config.LowerBoundVramPercentile)
	upperBoundVram := NewPercentileVramEstimator(config.UpperBoundVramPercentile)

	targetVram = WithVramMargin(config.RequestMarginFraction, targetVram)
	lowerBoundVram = WithVramMargin(config.RequestMarginFraction, lowerBoundVram)
	upperBoundVram = WithVramMargin(config.RequestMarginFraction, upperBoundVram)

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
		{asr.TargetComputePercentile, &cfg.TargetTflopsPercentile},
		{asr.LowerBoundComputePercentile, &cfg.LowerBoundTflopsPercentile},
		{asr.UpperBoundComputePercentile, &cfg.UpperBoundTflopsPercentile},
		{asr.TargetVRAMPercentile, &cfg.TargetVramPercentile},
		{asr.LowerBoundVRAMPercentile, &cfg.LowerBoundVramPercentile},
		{asr.UpperBoundVRAMPercentile, &cfg.UpperBoundVramPercentile},
		{asr.MarginFraction, &cfg.RequestMarginFraction},
		{asr.UpdateThreshold, &cfg.UpdateThreshold},
		{asr.MinVRAMResourcesRatio, &cfg.MinVRAMResourcesRatio},
		{asr.MaxVRAMResourcesRatio, &cfg.MaxVRAMResourcesRatio},
		{asr.MinComputeResourcesRatio, &cfg.MinComputeResourcesRatio},
		{asr.MaxComputeResourcesRatio, &cfg.MaxComputeResourcesRatio},
	}
	for _, f := range fields {
		if f.val == "" {
			continue
		}
		if v, err := strconv.ParseFloat(f.val, 64); err == nil {
			*f.dst = v
		}
	}

	return &cfg
}

func parseDurationOrDefault(durationStr string, defaultDuration time.Duration) (time.Duration, error) {
	if durationStr == "" {
		return defaultDuration, nil
	}
	return time.ParseDuration(durationStr)
}
