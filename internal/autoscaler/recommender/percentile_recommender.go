package recommender

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
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
	defaultUpperBoundTflopsPercentile = 0.99
	// Default update threshold
	defaultUpdateThreshold = 0.1
	// Default min/max scaling ratios
	defaultMinVRAMResourcesRatio    = 0.2
	defaultMaxVRAMResourcesRatio    = 5.0
	defaultMinComputeResourcesRatio = 0.1
	defaultMaxComputeResourcesRatio = 10.0
	// Minimum resource values

	scaleResourceCompute = "Compute"
	scaleResourceVram    = "VRAM"
)

var (
	minComputeResource = resource.MustParse("1")
	minVRAMResource    = resource.MustParse("1Gi")
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
	asr := workload.AutoSetResources()
	if asr == nil {
		return nil, nil
	}
	config := getPercentileConfig(asr)
	initialDelay, err := parseDurationOrDefault(asr.InitialDelayPeriod, 30*time.Minute)
	if err != nil {
		log.Error(err, "failed to parse initial delay period, using default")
		initialDelay = 30 * time.Minute
	}

	workloadCreationTime := workload.CreationTime()
	if workloadCreationTime.IsZero() {
		// Fallback: use current time if creation timestamp is not set
		workloadCreationTime = time.Now()
	}
	_, name := workload.Coordinates()
	qos := workload.Qos()

	timeSinceCreation := time.Since(workloadCreationTime)
	if timeSinceCreation < initialDelay {
		workload.UpsertStatusCondition(metav1.Condition{
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

	log.V(4).Info("estimated resources", "workload", name, "estimations", estimations)

	curRes := workload.GetCurrentResourcesSpec()
	originalRes := workload.GetOriginalResourcesSpec()
	recommendation := tfv1.Resources{}

	// A resource without a single usage sample is left as it is: the estimate
	// of an empty histogram is zero, which would read as "scale down to the
	// minimum" although nothing was measured.
	scaleCompute := func() *scalingResult { return nil }
	if !estimations.NoTflopsSamples {
		scaleCompute = func() *scalingResult {
			return p.handleResourceScaling(
				scaleResourceCompute,
				&curRes.Requests.Tflops,
				&curRes.Limits.Tflops,
				&estimations.TargetTflops,
				&estimations.LowerBoundTflops,
				&estimations.UpperBoundTflops,
				&originalRes.Requests.Tflops,
				&originalRes.Limits.Tflops,
				config,
				qos,
			)
		}
	}
	scaleVram := func() *scalingResult { return nil }
	if !estimations.NoVramSamples {
		scaleVram = func() *scalingResult {
			return p.handleResourceScaling(
				scaleResourceVram,
				&curRes.Requests.Vram,
				&curRes.Limits.Vram,
				&estimations.TargetVram,
				&estimations.LowerBoundVram,
				&estimations.UpperBoundVram,
				&originalRes.Requests.Vram,
				&originalRes.Limits.Vram,
				config,
				qos,
			)
		}
	}

	// Handle TFLOPS scaling
	if result := scaleCompute(); result != nil {
		recommendation.Requests.Tflops = result.targetRequest
		recommendation.Limits.Tflops = result.targetLimit
	} else {
		recommendation.Requests.Tflops = curRes.Requests.Tflops
		recommendation.Limits.Tflops = curRes.Limits.Tflops
	}

	// Handle VRAM scaling
	if result := scaleVram(); result != nil {
		recommendation.Requests.Vram = result.targetRequest
		recommendation.Limits.Vram = result.targetLimit
	} else {
		recommendation.Requests.Vram = curRes.Requests.Vram
		recommendation.Limits.Vram = curRes.Limits.Vram
	}

	// Avoid fluctuation when scale up/down is too small
	if !recommendation.IsZero() {
		if thresholdMessage := withinUpdateThreshold(curRes, &recommendation, config.UpdateThreshold); thresholdMessage != "" {
			workload.UpsertStatusCondition(metav1.Condition{
				Type:               constants.ConditionStatusTypeResourceUpdate,
				Status:             metav1.ConditionTrue,
				LastTransitionTime: metav1.Now(),
				Reason:             "InsideUpdateThreshold",
				Message:            thresholdMessage + "no update performed",
			})
			// Still update recommendation in status
			return &RecResult{
				Resources:        recommendation,
				HasApplied:       true,
				ScaleDownLocking: false,
			}, nil
		}
	}

	if recommendation.IsZero() {
		return nil, nil
	}

	// A running worker is taken to the recommended resources in bounded steps,
	// and never below what it was observed using.
	recommendation, boundNotes := boundChange(workload, curRes, recommendation, estimations)
	bounds := describeBounds(boundNotes)

	if p.recommendationProcessor != nil {
		var err error
		var msg string
		recommendation, msg, err = p.recommendationProcessor.Apply(ctx, workload, &recommendation)
		if err != nil {
			return nil, fmt.Errorf("failed to apply recommendation processor: %v", err)
		}
		if msg != "" {
			if bounds != "" {
				bounds += ", "
			}
			bounds += msg
		}
	}

	hasApplied := recommendation.Equal(curRes)
	if !hasApplied {
		if waiting := waitForMinChangeInterval(workload); waiting != nil {
			return waiting, nil
		}

		// The message tells what is applied; when a bound made that differ from
		// what the usage calls for, it also tells what was recommended and why
		// it was not applied as it is.
		message := describeChange(curRes, &recommendation)
		if bounds != "" {
			message += "; " + bounds
			log.Info("recommendation bounded", "workload", name, "message", message)
		}
		workload.SetChangeBounds(bounds)
		workload.UpsertStatusCondition(metav1.Condition{
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
	targetRequest resource.Quantity
	targetLimit   resource.Quantity
}

func (p *PercentileRecommender) handleResourceScaling(
	resourceName string,
	currentRequest, currentLimit, targetRequest, lowerBound, upperBound, originalRequest, originalLimit *resource.Quantity,
	config *PercentileConfig,
	qos tfv1.QoSLevel,
) *scalingResult {
	// UpperBound becomes limit, Target becomes request
	targetLim := *upperBound
	targetReq := *lowerBound
	switch qos {
	case tfv1.QoSCritical:
		targetReq = *upperBound
	case tfv1.QoSHigh:
		targetReq = *targetRequest
	}

	// Apply min/max scaling ratio constraints
	var minRatio, maxRatio float64
	if resourceName == scaleResourceCompute {
		minRatio = config.MinComputeResourcesRatio
		maxRatio = config.MaxComputeResourcesRatio
	} else {
		minRatio = config.MinVRAMResourcesRatio
		maxRatio = config.MaxVRAMResourcesRatio
	}

	// Calculate min and max allowed values based on original request
	originalRequestValue := originalRequest.AsApproximateFloat64()
	originalLimitValue := originalLimit.AsApproximateFloat64()
	minAllowedReq := originalRequestValue * minRatio
	maxAllowedReq := originalRequestValue * maxRatio
	minAllowedLim := originalLimitValue * minRatio
	maxAllowedLim := originalLimitValue * maxRatio

	// Apply minimum resource constraints
	minResource := minVRAMResource
	if resourceName == scaleResourceCompute {
		minResource = minComputeResource
	}

	// Must assign a minimum value to target request and limit
	if targetLim.Cmp(minResource) < 0 {
		targetLim = minResource
	}
	if targetReq.Cmp(minResource) < 0 {
		targetReq = minResource
	}

	// Must inside scaling range
	targetReqValue := targetReq.AsApproximateFloat64()
	if minAllowedReq != 0 && targetReqValue < minAllowedReq {
		targetReqValue = minAllowedReq
		targetReq = *resource.NewQuantity(int64(targetReqValue), targetReq.Format)
	}
	if maxAllowedReq != 0 && targetReqValue > maxAllowedReq {
		targetReqValue = maxAllowedReq
		targetReq = *resource.NewQuantity(int64(targetReqValue), targetReq.Format)
	}
	targetLimValue := targetLim.AsApproximateFloat64()
	if minAllowedLim != 0 && targetLimValue < minAllowedLim {
		targetLimValue = minAllowedLim
		targetLim = *resource.NewQuantity(int64(targetLimValue), targetLim.Format)
	}
	if maxAllowedLim != 0 && targetLimValue > maxAllowedLim {
		targetLimValue = maxAllowedLim
		targetLim = *resource.NewQuantity(int64(targetLimValue), targetLim.Format)
	}

	// Make sure compute limit is not less than original to avoid performance downgrade
	if resourceName == "Compute" {
		if targetLimValue < originalLimitValue {
			targetLimValue = originalLimitValue
			targetLim = *resource.NewQuantity(int64(targetLimValue), targetLim.Format)
		}
	}

	// Check if scaling is needed
	isReqNoChange := currentRequest.Cmp(targetReq) == 0
	isLimNoChange := currentLimit.Cmp(targetLim) == 0
	if isReqNoChange && isLimNoChange {
		return nil
	}

	return &scalingResult{
		targetRequest: targetReq,
		targetLimit:   targetLim,
	}
}

// withinUpdateThreshold returns a message naming the differences when no
// request or limit differs from the current one by more than the threshold,
// and "" when a change is worth making. Limits count as well as requests: a
// change is made in bounded steps, and a limit may still be on its way when
// the request has arrived.
func withinUpdateThreshold(current, recommended *tfv1.Resources, updateThreshold float64) string {
	message := ""
	for _, quantity := range []struct {
		name                 string
		current, recommended resource.Quantity
	}{
		{scaleResourceCompute, current.Requests.Tflops, recommended.Requests.Tflops},
		{computeLimit, current.Limits.Tflops, recommended.Limits.Tflops},
		{scaleResourceVram, current.Requests.Vram, recommended.Requests.Vram},
		{vramLimit, current.Limits.Vram, recommended.Limits.Vram},
	} {
		if quantity.current.IsZero() || quantity.recommended.IsZero() {
			continue
		}
		diff := absDiff(quantity.current, quantity.recommended)
		threshold := multiplyQuantity(quantity.current, updateThreshold)
		if diff.Cmp(threshold) > 0 {
			return ""
		}
		message += fmt.Sprintf("%s change (%s) within threshold (%s), ", quantity.name, diff.String(), threshold.String())
	}
	return message
}

// waitForMinChangeInterval returns the result that leaves the workload as it
// is while the previous change of its workers is less than minChangeInterval
// ago, and nil once a further change may be made. The workers get that time to
// show their usage under the resources of the previous change.
func waitForMinChangeInterval(state *workload.State) *RecResult {
	lastChange := state.LastResourceChange()
	if lastChange.IsZero() || timeNow().Sub(lastChange) >= minChangeInterval {
		return nil
	}
	state.UpsertStatusCondition(metav1.Condition{
		Type:               constants.ConditionStatusTypeResourceUpdate,
		Status:             metav1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Reason:             "MinChangeInterval",
		Message: fmt.Sprintf("Resources were changed at %s, no further change before %s (minimum interval %s)",
			lastChange.UTC().Format(time.RFC3339), lastChange.Add(minChangeInterval).UTC().Format(time.RFC3339), minChangeInterval),
	})
	return &RecResult{
		Resources:        tfv1.Resources{},
		HasApplied:       true,
		ScaleDownLocking: false,
	}
}

// describeChange words the change from the current resources to the ones that
// are applied, per resource that changes.
func describeChange(current, applied *tfv1.Resources) string {
	parts := []string{}
	for _, change := range []struct {
		name                                       string
		fromRequest, toRequest, fromLimit, toLimit *resource.Quantity
	}{
		{scaleResourceCompute, &current.Requests.Tflops, &applied.Requests.Tflops, &current.Limits.Tflops, &applied.Limits.Tflops},
		{scaleResourceVram, &current.Requests.Vram, &applied.Requests.Vram, &current.Limits.Vram, &applied.Limits.Vram},
	} {
		if change.fromRequest.Cmp(*change.toRequest) == 0 && change.fromLimit.Cmp(*change.toLimit) == 0 {
			continue
		}
		parts = append(parts, fmt.Sprintf("%s scaled: request %s -> %s, limit %s -> %s", change.name,
			change.fromRequest.String(), change.toRequest.String(), change.fromLimit.String(), change.toLimit.String()))
	}
	return strings.Join(parts, ", ")
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
	// NoTflopsSamples / NoVramSamples are set when the estimate of that
	// resource rests on no usage sample at all and must not be acted on.
	NoTflopsSamples bool
	NoVramSamples   bool
	// RequestFloor and LimitFloor are the observed usage a request and a limit
	// are not lowered below (see usageFloorRequestPercentile); zero for none.
	RequestFloor tfv1.Resource
	LimitFloor   tfv1.Resource
}

// resourcesEstimator is stateless. It holds no per-call data; estimators are
// constructed locally per GetResourcesEstimation call so concurrent invocations
// for different workloads cannot stomp on each other's percentile / margin
// configuration (the previous "*r = resourcesEstimator{...}" pattern was a
// data race across workloads sharing the same PercentileRecommender).
type resourcesEstimator struct{}

type percentileEstimators struct {
	lowerBoundTflops TflopsEstimator
	targetTflops     TflopsEstimator
	upperBoundTflops TflopsEstimator
	lowerBoundVram   VramEstimator
	targetVram       VramEstimator
	upperBoundVram   VramEstimator
}

func (r *resourcesEstimator) GetResourcesEstimation(wl *workload.State) *EstimatedResources {
	asr := wl.AutoSetResources()
	if asr == nil {
		return nil
	}
	est := buildPercentileEstimators(getPercentileConfig(asr))

	var result *EstimatedResources
	wl.ConsumeAggregator(func(aggregator *metrics.WorkerUsageAggregator) {
		if aggregator == nil || aggregator.IsEmpty() {
			return
		}
		result = &EstimatedResources{
			LowerBoundTflops: QuantityFromAmount(est.lowerBoundTflops.GetTflopsEstimation(aggregator), resource.DecimalSI),
			TargetTflops:     QuantityFromAmount(est.targetTflops.GetTflopsEstimation(aggregator), resource.DecimalSI),
			UpperBoundTflops: QuantityFromAmount(est.upperBoundTflops.GetTflopsEstimation(aggregator), resource.DecimalSI),
			LowerBoundVram:   QuantityFromAmount(est.lowerBoundVram.GetVramEstimation(aggregator), resource.BinarySI),
			TargetVram:       QuantityFromAmount(est.targetVram.GetVramEstimation(aggregator), resource.BinarySI),
			UpperBoundVram:   QuantityFromAmount(est.upperBoundVram.GetVramEstimation(aggregator), resource.BinarySI),
			NoTflopsSamples:  aggregator.TflopsHistogram.IsEmpty(),
			NoVramSamples:    aggregator.VramHistogram.IsEmpty(),
			RequestFloor: tfv1.Resource{
				Tflops: QuantityFromAmount(NewPercentileTflopsEstimator(usageFloorRequestPercentile).GetTflopsEstimation(aggregator), resource.DecimalSI),
				Vram:   QuantityFromAmount(NewPercentileVramEstimator(usageFloorRequestPercentile).GetVramEstimation(aggregator), resource.BinarySI),
			},
			LimitFloor: tfv1.Resource{
				Tflops: QuantityFromAmount(NewPercentileTflopsEstimator(usageFloorLimitPercentile).GetTflopsEstimation(aggregator), resource.DecimalSI),
				Vram:   QuantityFromAmount(NewPercentileVramEstimator(usageFloorLimitPercentile).GetVramEstimation(aggregator), resource.BinarySI),
			},
		}
	})
	return result
}

func buildPercentileEstimators(config *PercentileConfig) percentileEstimators {
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

	return percentileEstimators{
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
