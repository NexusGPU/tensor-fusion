package quota

import (
	"strconv"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// Calculator provides shared quota calculation logic
type Calculator struct{}

// NewCalculator creates a new quota calculator
func NewCalculator() *Calculator {
	return &Calculator{}
}

// IncreaseUsage safely increases resource usage
func (c *Calculator) IncreaseUsage(usage *tfv1.GPUResourceUsage, allocation *tfv1.AllocRequest) {
	// Update requests
	c.SafeAdd(&usage.Requests.Tflops, allocation.Request.Tflops)
	c.SafeAdd(&usage.Requests.Vram, allocation.Request.Vram)

	// Update limits (assume limits = requests for now)
	c.SafeAdd(&usage.Limits.Tflops, allocation.Request.Tflops)
	c.SafeAdd(&usage.Limits.Vram, allocation.Request.Vram)

	// Update workers
	if usage.Workers != nil {
		*usage.Workers += 1
	}
}

// DecreaseUsage safely decreases resource usage with bounds checking
func (c *Calculator) DecreaseUsage(usage *tfv1.GPUResourceUsage, allocation *tfv1.AllocRequest) {
	// Update requests with bounds checking
	c.SafeSub(&usage.Requests.Tflops, allocation.Request.Tflops)
	c.SafeSub(&usage.Requests.Vram, allocation.Request.Vram)

	// Update limits with bounds checking
	c.SafeSub(&usage.Limits.Tflops, allocation.Request.Tflops)
	c.SafeSub(&usage.Limits.Vram, allocation.Request.Vram)

	// Update workers with bounds checking
	if usage.Workers != nil {
		if *usage.Workers >= int32(allocation.Count) {
			*usage.Workers -= int32(allocation.Count)
		} else {
			*usage.Workers = 0
		}
	}
}

// ApplyUsageOperation atomically applies a usage operation (increase or decrease)
func (c *Calculator) ApplyUsageOperation(currentUsage *tfv1.GPUResourceUsage, allocation *tfv1.AllocRequest, isIncrease bool) {
	if isIncrease {
		c.IncreaseUsage(currentUsage, allocation)
	} else {
		c.DecreaseUsage(currentUsage, allocation)
	}
}

// CalculateActualDeallocation calculates actual deallocation amounts clamped to current usage
func (c *Calculator) CalculateActualDeallocation(currentUsage *tfv1.GPUResourceUsage, allocation *tfv1.AllocRequest) *tfv1.Resource {

	// // Clamp TFlops deallocation
	// if !currentUsage.Requests.Tflops.IsZero() && currentUsage.Requests.Tflops.Cmp(allocation.Request.Tflops) < 0 {
	// 	actual.TFlops = currentUsage.Requests.Tflops.DeepCopy()
	// } else {
	// 	actual.TFlops = allocation.Request.Tflops.DeepCopy()
	// }

	// // Clamp VRAM deallocation
	// if !currentUsage.Requests.Vram.IsZero() && currentUsage.Requests.Vram.Cmp(allocation.Request.Vram) < 0 {
	// 	actual.VRAM = currentUsage.Requests.Vram.DeepCopy()
	// } else {
	// 	actual.VRAM = allocation.Request.Vram.DeepCopy()
	// }

	// // Clamp workers deallocation
	// if currentUsage.Workers != nil {
	// 	actual.Workers = min(*currentUsage.Workers, int32(allocation.Count))
	// } else {
	// 	actual.Workers = 0
	// }

	// return actual
	return &tfv1.Resource{}
}

// CalculateAvailablePercent calculates available percentage for each resource
func (c *Calculator) CalculateAvailablePercent(quota *tfv1.GPUResourceQuota, usage *tfv1.GPUResourceUsage) *tfv1.GPUResourceAvailablePercent {
	percent := &tfv1.GPUResourceAvailablePercent{}

	// Calculate requests.tflops percentage
	if !quota.Spec.Total.Requests.Tflops.IsZero() && !usage.Requests.Tflops.IsZero() {
		if p := c.calculateResourcePercent(quota.Spec.Total.Requests.Tflops.Value(), usage.Requests.Tflops.Value()); p != nil {
			percent.RequestsTFlops = strconv.FormatFloat(*p, 'f', 2, 64)
		}
	}
	// Calculate requests.vram percentage
	if !quota.Spec.Total.Requests.Vram.IsZero() && !usage.Requests.Vram.IsZero() {
		if p := c.calculateResourcePercent(quota.Spec.Total.Requests.Vram.Value(), usage.Requests.Vram.Value()); p != nil {
			percent.RequestsVRAM = strconv.FormatFloat(*p, 'f', 2, 64)
		}
	}

	// Calculate limits.tflops percentage
	if !quota.Spec.Total.Limits.Tflops.IsZero() && !usage.Limits.Tflops.IsZero() {
		if p := c.calculateResourcePercent(quota.Spec.Total.Limits.Tflops.Value(), usage.Limits.Tflops.Value()); p != nil {
			percent.LimitsTFlops = strconv.FormatFloat(*p, 'f', 2, 64)
		}
	}

	// Calculate limits.vram percentage
	if !quota.Spec.Total.Limits.Vram.IsZero() && !usage.Limits.Vram.IsZero() {
		if p := c.calculateResourcePercent(quota.Spec.Total.Limits.Vram.Value(), usage.Limits.Vram.Value()); p != nil {
			percent.LimitsVRAM = strconv.FormatFloat(*p, 'f', 2, 64)
		}
	}

	// Calculate workers percentage
	if quota.Spec.Total.MaxWorkers != nil && usage.Workers != nil {
		total := int64(*quota.Spec.Total.MaxWorkers)
		used := int64(*usage.Workers)
		if p := c.calculateResourcePercent(total, used); p != nil {
			percent.Workers = strconv.FormatFloat(*p, 'f', 2, 64)
		}
	}

	return percent
}

// calculateResourcePercent calculates available percentage for a single resource
func (c *Calculator) calculateResourcePercent(total, used int64) *float64 {
	if total <= 0 {
		return nil
	}
	available := max(0, float64(total-used)*100.0/float64(total))
	return &available
}

// IsQuotaExceeded checks if any quota limit is exceeded
func (c *Calculator) IsQuotaExceeded(quota *tfv1.GPUResourceQuota, usage *tfv1.GPUResourceUsage) bool {
	if !quota.Spec.Total.Requests.Tflops.IsZero() && !usage.Requests.Tflops.IsZero() {
		if usage.Requests.Tflops.Cmp(quota.Spec.Total.Requests.Tflops) > 0 {
			return true
		}
	}
	if !quota.Spec.Total.Requests.Vram.IsZero() && !usage.Requests.Vram.IsZero() {
		if usage.Requests.Vram.Cmp(quota.Spec.Total.Requests.Vram) > 0 {
			return true
		}
	}
	if !quota.Spec.Total.Limits.Tflops.IsZero() && !usage.Limits.Tflops.IsZero() {
		if usage.Limits.Tflops.Cmp(quota.Spec.Total.Limits.Tflops) > 0 {
			return true
		}
	}
	if !quota.Spec.Total.Limits.Vram.IsZero() && !usage.Limits.Vram.IsZero() {
		if usage.Limits.Vram.Cmp(quota.Spec.Total.Limits.Vram) > 0 {
			return true
		}
	}
	if quota.Spec.Total.MaxWorkers != nil && usage.Workers != nil {
		if *usage.Workers > *quota.Spec.Total.MaxWorkers {
			return true
		}
	}
	return false
}

// IsAlertThresholdReached checks if alert threshold is reached for any resource
func (c *Calculator) IsAlertThresholdReached(availablePercent *tfv1.GPUResourceAvailablePercent, threshold int32) bool {
	thresholdFloat := float64(100 - threshold)

	tflops, _ := strconv.ParseFloat(availablePercent.RequestsTFlops, 64)
	if availablePercent.RequestsTFlops != "" && tflops < thresholdFloat {
		return true
	}
	vram, _ := strconv.ParseFloat(availablePercent.RequestsVRAM, 64)
	if availablePercent.RequestsVRAM != "" && vram < thresholdFloat {
		return true
	}
	tflopsLimits, _ := strconv.ParseFloat(availablePercent.LimitsTFlops, 64)
	if availablePercent.LimitsTFlops != "" && tflopsLimits < thresholdFloat {
		return true
	}
	vramLimits, _ := strconv.ParseFloat(availablePercent.LimitsVRAM, 64)
	if availablePercent.LimitsVRAM != "" && vramLimits < thresholdFloat {
		return true
	}
	workers, _ := strconv.ParseFloat(availablePercent.Workers, 64)
	if availablePercent.Workers != "" && workers < thresholdFloat {
		return true
	}
	return false
}

// CreateStandardConditions creates standard quota conditions
func (c *Calculator) CreateStandardConditions(exceeded, alertReached bool, alertThreshold int32) []metav1.Condition {
	now := metav1.Now()

	conditions := []metav1.Condition{
		{
			Type:               string(tfv1.GPUResourceQuotaConditionReady),
			Status:             metav1.ConditionTrue,
			LastTransitionTime: now,
			Reason:             "QuotaReady",
			Message:            "GPUResourceQuota is ready and monitoring resource usage",
		},
	}

	// Exceeded condition
	exceededCondition := metav1.Condition{
		Type:               string(tfv1.GPUResourceQuotaConditionExceeded),
		Status:             metav1.ConditionFalse,
		LastTransitionTime: now,
		Reason:             "QuotaNotExceeded",
		Message:            "Resource usage is within quota limits",
	}
	if exceeded {
		exceededCondition.Status = metav1.ConditionTrue
		exceededCondition.Reason = "QuotaExceeded"
		exceededCondition.Message = "Resource usage has exceeded quota limits"
	}
	conditions = append(conditions, exceededCondition)

	// Alert threshold condition
	alertCondition := metav1.Condition{
		Type:               string(tfv1.GPUResourceQuotaConditionAlertThresholdReached),
		Status:             metav1.ConditionFalse,
		LastTransitionTime: now,
		Reason:             "AlertThresholdNotReached",
		Message:            "Resource usage is below alert threshold",
	}
	if alertReached {
		alertCondition.Status = metav1.ConditionTrue
		alertCondition.Reason = "AlertThresholdReached"
		alertCondition.Message = "Resource usage has reached alert threshold"
	}
	conditions = append(conditions, alertCondition)

	return conditions
}

// AggregateWorkloadResources aggregates workload resources with replicas
func (c *Calculator) AggregateWorkloadResources(baseResource tfv1.Resource, replicas int32) tfv1.Resource {
	result := tfv1.Resource{}

	// Calculate total requests
	tflops := baseResource.Tflops.DeepCopy()
	tflops.Set(tflops.Value() * int64(replicas))
	result.Tflops = tflops

	vram := baseResource.Vram.DeepCopy()
	vram.Set(vram.Value() * int64(replicas))
	result.Vram = vram

	return result
}

// CreateZeroUsage creates a zero-initialized usage object
func (c *Calculator) CreateZeroUsage() *tfv1.GPUResourceUsage {
	return &tfv1.GPUResourceUsage{
		Requests: tfv1.Resource{
			Tflops: *resource.NewQuantity(0, resource.DecimalSI),
			Vram:   *resource.NewQuantity(0, resource.BinarySI),
		},
		Limits: tfv1.Resource{
			Tflops: *resource.NewQuantity(0, resource.DecimalSI),
			Vram:   *resource.NewQuantity(0, resource.BinarySI),
		},
		Workers: new(int32),
	}
}

// CalculateUsagePercent calculates usage percentage for alert checking
func (c *Calculator) CalculateUsagePercent(quota *tfv1.GPUResourceQuota, usage *tfv1.GPUResourceUsage) map[string]int64 {
	percentages := make(map[string]int64)

	if !quota.Spec.Total.Requests.Tflops.IsZero() && !usage.Requests.Tflops.IsZero() && quota.Spec.Total.Requests.Tflops.Value() > 0 {
		percentages["requests.tflops"] = usage.Requests.Tflops.Value() * 100 / quota.Spec.Total.Requests.Tflops.Value()
	}

	if !quota.Spec.Total.Requests.Vram.IsZero() && !usage.Requests.Vram.IsZero() && quota.Spec.Total.Requests.Vram.Value() > 0 {
		percentages["requests.vram"] = usage.Requests.Vram.Value() * 100 / quota.Spec.Total.Requests.Vram.Value()
	}

	if quota.Spec.Total.MaxWorkers != nil && usage.Workers != nil && *quota.Spec.Total.MaxWorkers > 0 {
		percentages["workers"] = int64(*usage.Workers) * 100 / int64(*quota.Spec.Total.MaxWorkers)
	}

	return percentages
}

// SafeAdd safely adds quantities with nil checks
func (c *Calculator) SafeAdd(a *resource.Quantity, b resource.Quantity) {
	if a == nil || a.IsZero() {
		return
	}
	a.Add(b)
}

// SafeSub safely subtracts b from a, ensuring a doesn't go negative
func (c *Calculator) SafeSub(a *resource.Quantity, b resource.Quantity) {
	if a == nil || a.IsZero() {
		return
	}
	a.Sub(b)
	// Ensure quantity doesn't go negative
	if a.Sign() < 0 {
		a.Set(0)
	}
}
