package quota

import (
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

// ResourceOperation represents a resource operation (increase/decrease)
type ResourceOperation struct {
	TFlops  resource.Quantity
	VRAM    resource.Quantity
	Workers int32
}

// NewResourceOperation creates a resource operation from base resource and replicas
func (c *Calculator) NewResourceOperation(baseResource tfv1.Resource, replicas int32) *ResourceOperation {
	return &ResourceOperation{
		TFlops:  c.multiplyQuantity(baseResource.Tflops, replicas),
		VRAM:    c.multiplyQuantity(baseResource.Vram, replicas),
		Workers: replicas,
	}
}

// NewResourceOperationFromUsage creates a resource operation from current usage
func (c *Calculator) NewResourceOperationFromUsage(usage *tfv1.GPUResourceUsage) *ResourceOperation {
	op := &ResourceOperation{}

	if usage.RequestsTFlops != nil {
		op.TFlops = usage.RequestsTFlops.DeepCopy()
	}
	if usage.RequestsVRAM != nil {
		op.VRAM = usage.RequestsVRAM.DeepCopy()
	}
	if usage.Workers != nil {
		op.Workers = *usage.Workers
	}

	return op
}

// IncreaseUsage safely increases resource usage
func (c *Calculator) IncreaseUsage(usage *tfv1.GPUResourceUsage, operation *ResourceOperation) {
	// Update requests
	c.SafeAdd(usage.RequestsTFlops, operation.TFlops)
	c.SafeAdd(usage.RequestsVRAM, operation.VRAM)

	// Update limits (assume limits = requests for now)
	c.SafeAdd(usage.LimitsTFlops, operation.TFlops)
	c.SafeAdd(usage.LimitsVRAM, operation.VRAM)

	// Update workers
	if usage.Workers != nil {
		*usage.Workers += operation.Workers
	}
}

// DecreaseUsage safely decreases resource usage with bounds checking
func (c *Calculator) DecreaseUsage(usage *tfv1.GPUResourceUsage, operation *ResourceOperation) {
	// Update requests with bounds checking
	c.SafeSub(usage.RequestsTFlops, operation.TFlops)
	c.SafeSub(usage.RequestsVRAM, operation.VRAM)

	// Update limits with bounds checking
	c.SafeSub(usage.LimitsTFlops, operation.TFlops)
	c.SafeSub(usage.LimitsVRAM, operation.VRAM)

	// Update workers with bounds checking
	if usage.Workers != nil {
		if *usage.Workers >= operation.Workers {
			*usage.Workers -= operation.Workers
		} else {
			*usage.Workers = 0
		}
	}
}

// UpdateAvailableQuota updates available quota after usage change
func (c *Calculator) UpdateAvailableQuota(available *tfv1.GPUResourceUsage, operation *ResourceOperation, isIncrease bool) {
	if isIncrease {
		// Increase usage means decrease available
		c.SafeSub(available.RequestsTFlops, operation.TFlops)
		c.SafeSub(available.RequestsVRAM, operation.VRAM)
		c.SafeSub(available.LimitsTFlops, operation.TFlops)
		c.SafeSub(available.LimitsVRAM, operation.VRAM)
		if available.Workers != nil {
			if *available.Workers >= operation.Workers {
				*available.Workers -= operation.Workers
			} else {
				*available.Workers = 0
			}
		}
	} else {
		// Decrease usage means increase available
		c.SafeAdd(available.RequestsTFlops, operation.TFlops)
		c.SafeAdd(available.RequestsVRAM, operation.VRAM)
		c.SafeAdd(available.LimitsTFlops, operation.TFlops)
		c.SafeAdd(available.LimitsVRAM, operation.VRAM)
		if available.Workers != nil {
			*available.Workers += operation.Workers
		}
	}
}

// ApplyUsageOperation atomically applies a usage operation (increase or decrease)
func (c *Calculator) ApplyUsageOperation(currentUsage, available *tfv1.GPUResourceUsage, operation *ResourceOperation, isIncrease bool) {
	if isIncrease {
		c.IncreaseUsage(currentUsage, operation)
		c.UpdateAvailableQuota(available, operation, true)
	} else {
		c.DecreaseUsage(currentUsage, operation)
		c.UpdateAvailableQuota(available, operation, false)
	}
}

// CalculateActualDeallocation calculates actual deallocation amounts clamped to current usage
func (c *Calculator) CalculateActualDeallocation(currentUsage *tfv1.GPUResourceUsage, operation *ResourceOperation) *ResourceOperation {
	actual := &ResourceOperation{}

	// Clamp TFlops deallocation
	if currentUsage.RequestsTFlops != nil && currentUsage.RequestsTFlops.Cmp(operation.TFlops) < 0 {
		actual.TFlops = currentUsage.RequestsTFlops.DeepCopy()
	} else {
		actual.TFlops = operation.TFlops.DeepCopy()
	}

	// Clamp VRAM deallocation
	if currentUsage.RequestsVRAM != nil && currentUsage.RequestsVRAM.Cmp(operation.VRAM) < 0 {
		actual.VRAM = currentUsage.RequestsVRAM.DeepCopy()
	} else {
		actual.VRAM = operation.VRAM.DeepCopy()
	}

	// Clamp workers deallocation
	if currentUsage.Workers != nil {
		actual.Workers = min(*currentUsage.Workers, operation.Workers)
	} else {
		actual.Workers = 0
	}

	return actual
}

// CopyUsageToTotal copies available quota from total quota specification
func (c *Calculator) CopyUsageToTotal(quota *tfv1.GPUResourceQuota) *tfv1.GPUResourceUsage {
	available := &tfv1.GPUResourceUsage{
		Workers: new(int32),
	}

	if quota.Spec.Total.RequestsTFlops != nil {
		copy := quota.Spec.Total.RequestsTFlops.DeepCopy()
		available.RequestsTFlops = &copy
	}
	if quota.Spec.Total.RequestsVRAM != nil {
		copy := quota.Spec.Total.RequestsVRAM.DeepCopy()
		available.RequestsVRAM = &copy
	}
	if quota.Spec.Total.LimitsTFlops != nil {
		copy := quota.Spec.Total.LimitsTFlops.DeepCopy()
		available.LimitsTFlops = &copy
	}
	if quota.Spec.Total.LimitsVRAM != nil {
		copy := quota.Spec.Total.LimitsVRAM.DeepCopy()
		available.LimitsVRAM = &copy
	}
	if quota.Spec.Total.Workers != nil {
		*available.Workers = *quota.Spec.Total.Workers
	}

	return available
}

// multiplyQuantity helper method to multiply quantity by replicas
func (c *Calculator) multiplyQuantity(base resource.Quantity, replicas int32) resource.Quantity {
	result := base.DeepCopy()
	result.Set(result.Value() * int64(replicas))
	return result
}

// CalculateAvailablePercent calculates available percentage for each resource
func (c *Calculator) CalculateAvailablePercent(quota *tfv1.GPUResourceQuota, usage *tfv1.GPUResourceUsage) *tfv1.GPUResourceAvailablePercent {
	percent := &tfv1.GPUResourceAvailablePercent{}

	// Calculate requests.tflops percentage
	if quota.Spec.Total.RequestsTFlops != nil && usage.RequestsTFlops != nil {
		if p := c.calculateResourcePercent(quota.Spec.Total.RequestsTFlops.Value(), usage.RequestsTFlops.Value()); p != nil {
			percent.RequestsTFlops = p
		}
	}

	// Calculate requests.vram percentage
	if quota.Spec.Total.RequestsVRAM != nil && usage.RequestsVRAM != nil {
		if p := c.calculateResourcePercent(quota.Spec.Total.RequestsVRAM.Value(), usage.RequestsVRAM.Value()); p != nil {
			percent.RequestsVRAM = p
		}
	}

	// Calculate limits.tflops percentage
	if quota.Spec.Total.LimitsTFlops != nil && usage.LimitsTFlops != nil {
		if p := c.calculateResourcePercent(quota.Spec.Total.LimitsTFlops.Value(), usage.LimitsTFlops.Value()); p != nil {
			percent.LimitsTFlops = p
		}
	}

	// Calculate limits.vram percentage
	if quota.Spec.Total.LimitsVRAM != nil && usage.LimitsVRAM != nil {
		if p := c.calculateResourcePercent(quota.Spec.Total.LimitsVRAM.Value(), usage.LimitsVRAM.Value()); p != nil {
			percent.LimitsVRAM = p
		}
	}

	// Calculate workers percentage
	if quota.Spec.Total.Workers != nil && usage.Workers != nil {
		total := int64(*quota.Spec.Total.Workers)
		used := int64(*usage.Workers)
		if p := c.calculateResourcePercent(total, used); p != nil {
			percent.Workers = p
		}
	}

	return percent
}

// calculateResourcePercent calculates available percentage for a single resource
func (c *Calculator) calculateResourcePercent(total, used int64) *int64 {
	if total <= 0 {
		return nil
	}
	available := max(0, (total-used)*100/total)
	return &available
}

// IsQuotaExceeded checks if any quota limit is exceeded
func (c *Calculator) IsQuotaExceeded(quota *tfv1.GPUResourceQuota, usage *tfv1.GPUResourceUsage) bool {
	if quota.Spec.Total.RequestsTFlops != nil && usage.RequestsTFlops != nil {
		if usage.RequestsTFlops.Cmp(*quota.Spec.Total.RequestsTFlops) > 0 {
			return true
		}
	}
	if quota.Spec.Total.RequestsVRAM != nil && usage.RequestsVRAM != nil {
		if usage.RequestsVRAM.Cmp(*quota.Spec.Total.RequestsVRAM) > 0 {
			return true
		}
	}
	if quota.Spec.Total.LimitsTFlops != nil && usage.LimitsTFlops != nil {
		if usage.LimitsTFlops.Cmp(*quota.Spec.Total.LimitsTFlops) > 0 {
			return true
		}
	}
	if quota.Spec.Total.LimitsVRAM != nil && usage.LimitsVRAM != nil {
		if usage.LimitsVRAM.Cmp(*quota.Spec.Total.LimitsVRAM) > 0 {
			return true
		}
	}
	if quota.Spec.Total.Workers != nil && usage.Workers != nil {
		if *usage.Workers > *quota.Spec.Total.Workers {
			return true
		}
	}
	return false
}

// IsAlertThresholdReached checks if alert threshold is reached for any resource
func (c *Calculator) IsAlertThresholdReached(availablePercent *tfv1.GPUResourceAvailablePercent, threshold int32) bool {
	thresholdInt64 := int64(100 - threshold)

	if availablePercent.RequestsTFlops != nil && *availablePercent.RequestsTFlops < thresholdInt64 {
		return true
	}
	if availablePercent.RequestsVRAM != nil && *availablePercent.RequestsVRAM < thresholdInt64 {
		return true
	}
	if availablePercent.LimitsTFlops != nil && *availablePercent.LimitsTFlops < thresholdInt64 {
		return true
	}
	if availablePercent.LimitsVRAM != nil && *availablePercent.LimitsVRAM < thresholdInt64 {
		return true
	}
	if availablePercent.Workers != nil && *availablePercent.Workers < thresholdInt64 {
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
		RequestsTFlops: resource.NewQuantity(0, resource.DecimalSI),
		RequestsVRAM:   resource.NewQuantity(0, resource.BinarySI),
		LimitsTFlops:   resource.NewQuantity(0, resource.DecimalSI),
		LimitsVRAM:     resource.NewQuantity(0, resource.BinarySI),
		Workers:        new(int32),
	}
}

// CalculateUsagePercent calculates usage percentage for alert checking
func (c *Calculator) CalculateUsagePercent(quota *tfv1.GPUResourceQuota, usage *tfv1.GPUResourceUsage) map[string]int64 {
	percentages := make(map[string]int64)

	if quota.Spec.Total.RequestsTFlops != nil && usage.RequestsTFlops != nil && quota.Spec.Total.RequestsTFlops.Value() > 0 {
		percentages["requests.tflops"] = usage.RequestsTFlops.Value() * 100 / quota.Spec.Total.RequestsTFlops.Value()
	}

	if quota.Spec.Total.RequestsVRAM != nil && usage.RequestsVRAM != nil && quota.Spec.Total.RequestsVRAM.Value() > 0 {
		percentages["requests.vram"] = usage.RequestsVRAM.Value() * 100 / quota.Spec.Total.RequestsVRAM.Value()
	}

	if quota.Spec.Total.Workers != nil && usage.Workers != nil && *quota.Spec.Total.Workers > 0 {
		percentages["workers"] = int64(*usage.Workers) * 100 / int64(*quota.Spec.Total.Workers)
	}

	return percentages
}

// SafeAdd safely adds quantities with nil checks
func (c *Calculator) SafeAdd(a *resource.Quantity, b resource.Quantity) {
	if a != nil {
		a.Add(b)
	}
}

// SafeSub safely subtracts b from a, ensuring a doesn't go negative
func (c *Calculator) SafeSub(a *resource.Quantity, b resource.Quantity) {
	if a == nil {
		return
	}
	a.Sub(b)
	// Ensure quantity doesn't go negative
	if a.Sign() < 0 {
		a.Set(0)
	}
}
