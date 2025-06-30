package quota

import (
	"fmt"
	"strconv"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

// Calculator provides shared quota calculation logic
type Calculator struct{}

const QuotaReadyAndCapacitySufficient = "GPUResourceQuota is ready and capacity is sufficient"

// NewCalculator creates a new quota calculator
func NewCalculator() *Calculator {
	return &Calculator{}
}

// IncreaseUsage safely increases resource usage
func (c *Calculator) IncreaseUsage(usage *tfv1.GPUResourceUsage, allocation *tfv1.AllocRequest) {
	if allocation == nil {
		return
	}

	count := int64(allocation.Count)
	c.SafeAdd(&usage.Requests.Tflops, allocation.Request.Tflops, count)
	c.SafeAdd(&usage.Requests.Vram, allocation.Request.Vram, count)
	c.SafeAdd(&usage.Limits.Tflops, allocation.Limit.Tflops, count)
	c.SafeAdd(&usage.Limits.Vram, allocation.Limit.Vram, count)
	usage.Workers += 1
}

// DecreaseUsage safely decreases resource usage with bounds checking
func (c *Calculator) DecreaseUsage(usage *tfv1.GPUResourceUsage, allocation *tfv1.AllocRequest) {
	count := int64(allocation.Count)
	c.SafeSub(&usage.Requests.Tflops, allocation.Request.Tflops, count)
	c.SafeSub(&usage.Requests.Vram, allocation.Request.Vram, count)
	c.SafeSub(&usage.Limits.Tflops, allocation.Limit.Tflops, count)
	c.SafeSub(&usage.Limits.Vram, allocation.Limit.Vram, count)
	usage.Workers -= 1
}

// ApplyUsageOperation atomically applies a usage operation (increase or decrease)
func (c *Calculator) ApplyUsageOperation(currentUsage *tfv1.GPUResourceUsage, allocation *tfv1.AllocRequest, isIncrease bool) {
	if isIncrease {
		c.IncreaseUsage(currentUsage, allocation)
	} else {
		c.DecreaseUsage(currentUsage, allocation)
	}
}

// CalculateAvailablePercent calculates available percentage for each resource
func (c *Calculator) CalculateAvailablePercent(quota *tfv1.GPUResourceQuota, usage *tfv1.GPUResourceUsage) *tfv1.GPUResourceAvailablePercent {
	percent := &tfv1.GPUResourceAvailablePercent{
		RequestsTFlops: "100",
		RequestsVRAM:   "100",
		LimitsTFlops:   "100",
		LimitsVRAM:     "100",
		Workers:        "100",
	}
	if quota == nil || usage == nil {
		return percent
	}

	totalReq := quota.Spec.Total.Requests
	if totalReq != nil {
		percent.RequestsTFlops = c.calculateResourcePercent(totalReq.Tflops, usage.Requests.Tflops)
		percent.RequestsVRAM = c.calculateResourcePercent(totalReq.Vram, usage.Requests.Vram)
	}

	totalLimit := quota.Spec.Total.Limits
	if totalLimit != nil {
		percent.LimitsTFlops = c.calculateResourcePercent(totalLimit.Tflops, usage.Limits.Tflops)
		percent.LimitsVRAM = c.calculateResourcePercent(totalLimit.Vram, usage.Limits.Vram)
	}

	if quota.Spec.Total.MaxWorkers != nil {
		total := float64(*quota.Spec.Total.MaxWorkers)
		used := float64(usage.Workers)
		if total > 0 {
			percent.Workers = strconv.FormatFloat((total-used)/total*100.0, 'f', 2, 64)
		}
	}

	return percent
}

// calculateResourcePercent calculates available percentage for a single resource
// returns round(2) float string, range [0, 100]
func (c *Calculator) calculateResourcePercent(total, used resource.Quantity) string {
	if total.Value() <= 0 {
		return "100"
	}
	totalFloat := total.AsApproximateFloat64()
	usedFloat := used.AsApproximateFloat64()
	available := max(0, (totalFloat-usedFloat)/totalFloat*100)
	return strconv.FormatFloat(available, 'f', 2, 64)
}

// IsAlertThresholdReached checks if alert threshold is reached for any resource
func (c *Calculator) IsAlertThresholdReached(availablePercent string, threshold int32) bool {
	minAvailablePercent := int64(100 - threshold)
	if availablePercent == "" {
		// should not happen, default to "100"
		return true
	}
	available, _ := strconv.ParseInt(availablePercent, 10, 64)
	return available < minAvailablePercent
}

// CreateStandardConditions creates standard quota conditions
func (c *Calculator) CalculateConditions(availablePercent *tfv1.GPUResourceAvailablePercent, alertThreshold int32, previousMsg string) []metav1.Condition {
	now := metav1.Now()

	alertReachedTFlops := c.IsAlertThresholdReached(availablePercent.RequestsTFlops, alertThreshold)
	alertReachedVRAM := c.IsAlertThresholdReached(availablePercent.RequestsVRAM, alertThreshold)
	alertReachedLimitsTFlops := c.IsAlertThresholdReached(availablePercent.LimitsTFlops, alertThreshold)
	alertReachedLimitsVRAM := c.IsAlertThresholdReached(availablePercent.LimitsVRAM, alertThreshold)
	alertReachedWorkers := c.IsAlertThresholdReached(availablePercent.Workers, alertThreshold)

	if alertReachedTFlops || alertReachedVRAM || alertReachedLimitsTFlops || alertReachedLimitsVRAM || alertReachedWorkers {
		msg := fmt.Sprintf("One of the following resources has reached alert threshold:"+
			" tflops limits: %t, tflops requests: %t, vram limits: %t, vram requests: %t, workers: %t",
			alertReachedLimitsTFlops, alertReachedLimitsVRAM,
			alertReachedTFlops, alertReachedVRAM, alertReachedWorkers)

		if msg != previousMsg {
			return []metav1.Condition{
				{
					Type:               constants.ConditionStatusTypeReady,
					Status:             metav1.ConditionFalse,
					LastTransitionTime: now,
					Reason:             "AlertThresholdReached",
					Message:            msg,
				},
			}
		} else {
			return nil
		}
	}
	if previousMsg != QuotaReadyAndCapacitySufficient {
		return []metav1.Condition{
			{
				Type:               constants.ConditionStatusTypeReady,
				Status:             metav1.ConditionFalse,
				LastTransitionTime: now,
				Reason:             "CapacitySufficient",
				Message:            QuotaReadyAndCapacitySufficient,
			},
		}
	} else {
		return nil
	}
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
		Workers: 0,
	}
}

// SafeAdd safely adds quantities with nil checks
func (c *Calculator) SafeAdd(a *resource.Quantity, b resource.Quantity, count int64) {
	if a == nil {
		return
	}
	b.Mul(count)
	a.Add(b)
}

// SafeSub safely subtracts b from a, ensuring a doesn't go negative
func (c *Calculator) SafeSub(a *resource.Quantity, b resource.Quantity, count int64) {
	if a == nil {
		return
	}
	b.Mul(count)
	a.Sub(b)
	if a.Sign() < 0 {
		a.Set(0)
	}
}
