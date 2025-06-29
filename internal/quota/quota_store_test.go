package quota

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Test constants
const (
	TestNamespace = "test-ns"
	TestPoolName  = "test-pool"
	TestWorkload  = "test-workload"
	TestQuotaName = "test-quota"
)

func TestQuotaCalculator_EdgeCases(t *testing.T) {
	calc := NewCalculator()

	t.Run("zero division safety", func(t *testing.T) {
		quota := createTestQuota(0, 0, 0)
		usage := createZeroUsage()

		percent := calc.CalculateAvailablePercent(quota, usage)
		assert.Equal(t, percent.RequestsTFlops, "100")
		assert.Equal(t, percent.RequestsVRAM, "100")
		assert.Equal(t, percent.Workers, "100")
	})

	t.Run("negative usage protection", func(t *testing.T) {
		usage := createZeroUsage()
		usage.Requests.Tflops.Set(10)

		// Try to subtract more than available
		toSubtract := *resource.NewQuantity(20, resource.DecimalSI)
		calc.SafeSub(&usage.Requests.Tflops, toSubtract, 1)

		// Should clamp to zero
		assert.Equal(t, int64(0), usage.Requests.Tflops.Value())
	})

	t.Run("nil quantity handling", func(t *testing.T) {
		// Should not panic with nil quantities
		var nilQty *resource.Quantity
		testQty := *resource.NewQuantity(10, resource.DecimalSI)

		calc.SafeAdd(nilQty, testQty, 1) // Should not panic
		calc.SafeSub(nilQty, testQty, 1) // Should not panic
	})
}

func TestQuotaStore_ValidationRules(t *testing.T) {
	qs := NewQuotaStore(nil, context.Background())

	t.Run("valid quota configuration", func(t *testing.T) {
		quota := createTestQuota(100, 1000, 10)
		err := qs.validateQuotaConfig(quota)
		assert.NoError(t, err)
	})

	t.Run("negative values validation", func(t *testing.T) {
		quota := createTestQuota(-10, 1000, 10)
		err := qs.validateQuotaConfig(quota)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "requests.tflops cannot be negative")
	})

	t.Run("limits less than requests validation", func(t *testing.T) {
		quota := createTestQuota(100, 1000, 10)
		// Set limits lower than requests
		quota.Spec.Total.Limits.Tflops = *resource.NewQuantity(50, resource.DecimalSI)

		err := qs.validateQuotaConfig(quota)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "limits.tflops cannot be less than requests.tflops")
	})

	t.Run("invalid alert threshold validation", func(t *testing.T) {
		quota := createTestQuota(100, 1000, 10)
		invalidThreshold := int32(150)
		quota.Spec.Total.AlertThresholdPercent = &invalidThreshold

		err := qs.validateQuotaConfig(quota)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "alertThresholdPercent must be between 0 and 100")
	})
}

func createTestQuota(tflops, vram int64, workers int32) *tfv1.GPUResourceQuota {
	return &tfv1.GPUResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name:      TestQuotaName,
			Namespace: TestNamespace,
		},
		Spec: tfv1.GPUResourceQuotaSpec{
			Total: tfv1.GPUResourceQuotaTotal{
				Requests: &tfv1.Resource{
					Tflops: *resource.NewQuantity(tflops, resource.DecimalSI),
					Vram:   *resource.NewQuantity(vram, resource.BinarySI),
				},
				Limits: &tfv1.Resource{
					Tflops: *resource.NewQuantity(tflops, resource.DecimalSI),
					Vram:   *resource.NewQuantity(vram, resource.BinarySI),
				},
				MaxWorkers: &workers,
			},
		},
	}
}

func createZeroUsage() *tfv1.GPUResourceUsage {
	calc := NewCalculator()
	return calc.CreateZeroUsage()
}
