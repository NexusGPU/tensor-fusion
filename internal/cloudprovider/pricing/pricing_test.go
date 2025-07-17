package pricing

import (
	_ "embed"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/stretchr/testify/assert"
)

func TestStaticPricingProvider_AWS(t *testing.T) {
	provider := NewStaticPricingProvider()
	SetTflopsMapAndInitGPUPricingInfo(t.Context(), config.MockGpuInfo())

	// Test AWS instance pricing - use an instance type with available pricing
	instanceType := "g6.xlarge"
	region := "us-east-2"

	// Test on-demand pricing
	price, found := provider.GetPricing(instanceType, tfv1.CapacityTypeOnDemand, region)
	if found {
		assert.Greater(t, price, 0.0, "AWS on-demand price should be greater than 0")
		t.Logf("AWS %s on-demand price: $%.4f/hour", instanceType, price)
	}

	// Test getting pricing with capacity type
	onDemandPrice, foundOnDemand := provider.GetPricing(instanceType, tfv1.CapacityTypeOnDemand, region)
	if foundOnDemand {
		assert.Greater(t, onDemandPrice, 0.0, "AWS on-demand price should be greater than 0")
		t.Logf("AWS %s on-demand price (via GetPricing): $%.4f/hour", instanceType, onDemandPrice)
	}

	reservedPrice, foundReserved := provider.GetPricing(instanceType, tfv1.CapacityTypeReserved, region)
	if foundReserved {
		assert.GreaterOrEqual(t, reservedPrice, 0.0, "AWS reserved price should be >= 0")
		t.Logf("AWS %s reserved price: $%.4f/hour", instanceType, reservedPrice)
	}

	// Test getting GPU instance info by instance type
	info, foundInfo := provider.GetGPUNodeInstanceTypeInfoByInstance(instanceType, region)
	if foundInfo {
		assert.Equal(t, instanceType, info.InstanceType, "Instance type should match")
		assert.Greater(t, info.GPUCount, int32(0), "GPU count should be greater than 0")
		assert.NotEmpty(t, info.GPUModel, "GPU model should not be empty")

		t.Logf("AWS %s GPU info:", instanceType)
		t.Logf("  GPU Model: %s", info.GPUModel)
		t.Logf("  GPU Count: %d", info.GPUCount)
		t.Logf("  FP16 TFlops per GPU: %f", info.FP16TFlopsPerGPU)
		t.Logf("  VRAM per GPU: %d GB", info.VRAMGigabytesPerGPU)
		t.Logf("  Memory: %d GiB", info.MemoryGiB)
		t.Logf("  CPU Architecture: %s", info.CPUArchitecture)
	}

	// Test getting all instance types
	instanceTypes, foundTypes := provider.GetRegionalGPUNodeInstanceTypes(region)
	if foundTypes {
		assert.Greater(t, len(instanceTypes), 0, "Should have some instance types")
		t.Logf("Found %d total instance types in region %s", len(instanceTypes), region)

		// Log first few instance types
		for i, instanceType := range instanceTypes {
			if i < 5 {
				t.Logf("  Instance type %d: %s", i+1, instanceType.InstanceType)
			}
		}
	}
}

func TestStaticPricingProvider_Azure(t *testing.T) {
	provider := NewStaticPricingProvider()
	SetTflopsMapAndInitGPUPricingInfo(t.Context(), config.MockGpuInfo())

	// Test Azure instance pricing
	instanceType := "ND12s"
	region := "eastus"

	// Test on-demand pricing
	price, found := provider.GetPricing(instanceType, tfv1.CapacityTypeOnDemand, region)
	if found {
		assert.Greater(t, price, 0.0, "Azure on-demand price should be greater than 0")
		t.Logf("Azure %s on-demand price: $%.4f/hour", instanceType, price)
	}

	// Test getting pricing with capacity type
	onDemandPrice, foundOnDemand := provider.GetPricing(instanceType, tfv1.CapacityTypeOnDemand, region)
	if foundOnDemand {
		assert.Greater(t, onDemandPrice, 0.0, "Azure on-demand price should be greater than 0")
		t.Logf("Azure %s on-demand price (via GetPricing): $%.4f/hour", instanceType, onDemandPrice)
	}

	reservedPrice, foundReserved := provider.GetPricing(instanceType, tfv1.CapacityTypeReserved, region)
	if foundReserved {
		assert.GreaterOrEqual(t, reservedPrice, 0.0, "Azure reserved price should be >= 0")
		t.Logf("Azure %s reserved price: $%.4f/hour", instanceType, reservedPrice)
	}

	spotPrice, foundSpot := provider.GetPricing(instanceType, tfv1.CapacityTypeSpot, region)
	if foundSpot {
		assert.GreaterOrEqual(t, spotPrice, 0.0, "Azure spot price should be >= 0")
		t.Logf("Azure %s spot price: $%.4f/hour", instanceType, spotPrice)
	}

	// Test getting GPU instance info by instance type
	info, foundInfo := provider.GetGPUNodeInstanceTypeInfoByInstance(instanceType, region)
	if foundInfo {
		assert.Equal(t, instanceType, info.InstanceType, "Instance type should match")
		assert.Greater(t, info.GPUCount, int32(0), "GPU count should be greater than 0")
		assert.NotEmpty(t, info.GPUModel, "GPU model should not be empty")

		t.Logf("Azure %s GPU info:", instanceType)
		t.Logf("  GPU Model: %s", info.GPUModel)
		t.Logf("  GPU Count: %d", info.GPUCount)
		t.Logf("  FP16 TFlops per GPU: %f", info.FP16TFlopsPerGPU)
		t.Logf("  VRAM per GPU: %d GB", info.VRAMGigabytesPerGPU)
		t.Logf("  Memory: %d GiB", info.MemoryGiB)
		t.Logf("  CPU Architecture: %s", info.CPUArchitecture)
	}
}

func TestParseHelperFunctions(t *testing.T) {
	// Test parsePrice
	assert.Equal(t, 21.5, parsePrice("$21.5000 hourly"))
	assert.Equal(t, 0.0, parsePrice("unavailable"))
	assert.Equal(t, 98.32, parsePrice("$98.3200 hourly"))

	// Test parseGPUCount
	assert.Equal(t, int32(8), parseGPUCount("8"))
	assert.Equal(t, int32(16), parseGPUCount("16"))
	assert.Equal(t, int32(0), parseGPUCount("invalid"))

	// Test parseMemory
	assert.Equal(t, int32(512), parseMemory("512 GiB"))
	assert.Equal(t, int32(2048), parseMemory("2048 GiB"))
	assert.Equal(t, int32(0), parseMemory("invalid"))

	// Test parseAzureGPUSpec
	count, model := parseAzureGPUSpec("8X V100 (NVlink)")
	assert.Equal(t, int32(8), count)
	assert.Equal(t, "V100", model)

	count, model = parseAzureGPUSpec("1X H100")
	assert.Equal(t, int32(1), count)
	assert.Equal(t, "H100", model)

	count, model = parseAzureGPUSpec("4X A100")
	assert.Equal(t, int32(4), count)
	assert.Equal(t, "A100", model)
}

func TestIsFractionalGPUCount(t *testing.T) {
	provider := NewStaticPricingProvider()
	price, found := provider.GetPricing("NV12ads v710 v5", tfv1.CapacityTypeOnDemand, "us-east-2")
	assert.False(t, found)
	assert.Equal(t, 0.0, price)
}

func TestUnavailableGPUCount(t *testing.T) {
	provider := NewStaticPricingProvider()
	price, found := provider.GetPricing("NG32ads V620 v1", tfv1.CapacityTypeOnDemand, "us-east-2")
	assert.False(t, found)
	assert.Equal(t, 0.0, price)
}

func TestAzureGPUNodeInstanceInfo(t *testing.T) {
	provider := NewStaticPricingProvider()
	SetTflopsMapAndInitGPUPricingInfo(t.Context(), config.MockGpuInfo())

	ND96isr, found := provider.GetGPUNodeInstanceTypeInfoByInstance("ND96isr H200 v5", "eastus")
	assert.True(t, found)
	assert.Equal(t, int32(8), ND96isr.GPUCount)
	assert.Equal(t, "H200", ND96isr.GPUModel)
	assert.Equal(t, int32(141), ND96isr.VRAMGigabytesPerGPU)
}
