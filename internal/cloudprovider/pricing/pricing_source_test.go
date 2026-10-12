package pricing

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A rate that was derived from an assumption says so. The AWS data lists no
// spot prices, so an AWS spot rate is an estimate; the Azure data lists them,
// so an Azure spot rate is a listed price; on-demand and reserved rates are
// always listed prices.
func TestHourlyPriceWithSource_TellsListedPricesFromEstimates(t *testing.T) {
	provider, store := newIsolatedProvider(longTestTimeout)
	store.load(t.Context(), []config.GpuInfo{testGpuInfo("L4", "121"), testGpuInfo("V100", "125")})

	price, source, err := provider.HourlyPriceWithSource(awsL4Instance, tfv1.CapacityTypeSpot, testRegion)
	require.NoError(t, err)
	assert.Equal(t, PriceSourceSpotEstimate, source, "the AWS data has no spot price: the rate is an estimate")
	assert.InDelta(t, l4OnDemandPrice*AWSSpotAssumedPriceRatio, price, 1e-9)

	price, source, err = provider.HourlyPriceWithSource(awsL4Instance, tfv1.CapacityTypeOnDemand, testRegion)
	require.NoError(t, err)
	assert.Equal(t, PriceSourceList, source)
	assert.InDelta(t, l4OnDemandPrice, price, 1e-9)

	price, source, err = provider.HourlyPriceWithSource(awsL4Instance, tfv1.CapacityTypeReserved, testRegion)
	require.NoError(t, err)
	assert.Equal(t, PriceSourceList, source)
	assert.InDelta(t, l4ReservedPrice, price, 1e-9)

	price, source, err = provider.HourlyPriceWithSource(azureV100Spot, tfv1.CapacityTypeSpot, "eastus")
	require.NoError(t, err)
	assert.Equal(t, PriceSourceList, source, "the Azure data lists the spot price")
	assert.InDelta(t, v100Spot, price, 1e-9)

	// Without a price there is no source either.
	for _, tc := range []struct {
		instanceType string
		capacityType tfv1.CapacityTypeEnum
		want         error
	}{
		{azureV100Spot, tfv1.CapacityTypeReserved, ErrPriceUnavailable},
		{awsL4Instance, tfv1.CapacityTypeEnum("burstable"), ErrUnsupportedCapacityType},
		{"no-such-instance", tfv1.CapacityTypeSpot, ErrInstanceTypeNotFound},
	} {
		price, source, err = provider.HourlyPriceWithSource(tc.instanceType, tc.capacityType, testRegion)
		require.ErrorIs(t, err, tc.want)
		assert.Zero(t, price)
		assert.Empty(t, source)
	}

	// The function without the source keeps answering with the same rate.
	for _, capacityType := range []tfv1.CapacityTypeEnum{
		tfv1.CapacityTypeOnDemand, tfv1.CapacityTypeReserved, tfv1.CapacityTypeSpot,
	} {
		want, _, err := provider.HourlyPriceWithSource(awsL4Instance, capacityType, testRegion)
		require.NoError(t, err)
		got, err := provider.HourlyPriceForCapacityType(awsL4Instance, capacityType, testRegion)
		require.NoError(t, err)
		assert.Equal(t, want, got, capacityType)
	}
}

// Over the whole embedded data: every spot rate that is marked as an estimate
// is exactly the assumed share of the on-demand price, a row without an
// on-demand price has no estimate, and Azure spot rates are never estimated.
func TestEmbeddedData_EstimatedSpotRatesAreMarked(t *testing.T) {
	aws, azure := embeddedInstanceData()
	require.NotEmpty(t, aws)
	require.NotEmpty(t, azure)

	estimated := 0
	for _, row := range aws {
		if !row.spotEstimated {
			continue
		}
		estimated++
		assert.Positive(t, row.prices[0], "%s: an estimate needs an on-demand price", row.info.InstanceType)
		assert.InDelta(t, row.prices[0]*AWSSpotAssumedPriceRatio, row.prices[2], 1e-9, row.info.InstanceType)
	}
	assert.Positive(t, estimated, "the AWS data is expected to need estimates; drop this once it lists spot prices")

	listed := 0
	for _, row := range azure {
		assert.False(t, row.spotEstimated, "%s: Azure spot rates are never estimated", row.info.InstanceType)
		if row.prices[2] > 0 {
			listed++
		}
	}
	assert.Positive(t, listed, "the Azure data lists spot prices")
}

// A spot price listed in an AWS record is used as it is; only a record without
// one falls back to the estimate.
func TestParseAWSRecord_PrefersAListedSpotPrice(t *testing.T) {
	record := func(onDemand, spot string) []string {
		return []string{
			"G6 Extra Large", "g6.xlarge", "16 GiB", "4 vCPUs", "1", "L4", "24 GiB", "x86_64",
			"Up to 10 Gigabit", onDemand, "$0.5239 hourly", spot, "unavailable", "unavailable",
		}
	}

	_, prices, estimated := parseAWSRecord(record("$0.8048 hourly", "$0.2500 hourly"))
	assert.False(t, estimated)
	assert.InDelta(t, 0.25, prices[2], 1e-9)

	_, prices, estimated = parseAWSRecord(record("$0.8048 hourly", "unavailable"))
	assert.True(t, estimated)
	assert.InDelta(t, 0.8048*AWSSpotAssumedPriceRatio, prices[2], 1e-9)

	// Nothing to estimate from: no spot rate at all, rather than an estimate of 0.
	_, prices, estimated = parseAWSRecord(record("unavailable", "unavailable"))
	assert.False(t, estimated)
	assert.Zero(t, prices[2])
}

// The factor keeps the value other code uses under its older name.
func TestAWSSpotAssumedPriceRatioKeepsItsValue(t *testing.T) {
	assert.Equal(t, constants.SpotInstanceAssumedDiscountRatio, AWSSpotAssumedPriceRatio)
	assert.Greater(t, AWSSpotAssumedPriceRatio, 0.0)
	assert.Less(t, AWSSpotAssumedPriceRatio, 1.0)
}

// A node carries its instance type as the cloud API spells it. An Azure node's
// spelling differs from the price list's; it still finds its listed prices,
// the real spot price among them.
func TestHourlyPriceWithSource_FindsAzureSizesByTheirNodeLabel(t *testing.T) {
	provider, store := newIsolatedProvider(longTestTimeout)
	store.load(t.Context(), []config.GpuInfo{testGpuInfo("V100", "125")})

	listed, source, err := provider.HourlyPriceWithSource(azureV100Spot, tfv1.CapacityTypeSpot, "eastus")
	require.NoError(t, err)
	require.Equal(t, PriceSourceList, source)

	byLabel, source, err := provider.HourlyPriceWithSource("Standard_NC24s_v3", tfv1.CapacityTypeSpot, "eastus")
	require.NoError(t, err)
	require.Equal(t, PriceSourceList, source)
	require.Equal(t, listed, byLabel)

	_, _, err = provider.HourlyPriceWithSource("Standard_Not_A_Size", tfv1.CapacityTypeSpot, "eastus")
	require.ErrorIs(t, err, ErrInstanceTypeNotFound)
	require.Equal(t, "g6.xlarge", azureListName("g6.xlarge"))
}

// A price the data lists as "unavailable" is no price: the instance type is
// not offered for that capacity type, instead of being offered for nothing and
// winning every comparison as the cheapest.
func TestLookupPricing_UnavailableIsNotFree(t *testing.T) {
	provider, store := newIsolatedProvider(longTestTimeout)
	store.load(t.Context(), []config.GpuInfo{testGpuInfo("H200", "500"), testGpuInfo("H100_SXM", "400")})

	// p5en.48xlarge (8x H200) lists an on-demand price and no reserved one.
	price, err := provider.LookupPricing("p5en.48xlarge", tfv1.CapacityTypeOnDemand, testRegion)
	require.NoError(t, err)
	require.Positive(t, price)
	_, err = provider.LookupPricing("p5en.48xlarge", tfv1.CapacityTypeReserved, testRegion)
	require.ErrorIs(t, err, ErrPriceUnavailable)
	_, found := provider.GetPricing("p5en.48xlarge", tfv1.CapacityTypeReserved, testRegion)
	require.False(t, found)

	// p5.48xlarge (8x H100) lists no price at all.
	for _, capacityType := range []tfv1.CapacityTypeEnum{
		tfv1.CapacityTypeOnDemand, tfv1.CapacityTypeSpot, tfv1.CapacityTypeReserved,
	} {
		price, err := provider.LookupPricing("p5.48xlarge", capacityType, testRegion)
		require.Error(t, err, "capacity type %s", capacityType)
		require.Zero(t, price)
		_, found := provider.GetPricing("p5.48xlarge", capacityType, testRegion)
		require.False(t, found, "capacity type %s", capacityType)
	}
}

// Alibaba Cloud ECS instance types are a table of their own: they are found by
// name and priced, and they never show up among the AWS instance types.
func TestAlibabaInstanceTypes_AreKnownAndKeptApartFromAWS(t *testing.T) {
	provider, store := newIsolatedProvider(longTestTimeout)
	store.load(t.Context(), []config.GpuInfo{testGpuInfo("T4", "65")})

	alibaba, err := provider.LookupAlibabaGPUNodeInstanceTypes()
	require.NoError(t, err)
	require.NotEmpty(t, alibaba)
	for _, instance := range alibaba {
		assert.Equal(t, "T4", instance.GPUModel, instance.InstanceType)
		assert.InDelta(t, 65, instance.FP16TFlopsPerGPU, 1e-9, instance.InstanceType)
		assert.Equal(t, int32(16), instance.VRAMGigabytesPerGPU, instance.InstanceType)
	}

	info, found := provider.GetGPUNodeInstanceTypeInfoByInstance("ecs.gn6i-c8g1.2xlarge", "us-east-1")
	require.True(t, found)
	assert.Equal(t, int32(1), info.GPUCount)
	assert.Equal(t, int32(8), info.CPUs)

	price, source, err := provider.HourlyPriceWithSource("ecs.gn6i-c8g1.2xlarge", tfv1.CapacityTypeOnDemand, "us-east-1")
	require.NoError(t, err)
	assert.Equal(t, PriceSourceList, source)
	assert.Positive(t, price)
	_, source, err = provider.HourlyPriceWithSource("ecs.gn6i-c8g1.2xlarge", tfv1.CapacityTypeSpot, "us-east-1")
	require.NoError(t, err)
	assert.Equal(t, PriceSourceSpotEstimate, source, "the Alibaba data has no spot price: the rate is an estimate")

	aws, err := provider.LookupRegionalGPUNodeInstanceTypes(testRegion)
	require.NoError(t, err)
	for _, instance := range aws {
		assert.NotContains(t, instance.InstanceType, "ecs.", "AWS instance types must not contain Alibaba Cloud ones")
	}
}
