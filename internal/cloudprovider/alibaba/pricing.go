package alibaba

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	pricing "github.com/NexusGPU/tensor-fusion/internal/cloudprovider/pricing"
	types "github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
)

var GPUInstanceTypeInfo []types.GPUNodeInstanceInfo

// Some regions are more expensive or cheaper than others, if not found in this map, use 1.0 as default ratio
// TODO: this should be configurable, also indicating some special discounts
var RegionCostDifferenceRatio = map[string]float64{
	"us-west-1":   1.0,
	"cn-hangzhou": 0.7,
}

var PricingMap = map[string]*types.GPUNodeInstanceInfo{}
var pricingProvider *pricing.StaticPricingProvider

func init() {
	// TODO: this is just mock data, should refer Karpenter to get price and append GPU info.
	// Refer
	// https://github.com/aws/karpenter-provider-aws/blob/2cb43bc468d04f2263d43109e0ccc7cbe616e226/pkg/providers/pricing/zz_generated.pricing_aws.go
	// https://github.com/cloudpilot-ai/karpenter-provider-alibabacloud/blob/main/hack/tools/price_gen/price_gen.go
	GPUInstanceTypeInfo = []types.GPUNodeInstanceInfo{
		// Out of stock
		// {
		// 	InstanceType: "ecs.gn6i-c4g1.xlarge",
		// 	CostPerHour:  1.152,

		// 	CPUs:      4,
		// 	MemoryGiB: 15,

		// 	FP16TFlopsPerGPU:    65,
		// 	VRAMGigabytesPerGPU: 15,

		// 	GPUModel:        "NVIDIA T4",
		// 	GPUCount:        1,
		// 	GPUArchitecture: types.GPUArchitectureNvidiaTuring,
		// 	CPUArchitecture: types.CPUArchitectureAMD64,
		// },

		{
			InstanceType: "ecs.gn6i-c8g1.2xlarge",
			// CostPerHour:  1.344,

			CPUs:      8,
			MemoryGiB: 31,

			FP16TFlopsPerGPU:    65,
			VRAMGigabytesPerGPU: 15,

			GPUModel:        "NVIDIA T4",
			GPUCount:        1,
			GPUVendor:       types.GPUVendorNvidia,
			CPUArchitecture: types.CPUArchitectureAMD64,
		},

		{
			InstanceType: "ecs.gn6i-c16g1.4xlarge",
			// CostPerHour:  1.729,

			CPUs:      16,
			MemoryGiB: 60,

			FP16TFlopsPerGPU:    65,
			VRAMGigabytesPerGPU: 15,

			GPUModel:        "NVIDIA T4",
			GPUCount:        1,
			GPUVendor:       types.GPUVendorNvidia,
			CPUArchitecture: types.CPUArchitectureAMD64,
		},
	}

	for _, instanceType := range GPUInstanceTypeInfo {
		PricingMap[instanceType.InstanceType] = &instanceType
	}
}

func (p AlibabaGPUNodeProvider) GetGPUNodeInstanceTypeInfo(region string) []types.GPUNodeInstanceInfo {
	return GPUInstanceTypeInfo
}

func (p AlibabaGPUNodeProvider) GetInstancePricing(instanceType string, capacityType tfv1.CapacityTypeEnum, region string) (float64, error) {
	if price, exists := pricingProvider.GetPricing(instanceType, capacityType, region); exists {
		return price, nil
	}
	return 0, nil
}
