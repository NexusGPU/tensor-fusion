package alibaba

import (
	"fmt"

	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
)

var GPUInstanceTypeInfo []types.GPUNodeInstanceInfo

// Some regions are more expensive or cheaper than others, if not found in this map, use 1.0 as default ratio
// TODO: this should be configurable, also indicating some special discounts
var RegionCostDifferenceRatio = map[string]float64{
	"us-west-1":   1.0,
	"cn-hangzhou": 0.7,
}

var PricingMap = map[string]*types.GPUNodeInstanceInfo{}

const SPOT_DISCOUNT_RATIO = 0.3

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
			CostPerHour:  1.344,

			CPUs:      8,
			MemoryGiB: 31,

			FP16TFlopsPerGPU:    65,
			VRAMGigabytesPerGPU: 15,

			GPUModel:        "NVIDIA T4",
			GPUCount:        1,
			GPUArchitecture: types.GPUArchitectureNvidiaTuring,
			CPUArchitecture: types.CPUArchitectureAMD64,
		},

		{
			InstanceType: "ecs.gn6i-c16g1.4xlarge",
			CostPerHour:  1.729,

			CPUs:      16,
			MemoryGiB: 60,

			FP16TFlopsPerGPU:    65,
			VRAMGigabytesPerGPU: 15,

			GPUModel:        "NVIDIA T4",
			GPUCount:        1,
			GPUArchitecture: types.GPUArchitectureNvidiaTuring,
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

func (p AlibabaGPUNodeProvider) GetInstancePricing(instanceType string, region string, capacityType types.CapacityTypeEnum) (float64, error) {
	discountRatio := 1.0
	if ratio, ok := RegionCostDifferenceRatio[region]; ok {
		discountRatio = ratio
	}

	if capacityType == types.CapacityTypeSpot {
		// TODO: this should be dynamic, on average Spot instance can save 70% more, before get accurate Spot discount ratio, just use fix value for now
		discountRatio = discountRatio * SPOT_DISCOUNT_RATIO
	}

	if PricingMap[instanceType] == nil {
		// Should not happen after get all pricing data of all popular instance types
		return 0, fmt.Errorf("instance type not found: %s", instanceType)
	}
	return PricingMap[instanceType].CostPerHour * discountRatio, nil
}
