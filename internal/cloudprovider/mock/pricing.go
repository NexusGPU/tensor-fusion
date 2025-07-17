package mock

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	types "github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
)

var GPUInstanceTypeInfo []types.GPUNodeInstanceInfo

func init() {
	GPUInstanceTypeInfo = []types.GPUNodeInstanceInfo{
		{
			InstanceType: "mock.g4dn.large",
			// CostPerHour:  2.152,

			CPUs:      4,
			MemoryGiB: 15,

			FP16TFlopsPerGPU:    65,
			VRAMGigabytesPerGPU: 16,

			GPUModel:        "NVIDIA T4",
			GPUCount:        1,
			GPUVendor:       types.GPUVendorNvidia,
			CPUArchitecture: types.CPUArchitectureAMD64,
		},
	}
}

func (p MockGPUNodeProvider) GetGPUNodeInstanceTypeInfo(region string) []types.GPUNodeInstanceInfo {
	return GPUInstanceTypeInfo
}

func (p MockGPUNodeProvider) GetInstancePricing(instanceType string, capacityType tfv1.CapacityTypeEnum, region string) (float64, error) {
	return 42, nil
}
