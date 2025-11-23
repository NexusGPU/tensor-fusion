package config

import (
	"k8s.io/apimachinery/pkg/api/resource"
)

type GpuInfo struct {
	Model         string            `json:"model"`
	Vendor        string            `json:"vendor"`
	CostPerHour   float64           `json:"costPerHour"`
	Fp16TFlops    resource.Quantity `json:"fp16TFlops"`
	FullModelName string            `json:"fullModelName"`

	// PartitionTemplates contains available partition templates for this GPU (e.g., MIG profiles)
	// Only applicable for GPUs that support hardware partitioning
	PartitionTemplates []PartitionTemplateInfo `json:"partitionTemplates,omitempty"`

	// MaxPartitions is the maximum number of partitions this GPU can support (e.g., 7 for MIG)
	MaxPartitions uint32 `json:"maxPartitions,omitempty"`

	// MaxPlacementSlots is the maximum number of placement slots this GPU can support (e.g., 8 for NVIDIA MIG)
	MaxPlacementSlots uint32 `json:"maxPlacementSlots,omitempty"`
}

// PartitionTemplateInfo contains detailed resource information for a partition template
type PartitionTemplateInfo struct {
	// TemplateID is the unique identifier for this partition template Profile `19` for 1g.10gb in A100
	TemplateID string `json:"templateId"`

	// TemplateID is the unique identifier (e.g., "1g.24gb", "4g.94gb")
	Name string `json:"name"`

	// MemoryGigabytes is the memory allocated to this partition in gigabytes
	MemoryGigabytes uint64 `json:"memoryGigabytes"`

	// ComputePercent is the percent of sliced GPU (0-100)
	ComputePercent float64 `json:"computePercent"`

	// Description provides additional information about this template
	Description string `json:"description,omitempty"`

	// MaxPartition for this single template, eg. 1g.10gb+me can only be allocate once
	MaxPartition uint32 `json:"maxPartition"`

	// The placement limit for this template, use a bitmask to represent the placement limit
	// e.g. sudo nvidia-smi mig -i 0 -lgipp
	// GPU  0 Profile ID 19 Placements: {0,1,2,3,4,5,6}:1
	// GPU  0 Profile ID 20 Placements: {0,1,2,3,4,5,6}:1
	// GPU  0 Profile ID 15 Placements: {0,2,4,6}:2
	// GPU  0 Profile ID 14 Placements: {0,2,4}:2
	// GPU  0 Profile ID  9 Placements: {0,4}:4
	// GPU  0 Profile ID  5 Placement : {0}:4
	// GPU  0 Profile ID  0 Placement : {0}:8
	PlacementLimit  []uint32 `json:"placementLimit"`
	PlacementOffSet uint32   `json:"placementOffSet"`
}

func MockGpuInfo() *[]GpuInfo {
	return &[]GpuInfo{
		{
			Model:         "mock",
			Vendor:        "mock",
			CostPerHour:   0.1,
			Fp16TFlops:    resource.MustParse("1000"),
			FullModelName: "mock",
		},
		{
			Model:         "L4",
			Vendor:        "NVIDIA",
			CostPerHour:   0.8,
			Fp16TFlops:    resource.MustParse("121"),
			FullModelName: "NVIDIA L4",
		},
		{
			Model:         "A100",
			Vendor:        "NVIDIA",
			CostPerHour:   2.0,
			Fp16TFlops:    resource.MustParse("312"),
			FullModelName: "NVIDIA A100",
		},
		{
			Model:         "H100",
			Vendor:        "NVIDIA",
			CostPerHour:   2.5,
			Fp16TFlops:    resource.MustParse("989"),
			FullModelName: "NVIDIA H100",
		},
		{
			Model:         "H200",
			Vendor:        "NVIDIA",
			CostPerHour:   2.5,
			Fp16TFlops:    resource.MustParse("989"),
			FullModelName: "NVIDIA H200",
		},
	}
}
