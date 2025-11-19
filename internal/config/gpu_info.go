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
}

// PartitionTemplateInfo contains detailed resource information for a partition template
type PartitionTemplateInfo struct {
	// TemplateID is the unique identifier (e.g., "1g.24gb", "4g.94gb")
	TemplateID string `json:"templateId"`

	// Name is a human-readable name
	Name string `json:"name"`

	// MemoryBytes is the memory allocated to this partition in bytes
	MemoryBytes uint64 `json:"memoryBytes"`

	// ComputeUnits is the number of compute units (SMs) allocated
	ComputeUnits uint64 `json:"computeUnits"`

	// Tflops is the TFLOPS capacity of this partition
	Tflops float64 `json:"tflops"`

	// SliceCount is the number of slices (for MIG, this is the denominator, e.g., 7 for 1/7)
	SliceCount uint32 `json:"sliceCount"`

	// IsDefault indicates if this is a default template
	IsDefault bool `json:"isDefault,omitempty"`

	// Description provides additional information about this template
	Description string `json:"description,omitempty"`
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
