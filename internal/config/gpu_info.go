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
