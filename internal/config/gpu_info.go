package config

import (
	"os"

	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/yaml"
)

type GpuInfo struct {
	Model         string            `json:"model"`
	Vendor        string            `json:"vendor"`
	CostPerHour   float64           `json:"costPerHour"`
	Fp16TFlops    resource.Quantity `json:"fp16TFlops"`
	FullModelName string            `json:"fullModelName"`
}

func LoadGpuInfoFromFile(filename string) ([]GpuInfo, error) {
	infos := make([]GpuInfo, 0)
	data, err := os.ReadFile(filename)
	if err != nil {
		return infos, err
	}
	err = yaml.Unmarshal(data, &infos)
	if err != nil {
		return nil, err
	}
	return infos, nil
}

func MockGpuInfo() []GpuInfo {
	return []GpuInfo{
		{
			Model:         "mock",
			Vendor:        "mock",
			CostPerHour:   0.1,
			Fp16TFlops:    resource.MustParse("1000"),
			FullModelName: "mock",
		},
	}
}
