package config

type GPUFitConfig struct {
	MaxWorkerPerNode int `json:"maxWorkerPerNode"`

	VramWeight   float64 `json:"vramWeight"`
	TflopsWeight float64 `json:"tflopsWeight"`

	// EnableNvLinkAware turns on topology-aware score and reserve behavior.
	EnableNvLinkAware bool `json:"enableNvlinkAware"`

	// ResourceScoreWeight is used when combining legacy resource score and topology score.
	ResourceScoreWeight float64 `json:"resourceScoreWeight"`

	// TopologyScoreWeight is used when combining legacy resource score and topology score.
	TopologyScoreWeight float64 `json:"topologyScoreWeight"`

	// SingleGPUProtectWeight controls how much single-GPU workloads avoid high-connectivity GPUs.
	SingleGPUProtectWeight float64 `json:"singleGpuProtectWeight"`
}

type GPUNetworkTopologyAwareConfig struct {
	TotalIntranetBandWidthGBps int64 `json:"totalIntranetBandWidthGBps"`
}
