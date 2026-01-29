package config

type GPUFitConfig struct {
	MaxWorkerPerNode int `json:"maxWorkerPerNode"`

	VramWeight   float64 `json:"vramWeight"`
	TflopsWeight float64 `json:"tflopsWeight"`

	// PreemptClusterWide when true or unset allows preempting pods cluster-wide; when false only victims
	// in the preemptor pod's namespace are accepted (validated in Filter phase after DefaultPreemption selects victims).
	PreemptClusterWide *bool `json:"preemptClusterWide,omitempty"`
}

type GPUNetworkTopologyAwareConfig struct {
	TotalIntranetBandWidthGBps int64 `json:"totalIntranetBandWidthGBps"`
}
