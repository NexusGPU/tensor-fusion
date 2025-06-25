package config

type GPUFitConfig struct {
	MaxWorkerPerNode int `json:"maxWorkerPerNode"`

	VramWeight   float64 `json:"vramWeight"`
	TflopsWeight float64 `json:"tflopsWeight"`
}
