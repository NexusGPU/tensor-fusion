package config

import "github.com/NexusGPU/tensor-fusion/internal/alert"

type GlobalConfig struct {
	MetricsTTL string       `yaml:"metricsTTL"`
	AlertRules []alert.Rule `yaml:"alertRules"`
}

func MockGlobalConfig() *GlobalConfig {
	return &GlobalConfig{
		MetricsTTL: "30d",
		AlertRules: []alert.Rule{
			{
				Name:               "mock",
				Query:              "mock",
				Threshold:          1,
				EvaluationInterval: "1m",
				ConsecutiveCount:   2,
				Severity:           "P1",
				Summary:            "mock",
				Description:        "mock",
			},
		},
	}
}
