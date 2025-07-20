package config

import "fmt"

type GlobalConfig struct {
	MetricsTTL            string   `yaml:"metricsTTL"`
	MetricsFormat         string   `yaml:"metricsFormat"`
	MetricsExtraPodLabels []string `yaml:"metricsExtraPodLabels"`

	AlertRules []AlertRule `yaml:"alertRules"`
}

var globalConfig *GlobalConfig

func GetGlobalConfig() *GlobalConfig {
	if globalConfig == nil {
		fmt.Println("[WARN] trying to get global config before initialized")
		return &GlobalConfig{}
	}
	return globalConfig
}

func SetGlobalConfig(config *GlobalConfig) {
	globalConfig = config
}

const (
	// Default format for fast greptimedb ingestion
	// See https://docs.influxdata.com/influxdb/v2/reference/syntax/line-protocol/
	MetricsFormatInflux = "influx"

	// Json format with { "measure", "tag", "field", "ts"}
	MetricsFormatJson = "json"

	// Open telemetry format
	MetricsFormatOTel = "otel"
)

func MockGlobalConfig() *GlobalConfig {
	return &GlobalConfig{
		MetricsTTL:            "30d",
		MetricsFormat:         "influx",
		MetricsExtraPodLabels: []string{"kubernetes.io/app"},
		AlertRules: []AlertRule{
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
