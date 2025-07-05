package config

type GlobalConfig struct {
	MetricsTTL            string   `yaml:"metricsTTL"`
	MetricsFormat         string   `yaml:"metricsFormat"`
	MetricsExtraPodLabels []string `yaml:"metricsExtraPodLabels"`

	AlertRules []AlertRule `yaml:"alertRules"`
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
