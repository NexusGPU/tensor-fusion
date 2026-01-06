package config

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type GlobalConfig struct {
	MetricsTTL            string            `yaml:"metricsTTL"`
	MetricsFormat         string            `yaml:"metricsFormat"`
	MetricsExtraPodLabels map[string]string `yaml:"metricsExtraPodLabels"`

	AlertRules    []AlertRule          `yaml:"alertRules"`
	AutoMigration *AutoMigrationConfig `yaml:"autoMigration"`

	AutoScalingInterval  string `yaml:"autoScalingInterval"`
	GPUOperatorNamespace string `yaml:"gpuOperatorNamespace"`
}

type AutoMigrationConfig struct {
	Enable bool                `yaml:"enable"`
	Scope  *AutoMigrationScope `yaml:"scope"`
}

type AutoMigrationScope struct {
	Includes *AutoMigrationRules `yaml:"includes"`
	Excludes *AutoMigrationRules `yaml:"excludes"`
}

type AutoMigrationRules struct {
	NamespaceNames    []string              `yaml:"namespaceNames"`
	NamespaceSelector *metav1.LabelSelector `yaml:"namespaceSelector"`
	PodSelector       *metav1.LabelSelector `yaml:"podSelector"`
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

	// Default GPU operator namespace
	DefaultGPUOperatorNamespace = "gpu-operator"
)

// GetGPUOperatorNamespace returns the configured GPU operator namespace or default value
func GetGPUOperatorNamespace() string {
	cfg := GetGlobalConfig()
	if cfg.GPUOperatorNamespace == "" {
		return DefaultGPUOperatorNamespace
	}
	return cfg.GPUOperatorNamespace
}

func MockGlobalConfig() *GlobalConfig {
	return &GlobalConfig{
		MetricsTTL:            "30d",
		MetricsFormat:         "influx",
		MetricsExtraPodLabels: map[string]string{"kubernetes.io/app": "app"},
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
		GPUOperatorNamespace: DefaultGPUOperatorNamespace,
	}
}
