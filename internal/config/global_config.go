package config

import (
	"fmt"
	"sync/atomic"

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

	// HypervisorMaxCrashCount is the container RestartCount threshold that triggers
	// a full hypervisor pod recreation. Once any container on the hypervisor pod
	// exceeds this count, the pod is deleted so the next reconcile recreates it.
	// Set to 0 to disable. If nil, falls back to DefaultHypervisorMaxCrashCount.
	HypervisorMaxCrashCount *int32 `yaml:"hypervisorMaxCrashCount"`

	// MaxInFlightNodes caps how many nodes the auto-expander may provision
	// concurrently. If nil or <= 0, falls back to DefaultMaxInFlightNodes.
	MaxInFlightNodes *int `yaml:"maxInFlightNodes"`
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

// globalConfig is read from the scheduler hot path (e.g. GetMaxInFlightNodes)
// while a config-watch goroutine updates it via SetGlobalConfig, so it must be
// accessed atomically to avoid data races.
var globalConfig atomic.Pointer[GlobalConfig]

func GetGlobalConfig() *GlobalConfig {
	cfg := globalConfig.Load()
	if cfg == nil {
		fmt.Println("[WARN] trying to get global config before initialized")
		return &GlobalConfig{}
	}
	return cfg
}

func SetGlobalConfig(config *GlobalConfig) {
	globalConfig.Store(config)
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

	// DefaultHypervisorMaxCrashCount is the fallback threshold when
	// HypervisorMaxCrashCount is not configured.
	DefaultHypervisorMaxCrashCount int32 = 5

	// DefaultMaxInFlightNodes is the fallback cap on concurrently
	// provisioning nodes when MaxInFlightNodes is not configured.
	DefaultMaxInFlightNodes = 15
)

// GetMaxInFlightNodes returns the configured cap on concurrently provisioning
// nodes for the auto-expander, or the default value when not configured.
func GetMaxInFlightNodes() int {
	cfg := GetGlobalConfig()
	if cfg.MaxInFlightNodes == nil || *cfg.MaxInFlightNodes <= 0 {
		return DefaultMaxInFlightNodes
	}
	return *cfg.MaxInFlightNodes
}

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
