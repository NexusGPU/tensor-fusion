package config

import (
	"testing"

	"k8s.io/utils/ptr"
	"sigs.k8s.io/yaml"
)

func TestGetMaxInFlightNodes(t *testing.T) {
	original := globalConfig.Load()
	t.Cleanup(func() { globalConfig.Store(original) })

	tests := []struct {
		name  string
		value *int
		want  int
	}{
		{name: "unset falls back to default", value: nil, want: DefaultMaxInFlightNodes},
		{name: "zero falls back to default", value: ptr.To(0), want: DefaultMaxInFlightNodes},
		{name: "negative falls back to default", value: ptr.To(-3), want: DefaultMaxInFlightNodes},
		{name: "configured value wins", value: ptr.To(42), want: 42},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			SetGlobalConfig(&GlobalConfig{MaxInFlightNodes: tt.value})
			if got := GetMaxInFlightNodes(); got != tt.want {
				t.Fatalf("GetMaxInFlightNodes() = %d, want %d", got, tt.want)
			}
		})
	}
}

// The currency of cost figures is read from the "currency" key of the global
// config and is unspecified unless an administrator sets it.
func TestGetCurrency(t *testing.T) {
	original := globalConfig.Load()
	t.Cleanup(func() { globalConfig.Store(original) })

	tests := []struct {
		name string
		yaml string
		want string
	}{
		{name: "unset means unspecified", yaml: "metricsTTL: 30d\n", want: ""},
		{name: "empty means unspecified", yaml: "currency: \"\"\n", want: ""},
		{name: "configured value", yaml: "currency: USD\n", want: "USD"},
		{name: "surrounding blanks are dropped", yaml: "currency: \" CNY \"\n", want: "CNY"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &GlobalConfig{}
			if err := yaml.Unmarshal([]byte(tt.yaml), cfg); err != nil {
				t.Fatalf("unmarshal %q: %v", tt.yaml, err)
			}
			SetGlobalConfig(cfg)
			if got := GetCurrency(); got != tt.want {
				t.Fatalf("GetCurrency() = %q, want %q", got, tt.want)
			}
		})
	}
}
