package config

import (
	"testing"

	"k8s.io/utils/ptr"
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
