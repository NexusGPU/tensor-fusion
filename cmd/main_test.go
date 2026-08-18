package main

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/require"
)

func TestResolveIsolationModePolicy(t *testing.T) {
	tests := []struct {
		name    string
		raw     string
		want    tfv1.IsolationModePolicyType
		wantErr bool
	}{
		{name: "unset defaults to static", want: tfv1.IsolationModePolicyStatic},
		{name: "static", raw: " static ", want: tfv1.IsolationModePolicyStatic},
		{name: "dynamic", raw: "DYNAMIC", want: tfv1.IsolationModePolicyDynamic},
		{name: "invalid", raw: "mixed", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := resolveIsolationModePolicy(tt.raw)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}
}
