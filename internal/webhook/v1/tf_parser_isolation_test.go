package v1

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

func TestValidateIsolationAndExecutionMode(t *testing.T) {
	tests := []struct {
		name    string
		profile *tfv1.WorkloadProfileSpec
		wantErr bool
	}{
		{
			name: "hard remote is allowed",
			profile: &tfv1.WorkloadProfileSpec{
				Isolation:  tfv1.IsolationModeHard,
				IsLocalGPU: false,
			},
			wantErr: false,
		},
		{
			name: "hard local without sidecar is rejected",
			profile: &tfv1.WorkloadProfileSpec{
				Isolation:     tfv1.IsolationModeHard,
				IsLocalGPU:    true,
				SidecarWorker: false,
			},
			wantErr: true,
		},
		{
			name: "hard local sidecar is allowed",
			profile: &tfv1.WorkloadProfileSpec{
				Isolation:     tfv1.IsolationModeHard,
				IsLocalGPU:    true,
				SidecarWorker: true,
			},
			wantErr: false,
		},
		{
			name: "soft local is allowed",
			profile: &tfv1.WorkloadProfileSpec{
				Isolation:  tfv1.IsolationModeSoft,
				IsLocalGPU: true,
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateIsolationAndExecutionMode(tt.profile)
			if tt.wantErr && err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !tt.wantErr && err != nil {
				t.Fatalf("expected no error, got %v", err)
			}
		})
	}
}
