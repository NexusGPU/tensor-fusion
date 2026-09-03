package main

import "testing"

const (
	testIsolationPolicyDynamic = "dynamic"
	testIsolationPolicyStatic  = "static"
)

func TestLegacyCompatibleFlagArgs(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want []string
	}{
		{
			name: "legacy daemon",
			args: []string{"daemon", "--isolation-policy=" + testIsolationPolicyDynamic},
			want: []string{"--isolation-policy=" + testIsolationPolicyDynamic},
		},
		{
			name: "normal flags",
			args: []string{"--isolation-policy=" + testIsolationPolicyDynamic},
			want: []string{"--isolation-policy=" + testIsolationPolicyDynamic},
		},
		{name: "empty", args: nil, want: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := legacyCompatibleFlagArgs(tt.args)
			if len(got) != len(tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
			for i := range tt.want {
				if got[i] != tt.want[i] {
					t.Fatalf("got %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestEnvOrDefault(t *testing.T) {
	t.Setenv("TF_TEST_HYPERVISOR_VALUE", "  "+testIsolationPolicyDynamic+" ")
	if got := envOrDefault("TF_TEST_HYPERVISOR_VALUE", testIsolationPolicyStatic); got != testIsolationPolicyDynamic {
		t.Fatalf("envOrDefault() = %q, want %q", got, testIsolationPolicyDynamic)
	}
	t.Setenv("TF_TEST_HYPERVISOR_VALUE", " ")
	if got := envOrDefault("TF_TEST_HYPERVISOR_VALUE", testIsolationPolicyStatic); got != testIsolationPolicyStatic {
		t.Fatalf("envOrDefault() = %q, want %q", got, testIsolationPolicyStatic)
	}
}
