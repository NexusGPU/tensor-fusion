package kubernetes

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGetPodUIDFromCgroup(t *testing.T) {
	tests := []struct {
		name    string
		cgroup  string
		wantUID string
	}{
		{
			name:    "systemd driver (underscores)",
			cgroup:  "0::/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pode38c7b11_d82c_4169_b866_361c5d7103af.slice/cri-containerd-abc123.scope\n",
			wantUID: "e38c7b11-d82c-4169-b866-361c5d7103af",
		},
		{
			name:    "cgroupfs driver (dashes)",
			cgroup:  "11:memory:/kubepods/burstable/pode38c7b11-d82c-4169-b866-361c5d7103af/abc123\n",
			wantUID: "e38c7b11-d82c-4169-b866-361c5d7103af",
		},
		{
			name:    "guaranteed qos (no qos sub-slice)",
			cgroup:  "0::/kubepods.slice/kubepods-pode38c7b11_d82c_4169_b866_361c5d7103af.slice/cri-containerd-abc.scope\n",
			wantUID: "e38c7b11-d82c-4169-b866-361c5d7103af",
		},
		{
			name:    "non-pod cgroup yields empty",
			cgroup:  "0::/system.slice/containerd.service\n",
			wantUID: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "cgroup"), []byte(tt.cgroup), 0o644); err != nil {
				t.Fatalf("write cgroup: %v", err)
			}
			if got := getPodUIDFromCgroup(dir); got != tt.wantUID {
				t.Errorf("getPodUIDFromCgroup() = %q, want %q", got, tt.wantUID)
			}
		})
	}
}

func TestGetPodUIDFromCgroupMissingFile(t *testing.T) {
	if got := getPodUIDFromCgroup(t.TempDir()); got != "" {
		t.Errorf("expected empty UID for missing cgroup file, got %q", got)
	}
}
