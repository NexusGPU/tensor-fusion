package worker

import (
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
)

func TestResolveWorkerUID(t *testing.T) {
	lookup := map[string]string{"ns1/pod-a": "uid-a"}
	uids := map[string]struct{}{"uid-a": {}, "uid-b": {}}

	tests := []struct {
		name    string
		mi      *framework.ProcessMappingInfo
		wantUID string
		wantOK  bool
	}{
		{
			name:    "environ identity wins",
			mi:      &framework.ProcessMappingInfo{Namespace: "ns1", PodName: "pod-a", PodUID: "uid-b"},
			wantUID: "uid-a",
			wantOK:  true,
		},
		{
			name:    "falls back to cgroup pod UID when environ stripped",
			mi:      &framework.ProcessMappingInfo{Namespace: "", PodName: "", PodUID: "uid-b"},
			wantUID: "uid-b",
			wantOK:  true,
		},
		{
			name:   "unknown pod UID is not attributed",
			mi:     &framework.ProcessMappingInfo{PodUID: "uid-unknown"},
			wantOK: false,
		},
		{
			name:   "no identity at all",
			mi:     &framework.ProcessMappingInfo{},
			wantOK: false,
		},
		{
			name:   "nil mapping info",
			mi:     nil,
			wantOK: false,
		},
		{
			name:    "environ miss falls through to cgroup UID",
			mi:      &framework.ProcessMappingInfo{Namespace: "ns1", PodName: "pod-unknown", PodUID: "uid-b"},
			wantUID: "uid-b",
			wantOK:  true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotUID, gotOK := resolveWorkerUID(tt.mi, lookup, uids)
			if gotOK != tt.wantOK || (tt.wantOK && gotUID != tt.wantUID) {
				t.Errorf("resolveWorkerUID() = (%q, %v), want (%q, %v)", gotUID, gotOK, tt.wantUID, tt.wantOK)
			}
		})
	}
}
