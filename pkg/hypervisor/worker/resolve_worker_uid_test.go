package worker

import (
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
)

func TestResolveWorkerUID(t *testing.T) {
	lookup := map[string]string{"ns1/pod-a": "uid-a", "ns1/pod-b": "uid-b"}
	uids := map[string]struct{}{"uid-a": {}, "uid-b": {}}

	tests := []struct {
		name    string
		mi      *framework.ProcessMappingInfo
		wantUID string
		wantOK  bool
	}{
		{
			name:    "cgroup and environment agree",
			mi:      &framework.ProcessMappingInfo{Namespace: "ns1", PodName: "pod-a", PodUID: "uid-a"},
			wantUID: "uid-a",
			wantOK:  true,
		},
		{
			name:    "cgroup identity is used when the environment names a different worker",
			mi:      &framework.ProcessMappingInfo{Namespace: "ns1", PodName: "pod-a", PodUID: "uid-b"},
			wantUID: "uid-b",
			wantOK:  true,
		},
		{
			name:    "cgroup identity is used when the environment was stripped",
			mi:      &framework.ProcessMappingInfo{Namespace: "", PodName: "", PodUID: "uid-b"},
			wantUID: "uid-b",
			wantOK:  true,
		},
		{
			name:    "cgroup identity is used when the environment names an unknown pod",
			mi:      &framework.ProcessMappingInfo{Namespace: "ns1", PodName: "pod-unknown", PodUID: "uid-b"},
			wantUID: "uid-b",
			wantOK:  true,
		},
		{
			name:   "untracked pod is not attributed",
			mi:     &framework.ProcessMappingInfo{PodUID: "uid-unknown"},
			wantOK: false,
		},
		{
			name:   "untracked pod is not attributed to the worker its environment names",
			mi:     &framework.ProcessMappingInfo{Namespace: "ns1", PodName: "pod-a", PodUID: "uid-unknown"},
			wantOK: false,
		},
		{
			name:    "environment is used when there is no cgroup identity",
			mi:      &framework.ProcessMappingInfo{Namespace: "ns1", PodName: "pod-a"},
			wantUID: "uid-a",
			wantOK:  true,
		},
		{
			name:   "environment is not used when the cgroup could not be read",
			mi:     &framework.ProcessMappingInfo{Namespace: "ns1", PodName: "pod-a", PodUIDUnknown: true},
			wantOK: false,
		},
		{
			name:   "environment naming an unknown pod without cgroup identity",
			mi:     &framework.ProcessMappingInfo{Namespace: "ns1", PodName: "pod-unknown"},
			wantOK: false,
		},
		{
			name:   "incomplete environment without cgroup identity",
			mi:     &framework.ProcessMappingInfo{PodName: "pod-a"},
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

// GPU memory is accounted to the pod whose cgroup the process runs in, whatever
// pod identity the process carries in its own environment.
func TestGPUMemoryIsAccountedToTheCgroupOwner(t *testing.T) {
	const deviceUUID = "gpu-0"
	owner := &api.WorkerInfo{WorkerUID: "uid-a", Namespace: "ns1", WorkerName: "pod-a"}
	neighbour := &api.WorkerInfo{WorkerUID: "uid-b", Namespace: "ns1", WorkerName: "pod-b"}
	controller := &WorkerController{
		workers: map[string]*api.WorkerInfo{owner.WorkerUID: owner, neighbour.WorkerUID: neighbour},
		backend: &fakeWorkerBackend{mappings: map[uint32]*framework.ProcessMappingInfo{
			// Runs in pod-a, started with pod-b's name in its environment.
			100: {HostPID: 100, GuestPID: 7, PodUID: owner.WorkerUID, Namespace: "ns1", PodName: "pod-b"},
			// Runs in pod-b with a consistent environment.
			200: {HostPID: 200, GuestPID: 8, PodUID: neighbour.WorkerUID, Namespace: "ns1", PodName: "pod-b"},
			// Runs in a pod this hypervisor does not manage.
			300: {HostPID: 300, GuestPID: 9, PodUID: "uid-untracked", Namespace: "ns1", PodName: "pod-b"},
		}},
		deviceController: &fakeDeviceController{processInfos: []api.ProcessInformation{
			{ProcessID: "100", DeviceUUID: deviceUUID, MemoryUsedBytes: 4 << 30},
			{ProcessID: "200", DeviceUUID: deviceUUID, MemoryUsedBytes: 1 << 30},
			{ProcessID: "300", DeviceUUID: deviceUUID, MemoryUsedBytes: 2 << 30},
		}},
	}

	usage, _, _ := controller.collectWorkerMemoryUsage(controller.buildWorkerLookupMap())
	if got := usage[owner.WorkerUID][deviceUUID]; got != 4<<30 {
		t.Errorf("owner usage = %d, want %d", got, uint64(4<<30))
	}
	if got := usage[neighbour.WorkerUID][deviceUUID]; got != 1<<30 {
		t.Errorf("neighbour usage = %d, want %d", got, uint64(1<<30))
	}

	metrics, err := controller.GetWorkerMetrics()
	if err != nil {
		t.Fatalf("GetWorkerMetrics() error = %v", err)
	}
	if got := len(metrics[deviceUUID][owner.WorkerUID]); got != 1 {
		t.Errorf("owner has %d processes in metrics, want 1", got)
	}
	if got := len(metrics[deviceUUID][neighbour.WorkerUID]); got != 1 {
		t.Errorf("neighbour has %d processes in metrics, want 1", got)
	}
}
