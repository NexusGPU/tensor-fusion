package kubernetes

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

const (
	testPodUID       = "e38c7b11-d82c-4169-b866-361c5d7103af"
	testPodUIDSlice  = "e38c7b11_d82c_4169_b866_361c5d7103af"
	otherPodUID      = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
	otherPodUIDSlice = "0f1e2d3c_4b5a_6978_8796_a5b4c3d2e1f0"
	containerID      = "5f3c1b7a9d2e4c6f8a0b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f"
)

func TestGetPodUIDFromCgroup(t *testing.T) {
	tests := []struct {
		name    string
		cgroup  string
		wantUID string
	}{
		{
			name:    "systemd driver (underscores)",
			cgroup:  "12:devices:/kubepods-burstable-pode38c7b11_d82c_4169_b866_361c5d7103af.slice\n",
			wantUID: testPodUID,
		},
		{
			name:    "cgroupfs driver (dashes)",
			cgroup:  "11:memory:/kubepods/burstable/pode38c7b11-d82c-4169-b866-361c5d7103af/abc123\n",
			wantUID: testPodUID,
		},
		{
			name:    "guaranteed qos (no qos sub-slice)",
			cgroup:  "0::/kubepods.slice/kubepods-pode38c7b11_d82c_4169_b866_361c5d7103af.slice/cri-containerd-abc.scope\n",
			wantUID: testPodUID,
		},
		{
			name:    "non-pod cgroup yields empty",
			cgroup:  "0::/system.slice/containerd.service\n",
			wantUID: "",
		},
		{
			name: "cgroup v2, systemd driver, burstable",
			cgroup: "0::/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod" + testPodUIDSlice +
				".slice/cri-containerd-" + containerID + ".scope\n",
			wantUID: testPodUID,
		},
		{
			name: "cgroup v2, systemd driver, besteffort, cri-o",
			cgroup: "0::/kubepods.slice/kubepods-besteffort.slice/kubepods-besteffort-pod" + testPodUIDSlice +
				".slice/crio-" + containerID + ".scope\n",
			wantUID: testPodUID,
		},
		{
			name:    "cgroup v2, cgroupfs driver",
			cgroup:  "0::/kubepods/besteffort/pod" + testPodUID + "/" + containerID + "\n",
			wantUID: testPodUID,
		},
		{
			name:    "cgroup v2, cgroupfs driver, guaranteed",
			cgroup:  "0::/kubepods/pod" + testPodUID + "/" + containerID + "\n",
			wantUID: testPodUID,
		},
		{
			name: "cgroup v1, cgroupfs driver, one line per hierarchy",
			cgroup: "12:pids:/kubepods/burstable/pod" + testPodUID + "/" + containerID + "\n" +
				"11:devices:/kubepods/burstable/pod" + testPodUID + "/" + containerID + "\n" +
				"1:name=systemd:/kubepods/burstable/pod" + testPodUID + "/" + containerID + "\n" +
				"0::/\n",
			wantUID: testPodUID,
		},
		{
			name: "cgroup v1, systemd driver, one line per hierarchy",
			cgroup: "0::/\n" +
				"11:memory:/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod" + testPodUIDSlice +
				".slice/docker-" + containerID + ".scope\n" +
				"1:name=systemd:/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod" + testPodUIDSlice +
				".slice/docker-" + containerID + ".scope\n",
			wantUID: testPodUID,
		},
		{
			name: "path seen from another cgroup namespace",
			cgroup: "0::/../../kubepods-burstable.slice/kubepods-burstable-pod" + testPodUIDSlice +
				".slice/cri-containerd-" + containerID + ".scope\n",
			wantUID: testPodUID,
		},
		{
			name: "kubelet running under its own cgroup root",
			cgroup: "0::/kubelet.slice/kubelet-kubepods.slice/kubelet-kubepods-burstable.slice/" +
				"kubelet-kubepods-burstable-pod" + testPodUIDSlice + ".slice/cri-containerd-" + containerID + ".scope\n",
			wantUID: testPodUID,
		},
		{
			name:    "pod-level cgroup without a container level",
			cgroup:  "0::/kubepods/burstable/pod" + testPodUID + "\n",
			wantUID: testPodUID,
		},
		{
			// A container may create cgroups below its own; only the level
			// created by the kubelet identifies the pod.
			name: "nested cgroup carrying a second pod UID, cgroupfs driver",
			cgroup: "0::/kubepods/burstable/pod" + testPodUID + "/" + containerID +
				"/kubepods/burstable/pod" + otherPodUID + "/" + containerID + "\n",
			wantUID: testPodUID,
		},
		{
			name: "nested cgroup carrying a second pod UID, systemd driver",
			cgroup: "0::/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod" + testPodUIDSlice +
				".slice/cri-containerd-" + containerID + ".scope/kubepods-burstable-pod" + otherPodUIDSlice + ".slice\n",
			wantUID: testPodUID,
		},
		{
			name: "static pod keeps its own identity",
			cgroup: "0::/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod0123456789abcdef0123456789abcdef" +
				".slice/cri-containerd-" + containerID + ".scope\n",
			wantUID: "0123456789abcdef0123456789abcdef",
		},
		{
			name:    "pod-like text that is not a path component",
			cgroup:  "0::/user.slice/tripod" + testPodUID + "/session.scope\n",
			wantUID: "",
		},
		{
			name:    "container ID is not mistaken for a pod",
			cgroup:  "0::/system.slice/docker-" + containerID + ".scope\n",
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

// A cgroup that cannot be read leaves the pod of the process unknown, which is
// not the same as a cgroup that names no pod: only the latter lets the
// environment say which worker the process is.
func TestProcessMappingTellsAnUnreadableCgroupFromOneWithoutAPod(t *testing.T) {
	environ := []string{constants.PodNameEnv + "=pod-a", constants.PodNamespaceEnv + "=ns1"}

	outside, err := getWorkerInfoFromProcDir(writeFakeProcEntry(t, fakeProcEntry{
		environ: environ, cgroup: "0::/user.slice/user-1000.slice/session-3.scope\n",
	}), 4242)
	if err != nil {
		t.Fatal(err)
	}
	if outside.PodUID != "" || outside.PodUIDUnknown {
		t.Errorf("process outside every pod: PodUID=%q unknown=%v, want empty and known",
			outside.PodUID, outside.PodUIDUnknown)
	}

	// No cgroup file: the read fails.
	unread, err := getWorkerInfoFromProcDir(writeFakeProcEntry(t, fakeProcEntry{environ: environ}), 4242)
	if err != nil {
		t.Fatal(err)
	}
	if unread.PodUID != "" || !unread.PodUIDUnknown {
		t.Errorf("unreadable cgroup: PodUID=%q unknown=%v, want empty and unknown", unread.PodUID, unread.PodUIDUnknown)
	}
	// The environment is still reported; it just does not identify the pod.
	if unread.PodName != "pod-a" || unread.Namespace != "ns1" {
		t.Errorf("environment not reported: %q/%q", unread.Namespace, unread.PodName)
	}
}

type fakeProcEntry struct {
	environ    []string // nil leaves the file out
	unreadable bool     // environ exists but cannot be read
	cgroup     string
	status     string
}

func writeFakeProcEntry(t *testing.T, entry fakeProcEntry) string {
	t.Helper()
	dir := t.TempDir()
	write := func(name, content string) {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	switch {
	case entry.unreadable:
		// Reading a directory fails for every user, including root.
		if err := os.Mkdir(filepath.Join(dir, "environ"), 0o755); err != nil {
			t.Fatalf("create environ: %v", err)
		}
	case entry.environ != nil:
		write("environ", strings.Join(entry.environ, "\x00")+"\x00")
	}
	if entry.cgroup != "" {
		write("cgroup", entry.cgroup)
	}
	if entry.status != "" {
		write("status", entry.status)
	}
	return dir
}

// The cgroup is assigned by the kubelet; the environment is whatever the
// process was started with. Both are reported, and a process without a readable
// environment is still identified by its cgroup.
func TestProcessMappingReportsCgroupIdentityIndependentlyOfEnvironment(t *testing.T) {
	ownCgroup := "0::/kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod" + testPodUIDSlice +
		".slice/cri-containerd-" + containerID + ".scope\n"
	tests := []struct {
		name          string
		entry         fakeProcEntry
		wantPodUID    string
		wantNamespace string
		wantPodName   string
		wantContainer string
		wantGuestPID  uint32
	}{
		{
			name: "environment and cgroup agree",
			entry: fakeProcEntry{
				environ: []string{"PATH=/usr/bin", "POD_NAMESPACE=team-a", "POD_NAME=train-0", "CONTAINER_NAME=main"},
				cgroup:  ownCgroup,
				status:  "Name:\tpython\nNSpid:\t4242\t17\n",
			},
			wantPodUID: testPodUID, wantNamespace: "team-a", wantPodName: "train-0", wantContainer: "main",
			wantGuestPID: 17,
		},
		{
			name: "environment names a different pod than the cgroup",
			entry: fakeProcEntry{
				environ: []string{"POD_NAMESPACE=team-b", "POD_NAME=serve-0", "CONTAINER_NAME=main"},
				cgroup:  ownCgroup,
			},
			wantPodUID: testPodUID, wantNamespace: "team-b", wantPodName: "serve-0", wantContainer: "main",
			wantGuestPID: 4242,
		},
		{
			name:         "environment without pod identity",
			entry:        fakeProcEntry{environ: []string{"PATH=/usr/bin"}, cgroup: ownCgroup},
			wantPodUID:   testPodUID,
			wantGuestPID: 4242,
		},
		{
			name:         "environment cannot be read",
			entry:        fakeProcEntry{unreadable: true, cgroup: ownCgroup},
			wantPodUID:   testPodUID,
			wantGuestPID: 4242,
		},
		{
			name:         "environment file is gone",
			entry:        fakeProcEntry{cgroup: ownCgroup},
			wantPodUID:   testPodUID,
			wantGuestPID: 4242,
		},
		{
			name: "process outside any pod cgroup",
			entry: fakeProcEntry{
				environ: []string{"POD_NAMESPACE=team-a", "POD_NAME=train-0", "CONTAINER_NAME=main"},
				cgroup:  "0::/system.slice/containerd.service\n",
			},
			wantNamespace: "team-a", wantPodName: "train-0", wantContainer: "main",
			wantGuestPID: 4242,
		},
		{
			name:         "neither environment nor cgroup",
			entry:        fakeProcEntry{},
			wantGuestPID: 4242,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info, err := getWorkerInfoFromProcDir(writeFakeProcEntry(t, tt.entry), 4242)
			if err != nil {
				t.Fatalf("getWorkerInfoFromProcDir() error = %v", err)
			}
			if info.PodUID != tt.wantPodUID {
				t.Errorf("PodUID = %q, want %q", info.PodUID, tt.wantPodUID)
			}
			if info.Namespace != tt.wantNamespace || info.PodName != tt.wantPodName {
				t.Errorf("environment identity = %q/%q, want %q/%q",
					info.Namespace, info.PodName, tt.wantNamespace, tt.wantPodName)
			}
			if info.ContainerName != tt.wantContainer {
				t.Errorf("ContainerName = %q, want %q", info.ContainerName, tt.wantContainer)
			}
			if info.HostPID != 4242 || info.GuestPID != tt.wantGuestPID {
				t.Errorf("PIDs = host %d guest %d, want host 4242 guest %d", info.HostPID, info.GuestPID, tt.wantGuestPID)
			}
		})
	}
}

func TestProcessMappingFailsForMissingProcess(t *testing.T) {
	if _, err := getWorkerInfoFromProcDir(filepath.Join(t.TempDir(), "gone"), 4242); err == nil {
		t.Fatal("expected an error for a process that does not exist")
	}
}
