package kubernetes

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	"k8s.io/klog/v2"
)

// podUIDCgroupRe matches the Kubernetes pod UID embedded in a cgroup path, on
// cgroup v1 and v2 alike. Handles both the systemd driver (underscores, e.g.
// "kubepods-burstable-pode38c7b11_d82c_4169_b866_361c5d7103af.slice") and the
// cgroupfs driver (dashes, e.g. "kubepods/burstable/pode38c7b11-...-..."), as
// well as the hash a static pod uses in place of a UUID. The match has to be a
// whole path component (or its systemd slice prefix).
var podUIDCgroupRe = regexp.MustCompile(
	`[/-]pod([0-9a-fA-F]{8}[-_][0-9a-fA-F]{4}[-_]` +
		`[0-9a-fA-F]{4}[-_][0-9a-fA-F]{4}[-_][0-9a-fA-F]{12}|[0-9a-fA-F]{32})(?:[./\n]|$)`,
)

// GetWorkerInfoFromHostPID extracts worker information for a host process from
// /proc/{hostPID}: the pod UID from its cgroup, and the pod/container names it
// carries in its environment.
// Returns ProcessMappingInfo with PodUID, Namespace, PodName, ContainerName for worker lookup
func GetWorkerInfoFromHostPID(hostPID uint32) (*framework.ProcessMappingInfo, error) {
	return getWorkerInfoFromProcDir(fmt.Sprintf("/proc/%d", hostPID), hostPID)
}

func getWorkerInfoFromProcDir(procDir string, hostPID uint32) (*framework.ProcessMappingInfo, error) {
	// Check if process exists
	if _, err := os.Stat(procDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("process %d does not exist", hostPID)
	}

	// Read environment variables from /proc/{pid}/environ. The environment is
	// only descriptive (the cgroup below identifies the pod) and is not always
	// readable, so a failure here must not hide the process.
	envPath := filepath.Join(procDir, "environ")
	envData, err := os.ReadFile(envPath)
	if err != nil {
		klog.V(5).Infof("Failed to read environment from %s: %v", envPath, err)
	}

	// Parse environment variables (null-separated)
	envMap := make(map[string]string)
	envPairs := strings.SplitSeq(string(envData), "\x00")
	for pair := range envPairs {
		if pair == "" {
			continue
		}
		parts := strings.SplitN(pair, "=", 2)
		if len(parts) == 2 {
			envMap[parts[0]] = parts[1]
		}
	}

	// Extract Kubernetes pod information from environment (injected by webhook)
	podName := envMap[constants.PodNameEnv]
	namespace := envMap[constants.PodNamespaceEnv]
	containerName := envMap[constants.ContainerNameEnv]

	// Read container PID (namespaced PID) from /proc/{pid}/status
	containerPID, err := getContainerPIDFromStatus(procDir)
	if err != nil {
		// If we can't get container PID, use host PID as fallback
		containerPID = hostPID
	}

	// Parse the pod UID from the cgroup. This is the identity callers rely on:
	// the cgroup is set by the kubelet and inherited by every child, so unlike
	// the environment it neither depends on how the process was started nor
	// gets lost when a workload spawns the process that actually holds device
	// memory with a stripped environment (e.g. vLLM's EngineCore subprocess
	// drops POD_NAME/POD_NAMESPACE).
	//
	// A cgroup that names no pod is a statement too (the process runs outside
	// every pod), and only then may the environment say which worker it is. A
	// cgroup that could not be read says nothing, and is reported as such.
	podUID, cgroupRead := readPodUIDFromCgroup(procDir)

	return &framework.ProcessMappingInfo{
		Namespace:     namespace,
		PodName:       podName,
		ContainerName: containerName,
		GuestID:       fmt.Sprintf("%s_%s_%s", namespace, podName, containerName),
		PodUID:        podUID,
		PodUIDUnknown: !cgroupRead,
		HostPID:       hostPID,
		GuestPID:      containerPID,
	}, nil
}

// getPodUIDFromCgroup reads /proc/<pid>/cgroup and extracts the Kubernetes pod
// UID, normalized to the canonical dash-separated form (matching pod.UID, i.e.
// WorkerInfo.WorkerUID). Returns "" when no pod UID can be found.
//
// The leftmost match is the pod-level cgroup created by the kubelet. Anything
// further down the path can be created from inside the container and is
// ignored.
func getPodUIDFromCgroup(procDir string) string {
	podUID, _ := readPodUIDFromCgroup(procDir)
	return podUID
}

// readPodUIDFromCgroup is getPodUIDFromCgroup that also reports whether the
// cgroup could be read at all. An empty UID with read set means the process is
// in no pod; with read unset its pod is not known.
func readPodUIDFromCgroup(procDir string) (podUID string, read bool) {
	data, err := os.ReadFile(filepath.Join(procDir, "cgroup"))
	if err != nil {
		klog.V(5).Infof("Failed to read cgroup from %s: %v", procDir, err)
		return "", false
	}
	m := podUIDCgroupRe.FindSubmatch(data)
	if m == nil {
		return "", true
	}
	return strings.ReplaceAll(string(m[1]), "_", "-"), true
}

// getContainerPIDFromStatus reads the container PID (NSpid) from /proc/{pid}/status
func getContainerPIDFromStatus(procDir string) (uint32, error) {
	statusPath := filepath.Join(procDir, "status")
	file, err := os.Open(statusPath)
	if err != nil {
		return 0, fmt.Errorf("failed to open status file: %w", err)
	}
	defer func() {
		_ = file.Close()
	}()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "NSpid:") {
			// NSpid format: "NSpid:	1234	5678" (host PID, then container PID)
			// or "NSpid:	1234" (if same namespace)
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				// The last field is typically the container PID
				// If there are multiple PIDs, the last one is in the innermost namespace
				pidStr := fields[len(fields)-1]
				pid, err := strconv.ParseUint(pidStr, 10, 32)
				if err != nil {
					return 0, fmt.Errorf("failed to parse container PID: %w", err)
				}
				return uint32(pid), nil
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return 0, fmt.Errorf("failed to read status file: %w", err)
	}

	return 0, fmt.Errorf("NSpid not found in status file")
}
