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
)

// podUIDCgroupRe matches the Kubernetes pod UID embedded in a cgroup path.
// Handles both the systemd driver (underscores, e.g.
// "kubepods-burstable-pode38c7b11_d82c_4169_b866_361c5d7103af.slice") and the
// cgroupfs driver (dashes, e.g. "kubepods/burstable/pode38c7b11-...-...").
var podUIDCgroupRe = regexp.MustCompile(`pod([0-9a-fA-F]{8}[-_][0-9a-fA-F]{4}[-_][0-9a-fA-F]{4}[-_][0-9a-fA-F]{4}[-_][0-9a-fA-F]{12})`)

// GetWorkerInfoFromHostPID extracts worker information from a process's environment
// by reading /proc/{hostPID}/environ and /proc/{hostPID}/status
// Returns ProcessMappingInfo with Namespace, PodName, ContainerName for worker lookup
func GetWorkerInfoFromHostPID(hostPID uint32) (*framework.ProcessMappingInfo, error) {
	procDir := fmt.Sprintf("/proc/%d", hostPID)

	// Check if process exists
	if _, err := os.Stat(procDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("process %d does not exist", hostPID)
	}

	// Read environment variables from /proc/{pid}/environ
	envPath := filepath.Join(procDir, "environ")
	envData, err := os.ReadFile(envPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read environment from %s: %w", envPath, err)
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

	// Parse the pod UID from the cgroup as an environ-independent fallback.
	// Some GPU workloads spawn the process that actually holds device memory
	// with a stripped environment (e.g. vLLM's EngineCore subprocess drops
	// POD_NAME/POD_NAMESPACE), which would otherwise make this process
	// unattributable. The cgroup is set by the kubelet and inherited by every
	// child, so it survives such stripping.
	podUID := getPodUIDFromCgroup(procDir)

	return &framework.ProcessMappingInfo{
		Namespace:     namespace,
		PodName:       podName,
		ContainerName: containerName,
		GuestID:       fmt.Sprintf("%s_%s_%s", namespace, podName, containerName),
		PodUID:        podUID,
		HostPID:       hostPID,
		GuestPID:      containerPID,
	}, nil
}

// getPodUIDFromCgroup reads /proc/<pid>/cgroup and extracts the Kubernetes pod
// UID, normalized to the canonical dash-separated form (matching pod.UID, i.e.
// WorkerInfo.WorkerUID). Returns "" when no pod UID can be found.
func getPodUIDFromCgroup(procDir string) string {
	data, err := os.ReadFile(filepath.Join(procDir, "cgroup"))
	if err != nil {
		return ""
	}
	m := podUIDCgroupRe.FindSubmatch(data)
	if m == nil {
		return ""
	}
	return strings.ReplaceAll(string(m[1]), "_", "-")
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
