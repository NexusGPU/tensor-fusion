package kubernetes

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
)

// GetWorkerInfoFromHostPID extracts worker information from a process's environment
// by reading /proc/{hostPID}/environ and /proc/{hostPID}/status
// workerUID (podUID) is provided as input parameter, not extracted from environment
func GetWorkerInfoFromHostPID(hostPID uint32, workerUID string) (*framework.ProcessMappingInfo, error) {
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
	envPairs := strings.Split(string(envData), "\x00")
	for _, pair := range envPairs {
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

	// Validate required fields (must exist as they are injected by webhook)
	if podName == "" {
		return nil, fmt.Errorf("POD_NAME not found in environment for process %d", hostPID)
	}
	if namespace == "" {
		return nil, fmt.Errorf("POD_NAMESPACE not found in environment for process %d", hostPID)
	}
	if containerName == "" {
		return nil, fmt.Errorf("CONTAINER_NAME not found in environment for process %d", hostPID)
	}

	return &framework.ProcessMappingInfo{
		GuestID:  fmt.Sprintf("%s_%s_%s", namespace, podName, containerName),
		HostPID:  hostPID,
		GuestPID: containerPID,
	}, nil
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
