/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package handlers

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/framework"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/gin-gonic/gin"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

// LegacyHandler handles legacy endpoints
type LegacyHandler struct {
	workerController     framework.WorkerController
	allocationController framework.WorkerAllocationController
	backend              framework.Backend
	deviceController     framework.DeviceController
	listHostPIDsFunc     func() ([]uint32, error)
	processMappingFunc   func(hostPID uint32) (*framework.ProcessMappingInfo, error)
	shmBasePath          string
}

// NewLegacyHandler creates a new legacy handler
func NewLegacyHandler(
	workerController framework.WorkerController,
	allocationController framework.WorkerAllocationController,
	backend framework.Backend,
	deviceController framework.DeviceController,
) *LegacyHandler {
	handler := &LegacyHandler{
		workerController:     workerController,
		allocationController: allocationController,
		backend:              backend,
		deviceController:     deviceController,
		shmBasePath: filepath.Join(
			constants.TFDataPath,
			strings.TrimPrefix(constants.SharedMemMountSubPath, "/"),
		),
	}
	handler.listHostPIDsFunc = defaultListHostPIDs
	handler.processMappingFunc = func(hostPID uint32) (*framework.ProcessMappingInfo, error) {
		if handler.backend == nil {
			return nil, fmt.Errorf("kubernetes backend not enabled")
		}
		return handler.backend.GetProcessMappingInfo(hostPID)
	}
	return handler
}

// HandleGetLimiter handles GET /api/v1/limiter
func (h *LegacyHandler) HandleGetLimiter(c *gin.Context) {
	workers, err := h.workerController.ListWorkers()
	if err != nil {
		c.JSON(http.StatusInternalServerError, api.ErrorResponse{Error: err.Error()})
		return
	}

	limiterInfos := make([]api.LimiterInfo, 0, len(workers))
	for _, worker := range workers {
		allocation, exists := h.allocationController.GetWorkerAllocation(worker.WorkerUID)
		if !exists || allocation == nil {
			continue
		}

		var requests, limits *tfv1.Resource
		if allocation.WorkerInfo != nil {
			requests = &allocation.WorkerInfo.Requests
			limits = &allocation.WorkerInfo.Limits
		}

		limiterInfos = append(limiterInfos, api.LimiterInfo{
			WorkerUID: worker.WorkerUID,
			Requests:  requests,
			Limits:    limits,
		})
	}

	c.JSON(http.StatusOK, api.ListLimitersResponse{Limiters: limiterInfos})
}

// HandleTrap handles POST /api/v1/trap
// When VRAM pressure is detected, this endpoint identifies low-QoS workers
// that can be snapshotted to release VRAM for higher-priority workloads.
func (h *LegacyHandler) HandleTrap(c *gin.Context) {
	workers, err := h.workerController.ListWorkers()
	if err != nil {
		c.JSON(http.StatusInternalServerError, api.ErrorResponse{Error: err.Error()})
		return
	}

	snapshotCount := 0
	for _, worker := range workers {
		allocation, exists := h.allocationController.GetWorkerAllocation(worker.WorkerUID)
		if !exists || allocation == nil {
			continue
		}

		// Only snapshot low QoS workers to release VRAM for higher priority workloads
		if worker.QoS == tfv1.QoSLow || worker.QoS == tfv1.QoSMedium {
			snapshotCount++
			klog.V(2).Infof("VRAM trap: worker %s (QoS=%s) selected for snapshot", worker.WorkerUID, worker.QoS)
		}
	}

	c.JSON(http.StatusOK, api.TrapResponse{
		Message:       "trap initiated",
		SnapshotCount: snapshotCount,
	})
}

// HandleGetPods handles GET /api/v1/pod
func (h *LegacyHandler) HandleGetPods(c *gin.Context) {
	if isModernPodInfoRequest(c) {
		h.handleGetPodInfo(c)
		return
	}

	// Only available when k8s backend is enabled
	if h.backend == nil {
		c.JSON(http.StatusServiceUnavailable, api.ErrorResponse{Error: "kubernetes backend not enabled"})
		return
	}

	workers, err := h.workerController.ListWorkers()
	if err != nil {
		c.JSON(http.StatusInternalServerError, api.ErrorResponse{Error: err.Error()})
		return
	}

	pods := make([]api.PodInfo, 0)
	for _, worker := range workers {
		allocation, exists := h.allocationController.GetWorkerAllocation(worker.WorkerUID)
		if !exists || allocation == nil {
			continue
		}

		var vramLimit *uint64
		var tflopsLimit *float64
		if allocation.WorkerInfo != nil {
			if allocation.WorkerInfo.Limits.Vram.Value() > 0 {
				vramLimit = ptr.To(uint64(allocation.WorkerInfo.Limits.Vram.Value()))
			}
			if allocation.WorkerInfo.Limits.Tflops.Value() > 0 {
				tflopsLimit = ptr.To(allocation.WorkerInfo.Limits.Tflops.AsApproximateFloat64())
			}
		}
		pods = append(pods, api.PodInfo{
			PodName:     getAllocationPodName(allocation),
			Namespace:   getAllocationNamespace(allocation),
			GPUIDs:      getDeviceUUIDs(allocation),
			TflopsLimit: tflopsLimit,
			VramLimit:   vramLimit,
			QoSLevel:    allocation.WorkerInfo.QoS,
		})
	}

	c.JSON(http.StatusOK, api.ListPodsResponse{Pods: pods})
}

// HandleInitProcess handles POST /api/v1/process.
func (h *LegacyHandler) HandleInitProcess(c *gin.Context) {
	var query processInitQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		c.JSON(http.StatusBadRequest, api.ErrorResponse{Error: err.Error()})
		return
	}

	token, err := extractBearerToken(c)
	if err != nil {
		c.JSON(http.StatusUnauthorized, api.ErrorResponse{Error: err.Error()})
		return
	}

	payload, err := extractJWTPayload(token)
	if err != nil {
		c.JSON(http.StatusUnauthorized, api.ErrorResponse{Error: err.Error()})
		return
	}

	namespace := payload.Kubernetes.Namespace
	podName := payload.Kubernetes.Pod.Name
	allocation, found, err := h.findWorkerAllocation(namespace, podName)
	if err != nil {
		c.JSON(http.StatusOK, api.ProcessInitResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to find pod in registry: %v", err),
		})
		return
	}
	if !found {
		c.JSON(http.StatusOK, api.ProcessInitResponse{
			Success: false,
			Message: fmt.Sprintf("Pod %s not found in namespace %s", podName, namespace),
		})
		return
	}

	if err := h.ensureWorkerSharedMemory(namespace, podName, allocation); err != nil {
		c.JSON(http.StatusOK, api.ProcessInitResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to prepare shared memory: %v", err),
		})
		return
	}

	hostPID, err := h.findHostPID(namespace, podName, query.ContainerName, query.ContainerPID)
	if err != nil {
		c.JSON(http.StatusOK, api.ProcessInitResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to initialize process: %v", err),
		})
		return
	}

	// Register host PID in shared memory so cuda-limiter can track active processes
	if err := h.registerPIDInSharedMemory(namespace, podName, hostPID); err != nil {
		klog.Warningf("Failed to register PID %d in shared memory for %s/%s: %v",
			hostPID, namespace, podName, err)
	}

	c.JSON(http.StatusOK, api.ProcessInitResponse{
		Success: true,
		Data: &api.ProcessInitInfo{
			HostPID:       hostPID,
			ContainerPID:  query.ContainerPID,
			ContainerName: query.ContainerName,
			PodName:       podName,
			Namespace:     namespace,
		},
		Message: "Process initialized successfully",
	})
}

// Helper functions for WorkerAllocation field access
func getAllocationPodName(allocation *api.WorkerAllocation) string {
	if allocation.WorkerInfo != nil {
		return allocation.WorkerInfo.WorkerName
	}
	return ""
}

func getAllocationNamespace(allocation *api.WorkerAllocation) string {
	if allocation.WorkerInfo != nil {
		return allocation.WorkerInfo.Namespace
	}
	return ""
}

func getDeviceUUIDs(allocation *api.WorkerAllocation) []string {
	uuids := make([]string, 0, len(allocation.DeviceInfos))
	for _, device := range allocation.DeviceInfos {
		uuids = append(uuids, device.UUID)
	}
	if len(uuids) == 0 && allocation.WorkerInfo != nil {
		uuids = append(uuids, allocation.WorkerInfo.AllocatedDevices...)
	}
	return uuids
}

type podInfoQuery struct {
	ContainerName string `form:"container_name"`
}

type processInitQuery struct {
	ContainerName string `form:"container_name" binding:"required"`
	ContainerPID  uint32 `form:"container_pid" binding:"required"`
}

type jwtPayload struct {
	Kubernetes kubernetesInfo `json:"kubernetes.io"`
}

type kubernetesInfo struct {
	Namespace string        `json:"namespace"`
	Pod       kubernetesPod `json:"pod"`
}

type kubernetesPod struct {
	Name string `json:"name"`
}

func isModernPodInfoRequest(c *gin.Context) bool {
	if c.GetHeader(constants.AuthorizationHeader) != "" {
		return true
	}
	return c.Query("container_name") != ""
}

func (h *LegacyHandler) handleGetPodInfo(c *gin.Context) {
	var query podInfoQuery
	if err := c.ShouldBindQuery(&query); err != nil {
		c.JSON(http.StatusBadRequest, api.ErrorResponse{Error: err.Error()})
		return
	}

	token, err := extractBearerToken(c)
	if err != nil {
		c.JSON(http.StatusUnauthorized, api.ErrorResponse{Error: err.Error()})
		return
	}

	payload, err := extractJWTPayload(token)
	if err != nil {
		c.JSON(http.StatusUnauthorized, api.ErrorResponse{Error: err.Error()})
		return
	}

	allocation, found, err := h.findWorkerAllocation(payload.Kubernetes.Namespace, payload.Kubernetes.Pod.Name)
	if err != nil {
		c.JSON(http.StatusOK, api.PodInfoResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to find pod in registry: %v", err),
		})
		return
	}
	if !found {
		c.JSON(http.StatusOK, api.PodInfoResponse{
			Success: false,
			Message: fmt.Sprintf(
				"Pod %s not found in namespace %s",
				payload.Kubernetes.Pod.Name,
				payload.Kubernetes.Namespace,
			),
		})
		return
	}

	if err := h.ensureWorkerSharedMemory(
		payload.Kubernetes.Namespace,
		payload.Kubernetes.Pod.Name,
		allocation,
	); err != nil {
		c.JSON(http.StatusOK, api.PodInfoResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to prepare shared memory: %v", err),
		})
		return
	}

	c.JSON(http.StatusOK, api.PodInfoResponse{
		Success: true,
		Data: &api.RemotePodInfo{
			PodName:      getAllocationPodName(allocation),
			Namespace:    getAllocationNamespace(allocation),
			GPUIDs:       getContainerDeviceUUIDs(allocation, query.ContainerName),
			TflopsLimit:  getAllocationTflopsLimit(allocation),
			VramLimit:    getAllocationVramLimit(allocation),
			QoSLevel:     getPascalCaseQoSLevel(allocation),
			ComputeShard: false,
			Isolation:    getAllocationIsolation(allocation),
		},
		Message: fmt.Sprintf("Pod %s information retrieved successfully", payload.Kubernetes.Pod.Name),
	})
}

func extractBearerToken(c *gin.Context) (string, error) {
	token := strings.TrimSpace(c.GetHeader(constants.AuthorizationHeader))
	if token == "" {
		return "", fmt.Errorf("missing authorization header")
	}
	return strings.TrimPrefix(token, "Bearer "), nil
}

func extractJWTPayload(token string) (*jwtPayload, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, fmt.Errorf("invalid JWT format")
	}

	payloadBytes, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, fmt.Errorf("failed to decode JWT payload: %w", err)
	}

	var payload jwtPayload
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return nil, fmt.Errorf("failed to parse JWT payload: %w", err)
	}
	if payload.Kubernetes.Namespace == "" || payload.Kubernetes.Pod.Name == "" {
		return nil, fmt.Errorf("JWT payload missing kubernetes namespace or pod name")
	}
	return &payload, nil
}

func (h *LegacyHandler) findWorkerAllocation(namespace, podName string) (*api.WorkerAllocation, bool, error) {
	workers, err := h.workerController.ListWorkers()
	if err != nil {
		return nil, false, err
	}

	for _, worker := range workers {
		if worker == nil {
			continue
		}
		if worker.Namespace != namespace || worker.WorkerName != podName {
			continue
		}
		allocation, exists := h.allocationController.GetWorkerAllocation(worker.WorkerUID)
		if !exists || allocation == nil {
			continue
		}
		return allocation, true, nil
	}

	return nil, false, nil
}

func (h *LegacyHandler) findHostPID(namespace, podName, containerName string, containerPID uint32) (uint32, error) {
	if h.processMappingFunc == nil {
		return 0, fmt.Errorf("kubernetes backend not enabled")
	}

	pids, err := h.listHostPIDs()
	if err != nil {
		return 0, err
	}

	for _, hostPID := range pids {
		mappingInfo, err := h.processMappingFunc(hostPID)
		if err != nil || mappingInfo == nil {
			continue
		}
		if mappingInfo.Namespace != namespace || mappingInfo.PodName != podName {
			continue
		}
		if mappingInfo.ContainerName != containerName || mappingInfo.GuestPID != containerPID {
			continue
		}
		return mappingInfo.HostPID, nil
	}

	return 0, fmt.Errorf(
		"process not found for pod %s/%s container %s pid %d",
		namespace,
		podName,
		containerName,
		containerPID,
	)
}

func (h *LegacyHandler) listHostPIDs() ([]uint32, error) {
	if h.listHostPIDsFunc == nil {
		return nil, fmt.Errorf("host PID lister is not configured")
	}
	return h.listHostPIDsFunc()
}

func defaultListHostPIDs() ([]uint32, error) {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return nil, fmt.Errorf("failed to read /proc: %w", err)
	}

	pids := make([]uint32, 0, len(entries))
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		pid, err := strconv.ParseUint(entry.Name(), 10, 32)
		if err != nil {
			continue
		}
		pids = append(pids, uint32(pid))
	}
	sort.Slice(pids, func(i, j int) bool {
		return pids[i] < pids[j]
	})
	return pids, nil
}

func getContainerDeviceUUIDs(allocation *api.WorkerAllocation, containerName string) []string {
	if containerName != "" && allocation.WorkerInfo != nil && allocation.WorkerInfo.Annotations != nil {
		if rawMapping, exists := allocation.WorkerInfo.Annotations[constants.ContainerGPUsAnnotation]; exists &&
			rawMapping != "" {
			var containerMapping map[string][]string
			if err := json.Unmarshal([]byte(rawMapping), &containerMapping); err == nil {
				if gpuIDs, exists := containerMapping[containerName]; exists {
					return normalizeGPUUUIDs(gpuIDs)
				}
			}
		}
	}
	return normalizeGPUUUIDs(getDeviceUUIDs(allocation))
}

func normalizeGPUUUIDs(deviceUUIDs []string) []string {
	normalized := make([]string, 0, len(deviceUUIDs))
	for _, deviceUUID := range deviceUUIDs {
		normalized = append(normalized, normalizeGPUUUID(deviceUUID))
	}
	return normalized
}

func normalizeGPUUUID(deviceUUID string) string {
	if strings.HasPrefix(deviceUUID, "gpu-") {
		return "GPU-" + strings.TrimPrefix(deviceUUID, "gpu-")
	}
	return deviceUUID
}

func getAllocationVramLimit(allocation *api.WorkerAllocation) *uint64 {
	if allocation.WorkerInfo == nil || allocation.WorkerInfo.Limits.Vram.Value() <= 0 {
		return nil
	}
	return ptr.To(uint64(allocation.WorkerInfo.Limits.Vram.Value()))
}

func getAllocationTflopsLimit(allocation *api.WorkerAllocation) *float64 {
	if allocation.WorkerInfo == nil || allocation.WorkerInfo.Limits.Tflops.Value() <= 0 {
		return nil
	}
	return ptr.To(allocation.WorkerInfo.Limits.Tflops.AsApproximateFloat64())
}

func getPascalCaseQoSLevel(allocation *api.WorkerAllocation) *string {
	if allocation.WorkerInfo == nil || allocation.WorkerInfo.QoS == "" {
		return nil
	}
	qosLevel := string(allocation.WorkerInfo.QoS)
	qosLevel = strings.ToLower(qosLevel)
	qosLevel = strings.ToUpper(qosLevel[:1]) + qosLevel[1:]
	return ptr.To(qosLevel)
}

func getAllocationIsolation(allocation *api.WorkerAllocation) string {
	if allocation.WorkerInfo == nil {
		return ""
	}
	return string(allocation.WorkerInfo.IsolationMode)
}

func (h *LegacyHandler) registerPIDInSharedMemory(namespace, podName string, hostPID uint32) error {
	podId := workerstate.NewPodIdentifier(namespace, podName)
	handle, err := workerstate.OpenSharedMemoryHandle(h.shmBasePath, podId)
	if err != nil {
		return fmt.Errorf("failed to open shm for PID registration: %w", err)
	}
	// Each /process init opens a fresh handle (mmap + fd). Without Close()
	// the mapping and file descriptor leak — over a churning workload the
	// hypervisor process accumulates one mmap region per pod-init.
	defer func() {
		_ = handle.Close()
	}()
	state := handle.GetState()
	if state != nil {
		state.AddPID(int(hostPID))
	}
	return nil
}

func (h *LegacyHandler) ensureWorkerSharedMemory(namespace, podName string, allocation *api.WorkerAllocation) error {
	if allocation == nil || allocation.WorkerInfo == nil {
		return nil
	}

	configs := buildWorkerDeviceConfigs(allocation)
	if len(configs) == 0 {
		return nil
	}

	podId := workerstate.NewPodIdentifier(namespace, podName)
	// /process init can fire many times for a single pod (one per container
	// and each retry). CreateSharedMemoryHandle truncates the underlying file
	// (O_TRUNC), which would zero out live ERL state — used tokens, mem
	// counters — every call. Open the existing shm if it is already there
	// and only fall back to Create on the FIRST init (file does not exist
	// yet). Any other Open error — legacy layout, size mismatch, discriminant
	// mismatch, mmap failure, permission denied, transient I/O — must NOT
	// trigger Create, otherwise we would silently O_TRUNC a healthy shm and
	// destroy live state. Surface the error instead.
	handle, err := workerstate.OpenSharedMemoryHandle(h.shmBasePath, podId)
	if err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("failed to open existing shm (refusing to recreate over a non-missing file): %w", err)
		}
		handle, err = workerstate.CreateSharedMemoryHandle(h.shmBasePath, podId, configs)
		if err != nil {
			return err
		}
	}
	return handle.Close()
}

func buildWorkerDeviceConfigs(allocation *api.WorkerAllocation) []workerstate.DeviceConfig {
	if allocation == nil || allocation.WorkerInfo == nil {
		return nil
	}
	configs := make([]workerstate.DeviceConfig, 0, len(allocation.DeviceInfos))
	for _, deviceInfo := range allocation.DeviceInfos {
		if deviceInfo == nil {
			continue
		}
		memLimit := deviceInfo.TotalMemoryBytes
		if allocation.WorkerInfo.Limits.Vram.Value() > 0 {
			memLimit = uint64(allocation.WorkerInfo.Limits.Vram.Value())
		}
		smCount := parseUint32Property(deviceInfo.Properties, "totalComputeUnits")
		computeCapability := ""
		if deviceInfo.Properties != nil {
			computeCapability = deviceInfo.Properties["computeCapability"]
		}
		configs = append(configs, workerstate.DeviceConfig{
			DeviceIdx:  uint32(deviceInfo.Index),
			DeviceUUID: normalizeGPUUUID(deviceInfo.UUID),
			UpLimit:    computeLimitPercent(allocation.WorkerInfo, deviceInfo),
			MemLimit:   memLimit,
			SMCount:    smCount * coresPerSM(computeCapability),
		})
	}
	return configs
}

func computeLimitPercent(workerInfo *api.WorkerInfo, deviceInfo *api.DeviceInfo) uint32 {
	if workerInfo == nil {
		return 100
	}
	if workerInfo.Limits.ComputePercent.Value() > 0 {
		return uint32(workerInfo.Limits.ComputePercent.Value())
	}
	if workerInfo.Limits.Tflops.Value() > 0 && deviceInfo != nil && deviceInfo.MaxTflops > 0 {
		percent := math.Ceil(workerInfo.Limits.Tflops.AsApproximateFloat64() / deviceInfo.MaxTflops * 100.0)
		if percent < 1 {
			return 1
		}
		if percent > 100 {
			return 100
		}
		return uint32(percent)
	}
	return 100
}

func parseUint32Property(properties map[string]string, key string) uint32 {
	if properties == nil {
		return 0
	}
	value := strings.TrimSpace(properties[key])
	if value == "" {
		return 0
	}
	parsed, err := strconv.ParseUint(value, 10, 32)
	if err != nil {
		return 0
	}
	return uint32(parsed)
}

func coresPerSM(computeCapability string) uint32 {
	parts := strings.Split(strings.TrimSpace(computeCapability), ".")
	if len(parts) != 2 {
		return 0
	}

	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0
	}
	minor, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0
	}

	switch (major * 10) + minor {
	case 20:
		return 32
	case 21:
		return 48
	case 30, 32, 35, 37:
		return 192
	case 50, 52, 53:
		return 128
	case 60:
		return 64
	case 61, 62:
		return 128
	case 70, 72, 75, 80:
		return 64
	case 86, 87, 89, 90, 100, 101, 103, 110, 120, 121:
		return 128
	default:
		return 0
	}
}
