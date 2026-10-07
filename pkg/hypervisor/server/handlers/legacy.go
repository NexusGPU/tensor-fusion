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
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
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
	autoFreeze           []tfv1.AutoFreeze
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
	}
	if raw := os.Getenv(constants.HypervisorSchedulingConfigEnv); raw != "" {
		var scheduling tfv1.HypervisorScheduling
		if err := json.Unmarshal([]byte(raw), &scheduling); err != nil {
			klog.Warningf("Failed to parse %s for legacy API compatibility: %v",
				constants.HypervisorSchedulingConfigEnv, err)
		} else {
			handler.autoFreeze = scheduling.AutoFreezeAndResume.AutoFreeze
		}
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

	workerUID, err := h.authenticatePodToken(c)
	if err != nil {
		c.JSON(http.StatusUnauthorized, api.ErrorResponse{Error: err.Error()})
		return
	}

	allocation, found := h.allocationController.GetWorkerAllocation(workerUID)
	if !found || allocation == nil || allocation.WorkerInfo == nil {
		c.JSON(http.StatusOK, api.ProcessInitResponse{
			Success: false,
			Message: "Pod has no GPU allocation on this node",
		})
		return
	}
	namespace := allocation.WorkerInfo.Namespace
	podName := allocation.WorkerInfo.WorkerName

	if err := h.workerController.WithWorkerSharedMemory(workerUID, nil); err != nil {
		c.JSON(http.StatusOK, api.ProcessInitResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to prepare shared memory: %v", err),
		})
		return
	}

	hostPID, err := h.findHostPID(workerUID, namespace, podName, query.ContainerName, query.ContainerPID)
	if err != nil {
		c.JSON(http.StatusOK, api.ProcessInitResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to initialize process: %v", err),
		})
		return
	}

	// Register host PID in shared memory so cuda-limiter can track active processes
	registered := false
	if err := h.workerController.WithWorkerSharedMemory(workerUID, func(state *workerstate.SharedDeviceState) {
		registered = state.TryAddPID(int(hostPID))
	}); err != nil {
		c.JSON(http.StatusOK, api.ProcessInitResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to register process: %v", err),
		})
		return
	}
	if !registered {
		c.JSON(http.StatusOK, api.ProcessInitResponse{Success: false, Message: "Process registration busy; retry"})
		return
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

	workerUID, err := h.authenticatePodToken(c)
	if err != nil {
		c.JSON(http.StatusUnauthorized, api.ErrorResponse{Error: err.Error()})
		return
	}

	allocation, found := h.allocationController.GetWorkerAllocation(workerUID)
	if !found || allocation == nil || allocation.WorkerInfo == nil {
		c.JSON(http.StatusOK, api.PodInfoResponse{
			Success: false,
			Message: "Pod has no GPU allocation on this node",
		})
		return
	}

	if err := h.workerController.WithWorkerSharedMemory(workerUID, nil); err != nil {
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
			AutoFreeze:   getAutoFreezeConfig(allocation, h.autoFreeze),
		},
		Message: fmt.Sprintf("Pod %s information retrieved successfully", allocation.WorkerInfo.WorkerName),
	})
}

// Authenticate through Kubernetes before using any workload identity. JWT
// payloads alone are untrusted, and Pod names can be reused after deletion.
func (h *LegacyHandler) authenticatePodToken(c *gin.Context) (string, error) {
	fields := strings.Fields(c.GetHeader(constants.AuthorizationHeader))
	if len(fields) != 2 || !strings.EqualFold(fields[0], "Bearer") {
		return "", fmt.Errorf("missing Bearer token")
	}
	authenticator, ok := h.backend.(interface {
		AuthenticatePodToken(context.Context, string) (string, error)
	})
	if !ok {
		return "", fmt.Errorf("pod token authentication is unavailable")
	}
	uid, err := authenticator.AuthenticatePodToken(c.Request.Context(), fields[1])
	if err != nil || uid == "" {
		return "", fmt.Errorf("invalid Pod token")
	}
	return uid, nil
}

func (h *LegacyHandler) findHostPID(
	workerUID, namespace, podName, containerName string, containerPID uint32,
) (uint32, error) {
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
		if mappingInfo.PodUID != workerUID {
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

func getAutoFreezeConfig(allocation *api.WorkerAllocation, configs []tfv1.AutoFreeze) *api.AutoFreezeConfig {
	if allocation.WorkerInfo == nil {
		return nil
	}
	for _, config := range configs {
		if config.Qos != allocation.WorkerInfo.QoS {
			continue
		}
		result := &api.AutoFreezeConfig{
			Enable: config.Enable != nil && *config.Enable,
		}
		if config.FreezeToMemTTL != "" {
			result.FreezeToMemTTL = ptr.To(config.FreezeToMemTTL)
		}
		if config.FreezeToDiskTTL != "" {
			result.FreezeToDiskTTL = ptr.To(config.FreezeToDiskTTL)
		}
		return result
	}
	return nil
}
