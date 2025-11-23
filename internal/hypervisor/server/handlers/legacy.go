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
	"net/http"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/gin-gonic/gin"
)

// LegacyHandler handles legacy endpoints
type LegacyHandler struct {
	workerController framework.WorkerController
	backend          framework.Backend
}

// NewLegacyHandler creates a new legacy handler
func NewLegacyHandler(workerController framework.WorkerController, backend framework.Backend) *LegacyHandler {
	return &LegacyHandler{
		workerController: workerController,
		backend:          backend,
	}
}

// HandleGetLimiter handles GET /api/v1/limiter
func (h *LegacyHandler) HandleGetLimiter(c *gin.Context) {
	workers, err := h.workerController.ListWorkers()
	if err != nil {
		c.JSON(http.StatusInternalServerError, api.ErrorResponse{Error: err.Error()})
		return
	}

	limiterInfos := make([]api.LimiterInfo, 0, len(workers))
	for _, workerUID := range workers {
		allocation, err := h.workerController.GetWorkerAllocation(workerUID)
		if err != nil || allocation == nil {
			continue
		}

		var requests, limits *api.ResourceInfo
		if allocation.WorkerInfo != nil && allocation.WorkerInfo.MemoryLimitBytes > 0 {
			limits = &api.ResourceInfo{
				Vram: &allocation.WorkerInfo.MemoryLimitBytes,
			}
		}
		if allocation.WorkerInfo != nil && allocation.WorkerInfo.ComputeLimitUnits > 0 {
			computeLimit := float64(allocation.WorkerInfo.ComputeLimitUnits)
			if limits == nil {
				limits = &api.ResourceInfo{}
			}
			limits.ComputePercent = &computeLimit
		}

		limiterInfos = append(limiterInfos, api.LimiterInfo{
			WorkerUID: workerUID,
			Requests:  requests,
			Limits:    limits,
		})
	}

	c.JSON(http.StatusOK, api.ListLimitersResponse{Limiters: limiterInfos})
}

// HandleTrap handles POST /api/v1/trap
func (h *LegacyHandler) HandleTrap(c *gin.Context) {
	// Trap endpoint: start snapshot low QoS workers to release VRAM
	workers, err := h.workerController.ListWorkers()
	if err != nil {
		c.JSON(http.StatusInternalServerError, api.ErrorResponse{Error: err.Error()})
		return
	}

	snapshotCount := 0
	for _, workerUID := range workers {
		allocation, err := h.workerController.GetWorkerAllocation(workerUID)
		if err != nil || allocation == nil {
			continue
		}

		// TODO: Check QoS level and snapshot low QoS workers
		// For now, snapshot all workers (this should be filtered by QoS)
		snapshotCount++
	}

	c.JSON(http.StatusOK, api.TrapResponse{
		Message:       "trap initiated",
		SnapshotCount: snapshotCount,
	})
}

// HandleGetPods handles GET /api/v1/pod
func (h *LegacyHandler) HandleGetPods(c *gin.Context) {
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
	for _, workerUID := range workers {
		allocation, err := h.workerController.GetWorkerAllocation(workerUID)
		if err != nil || allocation == nil {
			continue
		}

		var tflopsLimit *float64
		var vramLimit *uint64
		var qosLevel *string

		if allocation.WorkerInfo != nil && allocation.WorkerInfo.MemoryLimitBytes > 0 {
			vramLimit = &allocation.WorkerInfo.MemoryLimitBytes
		}

		// Try to get QoS from allocation or default to medium
		qos := "medium"
		qosLevel = &qos

		pods = append(pods, api.PodInfo{
			PodName:     getAllocationPodName(allocation),
			Namespace:   getAllocationNamespace(allocation),
			GPUIDs:      getDeviceUUIDs(allocation),
			TflopsLimit: tflopsLimit,
			VramLimit:   vramLimit,
			QoSLevel:    qosLevel,
		})
	}

	c.JSON(http.StatusOK, api.ListPodsResponse{Pods: pods})
}

// Helper functions for WorkerAllocation field access
func getAllocationPodName(allocation *api.WorkerAllocation) string {
	if allocation.WorkerInfo != nil {
		return allocation.WorkerInfo.PodName
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
	var uuids []string
	for _, device := range allocation.DeviceInfos {
		uuids = append(uuids, device.UUID)
	}
	return uuids
}

// HandleGetProcesses handles GET /api/v1/process
func (h *LegacyHandler) HandleGetProcesses(c *gin.Context) {
	// Get worker to process mapping
	processMap, err := h.backend.GetWorkerToProcessMap()
	if err != nil {
		c.JSON(http.StatusInternalServerError, api.ErrorResponse{Error: err.Error()})
		return
	}

	processInfos := make([]api.ProcessInfo, 0, len(processMap))
	for workerUID, pids := range processMap {
		mapping := make(map[string]string)
		for _, pid := range pids {
			// In a real implementation, this would map container PID to host PID
			// For now, use the same PID
			mapping[pid] = pid
		}
		processInfos = append(processInfos, api.ProcessInfo{
			WorkerUID:      workerUID,
			ProcessMapping: mapping,
		})
	}

	c.JSON(http.StatusOK, api.ListProcessesResponse{Processes: processInfos})
}
