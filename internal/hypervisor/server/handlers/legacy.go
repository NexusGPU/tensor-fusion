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

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/gin-gonic/gin"
	"k8s.io/utils/ptr"
)

// LegacyHandler handles legacy endpoints
type LegacyHandler struct {
	workerController     framework.WorkerController
	allocationController framework.WorkerAllocationController
	backend              framework.Backend
}

// NewLegacyHandler creates a new legacy handler
func NewLegacyHandler(workerController framework.WorkerController, allocationController framework.WorkerAllocationController, backend framework.Backend) *LegacyHandler {
	return &LegacyHandler{
		workerController:     workerController,
		allocationController: allocationController,
		backend:              backend,
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
func (h *LegacyHandler) HandleTrap(c *gin.Context) {
	// Trap endpoint: start snapshot low QoS workers to release VRAM
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
	return uuids
}
