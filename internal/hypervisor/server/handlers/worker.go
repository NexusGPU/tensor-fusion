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

// WorkerHandler handles worker-related endpoints
type WorkerHandler struct {
	workerController framework.WorkerController
}

// NewWorkerHandler creates a new worker handler
func NewWorkerHandler(workerController framework.WorkerController) *WorkerHandler {
	return &WorkerHandler{
		workerController: workerController,
	}
}

// HandleGetWorkers handles GET /api/v1/workers
func (h *WorkerHandler) HandleGetWorkers(c *gin.Context) {
	workers, err := h.workerController.ListWorkers()
	if err != nil {
		c.JSON(http.StatusInternalServerError, api.ErrorResponse{Error: err.Error()})
		return
	}

	// Get worker details
	workerDetails := make([]api.WorkerDetail, 0, len(workers))
	for _, workerUID := range workers {
		allocation, err := h.workerController.GetWorkerAllocation(workerUID)
		if err != nil {
			continue
		}
		workerDetails = append(workerDetails, api.WorkerDetail{
			WorkerUID:  workerUID,
			Allocation: allocation,
		})
	}

	c.JSON(http.StatusOK, api.ListWorkersResponse{Workers: workerDetails})
}

// HandleGetWorker handles GET /api/v1/workers/:id
func (h *WorkerHandler) HandleGetWorker(c *gin.Context) {
	workerID := c.Param("id")
	allocation, err := h.workerController.GetWorkerAllocation(workerID)
	if err != nil {
		c.JSON(http.StatusNotFound, api.ErrorResponse{Error: err.Error()})
		return
	}
	if allocation == nil {
		c.JSON(http.StatusNotFound, api.ErrorResponse{Error: "worker not found"})
		return
	}

	// Get worker metrics
	metrics, err := h.workerController.GetWorkerMetrics()
	if err != nil {
		c.JSON(http.StatusOK, api.GetWorkerResponse{
			WorkerUID:  workerID,
			Allocation: allocation,
		})
		return
	}

	// Filter metrics for this worker
	workerMetrics := make(map[string]map[string]map[string]*api.WorkerMetrics)
	// Get metrics for all devices in the allocation
	for _, device := range allocation.DeviceInfos {
		if allMetrics, exists := metrics[device.UUID]; exists {
			if wm, exists := allMetrics[workerID]; exists {
				if workerMetrics[device.UUID] == nil {
					workerMetrics[device.UUID] = make(map[string]map[string]*api.WorkerMetrics)
				}
				workerMetrics[device.UUID][workerID] = wm
			}
		}
	}

	c.JSON(http.StatusOK, api.GetWorkerResponse{
		WorkerUID:  workerID,
		Allocation: allocation,
		Metrics:    workerMetrics,
	})
}

// HandleSnapshotWorker handles POST /api/v1/workers/:id/snapshot
func (h *WorkerHandler) HandleSnapshotWorker(c *gin.Context) {
	workerID := c.Param("id")
	// TODO: Implement actual snapshot logic using accelerator interface
	// For now, return success
	c.JSON(http.StatusOK, api.SnapshotWorkerResponse{
		Message:  "worker snapshot initiated",
		WorkerID: workerID,
	})
}

// HandleResumeWorker handles POST /api/v1/workers/:id/resume
func (h *WorkerHandler) HandleResumeWorker(c *gin.Context) {
	workerID := c.Param("id")
	// TODO: Implement actual resume logic using accelerator interface
	// For now, return success
	c.JSON(http.StatusOK, api.ResumeWorkerResponse{
		Message:  "worker resume initiated",
		WorkerID: workerID,
	})
}
