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
	workerController     framework.WorkerController
	allocationController framework.WorkerAllocationController
}

// NewWorkerHandler creates a new worker handler
func NewWorkerHandler(workerController framework.WorkerController, allocationController framework.WorkerAllocationController) *WorkerHandler {
	return &WorkerHandler{
		workerController:     workerController,
		allocationController: allocationController,
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
	workerDetails := make([]*api.WorkerAllocation, 0, len(workers))
	for _, worker := range workers {
		allocation, exists := h.allocationController.GetWorkerAllocation(worker.WorkerUID)
		if !exists || allocation == nil {
			continue
		}
		workerDetails = append(workerDetails, allocation)
	}

	c.JSON(http.StatusOK, api.DataResponse[[]*api.WorkerAllocation]{Data: workerDetails})
}

// HandleGetWorker handles GET /api/v1/workers/:id
func (h *WorkerHandler) HandleGetWorker(c *gin.Context) {
	workerID := c.Param("id")
	allocation, exists := h.allocationController.GetWorkerAllocation(workerID)
	if !exists || allocation == nil {
		c.JSON(http.StatusNotFound, api.ErrorResponse{Error: "worker not found"})
		return
	}

	// Get worker metrics
	workerMetrics, err := h.workerController.GetWorkerMetrics()
	if err != nil {
		c.JSON(http.StatusInternalServerError, api.ErrorResponse{Error: err.Error()})
		return
	}

	metrics, exists := workerMetrics[workerID]
	if !exists || metrics == nil {
		c.JSON(http.StatusOK, api.DataResponse[map[string]any]{
			Data: map[string]any{
				"worker_uid": workerID,
				"allocation": allocation,
			},
		})
		return
	}
	// TODO
}

// HandleSnapshotWorker handles POST /api/v1/workers/:id/snapshot
func (h *WorkerHandler) HandleSnapshotWorker(c *gin.Context) {
	workerID := c.Param("id")
	// TODO: Implement actual snapshot logic using accelerator interface
	// For now, return success
	c.JSON(http.StatusOK, api.MessageAndDataResponse[string]{
		Message: "worker snapshot initiated",
		Data:    workerID,
	})
}

// HandleResumeWorker handles POST /api/v1/workers/:id/resume
func (h *WorkerHandler) HandleResumeWorker(c *gin.Context) {
	workerID := c.Param("id")
	// TODO: Implement actual resume logic using accelerator interface
	// For now, return success
	c.JSON(http.StatusOK, api.MessageAndDataResponse[string]{
		Message: "worker resume initiated",
		Data:    workerID,
	})
}
