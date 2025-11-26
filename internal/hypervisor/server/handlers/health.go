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

// HealthHandler handles health check endpoints
type HealthHandler struct{}

// NewHealthHandler creates a new health handler
func NewHealthHandler() *HealthHandler {
	return &HealthHandler{}
}

// HandleHealthz handles GET /healthz
func (h *HealthHandler) HandleHealthz(c *gin.Context) {
	c.JSON(http.StatusOK, api.StatusResponse{Status: "ok"})
}

// HandleReadyz handles GET /readyz
func (h *HealthHandler) HandleReadyz(c *gin.Context, deviceController framework.DeviceController, workerController framework.WorkerController) {
	if deviceController == nil || workerController == nil {
		c.JSON(http.StatusServiceUnavailable, api.StatusResponse{Status: "not ready"})
		return
	}
	c.JSON(http.StatusOK, api.StatusResponse{Status: "ready"})
}
