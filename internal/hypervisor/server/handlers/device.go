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

// DeviceHandler handles device-related endpoints
type DeviceHandler struct {
	deviceController framework.DeviceController
}

// NewDeviceHandler creates a new device handler
func NewDeviceHandler(deviceController framework.DeviceController) *DeviceHandler {
	return &DeviceHandler{
		deviceController: deviceController,
	}
}

// HandleGetDevices handles GET /api/v1/devices
func (h *DeviceHandler) HandleGetDevices(c *gin.Context) {
	devices, err := h.deviceController.ListDevices(c.Request.Context())
	if err != nil {
		c.JSON(http.StatusInternalServerError, api.ErrorResponse{Error: err.Error()})
		return
	}
	c.JSON(http.StatusOK, api.ListDevicesResponse{Devices: devices})
}

// HandleGetDevice handles GET /api/v1/devices/:uuid
func (h *DeviceHandler) HandleGetDevice(c *gin.Context) {
	uuid := c.Param("uuid")
	device, err := h.deviceController.GetDevice(c.Request.Context(), uuid)
	if err != nil {
		c.JSON(http.StatusNotFound, api.ErrorResponse{Error: err.Error()})
		return
	}
	c.JSON(http.StatusOK, api.GetDeviceResponse{DeviceInfo: device})
}

// HandleDiscoverDevices handles POST /api/v1/devices/discover
func (h *DeviceHandler) HandleDiscoverDevices(c *gin.Context) {
	if err := h.deviceController.DiscoverDevices(); err != nil {
		c.JSON(http.StatusInternalServerError, api.ErrorResponse{Error: err.Error()})
		return
	}
	c.JSON(http.StatusOK, api.DiscoverDevicesResponse{Message: "device discovery triggered"})
}

