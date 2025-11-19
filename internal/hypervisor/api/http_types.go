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

package api

// HTTP API Response Types

// HealthResponse represents health check response
type HealthResponse struct {
	Status string `json:"status"`
}

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error string `json:"error"`
}

// MessageResponse represents a message response
type MessageResponse struct {
	Message string `json:"message"`
}

// ListDevicesResponse represents the response from GET /api/v1/devices
type ListDevicesResponse struct {
	Devices []*DeviceInfo `json:"devices"`
}

// GetDeviceResponse represents the response from GET /api/v1/devices/:uuid
type GetDeviceResponse struct {
	*DeviceInfo
}

// DiscoverDevicesResponse represents the response from POST /api/v1/devices/discover
type DiscoverDevicesResponse struct {
	Message string `json:"message"`
}

// WorkerDetail represents a worker with its allocation
type WorkerDetail struct {
	WorkerUID  string            `json:"worker_uid"`
	Allocation *DeviceAllocation `json:"allocation"`
}

// ListWorkersResponse represents the response from GET /api/v1/workers
type ListWorkersResponse struct {
	Workers []WorkerDetail `json:"workers"`
}

// GetWorkerResponse represents the response from GET /api/v1/workers/:id
type GetWorkerResponse struct {
	WorkerUID  string                                          `json:"worker_uid"`
	Allocation *DeviceAllocation                               `json:"allocation"`
	Metrics    map[string]map[string]map[string]*WorkerMetrics `json:"metrics,omitempty"`
}

// SnapshotWorkerResponse represents the response from POST /api/v1/workers/:id/snapshot
type SnapshotWorkerResponse struct {
	Message  string `json:"message"`
	WorkerID string `json:"worker_id"`
}

// ResumeWorkerResponse represents the response from POST /api/v1/workers/:id/resume
type ResumeWorkerResponse struct {
	Message  string `json:"message"`
	WorkerID string `json:"worker_id"`
}

// ResourceInfo represents resource requests/limits
type ResourceInfo struct {
	Tflops         *float64 `json:"tflops,omitempty"`
	Vram           *uint64  `json:"vram,omitempty"`
	ComputePercent *float64 `json:"compute_percent,omitempty"`
}

// LimiterInfo represents worker limiter information
type LimiterInfo struct {
	WorkerUID string        `json:"worker_uid"`
	Requests  *ResourceInfo `json:"requests,omitempty"`
	Limits    *ResourceInfo `json:"limits,omitempty"`
}

// ListLimitersResponse represents the response from GET /api/v1/limiter
type ListLimitersResponse struct {
	Limiters []LimiterInfo `json:"limiters"`
}

// TrapResponse represents the response from POST /api/v1/trap
type TrapResponse struct {
	Message       string `json:"message"`
	SnapshotCount int    `json:"snapshot_count"`
}

// PodInfo represents pod information for the /api/v1/pod endpoint
type PodInfo struct {
	PodName     string   `json:"pod_name"`
	Namespace   string   `json:"namespace"`
	GPUIDs      []string `json:"gpu_uuids"`
	TflopsLimit *float64 `json:"tflops_limit,omitempty"`
	VramLimit   *uint64  `json:"vram_limit,omitempty"`
	QoSLevel    *string  `json:"qos_level,omitempty"`
}

// ListPodsResponse represents the response from GET /api/v1/pod
type ListPodsResponse struct {
	Pods []PodInfo `json:"pods"`
}

// ProcessInfo represents process mapping information
type ProcessInfo struct {
	WorkerUID      string            `json:"worker_uid"`
	ProcessMapping map[string]string `json:"process_mapping"` // container PID -> host PID
}

// ListProcessesResponse represents the response from GET /api/v1/process
type ListProcessesResponse struct {
	Processes []ProcessInfo `json:"processes"`
}
