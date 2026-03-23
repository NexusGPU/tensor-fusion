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

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

// HTTP API Response Types

// ErrorResponse represents an error response
type ErrorResponse struct {
	Error string `json:"error"`
}

// DataResponse is a generic response wrapper for data-only responses
// +k8s:deepcopy-gen=false
type DataResponse[T any] struct {
	Data T `json:"data"`
}

// MessageAndDataResponse is a generic response wrapper for responses with message and data
type MessageAndDataResponse[T any] struct {
	Message string `json:"message"`
	Data    T      `json:"data"`
}

// StatusResponse represents a simple status response
type StatusResponse struct {
	Status string `json:"status"`
}

// Types to be compatible with legacy APIs

// LimiterInfo represents worker limiter information (used in legacy.go)
type LimiterInfo struct {
	WorkerUID string         `json:"worker_uid"`
	Requests  *tfv1.Resource `json:"requests,omitempty"`
	Limits    *tfv1.Resource `json:"limits,omitempty"`
}

// ListLimitersResponse represents the response from GET /api/v1/limiter (used in legacy.go)
type ListLimitersResponse struct {
	Limiters []LimiterInfo `json:"limiters"`
}

// TrapResponse represents the response from POST /api/v1/trap (used in legacy.go)
type TrapResponse struct {
	Message       string `json:"message"`
	SnapshotCount int    `json:"snapshot_count"`
}

// PodInfo represents pod information for the /api/v1/pod endpoint (used in legacy.go)
type PodInfo struct {
	PodName     string        `json:"pod_name"`
	Namespace   string        `json:"namespace"`
	GPUIDs      []string      `json:"gpu_uuids"`
	TflopsLimit *float64      `json:"tflops_limit,omitempty"`
	VramLimit   *uint64       `json:"vram_limit,omitempty"`
	QoSLevel    tfv1.QoSLevel `json:"qos_level,omitempty"`
}

// ListPodsResponse represents the response from GET /api/v1/pod (used in legacy.go)
type ListPodsResponse struct {
	Pods []PodInfo `json:"pods"`
}

// AutoFreezeConfig represents auto-freeze settings for remote workers.
type AutoFreezeConfig struct {
	FreezeToMemTTL  *string `json:"freeze_to_mem_ttl,omitempty"`
	FreezeToDiskTTL *string `json:"freeze_to_disk_ttl,omitempty"`
	Enable          bool    `json:"enable"`
}

// RemotePodInfo represents the modern pod info payload expected by remote workers.
type RemotePodInfo struct {
	PodName      string            `json:"pod_name"`
	Namespace    string            `json:"namespace"`
	GPUIDs       []string          `json:"gpu_uuids"`
	TflopsLimit  *float64          `json:"tflops_limit,omitempty"`
	VramLimit    *uint64           `json:"vram_limit,omitempty"`
	QoSLevel     *string           `json:"qos_level,omitempty"`
	ComputeShard bool              `json:"compute_shard"`
	Isolation    string            `json:"isolation,omitempty"`
	AutoFreeze   *AutoFreezeConfig `json:"auto_freeze,omitempty"`
}

// PodInfoResponse represents the modern response from GET /api/v1/pod.
type PodInfoResponse struct {
	Success bool           `json:"success"`
	Data    *RemotePodInfo `json:"data"`
	Message string         `json:"message"`
}

// LegacyProcessInfo represents process mapping information for the old list API.
type LegacyProcessInfo struct {
	WorkerUID      string            `json:"worker_uid"`
	ProcessMapping map[string]string `json:"process_mapping"` // container PID -> host PID
}

// ListProcessesResponse represents the response from GET /api/v1/process (used in legacy.go)
type ListProcessesResponse struct {
	Processes []LegacyProcessInfo `json:"processes"`
}

// ProcessInitInfo represents process mapping info for remote worker initialization.
type ProcessInitInfo struct {
	HostPID       uint32 `json:"host_pid"`
	ContainerPID  uint32 `json:"container_pid"`
	ContainerName string `json:"container_name"`
	PodName       string `json:"pod_name"`
	Namespace     string `json:"namespace"`
}

// ProcessInitResponse represents the modern response from POST /api/v1/process.
type ProcessInitResponse struct {
	Success bool             `json:"success"`
	Data    *ProcessInitInfo `json:"data"`
	Message string           `json:"message"`
}
