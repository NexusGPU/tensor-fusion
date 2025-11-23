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

package tui

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
)

// Client is an HTTP client for fetching data from the hypervisor server
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// NewClient creates a new HTTP client for the hypervisor
func NewClient(host string, port int) *Client {
	return &Client{
		baseURL: fmt.Sprintf("http://%s:%d/api/v1", host, port),
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
	}
}

// doRequest performs an HTTP request and decodes the JSON response
//
//nolint:unparam // method parameter is kept for API consistency, even though it's always "GET"
func (c *Client) doRequest(ctx context.Context, method, path string, result interface{}) error {
	url := fmt.Sprintf("%s/%s", c.baseURL, path)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("execute request: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
		return fmt.Errorf("decode response: %w", err)
	}

	return nil
}

// ListDevices fetches all devices from the hypervisor
func (c *Client) ListDevices(ctx context.Context) ([]*api.DeviceInfo, error) {
	var result api.ListDevicesResponse
	if err := c.doRequest(ctx, "GET", "devices", &result); err != nil {
		return nil, fmt.Errorf("list devices: %w", err)
	}
	return result.Devices, nil
}

// GetDevice fetches a specific device by UUID
func (c *Client) GetDevice(ctx context.Context, uuid string) (*api.DeviceInfo, error) {
	var result api.GetDeviceResponse
	if err := c.doRequest(ctx, "GET", fmt.Sprintf("devices/%s", uuid), &result); err != nil {
		return nil, fmt.Errorf("get device %s: %w", uuid, err)
	}
	return result.DeviceInfo, nil
}

// GetDeviceAllocations fetches allocations for a specific device
func (c *Client) GetDeviceAllocations(ctx context.Context, uuid string) ([]*api.DeviceAllocation, error) {
	workers, err := c.ListWorkers(ctx)
	if err != nil {
		return nil, fmt.Errorf("list workers: %w", err)
	}

	allocations := make([]*api.DeviceAllocation, 0)
	for _, worker := range workers {
		if worker.Allocation != nil {
			// Check if any device in the allocation matches the UUID
			for _, device := range worker.Allocation.DeviceInfos {
				if device.UUID == uuid {
					allocations = append(allocations, worker.Allocation)
					break
				}
			}
		}
	}

	return allocations, nil
}

// GetGPUMetrics fetches GPU metrics for all devices
// Note: This is a placeholder until a dedicated metrics endpoint is available
func (c *Client) GetGPUMetrics(ctx context.Context) (map[string]*api.GPUUsageMetrics, error) {
	// TODO: Implement when metrics endpoint is available
	// For now, return empty metrics to avoid errors
	return make(map[string]*api.GPUUsageMetrics), nil
}

// ListWorkers fetches all workers from the hypervisor
func (c *Client) ListWorkers(ctx context.Context) ([]api.WorkerDetail, error) {
	var result api.ListWorkersResponse
	if err := c.doRequest(ctx, "GET", "workers", &result); err != nil {
		return nil, fmt.Errorf("list workers: %w", err)
	}
	return result.Workers, nil
}

// GetWorker fetches a specific worker by ID
func (c *Client) GetWorker(ctx context.Context, workerID string) (*api.GetWorkerResponse, error) {
	var result api.GetWorkerResponse
	if err := c.doRequest(ctx, "GET", fmt.Sprintf("workers/%s", workerID), &result); err != nil {
		return nil, fmt.Errorf("get worker %s: %w", workerID, err)
	}
	return &result, nil
}

// GetWorkerMetrics fetches worker metrics for all workers
// This is optimized to batch requests when possible
func (c *Client) GetWorkerMetrics(ctx context.Context) (map[string]map[string]map[string]*api.WorkerMetrics, error) {
	workers, err := c.ListWorkers(ctx)
	if err != nil {
		return nil, err
	}

	metrics := make(map[string]map[string]map[string]*api.WorkerMetrics)
	for _, worker := range workers {
		workerDetail, err := c.GetWorker(ctx, worker.WorkerUID)
		if err != nil {
			// Continue on individual worker errors to get as much data as possible
			continue
		}

		if workerDetail.Metrics != nil {
			// Merge metrics by device UUID
			for deviceUUID, deviceMetrics := range workerDetail.Metrics {
				if metrics[deviceUUID] == nil {
					metrics[deviceUUID] = make(map[string]map[string]*api.WorkerMetrics)
				}
				// Copy worker metrics for this device
				for workerUID, workerMetrics := range deviceMetrics {
					metrics[deviceUUID][workerUID] = workerMetrics
				}
			}
		}
	}

	return metrics, nil
}
