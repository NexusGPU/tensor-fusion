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
	"fmt"
	"strings"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/charmbracelet/bubbles/viewport"
)

// updateMetricsView updates the metrics viewport
func updateMetricsView(
	metricsView *viewport.Model,
	devices []*api.DeviceInfo,
	workers []*api.WorkerInfo,
	metrics map[string]*api.GPUUsageMetrics,
	workerMetrics map[string]map[string]map[string]*api.WorkerMetrics,
	lastUpdate time.Time,
) {
	var content strings.Builder
	content.WriteString(TitleStyle.Render("System Metrics\n\n"))
	content.WriteString(fmt.Sprintf("Last Update: %s\n\n", lastUpdate.Format(time.RFC3339)))

	// Device metrics overview
	content.WriteString(TitleStyle.Render("Device Metrics Overview\n\n"))
	for _, device := range devices {
		metrics, hasMetrics := metrics[device.UUID]
		content.WriteString(fmt.Sprintf("%s [%s]\n", device.Model, device.UUID[:8]))
		if hasMetrics && metrics != nil {
			content.WriteString(fmt.Sprintf(
				"  Memory: %.1f%% %s\n",
				metrics.MemoryPercentage,
				renderBarChart(metrics.MemoryPercentage, 20),
			))
			content.WriteString(fmt.Sprintf(
				"  Compute: %.1f%% %s\n",
				metrics.ComputePercentage,
				renderBarChart(metrics.ComputePercentage, 20),
			))
			content.WriteString(fmt.Sprintf("  Temperature: %.1f°C  Power: %dW\n", metrics.Temperature, metrics.PowerUsage))
		} else {
			content.WriteString("  No metrics available\n")
		}
		content.WriteString("\n")
	}

	// Worker metrics overview
	content.WriteString(TitleStyle.Render("Worker Metrics Overview\n\n"))
	for _, worker := range workers {
		content.WriteString(fmt.Sprintf("%s/%s\n", worker.Namespace, worker.WorkerName))
		for _, deviceUUID := range worker.AllocatedDevices {
			content.WriteString(fmt.Sprintf("  Device: %s\n", deviceUUID))
			if workerMetrics, exists := workerMetrics[deviceUUID]; exists {
				if wm, exists := workerMetrics[worker.WorkerUID]; exists {
					var totalMemory uint64
					var totalCompute float64
					for _, metrics := range wm {
						totalMemory += metrics.MemoryBytes
						totalCompute += metrics.ComputePercentage
					}
					content.WriteString(fmt.Sprintf("  Memory: %s\n", formatBytes(totalMemory)))
					content.WriteString(fmt.Sprintf("  Compute: %.1f%% %s\n", totalCompute, renderBarChart(totalCompute, 20)))
				} else {
					content.WriteString("  No metrics available\n")
				}
			} else {
				content.WriteString("  No metrics available\n")
			}
			content.WriteString("\n")
		}
	}

	metricsView.SetContent(content.String())
}
