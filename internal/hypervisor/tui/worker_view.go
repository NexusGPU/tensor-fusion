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

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/viewport"
)

func newWorkerDelegate() list.DefaultDelegate {
	d := list.NewDefaultDelegate()
	d.Styles.SelectedTitle = SelectedStyle
	d.Styles.SelectedDesc = SelectedStyle
	d.Styles.NormalTitle = NormalStyle
	d.Styles.NormalDesc = NormalStyle
	return d
}

// updateWorkerDetail updates the worker detail viewport
func updateWorkerDetail(
	workerDetail *viewport.Model,
	selectedWorkerUID string,
	workers []*api.WorkerInfo,
	workerMetrics map[string]map[string]map[string]*api.WorkerMetrics,
	workerMetricsHistory map[string]*WorkerMetricsHistory,
) {
	var worker *api.WorkerInfo
	for _, w := range workers {
		if w.WorkerUID == selectedWorkerUID {
			worker = w
			break
		}
	}
	if worker == nil {
		workerDetail.SetContent("Worker not found")
		return
	}

	var content strings.Builder
	content.WriteString(TitleStyle.Render("Worker Details\n\n"))

	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Worker UID"), MetricValueStyle.Render(worker.WorkerUID)))
	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Pod Name"), MetricValueStyle.Render(worker.WorkerName)))
	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Namespace"), MetricValueStyle.Render(worker.Namespace)))
	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Device UUIDs"), MetricValueStyle.Render(strings.Join(worker.AllocatedDevices, ", "))))

	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Isolation Mode"), MetricValueStyle.Render(string(worker.IsolationMode))))
	if worker.Limits.Vram.Value() > 0 {
		content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Memory Limit"), formatBytes(uint64(worker.Limits.Vram.Value()))))
	}
	if worker.Limits.Tflops.Value() > 0 {
		content.WriteString(fmt.Sprintf("%s: %.2f\n", MetricLabelStyle.Render("Compute Limit"), worker.Limits.Tflops.AsApproximateFloat64()))
	}

	// Get worker metrics
	for _, deviceUUID := range worker.AllocatedDevices {
		if deviceWorkerMetrics, exists := workerMetrics[deviceUUID]; exists {
			if wm, exists := deviceWorkerMetrics[worker.WorkerUID]; exists {
				content.WriteString(TitleStyle.Render("Current Metrics\n\n"))
				var totalMemory uint64
				var totalCompute float64
				var totalTflops float64

				for _, metrics := range wm {
					totalMemory += metrics.MemoryBytes
					totalCompute += metrics.ComputePercentage
					totalTflops += metrics.ComputeTflops
				}

				content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Memory Used"), formatBytes(totalMemory)))
				content.WriteString(fmt.Sprintf("%s: %.1f%%\n", MetricLabelStyle.Render("Compute Usage"), totalCompute))
				content.WriteString(fmt.Sprintf("%s: %.2f TFLOPS\n\n", MetricLabelStyle.Render("Compute TFLOPS"), totalTflops))

				// Time-series charts
				if history, exists := workerMetricsHistory[deviceUUID]; exists && history != nil {
					content.WriteString("\n")
					content.WriteString(history.MemoryChart.Render())
					content.WriteString("\n")
					content.WriteString(history.ComputeChart.Render())
					content.WriteString("\n")
				}
			}
		}
	}

	workerDetail.SetContent(content.String())
}
