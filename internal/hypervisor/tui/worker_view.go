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

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/viewport"
)

// WorkerInfo represents worker information
type WorkerInfo struct {
	UID        string
	PodName    string
	Namespace  string
	DeviceUUID string
	Allocation *api.DeviceAllocation
}

// workerItem represents a worker in the list
type workerItem struct {
	uid       string
	podName   string
	namespace string
}

func (w workerItem) FilterValue() string {
	return fmt.Sprintf("%s %s %s", w.uid, w.podName, w.namespace)
}

func (w workerItem) Title() string {
	return fmt.Sprintf("%s/%s", w.namespace, w.podName)
}

func (w workerItem) Description() string {
	return w.uid
}

func newWorkerDelegate() list.DefaultDelegate {
	d := list.NewDefaultDelegate()
	d.Styles.SelectedTitle = SelectedStyle
	d.Styles.SelectedDesc = SelectedStyle
	d.Styles.NormalTitle = NormalStyle
	d.Styles.NormalDesc = NormalStyle
	return d
}

// updateWorkerList updates the worker list with current workers
func updateWorkerList(workerList *list.Model, workers []WorkerInfo) {
	workerItems := make([]list.Item, len(workers))
	for i, worker := range workers {
		workerItems[i] = workerItem{
			uid:       worker.UID,
			podName:   worker.PodName,
			namespace: worker.Namespace,
		}
	}
	workerList.SetItems(workerItems)
}

// updateWorkerDetail updates the worker detail viewport
func updateWorkerDetail(
	workerDetail *viewport.Model,
	selectedWorkerUID string,
	workers []WorkerInfo,
	workerMetrics map[string]map[string]map[string]*api.WorkerMetrics,
	workerMetricsHistory map[string]*WorkerMetricsHistory,
) {
	var worker *WorkerInfo
	for _, w := range workers {
		if w.UID == selectedWorkerUID {
			worker = &w
			break
		}
	}
	if worker == nil {
		workerDetail.SetContent("Worker not found")
		return
	}

	var content strings.Builder
	content.WriteString(TitleStyle.Render("Worker Details\n\n"))

	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Worker UID"), MetricValueStyle.Render(worker.UID)))
	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Pod Name"), MetricValueStyle.Render(worker.PodName)))
	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Namespace"), MetricValueStyle.Render(worker.Namespace)))
	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Device UUID"), MetricValueStyle.Render(worker.DeviceUUID)))

	if worker.Allocation != nil {
		content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Isolation Mode"), MetricValueStyle.Render(string(worker.Allocation.IsolationMode))))
		if worker.Allocation.MemoryLimit > 0 {
			content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Memory Limit"), formatBytes(worker.Allocation.MemoryLimit)))
		}
		if worker.Allocation.ComputeLimit > 0 {
			content.WriteString(fmt.Sprintf("%s: %d%%\n", MetricLabelStyle.Render("Compute Limit"), worker.Allocation.ComputeLimit))
		}
		content.WriteString(fmt.Sprintf("%s: %s\n\n", MetricLabelStyle.Render("Allocated At"), worker.Allocation.AllocatedAt.Format(time.RFC3339)))
	}

	// Get worker metrics
	if deviceWorkerMetrics, exists := workerMetrics[worker.DeviceUUID]; exists {
		if wm, exists := deviceWorkerMetrics[worker.UID]; exists {
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
			if history, exists := workerMetricsHistory[selectedWorkerUID]; exists && history != nil {
				content.WriteString("\n")
				content.WriteString(history.MemoryChart.Render())
				content.WriteString("\n")
				content.WriteString(history.ComputeChart.Render())
				content.WriteString("\n")
			}
		}
	}

	workerDetail.SetContent(content.String())
}
