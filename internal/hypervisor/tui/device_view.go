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
	"fmt"
	"strings"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/viewport"
)

// deviceItem represents a device in the list
type deviceItem struct {
	uuid  string
	model string
	index int32
}

func (d deviceItem) FilterValue() string {
	return fmt.Sprintf("%s %s %d", d.uuid, d.model, d.index)
}

func (d deviceItem) Title() string {
	return fmt.Sprintf("[%d] %s", d.index, d.model)
}

func (d deviceItem) Description() string {
	return d.uuid
}

func newDeviceDelegate() list.DefaultDelegate {
	d := list.NewDefaultDelegate()
	d.Styles.SelectedTitle = SelectedStyle
	d.Styles.SelectedDesc = SelectedStyle
	d.Styles.NormalTitle = NormalStyle
	d.Styles.NormalDesc = NormalStyle
	return d
}

// updateDeviceList updates the device list with current devices
func updateDeviceList(deviceList *list.Model, devices []*api.DeviceInfo) {
	deviceItems := make([]list.Item, len(devices))
	for i, device := range devices {
		deviceItems[i] = deviceItem{
			uuid:  device.UUID,
			model: device.Model,
			index: device.Index,
		}
	}
	deviceList.SetItems(deviceItems)
}

// updateDeviceDetail updates the device detail viewport
func updateDeviceDetail(
	ctx context.Context,
	client *Client,
	deviceDetail *viewport.Model,
	selectedDeviceUUID string,
	devices []*api.DeviceInfo,
	metrics map[string]*api.GPUUsageMetrics,
	deviceMetricsHistory map[string]*DeviceMetricsHistory,
) {
	var device *api.DeviceInfo
	for _, d := range devices {
		if d.UUID == selectedDeviceUUID {
			device = d
			break
		}
	}
	if device == nil {
		deviceDetail.SetContent("Device not found")
		return
	}

	deviceMetrics, hasMetrics := metrics[device.UUID]

	var content strings.Builder
	content.WriteString(TitleStyle.Render("Device Details\n\n"))

	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("UUID"), MetricValueStyle.Render(device.UUID)))
	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Vendor"), MetricValueStyle.Render(device.Vendor)))
	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Model"), MetricValueStyle.Render(device.Model)))
	content.WriteString(fmt.Sprintf("%s: %d\n", MetricLabelStyle.Render("Index"), device.Index))
	content.WriteString(fmt.Sprintf("%s: %d\n", MetricLabelStyle.Render("NUMA Node"), device.NUMANode))
	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Total Memory"), formatBytes(device.TotalMemory)))
	content.WriteString(fmt.Sprintf("%s: %.2f TFLOPS\n", MetricLabelStyle.Render("Max TFLOPS"), device.MaxTflops))
	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Driver Version"), device.DriverVersion))
	content.WriteString(fmt.Sprintf("%s: %s\n\n", MetricLabelStyle.Render("Firmware Version"), device.FirmwareVersion))

	if hasMetrics && deviceMetrics != nil {
		content.WriteString(TitleStyle.Render("Current Metrics\n\n"))
		content.WriteString(fmt.Sprintf("%s: %.1f%%\n", MetricLabelStyle.Render("Memory Usage"), deviceMetrics.MemoryPercentage))
		content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Memory Used"), formatBytes(deviceMetrics.MemoryBytes)))
		content.WriteString(fmt.Sprintf("%s: %.1f%%\n", MetricLabelStyle.Render("Compute Usage"), deviceMetrics.ComputePercentage))
		content.WriteString(fmt.Sprintf("%s: %.2f TFLOPS\n", MetricLabelStyle.Render("Compute TFLOPS"), deviceMetrics.ComputeTflops))
		content.WriteString(fmt.Sprintf("%s: %.1f°C\n", MetricLabelStyle.Render("Temperature"), deviceMetrics.Temperature))
		content.WriteString(fmt.Sprintf("%s: %d W\n", MetricLabelStyle.Render("Power Usage"), deviceMetrics.PowerUsage))
		content.WriteString(fmt.Sprintf("%s: %.1f MHz\n", MetricLabelStyle.Render("Graphics Clock"), deviceMetrics.GraphicsClockMHz))
		content.WriteString(fmt.Sprintf("%s: %.1f MHz\n\n", MetricLabelStyle.Render("SM Clock"), deviceMetrics.SMClockMHz))

		// Time-series charts
		if history, exists := deviceMetricsHistory[selectedDeviceUUID]; exists && history != nil {
			content.WriteString("\n")
			content.WriteString(history.MemoryChart.Render())
			content.WriteString("\n")
			content.WriteString(history.ComputeChart.Render())
			content.WriteString("\n")
			content.WriteString(history.TempChart.Render())
			content.WriteString("\n")
			content.WriteString(history.PowerChart.Render())
			content.WriteString("\n")
		}
	}

	// Get allocations for this device
	allocations, err := client.GetDeviceAllocations(ctx, device.UUID)
	if err == nil && len(allocations) > 0 {
		content.WriteString(TitleStyle.Render("Allocations\n\n"))
		for _, alloc := range allocations {
			content.WriteString(fmt.Sprintf("  Worker: %s\n", alloc.WorkerID))
			content.WriteString(fmt.Sprintf("  Pod: %s/%s\n", alloc.Namespace, alloc.PodName))
			content.WriteString(fmt.Sprintf("  Mode: %s\n", alloc.IsolationMode))
			if alloc.MemoryLimit > 0 {
				content.WriteString(fmt.Sprintf("  Memory Limit: %s\n", formatBytes(alloc.MemoryLimit)))
			}
			content.WriteString("\n")
		}
	}

	deviceDetail.SetContent(content.String())
}
