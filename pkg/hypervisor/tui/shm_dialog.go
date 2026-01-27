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
	"path/filepath"
	"strings"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

var (
	shmBasePath = filepath.Join(constants.TFDataPath, constants.SharedMemMountSubPath)
)

// ShmDialogModel represents the shared memory detail dialog
type ShmDialogModel struct {
	viewport   viewport.Model
	content    string
	width      int
	height     int
	isVisible  bool
	workerInfo *api.WorkerInfo
}

// NewShmDialogModel creates a new SHM dialog model
func NewShmDialogModel() *ShmDialogModel {
	return &ShmDialogModel{
		viewport:  viewport.New(0, 0),
		isVisible: false,
	}
}

// Init initializes the dialog
func (m *ShmDialogModel) Init() tea.Cmd {
	return nil
}

// Update updates the dialog
func (m *ShmDialogModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	if !m.isVisible {
		return m, nil
	}

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "esc", "q":
			m.isVisible = false
			return m, nil
		}
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.resize()
		return m, nil
	}

	var cmd tea.Cmd
	m.viewport, cmd = m.viewport.Update(msg)
	return m, cmd
}

// View renders the dialog
func (m *ShmDialogModel) View() string {
	if !m.isVisible {
		return ""
	}

	// Calculate dialog dimensions (80% of screen, centered)
	dialogWidth := int(float64(m.width) * 0.8)
	dialogHeight := int(float64(m.height) * 0.8)

	if dialogWidth < 40 {
		dialogWidth = 40
	}
	if dialogHeight < 10 {
		dialogHeight = 10
	}

	// Create dialog box
	box := BorderStyle.
		Width(dialogWidth).
		Height(dialogHeight).
		Render(m.viewport.View())

	// Center the dialog
	return lipgloss.Place(
		m.width,
		m.height,
		lipgloss.Center,
		lipgloss.Center,
		box,
	)
}

// Show displays the dialog with SHM details for the given worker
func (m *ShmDialogModel) Show(workerInfo *api.WorkerInfo) {
	m.workerInfo = workerInfo
	m.isVisible = true
	m.resize()
	m.updateContent()
}

// Hide hides the dialog
func (m *ShmDialogModel) Hide() {
	m.isVisible = false
}

// IsVisible returns whether the dialog is visible
func (m *ShmDialogModel) IsVisible() bool {
	return m.isVisible
}

// resize resizes the dialog viewport
func (m *ShmDialogModel) resize() {
	if !m.isVisible {
		return
	}

	dialogWidth := int(float64(m.width) * 0.8)
	dialogHeight := int(float64(m.height) * 0.8)

	if dialogWidth < 40 {
		dialogWidth = 40
	}
	if dialogHeight < 10 {
		dialogHeight = 10
	}

	// Account for border
	m.viewport.Width = dialogWidth - 2
	m.viewport.Height = dialogHeight - 2
}

// updateContent updates the dialog content with SHM details
func (m *ShmDialogModel) updateContent() {
	if m.workerInfo == nil {
		m.content = "No worker information available"
		m.viewport.SetContent(m.content)
		return
	}

	var content strings.Builder

	// Title
	content.WriteString(TitleStyle.Render("Shared Memory Details\n\n"))

	// Construct pod identifier and path
	podIdentifier := workerstate.NewPodIdentifier(m.workerInfo.Namespace, m.workerInfo.WorkerName)
	podPath := podIdentifier.ToPath(shmBasePath)
	shmPath := filepath.Join(podPath, workerstate.ShmPathSuffix)

	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Pod"), MetricValueStyle.Render(podIdentifier.String())))
	content.WriteString(fmt.Sprintf("%s: %s\n\n", MetricLabelStyle.Render("SHM Path"), MetricValueStyle.Render(shmPath)))

	// Try to open the shared memory handle
	handle, err := workerstate.OpenSharedMemoryHandle(podPath)
	if err != nil {
		content.WriteString(fmt.Sprintf("%s: %s\n\n", MetricLabelStyle.Render("Error"), MetricValueStyle.Render(err.Error())))
		m.content = content.String()
		m.viewport.SetContent(m.content)
		return
	}
	defer func() {
		_ = handle.Close()
	}()

	// Get the state
	state := handle.GetState()
	if state == nil {
		content.WriteString(fmt.Sprintf("%s: %s\n\n", MetricLabelStyle.Render("Error"), MetricValueStyle.Render("Shared memory state is null")))
		m.content = content.String()
		m.viewport.SetContent(m.content)
		return
	}

	// Basic information
	deviceCount := state.DeviceCount()
	content.WriteString(fmt.Sprintf("%s: %d\n", MetricLabelStyle.Render("Device Count"), deviceCount))

	lastHeartbeat := state.GetLastHeartbeat()
	heartbeatTime := time.Unix(int64(lastHeartbeat), 0)
	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Last Heartbeat"), heartbeatTime.Format(time.RFC3339)))

	// Health check (2 seconds timeout)
	isHealthy := state.IsHealthy(2 * time.Second)
	healthStatus := "Healthy"
	if !isHealthy {
		healthStatus = "Unhealthy"
	}
	content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Health Status"), MetricValueStyle.Render(healthStatus)))

	// Version information
	version := state.Version()
	content.WriteString(fmt.Sprintf("%s: v%d\n\n", MetricLabelStyle.Render("State Version"), version))

	// Device details based on version
	if version == 1 && state.V1 != nil {
		// V1 format
		for i := 0; i < deviceCount; i++ {
			if !state.V1.HasDevice(i) {
				continue
			}

			device := &state.V1.Devices[i]
			if !device.IsActive() {
				continue
			}

			uuid := device.GetUUID()
			availableCores := device.DeviceInfo.AvailableCudaCores
			totalCores := device.DeviceInfo.TotalCudaCores
			memLimit := device.DeviceInfo.MemLimit
			podMemoryUsed := device.DeviceInfo.PodMemoryUsed
			upLimit := device.DeviceInfo.UpLimit

			content.WriteString(fmt.Sprintf("Device %d:\n", i))
			content.WriteString(fmt.Sprintf("  %s: %s\n", MetricLabelStyle.Render("UUID"), MetricValueStyle.Render(uuid)))
			content.WriteString(fmt.Sprintf("  %s: %d / %d\n", MetricLabelStyle.Render("Cores"), availableCores, totalCores))
			content.WriteString(fmt.Sprintf("  %s: %s\n", MetricLabelStyle.Render("Mem Limit"), formatBytes(memLimit)))
			content.WriteString(fmt.Sprintf("  %s: %s\n", MetricLabelStyle.Render("Mem Used"), formatBytes(podMemoryUsed)))
			content.WriteString(fmt.Sprintf("  %s: %d%%\n\n", MetricLabelStyle.Render("Up Limit"), upLimit))
		}
	} else if version == 2 && state.V2 != nil {
		// V2 format with ERL
		for i := 0; i < deviceCount; i++ {
			if !state.V2.HasDevice(i) {
				continue
			}

			device := &state.V2.Devices[i]
			if !device.IsActive() {
				continue
			}

			uuid := device.GetUUID()
			totalCores := device.DeviceInfo.TotalCudaCores
			memLimit := device.DeviceInfo.MemLimit
			podMemoryUsed := device.DeviceInfo.PodMemoryUsed
			upLimit := device.DeviceInfo.UpLimit

			// ERL information
			erlCurrentTokens := device.DeviceInfo.GetERLCurrentTokens()
			erlTokenCapacity := device.DeviceInfo.GetERLTokenCapacity()
			erlTokenRefillRate := device.DeviceInfo.GetERLTokenRefillRate()
			erlLastTokenUpdate := device.DeviceInfo.GetERLLastTokenUpdate()

			content.WriteString(fmt.Sprintf("Device %d:\n", i))
			content.WriteString(fmt.Sprintf("  %s: %s\n", MetricLabelStyle.Render("UUID"), MetricValueStyle.Render(uuid)))
			content.WriteString(fmt.Sprintf("  %s: %d\n", MetricLabelStyle.Render("Total Cores"), totalCores))
			content.WriteString(fmt.Sprintf("  %s: %s\n", MetricLabelStyle.Render("Mem Limit"), formatBytes(memLimit)))
			content.WriteString(fmt.Sprintf("  %s: %s\n", MetricLabelStyle.Render("Mem Used"), formatBytes(podMemoryUsed)))
			content.WriteString(fmt.Sprintf("  %s: %d%%\n", MetricLabelStyle.Render("Up Limit"), upLimit))
			content.WriteString(fmt.Sprintf("  %s: %.1f / %.1f (rate: %.1f/s, updated: %.0fµs)\n\n",
				MetricLabelStyle.Render("ERL Tokens"),
				erlCurrentTokens,
				erlTokenCapacity,
				erlTokenRefillRate,
				erlLastTokenUpdate))
		}
	} else {
		content.WriteString(fmt.Sprintf("Unknown shared memory version: %d\n\n", version))
	}

	// Additional state information
	pids := state.GetAllPIDs()
	content.WriteString(fmt.Sprintf("%s: %d\n", MetricLabelStyle.Render("Active PIDs Count"), len(pids)))
	if len(pids) > 0 {
		pidStrs := make([]string, len(pids))
		for i, pid := range pids {
			pidStrs[i] = fmt.Sprintf("%d", pid)
		}
		content.WriteString(fmt.Sprintf("%s: %s\n", MetricLabelStyle.Render("Active PIDs"), strings.Join(pidStrs, ", ")))
	}

	m.content = content.String()
	m.viewport.SetContent(m.content)
	m.viewport.GotoTop()
}
