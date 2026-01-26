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
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

const (
	viewDevices = iota
	viewWorkers
	viewMetrics
	viewDeviceDetail
	viewWorkerDetail
)

// Model represents the TUI model
type Model struct {
	ctx    context.Context
	client *Client

	currentView   int
	devices       []*api.DeviceInfo
	workers       []*api.WorkerInfo
	metrics       map[string]*api.GPUUsageMetrics
	workerMetrics map[string]map[string]map[string]*api.WorkerMetrics

	// Metrics history for time-series charts
	deviceMetricsHistory map[string]*DeviceMetricsHistory
	workerMetricsHistory map[string]*WorkerMetricsHistory

	deviceList   list.Model
	workerList   list.Model
	deviceDetail viewport.Model
	workerDetail viewport.Model
	metricsView  viewport.Model

	shmDialog *ShmDialogModel

	selectedDeviceUUID string
	selectedWorkerUID  string

	width  int
	height int

	lastUpdate time.Time
}

// DeviceMetricsHistory tracks historical metrics for a device
type DeviceMetricsHistory struct {
	MemoryChart  *TimeSeriesChart
	ComputeChart *TimeSeriesChart
	TempChart    *TimeSeriesChart
	PowerChart   *TimeSeriesChart
}

// WorkerMetricsHistory tracks historical metrics for a worker
type WorkerMetricsHistory struct {
	MemoryChart  *TimeSeriesChart
	ComputeChart *TimeSeriesChart
}

type tickMsg time.Time
type updateDataMsg struct {
	devices       []*api.DeviceInfo
	workers       []*api.WorkerInfo
	metrics       map[string]*api.GPUUsageMetrics
	workerMetrics map[string]map[string]map[string]*api.WorkerMetrics
}

// NewModel creates a new TUI model
func NewModel(ctx context.Context, client *Client) *Model {
	m := &Model{
		ctx:                  ctx,
		client:               client,
		currentView:          viewDevices,
		metrics:              make(map[string]*api.GPUUsageMetrics),
		workerMetrics:        make(map[string]map[string]map[string]*api.WorkerMetrics),
		deviceMetricsHistory: make(map[string]*DeviceMetricsHistory),
		workerMetricsHistory: make(map[string]*WorkerMetricsHistory),
	}

	// Initialize device list
	deviceItems := []list.Item{}
	m.deviceList = list.New(deviceItems, newDeviceDelegate(), 0, 0)
	m.deviceList.Title = "GPU Devices"
	m.deviceList.SetShowStatusBar(false)
	m.deviceList.SetFilteringEnabled(true)
	m.deviceList.Styles.Title = TitleStyle
	m.deviceList.Styles.FilterPrompt = SubtitleStyle
	m.deviceList.Styles.FilterCursor = SelectedStyle

	// Initialize worker list
	workerItems := []list.Item{}
	m.workerList = list.New(workerItems, newWorkerDelegate(), 0, 0)
	m.workerList.Title = "Workers"
	m.workerList.SetShowStatusBar(false)
	m.workerList.SetFilteringEnabled(true)
	m.workerList.Styles.Title = TitleStyle
	m.workerList.Styles.FilterPrompt = SubtitleStyle
	m.workerList.Styles.FilterCursor = SelectedStyle

	// Initialize detail viewports
	m.deviceDetail = viewport.New(0, 0)
	m.workerDetail = viewport.New(0, 0)
	m.metricsView = viewport.New(0, 0)

	// Initialize SHM dialog
	m.shmDialog = NewShmDialogModel()

	return m
}

func (m *Model) Init() tea.Cmd {
	return tea.Batch(
		m.updateData(),
		tick(),
	)
}

func (m *Model) updateData() tea.Cmd {
	return func() tea.Msg {
		ctx, cancel := context.WithTimeout(m.ctx, 5*time.Second)
		defer cancel()

		// Get devices
		devices, err := m.client.ListDevices(ctx)
		if err != nil {
			devices = []*api.DeviceInfo{}
		}

		// Get workers
		workerDetails, err := m.client.ListWorkers(ctx)
		if err != nil {
			workerDetails = []*api.WorkerAllocation{}
		}

		workers := make([]*api.WorkerInfo, 0, len(workerDetails))
		for _, worker := range workerDetails {
			if worker == nil {
				continue
			}
			workers = append(workers, worker.WorkerInfo)
		}

		// Get GPU metrics - for now, we'll need to add a metrics endpoint
		// For now, return empty metrics
		metrics := make(map[string]*api.GPUUsageMetrics)

		// Get worker metrics
		workerMetrics, err := m.client.GetWorkerMetrics(ctx)
		if err != nil {
			workerMetrics = make(map[string]map[string]map[string]*api.WorkerMetrics)
		}

		return updateDataMsg{
			devices:       devices,
			workers:       workers,
			metrics:       metrics,
			workerMetrics: workerMetrics,
		}
	}
}

func tick() tea.Cmd {
	return tea.Tick(2*time.Second, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

//nolint:gocyclo // Complex state machine with many message types and view transitions
func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmds []tea.Cmd

	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.resizeViews()
		if m.shmDialog != nil {
			m.shmDialog.width = msg.Width
			m.shmDialog.height = msg.Height
		}
		return m, nil

	case tea.KeyMsg:
		switch msg.String() {
		case "q", "ctrl+c":
			return m, tea.Quit
		case "1":
			m.currentView = viewDevices
			return m, nil
		case "2":
			m.currentView = viewWorkers
			return m, nil
		case "3":
			m.currentView = viewMetrics
			return m, nil
		case "esc":
			// Close SHM dialog if visible
			if m.shmDialog != nil && m.shmDialog.IsVisible() {
				m.shmDialog.Hide()
				return m, nil
			}
			if m.currentView == viewDeviceDetail || m.currentView == viewWorkerDetail {
				if m.currentView == viewDeviceDetail {
					m.currentView = viewDevices
				} else {
					m.currentView = viewWorkers
				}
				return m, nil
			}
		case "enter":
			switch m.currentView {
			case viewDevices:
				if selectedItem := m.deviceList.SelectedItem(); selectedItem != nil {
					item := selectedItem.(deviceItem)
					m.selectedDeviceUUID = item.uuid
					m.currentView = viewDeviceDetail
					// Initialize history if needed
					if m.deviceMetricsHistory[m.selectedDeviceUUID] == nil {
						m.initDeviceHistory(m.selectedDeviceUUID)
					}
					updateDeviceDetail(m.ctx, m.client, &m.deviceDetail, m.selectedDeviceUUID, m.devices, m.metrics, m.deviceMetricsHistory)
					return m, nil
				}
			case viewWorkers:
				if selectedItem := m.workerList.SelectedItem(); selectedItem != nil {
					item := selectedItem.(*api.WorkerInfo)
					m.selectedWorkerUID = item.WorkerUID
					m.currentView = viewWorkerDetail
					// Initialize history if needed
					if m.workerMetricsHistory[m.selectedWorkerUID] == nil {
						m.initWorkerHistory(m.selectedWorkerUID)
					}
					updateWorkerDetail(&m.workerDetail, m.selectedWorkerUID, m.workers, m.workerMetrics, m.workerMetricsHistory)
					return m, nil
				}
			case viewWorkerDetail:
				// Check if SHM dialog is visible, if so, close it
				if m.shmDialog != nil && m.shmDialog.IsVisible() {
					m.shmDialog.Hide()
					return m, nil
				}
				// Otherwise, show SHM dialog if isolation mode is soft
				var worker *api.WorkerInfo
				for _, w := range m.workers {
					if w.WorkerUID == m.selectedWorkerUID {
						worker = w
					}
				}
				if worker != nil {
					m.shmDialog.Show(worker)
					return m, nil
				}
			}
		}

	case tickMsg:
		return m, tea.Batch(m.updateData(), tick())

	case updateDataMsg:
		m.devices = msg.devices
		m.workers = msg.workers
		m.metrics = msg.metrics
		m.workerMetrics = msg.workerMetrics
		m.lastUpdate = time.Now()

		// Update metrics history for charts
		m.updateMetricsHistory()

		updateDeviceList(&m.deviceList, m.devices)

		workerItems := make([]list.Item, len(m.workers))
		for i, worker := range m.workers {
			workerItems[i] = worker
		}
		m.workerList.SetItems(workerItems)
		switch m.currentView {
		case viewDeviceDetail:
			updateDeviceDetail(m.ctx, m.client, &m.deviceDetail, m.selectedDeviceUUID, m.devices, m.metrics, m.deviceMetricsHistory)
		case viewWorkerDetail:
			updateWorkerDetail(&m.workerDetail, m.selectedWorkerUID, m.workers, m.workerMetrics, m.workerMetricsHistory)
		case viewMetrics:
			updateMetricsView(&m.metricsView, m.devices, m.workers, m.metrics, m.workerMetrics, m.lastUpdate)
		}
		return m, nil
	}

	// Update sub-views
	// If SHM dialog is visible, it should handle input first
	if m.shmDialog != nil && m.shmDialog.IsVisible() {
		var cmd tea.Cmd
		_, cmd = m.shmDialog.Update(msg)
		cmds = append(cmds, cmd)
		return m, tea.Batch(cmds...)
	}

	switch m.currentView {
	case viewDevices:
		var cmd tea.Cmd
		m.deviceList, cmd = m.deviceList.Update(msg)
		cmds = append(cmds, cmd)
	case viewWorkers:
		var cmd tea.Cmd
		m.workerList, cmd = m.workerList.Update(msg)
		cmds = append(cmds, cmd)
	case viewDeviceDetail:
		var cmd tea.Cmd
		m.deviceDetail, cmd = m.deviceDetail.Update(msg)
		cmds = append(cmds, cmd)
	case viewWorkerDetail:
		var cmd tea.Cmd
		m.workerDetail, cmd = m.workerDetail.Update(msg)
		cmds = append(cmds, cmd)
	case viewMetrics:
		var cmd tea.Cmd
		m.metricsView, cmd = m.metricsView.Update(msg)
		cmds = append(cmds, cmd)
	}

	return m, tea.Batch(cmds...)
}

func (m *Model) resizeViews() {
	headerHeight := 3
	footerHeight := 2
	availableHeight := m.height - headerHeight - footerHeight

	switch m.currentView {
	case viewDevices:
		m.deviceList.SetWidth(m.width)
		m.deviceList.SetHeight(availableHeight)
	case viewWorkers:
		m.workerList.SetWidth(m.width)
		m.workerList.SetHeight(availableHeight)
	case viewDeviceDetail, viewWorkerDetail, viewMetrics:
		width := m.width
		height := availableHeight
		m.deviceDetail.Width = width
		m.deviceDetail.Height = height
		m.workerDetail.Width = width
		m.workerDetail.Height = height
		m.metricsView.Width = width
		m.metricsView.Height = height

		// Update chart dimensions when resizing
		chartWidth := width - 20
		if chartWidth < 40 {
			chartWidth = 40
		}
		chartHeight := 8

		if m.currentView == viewDeviceDetail && m.selectedDeviceUUID != "" {
			if history := m.deviceMetricsHistory[m.selectedDeviceUUID]; history != nil {
				history.MemoryChart.SetDimensions(chartWidth, chartHeight)
				history.ComputeChart.SetDimensions(chartWidth, chartHeight)
				history.TempChart.SetDimensions(chartWidth, chartHeight)
				history.PowerChart.SetDimensions(chartWidth, chartHeight)
			}
		} else if m.currentView == viewWorkerDetail && m.selectedWorkerUID != "" {
			if history := m.workerMetricsHistory[m.selectedWorkerUID]; history != nil {
				history.MemoryChart.SetDimensions(chartWidth, chartHeight)
				history.ComputeChart.SetDimensions(chartWidth, chartHeight)
			}
		}
	}
}

func (m *Model) View() string {
	if m.width == 0 || m.height == 0 {
		return "Initializing..."
	}

	var view string
	switch m.currentView {
	case viewDevices:
		view = m.deviceList.View()
	case viewWorkers:
		view = m.workerList.View()
	case viewDeviceDetail:
		view = m.deviceDetail.View()
	case viewWorkerDetail:
		view = m.workerDetail.View()
	case viewMetrics:
		view = m.metricsView.View()
	}

	header := m.renderHeader()
	footer := m.renderFooter()

	mainView := lipgloss.JoinVertical(lipgloss.Left, header, view, footer)

	// Render SHM dialog on top if visible
	if m.shmDialog != nil && m.shmDialog.IsVisible() {
		dialogView := m.shmDialog.View()
		// The dialog already handles centering, so we just return it
		// It will overlay on top of the main view
		return dialogView
	}

	return mainView
}

// initDeviceHistory initializes metrics history for a device
func (m *Model) initDeviceHistory(deviceUUID string) {
	chartWidth := m.width - 20
	if chartWidth < 40 {
		chartWidth = 40
	}
	chartHeight := 8

	m.deviceMetricsHistory[deviceUUID] = &DeviceMetricsHistory{
		MemoryChart:  NewTimeSeriesChart(chartWidth, chartHeight, "Memory Usage"),
		ComputeChart: NewTimeSeriesChart(chartWidth, chartHeight, "Compute Usage"),
		TempChart:    NewTimeSeriesChart(chartWidth, chartHeight, "Temperature"),
		PowerChart:   NewTimeSeriesChart(chartWidth, chartHeight, "Power Usage"),
	}

	// Set max values
	m.deviceMetricsHistory[deviceUUID].MemoryChart.SetMaxValue(100.0)
	m.deviceMetricsHistory[deviceUUID].ComputeChart.SetMaxValue(100.0)
	m.deviceMetricsHistory[deviceUUID].TempChart.SetMaxValue(100.0)  // Will auto-scale
	m.deviceMetricsHistory[deviceUUID].PowerChart.SetMaxValue(500.0) // Will auto-scale
}

// initWorkerHistory initializes metrics history for a worker
func (m *Model) initWorkerHistory(workerUID string) {
	chartWidth := m.width - 20
	if chartWidth < 40 {
		chartWidth = 40
	}
	chartHeight := 8

	m.workerMetricsHistory[workerUID] = &WorkerMetricsHistory{
		MemoryChart:  NewTimeSeriesChart(chartWidth, chartHeight, "Memory Usage"),
		ComputeChart: NewTimeSeriesChart(chartWidth, chartHeight, "Compute Usage"),
	}

	// Set max values
	m.workerMetricsHistory[workerUID].MemoryChart.SetMaxValue(100.0)
	m.workerMetricsHistory[workerUID].ComputeChart.SetMaxValue(100.0)
}

// updateMetricsHistory updates the metrics history with current values
func (m *Model) updateMetricsHistory() {
	// Update device metrics history
	for deviceUUID, metrics := range m.metrics {
		if metrics == nil {
			continue
		}

		history := m.deviceMetricsHistory[deviceUUID]
		if history == nil {
			// Only initialize if we're viewing this device
			if m.currentView == viewDeviceDetail && m.selectedDeviceUUID == deviceUUID {
				m.initDeviceHistory(deviceUUID)
				history = m.deviceMetricsHistory[deviceUUID]
			} else {
				continue
			}
		}

		history.MemoryChart.AddDataPoint(metrics.MemoryPercentage)
		history.ComputeChart.AddDataPoint(metrics.ComputePercentage)
		history.TempChart.AddDataPoint(metrics.Temperature)
		history.PowerChart.AddDataPoint(float64(metrics.PowerUsage))
	}

	// Update worker metrics history
	for _, deviceWorkers := range m.workerMetrics {
		for workerUID, workerMetrics := range deviceWorkers {
			history := m.workerMetricsHistory[workerUID]
			if history == nil {
				// Only initialize if we're viewing this worker
				if m.currentView == viewWorkerDetail && m.selectedWorkerUID == workerUID {
					m.initWorkerHistory(workerUID)
					history = m.workerMetricsHistory[workerUID]
				} else {
					continue
				}
			}

			// Aggregate metrics for this worker
			var totalMemory uint64
			var totalCompute float64
			for _, metrics := range workerMetrics {
				totalMemory += metrics.MemoryBytes
				totalCompute += metrics.ComputePercentage
			}

			// Calculate percentage if we have allocation info
			var memPercent float64
			for _, worker := range m.workers {
				if worker.WorkerUID == workerUID && worker.Limits.Vram.Value() > 0 {
					memPercent = float64(totalMemory) / float64(worker.Limits.Vram.Value()) * 100.0
					break
				}
			}

			history.MemoryChart.AddDataPoint(memPercent)
			history.ComputeChart.AddDataPoint(totalCompute)
		}
	}
}

func (m *Model) renderHeader() string {
	title := TitleStyle.Render("Tensor Fusion Hypervisor")
	tabs := []string{
		m.renderTab("Devices [1]", m.currentView == viewDevices),
		m.renderTab("Workers [2]", m.currentView == viewWorkers),
		m.renderTab("Metrics [3]", m.currentView == viewMetrics),
	}
	tabLine := lipgloss.JoinHorizontal(lipgloss.Left, tabs...)
	return lipgloss.JoinVertical(lipgloss.Left, title, tabLine)
}

func (m *Model) renderTab(text string, active bool) string {
	if active {
		return SelectedStyle.Render(text)
	}
	return NormalStyle.Render(text)
}

func (m *Model) renderFooter() string {
	help := "Press 'q' to quit | 'Enter' to view details"
	if m.currentView == viewWorkerDetail {
		help += " (Enter again for SHM details if soft isolation)"
	}
	help += " | 'Esc' to go back | '1/2/3' to switch views"
	return SubtitleStyle.Render(help)
}
