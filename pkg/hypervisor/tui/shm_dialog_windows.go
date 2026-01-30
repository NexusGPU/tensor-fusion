//go:build windows

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
	"os"

	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
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
	fmt.Fprintf(os.Stderr, "Shared memory dialog is not supported on Windows\n")
	return nil
}

// Update updates the dialog
func (m *ShmDialogModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	return m, nil
}

// View renders the dialog
func (m *ShmDialogModel) View() string {
	return ""
}

// Show displays the dialog with SHM details for the given worker
func (m *ShmDialogModel) Show(workerInfo *api.WorkerInfo) {
	// Not supported on Windows
}

// Hide hides the dialog
func (m *ShmDialogModel) Hide() {
	// Not supported on Windows
}

// IsVisible returns whether the dialog is visible
func (m *ShmDialogModel) IsVisible() bool {
	return false
}
