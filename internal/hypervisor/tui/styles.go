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
	"github.com/charmbracelet/lipgloss"
)

var (
	TitleStyle        = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("63"))
	SubtitleStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("241"))
	BorderStyle       = lipgloss.NewStyle().Border(lipgloss.RoundedBorder()).BorderForeground(lipgloss.Color("62"))
	SelectedStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("212")).Bold(true)
	NormalStyle       = lipgloss.NewStyle().Foreground(lipgloss.Color("250"))
	MetricLabelStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("243")).Width(20)
	MetricValueStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("39")).Bold(true)
	ChartBarStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("46"))
	ChartEmptyStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("238"))
)

