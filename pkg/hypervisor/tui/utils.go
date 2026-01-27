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
)

// formatBytes formats bytes into human-readable format
func formatBytes(bytes uint64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// renderBarChart renders a bar chart for a percentage value
// This is a simple wrapper that calls the chart implementation
func renderBarChart(percentage float64, width int) string {
	if percentage > 100 {
		percentage = 100
	}
	if percentage < 0 {
		percentage = 0
	}

	filled := int(percentage / 100.0 * float64(width))
	empty := width - filled

	var bar strings.Builder
	bar.WriteString(ChartBarStyle.Render(strings.Repeat("█", filled)))
	bar.WriteString(ChartEmptyStyle.Render(strings.Repeat("░", empty)))
	bar.WriteString(fmt.Sprintf(" %.1f%%", percentage))

	return bar.String()
}
