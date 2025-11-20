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

const (
	maxHistorySize = 60 // Keep 60 data points for ~2 minutes at 2s intervals
)

// TimeSeriesChart represents a time-series chart for metrics
type TimeSeriesChart struct {
	data     []float64
	width    int
	height   int
	maxValue float64
	minValue float64
	label    string
}

// NewTimeSeriesChart creates a new time-series chart
func NewTimeSeriesChart(width, height int, label string) *TimeSeriesChart {
	return &TimeSeriesChart{
		data:     make([]float64, 0, maxHistorySize),
		width:    width,
		height:   height,
		maxValue: 100.0, // Default max for percentages
		minValue: 0.0,
		label:    label,
	}
}

// AddDataPoint adds a new data point to the chart
func (c *TimeSeriesChart) AddDataPoint(value float64) {
	c.data = append(c.data, value)
	if len(c.data) > maxHistorySize {
		c.data = c.data[1:] // Remove oldest point
	}

	// Auto-scale max value
	if value > c.maxValue {
		c.maxValue = value * 1.1 // Add 10% padding
	}
	if value < c.minValue {
		c.minValue = value
	}
}

// SetMaxValue sets the maximum value for the chart scale
func (c *TimeSeriesChart) SetMaxValue(max float64) {
	c.maxValue = max
}

// SetDimensions sets the width and height of the chart
func (c *TimeSeriesChart) SetDimensions(width, height int) {
	c.width = width
	c.height = height
}

// Render renders the time-series chart as a string
//
//nolint:gocyclo // Complex rendering logic with multiple conditional branches
func (c *TimeSeriesChart) Render() string {
	if len(c.data) == 0 {
		return fmt.Sprintf("%s: No data\n", c.label)
	}

	var result strings.Builder
	result.WriteString(fmt.Sprintf("%s (max: %.1f)\n", c.label, c.maxValue))

	if c.height < 2 {
		// Single line mode - just show current value
		lastValue := c.data[len(c.data)-1]
		result.WriteString(renderBarChart(lastValue, c.width))
		return result.String()
	}

	// Multi-line chart
	chartHeight := c.height - 1 // Reserve one line for label
	if chartHeight < 1 {
		chartHeight = 1
	}

	// Create a grid for the chart
	grid := make([][]rune, chartHeight)
	for i := range grid {
		grid[i] = make([]rune, c.width)
		for j := range grid[i] {
			grid[i][j] = ' '
		}
	}

	// Handle edge case: maxValue == minValue
	valueRange := c.maxValue - c.minValue
	if valueRange == 0 {
		valueRange = 1.0 // Avoid division by zero
	}

	// Draw the data
	dataLen := len(c.data)
	if dataLen > c.width {
		// Downsample if we have more data points than width
		step := float64(dataLen) / float64(c.width)
		for x := 0; x < c.width; x++ {
			idx := int(float64(x) * step)
			if idx >= dataLen {
				idx = dataLen - 1
			}
			value := c.data[idx]
			y := int((c.maxValue - value) / valueRange * float64(chartHeight-1))
			if y < 0 {
				y = 0
			}
			if y >= chartHeight {
				y = chartHeight - 1
			}
			grid[y][x] = '█'

			// Draw line connecting to previous point
			if x > 0 {
				prevIdx := int(float64(x-1) * step)
				if prevIdx >= dataLen {
					prevIdx = dataLen - 1
				}
				prevValue := c.data[prevIdx]
				prevY := int((c.maxValue - prevValue) / valueRange * float64(chartHeight-1))
				if prevY < 0 {
					prevY = 0
				}
				if prevY >= chartHeight {
					prevY = chartHeight - 1
				}

				// Draw connecting line
				startY, endY := prevY, y
				if startY > endY {
					startY, endY = endY, startY
				}
				for lineY := startY; lineY <= endY; lineY++ {
					if lineY < chartHeight {
						if grid[lineY][x] == ' ' {
							grid[lineY][x] = '│'
						}
					}
				}
			}
		}
	} else {
		// Draw all data points
		for x, value := range c.data {
			if x >= c.width {
				break
			}
			y := int((c.maxValue - value) / valueRange * float64(chartHeight-1))
			if y < 0 {
				y = 0
			}
			if y >= chartHeight {
				y = chartHeight - 1
			}
			grid[y][x] = '█'

			// Draw connecting line
			if x > 0 {
				prevValue := c.data[x-1]
				prevY := int((c.maxValue - prevValue) / valueRange * float64(chartHeight-1))
				if prevY < 0 {
					prevY = 0
				}
				if prevY >= chartHeight {
					prevY = chartHeight - 1
				}

				startY, endY := prevY, y
				if startY > endY {
					startY, endY = endY, startY
				}
				for lineY := startY; lineY <= endY; lineY++ {
					if lineY < chartHeight {
						if grid[lineY][x] == ' ' {
							grid[lineY][x] = '│'
						}
					}
				}
			}
		}
	}

	// Render the grid
	for _, row := range grid {
		result.WriteString(ChartBarStyle.Render(string(row)))
		result.WriteString("\n")
	}

	// Add current value
	if len(c.data) > 0 {
		lastValue := c.data[len(c.data)-1]
		result.WriteString(fmt.Sprintf("Current: %.1f", lastValue))
	}

	return result.String()
}
