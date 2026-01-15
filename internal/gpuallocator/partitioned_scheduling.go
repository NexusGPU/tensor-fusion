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

package gpuallocator

import (
	"fmt"
	"math"
	"sort"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"k8s.io/apimachinery/pkg/api/resource"
)

const DefaultMaxPartitionNum = 32
const PartitionMatchingComputingWeight = 0.6
const PartitionMatchingVRAMWeight = 0.4

// PartitionMatchResult represents the result of matching a partition template to a request
type PartitionMatchResult struct {
	Template    *config.PartitionTemplateInfo // Template info from config
	TemplateID  string                        // Template ID
	Score       float64                       // Lower score means better match (less waste)
	CanAllocate bool
	Reason      string
}

// MatchPartitionTemplate matches a partition template to an allocation request.
// Gets template info from config (PartitionTemplateMap) based on GPU model.
// In partitioned mode, we find the smallest template that can satisfy the request.
func MatchPartitionTemplate(gpuStatus tfv1.GPUStatus, req *tfv1.AllocRequest) (*PartitionMatchResult, error) {
	gpuModel := gpuStatus.GPUModel

	// Get template configs from global map
	templateConfigs, exists := PartitionTemplateMap[gpuModel]
	if !exists || len(templateConfigs) == 0 {
		return nil, fmt.Errorf("no partition template configs found for GPU model %s", gpuModel)
	}

	if req.PartitionTemplateID != "" {
		templateInfo, exists := templateConfigs[req.PartitionTemplateID]
		if !exists {
			return nil, fmt.Errorf("specified partition template %s not found for GPU model %s", req.PartitionTemplateID, gpuModel)
		}
		return &PartitionMatchResult{
			Template:    &templateInfo,
			TemplateID:  req.PartitionTemplateID,
			CanAllocate: true,
			Reason:      "partition template is specified in request",
		}, nil
	}

	// Convert request to comparable values
	// Handle ComputePercent: convert to TFLOPs if specified
	var requestTflops float64
	if !req.Request.ComputePercent.IsZero() {
		// Get GPU capacity from global map to convert ComputePercent to TFLOPs
		mu.Lock()
		gpuCapacity, exists := GPUCapacityMap[gpuModel]
		mu.Unlock()
		if !exists {
			return nil, fmt.Errorf("GPU capacity not found for model %s, cannot convert ComputePercent to TFLOPs", gpuModel)
		}
		requiredTflops := utils.ComputePercentToTflops(gpuCapacity.Tflops, req.Request)
		requestTflops = requiredTflops.AsApproximateFloat64()
	} else {
		requestTflops = req.Request.Tflops.AsApproximateFloat64()
	}
	requestVramBytes := req.Request.Vram.Value()

	// Get max partitions from config
	maxPartitions := MaxPartitionsMap[gpuModel]
	if maxPartitions <= 0 {
		maxPartitions = DefaultMaxPartitionNum
	}

	// Find the best matching template
	var bestMatch *PartitionMatchResult
	bestScore := math.MaxFloat64 // Lower is better (we want smallest that fits)

	for templateID, templateInfo := range templateConfigs {
		// If a specific template is required, only consider that one
		if req.PartitionTemplateID != "" && templateID != req.PartitionTemplateID {
			continue
		}

		templateInfoCopy := templateInfo // Create a copy to avoid pointer issues
		result := &PartitionMatchResult{
			Template:    &templateInfoCopy,
			TemplateID:  templateID,
			CanAllocate: false,
		}

		// Check if template resources can satisfy the request
		templateTflops := templateInfo.ComputePercent * gpuStatus.Capacity.Tflops.AsApproximateFloat64()
		templateVramBytes := int64(templateInfo.MemoryGigabytes * 1024 * 1024 * 1024)

		// Check if template has enough resources
		if templateTflops < requestTflops {
			result.Reason = fmt.Sprintf("template %s has insufficient TFLOPs: %.2f < %.2f",
				templateID, templateTflops, requestTflops)
			continue
		}

		if templateVramBytes < requestVramBytes {
			result.Reason = fmt.Sprintf("template %s has insufficient VRAM: %d < %d",
				templateID, templateVramBytes, requestVramBytes)
			continue
		}

		// Check if we can allocate more partitions (MIG constraint)
		currentPartitionCount := len(gpuStatus.AllocatedPartitions)
		if maxPartitions > 0 && uint32(currentPartitionCount) >= maxPartitions {
			result.Reason = fmt.Sprintf("GPU has reached maximum partition count: %d/%d",
				currentPartitionCount, maxPartitions)
			continue
		}

		// Calculate score: prefer templates that are just large enough (minimize waste)
		tflopsWaste := (templateTflops - requestTflops) / math.Max(requestTflops, 1.0)
		vramWaste := float64(templateVramBytes-requestVramBytes) / math.Max(float64(requestVramBytes), 1.0)
		score := tflopsWaste*PartitionMatchingComputingWeight + vramWaste*PartitionMatchingVRAMWeight

		result.Score = score
		result.CanAllocate = true
		result.Reason = "template can satisfy request"

		// Update best match if this is better
		if bestMatch == nil || score < bestScore {
			bestMatch = result
			bestScore = score
		}
	}

	if bestMatch == nil {
		return nil, fmt.Errorf("no partition template can satisfy request: TFLOPs=%.2f, VRAM=%d",
			requestTflops, requestVramBytes)
	}

	return bestMatch, nil
}

// CalculatePartitionResourceUsage calculates the resource usage for a partition template.
// Gets template info from config.
func CalculatePartitionResourceUsage(capacityTflops resource.Quantity, gpuModel, templateID string) (tflops resource.Quantity, vram resource.Quantity, err error) {
	templateConfigs, exists := PartitionTemplateMap[gpuModel]
	if !exists {
		return resource.Quantity{}, resource.Quantity{}, fmt.Errorf("no partition template configs for GPU model %s", gpuModel)
	}

	templateInfo, exists := templateConfigs[templateID]
	if !exists {
		return resource.Quantity{}, resource.Quantity{}, fmt.Errorf("partition template %s not found for GPU model %s", templateID, gpuModel)
	}

	tflops = resource.MustParse(fmt.Sprintf("%.2f", templateInfo.ComputePercent*capacityTflops.AsApproximateFloat64()/100.0))
	vram = resource.MustParse(fmt.Sprintf("%dGi", templateInfo.MemoryGigabytes))

	return tflops, vram, nil
}

// areSlotsFree checks if slots starting from startPos for offset slots are all free.
func areSlotsFree(occupiedSlots map[uint32]bool, startPos, offset uint32) bool {
	for i := range offset {
		if occupiedSlots[startPos+i] {
			return false
		}
	}
	return true
}

// buildSlotOccupancyMap builds a map of occupied slots from existing partitions.
// Uses AllocatedSlotStart/End if available, otherwise falls back to greedy assignment.
func buildSlotOccupancyMap(
	gpu *tfv1.GPU,
	templateConfigs map[string]config.PartitionTemplateInfo,
) map[uint32]bool {
	occupiedSlots := make(map[uint32]bool)

	// First, use explicit slot assignments if available
	for _, partition := range gpu.Status.AllocatedPartitions {
		if partition.AllocatedSlotStart != nil && partition.AllocatedSlotEnd != nil {
			start := *partition.AllocatedSlotStart
			end := *partition.AllocatedSlotEnd
			for slot := start; slot < end; slot++ {
				occupiedSlots[slot] = true
			}
		}
	}

	// For partitions without explicit slot assignments, use greedy approach
	// Convert map to slice and sort by AllocatedAt timestamp (ASC)
	partitions := make([]tfv1.AllocatedPartition, 0, len(gpu.Status.AllocatedPartitions))
	for _, partition := range gpu.Status.AllocatedPartitions {
		// Skip if already has explicit slot assignment
		if partition.AllocatedSlotStart != nil && partition.AllocatedSlotEnd != nil {
			continue
		}
		partitions = append(partitions, partition)
	}

	if len(partitions) > 0 {
		sort.Slice(partitions, func(i, j int) bool {
			// If both have valid timestamps, compare by time
			if !partitions[i].AllocatedAt.IsZero() && !partitions[j].AllocatedAt.IsZero() {
				if !partitions[i].AllocatedAt.Equal(&partitions[j].AllocatedAt) {
					return partitions[i].AllocatedAt.Before(&partitions[j].AllocatedAt)
				}
			}
			// Fallback to PodUID for stable ordering when timestamps are zero or equal
			return partitions[i].PodUID < partitions[j].PodUID
		})

		// Process each partition without explicit slots in allocation order
		for _, partition := range partitions {
			templateInfo, exists := templateConfigs[partition.TemplateID]
			if !exists || len(templateInfo.PlacementLimit) == 0 || templateInfo.PlacementOffSet == 0 {
				continue
			}

			// Find first available starting position for this partition
			for _, startPos := range templateInfo.PlacementLimit {
				if areSlotsFree(occupiedSlots, startPos, templateInfo.PlacementOffSet) {
					// Assign this partition to this position
					for i := uint32(0); i < templateInfo.PlacementOffSet; i++ {
						occupiedSlots[startPos+i] = true
					}
					break
				}
			}
		}
	}

	return occupiedSlots
}

// findAvailableSlotPosition finds the first available slot position for a template.
// Returns the starting position and true if found, 0 and false otherwise.
func findAvailableSlotPosition(
	templateInfo config.PartitionTemplateInfo,
	occupiedSlots map[uint32]bool,
) (uint32, bool) {
	if len(templateInfo.PlacementLimit) == 0 || templateInfo.PlacementOffSet == 0 {
		return 0, false
	}

	for _, startPos := range templateInfo.PlacementLimit {
		if areSlotsFree(occupiedSlots, startPos, templateInfo.PlacementOffSet) {
			return startPos, true
		}
	}

	return 0, false
}

// CheckPartitionAvailability checks if a GPU has enough resources to allocate a partition.
// Gets template info from config. Uses vendor-specific strategy for slot/group checking.
func CheckPartitionAvailability(
	gpu *tfv1.GPU,
	templateID string,
) error {
	// Get template info from config first to check template-specific constraints
	templateConfigs, exists := PartitionTemplateMap[gpu.Status.GPUModel]
	if !exists {
		return fmt.Errorf("no partition template configs for GPU model %s", gpu.Status.GPUModel)
	}

	templateInfo, exists := templateConfigs[templateID]
	if !exists {
		return fmt.Errorf("partition template %s not found for GPU model %s", templateID, gpu.Status.GPUModel)
	}

	// Get GPU config for strategy
	gpuConfig := getGpuConfigFromMaps(gpu.Status.GPUModel)

	// Use vendor-specific strategy for slot/group checking
	strategy := GetPartitionStrategy(gpu.Status.Vendor)
	if err := strategy.CheckAvailability(gpu, templateInfo, gpuConfig); err != nil {
		return err
	}

	// Calculate required resources from config
	requiredTflops, requiredVram, err := CalculatePartitionResourceUsage(gpu.Status.Capacity.Tflops, gpu.Status.GPUModel, templateID)
	if err != nil {
		return err
	}

	// Check TFLOPs availability
	if gpu.Status.Available.Tflops.Cmp(requiredTflops) < 0 {
		return fmt.Errorf("GPU %s insufficient TFLOPs for partition: available %s, required %s",
			gpu.Name, gpu.Status.Available.Tflops.String(), requiredTflops.String())
	}

	// Check VRAM availability
	if gpu.Status.Available.Vram.Cmp(requiredVram) < 0 {
		return fmt.Errorf("GPU %s insufficient VRAM for partition: available %s, required %s",
			gpu.Name, gpu.Status.Available.Vram.String(), requiredVram.String())
	}

	return nil
}

// getGpuConfigFromMaps constructs a GpuInfo from the global maps for strategy use
func getGpuConfigFromMaps(gpuModel string) *config.GpuInfo {
	gpuConfig := &config.GpuInfo{
		Model:         gpuModel,
		MaxPartitions: MaxPartitionsMap[gpuModel],
	}

	// Get MaxPlacementSlots
	if slots, exists := MaxPlacementSlotsMap[gpuModel]; exists {
		gpuConfig.MaxPlacementSlots = slots
	}

	// Get MaxIsolationGroups (defaults to MaxPlacementSlots if not set)
	if groups, exists := MaxIsolationGroupsMap[gpuModel]; exists {
		gpuConfig.MaxIsolationGroups = groups
	} else if gpuConfig.MaxPlacementSlots > 0 {
		gpuConfig.MaxIsolationGroups = gpuConfig.MaxPlacementSlots
	}

	// Get TotalExtendedResources
	if resources, exists := TotalExtendedResourcesMap[gpuModel]; exists {
		gpuConfig.TotalExtendedResources = resources
	}

	// Convert template map to slice
	if templates, exists := PartitionTemplateMap[gpuModel]; exists {
		for _, t := range templates {
			gpuConfig.PartitionTemplates = append(gpuConfig.PartitionTemplates, t)
		}
	}

	return gpuConfig
}
