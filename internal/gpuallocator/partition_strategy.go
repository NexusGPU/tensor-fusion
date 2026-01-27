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
	"sort"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

// PartitionStrategy defines the interface for vendor-specific partition allocation strategies
type PartitionStrategy interface {
	// Name returns the strategy name (e.g., "nvidia-mig", "ascend-vnpu")
	Name() string

	// CheckAvailability verifies if a partition can be allocated on the given GPU
	// Returns nil if allocation is possible, error otherwise
	CheckAvailability(gpu *tfv1.GPU, templateInfo config.PartitionTemplateInfo, gpuConfig *config.GpuInfo) error

	// AllocateSlot finds and returns allocation position information
	// Returns (isolationGroupID, slotStart, slotEnd, error)
	// - For NVIDIA MIG: slotStart/slotEnd are physical slot positions
	// - For Ascend: isolationGroupID is the vGroup (0-3), slots may be nil
	AllocateSlot(gpu *tfv1.GPU, templateInfo config.PartitionTemplateInfo, gpuConfig *config.GpuInfo) (*uint32, *uint32, *uint32, error)
}

// GetPartitionStrategy returns the appropriate partition strategy based on GPU vendor
func GetPartitionStrategy(vendor string) PartitionStrategy {
	switch vendor {
	case constants.AcceleratorVendorHuaweiAscendNPU:
		return &AscendPartitionStrategy{}
	case constants.AcceleratorVendorNvidia, constants.AcceleratorVendorExample:
		// Default to NVIDIA MIG strategy for backward compatibility
		return &NVIDIAMIGStrategy{}
	default:
		return &NotSupportedPartitionStrategy{}
	}
}

// NotSupportedPartitionStrategy implements PartitionStrategy for unknown vendors
type NotSupportedPartitionStrategy struct{}

func (s *NotSupportedPartitionStrategy) Name() string {
	return "not-supported"
}

func (s *NotSupportedPartitionStrategy) CheckAvailability(gpu *tfv1.GPU, templateInfo config.PartitionTemplateInfo, gpuConfig *config.GpuInfo) error {
	return fmt.Errorf("vendor %s is not supported for GPU partition strategy", gpu.Status.Vendor)
}

func (s *NotSupportedPartitionStrategy) AllocateSlot(gpu *tfv1.GPU, templateInfo config.PartitionTemplateInfo, gpuConfig *config.GpuInfo) (*uint32, *uint32, *uint32, error) {
	return nil, nil, nil, fmt.Errorf("vendor %s is not supported for GPU partition strategy", gpu.Status.Vendor)
}

// NVIDIAMIGStrategy implements PartitionStrategy for NVIDIA MIG
type NVIDIAMIGStrategy struct{}

func (s *NVIDIAMIGStrategy) Name() string {
	return "nvidia-mig"
}

func (s *NVIDIAMIGStrategy) CheckAvailability(gpu *tfv1.GPU, templateInfo config.PartitionTemplateInfo, gpuConfig *config.GpuInfo) error {
	// Check general partition count limit
	currentCount := len(gpu.Status.AllocatedPartitions)
	maxPartitions := gpuConfig.MaxPartitions
	if maxPartitions == 0 {
		maxPartitions = 7 // Default MIG limit
	}
	if uint32(currentCount) >= maxPartitions {
		return fmt.Errorf("GPU %s has reached maximum partition count: %d/%d",
			gpu.Name, currentCount, maxPartitions)
	}

	// Check template-specific MaxPartition limit
	templateCount := uint32(0)
	for _, partition := range gpu.Status.AllocatedPartitions {
		if partition.TemplateID == templateInfo.TemplateID {
			templateCount++
		}
	}
	if templateInfo.MaxPartition > 0 && templateCount >= templateInfo.MaxPartition {
		return fmt.Errorf("GPU %s has reached maximum partition count for template %s: %d/%d",
			gpu.Name, templateInfo.TemplateID, templateCount, templateInfo.MaxPartition)
	}

	// Check placement slots
	if len(templateInfo.PlacementLimit) > 0 && templateInfo.PlacementOffSet > 0 {
		occupiedSlots := s.buildSlotOccupancyMap(gpu, gpuConfig)
		_, found := s.findAvailableSlotPosition(templateInfo, occupiedSlots)
		if !found {
			return fmt.Errorf("GPU %s has no available placement slots for template %s",
				gpu.Name, templateInfo.TemplateID)
		}
	}

	return nil
}

func (s *NVIDIAMIGStrategy) AllocateSlot(gpu *tfv1.GPU, templateInfo config.PartitionTemplateInfo, gpuConfig *config.GpuInfo) (*uint32, *uint32, *uint32, error) {
	if len(templateInfo.PlacementLimit) == 0 || templateInfo.PlacementOffSet == 0 {
		return nil, nil, nil, nil
	}

	occupiedSlots := s.buildSlotOccupancyMap(gpu, gpuConfig)
	startPos, found := s.findAvailableSlotPosition(templateInfo, occupiedSlots)
	if !found {
		return nil, nil, nil, fmt.Errorf("no available slot position for template %s", templateInfo.TemplateID)
	}

	endPos := startPos + templateInfo.PlacementOffSet
	return nil, &startPos, &endPos, nil
}

func (s *NVIDIAMIGStrategy) buildSlotOccupancyMap(gpu *tfv1.GPU, gpuConfig *config.GpuInfo) map[uint32]bool {
	occupiedSlots := make(map[uint32]bool)

	templateConfigs := make(map[string]config.PartitionTemplateInfo)
	for _, t := range gpuConfig.PartitionTemplates {
		// Index by both TemplateID and Name for flexible lookup
		templateConfigs[t.TemplateID] = t
		if t.Name != "" && t.Name != t.TemplateID {
			templateConfigs[t.Name] = t
		}
	}

	// Use explicit slot assignments if available
	for _, partition := range gpu.Status.AllocatedPartitions {
		if partition.AllocatedSlotStart != nil && partition.AllocatedSlotEnd != nil {
			for slot := *partition.AllocatedSlotStart; slot < *partition.AllocatedSlotEnd; slot++ {
				occupiedSlots[slot] = true
			}
		}
	}

	// Collect partitions without explicit slots and sort by timestamp
	partitionsToAssign := make([]tfv1.AllocatedPartition, 0)
	for _, partition := range gpu.Status.AllocatedPartitions {
		if partition.AllocatedSlotStart != nil && partition.AllocatedSlotEnd != nil {
			continue // Already processed
		}
		partitionsToAssign = append(partitionsToAssign, partition)
	}

	// Sort by AllocatedAt timestamp (ASC), fallback to PodUID for stable ordering
	sort.Slice(partitionsToAssign, func(i, j int) bool {
		if !partitionsToAssign[i].AllocatedAt.IsZero() && !partitionsToAssign[j].AllocatedAt.IsZero() {
			if !partitionsToAssign[i].AllocatedAt.Equal(&partitionsToAssign[j].AllocatedAt) {
				return partitionsToAssign[i].AllocatedAt.Before(&partitionsToAssign[j].AllocatedAt)
			}
		}
		return partitionsToAssign[i].PodUID < partitionsToAssign[j].PodUID
	})

	// Assign slots to partitions in order
	for _, partition := range partitionsToAssign {
		templateInfo, exists := templateConfigs[partition.TemplateID]
		if !exists || len(templateInfo.PlacementLimit) == 0 || templateInfo.PlacementOffSet == 0 {
			continue
		}

		// Find first available position
		for _, startPos := range templateInfo.PlacementLimit {
			allFree := true
			for i := uint32(0); i < templateInfo.PlacementOffSet; i++ {
				if occupiedSlots[startPos+i] {
					allFree = false
					break
				}
			}
			if allFree {
				for i := uint32(0); i < templateInfo.PlacementOffSet; i++ {
					occupiedSlots[startPos+i] = true
				}
				break
			}
		}
	}

	return occupiedSlots
}

func (s *NVIDIAMIGStrategy) findAvailableSlotPosition(templateInfo config.PartitionTemplateInfo, occupiedSlots map[uint32]bool) (uint32, bool) {
	for _, startPos := range templateInfo.PlacementLimit {
		allFree := true
		for i := uint32(0); i < templateInfo.PlacementOffSet; i++ {
			if occupiedSlots[startPos+i] {
				allFree = false
				break
			}
		}
		if allFree {
			return startPos, true
		}
	}
	return 0, false
}

// AscendPartitionStrategy implements PartitionStrategy for Huawei Ascend NPU
// Key concepts:
// - vNPU: virtual NPU created from a template (e.g., vir01, vir04)
// - vGroup: isolation group (0-3), provides hardware isolation between groups
// - Templates sharing same vGroup use time-sharing (soft isolation)
// - Different vGroups provide hard isolation
type AscendPartitionStrategy struct{}

func (s *AscendPartitionStrategy) Name() string {
	return "ascend-vnpu"
}

func (s *AscendPartitionStrategy) CheckAvailability(gpu *tfv1.GPU, templateInfo config.PartitionTemplateInfo, gpuConfig *config.GpuInfo) error {
	// Check global partition limit
	currentCount := len(gpu.Status.AllocatedPartitions)
	maxPartitions := gpuConfig.MaxPartitions
	if maxPartitions == 0 {
		maxPartitions = 7 // Default limit
	}
	if uint32(currentCount) >= maxPartitions {
		return fmt.Errorf("GPU %s has reached maximum partition count: %d/%d",
			gpu.Name, currentCount, maxPartitions)
	}

	// Check template-specific limits
	templateCount := uint32(0)
	for _, partition := range gpu.Status.AllocatedPartitions {
		if partition.TemplateID == templateInfo.TemplateID {
			templateCount++
		}
	}
	if templateInfo.MaxPartition > 0 && templateCount >= templateInfo.MaxPartition {
		return fmt.Errorf("GPU %s has reached maximum partition count for template %s: %d/%d",
			gpu.Name, templateInfo.TemplateID, templateCount, templateInfo.MaxPartition)
	}

	// Check extended resources (AICORE, AICPU, etc.)
	if err := s.checkExtendedResources(gpu, templateInfo, gpuConfig); err != nil {
		return err
	}

	// Check isolation group (vGroup) availability
	if err := s.checkIsolationGroupAvailability(gpu, templateInfo, gpuConfig); err != nil {
		return err
	}

	return nil
}

func (s *AscendPartitionStrategy) AllocateSlot(gpu *tfv1.GPU, templateInfo config.PartitionTemplateInfo, gpuConfig *config.GpuInfo) (*uint32, *uint32, *uint32, error) {
	// Find available isolation group (vGroup)
	groupID, err := s.findAvailableIsolationGroup(gpu, templateInfo, gpuConfig)
	if err != nil {
		return nil, nil, nil, err
	}

	return &groupID, nil, nil, nil
}

// checkExtendedResources verifies that the GPU has enough extended resources
func (s *AscendPartitionStrategy) checkExtendedResources(gpu *tfv1.GPU, templateInfo config.PartitionTemplateInfo, gpuConfig *config.GpuInfo) error {
	if len(templateInfo.ExtendedResources) == 0 || len(gpuConfig.TotalExtendedResources) == 0 {
		return nil // No extended resources to check
	}

	// Calculate currently used resources
	usedResources := make(map[string]uint32)
	templateConfigs := make(map[string]config.PartitionTemplateInfo)
	for _, t := range gpuConfig.PartitionTemplates {
		templateConfigs[t.TemplateID] = t
	}

	for _, partition := range gpu.Status.AllocatedPartitions {
		if allocatedTemplate, exists := templateConfigs[partition.TemplateID]; exists {
			for resource, amount := range allocatedTemplate.ExtendedResources {
				usedResources[resource] += amount
			}
		}
	}

	// Check if new allocation would exceed capacity
	for resource, required := range templateInfo.ExtendedResources {
		total, hasTotal := gpuConfig.TotalExtendedResources[resource]
		if !hasTotal {
			continue // Resource not tracked at GPU level
		}
		used := usedResources[resource]
		if used+required > total {
			return fmt.Errorf("GPU %s insufficient %s: used %d + required %d > total %d",
				gpu.Name, resource, used, required, total)
		}
	}

	return nil
}

// checkIsolationGroupAvailability verifies that there's an available isolation group
func (s *AscendPartitionStrategy) checkIsolationGroupAvailability(gpu *tfv1.GPU, templateInfo config.PartitionTemplateInfo, gpuConfig *config.GpuInfo) error {
	_, err := s.findAvailableIsolationGroup(gpu, templateInfo, gpuConfig)
	return err
}

// findAvailableIsolationGroup finds a suitable vGroup for the partition
// Ascend rules:
// - Max 4 vGroups (0-3)
// - Each vGroup requires at least 2 AI Cores
// - vir01 can share a vGroup (2 per group, time-sharing)
// - vir02, vir04 require exclusive vGroup
func (s *AscendPartitionStrategy) findAvailableIsolationGroup(gpu *tfv1.GPU, templateInfo config.PartitionTemplateInfo, gpuConfig *config.GpuInfo) (uint32, error) {
	maxGroups := gpuConfig.MaxIsolationGroups
	if maxGroups == 0 {
		maxGroups = 4 // Ascend default
	}

	// Build vGroup occupancy map
	groupOccupancy := s.buildGroupOccupancyMap(gpu, gpuConfig)

	// Determine sharing mode
	isShared := templateInfo.IsolationGroupSharing == config.IsolationGroupSharingShared
	maxPerGroup := templateInfo.MaxPartitionsPerIsolationGroup
	if maxPerGroup == 0 && isShared {
		maxPerGroup = 2 // Default for shared mode (e.g., vir01)
	}

	// For shared templates, first try to find an existing group with same template
	if isShared {
		for groupID := uint32(0); groupID < maxGroups; groupID++ {
			occupancy := groupOccupancy[groupID]
			// Check if group has only partitions of same template and not at capacity
			if occupancy.totalCount > 0 && occupancy.templateCounts[templateInfo.TemplateID] == occupancy.totalCount {
				if maxPerGroup == 0 || uint32(occupancy.totalCount) < maxPerGroup {
					return groupID, nil
				}
			}
		}
	}

	// Find a free group
	for groupID := uint32(0); groupID < maxGroups; groupID++ {
		if groupOccupancy[groupID].totalCount == 0 {
			return groupID, nil
		}
	}

	return 0, fmt.Errorf("no available isolation group (vGroup) for template %s on GPU %s: all %d groups are occupied",
		templateInfo.TemplateID, gpu.Name, maxGroups)
}

// groupOccupancy tracks what's allocated in each isolation group
type groupOccupancy struct {
	totalCount     int
	templateCounts map[string]int
}

// buildGroupOccupancyMap builds a map of isolation group occupancy
func (s *AscendPartitionStrategy) buildGroupOccupancyMap(gpu *tfv1.GPU, gpuConfig *config.GpuInfo) map[uint32]groupOccupancy {
	result := make(map[uint32]groupOccupancy)

	maxGroups := gpuConfig.MaxIsolationGroups
	if maxGroups == 0 {
		maxGroups = 4
	}

	// Initialize all groups
	for i := uint32(0); i < maxGroups; i++ {
		result[i] = groupOccupancy{
			templateCounts: make(map[string]int),
		}
	}

	// Process allocated partitions
	for _, partition := range gpu.Status.AllocatedPartitions {
		groupID := uint32(0)
		if partition.IsolationGroupID != nil {
			groupID = *partition.IsolationGroupID
		}

		if groupID >= maxGroups {
			continue // Invalid group ID
		}

		occ := result[groupID]
		occ.totalCount++
		occ.templateCounts[partition.TemplateID]++
		result[groupID] = occ
	}

	return result
}

// CalculateAscendExtendedResourceUsage calculates total extended resource usage for a template
func CalculateAscendExtendedResourceUsage(gpuConfig *config.GpuInfo, templateID string) (map[string]uint32, error) {
	for _, template := range gpuConfig.PartitionTemplates {
		if template.TemplateID == templateID || template.Name == templateID {
			return template.ExtendedResources, nil
		}
	}
	return nil, fmt.Errorf("template %s not found", templateID)
}
