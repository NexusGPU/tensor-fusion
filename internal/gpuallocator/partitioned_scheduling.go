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

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"k8s.io/apimachinery/pkg/api/resource"
)

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
func MatchPartitionTemplate(
	gpuModel string,
	gpuTemplates []tfv1.PartitionTemplate, // Only has TemplateID and Name
	req *tfv1.AllocRequest,
	allocatedPartitions map[string]tfv1.AllocatedPartition,
) (*PartitionMatchResult, error) {
	if len(gpuTemplates) == 0 {
		return nil, fmt.Errorf("no partition templates available for GPU model %s", gpuModel)
	}

	// Get template configs from global map
	templateConfigs, exists := PartitionTemplateMap[gpuModel]
	if !exists || len(templateConfigs) == 0 {
		return nil, fmt.Errorf("no partition template configs found for GPU model %s", gpuModel)
	}

	// Convert request to comparable values
	requestTflops := req.Request.Tflops.AsApproximateFloat64()
	requestVramBytes := req.Request.Vram.Value()

	// Get max partitions from config
	maxPartitions := MaxPartitionsMap[gpuModel]
	if maxPartitions == 0 {
		maxPartitions = 7 // Default MIG limit
	}

	// Find the best matching template
	var bestMatch *PartitionMatchResult
	bestScore := math.MaxFloat64 // Lower is better (we want smallest that fits)

	for _, gpuTemplate := range gpuTemplates {
		// Get detailed template info from config
		templateInfo, exists := templateConfigs[gpuTemplate.TemplateID]
		if !exists {
			continue // Skip if template not found in config
		}

		// If a specific template is required, only consider that one
		if req.PartitionTemplateID != "" && gpuTemplate.TemplateID != req.PartitionTemplateID {
			continue
		}

		result := &PartitionMatchResult{
			Template:    &templateInfo,
			TemplateID:  gpuTemplate.TemplateID,
			CanAllocate: false,
		}

		// Check if template resources can satisfy the request
		templateTflops := templateInfo.Tflops
		templateVramBytes := int64(templateInfo.MemoryBytes)

		// Check if template has enough resources
		if templateTflops < requestTflops {
			result.Reason = fmt.Sprintf("template %s has insufficient TFLOPs: %.2f < %.2f",
				gpuTemplate.TemplateID, templateTflops, requestTflops)
			continue
		}

		if templateVramBytes < requestVramBytes {
			result.Reason = fmt.Sprintf("template %s has insufficient VRAM: %d < %d",
				gpuTemplate.TemplateID, templateVramBytes, requestVramBytes)
			continue
		}

		// Check if we can allocate more partitions (MIG constraint)
		currentPartitionCount := len(allocatedPartitions)
		if maxPartitions > 0 && uint32(currentPartitionCount) >= maxPartitions {
			result.Reason = fmt.Sprintf("GPU has reached maximum partition count: %d/%d",
				currentPartitionCount, maxPartitions)
			continue
		}

		// Calculate score: prefer templates that are just large enough (minimize waste)
		tflopsWaste := (templateTflops - requestTflops) / math.Max(requestTflops, 0.1)
		vramWaste := float64(templateVramBytes-requestVramBytes) / math.Max(float64(requestVramBytes), 1.0)
		// Weighted average: TFLOPs waste is more important
		score := tflopsWaste*0.7 + vramWaste*0.3

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
func CalculatePartitionResourceUsage(gpuModel, templateID string) (tflops resource.Quantity, vram resource.Quantity, err error) {
	templateConfigs, exists := PartitionTemplateMap[gpuModel]
	if !exists {
		return resource.Quantity{}, resource.Quantity{}, fmt.Errorf("no partition template configs for GPU model %s", gpuModel)
	}

	templateInfo, exists := templateConfigs[templateID]
	if !exists {
		return resource.Quantity{}, resource.Quantity{}, fmt.Errorf("partition template %s not found for GPU model %s", templateID, gpuModel)
	}

	// TFLOPs: use the template's TFLOPs value
	tflops = resource.MustParse(fmt.Sprintf("%.2f", templateInfo.Tflops))

	// VRAM: template memory (no overhead)
	vram = *resource.NewQuantity(int64(templateInfo.MemoryBytes), resource.BinarySI)

	return tflops, vram, nil
}

// CheckPartitionAvailability checks if a GPU has enough resources to allocate a partition.
// Gets template info from config.
func CheckPartitionAvailability(
	gpu *tfv1.GPU,
	templateID string,
	allocatedPartitions map[string]tfv1.AllocatedPartition,
) error {
	if gpu.Status.Available == nil {
		return fmt.Errorf("GPU %s has nil available resources", gpu.Name)
	}

	// Get max partitions from config
	maxPartitions := MaxPartitionsMap[gpu.Status.GPUModel]
	if maxPartitions == 0 {
		maxPartitions = 7 // Default MIG limit
	}

	// Check partition count limit
	currentCount := len(allocatedPartitions)
	if maxPartitions > 0 && uint32(currentCount) >= maxPartitions {
		return fmt.Errorf("GPU %s has reached maximum partition count: %d/%d",
			gpu.Name, currentCount, maxPartitions)
	}

	// Calculate required resources from config
	requiredTflops, requiredVram, err := CalculatePartitionResourceUsage(gpu.Status.GPUModel, templateID)
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
