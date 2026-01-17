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

package filter

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/samber/lo"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// PartitionTemplateFilter filters GPUs based on partition template availability
// Only applies when isolation mode is partitioned
type PartitionTemplateFilter struct {
	isolationMode        tfv1.IsolationModeType
	requiredTemplateID   string
	maxPartitionsMap     map[string]uint32                                  // GPU model -> max partitions
	partitionTemplateMap map[string]map[string]config.PartitionTemplateInfo // GPU model -> templateID -> template info
}

// NewPartitionTemplateFilter creates a new PartitionTemplateFilter
func NewPartitionTemplateFilter(
	isolationMode tfv1.IsolationModeType,
	requiredTemplateID string,
	maxPartitionsMap map[string]uint32,
	partitionTemplateMap map[string]map[string]config.PartitionTemplateInfo,
) *PartitionTemplateFilter {
	return &PartitionTemplateFilter{
		isolationMode:        isolationMode,
		requiredTemplateID:   requiredTemplateID,
		maxPartitionsMap:     maxPartitionsMap,
		partitionTemplateMap: partitionTemplateMap,
	}
}

// Filter implements GPUFilter.Filter
func (f *PartitionTemplateFilter) Filter(ctx context.Context, workerPodKey tfv1.NameNamespace, gpus []*tfv1.GPU) ([]*tfv1.GPU, error) {
	// Only apply filter for partitioned isolation mode
	if f.isolationMode != tfv1.IsolationModePartitioned {
		return gpus, nil
	}

	logger := log.FromContext(ctx)

	return lo.Filter(gpus, func(gpu *tfv1.GPU, _ int) bool {
		// If a specific template ID is required, check if GPU model has it in config
		if f.requiredTemplateID != "" {
			templateConfigs, hasTemplates := f.partitionTemplateMap[gpu.Status.GPUModel]
			if !hasTemplates {
				logger.V(5).Info("GPU model has no partition templates configured",
					"gpu", gpu.Name, "model", gpu.Status.GPUModel)
				return false
			}
			if _, hasTemplate := templateConfigs[f.requiredTemplateID]; !hasTemplate {
				logger.V(5).Info("GPU model does not have required partition template",
					"gpu", gpu.Name, "model", gpu.Status.GPUModel, "template", f.requiredTemplateID)
				return false
			}
		}

		// Check partition count limit
		allocatedCount := 0
		if gpu.Status.AllocatedPartitions != nil {
			allocatedCount = len(gpu.Status.AllocatedPartitions)
		}

		// Get max partitions from config
		maxPartitions := f.maxPartitionsMap[gpu.Status.GPUModel]
		if maxPartitions == 0 {
			// Default to 7 for MIG if not configured
			maxPartitions = 7
		}

		if maxPartitions > 0 && uint32(allocatedCount) >= maxPartitions {
			logger.V(5).Info("GPU has reached maximum partition count",
				"gpu", gpu.Name, "allocated", allocatedCount, "max", maxPartitions)
			return false
		}

		return true
	}), nil
}

func (f *PartitionTemplateFilter) Name() string {
	return "PartitionTemplateFilter"
}
