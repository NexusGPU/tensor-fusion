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
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPartitionTemplateFilter(t *testing.T) {
	testPodKey := tfv1.NameNamespace{
		Name:      "test-pod",
		Namespace: "test-namespace",
	}

	// Setup partition template map (global config)
	partitionTemplateMap := map[string]map[string]config.PartitionTemplateInfo{
		"A100": {
			"1g.24gb": {TemplateID: "1g.24gb", Name: "1g.24gb", MemoryGigabytes: 24, ComputePercent: 14.28},
			"4g.94gb": {TemplateID: "4g.94gb", Name: "4g.94gb", MemoryGigabytes: 94, ComputePercent: 57.14},
		},
	}

	tests := []struct {
		name                 string
		isolationMode        tfv1.IsolationModeType
		requiredTemplate     string
		maxPartitionsMap     map[string]uint32
		partitionTemplateMap map[string]map[string]config.PartitionTemplateInfo
		gpus                 []*tfv1.GPU
		expectedCount        int
		expectedGPUNames     []string
	}{
		{
			name:                 "non-partitioned mode should pass all GPUs",
			isolationMode:        tfv1.IsolationModeSoft,
			requiredTemplate:     "",
			maxPartitionsMap:     map[string]uint32{},
			partitionTemplateMap: partitionTemplateMap,
			gpus: []*tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						GPUModel: "A100",
					},
				},
			},
			expectedCount:    1,
			expectedGPUNames: []string{"gpu-1"},
		},
		{
			name:                 "partitioned mode - GPU model without templates in config filtered out",
			isolationMode:        tfv1.IsolationModePartitioned,
			requiredTemplate:     "1g.24gb",
			maxPartitionsMap:     map[string]uint32{"A100": 7},
			partitionTemplateMap: partitionTemplateMap,
			gpus: []*tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						GPUModel: "H100", // Not in partitionTemplateMap
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-2"},
					Status: tfv1.GPUStatus{
						GPUModel: "A100", // Has templates in config
					},
				},
			},
			expectedCount:    1,
			expectedGPUNames: []string{"gpu-2"},
		},
		{
			name:                 "partitioned mode - specific template required",
			isolationMode:        tfv1.IsolationModePartitioned,
			requiredTemplate:     "1g.24gb",
			maxPartitionsMap:     map[string]uint32{"A100": 7},
			partitionTemplateMap: partitionTemplateMap,
			gpus: []*tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						GPUModel: "A100", // Has templates but checking if specific one exists
					},
				},
			},
			expectedCount:    1,
			expectedGPUNames: []string{"gpu-1"},
		},
		{
			name:             "partitioned mode - required template not in config",
			isolationMode:    tfv1.IsolationModePartitioned,
			requiredTemplate: "2g.48gb", // Not in config for A100
			maxPartitionsMap: map[string]uint32{"A100": 7},
			partitionTemplateMap: map[string]map[string]config.PartitionTemplateInfo{
				"A100": {
					"1g.24gb": {TemplateID: "1g.24gb", Name: "1g.24gb"},
				},
			},
			gpus: []*tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						GPUModel: "A100",
					},
				},
			},
			expectedCount:    0,
			expectedGPUNames: []string{},
		},
		{
			name:                 "partitioned mode - max partitions reached",
			isolationMode:        tfv1.IsolationModePartitioned,
			requiredTemplate:     "",
			maxPartitionsMap:     map[string]uint32{"A100": 7},
			partitionTemplateMap: partitionTemplateMap,
			gpus: []*tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						GPUModel: "A100",
						AllocatedPartitions: map[string]tfv1.AllocatedPartition{
							"pod-1": {TemplateID: "1g.24gb", PodUID: "pod-1"},
							"pod-2": {TemplateID: "1g.24gb", PodUID: "pod-2"},
							"pod-3": {TemplateID: "1g.24gb", PodUID: "pod-3"},
							"pod-4": {TemplateID: "1g.24gb", PodUID: "pod-4"},
							"pod-5": {TemplateID: "1g.24gb", PodUID: "pod-5"},
							"pod-6": {TemplateID: "1g.24gb", PodUID: "pod-6"},
							"pod-7": {TemplateID: "1g.24gb", PodUID: "pod-7"},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-2"},
					Status: tfv1.GPUStatus{
						GPUModel: "A100",
						AllocatedPartitions: map[string]tfv1.AllocatedPartition{
							"pod-1": {TemplateID: "1g.24gb", PodUID: "pod-1"},
						},
					},
				},
			},
			expectedCount:    1,
			expectedGPUNames: []string{"gpu-2"},
		},
	}

	ctx := context.Background()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			filter := NewPartitionTemplateFilter(tt.isolationMode, tt.requiredTemplate, tt.maxPartitionsMap, tt.partitionTemplateMap)
			result, err := filter.Filter(ctx, testPodKey, tt.gpus)

			assert.NoError(t, err)
			assert.Len(t, result, tt.expectedCount)
			if len(tt.expectedGPUNames) > 0 {
				resultNames := make([]string, len(result))
				for i, gpu := range result {
					resultNames[i] = gpu.Name
				}
				assert.ElementsMatch(t, tt.expectedGPUNames, resultNames)
			}
		})
	}
}
