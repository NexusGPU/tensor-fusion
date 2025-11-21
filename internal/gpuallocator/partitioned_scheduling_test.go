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
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const testGPUModel = "A100_SXM_80G"

func TestMatchPartitionTemplate(t *testing.T) {
	// Setup: Initialize partition template map
	gpuModel := testGPUModel
	PartitionTemplateMap[gpuModel] = map[string]config.PartitionTemplateInfo{
		"1g.24gb": {
			TemplateID:   "1g.24gb",
			Name:         "1g.24gb",
			MemoryBytes:  24 * 1024 * 1024 * 1024, // 24GB
			Tflops:       50.0,
			ComputeUnits: 14,
			SliceCount:   7,
		},
		"4g.94gb": {
			TemplateID:   "4g.94gb",
			Name:         "4g.94gb",
			MemoryBytes:  94 * 1024 * 1024 * 1024, // 94GB
			Tflops:       200.0,
			ComputeUnits: 56,
			SliceCount:   7,
		},
	}
	// Setup: Initialize GPU capacity map for ComputePercent conversion
	// A100_SXM_80G has ~312 TFLOPs capacity
	mu.Lock()
	GPUCapacityMap[gpuModel] = tfv1.Resource{
		Tflops: resource.MustParse("312"),
		Vram:   resource.MustParse("80Gi"),
	}
	mu.Unlock()

	tests := []struct {
		name                string
		gpuTemplates        []tfv1.PartitionTemplate
		req                 *tfv1.AllocRequest
		allocatedPartitions map[string]tfv1.AllocatedPartition
		expectError         bool
		expectedTemplateID  string
	}{
		{
			name: "match smallest template that fits",
			gpuTemplates: []tfv1.PartitionTemplate{
				{TemplateID: "1g.24gb", Name: "1g.24gb"},
				{TemplateID: "4g.94gb", Name: "4g.94gb"},
			},
			req: &tfv1.AllocRequest{
				Request: tfv1.Resource{
					Tflops: resource.MustParse("30"),
					Vram:   resource.MustParse("20Gi"),
				},
			},
			allocatedPartitions: map[string]tfv1.AllocatedPartition{},
			expectError:         false,
			expectedTemplateID:  "1g.24gb", // Should match smallest that fits
		},
		{
			name: "match specific template when required",
			gpuTemplates: []tfv1.PartitionTemplate{
				{TemplateID: "1g.24gb", Name: "1g.24gb"},
				{TemplateID: "4g.94gb", Name: "4g.94gb"},
			},
			req: &tfv1.AllocRequest{
				Request: tfv1.Resource{
					Tflops: resource.MustParse("30"),
					Vram:   resource.MustParse("20Gi"),
				},
				PartitionTemplateID: "4g.94gb",
			},
			allocatedPartitions: map[string]tfv1.AllocatedPartition{},
			expectError:         false,
			expectedTemplateID:  "4g.94gb",
		},
		{
			name: "no template matches request",
			gpuTemplates: []tfv1.PartitionTemplate{
				{TemplateID: "1g.24gb", Name: "1g.24gb"},
			},
			req: &tfv1.AllocRequest{
				Request: tfv1.Resource{
					Tflops: resource.MustParse("300"), // Too large
					Vram:   resource.MustParse("100Gi"),
				},
			},
			allocatedPartitions: map[string]tfv1.AllocatedPartition{},
			expectError:         true,
		},
		{
			name:         "no templates available",
			gpuTemplates: []tfv1.PartitionTemplate{},
			req: &tfv1.AllocRequest{
				Request: tfv1.Resource{
					Tflops: resource.MustParse("30"),
					Vram:   resource.MustParse("20Gi"),
				},
			},
			allocatedPartitions: map[string]tfv1.AllocatedPartition{},
			expectError:         true,
		},
		{
			name: "match with ComputePercent - smallest template that fits",
			gpuTemplates: []tfv1.PartitionTemplate{
				{TemplateID: "1g.24gb", Name: "1g.24gb"},
				{TemplateID: "4g.94gb", Name: "4g.94gb"},
			},
			req: &tfv1.AllocRequest{
				Request: tfv1.Resource{
					// 10% of 312 TFLOPs = 31.2 TFLOPs, should match 1g.24gb (50 TFLOPs)
					ComputePercent: resource.MustParse("10"),
					Vram:           resource.MustParse("20Gi"),
				},
			},
			allocatedPartitions: map[string]tfv1.AllocatedPartition{},
			expectError:         false,
			expectedTemplateID:  "1g.24gb",
		},
		{
			name: "match with ComputePercent - requires larger template",
			gpuTemplates: []tfv1.PartitionTemplate{
				{TemplateID: "1g.24gb", Name: "1g.24gb"},
				{TemplateID: "4g.94gb", Name: "4g.94gb"},
			},
			req: &tfv1.AllocRequest{
				Request: tfv1.Resource{
					// 50% of 312 TFLOPs = 156 TFLOPs, should match 4g.94gb (200 TFLOPs)
					ComputePercent: resource.MustParse("50"),
					Vram:           resource.MustParse("50Gi"),
				},
			},
			allocatedPartitions: map[string]tfv1.AllocatedPartition{},
			expectError:         false,
			expectedTemplateID:  "4g.94gb",
		},
		{
			name: "match with ComputePercent - no template matches",
			gpuTemplates: []tfv1.PartitionTemplate{
				{TemplateID: "1g.24gb", Name: "1g.24gb"},
			},
			req: &tfv1.AllocRequest{
				Request: tfv1.Resource{
					// 80% of 312 TFLOPs = 249.6 TFLOPs, too large for 1g.24gb (50 TFLOPs)
					ComputePercent: resource.MustParse("80"),
					Vram:           resource.MustParse("100Gi"),
				},
			},
			allocatedPartitions: map[string]tfv1.AllocatedPartition{},
			expectError:         true,
		},
		{
			name: "match with ComputePercent - missing GPU capacity",
			gpuTemplates: []tfv1.PartitionTemplate{
				{TemplateID: "1g.24gb", Name: "1g.24gb"},
			},
			req: &tfv1.AllocRequest{
				Request: tfv1.Resource{
					ComputePercent: resource.MustParse("10"),
					Vram:           resource.MustParse("20Gi"),
				},
			},
			allocatedPartitions: map[string]tfv1.AllocatedPartition{},
			expectError:         true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use different GPU model for missing capacity test
			testGPUModel := gpuModel
			if tt.name == "match with ComputePercent - missing GPU capacity" {
				testGPUModel = "UNKNOWN_GPU_MODEL"
			}

			result, err := MatchPartitionTemplate(
				testGPUModel,
				tt.gpuTemplates,
				tt.req,
				tt.allocatedPartitions,
			)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, result)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, result)
				assert.True(t, result.CanAllocate)
				assert.Equal(t, tt.expectedTemplateID, result.TemplateID)
			}
		})
	}
}

func TestCalculatePartitionResourceUsage(t *testing.T) {
	// Setup
	gpuModel := testGPUModel
	templateID := "1g.24gb"
	PartitionTemplateMap[gpuModel] = map[string]config.PartitionTemplateInfo{
		templateID: {
			TemplateID:   templateID,
			Name:         "1g.24gb",
			MemoryBytes:  24 * 1024 * 1024 * 1024, // 24GB
			Tflops:       50.0,
			ComputeUnits: 14,
		},
	}

	tflops, vram, err := CalculatePartitionResourceUsage(gpuModel, templateID)

	assert.NoError(t, err)
	// Compare using Cmp to handle different formatting (50 vs 50.00)
	assert.Equal(t, 0, tflops.Cmp(resource.MustParse("50")))
	assert.Equal(t, resource.MustParse("24Gi"), vram)
}

func TestCheckPartitionAvailability(t *testing.T) {
	// Setup
	gpuModel := testGPUModel
	templateID := "1g.24gb"
	PartitionTemplateMap[gpuModel] = map[string]config.PartitionTemplateInfo{
		templateID: {
			TemplateID:   templateID,
			Name:         "1g.24gb",
			MemoryBytes:  24 * 1024 * 1024 * 1024, // 24GB
			Tflops:       50.0,
			ComputeUnits: 14,
		},
	}
	MaxPartitionsMap[gpuModel] = 7

	tests := []struct {
		name                string
		gpu                 *tfv1.GPU
		templateID          string
		allocatedPartitions map[string]tfv1.AllocatedPartition
		expectError         bool
		errorContains       string
	}{
		{
			name: "sufficient resources available",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("100"),
						Vram:   resource.MustParse("50Gi"),
					},
				},
			},
			templateID:          templateID,
			allocatedPartitions: map[string]tfv1.AllocatedPartition{},
			expectError:         false,
		},
		{
			name: "insufficient TFLOPs",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("10"), // Too low
						Vram:   resource.MustParse("50Gi"),
					},
				},
			},
			templateID:          templateID,
			allocatedPartitions: map[string]tfv1.AllocatedPartition{},
			expectError:         true,
			errorContains:       "insufficient TFLOPs",
		},
		{
			name: "insufficient VRAM",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("100"),
						Vram:   resource.MustParse("10Gi"), // Too low
					},
				},
			},
			templateID:          templateID,
			allocatedPartitions: map[string]tfv1.AllocatedPartition{},
			expectError:         true,
			errorContains:       "insufficient VRAM",
		},
		{
			name: "max partitions reached",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("100"),
						Vram:   resource.MustParse("50Gi"),
					},
					AllocatedPartitions: map[string]tfv1.AllocatedPartition{
						"pod-1": {TemplateID: templateID, PodUID: "pod-1"},
						"pod-2": {TemplateID: templateID, PodUID: "pod-2"},
						"pod-3": {TemplateID: templateID, PodUID: "pod-3"},
						"pod-4": {TemplateID: templateID, PodUID: "pod-4"},
						"pod-5": {TemplateID: templateID, PodUID: "pod-5"},
						"pod-6": {TemplateID: templateID, PodUID: "pod-6"},
						"pod-7": {TemplateID: templateID, PodUID: "pod-7"},
					},
				},
			},
			templateID: templateID,
			allocatedPartitions: map[string]tfv1.AllocatedPartition{
				"pod-1": {TemplateID: templateID, PodUID: "pod-1"},
				"pod-2": {TemplateID: templateID, PodUID: "pod-2"},
				"pod-3": {TemplateID: templateID, PodUID: "pod-3"},
				"pod-4": {TemplateID: templateID, PodUID: "pod-4"},
				"pod-5": {TemplateID: templateID, PodUID: "pod-5"},
				"pod-6": {TemplateID: templateID, PodUID: "pod-6"},
				"pod-7": {TemplateID: templateID, PodUID: "pod-7"},
			},
			expectError:   true,
			errorContains: "maximum partition count",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := CheckPartitionAvailability(tt.gpu, tt.templateID, tt.allocatedPartitions)

			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
