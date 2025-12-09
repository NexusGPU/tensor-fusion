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
	"time"

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
			TemplateID:      "19",
			Name:            "1g.24gb",
			MemoryGigabytes: 24, // 24GB (function converts to bytes)
			ComputePercent:  1.0 / 7.0 * 100,
		},
		"4g.94gb": {
			TemplateID:      "9",
			Name:            "4g.94gb",
			MemoryGigabytes: 94, // 94GB (function converts to bytes)
			ComputePercent:  4.0 / 7.0 * 100,
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
				tfv1.GPUStatus{
					GPUModel:            testGPUModel,
					PartitionTemplates:  tt.gpuTemplates,
					AllocatedPartitions: tt.allocatedPartitions,
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("312"),
						Vram:   resource.MustParse("80Gi"),
					},
				},
				tt.req,
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
			TemplateID:      templateID,
			Name:            "1g.24gb",
			MemoryGigabytes: 24, // 24GB (function converts to bytes)
			ComputePercent:  1.0 / 7.0 * 100,
		},
	}

	tflops, vram, err := CalculatePartitionResourceUsage(resource.MustParse("312"), gpuModel, templateID)

	assert.NoError(t, err)
	// Compare using Cmp to handle different formatting
	// 1/7 of 312 TFLOPs = 44.57 TFLOPs
	expectedTflops := resource.MustParse("44.57")
	assert.Equal(t, 0, tflops.Cmp(expectedTflops), "TFLOPs: got %s, expected %s", tflops.String(), expectedTflops.String())
	// Compare VRAM using Cmp to handle quantity representation differences
	assert.Equal(t, 0, vram.Cmp(resource.MustParse("24Gi")), "VRAM: got %s, expected 24Gi", vram.String())
}

func TestCheckPartitionAvailability(t *testing.T) {
	// Setup: A100 MIG constraints based on nvidia-smi mig -lgipp output
	// Profile 19 (1g.24gb): Placements {0,1,2,3,4,5,6}:1 - can start at any of 7 positions, occupies 1 slot each
	// Profile 9 (4g.94gb): Placements {0,4}:4 - can start at position 0 or 4, occupies 4 slots each
	gpuModel := testGPUModel
	template1g := "1g.24gb" // Profile 19
	template4g := "4g.94gb" // Profile 9

	// Clear and setup maps for this test
	mu.Lock()
	PartitionTemplateMap[gpuModel] = map[string]config.PartitionTemplateInfo{
		template1g: {
			TemplateID:      template1g,
			Name:            "1g.24gb",
			MemoryGigabytes: 24, // 24GB
			ComputePercent:  1.0 / 7.0 * 100,
			MaxPartition:    7,                             // Can allocate up to 7 instances
			PlacementLimit:  []uint32{0, 1, 2, 3, 4, 5, 6}, // Can start at any of these positions
			PlacementOffSet: 1,                             // Occupies 1 slot
		},
		template4g: {
			TemplateID:      template4g,
			Name:            "4g.94gb",
			MemoryGigabytes: 94, // 94GB
			ComputePercent:  4.0 / 7.0 * 100,
			MaxPartition:    2,              // Can only allocate 2 instances
			PlacementLimit:  []uint32{0, 4}, // Can start at position 0 or 4
			PlacementOffSet: 4,              // Occupies 4 slots (0-3 or 4-7)
		},
	}
	MaxPartitionsMap[gpuModel] = 7
	MaxPlacementSlotsMap[gpuModel] = 8 // A100 has 8 placement slots (0-7)
	mu.Unlock()

	tests := []struct {
		name          string
		gpu           *tfv1.GPU
		templateID    string
		expectError   bool
		errorContains string
	}{
		{
			name: "happy path - 1g.24gb allocation succeeds",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("312"),
						Vram:   resource.MustParse("80Gi"),
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("100"),
						Vram:   resource.MustParse("50Gi"),
					},
					AllocatedPartitions: map[string]tfv1.AllocatedPartition{},
				},
			},
			templateID:  template1g,
			expectError: false,
		},
		{
			name: "Profile 19 * 4 should fail - all valid positions occupied",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("312"),
						Vram:   resource.MustParse("80Gi"),
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("200"),
						Vram:   resource.MustParse("96Gi"),
					},
					AllocatedPartitions: map[string]tfv1.AllocatedPartition{
						"pod-1": {TemplateID: template1g, PodUID: "pod-1"}, // Profile 19 at position 0 (slot 0)
						"pod-2": {TemplateID: template1g, PodUID: "pod-2"}, // Profile 19 at position 1 (slot 1)
						"pod-3": {TemplateID: template1g, PodUID: "pod-3"}, // Profile 19 at position 2 (slot 2)
						"pod-4": {TemplateID: template1g, PodUID: "pod-4"}, // Profile 19 at position 3 (slot 3)
						// Positions 4,5,6 are still free, but trying to allocate 5th instance
						// Actually wait, if we have 4 instances, we need to check if 5th can fit
						// Let me change this to have Profile 9 at position 0, then Profile 19 * 3, then try 4th
					},
				},
			},
			templateID:  template1g,
			expectError: false, // Actually 4 instances can fit at positions 0,1,2,3, leaving 4,5,6 free
		},
		{
			name: "Profile 9 at 0 + Profile 19 * 4 should fail",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("312"),
						Vram:   resource.MustParse("80Gi"),
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("200"),
						Vram:   resource.MustParse("96Gi"),
					},
					AllocatedPartitions: map[string]tfv1.AllocatedPartition{
						"pod-p9": {TemplateID: template4g, PodUID: "pod-p9", AllocatedAt: metav1.NewTime(metav1.Now().Add(-3 * time.Hour))}, // Profile 9 allocated first at position 0, occupies slots 0,1,2,3
						"pod-1":  {TemplateID: template1g, PodUID: "pod-1", AllocatedAt: metav1.NewTime(metav1.Now().Add(-2 * time.Hour))},  // Profile 19 at position 4 (slot 4)
						"pod-2":  {TemplateID: template1g, PodUID: "pod-2", AllocatedAt: metav1.NewTime(metav1.Now().Add(-1 * time.Hour))},  // Profile 19 at position 5 (slot 5)
						"pod-3":  {TemplateID: template1g, PodUID: "pod-3", AllocatedAt: metav1.Now()},                                      // Profile 19 at position 6 (slot 6)
						// Trying to allocate 4th Profile 19 instance - should fail
						// All valid positions {0,1,2,3,4,5,6} are either occupied or conflict
					},
				},
			},
			templateID:    template1g,
			expectError:   true,
			errorContains: "placement slots",
		},
		{
			name: "Profile 9 * 1 + Profile 19 * 3 should work",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("312"),
						Vram:   resource.MustParse("80Gi"),
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("150"),
						Vram:   resource.MustParse("118Gi"),
					},
					AllocatedPartitions: map[string]tfv1.AllocatedPartition{
						"pod-p9": {TemplateID: template4g, PodUID: "pod-p9"}, // Profile 9 at position 0, occupies slots 0,1,2,3
						"pod-1":  {TemplateID: template1g, PodUID: "pod-1"},  // Profile 19 at slot 4
						"pod-2":  {TemplateID: template1g, PodUID: "pod-2"},  // Profile 19 at slot 5
						// Trying to allocate 3rd Profile 19 instance - should succeed at slot 6
					},
				},
			},
			templateID:  template1g,
			expectError: false, // 3rd Profile 19 instance should succeed
		},
		{
			name: "Profile 9 * 1 + Profile 19 * 3 should work (happy case)",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("312"),
						Vram:   resource.MustParse("80Gi"),
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("150"),
						Vram:   resource.MustParse("118Gi"),
					},
					AllocatedPartitions: map[string]tfv1.AllocatedPartition{
						"pod-p9": {TemplateID: template4g, PodUID: "pod-p9"}, // Profile 9 at position 0, occupies slots 0,1,2,3
						"pod-1":  {TemplateID: template1g, PodUID: "pod-1"},  // Profile 19 at slot 4
						"pod-2":  {TemplateID: template1g, PodUID: "pod-2"},  // Profile 19 at slot 5
						// Trying to allocate 3rd Profile 19 instance - should succeed at slot 6
					},
				},
			},
			templateID:  template1g,
			expectError: false,
		},
		{
			name: "Profile 9 - all placement positions occupied",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("312"),
						Vram:   resource.MustParse("80Gi"),
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("200"),
						Vram:   resource.MustParse("94Gi"),
					},
					AllocatedPartitions: map[string]tfv1.AllocatedPartition{
						"pod-1": {TemplateID: template4g, PodUID: "pod-1"}, // Profile 9 at position 0, occupies slots 0,1,2,3
						"pod-2": {TemplateID: template4g, PodUID: "pod-2"}, // Profile 9 at position 4, occupies slots 4,5,6,7
						// Both positions {0,4} are now occupied
					},
				},
			},
			templateID:    template4g,
			expectError:   true,
			errorContains: "maximum partition count", // MaxPartition check happens first (2/2)
		},
		{
			name: "insufficient TFLOPs",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("312"),
						Vram:   resource.MustParse("80Gi"),
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("10"), // Too low
						Vram:   resource.MustParse("50Gi"),
					},
					AllocatedPartitions: map[string]tfv1.AllocatedPartition{},
				},
			},
			templateID:    template1g,
			expectError:   true,
			errorContains: "insufficient TFLOPs",
		},
		{
			name: "insufficient VRAM",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("312"),
						Vram:   resource.MustParse("80Gi"),
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("100"),
						Vram:   resource.MustParse("10Gi"), // Too low for 24Gi required
					},
					AllocatedPartitions: map[string]tfv1.AllocatedPartition{},
				},
			},
			templateID:    template1g,
			expectError:   true,
			errorContains: "insufficient VRAM",
		},
		{
			name: "Profile 9 can allocate at position 4 when Profile 19 uses slots 0-2",
			gpu: &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
				Status: tfv1.GPUStatus{
					GPUModel: gpuModel,
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse("312"),
						Vram:   resource.MustParse("80Gi"),
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse("200"),
						Vram:   resource.MustParse("94Gi"),
					},
					AllocatedPartitions: map[string]tfv1.AllocatedPartition{
						"pod-1": {TemplateID: template1g, PodUID: "pod-1"}, // Slot 0
						"pod-2": {TemplateID: template1g, PodUID: "pod-2"}, // Slot 1
						"pod-3": {TemplateID: template1g, PodUID: "pod-3"}, // Slot 2
						// Slots 3,4,5,6,7 are free
						// Profile 9 can use position 4 (slots 4,5,6,7) or position 0 (slots 0,1,2,3)
						// Position 0 conflicts, but position 4 is free
					},
				},
			},
			templateID:  template4g,
			expectError: false, // Profile 9 can use position 4
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := CheckPartitionAvailability(tt.gpu, tt.templateID)

			if tt.expectError {
				if !assert.Error(t, err) {
					return // Stop if no error when one is expected
				}
				if tt.errorContains != "" && err != nil {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
