package cel_filter

import (
	"context"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Helper functions for creating test data
func createTestGPU(name, namespace, gpuModel, phase string, tflops, vram float64) *tfv1.GPU {
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Labels:      make(map[string]string),
			Annotations: make(map[string]string),
		},
		Status: tfv1.GPUStatus{
			GPUModel: gpuModel,
			UUID:     "test-uuid-" + name,
			Phase:    tfv1.TensorFusionGPUPhase(phase),
			Message:  "Test GPU",
		},
	}

	// Set available resources
	if tflops > 0 || vram > 0 {
		gpu.Status.Available = &tfv1.Resource{
			Tflops: *resource.NewMilliQuantity(int64(tflops*1000), resource.DecimalSI),
			Vram:   *resource.NewQuantity(int64(vram*1024*1024*1024), resource.BinarySI),
		}
	}

	return gpu
}

func createTestAllocRequest(namespace, name, gpuModel, celExpression string) *tfv1.AllocRequest {
	return &tfv1.AllocRequest{
		WorkloadNameNamespace: tfv1.NameNamespace{
			Name:      name,
			Namespace: namespace,
		},
		GPUModel:            gpuModel,
		CELFilterExpression: celExpression,
		Count:               1,
	}
}

// Test normal cases of CEL filter (including basic filtering, custom expression, labels/annotations, etc.)
func TestCELFilter_NormalCases(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name          string
		request       *tfv1.AllocRequest
		gpus          []*tfv1.GPU
		expectedCount int
		description   string
	}{
		{
			name:    "filter by GPU model",
			request: createTestAllocRequest("default", "test-workload", "A100", ""),
			gpus: []*tfv1.GPU{
				createTestGPU("gpu-1", "default", "A100", "Ready", 150.0, 40.0),
				createTestGPU("gpu-2", "default", "V100", "Ready", 100.0, 32.0),
				createTestGPU("gpu-3", "default", "A100", "Ready", 150.0, 40.0),
			},
			expectedCount: 2,
			description:   "Should filter GPUs matching the specified model A100",
		},
		{
			name:    "filter by GPU phase only",
			request: createTestAllocRequest("default", "test-workload", "", ""),
			gpus: []*tfv1.GPU{
				createTestGPU("gpu-1", "default", "A100", "Ready", 150.0, 40.0),
				createTestGPU("gpu-2", "default", "A100", "Pending", 150.0, 40.0),
				createTestGPU("gpu-3", "default", "A100", "Ready", 150.0, 40.0),
				createTestGPU("gpu-4", "default", "A100", "Failed", 150.0, 40.0),
			},
			expectedCount: 2,
			description:   "Should only return GPUs in Ready phase",
		},
		{
			name:    "custom CEL expression - filter by available TFLOPS",
			request: createTestAllocRequest("default", "test-workload", "", "gpu.available.tflops > 120.0"),
			gpus: []*tfv1.GPU{
				createTestGPU("gpu-1", "default", "A100", "Ready", 150.0, 40.0),
				createTestGPU("gpu-2", "default", "V100", "Ready", 100.0, 32.0),
				createTestGPU("gpu-3", "default", "H100", "Ready", 200.0, 80.0),
			},
			expectedCount: 2,
			description:   "Should filter GPUs with TFLOPS > 120 and Ready phase",
		},
		{
			name:    "custom CEL expression - filter by available VRAM",
			request: createTestAllocRequest("default", "test-workload", "", "gpu.available.vram > 35000000000"), // > 35GB in bytes
			gpus: []*tfv1.GPU{
				createTestGPU("gpu-1", "default", "A100", "Ready", 150.0, 40.0), // 40GB
				createTestGPU("gpu-2", "default", "V100", "Ready", 100.0, 32.0), // 32GB
				createTestGPU("gpu-3", "default", "H100", "Ready", 200.0, 80.0), // 80GB
			},
			expectedCount: 2,
			description:   "Should filter GPUs with VRAM > 35GB and Ready phase",
		},
		{
			name:    "combined model and custom CEL expression",
			request: createTestAllocRequest("default", "test-workload", "A100", "gpu.available.tflops >= 150.0"),
			gpus: []*tfv1.GPU{
				createTestGPU("gpu-1", "default", "A100", "Ready", 150.0, 40.0),
				createTestGPU("gpu-2", "default", "A100", "Ready", 120.0, 40.0),
				createTestGPU("gpu-3", "default", "V100", "Ready", 160.0, 32.0),
				createTestGPU("gpu-4", "default", "A100", "Ready", 180.0, 40.0),
			},
			expectedCount: 2,
			description:   "Should filter A100 GPUs with TFLOPS >= 150 and Ready phase",
		},
		{
			name:    "filter by labels",
			request: createTestAllocRequest("default", "test-workload", "", "gpu.labels['environment'] == 'production'"),
			gpus: func() []*tfv1.GPU {
				gpu1 := createTestGPU("gpu-1", "default", "A100", "Ready", 150.0, 40.0)
				gpu1.Labels["environment"] = "production"
				gpu2 := createTestGPU("gpu-2", "default", "A100", "Ready", 150.0, 40.0)
				gpu2.Labels["environment"] = "development"
				gpu3 := createTestGPU("gpu-3", "default", "A100", "Ready", 150.0, 40.0)
				gpu3.Labels["environment"] = "production"
				return []*tfv1.GPU{gpu1, gpu2, gpu3}
			}(),
			expectedCount: 2,
			description:   "Should filter GPUs with environment=production label",
		},
		{
			name:    "filter by annotations",
			request: createTestAllocRequest("default", "test-workload", "", "gpu.annotations['priority'] == 'critical'"),
			gpus: func() []*tfv1.GPU {
				gpu1 := createTestGPU("gpu-1", "default", "A100", "Ready", 150.0, 40.0)
				gpu1.Annotations["priority"] = "critical"
				gpu2 := createTestGPU("gpu-2", "default", "A100", "Ready", 150.0, 40.0)
				gpu2.Annotations["priority"] = "low"
				gpu3 := createTestGPU("gpu-3", "default", "A100", "Ready", 150.0, 40.0)
				gpu3.Annotations["priority"] = "critical"
				return []*tfv1.GPU{gpu1, gpu2, gpu3}
			}(),
			expectedCount: 2,
			description:   "Should filter GPUs with priority=critical annotation",
		},
		{
			name:    "combined labels and annotations filter",
			request: createTestAllocRequest("default", "test-workload", "", "gpu.labels['tier'] == 'high-performance' && gpu.annotations['priority'] == 'critical'"),
			gpus: func() []*tfv1.GPU {
				gpu1 := createTestGPU("gpu-1", "default", "A100", "Ready", 150.0, 40.0)
				gpu1.Labels["tier"] = "high-performance"
				gpu1.Annotations["priority"] = "critical"
				gpu2 := createTestGPU("gpu-2", "default", "A100", "Ready", 150.0, 40.0)
				gpu2.Labels["tier"] = "standard"
				gpu2.Annotations["priority"] = "critical"
				gpu3 := createTestGPU("gpu-3", "default", "A100", "Ready", 150.0, 40.0)
				gpu3.Labels["tier"] = "high-performance"
				gpu3.Annotations["priority"] = "low"
				return []*tfv1.GPU{gpu1, gpu2, gpu3}
			}(),
			expectedCount: 1,
			description:   "Should filter GPUs matching both label and annotation conditions",
		},
		{
			name:          "empty GPU list",
			request:       createTestAllocRequest("default", "test-workload", "A100", ""),
			gpus:          []*tfv1.GPU{},
			expectedCount: 0,
			description:   "Should handle empty GPU list gracefully",
		},
		{
			name:    "complex combined expression with model, resources, and metadata",
			request: createTestAllocRequest("default", "test-workload", "A100", "gpu.available.tflops >= 150.0 && gpu.labels['environment'] == 'production'"),
			gpus: func() []*tfv1.GPU {
				gpu1 := createTestGPU("gpu-1", "default", "A100", "Ready", 180.0, 40.0)
				gpu1.Labels["environment"] = "production"
				gpu2 := createTestGPU("gpu-2", "default", "A100", "Ready", 120.0, 40.0)
				gpu2.Labels["environment"] = "production"
				gpu3 := createTestGPU("gpu-3", "default", "A100", "Ready", 200.0, 40.0)
				gpu3.Labels["environment"] = "development"
				return []*tfv1.GPU{gpu1, gpu2, gpu3}
			}(),
			expectedCount: 1,
			description:   "Should filter A100 GPUs with TFLOPS >= 150, production environment, and Ready phase",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create cache and CEL filter
			cache, err := NewExpressionCache(10, 5*time.Minute)
			require.NoError(t, err, "Failed to create expression cache")

			celFilter, err := NewCELFilter(tt.request, cache)
			require.NoError(t, err, "Failed to create CEL filter")

			// Execute filter
			workerPodKey := tfv1.NameNamespace{Name: "worker-pod", Namespace: "default"}
			filteredGPUs, err := celFilter.Filter(ctx, workerPodKey, tt.gpus)

			// Verify results
			require.NoError(t, err, "Filter execution should not fail")
			assert.Len(t, filteredGPUs, tt.expectedCount, tt.description)

			// Verify filter name
			assert.Contains(t, celFilter.Name(), "AllocRequest-")
			assert.Contains(t, celFilter.Name(), tt.request.WorkloadNameNamespace.String())
		})
	}
}

// Test edge and exception cases of CEL filter
func TestCELFilter_EdgeAndExceptionCases(t *testing.T) {
	ctx := context.Background()

	// Test CEL expressions with various edge cases (compilation + execution)
	t.Run("CEL expressions edge cases", func(t *testing.T) {
		// Test GPUs for execution
		testGPUs := []*tfv1.GPU{
			createTestGPU("gpu-1", "default", "A100", "Ready", 150.0, 40.0),
			createTestGPU("gpu-2", "default", "V100", "Ready", 100.0, 32.0),
		}
		// Add GPU with nil resources
		gpuWithNilResources := createTestGPU("gpu-nil", "default", "A100", "Ready", 0, 0)
		gpuWithNilResources.Status.Available = nil
		testGPUs = append(testGPUs, gpuWithNilResources)

		workerPodKey := tfv1.NameNamespace{Name: "worker-pod", Namespace: "default"}

		edgeCases := []struct {
			name          string
			expression    string
			shouldFail    bool // Whether compilation/creation should fail
			expectedCount int  // Expected GPU count if execution succeeds
			description   string
		}{
			// Compilation failures
			{
				name:        "syntax error - missing quotes",
				expression:  "gpu.gpuModel == A100",
				shouldFail:  true,
				description: "Missing quotes should cause compilation error",
			},
			{
				name:        "syntax error - invalid operator",
				expression:  "gpu.phase === 'Ready'",
				shouldFail:  true,
				description: "Invalid operator should cause compilation error",
			},
			{
				name:        "undefined variable",
				expression:  "jdwquygfewqndwql",
				shouldFail:  true,
				description: "Undefined variable should fail when combined with other conditions",
			},
			{
				name:        "whitespace only expression",
				expression:  "   ",
				shouldFail:  true,
				description: "Whitespace-only expression should fail",
			},

			// Compilation success but runtime behavior testing
			{
				name:          "empty expression",
				expression:    "",
				shouldFail:    false,
				expectedCount: 3, // All Ready GPUs pass
				description:   "Empty expression should work (no additional filtering)",
			},
			{
				name:          "logically contradictory expression",
				expression:    "gpu.phase > 100 && gpu.phase < 100",
				shouldFail:    false,
				expectedCount: 0, // No GPUs pass impossible condition
				description:   "Contradictory logic should compile but filter out all GPUs",
			},
			{
				name:          "type mismatch comparison",
				expression:    "gpu.phase == 123",
				shouldFail:    false,
				expectedCount: 0, // No GPUs pass type mismatch
				description:   "Type mismatch should return false for all GPUs",
			},
			{
				name:          "undefined nested field access",
				expression:    "gpu.nonexistent.field == 'value'",
				shouldFail:    false,
				expectedCount: 0, // No GPUs pass undefined field check
				description:   "Undefined nested field should return false (fail-safe)",
			},
			{
				name:          "numeric comparison on string",
				expression:    "gpu.gpuModel > 50",
				shouldFail:    false,
				expectedCount: 0, // No GPUs pass invalid comparison
				description:   "Invalid type comparison should return false",
			},
			{
				name:          "null field access",
				expression:    "gpu.available.tflops > 100",
				shouldFail:    false,
				expectedCount: 1, // Only A100 with 150 TFLOPS passes (V100=100, nil=fails)
				description:   "Null field access should be handled gracefully",
			},
			{
				name:          "conditional null handling",
				expression:    "has(gpu.available) ? gpu.available.tflops > 120 : false",
				shouldFail:    false,
				expectedCount: 1, // Only A100 with 150 TFLOPS
				description:   "Conditional expressions should handle nulls correctly",
			},
			{
				name:          "always true expression",
				expression:    "true",
				shouldFail:    false,
				expectedCount: 3, // All Ready GPUs pass
				description:   "Tautology should pass all Ready phase GPUs",
			},
			{
				name:          "always false expression",
				expression:    "false",
				shouldFail:    false,
				expectedCount: 0, // No GPUs pass
				description:   "Contradiction should filter out all GPUs",
			},
		}

		for _, tt := range edgeCases {
			t.Run(tt.name, func(t *testing.T) {
				cache, err := NewExpressionCache(10, 5*time.Minute)
				require.NoError(t, err)

				request := createTestAllocRequest("default", "test-workload", "", tt.expression)
				celFilter, err := NewCELFilter(request, cache)

				if tt.shouldFail {
					// Should fail at creation or execution
					if err != nil {
						t.Logf("✅ Expected compilation failure: %v", err)
						return
					}

					// If creation succeeded, should fail at execution
					_, err = celFilter.Filter(ctx, workerPodKey, testGPUs)
					assert.Error(t, err, "Should fail during execution: %s", tt.description)
					t.Logf("✅ Expected execution failure: %v", err)
				} else {
					// Should succeed in both creation and execution
					require.NoError(t, err, "Filter creation should succeed: %s", tt.description)

					filteredGPUs, err := celFilter.Filter(ctx, workerPodKey, testGPUs)
					require.NoError(t, err, "Filter execution should succeed: %s", tt.description)

					assert.Len(t, filteredGPUs, tt.expectedCount, tt.description)
					t.Logf("✅ Expression '%s': %d/%d GPUs filtered", tt.expression, len(filteredGPUs), len(testGPUs))
				}
			})
		}
	})
}
