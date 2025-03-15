package scheduler

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestLowLoadFirst(t *testing.T) {
	// Create test cases
	testCases := []struct {
		name     string
		gpus     []tfv1.GPU
		expected string
		errorMsg string
	}{
		{
			name: "select highest available resources",
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-2"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("15"),
							Vram:   resource.MustParse("80Gi"),
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-3"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("20"),
							Vram:   resource.MustParse("60Gi"),
						},
					},
				},
			},
			expected: "gpu-2", // Has highest VRAM
		},
		{
			name: "select higher TFLOPS when VRAM is equal",
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-2"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("15"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
			},
			expected: "gpu-2", // Equal VRAM but higher TFLOPS
		},
		{
			name:     "no GPUs available",
			gpus:     []tfv1.GPU{},
			expected: "",
			errorMsg: "no GPUs available",
		},
		{
			name: "single GPU available",
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
			},
			expected: "gpu-1",
		},
	}

	// Run test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			strategy := LowLoadFirst{}
			selected, err := strategy.SelectGPU(tc.gpus)

			if tc.errorMsg != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tc.errorMsg)
				assert.Nil(t, selected)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, selected)
				assert.Equal(t, tc.expected, selected.Name)
			}
		})
	}
}

func TestCompactFirst(t *testing.T) {
	// Create test cases
	testCases := []struct {
		name     string
		gpus     []tfv1.GPU
		expected string
		errorMsg string
	}{
		{
			name: "select lowest available resources",
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-2"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("15"),
							Vram:   resource.MustParse("20Gi"),
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-3"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("20"),
							Vram:   resource.MustParse("60Gi"),
						},
					},
				},
			},
			expected: "gpu-2", // Has lowest VRAM
		},
		{
			name: "select lower TFLOPS when VRAM is equal",
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-2"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("15"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
			},
			expected: "gpu-1", // Equal VRAM but lower TFLOPS
		},
		{
			name:     "no GPUs available",
			gpus:     []tfv1.GPU{},
			expected: "",
			errorMsg: "no GPUs available",
		},
		{
			name: "single GPU available",
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
			},
			expected: "gpu-1",
		},
	}

	// Run test cases
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			strategy := CompactFirst{}
			selected, err := strategy.SelectGPU(tc.gpus)

			if tc.errorMsg != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tc.errorMsg)
				assert.Nil(t, selected)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, selected)
				assert.Equal(t, tc.expected, selected.Name)
			}
		})
	}
}

func TestStrategyEdgeCases(t *testing.T) {
	// Test both strategies with different edge cases
	edgeCases := []struct {
		name          string
		gpus          []tfv1.GPU
		lowLoadName   string // expected result for LowLoadFirst
		compactName   string // expected result for CompactFirst
		errorExpected bool
	}{
		{
			name: "identical resources",
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-2"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("40Gi"),
						},
					},
				},
			},
			lowLoadName:   "gpu-1", // Both strategies should pick the first one when values are equal
			compactName:   "gpu-1",
			errorExpected: false,
		},
		{
			name: "extreme value differences",
			gpus: []tfv1.GPU{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-1"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("1"),
							Vram:   resource.MustParse("1Gi"),
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu-2"},
					Status: tfv1.GPUStatus{
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("100"),
							Vram:   resource.MustParse("1000Gi"),
						},
					},
				},
			},
			lowLoadName:   "gpu-2", // Highest resources
			compactName:   "gpu-1", // Lowest resources
			errorExpected: false,
		},
	}

	// Test both strategies
	for _, tc := range edgeCases {
		t.Run("LowLoadFirst_"+tc.name, func(t *testing.T) {
			strategy := LowLoadFirst{}
			selected, err := strategy.SelectGPU(tc.gpus)

			if tc.errorExpected {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, selected)
				assert.Equal(t, tc.lowLoadName, selected.Name)
			}
		})

		t.Run("CompactFirst_"+tc.name, func(t *testing.T) {
			strategy := CompactFirst{}
			selected, err := strategy.SelectGPU(tc.gpus)

			if tc.errorExpected {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, selected)
				assert.Equal(t, tc.compactName, selected.Name)
			}
		})
	}
}
