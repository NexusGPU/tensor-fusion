package utils_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

func TestContainsGPUResources(t *testing.T) {
	tests := []struct {
		name     string
		res      corev1.ResourceList
		expected bool
	}{
		{
			name:     "empty resource list",
			res:      corev1.ResourceList{},
			expected: false,
		},
		{
			name: "nvidia.com/gpu with positive value",
			res: corev1.ResourceList{
				corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
			},
			expected: true,
		},
		{
			name: "nvidia.com/gpu with zero value",
			res: corev1.ResourceList{
				corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("0"),
			},
			expected: false,
		},
		{
			name: "amd.com/gpu with positive value",
			res: corev1.ResourceList{
				corev1.ResourceName("amd.com/gpu"): resource.MustParse("2"),
			},
			expected: true,
		},
		{
			name: "amd.com/gpu with zero value",
			res: corev1.ResourceList{
				corev1.ResourceName("amd.com/gpu"): resource.MustParse("0"),
			},
			expected: false,
		},
		{
			name: "nvidia.com/gpu with negative value (should not happen in practice but test edge case)",
			res: corev1.ResourceList{
				corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("-1"),
			},
			expected: false,
		},
		{
			name: "non-GPU resource",
			res: corev1.ResourceList{
				corev1.ResourceCPU: resource.MustParse("1"),
			},
			expected: false,
		},
		{
			name: "multiple resources including GPU",
			res: corev1.ResourceList{
				corev1.ResourceCPU:                    resource.MustParse("1"),
				corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
			},
			expected: true,
		},
		{
			name: "multiple resources with GPU set to zero",
			res: corev1.ResourceList{
				corev1.ResourceCPU:                    resource.MustParse("1"),
				corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("0"),
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// containsGPUResources is not exported, so we test it through HasGPUResourceRequest
			// or we can test it directly if we make it exported, but for now let's test via HasGPUResourceRequest
			pod := &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Requests: tt.res,
							},
						},
					},
				},
			}
			result := utils.HasGPUResourceRequest(pod)
			assert.Equal(t, tt.expected, result, "HasGPUResourceRequest should return %v for %s", tt.expected, tt.name)
		})
	}
}

func TestHasGPUResourceRequest(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected bool
	}{
		{
			name: "pod with no containers",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{},
				},
			},
			expected: false,
		},
		{
			name: "pod with GPU in requests",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "pod with GPU in limits (and requests is not nil)",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("1"),
								},
								Limits: corev1.ResourceList{
									corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "pod with GPU set to zero in requests",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("0"),
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "pod with GPU set to zero in limits",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("0"),
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "pod with multiple containers, one with GPU",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "container-1",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU: resource.MustParse("1"),
								},
							},
						},
						{
							Name: "container-2",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "pod with multiple containers, all with GPU set to zero",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "container-1",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("0"),
								},
							},
						},
						{
							Name: "container-2",
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									corev1.ResourceName("amd.com/gpu"): resource.MustParse("0"),
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "pod with AMD GPU",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceName("amd.com/gpu"): resource.MustParse("1"),
								},
							},
						},
					},
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.HasGPUResourceRequest(tt.pod)
			assert.Equal(t, tt.expected, result, "HasGPUResourceRequest should return %v for %s", tt.expected, tt.name)
		})
	}
}
