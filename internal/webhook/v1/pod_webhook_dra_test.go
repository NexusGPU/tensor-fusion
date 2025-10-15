package v1

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

func TestDRAProcessor_IsDRAEnabled(t *testing.T) {
	tests := []struct {
		name           string
		processorDRA   bool
		podAnnotations map[string]string
		expected       bool
	}{
		{
			name:         "global DRA enabled, no pod annotation",
			processorDRA: true,
			expected:     true,
		},
		{
			name:         "global DRA disabled, no pod annotation",
			processorDRA: false,
			expected:     false,
		},
		{
			name:         "global DRA disabled, pod annotation enabled",
			processorDRA: false,
			podAnnotations: map[string]string{
				constants.DRAEnabledAnnotation: constants.TrueStringValue,
			},
			expected: true,
		},
		{
			name:         "global DRA enabled, pod annotation disabled",
			processorDRA: true,
			podAnnotations: map[string]string{
				constants.DRAEnabledAnnotation: constants.FalseStringValue,
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			processor := &DRAProcessor{
				enableDRA:    tt.processorDRA,
				configLoaded: true, // Skip config loading in tests
			}

			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: tt.podAnnotations,
				},
			}

			result := processor.IsDRAEnabled(context.Background(), pod)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestDRAProcessor_HandleDRAAdmission(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))

	// Create a SchedulingConfigTemplate with DRA config
	template := &tfv1.SchedulingConfigTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-template",
		},
		Spec: tfv1.SchedulingConfigTemplateSpec{
			DRA: &tfv1.DRAConfig{
				Enable:                    &[]bool{true}[0],
				ResourceClaimTemplateName: "custom-gpu-template",
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(template).
		Build()

	processor := &DRAProcessor{
		Client: fakeClient,
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-namespace",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "test-container"},
			},
		},
	}

	tfInfo := &utils.TensorFusionInfo{
		Profile: &tfv1.WorkloadProfileSpec{
			GPUCount: 1,
			Resources: tfv1.Resources{
				Requests: tfv1.Resource{
					Tflops: resource.MustParse("10"),
					Vram:   resource.MustParse("8Gi"),
				},
			},
		},
	}

	containerIndices := []int{0}

	// Test HandleDRAAdmission
	err := processor.HandleDRAAdmission(context.Background(), pod, tfInfo, containerIndices)
	require.NoError(t, err)

	// Verify CEL expression is stored in Pod annotation
	celExpression := pod.Annotations[constants.DRACelExpressionAnnotation]
	require.NotEmpty(t, celExpression)
	assert.Contains(t, celExpression, `device.capacity["tflops"].AsApproximateFloat64() >= 10`)
	assert.Contains(t, celExpression, `device.capacity["vram"].AsApproximateFloat64() >=`)

	// Verify DRA enabled annotation is set
	assert.Equal(t, constants.TrueStringValue, pod.Annotations[constants.DRAEnabledAnnotation])

	// Verify ResourceClaimTemplate reference is added to Pod
	require.Len(t, pod.Spec.ResourceClaims, 1)
	podClaim := pod.Spec.ResourceClaims[0]
	assert.Equal(t, constants.DRAClaimDefineName, podClaim.Name)
	require.NotNil(t, podClaim.ResourceClaimTemplateName)
	assert.Equal(t, "custom-gpu-template", *podClaim.ResourceClaimTemplateName)

	// Verify processor has cached the ResourceClaimTemplateName
	assert.Equal(t, "custom-gpu-template", processor.resourceClaimTemplateName)
}

func TestBuildCELSelector(t *testing.T) {
	tests := []struct {
		name                 string
		pod                  *corev1.Pod
		tfInfo               *utils.TensorFusionInfo
		expectedConditions   []string
		unexpectedConditions []string
	}{
		{
			name: "Basic resource filters",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "test-namespace",
				},
			},
			tfInfo: &utils.TensorFusionInfo{
				Profile: &tfv1.WorkloadProfileSpec{
					GPUCount: 2,
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: resource.MustParse("20"),
							Vram:   resource.MustParse("16Gi"),
						},
					},
					GPUModel: "H100",
				},
			},
			expectedConditions: []string{
				`device.attributes["model"] == "H100"`,
				`device.capacity["tflops"].AsApproximateFloat64() >= 20`,
				`device.capacity["vram"].AsApproximateFloat64() >=`,
				`device.attributes["phase"] == "Running"`,
				`device.attributes["phase"] == "Pending"`,
			},
		},
		{
			name: "All filters including pool and QoS",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "production",
				},
			},
			tfInfo: &utils.TensorFusionInfo{
				Profile: &tfv1.WorkloadProfileSpec{
					GPUCount: 1,
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("8Gi"),
						},
					},
					GPUModel: "A100",
					PoolName: "high-priority",
					Qos:      "high",
				},
			},
			expectedConditions: []string{
				`device.attributes["model"] == "A100"`,
				`device.attributes["pool_name"] == "high-priority"`,
				`device.capacity["tflops"].AsApproximateFloat64() >= 10`,
				`device.capacity["vram"].AsApproximateFloat64() >=`,
				`device.attributes["qos"] == "high"`,
				`device.attributes["phase"] == "Running"`,
				`device.attributes["phase"] == "Pending"`,
			},
		},
		{
			name: "Only phase filter when no specific requirements",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
				},
			},
			tfInfo: &utils.TensorFusionInfo{
				Profile: &tfv1.WorkloadProfileSpec{
					GPUCount: 0, // Zero count should not add condition
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							// Zero resources
						},
					},
				},
			},
			expectedConditions: []string{
				`device.attributes["phase"] == "Running"`,
				`device.attributes["phase"] == "Pending"`,
			},
			unexpectedConditions: []string{
				`device.attributes["model"]`,
				`device.attributes["pool_name"]`,
				`device.capacity["tflops"]`,
				`device.capacity["vram"]`,
			},
		},
		{
			name: "Empty profile still includes phase filter",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "",
				},
			},
			tfInfo: &utils.TensorFusionInfo{
				Profile: &tfv1.WorkloadProfileSpec{
					// All empty/zero values
				},
			},
			expectedConditions: []string{
				`device.attributes["phase"] == "Running"`,
				`device.attributes["phase"] == "Pending"`,
			},
			unexpectedConditions: []string{
				`device.attributes["model"]`,
				`device.attributes["pool_name"]`,
				`device.capacity["tflops"]`,
				`device.capacity["vram"]`,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			celExpression, err := BuildCELSelector(tt.pod, tt.tfInfo)
			require.NoError(t, err)
			require.NotEmpty(t, celExpression)

			// Verify expected conditions are present
			for _, condition := range tt.expectedConditions {
				assert.Contains(t, celExpression, condition, "Expected condition not found: %s", condition)
			}

			// Verify unexpected conditions are not present
			for _, condition := range tt.unexpectedConditions {
				assert.NotContains(t, celExpression, condition, "Unexpected condition found: %s", condition)
			}

			// Verify proper AND joining between different condition types
			// Count unique condition types (model, pool_name, capacity, qos, phase)
			hasModel := false
			hasPool := false
			hasCapacity := false
			hasQoS := false
			for _, cond := range tt.expectedConditions {
				if strings.Contains(cond, `"model"`) {
					hasModel = true
				}
				if strings.Contains(cond, `"pool_name"`) {
					hasPool = true
				}
				if strings.Contains(cond, `capacity[`) {
					hasCapacity = true
				}
				if strings.Contains(cond, `"qos"`) {
					hasQoS = true
				}
			}
			conditionTypeCount := 0
			if hasModel {
				conditionTypeCount++
			}
			if hasPool {
				conditionTypeCount++
			}
			if hasCapacity {
				conditionTypeCount++
			}
			if hasQoS {
				conditionTypeCount++
			}
			// Phase is always present, so add 1
			conditionTypeCount++

			// If we have more than just the phase condition, there should be && operators
			if conditionTypeCount > 1 {
				assert.Contains(t, celExpression, " && ", "Multiple condition types should be joined with &&")
			}
		})
	}
}

func TestHasDRAClaim(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected bool
	}{
		{
			name: "pod with resource claims",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					ResourceClaims: []corev1.PodResourceClaim{
						{Name: "gpu-claim"},
					},
				},
			},
			expected: true,
		},
		{
			name: "pod without resource claims",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := HasDRAClaim(tt.pod)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestDRAProcessor_LazyConfigLoading(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))

	tests := []struct {
		name      string
		templates []tfv1.SchedulingConfigTemplate
		expected  bool
	}{
		{
			name: "DRA enabled in template",
			templates: []tfv1.SchedulingConfigTemplate{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "template1"},
					Spec: tfv1.SchedulingConfigTemplateSpec{
						DRA: &tfv1.DRAConfig{
							Enable:                    &[]bool{true}[0],
							ResourceClaimTemplateName: "test-gpu-template",
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "DRA disabled in template",
			templates: []tfv1.SchedulingConfigTemplate{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "template1"},
					Spec: tfv1.SchedulingConfigTemplateSpec{
						DRA: &tfv1.DRAConfig{
							Enable: &[]bool{false}[0],
						},
					},
				},
			},
			expected: false,
		},
		{
			name:     "no templates",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			objects := make([]client.Object, len(tt.templates))
			for i, template := range tt.templates {
				objects[i] = &template
			}

			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(objects...).
				Build()

			processor := &DRAProcessor{
				Client: fakeClient,
			}

			// Test lazy loading by calling a method that triggers config loading
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{},
				},
			}

			result := processor.IsDRAEnabled(context.Background(), pod)
			assert.Equal(t, tt.expected, result)

			// Verify config was loaded
			assert.True(t, processor.configLoaded)
		})
	}
}
