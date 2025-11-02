package v1

import (
	"context"
	"fmt"
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
		poolDRAEnabled bool
		poolName       string
		podAnnotations map[string]string
		expected       bool
	}{
		{
			name:           "pool DRA enabled, no pod annotation",
			poolDRAEnabled: true,
			poolName:       "test-pool",
			expected:       true,
		},
		{
			name:           "pool DRA disabled, no pod annotation",
			poolDRAEnabled: false,
			poolName:       "test-pool",
			expected:       false,
		},
		{
			name:           "pool DRA disabled, pod annotation enabled",
			poolDRAEnabled: false,
			poolName:       "test-pool",
			podAnnotations: map[string]string{
				constants.DRAEnabledAnnotation: constants.TrueStringValue,
			},
			expected: true,
		},
		{
			name:           "pool DRA enabled, pod annotation disabled",
			poolDRAEnabled: true,
			poolName:       "test-pool",
			podAnnotations: map[string]string{
				constants.DRAEnabledAnnotation: constants.FalseStringValue,
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			processor := &DRAProcessor{
				poolConfigs: map[string]*PoolDRAConfig{
					tt.poolName: {
						EnableDRA:                 tt.poolDRAEnabled,
						ResourceClaimTemplateName: "test-template",
					},
				},
			}

			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: tt.podAnnotations,
				},
			}

			result := processor.IsDRAEnabled(context.Background(), pod, tt.poolName)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestDRAProcessor_HandleDRAAdmission(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	// Create a GPUPool with DRA config
	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-pool",
		},
		Spec: tfv1.GPUPoolSpec{
			DRAConfig: &tfv1.DRAConfig{
				Enable:                    &[]bool{true}[0],
				ResourceClaimTemplateName: "custom-gpu-template",
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(pool).
		Build()

	processor := NewDRAProcessor(fakeClient)

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
			PoolName: "test-pool",
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
	driverDomain := constants.DRADriverName
	assert.Contains(t, celExpression, fmt.Sprintf(`"%s" in device.capacity["%s"] && device.capacity["%s"]["%s"].compareTo(quantity("10")) >= 0`,
		constants.DRACapacityTFlops,
		driverDomain,
		driverDomain,
		constants.DRACapacityTFlops,
	))
	assert.Contains(t, celExpression, fmt.Sprintf(`"%s" in device.capacity["%s"] && device.capacity["%s"]["%s"].compareTo(quantity("8Gi")) >= 0`,
		constants.DRACapacityVRAM,
		driverDomain,
		driverDomain,
		constants.DRACapacityVRAM,
	))

	// Verify DRA enabled annotation is set
	assert.Equal(t, constants.TrueStringValue, pod.Annotations[constants.DRAEnabledAnnotation])

	// Verify ResourceClaimTemplate reference is added to Pod
	require.Len(t, pod.Spec.ResourceClaims, 1)
	podClaim := pod.Spec.ResourceClaims[0]
	assert.Equal(t, constants.DRAClaimDefineName, podClaim.Name)
	require.NotNil(t, podClaim.ResourceClaimTemplateName)
	assert.Equal(t, "custom-gpu-template", *podClaim.ResourceClaimTemplateName)

	// Verify processor has cached the pool configuration
	config, exists := processor.poolConfigs["test-pool"]
	require.True(t, exists)
	assert.Equal(t, "custom-gpu-template", config.ResourceClaimTemplateName)
	assert.True(t, config.EnableDRA)
}

func TestBuildCELSelector(t *testing.T) {
	driverDomain := constants.DRADriverName
	attrEquals := func(attr, value string) string {
		return fmt.Sprintf(`"%s" in device.attributes["%s"] && device.attributes["%s"]["%s"] == "%s"`, attr, driverDomain, driverDomain, attr, value)
	}
	capacityGte := func(capacity, value string) string {
		return fmt.Sprintf(`"%s" in device.capacity["%s"] && device.capacity["%s"]["%s"].compareTo(quantity("%s")) >= 0`, capacity, driverDomain, driverDomain, capacity, value)
	}
	phaseCondition := fmt.Sprintf(`"%s" in device.attributes["%s"] && device.attributes["%s"]["%s"] in ["%s", "%s"]`,
		constants.DRAAttributePhase,
		driverDomain,
		driverDomain,
		constants.DRAAttributePhase,
		constants.PhaseRunning,
		constants.PhasePending,
	)

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
				attrEquals(constants.DRAAttributeModel, "H100"),
				capacityGte(constants.DRACapacityTFlops, "20"),
				capacityGte(constants.DRACapacityVRAM, "16Gi"),
				phaseCondition,
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
				attrEquals(constants.DRAAttributeModel, "A100"),
				attrEquals(constants.DRAAttributePoolName, "high-priority"),
				capacityGte(constants.DRACapacityTFlops, "10"),
				capacityGte(constants.DRACapacityVRAM, "8Gi"),
				attrEquals(constants.DRAAttributeQoS, "high"),
				phaseCondition,
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
				phaseCondition,
			},
			unexpectedConditions: []string{
				fmt.Sprintf(`"%s" in device.attributes["%s"]`, constants.DRAAttributeModel, driverDomain),
				fmt.Sprintf(`"%s" in device.attributes["%s"]`, constants.DRAAttributePoolName, driverDomain),
				fmt.Sprintf(`"%s" in device.capacity["%s"]`, constants.DRACapacityTFlops, driverDomain),
				fmt.Sprintf(`"%s" in device.capacity["%s"]`, constants.DRACapacityVRAM, driverDomain),
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
				phaseCondition,
			},
			unexpectedConditions: []string{
				fmt.Sprintf(`"%s" in device.attributes["%s"]`, constants.DRAAttributeModel, driverDomain),
				fmt.Sprintf(`"%s" in device.attributes["%s"]`, constants.DRAAttributePoolName, driverDomain),
				fmt.Sprintf(`"%s" in device.capacity["%s"]`, constants.DRACapacityTFlops, driverDomain),
				fmt.Sprintf(`"%s" in device.capacity["%s"]`, constants.DRACapacityVRAM, driverDomain),
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
	require.NoError(t, corev1.AddToScheme(scheme))

	tests := []struct {
		name     string
		pools    []tfv1.GPUPool
		poolName string
		expected bool
	}{
		{
			name: "DRA enabled in pool",
			pools: []tfv1.GPUPool{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "test-pool"},
					Spec: tfv1.GPUPoolSpec{
						DRAConfig: &tfv1.DRAConfig{
							Enable:                    &[]bool{true}[0],
							ResourceClaimTemplateName: "test-gpu-template",
						},
					},
				},
			},
			poolName: "test-pool",
			expected: true,
		},
		{
			name: "DRA disabled in pool",
			pools: []tfv1.GPUPool{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "test-pool"},
					Spec: tfv1.GPUPoolSpec{
						DRAConfig: &tfv1.DRAConfig{
							Enable: &[]bool{false}[0],
						},
					},
				},
			},
			poolName: "test-pool",
			expected: false,
		},
		{
			name:     "no pool found",
			poolName: "nonexistent-pool",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			objects := make([]client.Object, len(tt.pools))
			for i := range tt.pools {
				objects[i] = &tt.pools[i]
			}

			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(objects...).
				Build()

			processor := NewDRAProcessor(fakeClient)

			// Test lazy loading by calling a method that triggers config loading
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{},
				},
			}

			result := processor.IsDRAEnabled(context.Background(), pod, tt.poolName)
			assert.Equal(t, tt.expected, result)

			// Verify config was loaded and cached
			_, exists := processor.poolConfigs[tt.poolName]
			assert.True(t, exists, "pool config should be cached")
		})
	}
}
