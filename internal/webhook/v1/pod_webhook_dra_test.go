package v1

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
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

func TestDRAProcessor_convertToResourceClaim(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))
	require.NoError(t, resourcev1beta2.AddToScheme(scheme))

	// Create a SchedulingConfigTemplate with DRA config
	template := &tfv1.SchedulingConfigTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-template",
		},
		Spec: tfv1.SchedulingConfigTemplateSpec{
			DRA: &tfv1.DRAConfig{
				Enable:        &[]bool{true}[0],
				ResourceClass: "custom.tensorfusion.ai/gpu",
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

	// Initialize DRA config to set up the resource class cache
	err := processor.InitializeDRAConfig(context.Background())
	require.NoError(t, err)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:         "test-pod",
			Namespace:    "test-namespace",
			GenerateName: "test-pod-",
			UID:          types.UID("test-uid"),
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

	claim, err := processor.convertToResourceClaim(pod, tfInfo)
	require.NoError(t, err)
	require.NotNil(t, claim)

	// Verify claim structure
	assert.Contains(t, claim.Name, "test-pod-")
	assert.Contains(t, claim.Name, "-gpu-claim")
	assert.Equal(t, "test-namespace", claim.Namespace)
	assert.Equal(t, "resource.k8s.io/v1beta2", claim.APIVersion)
	assert.Equal(t, "ResourceClaim", claim.Kind)

	// Verify labels instead of owner references (since we removed owner references during admission)
	require.NotNil(t, claim.Labels)
	assert.Equal(t, "test-pod-", claim.Labels["tensorfusion.ai/pod"]) // Uses GenerateName as podIdentifier
	assert.Equal(t, "gpu", claim.Labels["tensorfusion.ai/claim-for"])

	// Verify device claim
	require.Len(t, claim.Spec.Devices.Requests, 1)
	deviceReq := claim.Spec.Devices.Requests[0]
	assert.Equal(t, "gpu", deviceReq.Name)

	// Verify ExactDeviceRequest structure
	require.NotNil(t, deviceReq.Exactly)
	exactReq := deviceReq.Exactly
	assert.Equal(t, "custom.tensorfusion.ai/gpu", exactReq.DeviceClassName) // Uses cached resource class from template
	assert.Equal(t, int64(1), exactReq.Count)

	// Verify CEL selector
	require.Len(t, exactReq.Selectors, 1)
	require.NotNil(t, exactReq.Selectors[0].CEL)

	// The simplified CEL selector should only contain basic resource requirements
	celExpression := exactReq.Selectors[0].CEL.Expression

	// Verify it contains the expected resource filters (simplified version)
	assert.Contains(t, celExpression, `device.attributes["tflops"].quantity >= quantity("10")`)
	assert.Contains(t, celExpression, `device.attributes["vram"].quantity >= quantity("8Gi")`)

	// Verify conditions are combined with AND
	assert.Contains(t, celExpression, " && ")
}

func TestDRAProcessor_injectResourceClaimRef(t *testing.T) {
	processor := &DRAProcessor{}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-namespace",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{Name: "container1"},
				{Name: "container2"},
			},
		},
	}

	claim := &resourcev1beta2.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-claim",
			Namespace: "test-namespace",
		},
	}

	containerIndices := []int{0, 1}

	processor.injectResourceClaimRef(pod, claim, containerIndices)

	// Verify pod resource claims
	require.Len(t, pod.Spec.ResourceClaims, 1)
	podClaim := pod.Spec.ResourceClaims[0]
	assert.Equal(t, "gpu-claim", podClaim.Name)
	require.NotNil(t, podClaim.ResourceClaimName)
	assert.Equal(t, "test-claim", *podClaim.ResourceClaimName)

	// Verify container resource claims
	for _, idx := range containerIndices {
		container := pod.Spec.Containers[idx]
		require.Len(t, container.Resources.Claims, 1)
		assert.Equal(t, "gpu-claim", container.Resources.Claims[0].Name)
	}

	// Verify annotations
	require.NotNil(t, pod.Annotations)
	assert.Equal(t, constants.TrueStringValue, pod.Annotations[constants.DRAEnabledAnnotation])
}

func TestDRAProcessor_createResourceClaim(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, resourcev1beta2.AddToScheme(scheme))

	tests := []struct {
		name          string
		existingClaim *resourcev1beta2.ResourceClaim
		expectError   bool
		errorType     string
	}{
		{
			name:        "successful creation",
			expectError: false,
		},
		{
			name: "claim already exists with same pod",
			existingClaim: &resourcev1beta2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "test-namespace",
					Labels: map[string]string{
						"tensorfusion.ai/pod":       "test-pod",
						"tensorfusion.ai/claim-for": "gpu",
					},
				},
			},
			expectError: false,
		},
		{
			name: "claim already exists with different pod",
			existingClaim: &resourcev1beta2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "test-namespace",
					Labels: map[string]string{
						"tensorfusion.ai/pod":       "different-pod",
						"tensorfusion.ai/claim-for": "gpu",
					},
				},
			},
			expectError: true,
			errorType:   "conflict",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var fakeClient client.Client
			if tt.existingClaim != nil {
				fakeClient = fake.NewClientBuilder().
					WithScheme(scheme).
					WithObjects(tt.existingClaim).
					Build()
			} else {
				fakeClient = fake.NewClientBuilder().
					WithScheme(scheme).
					Build()
			}

			processor := &DRAProcessor{
				Client: fakeClient,
			}

			claim := &resourcev1beta2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "test-namespace",
					Labels: map[string]string{
						"tensorfusion.ai/pod":       "test-pod",
						"tensorfusion.ai/claim-for": "gpu",
					},
				},
			}

			err := processor.createResourceClaim(context.Background(), claim)

			if tt.expectError {
				require.Error(t, err)
				if tt.errorType == "conflict" {
					assert.Contains(t, err.Error(), "already exists for a different pod")
				}
			} else {
				require.NoError(t, err)
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
							Enable:        &[]bool{true}[0],
							ResourceClass: "test.ai/gpu",
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
