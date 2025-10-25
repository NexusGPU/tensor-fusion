package dra

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

func TestResourceClaimReconciler_Reconcile(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, resourcev1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	tests := []struct {
		name           string
		resourceClaim  *resourcev1.ResourceClaim
		pod            *corev1.Pod
		expectedResult ctrl.Result
		expectError    bool
		expectUpdate   bool
	}{
		{
			name:           "ResourceClaim not found",
			expectedResult: ctrl.Result{},
			expectError:    false,
		},
		{
			name: "ResourceClaim without TensorFusion label",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
			},
			expectedResult: ctrl.Result{},
			expectError:    false,
		},
		{
			name: "ResourceClaim with wrong label value",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionResourceClaimTemplateLabel: "false",
					},
				},
			},
			expectedResult: ctrl.Result{},
			expectError:    false,
		},
		{
			name: "ResourceClaim without owner Pod",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionResourceClaimTemplateLabel: constants.TrueStringValue,
					},
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 1,
								},
							},
						},
					},
				},
			},
			expectedResult: ctrl.Result{RequeueAfter: constants.PendingRequeueDuration},
			expectError:    false,
		},
		{
			name: "Owner Pod without CEL annotation",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionResourceClaimTemplateLabel: constants.TrueStringValue,
					},
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "v1",
							Kind:       "Pod",
							Name:       "test-pod",
							UID:        "pod-uid-123",
						},
					},
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 1,
								},
							},
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					UID:       "pod-uid-123",
					Annotations: map[string]string{
						constants.TFLOPSRequestAnnotation: "10",
						constants.VRAMRequestAnnotation:   "16Gi",
					},
				},
			},
			expectedResult: ctrl.Result{},
			expectError:    false,
		},
		{
			name: "Successful CEL expression update",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionResourceClaimTemplateLabel: constants.TrueStringValue,
					},
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "v1",
							Kind:       "Pod",
							Name:       "test-pod",
							UID:        "pod-uid-123",
						},
					},
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 1,
								},
							},
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					UID:       "pod-uid-123",
					Annotations: map[string]string{
						constants.DRACelExpressionAnnotation: `device.attributes["tflops"].quantity >= quantity("10")`,
						constants.TFLOPSRequestAnnotation:    "10",
						constants.VRAMRequestAnnotation:      "16Gi",
					},
				},
			},
			expectedResult: ctrl.Result{},
			expectError:    false,
			expectUpdate:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objects []runtime.Object
			if tt.resourceClaim != nil {
				objects = append(objects, tt.resourceClaim)
			}
			if tt.pod != nil {
				objects = append(objects, tt.pod)
			}

			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithRuntimeObjects(objects...).
				Build()

			reconciler := &ResourceClaimReconciler{
				Client: fakeClient,
				Scheme: scheme,
			}

			req := ctrl.Request{
				NamespacedName: types.NamespacedName{
					Name:      "test-claim",
					Namespace: "default",
				},
			}

			result, err := reconciler.Reconcile(context.Background(), req)

			if tt.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}

			assert.Equal(t, tt.expectedResult, result)

			// Check if ResourceClaim was updated with CEL expression
			if tt.expectUpdate && tt.resourceClaim != nil {
				updatedClaim := &resourcev1.ResourceClaim{}
				err := fakeClient.Get(context.Background(), types.NamespacedName{
					Name:      tt.resourceClaim.Name,
					Namespace: tt.resourceClaim.Namespace,
				}, updatedClaim)
				require.NoError(t, err)

				require.Len(t, updatedClaim.Spec.Devices.Requests, 1)
				deviceReq := updatedClaim.Spec.Devices.Requests[0]
				require.NotNil(t, deviceReq.Exactly)
				require.Len(t, deviceReq.Exactly.Selectors, 1)
				require.NotNil(t, deviceReq.Exactly.Selectors[0].CEL)
				assert.Equal(t, `device.attributes["tflops"].quantity >= quantity("10")`, deviceReq.Exactly.Selectors[0].CEL.Expression)
			}
		})
	}
}

func TestResourceClaimReconciler_findOwnerPod(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, resourcev1.AddToScheme(scheme))

	tests := []struct {
		name          string
		resourceClaim *resourcev1.ResourceClaim
		pod           *corev1.Pod
		expectedPod   *corev1.Pod
		expectError   bool
	}{
		{
			name: "No owner references",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
			},
			expectedPod: nil,
			expectError: false,
		},
		{
			name: "No Pod owner reference",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "Deployment",
							Name:       "test-deployment",
							UID:        "deployment-uid-123",
						},
					},
				},
			},
			expectedPod: nil,
			expectError: false,
		},
		{
			name: "Pod owner not found",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "v1",
							Kind:       "Pod",
							Name:       "nonexistent-pod",
							UID:        "pod-uid-123",
						},
					},
				},
			},
			expectedPod: nil,
			expectError: false,
		},
		{
			name: "Pod UID mismatch",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "v1",
							Kind:       "Pod",
							Name:       "test-pod",
							UID:        "pod-uid-123",
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					UID:       "different-uid",
				},
			},
			expectedPod: nil,
			expectError: true,
		},
		{
			name: "Successful Pod lookup",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "v1",
							Kind:       "Pod",
							Name:       "test-pod",
							UID:        "pod-uid-123",
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					UID:       "pod-uid-123",
				},
			},
			expectedPod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					UID:       "pod-uid-123",
				},
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objects []runtime.Object
			if tt.pod != nil {
				objects = append(objects, tt.pod)
			}

			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithRuntimeObjects(objects...).
				Build()

			reconciler := &ResourceClaimReconciler{
				Client: fakeClient,
				Scheme: scheme,
			}

			pod, err := reconciler.findOwnerPod(context.Background(), tt.resourceClaim)

			if tt.expectError {
				require.Error(t, err)
				assert.Nil(t, pod)
			} else {
				require.NoError(t, err)
				if tt.expectedPod == nil {
					assert.Nil(t, pod)
				} else {
					require.NotNil(t, pod)
					assert.Equal(t, tt.expectedPod.Name, pod.Name)
					assert.Equal(t, tt.expectedPod.Namespace, pod.Namespace)
					assert.Equal(t, tt.expectedPod.UID, pod.UID)
				}
			}
		})
	}
}

func TestResourceClaimReconciler_updateResourceClaimCEL(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, resourcev1.AddToScheme(scheme))

	tests := []struct {
		name          string
		resourceClaim *resourcev1.ResourceClaim
		celExpression string
		expectError   bool
		expectUpdate  bool
	}{
		{
			name: "No device requests",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{},
					},
				},
			},
			celExpression: `device.attributes["tflops"].quantity >= quantity("10")`,
			expectError:   true,
		},
		{
			name: "No ExactDeviceRequest",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								// Exactly is nil
							},
						},
					},
				},
			},
			celExpression: `device.attributes["tflops"].quantity >= quantity("10")`,
			expectError:   true,
		},
		{
			name: "CEL expression already set correctly",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 1,
									Selectors: []resourcev1.DeviceSelector{
										{
											CEL: &resourcev1.CELDeviceSelector{
												Expression: `device.attributes["tflops"].quantity >= quantity("10")`,
											},
										},
									},
								},
							},
						},
					},
				},
			},
			celExpression: `device.attributes["tflops"].quantity >= quantity("10")`,
			expectError:   false,
			expectUpdate:  false, // No update needed
		},
		{
			name: "Successful CEL expression update - empty selectors",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 1,
								},
							},
						},
					},
				},
			},
			celExpression: `device.attributes["tflops"].quantity >= quantity("10")`,
			expectError:   false,
			expectUpdate:  true,
		},
		{
			name: "Successful CEL expression update - nil CEL",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 1,
									Selectors: []resourcev1.DeviceSelector{
										{
											// CEL is nil
										},
									},
								},
							},
						},
					},
				},
			},
			celExpression: `device.attributes["vram"].quantity >= quantity("8Gi")`,
			expectError:   false,
			expectUpdate:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithRuntimeObjects(tt.resourceClaim).
				Build()

			reconciler := &ResourceClaimReconciler{
				Client: fakeClient,
				Scheme: scheme,
			}

			mockPod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.DRACelExpressionAnnotation: tt.celExpression,
					},
				},
			}
			updated, err := reconciler.updateResourceClaimCEL(tt.resourceClaim, mockPod)

			if tt.expectError {
				require.Error(t, err)
				assert.False(t, updated)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.expectUpdate, updated)

				if tt.expectUpdate {
					// Verify the CEL expression was set correctly
					require.Len(t, tt.resourceClaim.Spec.Devices.Requests, 1)
					deviceReq := tt.resourceClaim.Spec.Devices.Requests[0]
					require.NotNil(t, deviceReq.Exactly)
					require.Len(t, deviceReq.Exactly.Selectors, 1)
					require.NotNil(t, deviceReq.Exactly.Selectors[0].CEL)
					assert.Equal(t, tt.celExpression, deviceReq.Exactly.Selectors[0].CEL.Expression)
				}
			}
		})
	}
}

func TestResourceClaimReconciler_updateCapacityRequest(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, resourcev1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	tests := []struct {
		name          string
		resourceClaim *resourcev1.ResourceClaim
		pod           *corev1.Pod
		expectError   bool
		expectUpdate  bool
	}{
		{
			name: "No device requests",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					Annotations: map[string]string{
						constants.TFLOPSRequestAnnotation: "10",
						constants.VRAMRequestAnnotation:   "16Gi",
					},
				},
			},
			expectError: true,
		},
		{
			name: "No ExactDeviceRequest",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								// Exactly is nil
							},
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					Annotations: map[string]string{
						constants.TFLOPSRequestAnnotation: "10",
						constants.VRAMRequestAnnotation:   "16Gi",
					},
				},
			},
			expectError: true,
		},
		{
			name: "Successful capacity request update - nil Requests map",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 1,
									// Capacity.Requests is nil
								},
							},
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					Annotations: map[string]string{
						constants.TFLOPSRequestAnnotation: "10",
						constants.VRAMRequestAnnotation:   "16Gi",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "main",
							Image: "test:latest",
						},
					},
				},
			},
			expectError:  false,
			expectUpdate: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objects []runtime.Object
			if tt.pod != nil {
				objects = append(objects, tt.pod)
			}
			if tt.resourceClaim != nil {
				objects = append(objects, tt.resourceClaim)
			}

			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithRuntimeObjects(objects...).
				Build()

			reconciler := &ResourceClaimReconciler{
				Client: fakeClient,
				Scheme: scheme,
			}

			updated, err := reconciler.updateCapacityRequest(tt.resourceClaim, tt.pod)

			if tt.expectError {
				require.Error(t, err)
				assert.False(t, updated)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.expectUpdate, updated)

				if tt.expectUpdate {
					// Verify capacity requests were set
					require.Len(t, tt.resourceClaim.Spec.Devices.Requests, 1)
					deviceReq := tt.resourceClaim.Spec.Devices.Requests[0]
					require.NotNil(t, deviceReq.Exactly)
					require.NotNil(t, deviceReq.Exactly.Capacity.Requests)

					// Check that tflops and vram are set
					_, hasTflops := deviceReq.Exactly.Capacity.Requests[constants.DRACapacityTFlops]
					_, hasVram := deviceReq.Exactly.Capacity.Requests[constants.DRACapacityVRAM]
					assert.True(t, hasTflops, "TFlops capacity should be set")
					assert.True(t, hasVram, "VRAM capacity should be set")
				}
			}
		})
	}
}

func TestResourceClaimReconciler_IdempotencyCheck(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, resourcev1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	// Create a ResourceClaim that is already allocated
	resourceClaim := &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-claim",
			Namespace: "default",
			Labels: map[string]string{
				constants.TensorFusionResourceClaimTemplateLabel: constants.TrueStringValue,
			},
		},
		Spec: resourcev1.ResourceClaimSpec{
			Devices: resourcev1.DeviceClaim{
				Requests: []resourcev1.DeviceRequest{
					{
						Name: "gpu-request",
						Exactly: &resourcev1.ExactDeviceRequest{
							Count: 1,
						},
					},
				},
			},
		},
		Status: resourcev1.ResourceClaimStatus{
			Allocation: &resourcev1.AllocationResult{
				Devices: resourcev1.DeviceAllocationResult{
					Results: []resourcev1.DeviceRequestAllocationResult{
						{
							Request: "gpu-request",
							Device:  "gpu-0",
						},
					},
				},
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithRuntimeObjects(resourceClaim).
		Build()

	reconciler := &ResourceClaimReconciler{
		Client: fakeClient,
		Scheme: scheme,
	}

	req := ctrl.Request{
		NamespacedName: types.NamespacedName{
			Name:      "test-claim",
			Namespace: "default",
		},
	}

	result, err := reconciler.Reconcile(context.Background(), req)

	// Should not error and should not requeue
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	// Verify ResourceClaim was not modified
	updatedClaim := &resourcev1.ResourceClaim{}
	err = fakeClient.Get(context.Background(), types.NamespacedName{
		Name:      resourceClaim.Name,
		Namespace: resourceClaim.Namespace,
	}, updatedClaim)
	require.NoError(t, err)

	// Status should still have allocation
	require.NotNil(t, updatedClaim.Status.Allocation)
}

func TestResourceClaimReconciler_EmptyCELAnnotation(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, resourcev1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	resourceClaim := &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-claim",
			Namespace: "default",
			Labels: map[string]string{
				constants.TensorFusionResourceClaimTemplateLabel: constants.TrueStringValue,
			},
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "v1",
					Kind:       "Pod",
					Name:       "test-pod",
					UID:        "pod-uid-123",
				},
			},
		},
		Spec: resourcev1.ResourceClaimSpec{
			Devices: resourcev1.DeviceClaim{
				Requests: []resourcev1.DeviceRequest{
					{
						Name: "gpu-request",
						Exactly: &resourcev1.ExactDeviceRequest{
							Count: 1,
						},
					},
				},
			},
		},
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
			UID:       "pod-uid-123",
			// No CEL annotation
			Annotations: map[string]string{
				constants.TFLOPSRequestAnnotation: "10",
				constants.VRAMRequestAnnotation:   "16Gi",
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "main",
					Image: "test:latest",
				},
			},
		},
	}

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithRuntimeObjects(resourceClaim, pod).
		Build()

	reconciler := &ResourceClaimReconciler{
		Client: fakeClient,
		Scheme: scheme,
	}

	req := ctrl.Request{
		NamespacedName: types.NamespacedName{
			Name:      "test-claim",
			Namespace: "default",
		},
	}

	result, err := reconciler.Reconcile(context.Background(), req)

	// Should succeed without error
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	// Verify ResourceClaim was updated (capacity should be set even without CEL)
	updatedClaim := &resourcev1.ResourceClaim{}
	err = fakeClient.Get(context.Background(), types.NamespacedName{
		Name:      resourceClaim.Name,
		Namespace: resourceClaim.Namespace,
	}, updatedClaim)
	require.NoError(t, err)

	// CEL should not be set (empty annotation means no CEL)
	deviceReq := updatedClaim.Spec.Devices.Requests[0]
	if len(deviceReq.Exactly.Selectors) > 0 && deviceReq.Exactly.Selectors[0].CEL != nil {
		assert.Empty(t, deviceReq.Exactly.Selectors[0].CEL.Expression)
	}

	// But capacity should be set
	require.NotNil(t, deviceReq.Exactly.Capacity.Requests)
	_, hasTflops := deviceReq.Exactly.Capacity.Requests[constants.DRACapacityTFlops]
	_, hasVram := deviceReq.Exactly.Capacity.Requests[constants.DRACapacityVRAM]
	assert.True(t, hasTflops)
	assert.True(t, hasVram)
}

func TestResourceClaimReconciler_updateDeviceCount(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, resourcev1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	tests := []struct {
		name          string
		resourceClaim *resourcev1.ResourceClaim
		pod           *corev1.Pod
		expectError   bool
		expectUpdate  bool
		expectedCount int64
	}{
		{
			name: "No device requests",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.GpuCountAnnotation: "2",
					},
				},
			},
			expectError: true,
		},
		{
			name: "No ExactDeviceRequest",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								// Exactly is nil
							},
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.GpuCountAnnotation: "2",
					},
				},
			},
			expectError: true,
		},
		{
			name: "GPU count defaults to 1 when annotation missing",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 0, // Will be updated to 1
								},
							},
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						// No GpuCountAnnotation
					},
				},
			},
			expectError:   false,
			expectUpdate:  true,
			expectedCount: 1,
		},
		{
			name: "GPU count already set correctly - no update",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 2,
								},
							},
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.GpuCountAnnotation: "2",
					},
				},
			},
			expectError:   false,
			expectUpdate:  false,
			expectedCount: 2,
		},
		{
			name: "Successful GPU count update",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 1,
								},
							},
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.GpuCountAnnotation: "4",
					},
				},
			},
			expectError:   false,
			expectUpdate:  true,
			expectedCount: 4,
		},
		{
			name: "Invalid GPU count - not a number",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 1,
								},
							},
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.GpuCountAnnotation: "invalid",
					},
				},
			},
			expectError: true,
		},
		{
			name: "Invalid GPU count - zero",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 1,
								},
							},
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.GpuCountAnnotation: "0",
					},
				},
			},
			expectError: true,
		},
		{
			name: "Invalid GPU count - negative",
			resourceClaim: &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1.ExactDeviceRequest{
									Count: 1,
								},
							},
						},
					},
				},
			},
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.GpuCountAnnotation: "-1",
					},
				},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var objects []runtime.Object
			if tt.pod != nil {
				objects = append(objects, tt.pod)
			}
			if tt.resourceClaim != nil {
				objects = append(objects, tt.resourceClaim)
			}

			fakeClient := fake.NewClientBuilder().
				WithScheme(scheme).
				WithRuntimeObjects(objects...).
				Build()

			reconciler := &ResourceClaimReconciler{
				Client: fakeClient,
				Scheme: scheme,
			}

			updated, err := reconciler.updateDeviceCount(tt.resourceClaim, tt.pod)

			if tt.expectError {
				require.Error(t, err)
				assert.False(t, updated)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.expectUpdate, updated)

				if tt.expectUpdate || tt.expectedCount > 0 {
					// Verify count was set correctly
					require.Len(t, tt.resourceClaim.Spec.Devices.Requests, 1)
					deviceReq := tt.resourceClaim.Spec.Devices.Requests[0]
					require.NotNil(t, deviceReq.Exactly)
					assert.Equal(t, tt.expectedCount, deviceReq.Exactly.Count)
				}
			}
		})
	}
}
