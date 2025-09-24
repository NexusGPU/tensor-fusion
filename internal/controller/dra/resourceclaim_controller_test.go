package dra

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1beta2 "k8s.io/api/resource/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

func TestResourceClaimReconciler_Reconcile(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, resourcev1beta2.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))

	tests := []struct {
		name           string
		resourceClaim  *resourcev1beta2.ResourceClaim
		pod            *corev1.Pod
		expectedResult ctrl.Result
		expectError    bool
		expectUpdate   bool
	}{
		{
			name: "ResourceClaim not found",
			expectedResult: ctrl.Result{},
			expectError:    false,
		},
		{
			name: "ResourceClaim without TensorFusion label",
			resourceClaim: &resourcev1beta2.ResourceClaim{
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
			resourceClaim: &resourcev1beta2.ResourceClaim{
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
			resourceClaim: &resourcev1beta2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionResourceClaimTemplateLabel: constants.TrueStringValue,
					},
				},
				Spec: resourcev1beta2.ResourceClaimSpec{
					Devices: resourcev1beta2.DeviceClaim{
						Requests: []resourcev1beta2.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1beta2.ExactDeviceRequest{
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
			resourceClaim: &resourcev1beta2.ResourceClaim{
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
				Spec: resourcev1beta2.ResourceClaimSpec{
					Devices: resourcev1beta2.DeviceClaim{
						Requests: []resourcev1beta2.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1beta2.ExactDeviceRequest{
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
				},
			},
			expectedResult: ctrl.Result{},
			expectError:    false,
		},
		{
			name: "Successful CEL expression update",
			resourceClaim: &resourcev1beta2.ResourceClaim{
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
				Spec: resourcev1beta2.ResourceClaimSpec{
					Devices: resourcev1beta2.DeviceClaim{
						Requests: []resourcev1beta2.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1beta2.ExactDeviceRequest{
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
				updatedClaim := &resourcev1beta2.ResourceClaim{}
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
	require.NoError(t, resourcev1beta2.AddToScheme(scheme))

	tests := []struct {
		name          string
		resourceClaim *resourcev1beta2.ResourceClaim
		pod           *corev1.Pod
		expectedPod   *corev1.Pod
		expectError   bool
	}{
		{
			name: "No owner references",
			resourceClaim: &resourcev1beta2.ResourceClaim{
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
			resourceClaim: &resourcev1beta2.ResourceClaim{
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
			resourceClaim: &resourcev1beta2.ResourceClaim{
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
			resourceClaim: &resourcev1beta2.ResourceClaim{
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
			resourceClaim: &resourcev1beta2.ResourceClaim{
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
	require.NoError(t, resourcev1beta2.AddToScheme(scheme))

	tests := []struct {
		name          string
		resourceClaim *resourcev1beta2.ResourceClaim
		celExpression string
		expectError   bool
		expectUpdate  bool
	}{
		{
			name: "No device requests",
			resourceClaim: &resourcev1beta2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1beta2.ResourceClaimSpec{
					Devices: resourcev1beta2.DeviceClaim{
						Requests: []resourcev1beta2.DeviceRequest{},
					},
				},
			},
			celExpression: `device.attributes["tflops"].quantity >= quantity("10")`,
			expectError:   true,
		},
		{
			name: "No ExactDeviceRequest",
			resourceClaim: &resourcev1beta2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1beta2.ResourceClaimSpec{
					Devices: resourcev1beta2.DeviceClaim{
						Requests: []resourcev1beta2.DeviceRequest{
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
			resourceClaim: &resourcev1beta2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1beta2.ResourceClaimSpec{
					Devices: resourcev1beta2.DeviceClaim{
						Requests: []resourcev1beta2.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1beta2.ExactDeviceRequest{
									Count: 1,
									Selectors: []resourcev1beta2.DeviceSelector{
										{
											CEL: &resourcev1beta2.CELDeviceSelector{
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
			resourceClaim: &resourcev1beta2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1beta2.ResourceClaimSpec{
					Devices: resourcev1beta2.DeviceClaim{
						Requests: []resourcev1beta2.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1beta2.ExactDeviceRequest{
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
			resourceClaim: &resourcev1beta2.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-claim",
					Namespace: "default",
				},
				Spec: resourcev1beta2.ResourceClaimSpec{
					Devices: resourcev1beta2.DeviceClaim{
						Requests: []resourcev1beta2.DeviceRequest{
							{
								Name: "gpu-request",
								Exactly: &resourcev1beta2.ExactDeviceRequest{
									Count: 1,
									Selectors: []resourcev1beta2.DeviceSelector{
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

			err := reconciler.updateResourceClaimCEL(context.Background(), tt.resourceClaim, tt.celExpression)

			if tt.expectError {
				require.Error(t, err)
			} else {
				require.NoError(t, err)

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