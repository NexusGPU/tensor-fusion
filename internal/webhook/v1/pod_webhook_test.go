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

package v1

import (
	"context"
	"encoding/json"
	"net/http"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/config"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
	"github.com/NexusGPU/tensor-fusion-operator/internal/scheduler"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

var _ = Describe("TensorFusionPodMutator", func() {
	var (
		mutator   *TensorFusionPodMutator
		ctx       context.Context
		scheme    *runtime.Scheme
		decoder   admission.Decoder
		k8sclient client.Client
	)

	BeforeEach(func() {
		ctx = context.Background()
		scheme = runtime.NewScheme()
		Expect(corev1.AddToScheme(scheme)).To(Succeed())
		Expect(tfv1.AddToScheme(scheme)).To(Succeed())

		decoder = admission.NewDecoder(scheme)
		k8sclient = k8sClient

		mutator = &TensorFusionPodMutator{
			Client:    k8sclient,
			scheduler: scheduler.NewScheduler(k8sclient),
		}
		Expect(mutator.InjectDecoder(decoder)).To(Succeed())

	})

	Context("Handle", func() {
		It("should successfully mutate a pod with TF resources", func() {
			// Set up a client profile for testing
			clientProfile := &tfv1.ClientProfile{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-profile-handle",
					Namespace: "default",
				},
				Spec: tfv1.ClientProfileSpec{
					PoolName: "mock",
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("1Gi"),
						},
						Limits: tfv1.Resource{
							Tflops: resource.MustParse("100"),
							Vram:   resource.MustParse("16Gi"),
						},
					},
				},
			}

			Expect(k8sclient.Create(ctx, clientProfile)).To(Succeed())

			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.GpuPoolKey:                "mock",
						constants.ClientProfileAnnotation:   "test-profile-handle",
						constants.InjectContainerAnnotation: "main",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "main",
							Image: "test-image",
							Env: []corev1.EnvVar{
								{
									Name: "A",
									ValueFrom: &corev1.EnvVarSource{
										FieldRef: &corev1.ObjectFieldSelector{
											FieldPath: "metadata.namespace",
										},
									},
								},
								{
									Name:  "B",
									Value: "B",
								},
							},
						},
					},
				},
			}
			podBytes, err := json.Marshal(pod)
			Expect(err).NotTo(HaveOccurred())

			req := admission.Request{
				AdmissionRequest: admissionv1.AdmissionRequest{
					Object: runtime.RawExtension{
						Raw: podBytes,
					},
					Operation: admissionv1.Create,
				},
			}

			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeTrue())
			Expect(resp.Patches).NotTo(BeEmpty())
			Expect(err).NotTo(HaveOccurred())
		})

		It("should handle pods without TF requirements", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-no-tf",
					Namespace: "default",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "main",
							Image: "test-image",
						},
					},
				},
			}

			podBytes, err := json.Marshal(pod)
			Expect(err).NotTo(HaveOccurred())

			req := admission.Request{
				AdmissionRequest: admissionv1.AdmissionRequest{
					Object: runtime.RawExtension{
						Raw: podBytes,
					},
					Operation: admissionv1.Create,
				},
			}

			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeFalse())
			Expect(resp.Patches).To(BeEmpty())
		})

		It("should handle invalid pod specification", func() {
			req := admission.Request{
				AdmissionRequest: admissionv1.AdmissionRequest{
					Object: runtime.RawExtension{
						Raw: []byte("invalid json"),
					},
					Operation: admissionv1.Create,
				},
			}

			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeFalse())
			Expect(resp.Result.Code).To(Equal(int32(http.StatusBadRequest)))
		})
	})

	Context("Handle with local GPU mode", func() {
		It("should successfully handle a pod with local GPU mode", func() {
			// Create a mock GPU
			mockGPU := &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{
					Name: "mock-gpu-1",
					Labels: map[string]string{
						constants.GpuPoolKey: "mock",
					},
				},
			}
			Expect(k8sclient.Create(ctx, mockGPU)).To(Succeed())
			mockGPU.Status = tfv1.GPUStatus{
				Phase: tfv1.TensorFusionGPUPhaseRunning,
				UUID:  "mock-gpu",
				NodeSelector: map[string]string{
					"kubernetes.io/hostname": "mock-node",
				},
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("200"),
					Vram:   resource.MustParse("200Gi"),
				},
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("200"),
					Vram:   resource.MustParse("200Gi"),
				},
			}
			Expect(k8sclient.Status().Update(ctx, mockGPU)).To(Succeed())
			// Set up a client profile with IsLocalGPU set to true
			clientProfile := &tfv1.ClientProfile{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "local-gpu-profile",
					Namespace: "default",
				},
				Spec: tfv1.ClientProfileSpec{
					PoolName:   "mock",
					IsLocalGPU: true,
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("1Gi"),
						},
						Limits: tfv1.Resource{
							Tflops: resource.MustParse("100"),
							Vram:   resource.MustParse("16Gi"),
						},
					},
				},
			}

			Expect(k8sclient.Create(ctx, clientProfile)).To(Succeed())
			// Create a pod with the local GPU profile
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-local-gpu",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.GpuPoolKey:                "mock",
						constants.ClientProfileAnnotation:   "local-gpu-profile",
						constants.InjectContainerAnnotation: "main",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "main",
							Image: "test-image",
						},
					},
				},
			}
			podBytes, err := json.Marshal(pod)
			Expect(err).NotTo(HaveOccurred())

			req := admission.Request{
				AdmissionRequest: admissionv1.AdmissionRequest{
					Object: runtime.RawExtension{
						Raw: podBytes,
					},
					Operation: admissionv1.Create,
				},
			}

			// Process the request
			resp := mutator.Handle(ctx, req)

			// Verify the response
			Expect(resp.Allowed).To(BeTrue())
			Expect(resp.Patches).NotTo(BeEmpty())

			// Check the GPU resources were updated
			updatedGPU := &tfv1.GPU{}
			Expect(k8sclient.Get(ctx, client.ObjectKeyFromObject(mockGPU), updatedGPU)).NotTo(HaveOccurred())
			// The available resources should be reduced by the requested amount
			Expect(updatedGPU.Status.Available.Tflops.Cmp(resource.MustParse("190"))).To(Equal(0))
			Expect(updatedGPU.Status.Available.Vram.String()).To(Equal("199Gi"))
		})
	})

	Context("ParseTFResources", func() {
		It("should correctly parse TF requirements from pod annotations", func() {
			// Set up a client profile for testing
			clientProfile := &tfv1.ClientProfile{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-profile-parse-tf-resources",
					Namespace: "default",
				},
				Spec: tfv1.ClientProfileSpec{
					PoolName: "mock",
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("1Gi"),
						},
						Limits: tfv1.Resource{
							Tflops: resource.MustParse("100"),
							Vram:   resource.MustParse("16Gi"),
						},
					},
				},
			}

			Expect(k8sclient.Create(ctx, clientProfile)).To(Succeed())

			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Annotations: map[string]string{
						constants.GpuPoolKey:              "mock",
						constants.ClientProfileAnnotation: "test-profile-parse-tf-resources",
						// override tflops request
						constants.TFLOPSRequestAnnotation:   "20",
						constants.InjectContainerAnnotation: "test-container",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Env: []corev1.EnvVar{
								{
									Name:  constants.ConnectionNameEnv,
									Value: "conn1",
								},
								{
									Name:  constants.ConnectionNamespaceEnv,
									Value: "ns",
								},
							},
						},
					},
				},
			}
			profile, containerNames, err := ParseTFResources(ctx, k8sclient, pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(containerNames).To(HaveLen(1))
			Expect(containerNames[0]).To(Equal("test-container"))
			Expect(profile.PoolName).To(Equal("mock"))
			Expect(profile.Resources.Requests.Tflops.String()).To(Equal("20"))
			Expect(profile.Resources.Requests.Vram.String()).To(Equal("1Gi"))
			Expect(profile.Resources.Limits.Tflops.String()).To(Equal("100"))
			Expect(profile.Resources.Limits.Vram.String()).To(Equal("16Gi"))
		})
	})

	Context("patchTFClient", func() {
		It("should apply the patch to the pod", func() {
			pod := &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
						},
					},
				},
			}

			patch, err := mutator.patchTFClient(pod, config.MockGPUPoolSpec.ComponentConfig.Client, []string{"test-container"}, nil)
			Expect(err).NotTo(HaveOccurred())
			Expect(patch).NotTo(BeEmpty())
			Expect(patch).To(HaveLen(2))
			Expect(patch[1].Path).To(Equal("/spec/initContainers"))
			Expect(patch[1].Operation).To(Equal("add"))
		})
	})
})
