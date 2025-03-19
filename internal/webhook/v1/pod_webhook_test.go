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

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
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
			Client:  k8sclient,
			decoder: decoder,
		}
	})

	Context("Handle", func() {
		It("should successfully mutate a pod with TF resources", func() {
			// Set up a workload profile for testing
			workloadProfile := &tfv1.WorkloadProfile{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-profile-handle",
					Namespace: "default",
				},
				Spec: tfv1.WorkloadProfileSpec{
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
			Expect(k8sclient.Create(ctx, workloadProfile)).To(Succeed())

			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.GpuPoolKey:                "mock",
						constants.WorkloadProfileAnnotation: "test-profile-handle",
						constants.InjectContainerAnnotation: "main",
						constants.WorkloadKey:               "test-workload",
						constants.GenWorkload:               "true",
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
			// Should fail because no annotations are found
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
			// Set up a workload profile with IsLocalGPU set to true
			workloadProfile := &tfv1.WorkloadProfile{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "local-gpu-profile",
					Namespace: "default",
				},
				Spec: tfv1.WorkloadProfileSpec{
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
			Expect(k8sclient.Create(ctx, workloadProfile)).To(Succeed())

			// Create a TensorFusionWorkload first
			workload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "local-gpu-workload",
					Namespace: "default",
				},
				Spec: tfv1.TensorFusionWorkloadSpec{
					PoolName:   "mock",
					IsLocalGPU: true,
				},
			}
			Expect(k8sclient.Create(ctx, workload)).To(Succeed())

			// Update workload status
			workload.Status.WorkerStatuses = []tfv1.WorkerStatus{
				{
					WorkerName:  "mock-worker",
					WorkerPhase: tfv1.WorkerRunning,
					NodeSelector: map[string]string{
						"kubernetes.io/hostname": "mock-node",
					},
				},
			}
			Expect(k8sclient.Status().Update(ctx, workload)).To(Succeed())

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
						constants.WorkloadProfileAnnotation: "local-gpu-profile",
						constants.InjectContainerAnnotation: "main",
						constants.WorkloadKey:               "local-gpu-workload",
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
		})
	})

	Context("ParseTensorFusionInfo", func() {
		It("should correctly parse TF requirements from pod annotations", func() {
			// Set up a workload profile for testing
			workloadProfile := &tfv1.WorkloadProfile{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-profile-parse-tf-resources",
					Namespace: "default",
				},
				Spec: tfv1.WorkloadProfileSpec{
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

			Expect(k8sclient.Create(ctx, workloadProfile)).To(Succeed())

			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Annotations: map[string]string{
						constants.GpuPoolKey:                "mock",
						constants.WorkloadProfileAnnotation: "test-profile-parse-tf-resources",
						constants.WorkloadKey:               "test-workload",
						// override tflops request
						constants.TFLOPSRequestAnnotation:   "20",
						constants.InjectContainerAnnotation: "test-container",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name: "test-container",
						},
					},
				},
			}
			tfInfo, err := ParseTensorFusionInfo(ctx, k8sclient, pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(tfInfo.ContainerNames).To(HaveLen(1))
			Expect(tfInfo.ContainerNames[0]).To(Equal("test-container"))
			Expect(tfInfo.Profile.PoolName).To(Equal("mock"))
			Expect(tfInfo.Profile.Resources.Requests.Tflops.String()).To(Equal("20"))
			Expect(tfInfo.Profile.Resources.Requests.Vram.String()).To(Equal("1Gi"))
			Expect(tfInfo.Profile.Resources.Limits.Tflops.String()).To(Equal("100"))
			Expect(tfInfo.Profile.Resources.Limits.Vram.String()).To(Equal("16Gi"))
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
			// There should be at least 2 patches (initContainers and the container env patches)
			Expect(len(patch)).To(BeNumerically(">=", 2))
		})
	})
})
