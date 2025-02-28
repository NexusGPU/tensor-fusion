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
	"fmt"
	"net/http"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/config"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
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
		mutator *TensorFusionPodMutator
		ctx     context.Context
		scheme  *runtime.Scheme
		decoder admission.Decoder
		client  client.Client
	)

	BeforeEach(func() {
		ctx = context.Background()
		scheme = runtime.NewScheme()
		Expect(corev1.AddToScheme(scheme)).To(Succeed())
		Expect(tfv1.AddToScheme(scheme)).To(Succeed())

		decoder = admission.NewDecoder(scheme)
		client = k8sClient

		mutator = &TensorFusionPodMutator{
			Client: client,
		}
		Expect(mutator.InjectDecoder(decoder)).To(Succeed())
	})

	Context("Handle", func() {
		It("should successfully mutate a pod with TF resources", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.GpuPoolKey: "mock",
						fmt.Sprintf(constants.TFLOPSRequestAnnotationFormat, "main"): "10",
						fmt.Sprintf(constants.VRAMRequestAnnotationFormat, "main"):   "1Gi",
						fmt.Sprintf(constants.TFLOPSLimitAnnotationFormat, "main"):   "100",
						fmt.Sprintf(constants.VRAMLimitAnnotationFormat, "main"):     "16Gi",
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
			Expect(resp.Allowed).To(BeTrue())
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

	Context("parseTFReq", func() {
		It("should correctly parse TF requirements from pod annotations", func() {

			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.GpuPoolKey: "mock",
						fmt.Sprintf(constants.TFLOPSRequestAnnotationFormat, "test-container"): "10",
						fmt.Sprintf(constants.VRAMRequestAnnotationFormat, "test-container"):   "1Gi",
						fmt.Sprintf(constants.TFLOPSLimitAnnotationFormat, "test-container"):   "100",
						fmt.Sprintf(constants.VRAMLimitAnnotationFormat, "test-container"):     "16Gi",
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
			_, resources := ParseTFResources(pod)
			Expect(resources).To(HaveLen(1))
			Expect(resources[0].ContainerName).To(Equal("test-container"))
			Expect(resources[0].TflopsRequest.String()).To(Equal("10"))
			Expect(resources[0].VramRequest.String()).To(Equal("1Gi"))
			Expect(resources[0].TflopsLimit.String()).To(Equal("100"))
			Expect(resources[0].VramLimit.String()).To(Equal("16Gi"))
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

			patch, err := mutator.patchTFClient(pod, config.MockGPUPoolSpec.ComponentConfig.Client, []TFResource{{ContainerName: "test-container", TflopsRequest: resource.MustParse("100"), VramRequest: resource.MustParse("16Gi")}})
			Expect(err).NotTo(HaveOccurred())
			Expect(patch).NotTo(BeEmpty())
			Expect(patch).To(HaveLen(2))
			Expect(patch[1].Path).To(Equal("/spec/initContainers"))
			Expect(patch[1].Operation).To(Equal("add"))
		})
	})
})
