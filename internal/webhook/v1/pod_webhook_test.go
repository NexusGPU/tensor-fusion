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

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/samber/lo"
	"gomodules.xyz/jsonpatch/v2"
	admissionv1 "k8s.io/api/admission/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

func isAddOperation(patch jsonpatch.JsonPatchOperation) bool {
	return patch.Operation == "add"
}

// nolint:goconst
var _ = Describe("TensorFusionPodMutator", func() {
	var (
		mutator *TensorFusionPodMutator
		ctx     context.Context
		scheme  *runtime.Scheme
		decoder admission.Decoder
	)

	BeforeEach(func() {
		ctx = context.Background()
		scheme = runtime.NewScheme()
		Expect(corev1.AddToScheme(scheme)).To(Succeed())
		Expect(tfv1.AddToScheme(scheme)).To(Succeed())

		decoder = admission.NewDecoder(scheme)

		mutator = &TensorFusionPodMutator{
			Client:  k8sClient,
			decoder: decoder,
		}
	})

	Context("Handle", func() {
		It("should handle pod with empty namespace", func() {
			// Create a pod with empty namespace
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod-empty-ns",
					// empty namespace
					Namespace: "",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.GpuPoolKey:                "mock",
						constants.InjectContainerAnnotation: "main",
						constants.TFLOPSLimitAnnotation:     "100",
						constants.VRAMLimitAnnotation:       "16Gi",

						constants.TFLOPSRequestAnnotation: "10",
						constants.VRAMRequestAnnotation:   "1Gi",
					},
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "ReplicaSet",
							Name:       "test-workload",
							UID:        "owner-uid",
							Controller: ptr.To(true),
						},
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "main",
						Image: "test-image",
					}},
				},
			}
			podBytes, err := json.Marshal(pod)
			Expect(err).NotTo(HaveOccurred())

			// Construct an admission request with the pod
			req := admission.Request{
				AdmissionRequest: admissionv1.AdmissionRequest{
					Object: runtime.RawExtension{
						Raw: podBytes,
					},
					Operation: admissionv1.Create,
					Namespace: "default",
				},
			}

			// Call mutator.Handle to process the admission request
			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeTrue())
			Expect(resp.Patches).NotTo(BeEmpty())

			_, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1tflops-request"
			})
			Expect(found).To(BeFalse())
			_, found = lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1vram-request"
			})
			Expect(found).To(BeFalse())
			op, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1isolation"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("soft"))
			op, found = lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/spec/schedulerName"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("tensor-fusion-scheduler"))
		})

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
			Expect(k8sClient.Create(ctx, workloadProfile)).To(Succeed())
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					OwnerReferences: []metav1.OwnerReference{{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "test-workload",
						UID:        "owner-uid",
						Controller: ptr.To(true),
					}},
					Annotations: map[string]string{
						constants.GpuPoolKey:                "mock",
						constants.WorkloadProfileAnnotation: "test-profile-handle",
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

			op, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1tflops-request"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("10"))
			op, found = lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1vram-request"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("1Gi"))
			op, found = lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1tflops-limit"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("100"))
			op, found = lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1vram-limit"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("16Gi"))
			op, found = lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1gpu-count"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("1"))

			// Check workload created
			Eventually(func(g Gomega) error {
				workload := &tfv1.TensorFusionWorkload{}
				err = k8sClient.Get(ctx, client.ObjectKey{Name: "test-workload", Namespace: "default"}, workload)
				g.Expect(workload.Spec.Replicas).To(BeNil())
				g.Expect(workload.OwnerReferences).To(HaveLen(1))
				g.Expect(workload.OwnerReferences[0].Name).To(Equal("test-workload"))
				return err
			}).Should(Succeed())
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
			Expect(resp.Allowed).To(BeTrue())
			Expect(resp.Patches).To(BeEmpty())
		})

		It("should handle dedicated GPU", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-local-gpu",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.DedicatedGPUAnnotation: constants.TrueStringValue,
						constants.GPUModelAnnotation:     "A100",
						constants.GpuPoolKey:             "mock",
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
					Namespace: "default",
				},
			}

			gpuallocator.GPUCapacityMap["A100"] = tfv1.Resource{
				Tflops: resource.MustParse("312"),
				Vram:   resource.MustParse("40Gi"),
			}
			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeTrue())

			op, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1tflops-request"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("312"))
		})

		It("should handle gpu resource limits and fill requests from annotation limits", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-local-gpu",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.TFLOPSLimitAnnotation: "100",
						constants.VRAMLimitAnnotation:   "16Gi",
						constants.GpuPoolKey:            "mock",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "main",
							Image: "test-image",
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									corev1.ResourceName(constants.NvidiaGPUKey): resource.MustParse("3"),
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
					Namespace: "default",
				},
			}
			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeTrue())

			op, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1tflops-request"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("100"))
			op, found = lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1vram-request"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("16Gi"))
			op, found = lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1gpu-count"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("3"))
		})

		It("should handle gpu resource limits with only one label", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-local-gpu",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.GpuPoolKey: "mock",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "main",
							Image: "test-image",
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									corev1.ResourceName(constants.NvidiaGPUKey): resource.MustParse("3"),
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
					Namespace: "default",
				},
			}
			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeTrue())

			op, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1compute-percent-request"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("100"))
			op, found = lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1gpu-count"
			})
			Expect(found).To(BeTrue())
			Expect(op.Value).To(Equal("3"))
		})

		It("should handle ignore gpu resource limits when annotation is present", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod-local-gpu",
					Namespace: "default",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.TFLOPSLimitAnnotation: "100",
						constants.VRAMLimitAnnotation:   "16Gi",
						constants.GpuCountAnnotation:    "2",
						constants.GpuPoolKey:            "mock",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "main",
							Image: "test-image",
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									corev1.ResourceName(constants.NvidiaGPUKey): resource.MustParse("3"),
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
					Namespace: "default",
				},
			}
			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeTrue())

			_, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return isAddOperation(patch) &&
					patch.Path == "/metadata/annotations/tensor-fusion.ai~1gpu-count"
			})
			Expect(found).To(BeFalse())
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
			Expect(k8sClient.Create(ctx, workloadProfile)).To(Succeed())

			// Create a TensorFusionWorkload first
			workload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "local-gpu-workload",
					Namespace: "default",
				},
				Spec: tfv1.WorkloadProfileSpec{
					PoolName:   "mock",
					IsLocalGPU: true,
				},
			}
			Expect(k8sClient.Create(ctx, workload)).To(Succeed())

			Expect(k8sClient.Status().Update(ctx, workload)).To(Succeed())

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

			scheduleMutation, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return patch.Path == "/spec/schedulerName"
			})
			Expect(found).To(BeTrue())
			Expect(scheduleMutation.Value).To(Equal(constants.SchedulerName))

			workloadAnnotationMutation, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return patch.Path == "/metadata/annotations/tensor-fusion.ai~1tflops-limit"
			})
			Expect(found).To(BeTrue())
			Expect(workloadAnnotationMutation.Value).To(Equal("100"))
		})
	})

	Context("Handle with EnabledReplicas", func() {
		It("should only patch enabledReplicas pods", func() {
			// Create a ReplicaSet as the owner for the pod
			replicaSet := &appsv1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-rs",
					Namespace: "default",
				},
				Spec: appsv1.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"app": "test-app",
						},
					},
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"app": "test-app",
							},
						},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "test-container",
									Image: "test-image",
								},
							},
						},
					},
				},
			}

			Expect(k8sClient.Create(ctx, replicaSet)).To(Succeed())

			// Get the ReplicaSet to obtain its UID
			createdReplicaSet := &appsv1.ReplicaSet{}
			Expect(k8sClient.Get(ctx, client.ObjectKey{Namespace: "default", Name: "test-rs"}, createdReplicaSet)).To(Succeed())
			replicaSetUID := createdReplicaSet.GetUID()

			// Create a workload profile
			workloadProfile := &tfv1.WorkloadProfile{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-profile-enabled-replicas",
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
			Expect(k8sClient.Create(ctx, workloadProfile)).To(Succeed())

			// Create a pod with TF resources and owner reference
			trueVal := true
			enabledReplicas := int32(1)

			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:    "default",
					GenerateName: "test-pod-enabled-replicas-",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
						"pod-template-hash":                   "test-hash",
					},
					Annotations: map[string]string{
						constants.GpuPoolKey:                            "mock",
						constants.WorkloadProfileAnnotation:             "test-profile-enabled-replicas",
						constants.InjectContainerAnnotation:             "main",
						constants.TensorFusionEnabledReplicasAnnotation: fmt.Sprintf("%d", enabledReplicas), // Using the correct constant
					},
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "ReplicaSet",
							Name:       "test-rs",
							UID:        replicaSetUID,
							Controller: &trueVal,
						},
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

			resp := mutator.Handle(ctx, req)
			// First call: Pod mutation should occur since enabledReplicas is 1,
			// so the response should be allowed and contain patches
			Expect(resp.Allowed).To(BeTrue())
			Expect(resp.Patches).NotTo(BeEmpty())

			counter := &TensorFusionPodCounter{Client: k8sClient}
			count, _, err := counter.Get(ctx, pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(count).To(Equal(int32(1)))

			resp = mutator.Handle(ctx, req)
			// Second call: Pod should be ignored since it's been processed already,
			// so the response should be allowed but patches should be empty
			Expect(resp.Allowed).To(BeTrue())
			Expect(resp.Patches).To(BeEmpty())

			// Clean up
			Expect(k8sClient.Delete(ctx, replicaSet)).To(Succeed())
			Expect(k8sClient.Delete(ctx, workloadProfile)).To(Succeed())
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

			Expect(k8sClient.Create(ctx, workloadProfile)).To(Succeed())

			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Annotations: map[string]string{
						constants.GpuPoolKey:                "mock",
						constants.WorkloadProfileAnnotation: "test-profile-parse-tf-resources",
						// override tflops request
						constants.TFLOPSRequestAnnotation:               "20",
						constants.InjectContainerAnnotation:             "test-container",
						constants.TensorFusionEnabledReplicasAnnotation: "3",
						constants.QoSLevelAnnotation:                    "high",
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
			tfInfo, err := ParseTensorFusionInfo(ctx, k8sClient, pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(tfInfo.ContainerNames).To(HaveLen(1))
			Expect(tfInfo.ContainerNames[0]).To(Equal("test-container"))
			Expect(tfInfo.Profile.PoolName).To(Equal("mock"))
			Expect(tfInfo.Profile.Resources.Requests.Tflops.String()).To(Equal("20"))
			Expect(tfInfo.Profile.Resources.Requests.Vram.String()).To(Equal("1Gi"))
			Expect(tfInfo.Profile.Resources.Limits.Tflops.String()).To(Equal("100"))
			Expect(tfInfo.Profile.Resources.Limits.Vram.String()).To(Equal("16Gi"))
			Expect(tfInfo.Profile.Qos).To(Equal(tfv1.QoSHigh))
			Expect(*tfInfo.EnabledReplicas).To(Equal(int32(3)))
		})

		It("should treat generateName as workload name if the pod has no controllerRef", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:    "default",
					GenerateName: "test-name",
					Annotations: map[string]string{
						constants.GpuPoolKey: "mock",
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
			tfInfo, _ := ParseTensorFusionInfo(ctx, k8sClient, pod)
			Expect(tfInfo.WorkloadName).To(HavePrefix("test-name"))
		})

		It("should treat controller name as workload name if the pod has controllerRef", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:    "default",
					GenerateName: "test-name",
					Annotations: map[string]string{
						constants.GpuPoolKey: "mock",
					},
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "ReplicaSet",
							Name:       "test-rs",
							UID:        "rs-uid",
							Controller: ptr.To(true),
						},
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
			tfInfo, _ := ParseTensorFusionInfo(ctx, k8sClient, pod)
			Expect(tfInfo.WorkloadName).To(Equal("test-rs"))
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

			pool := &tfv1.GPUPool{
				Spec: *config.MockGPUPoolSpec,
			}

			currentBytes, err := json.Marshal(pod)
			Expect(err).NotTo(HaveOccurred())
			patch, err := mutator.patchTFClient(context.Background(), pod, pool, false, currentBytes, []int{0}, false)
			Expect(err).NotTo(HaveOccurred())
			Expect(patch).NotTo(BeEmpty())
			// There should be at least 2 patches (initContainers and the container env patches)
			Expect(len(patch)).To(BeNumerically(">=", 2))
		})
	})

	Context("when handling workload", func() {
		It("should set the workload owner same with Pod's controllerRef", func() {
			expectedRef := metav1.OwnerReference{
				APIVersion: "apps/v1",
				Kind:       "ReplicaSet",
				Name:       "my-rs",
				UID:        "rs-uid",
				Controller: ptr.To(true),
			}
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					GenerateName: "test-name",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.GpuPoolKey:            "mock",
						constants.TFLOPSLimitAnnotation: "100",
						constants.VRAMLimitAnnotation:   "16Gi",
					},
					OwnerReferences: []metav1.OwnerReference{expectedRef},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "main",
						Image: "test-image",
					}},
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
					Namespace: "default",
				},
			}

			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeTrue())
			Expect(pod.Annotations[constants.SetPendingOwnedWorkloadAnnotation]).To(BeEmpty())

			Eventually(func(g Gomega) {
				workload := &tfv1.TensorFusionWorkload{}
				g.Expect(k8sClient.Get(ctx,
					client.ObjectKey{
						Name:      expectedRef.Name,
						Namespace: "default",
					}, workload)).To(Succeed())
				gotRef := metav1.GetControllerOfNoCopy(workload)
				g.Expect(*gotRef).To(Equal(expectedRef))
			}).Should(Succeed())
		})

		It("should set the workload owner to controlling deployment if the pod controlled by a deployment", func() {
			expectedRef := metav1.OwnerReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       "test-deployment",
				UID:        "deployment-uid",
				Controller: ptr.To(true),
			}
			rs := &appsv1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "test-rs",
					Namespace:       "default",
					UID:             "rs-uid",
					OwnerReferences: []metav1.OwnerReference{expectedRef},
				},
				Spec: appsv1.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"app": "test-app",
						},
					},
					Template: corev1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"app": "test-app",
							},
						},
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  "test-container",
									Image: "test-image",
								},
							},
						},
					},
				},
			}

			Expect(k8sClient.Create(ctx, rs)).To(Succeed())
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					GenerateName: "test-name",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.GpuPoolKey:            "mock",
						constants.TFLOPSLimitAnnotation: "100",
						constants.VRAMLimitAnnotation:   "16Gi",
					},
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "ReplicaSet",
							Name:       "test-rs",
							UID:        "rs-uid",
							Controller: ptr.To(true),
						},
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "main",
						Image: "test-image",
					}},
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
					Namespace: "default",
				},
			}

			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeTrue())
			Expect(pod.Annotations[constants.SetPendingOwnedWorkloadAnnotation]).To(BeEmpty())

			Eventually(func(g Gomega) {
				workload := &tfv1.TensorFusionWorkload{}
				g.Expect(k8sClient.Get(ctx,
					client.ObjectKey{
						Name:      expectedRef.Name,
						Namespace: "default",
					}, workload)).To(Succeed())
				gotRef := metav1.GetControllerOfNoCopy(workload)
				g.Expect(*gotRef).To(Equal(expectedRef))
			}).Should(Succeed())

			Expect(k8sClient.Delete(ctx, rs)).Should(Succeed())
		})

		It("should add SetPendingOwnedWorkload annotation to pod when workload has no controllerRef", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					GenerateName: "test-name",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.GpuPoolKey:            "mock",
						constants.TFLOPSLimitAnnotation: "100",
						constants.VRAMLimitAnnotation:   "16Gi",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "main",
						Image: "test-image",
					}},
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
					Namespace: "default",
				},
			}

			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeTrue())
			annotation, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return patch.Path == "/metadata/annotations/tensor-fusion.ai~1pending-owned-workload"
			})
			Expect(found).To(BeTrue())
			Expect(annotation.Value).To(HavePrefix("test-name"))
		})

		It("should accumulate GPU count from multiple containers with nvidia.com/gpu limits", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "multi-container-gpu-pod",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.GpuPoolKey:            "mock",
						constants.VRAMLimitAnnotation:   "16Gi",
						constants.TFLOPSLimitAnnotation: "100",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "container1",
							Image: "test-image1",
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									constants.NvidiaGPUKey: *resource.NewQuantity(2, resource.DecimalSI),
								},
							},
						},
						{
							Name:  "container2",
							Image: "test-image2",
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									constants.NvidiaGPUKey: *resource.NewQuantity(1, resource.DecimalSI),
								},
							},
						},
						{
							Name:  "container3",
							Image: "test-image3",
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									constants.NvidiaGPUKey: *resource.NewQuantity(1, resource.DecimalSI),
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
					Namespace: "default",
				},
			}

			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeTrue())

			// Verify inject-container annotation includes all containers
			injectContainerPatch, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return patch.Path == "/metadata/annotations/tensor-fusion.ai~1inject-container"
			})
			Expect(found).To(BeTrue())
			injectContainers := injectContainerPatch.Value.(string)
			Expect(injectContainers).To(ContainSubstring("container1"))
			Expect(injectContainers).To(ContainSubstring("container2"))
			Expect(injectContainers).To(ContainSubstring("container3"))

			// Verify gpu-count annotation is set to total (2+1+1=4)
			gpuCountPatch, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return patch.Path == "/metadata/annotations/tensor-fusion.ai~1gpu-count"
			})
			Expect(found).To(BeTrue())
			Expect(gpuCountPatch.Value).To(Equal("4"))

			// Verify container-gpu-count annotation is set
			containerGPUCountPatch, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return patch.Path == "/metadata/annotations/tensor-fusion.ai~1container-gpu-count"
			})
			Expect(found).To(BeTrue())

			// Parse and verify the container GPU count JSON
			var containerGPUCounts map[string]int
			err = json.Unmarshal([]byte(containerGPUCountPatch.Value.(string)), &containerGPUCounts)
			Expect(err).NotTo(HaveOccurred())
			Expect(containerGPUCounts["container1"]).To(Equal(2))
			Expect(containerGPUCounts["container2"]).To(Equal(1))
			Expect(containerGPUCounts["container3"]).To(Equal(1))
		})

		It("should not set container-gpu-count annotation for single container with GPU", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "single-container-gpu-pod",
					Labels: map[string]string{
						constants.TensorFusionEnabledLabelKey: "true",
					},
					Annotations: map[string]string{
						constants.GpuPoolKey:            "mock",
						constants.VRAMLimitAnnotation:   "16Gi",
						constants.TFLOPSLimitAnnotation: "100",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "main",
							Image: "test-image",
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									constants.NvidiaGPUKey: *resource.NewQuantity(2, resource.DecimalSI),
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
					Namespace: "default",
				},
			}

			resp := mutator.Handle(ctx, req)
			Expect(resp.Allowed).To(BeTrue())

			// Verify container-gpu-count annotation IS set even for single container
			// (the logic sets it whenever containerGPUCounts is not empty)
			containerGPUCountPatch, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return patch.Path == "/metadata/annotations/tensor-fusion.ai~1container-gpu-count"
			})
			Expect(found).To(BeTrue())

			// Verify the container GPU count content
			var containerGPUCounts map[string]int
			err = json.Unmarshal([]byte(containerGPUCountPatch.Value.(string)), &containerGPUCounts)
			Expect(err).NotTo(HaveOccurred())
			Expect(containerGPUCounts["main"]).To(Equal(2))

			// Verify gpu-count is set to 2
			gpuCountPatch, found := lo.Find(resp.Patches, func(patch jsonpatch.JsonPatchOperation) bool {
				return patch.Path == "/metadata/annotations/tensor-fusion.ai~1gpu-count"
			})
			Expect(found).To(BeTrue())
			Expect(gpuCountPatch.Value).To(Equal("2"))
		})
	})

	Context("ParseContainerGPUCounts", func() {
		It("should parse valid container GPU count annotation", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.ContainerGPUCountAnnotation: `{"container1": 2, "container2": 1}`,
					},
				},
			}

			counts, err := ParseContainerGPUCounts(pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(counts).To(HaveLen(2))
			Expect(counts["container1"]).To(Equal(2))
			Expect(counts["container2"]).To(Equal(1))
		})

		It("should return nil for pod without annotation", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{},
				},
			}

			counts, err := ParseContainerGPUCounts(pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(counts).To(BeNil())
		})

		It("should return nil for pod with nil annotations", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: nil,
				},
			}

			counts, err := ParseContainerGPUCounts(pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(counts).To(BeNil())
		})

		It("should return error for invalid JSON", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.ContainerGPUCountAnnotation: `invalid json`,
					},
				},
			}

			counts, err := ParseContainerGPUCounts(pod)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("failed to parse"))
			Expect(counts).To(BeNil())
		})

		It("should parse empty JSON object", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						constants.ContainerGPUCountAnnotation: `{}`,
					},
				},
			}

			counts, err := ParseContainerGPUCounts(pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(counts).To(BeEmpty())
		})
	})
})
