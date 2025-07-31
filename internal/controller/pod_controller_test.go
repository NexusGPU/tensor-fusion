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

package controller

import (
	"context"
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

var _ = Describe("Pod Controller", func() {
	var (
		tfEnv *TensorFusionEnv
		ctx   context.Context
	)

	BeforeEach(func() {
		ctx = context.Background()
		tfEnv = NewTensorFusionEnvBuilder().
			AddPoolWithNodeCount(2).
			SetGpuCountPerNode(1).
			Build()
	})

	AfterEach(func() {
		if tfEnv != nil {
			tfEnv.Cleanup()
		}
	})

	Context("When reconciling a Worker Pod", func() {
		var workerPod *corev1.Pod

		BeforeEach(func() {
			workerPod = &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-worker-pod",
					Namespace: "default",
					Labels: map[string]string{
						constants.LabelComponent: constants.ComponentWorker,
						constants.WorkloadKey:    "test-workload",
					},
					Annotations: map[string]string{
						constants.TFLOPSRequestAnnotation: "5",
						constants.VRAMRequestAnnotation:   "1Gi",
					},
					Finalizers: []string{constants.Finalizer},
				},
				Spec: corev1.PodSpec{
					NodeName: "skip-schedule",
					Containers: []corev1.Container{
						{
							Name:  "worker",
							Image: "test-image",
						},
					},
				},
			}
		})

		AfterEach(func() {
			if workerPod != nil {
				_ = k8sClient.Delete(ctx, workerPod)
			}
		})

		It("should successfully reconcile a worker pod creation", func() {
			By("creating a worker pod")
			Expect(k8sClient.Create(ctx, workerPod)).To(Succeed())

			By("verifying the pod exists and retains finalizer")
			Eventually(func() bool {
				updatedPod := &corev1.Pod{}
				err := k8sClient.Get(ctx, client.ObjectKeyFromObject(workerPod), updatedPod)
				if err != nil {
					return false
				}
				return controllerutil.ContainsFinalizer(updatedPod, constants.Finalizer)
			}).Should(BeTrue())
		})

		It("should handle worker pod deletion and cleanup resources", func() {
			By("creating a worker pod")
			Expect(k8sClient.Create(ctx, workerPod)).To(Succeed())

			By("waiting for pod to be processed")
			Eventually(func() bool {
				updatedPod := &corev1.Pod{}
				err := k8sClient.Get(ctx, client.ObjectKeyFromObject(workerPod), updatedPod)
				return err == nil
			}).Should(BeTrue())

			By("deleting the worker pod")
			Expect(k8sClient.Delete(ctx, workerPod)).To(Succeed())

			By("verifying the finalizer is removed and pod is deleted")
			Eventually(func() bool {
				updatedPod := &corev1.Pod{}
				err := k8sClient.Get(ctx, client.ObjectKeyFromObject(workerPod), updatedPod)
				if errors.IsNotFound(err) {
					return true
				}
				return !controllerutil.ContainsFinalizer(updatedPod, constants.Finalizer)
			}).Should(BeTrue())
		})
	})

	Context("When reconciling a Client Pod", func() {
		var clientPod *corev1.Pod
		var workload *tfv1.TensorFusionWorkload

		BeforeEach(func() {
			// Create a test workload first
			workload = &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-workload",
					Namespace: "default",
				},
				Spec: tfv1.WorkloadProfileSpec{
					PoolName: "cluster-0-pool-0",
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: resource.MustParse("5"),
							Vram:   resource.MustParse("1Gi"),
						},
						Limits: tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("16Gi"),
						},
					},
				},
			}
			Expect(k8sClient.Create(ctx, workload)).To(Succeed())

			clientPod = &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-client-pod",
					Namespace: "default",
					Labels: map[string]string{
						constants.LabelComponent: constants.ComponentClient,
					},
					Annotations: map[string]string{
						constants.SelectedWorkloadAnnotation:        workload.Name,
						constants.SetPendingOwnedWorkloadAnnotation: workload.Name,
						constants.TFLOPSRequestAnnotation:           "5",
						constants.VRAMRequestAnnotation:             "1Gi",
					},
				},
				Spec: corev1.PodSpec{
					NodeName: "skip-schedule",
					Containers: []corev1.Container{
						{
							Name:  "client",
							Image: "test-image",
							Env: []corev1.EnvVar{
								{
									Name:  constants.ConnectionNameEnv,
									Value: "test-connection-pod-controller",
								},
								{
									Name:  constants.ConnectionNamespaceEnv,
									Value: "default",
								},
							},
						},
					},
				},
			}
		})

		AfterEach(func() {
			if workload != nil {
				_ = k8sClient.Delete(ctx, workload)
			}
			if clientPod != nil {
				_ = k8sClient.Delete(ctx, clientPod)
			}
		})

		It("should successfully create TensorFusion connection for client pod", func() {
			By("creating a client pod")
			Expect(k8sClient.Create(ctx, clientPod)).To(Succeed())

			By("verifying TensorFusion connection is created")
			connection := &tfv1.TensorFusionConnection{}
			connectionKey := types.NamespacedName{
				Name:      "test-connection-pod-controller",
				Namespace: "default",
			}
			Eventually(func() error {
				return k8sClient.Get(ctx, connectionKey, connection)
			}).Should(Succeed())

			By("verifying connection has correct spec and owner reference")
			Eventually(func(g Gomega) error {
				g.Expect(connection.Spec.WorkloadName).To(Equal(workload.Name))
				g.Expect(connection.Spec.ClientPod).To(Equal(clientPod.Name))
				g.Expect(connection.Labels[constants.WorkloadKey]).To(Equal(workload.Name))
				g.Expect(connection.OwnerReferences).To(HaveLen(1))
				g.Expect(connection.OwnerReferences[0].Name).To(Equal(clientPod.Name))
				g.Expect(connection.OwnerReferences[0].Kind).To(Equal("Pod"))

				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(workload), workload)).To(Succeed())
				if len(workload.OwnerReferences) > 0 {
					g.Expect(workload.OwnerReferences[0].UID).To(Equal(clientPod.UID))
				} else {
					return fmt.Errorf("workload owner references is empty, wait next check")
				}
				return nil
			}).Should(Succeed())

		})

		It("should remove tensor-fusion finalizer from client pod if present", func() {
			By("creating a client pod with tensor-fusion finalizer")
			clientPod.Finalizers = []string{constants.Finalizer}
			Expect(k8sClient.Create(ctx, clientPod)).To(Succeed())

			By("verifying finalizer is eventually removed")
			Eventually(func() bool {
				updatedPod := &corev1.Pod{}
				err := k8sClient.Get(ctx, client.ObjectKeyFromObject(clientPod), updatedPod)
				if err != nil {
					return false
				}
				return !controllerutil.ContainsFinalizer(updatedPod, constants.Finalizer)
			}).Should(BeTrue())
		})

		It("should skip connection creation if pod is not a TensorFusion client", func() {
			By("creating a client pod without selected workload annotation")
			delete(clientPod.Annotations, constants.SelectedWorkloadAnnotation)
			Expect(k8sClient.Create(ctx, clientPod)).To(Succeed())

			By("verifying no TensorFusion connection is created")
			connection := &tfv1.TensorFusionConnection{}
			connectionKey := types.NamespacedName{
				Name:      "test-connection-pod-controller",
				Namespace: "default",
			}
			Consistently(func() bool {
				err := k8sClient.Get(ctx, connectionKey, connection)
				return errors.IsNotFound(err)
			}, 2*time.Second).Should(BeTrue())
		})

		It("should not create duplicate connections if connection already exists", func() {
			By("creating an existing TensorFusion connection")
			existingConnection := &tfv1.TensorFusionConnection{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-connection-pod-controller",
					Namespace: "default",
				},
				Spec: tfv1.TensorFusionConnectionSpec{
					WorkloadName: workload.Name,
					ClientPod:    "existing-pod",
				},
			}
			Expect(k8sClient.Create(ctx, existingConnection)).To(Succeed())

			By("creating a client pod")
			Expect(k8sClient.Create(ctx, clientPod)).To(Succeed())

			By("verifying the existing connection is not modified")
			connection := &tfv1.TensorFusionConnection{}
			connectionKey := types.NamespacedName{
				Name:      "test-connection-pod-controller",
				Namespace: "default",
			}
			Consistently(func() string {
				err := k8sClient.Get(ctx, connectionKey, connection)
				if err != nil {
					return ""
				}
				return connection.Spec.ClientPod
			}, 2*time.Second).Should(Equal("existing-pod"))

			Expect(k8sClient.Delete(ctx, existingConnection)).To(Succeed())
		})
	})

	Context("When handling pending owned workload", func() {
		var pod *corev1.Pod
		var workload *tfv1.TensorFusionWorkload

		BeforeEach(func() {
			workload = &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-workload",
					Namespace: "default",
				},
				Spec: tfv1.WorkloadProfileSpec{
					PoolName: "default-pool",
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
			Expect(k8sClient.Create(ctx, workload)).To(Succeed())

			pod = &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "owner-pod",
					Namespace: "default",
					Labels: map[string]string{
						constants.LabelComponent: constants.ComponentWorker,
					},
					Annotations: map[string]string{
						constants.SetPendingOwnedWorkloadAnnotation: workload.Name,
					},
				},
				Spec: corev1.PodSpec{
					NodeName: "skip-schedule",
					Containers: []corev1.Container{
						{
							Name:  "test",
							Image: "test-image",
						},
					},
				},
			}
		})

		AfterEach(func() {
			if workload != nil {
				_ = k8sClient.Delete(ctx, workload)
			}
			if pod != nil {
				_ = k8sClient.Delete(ctx, pod)
			}
		})

		It("should set owner reference for pending owned workload", func() {
			By("creating a pod with pending owned workload annotation")
			Expect(k8sClient.Create(ctx, pod)).To(Succeed())

			By("verifying owner reference is set on workload")
			Eventually(func() bool {
				updatedWorkload := &tfv1.TensorFusionWorkload{}
				err := k8sClient.Get(ctx, client.ObjectKeyFromObject(workload), updatedWorkload)
				if err != nil {
					return false
				}
				return len(updatedWorkload.OwnerReferences) == 1 &&
					updatedWorkload.OwnerReferences[0].Name == pod.Name &&
					updatedWorkload.OwnerReferences[0].Kind == "Pod"
			}).Should(BeTrue())

			By("verifying annotation is removed from pod")
			Eventually(func() bool {
				updatedPod := &corev1.Pod{}
				err := k8sClient.Get(ctx, client.ObjectKeyFromObject(pod), updatedPod)
				if err != nil {
					return false
				}
				_, exists := updatedPod.Annotations[constants.SetPendingOwnedWorkloadAnnotation]
				return !exists
			}).Should(BeTrue())
		})

		It("should handle orphaned pod when owned workload not found", func() {
			By("creating a pod with non-existent owned workload annotation")
			pod.Annotations[constants.SetPendingOwnedWorkloadAnnotation] = "non-existent-workload"
			Expect(k8sClient.Create(ctx, pod)).To(Succeed())

			By("verifying the pod still exists (orphaned pod handling)")
			Consistently(func() bool {
				updatedPod := &corev1.Pod{}
				err := k8sClient.Get(ctx, client.ObjectKeyFromObject(pod), updatedPod)
				return err == nil
			}, 2*time.Second).Should(BeTrue())
		})
	})

	Context("When testing helper functions", func() {
		It("should extract connection name and namespace from container environment", func() {
			pod := &corev1.Pod{
				Spec: corev1.PodSpec{
					NodeName: "skip-schedule",
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Env: []corev1.EnvVar{
								{
									Name:  constants.ConnectionNameEnv,
									Value: "test-connection-pod-controller",
								},
								{
									Name:  constants.ConnectionNamespaceEnv,
									Value: "test-namespace",
								},
							},
						},
					},
				},
			}

			result := findConnectionNameNamespace(pod)
			Expect(result.Name).To(Equal("test-connection-pod-controller"))
			Expect(result.Namespace).To(Equal("test-namespace"))
		})

		It("should return empty values when environment variables are missing", func() {
			pod := &corev1.Pod{
				Spec: corev1.PodSpec{
					NodeName: "skip-schedule",
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Env: []corev1.EnvVar{
								{
									Name:  "OTHER_ENV",
									Value: "other-value",
								},
							},
						},
					},
				},
			}

			result := findConnectionNameNamespace(pod)
			Expect(result.Name).To(BeEmpty())
			Expect(result.Namespace).To(BeEmpty())
		})

		It("should build connection object correctly", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					UID:       "test-uid",
					Annotations: map[string]string{
						constants.SelectedWorkloadAnnotation: "test-workload",
					},
				},
				Spec: corev1.PodSpec{
					NodeName: "skip-schedule",
					Containers: []corev1.Container{
						{
							Name: "test-container",
							Env: []corev1.EnvVar{
								{
									Name:  constants.ConnectionNameEnv,
									Value: "test-connection-pod-controller",
								},
								{
									Name:  constants.ConnectionNamespaceEnv,
									Value: "test-namespace",
								},
							},
						},
					},
				},
			}

			connection := buildTensorFusionConnectionObj(pod)
			Expect(connection).NotTo(BeNil())
			Expect(connection.Name).To(Equal("test-connection-pod-controller"))
			Expect(connection.Namespace).To(Equal("test-namespace"))
			Expect(connection.Labels[constants.WorkloadKey]).To(Equal("test-workload"))
			Expect(connection.Spec.WorkloadName).To(Equal("test-workload"))
			Expect(connection.Spec.ClientPod).To(Equal("test-pod"))
			Expect(connection.OwnerReferences).To(HaveLen(1))
			Expect(connection.OwnerReferences[0].Name).To(Equal("test-pod"))
			Expect(connection.OwnerReferences[0].UID).To(Equal(types.UID("test-uid")))
		})

		It("should return nil when selected workload annotation is missing", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
				},
				Spec: corev1.PodSpec{
					NodeName: "skip-schedule",
					Containers: []corev1.Container{
						{
							Name: "test-container",
						},
					},
				},
			}

			connection := buildTensorFusionConnectionObj(pod)
			Expect(connection).To(BeNil())
		})

		It("should return nil when connection environment variables are missing", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pod",
					Namespace: "default",
					Annotations: map[string]string{
						constants.SelectedWorkloadAnnotation: "test-workload",
					},
				},
				Spec: corev1.PodSpec{
					NodeName: "skip-schedule",
					Containers: []corev1.Container{
						{
							Name: "test-container",
						},
					},
				},
			}

			connection := buildTensorFusionConnectionObj(pod)
			Expect(connection).To(BeNil())
		})
	})
})
