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

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
	"github.com/NexusGPU/tensor-fusion-operator/internal/scheduler"
)

var _ = Describe("TensorFusionConnection Controller", func() {
	Context("When reconciling a resource", func() {
		const resourceName = "test-resource"

		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name:      resourceName,
			Namespace: "default",
		}
		scheduler := scheduler.NewNaiveScheduler()
		gpu := &tfv1.GPU{
			ObjectMeta: metav1.ObjectMeta{
				Name: "mock-gpu",
			},
		}
		BeforeEach(func() {
			connection := &tfv1.TensorFusionConnection{}
			By("creating the custom resource for the Kind TensorFusionConnection")
			err := k8sClient.Get(ctx, typeNamespacedName, connection)
			if err != nil && errors.IsNotFound(err) {
				resource := &tfv1.TensorFusionConnection{
					ObjectMeta: metav1.ObjectMeta{
						Name:      resourceName,
						Namespace: "default",
					},
					Spec: tfv1.TensorFusionConnectionSpec{
						PoolName: "mock",
						Resources: tfv1.Resources{
							Requests: tfv1.Resource{
								Tflops: resource.MustParse("1"),
								Vram:   resource.MustParse("1Gi"),
							},
							Limits: tfv1.Resource{
								Tflops: resource.MustParse("1"),
								Vram:   resource.MustParse("1Gi"),
							},
						},
					},
				}
				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}

			scheduler.OnAdd(gpu)
			Expect(k8sClient.Create(ctx, gpu)).To(Succeed())
			gpu.Status = tfv1.GPUStatus{
				Phase: tfv1.TensorFusionGPUPhaseRunning,
				UUID:  "mock-gpu",
				NodeSelector: map[string]string{
					"kubernetes.io/hostname": "mock-node",
				},
				Capacity: &tfv1.Resource{
					Tflops: resource.MustParse("2"),
					Vram:   resource.MustParse("2Gi"),
				},
				Available: &tfv1.Resource{
					Tflops: resource.MustParse("2"),
					Vram:   resource.MustParse("2Gi"),
				},
			}
			Expect(k8sClient.Status().Update(ctx, gpu)).To(Succeed())
		})

		AfterEach(func() {
			resource := &tfv1.TensorFusionConnection{}
			err := k8sClient.Get(ctx, typeNamespacedName, resource)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the specific resource instance TensorFusionConnection")
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())
		})
		It("should successfully reconcile the resource", func() {
			By("Reconciling the created resource")

			controllerReconciler := &TensorFusionConnectionReconciler{
				Client:    k8sClient,
				Scheme:    k8sClient.Scheme(),
				Scheduler: scheduler,
			}
			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())
			connection := &tfv1.TensorFusionConnection{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, connection)).NotTo(HaveOccurred())
			Expect(connection.Finalizers).Should(ConsistOf(constants.Finalizer))
			_, err = controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(k8sClient.Get(ctx, typeNamespacedName, connection)).NotTo(HaveOccurred())
			Expect(connection.Status.Phase).To(Equal(tfv1.TensorFusionConnectionStarting))

			workerPod := &corev1.Pod{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, workerPod)).NotTo(HaveOccurred())
			Expect(workerPod.Spec.NodeSelector).To(Equal(gpu.Status.NodeSelector))

			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: "mock-gpu"}, gpu)).NotTo(HaveOccurred())
			Expect(gpu.Status.Available.Tflops).To(Equal(resource.MustParse("1")))
			Expect(gpu.Status.Available.Vram).To(Equal(resource.MustParse("1Gi")))
		})
	})
})
