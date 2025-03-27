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
	"time"

	"github.com/aws/smithy-go/ptr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	tensorfusionaiv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	scheduler "github.com/NexusGPU/tensor-fusion/internal/scheduler"
)

var _ = Describe("TensorFusionWorkload Controller", func() {
	const (
		resourceName      = "test-workload"
		resourceNamespace = "default"
		poolName          = "mock"
	)

	ctx := context.Background()
	var reconciler *TensorFusionWorkloadReconciler

	typeNamespacedName := types.NamespacedName{
		Name:      resourceName,
		Namespace: resourceNamespace,
	}

	tflopsRequests := resource.MustParse("10")
	vramRequests := resource.MustParse("8Gi")
	tflopsLimits := resource.MustParse("20")
	vramLimits := resource.MustParse("16Gi")

	var gpu *tfv1.GPU
	BeforeEach(func() {

		gpu = &tfv1.GPU{
			ObjectMeta: metav1.ObjectMeta{
				Name: "mock-gpu",
				Labels: map[string]string{
					constants.GpuPoolKey: poolName,
				},
			},
		}
		Expect(k8sClient.Create(ctx, gpu)).To(Succeed())
		gpu.Status = tfv1.GPUStatus{
			Phase:    tfv1.TensorFusionGPUPhaseRunning,
			UUID:     "mock-gpu",
			GPUModel: "mock",
			NodeSelector: map[string]string{
				"kubernetes.io/hostname": "mock-node",
			},
			Capacity: &tfv1.Resource{
				Tflops: resource.MustParse("2000"),
				Vram:   resource.MustParse("2000Gi"),
			},
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("2000"),
				Vram:   resource.MustParse("2000Gi"),
			},
		}
		Expect(k8sClient.Status().Update(ctx, gpu)).To(Succeed())

		// Set up the reconciler with mocked scheduler
		reconciler = &TensorFusionWorkloadReconciler{
			Client:    k8sClient,
			Scheme:    k8sClient.Scheme(),
			Scheduler: scheduler.NewScheduler(k8sClient),
			Recorder:  record.NewFakeRecorder(3),
			GpuInfos:  config.MockGpuInfo(),
		}

		// Clean up any pods from previous tests
		podList := &corev1.PodList{}
		err := k8sClient.List(ctx, podList,
			client.InNamespace(resourceNamespace),
			client.MatchingLabels{constants.WorkloadKey: resourceName})
		Expect(err).NotTo(HaveOccurred())

		for i := range podList.Items {
			err = k8sClient.Delete(ctx, &podList.Items[i])
			Expect(err).NotTo(HaveOccurred(), "failed to delete pod")
		}
	})

	AfterEach(func() {
		// Clean up workload resources

		resource := &tensorfusionaiv1.TensorFusionWorkload{}
		err := k8sClient.Get(ctx, typeNamespacedName, resource)
		if err == nil {
			By("remove finalizers from workload")
			if len(resource.Finalizers) > 0 {
				resource.Finalizers = []string{}
				Expect(k8sClient.Update(ctx, resource)).To(Succeed())
			}

			By("Cleaning up the test workload")
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())
		}

		By("Cleaning up the test pods")
		// List the pods
		var podList corev1.PodList
		Expect(k8sClient.List(ctx, &podList,
			client.InNamespace(resourceNamespace),
			client.MatchingLabels{constants.WorkloadKey: resourceName})).To(Succeed())

		// remove finalizers from each pod
		for i := range podList.Items {
			pod := &podList.Items[i]
			if len(pod.Finalizers) > 0 {
				pod.Finalizers = []string{}
				Expect(k8sClient.Update(ctx, pod)).To(Succeed())
			}
		}
		Expect(k8sClient.DeleteAllOf(ctx, &corev1.Pod{},
			client.InNamespace(resourceNamespace),
			client.MatchingLabels{constants.WorkloadKey: resourceName},
			client.GracePeriodSeconds(0),
		)).To(Succeed())

		By("clean up the gpu")
		Expect(k8sClient.Delete(ctx, gpu)).To(Succeed())
	})

	Context("When reconciling a new workload", func() {
		It("Should create worker pods according to replicas", func() {
			// Create a workload with 2 replicas
			workload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      resourceName,
					Namespace: resourceNamespace,
				},
				Spec: tfv1.TensorFusionWorkloadSpec{
					Replicas: ptr.Int32(2),
					PoolName: poolName,
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: tflopsRequests,
							Vram:   vramRequests,
						},
						Limits: tfv1.Resource{
							Tflops: tflopsLimits,
							Vram:   vramLimits,
						},
					},
				},
			}

			Expect(k8sClient.Create(ctx, workload)).To(Succeed())

			// Reconcile the workload
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify pods were created
			podList := &corev1.PodList{}
			Eventually(func() int {
				err := k8sClient.List(ctx, podList,
					client.InNamespace(resourceNamespace),
					client.MatchingLabels{constants.WorkloadKey: resourceName})
				if err != nil {
					return 0
				}
				return len(podList.Items)
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(2))

			// Reconcile the workload
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify workload status was updated
			Eventually(func() int32 {
				workload := &tfv1.TensorFusionWorkload{}
				err := k8sClient.Get(ctx, typeNamespacedName, workload)
				if err != nil {
					return -1
				}
				return workload.Status.Replicas
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(int32(2)))
		})
	})

	Context("When scaling up a workload", func() {
		It("Should create additional worker pods", func() {
			// Create a workload with 1 replica
			workload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      resourceName,
					Namespace: resourceNamespace,
				},
				Spec: tfv1.TensorFusionWorkloadSpec{
					Replicas: ptr.Int32(1),
					PoolName: poolName,
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: tflopsRequests,
							Vram:   vramRequests,
						},
						Limits: tfv1.Resource{
							Tflops: tflopsLimits,
							Vram:   vramLimits,
						},
					},
				},
			}

			Expect(k8sClient.Create(ctx, workload)).To(Succeed())

			// First reconcile to create the initial pods
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Check initial pod count
			podList := &corev1.PodList{}
			Eventually(func() int {
				err := k8sClient.List(ctx, podList,
					client.InNamespace(resourceNamespace),
					client.MatchingLabels{constants.WorkloadKey: resourceName})
				if err != nil {
					return 0
				}
				return len(podList.Items)
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(1))

			// Scale up to 3 replicas
			workload = &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, workload)).To(Succeed())
			workload.Spec.Replicas = ptr.Int32(3)
			Expect(k8sClient.Update(ctx, workload)).To(Succeed())

			// Reconcile again to handle the scale up
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify pods were scaled up
			Eventually(func() int {
				err := k8sClient.List(ctx, podList,
					client.InNamespace(resourceNamespace),
					client.MatchingLabels{constants.WorkloadKey: resourceName})
				if err != nil {
					return 0
				}
				return len(podList.Items)
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(3))

			// Reconcile again to handle status
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify workload status was updated
			Eventually(func() int32 {
				workload := &tfv1.TensorFusionWorkload{}
				err := k8sClient.Get(ctx, typeNamespacedName, workload)
				if err != nil {
					return -1
				}
				return workload.Status.Replicas
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(int32(3)))
		})
	})

	Context("When resource limits change in a workload", func() {
		It("Should rebuild all worker pods", func() {
			// Create a workload with 2 replicas
			workload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      resourceName,
					Namespace: resourceNamespace,
				},
				Spec: tfv1.TensorFusionWorkloadSpec{
					Replicas: ptr.Int32(2),
					PoolName: poolName,
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: tflopsRequests,
							Vram:   vramRequests,
						},
						Limits: tfv1.Resource{
							Tflops: tflopsLimits,
							Vram:   vramLimits,
						},
					},
				},
			}

			Expect(k8sClient.Create(ctx, workload)).To(Succeed())

			// First reconcile to create the initial pods
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Check that pods are created
			podList := &corev1.PodList{}
			Eventually(func() int {
				err := k8sClient.List(ctx, podList,
					client.InNamespace(resourceNamespace),
					client.MatchingLabels{constants.WorkloadKey: resourceName})
				if err != nil {
					return 0
				}
				return len(podList.Items)
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(2))

			// Store the original pod template hash
			var originalPodNames []string
			var originalPodTemplateHash string
			for _, pod := range podList.Items {
				originalPodNames = append(originalPodNames, pod.Name)
				originalPodTemplateHash = pod.Labels[constants.LabelKeyPodTemplateHash]
			}
			Expect(originalPodTemplateHash).NotTo(BeEmpty())

			// Update workload with different resource limits
			workload = &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, workload)).To(Succeed())
			workload.Spec.Resources.Limits.Tflops = resource.MustParse("30") // Increase TFLOPS limit
			workload.Spec.Resources.Limits.Vram = resource.MustParse("24Gi") // Increase VRAM limit
			Expect(k8sClient.Update(ctx, workload)).To(Succeed())

			// Reconcile to handle the resource limits change
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Reconcile again to handle the Finalizer
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify old pods are deleted due to template hash change
			Eventually(func() bool {
				podList := &corev1.PodList{}
				err := k8sClient.List(ctx, podList,
					client.InNamespace(resourceNamespace),
					client.MatchingLabels{constants.WorkloadKey: resourceName})
				if err != nil || len(podList.Items) != 0 {
					return false
				}
				return true // All pods should be deleted
			}, 5*time.Second, 100*time.Millisecond).Should(BeTrue())

			// Reconcile again to create new pods
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify new pods are created
			Eventually(func() int {
				err := k8sClient.List(ctx, podList,
					client.InNamespace(resourceNamespace),
					client.MatchingLabels{constants.WorkloadKey: resourceName})
				if err != nil {
					return 0
				}
				return len(podList.Items)
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(2))

			// Verify new pods have different names and pod template hash
			var newPodNames []string
			var newPodTemplateHash string
			for _, pod := range podList.Items {
				newPodNames = append(newPodNames, pod.Name)
				newPodTemplateHash = pod.Labels[constants.LabelKeyPodTemplateHash]
			}
			Expect(newPodTemplateHash).NotTo(BeEmpty())
			Expect(newPodTemplateHash).NotTo(Equal(originalPodTemplateHash))

			// Verify that pod names have changed
			for _, originalName := range originalPodNames {
				Expect(newPodNames).NotTo(ContainElement(originalName))
			}

			// Reconcile again to handle status
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify workload status was updated
			Eventually(func() int32 {
				workload := &tfv1.TensorFusionWorkload{}
				err = k8sClient.Get(ctx, typeNamespacedName, workload)
				if err != nil {
					return -1
				}
				return workload.Status.Replicas
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(int32(2)))
		})
	})

	Context("When scaling down a workload", func() {
		It("Should delete excess worker pods", func() {
			// Create a workload with 3 replicas
			workload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      resourceName,
					Namespace: resourceNamespace,
				},
				Spec: tfv1.TensorFusionWorkloadSpec{
					Replicas: ptr.Int32(3),
					PoolName: poolName,
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: tflopsRequests,
							Vram:   vramRequests,
						},
						Limits: tfv1.Resource{
							Tflops: tflopsLimits,
							Vram:   vramLimits,
						},
					},
				},
			}

			Expect(k8sClient.Create(ctx, workload)).To(Succeed())

			// First reconcile to create the initial pods
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Check initial pod count
			podList := &corev1.PodList{}
			Eventually(func() int {
				err := k8sClient.List(ctx, podList,
					client.InNamespace(resourceNamespace),
					client.MatchingLabels{constants.WorkloadKey: resourceName})
				if err != nil {
					return 0
				}
				return len(podList.Items)
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(3))

			// Scale down to 1 replica
			workload = &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, workload)).To(Succeed())
			workload.Spec.Replicas = ptr.Int32(1)
			Expect(k8sClient.Update(ctx, workload)).To(Succeed())

			// Reconcile again to handle the scale down
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Reconcile again to handle the Finalizer
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify pods were scaled down
			Eventually(func() int {
				err := k8sClient.List(ctx, podList,
					client.InNamespace(resourceNamespace),
					client.MatchingLabels{constants.WorkloadKey: resourceName})
				if err != nil {
					return -1
				}
				return len(podList.Items)
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(1))

			// Reconcile again to handle status update
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify workload status was updated
			Eventually(func() int32 {
				workload := &tfv1.TensorFusionWorkload{}
				err := k8sClient.Get(ctx, typeNamespacedName, workload)
				if err != nil {
					return -1
				}
				return workload.Status.Replicas
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(int32(1)))
		})
	})

	Context("When handling pods with finalizers", func() {
		It("Should process GPU resource cleanup", func() {
			// Create a workload
			workload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      resourceName,
					Namespace: resourceNamespace,
				},
				Spec: tfv1.TensorFusionWorkloadSpec{
					Replicas: ptr.Int32(1),
					PoolName: poolName,
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: tflopsRequests,
							Vram:   vramRequests,
						},
						Limits: tfv1.Resource{
							Tflops: tflopsLimits,
							Vram:   vramLimits,
						},
					},
				},
			}

			Expect(k8sClient.Create(ctx, workload)).To(Succeed())

			// Reconcile to create the pod
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			var updatedGPU tfv1.GPU
			Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(gpu), &updatedGPU)).NotTo(HaveOccurred())
			Expect(updatedGPU.Status.Available.Tflops.Equal(resource.MustParse("1990"))).Should(BeTrue())
			Expect(updatedGPU.Status.Available.Vram.Equal(resource.MustParse("1992Gi"))).Should(BeTrue())

			// Get the created pod
			podList := &corev1.PodList{}
			Eventually(func() int {
				err := k8sClient.List(ctx, podList,
					client.InNamespace(resourceNamespace),
					client.MatchingLabels{constants.WorkloadKey: resourceName})
				if err != nil {
					return 0
				}
				return len(podList.Items)
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(1))

			pod := &podList.Items[0]

			Expect(k8sClient.Delete(ctx, pod)).NotTo(HaveOccurred())

			// Reconcile to process the finalizer
			_, err = reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(gpu), &updatedGPU)).NotTo(HaveOccurred())
			Expect(gpu.Status.Available.Tflops.Equal(resource.MustParse("2000"))).Should(BeTrue())
			Expect(gpu.Status.Available.Vram.Equal(resource.MustParse("2000Gi"))).Should(BeTrue())
		})
	})

	Context("When a workload is deleted", func() {
		It("Should not error when reconciling a deleted workload", func() {
			// Create and immediately delete a workload
			workload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      resourceName,
					Namespace: resourceNamespace,
				},
				Spec: tfv1.TensorFusionWorkloadSpec{
					Replicas: ptr.Int32(1),
					PoolName: poolName,
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: tflopsRequests,
							Vram:   vramRequests,
						},
						Limits: tfv1.Resource{
							Tflops: tflopsLimits,
							Vram:   vramLimits,
						},
					},
				},
			}

			Expect(k8sClient.Create(ctx, workload)).To(Succeed())
			Expect(k8sClient.Delete(ctx, workload)).To(Succeed())

			// Reconcile should not error even though workload is gone
			result, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(result).To(Equal(reconcile.Result{}))
		})
	})

	Context("When GPUPool doesn't exist", func() {
		It("Should return an error when reconciling a workload with non-existent pool", func() {
			// Create a workload with non-existent pool
			workload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      resourceName,
					Namespace: resourceNamespace,
				},
				Spec: tfv1.TensorFusionWorkloadSpec{
					Replicas: ptr.Int32(1),
					PoolName: "non-existent-pool",
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: tflopsRequests,
							Vram:   vramRequests,
						},
						Limits: tfv1.Resource{
							Tflops: tflopsLimits,
							Vram:   vramLimits,
						},
					},
				},
			}

			Expect(k8sClient.Create(ctx, workload)).To(Succeed())

			// Reconcile should return error
			_, err := reconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("does not exist"))
		})
	})
})
