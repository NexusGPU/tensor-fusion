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

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/pointer"
	"sigs.k8s.io/controller-runtime/pkg/client"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

var _ = Describe("TensorFusionWorkload Controller", func() {
	const (
		resourceName      = "test-workload"
		resourceNamespace = "default"
		poolName          = "mock"
	)

	ctx := context.Background()

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
		resource := &tfv1.TensorFusionWorkload{}
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
					Replicas: pointer.Int32(2),
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
					Replicas: pointer.Int32(1),
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

			// Wait for initial pod to be created
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

			// Scale up to 2 replicas
			workload = &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, workload)).To(Succeed())
			workload.Spec.Replicas = pointer.Int32(2)
			Expect(k8sClient.Update(ctx, workload)).To(Succeed())

			// Verify additional pod was created
			Eventually(func() int {
				err := k8sClient.List(ctx, podList,
					client.InNamespace(resourceNamespace),
					client.MatchingLabels{constants.WorkloadKey: resourceName})
				if err != nil {
					return 0
				}
				return len(podList.Items)
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(2))
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
					Replicas: pointer.Int32(2),
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

			// Wait for initial pods to be created
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

			// Store initial pod UIDs
			initialPodUIDs := make([]types.UID, len(podList.Items))
			for i, pod := range podList.Items {
				initialPodUIDs[i] = pod.UID
			}

			// Update resource limits
			workload = &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, typeNamespacedName, workload)).To(Succeed())
			workload.Spec.Resources.Limits.Tflops = resource.MustParse("30")
			workload.Spec.Resources.Limits.Vram = resource.MustParse("32Gi")
			Expect(k8sClient.Update(ctx, workload)).To(Succeed())

			// Verify pods were recreated with new UIDs
			Eventually(func() bool {
				err := k8sClient.List(ctx, podList,
					client.InNamespace(resourceNamespace),
					client.MatchingLabels{constants.WorkloadKey: resourceName})
				if err != nil || len(podList.Items) != 2 {
					return false
				}

				// Check if all pod UIDs are different from initial UIDs
				for _, pod := range podList.Items {
					for _, initialUID := range initialPodUIDs {
						if pod.UID == initialUID {
							return false
						}
					}
				}
				return true
			}, 5*time.Second, 100*time.Millisecond).Should(BeTrue())
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
					Replicas: pointer.Int32(3),
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

			// Wait for initial pods to be created
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
			workload.Spec.Replicas = pointer.Int32(1)
			Expect(k8sClient.Update(ctx, workload)).To(Succeed())

			// Verify excess pods were deleted
			Eventually(func() int {
				err := k8sClient.List(ctx, podList,
					client.InNamespace(resourceNamespace),
					client.MatchingLabels{constants.WorkloadKey: resourceName})
				if err != nil {
					return 999
				}
				return len(podList.Items)
			}, 5*time.Second, 100*time.Millisecond).Should(Equal(1))
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
					Replicas: pointer.Int32(1),
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

			var updatedGPU tfv1.GPU
			Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(gpu), &updatedGPU)).NotTo(HaveOccurred())
			Expect(updatedGPU.Status.Available.Tflops.Equal(resource.MustParse("1990"))).Should(BeTrue())
			Expect(updatedGPU.Status.Available.Vram.Equal(resource.MustParse("1992Gi"))).Should(BeTrue())

			pod := &podList.Items[0]
			Expect(k8sClient.Delete(ctx, pod)).NotTo(HaveOccurred())

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
					Replicas: pointer.Int32(1),
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

			// Verify the workload was deleted
			Eventually(func() error {
				err := k8sClient.Get(ctx, typeNamespacedName, workload)
				return err
			}, 5*time.Second, 100*time.Millisecond).Should(HaveOccurred())
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
					Replicas: pointer.Int32(1),
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

			// Verify the workload was created
			Eventually(func() error {
				err := k8sClient.Get(ctx, typeNamespacedName, workload)
				return err
			}, 5*time.Second, 100*time.Millisecond).ShouldNot(HaveOccurred())
		})
	})

	Context("When deleting a workload with multiple replicas", func() {
		It("should properly clean up workload and its pods when deleted", func() {
			By("Creating a workload with 2 replicas")
			workload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-cleanup-workload",
					Namespace: "default",
				},
				Spec: tfv1.TensorFusionWorkloadSpec{
					PoolName: "mock",
					Replicas: pointer.Int32(2),
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
			Expect(k8sClient.Create(ctx, workload)).To(Succeed())

			By("Waiting for 2 worker pods to be created")
			Eventually(func() int {
				podList := &corev1.PodList{}
				err := k8sClient.List(ctx, podList,
					client.InNamespace("default"),
					client.MatchingLabels{
						constants.LabelKeyOwner: workload.Name,
					})
				if err != nil {
					return 0
				}
				return len(podList.Items)
			}, time.Second*10, time.Millisecond*100).Should(Equal(2))

			By("Deleting the workload")
			Expect(k8sClient.Delete(ctx, workload)).To(Succeed())

			By("Verifying all resources are cleaned up")
			// Wait for workload to be deleted
			Eventually(func() bool {
				err := k8sClient.Get(ctx, types.NamespacedName{
					Name:      workload.Name,
					Namespace: workload.Namespace,
				}, &tfv1.TensorFusionWorkload{})
				return errors.IsNotFound(err)
			}, time.Second*10, time.Millisecond*100).Should(BeTrue())

			// Wait for all pods to be deleted
			Eventually(func() int {
				podList := &corev1.PodList{}
				err := k8sClient.List(ctx, podList,
					client.InNamespace("default"),
					client.MatchingLabels{
						constants.LabelKeyOwner: workload.Name,
					})
				if err != nil {
					return -1
				}
				return len(podList.Items)
			}, time.Second*10, time.Millisecond*100).Should(Equal(0))
		})
	})
})
