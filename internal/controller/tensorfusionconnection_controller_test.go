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
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

var _ = Describe("TensorFusionConnection Controller", func() {
	Context("When reconciling a resource", func() {
		const resourceName = "test-resource"
		const workloadName = "test-workload-1"
		var podList *corev1.PodList

		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name:      resourceName,
			Namespace: "default",
		}

		workloadNamespacedName := types.NamespacedName{
			Name:      workloadName,
			Namespace: "default",
		}

		var tfEnv *TensorFusionEnv
		BeforeEach(func() {
			tfEnv = NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(2).
				Build()
			cfg := tfEnv.GetConfig()
			go mockSchedulerLoop(ctx, cfg)
			pool := tfEnv.GetGPUPool(0)
			// Create workload first
			workload := createTensorFusionWorkload(pool.Name, workloadNamespacedName, 2)
			podList = checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)

			By("creating the custom resource for the Kind TensorFusionConnection")
			resource := &tfv1.TensorFusionConnection{
				ObjectMeta: metav1.ObjectMeta{
					Name:      resourceName,
					Namespace: "default",
					Labels: map[string]string{
						constants.WorkloadKey: workloadName,
					},
				},
				Spec: tfv1.TensorFusionConnectionSpec{
					WorkloadName: workloadName,
				},
			}
			Expect(k8sClient.Create(ctx, resource)).To(Succeed())
		})

		AfterEach(func() {
			// Clean up connection
			connection := &tfv1.TensorFusionConnection{}
			err := k8sClient.Get(ctx, typeNamespacedName, connection)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the specific resource instance TensorFusionConnection")
			Expect(k8sClient.Delete(ctx, connection)).To(Succeed())

			// Clean up workload
			cleanupWorkload(workloadNamespacedName)

			tfEnv.Cleanup()
		})

		It("should update connection status with worker information", func() {
			By("Verifying the connection status is updated")
			connection := &tfv1.TensorFusionConnection{}
			workload := &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, workloadNamespacedName, workload)).Should(Succeed())
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.Get(ctx, typeNamespacedName, connection)).Should(Succeed())
				g.Expect(connection.Status.WorkerName).ShouldNot(Equal(""))
			}).Should(Succeed())
		})

		It("should handle missing workload label", func() {
			By("Creating a connection without workload label")
			connectionNoLabel := &tfv1.TensorFusionConnection{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-connection-no-label",
					Namespace: "default",
				},
				Spec: tfv1.TensorFusionConnectionSpec{
					WorkloadName: workloadName,
				},
			}
			Expect(k8sClient.Create(ctx, connectionNoLabel)).To(Succeed())
			Consistently(func(g Gomega) {
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(connectionNoLabel), connectionNoLabel)).Should(Succeed())
				g.Expect(connectionNoLabel.Status.WorkerName).Should(Equal(""))
			}, 2*time.Second, 400*time.Millisecond).Should(Succeed())

			Expect(k8sClient.Delete(ctx, connectionNoLabel)).To(Succeed())
		})

		It("should handle worker selection when worker status changes", func() {
			By("Setting up a connection with an already assigned worker")
			// Verify initial worker assignment
			workload := &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, workloadNamespacedName, workload)).Should(Succeed())
			connection := &tfv1.TensorFusionConnection{}
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.Get(ctx, typeNamespacedName, connection)).Should(Succeed())
				g.Expect(connection.Status.WorkerName).ShouldNot(Equal(""))
			}).Should(Succeed())

			By("Updating the workload to mark the worker as failed")

			usedWorkerPod, ok := lo.Find(podList.Items, func(pod corev1.Pod) bool {
				return pod.Name == connection.Status.WorkerName
			})
			Expect(ok).To(BeTrue())

			failPod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      usedWorkerPod.Name,
					Namespace: usedWorkerPod.Namespace,
				},
			}
			failPod.Status.Phase = corev1.PodFailed
			failPod.Status.Reason = "Failed"
			failPod.Status.Message = "Test failure"
			Expect(k8sClient.Status().Patch(ctx, failPod, client.MergeFrom(&corev1.Pod{}))).To(Succeed())

			Eventually(func(g Gomega) {
				g.Expect(k8sClient.Get(ctx, typeNamespacedName, connection)).Should(Succeed())
				g.Expect(connection.Status.WorkerName).ShouldNot(Equal(failPod.Name))
			}).Should(Succeed())
		})

		It("should create dedicated worker for dynamic replica workload", func() {
			By("Creating a TensorFusionWorkload with dynamic replica, simulate pod webhook effect")

			workloadName := "test-workload-dynamic-replica"
			workload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      workloadName,
					Namespace: "default",
				},
				Spec: tfv1.WorkloadProfileSpec{
					PoolName: tfEnv.clusterKey.Name + "-pool-0",
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
				Status: tfv1.TensorFusionWorkloadStatus{
					Replicas:      0,
					ReadyReplicas: 0,
				},
			}
			Expect(k8sClient.Create(ctx, workload)).To(Succeed())

			By("Creating a connection use the dynamic replica workload")
			connectionName := "test-connection-dynamic-replica"
			connectionNamespacedName := types.NamespacedName{
				Name:      connectionName,
				Namespace: "default",
			}
			connection := &tfv1.TensorFusionConnection{
				ObjectMeta: metav1.ObjectMeta{
					Name:      connectionName,
					Namespace: "default",
					Labels: map[string]string{
						constants.WorkloadKey: workloadName,
					},
				},
				Spec: tfv1.TensorFusionConnectionSpec{
					WorkloadName: workloadName,
				},
			}
			Expect(k8sClient.Create(ctx, connection)).To(Succeed())

			By("Verifying the connection status is updated to WorkerRunning")
			Eventually(func() bool {
				if err := k8sClient.Get(ctx, connectionNamespacedName, connection); err != nil {
					return false
				}
				return connection.Status.Phase == tfv1.WorkerRunning
			}).Should(BeTrue())

			By("Cleaning up test resources")
			Expect(k8sClient.Delete(ctx, connection)).To(Succeed())
			Expect(k8sClient.Delete(ctx, workload)).To(Succeed())
		})
	})
})
