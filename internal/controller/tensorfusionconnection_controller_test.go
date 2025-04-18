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
		const workloadName = "test-workload"

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
			pool := tfEnv.GetGPUPool(0)
			// Create workload first
			workload := createTensorFusionWorkload(pool.Name, workloadNamespacedName, 2)
			checkWorkerPodCount(workload)
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
				workerStatus := workload.Status.WorkerStatuses[0]
				g.Expect(connection.Status.WorkerName).Should(Equal(workerStatus.WorkerName))
				g.Expect(connection.Status.Phase).Should(Equal(workerStatus.WorkerPhase))
				connectionUrl := fmt.Sprintf("native+%s+%d+%s-%s", workerStatus.WorkerIp, workerStatus.WorkerPort, workerStatus.WorkerName, workerStatus.ResourceVersion)
				g.Expect(connection.Status.ConnectionURL).Should(Equal(connectionUrl))
			}, timeout, interval).Should(Succeed())
		})

		It("should handle missing workload label", func() {
			By("Creating a connection without workload label")
			connectionNoLabel := &tfv1.TensorFusionConnection{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-connection-no-label",
					Namespace: "default",
					// No workload label
				},
				Spec: tfv1.TensorFusionConnectionSpec{
					WorkloadName: workloadName,
				},
			}
			Expect(k8sClient.Create(ctx, connectionNoLabel)).To(Succeed())
			Consistently(func(g Gomega) {
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(connectionNoLabel), connectionNoLabel)).Should(Succeed())
				g.Expect(connectionNoLabel.Status.WorkerName).Should(BeEmpty())
			}, 5*time.Second, interval).Should(Succeed())

			// Clean up the test connection
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
				workerStatus := workload.Status.WorkerStatuses[0]
				g.Expect(connection.Status.WorkerName).Should(Equal(workerStatus.WorkerName))
			}, timeout, interval).Should(Succeed())

			By("Updating the workload to mark the worker as failed")
			Expect(k8sClient.Get(ctx, workloadNamespacedName, workload)).To(Succeed())
			workload.Status.WorkerStatuses[0].WorkerPhase = tfv1.WorkerFailed
			Expect(k8sClient.Status().Update(ctx, workload)).To(Succeed())

			// Verify worker reselection
			Expect(k8sClient.Get(ctx, workloadNamespacedName, workload)).Should(Succeed())
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.Get(ctx, typeNamespacedName, connection)).Should(Succeed())
				workerStatus := workload.Status.WorkerStatuses[1]
				g.Expect(connection.Status.WorkerName).Should(Equal(workerStatus.WorkerName))
				g.Expect(connection.Status.Phase).Should(Equal(workerStatus.WorkerPhase))
				connectionUrl := fmt.Sprintf("native+%s+%d+%s-%s", workerStatus.WorkerIp, workerStatus.WorkerPort, workerStatus.WorkerName, workerStatus.ResourceVersion)
				g.Expect(connection.Status.ConnectionURL).Should(Equal(connectionUrl))
			}, timeout, interval).Should(Succeed())
		})

		It("should update status to WorkerPending when worker selection fails", func() {
			By("Creating a TensorFusionWorkload without worker status")

			// Create a workload with no workers (empty WorkerStatuses)
			failWorkloadName := "test-workload-no-workers"
			failWorkloadNamespacedName := types.NamespacedName{
				Name:      failWorkloadName,
				Namespace: "default",
			}

			failWorkload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      failWorkloadName,
					Namespace: "default",
				},
				Spec: tfv1.TensorFusionWorkloadSpec{
					PoolName: "mock-empty",
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
					// Empty WorkerStatuses to force selection failure
					WorkerStatuses: []tfv1.WorkerStatus{},
				},
			}
			Expect(k8sClient.Create(ctx, failWorkload)).To(Succeed())

			// Verify workload was created properly
			createdWorkload := &tfv1.TensorFusionWorkload{}
			Eventually(func() bool {
				if err := k8sClient.Get(ctx, failWorkloadNamespacedName, createdWorkload); err != nil {
					return false
				}
				return len(createdWorkload.Status.WorkerStatuses) == 0
			}, timeout, interval).Should(BeTrue())

			By("Creating a connection to the workload with no workers")
			failConnectionName := "test-connection-fail"
			failConnectionNamespacedName := types.NamespacedName{
				Name:      failConnectionName,
				Namespace: "default",
			}

			failConnection := &tfv1.TensorFusionConnection{
				ObjectMeta: metav1.ObjectMeta{
					Name:      failConnectionName,
					Namespace: "default",
					Labels: map[string]string{
						constants.WorkloadKey: failWorkloadName,
					},
				},
				Spec: tfv1.TensorFusionConnectionSpec{
					WorkloadName: failWorkloadName,
				},
			}
			Expect(k8sClient.Create(ctx, failConnection)).To(Succeed())

			By("Verifying the connection status is updated to WorkerPending")
			Eventually(func() bool {
				if err := k8sClient.Get(ctx, failConnectionNamespacedName, failConnection); err != nil {
					return false
				}
				return failConnection.Status.Phase == tfv1.WorkerPending
			}, timeout, interval).Should(BeTrue())

			By("Cleaning up test resources")
			Expect(k8sClient.Delete(ctx, failConnection)).To(Succeed())
			Expect(k8sClient.Delete(ctx, failWorkload)).To(Succeed())
		})
	})
})
