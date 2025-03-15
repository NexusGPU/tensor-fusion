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
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

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

		BeforeEach(func() {
			// Create workload first
			workload := &tfv1.TensorFusionWorkload{}
			err := k8sClient.Get(ctx, workloadNamespacedName, workload)
			if err != nil && errors.IsNotFound(err) {
				By("creating the TensorFusionWorkload resource")
				workload = &tfv1.TensorFusionWorkload{
					ObjectMeta: metav1.ObjectMeta{
						Name:      workloadName,
						Namespace: "default",
					},
					Spec: tfv1.TensorFusionWorkloadSpec{
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
				Expect(k8sClient.Create(ctx, workload)).To(Succeed())
				workload.Status = tfv1.TensorFusionWorkloadStatus{
					Replicas:      1,
					ReadyReplicas: 1,
					WorkerStatuses: []tfv1.WorkerStatus{
						{
							WorkerPhase: tfv1.WorkerRunning,
							WorkerName:  "test-worker-1",
							WorkerIp:    "192.168.1.1",
							WorkerPort:  8080,
						},
					},
				}
				// Update status
				Expect(k8sClient.Status().Update(ctx, workload)).To(Succeed())
			}

			connection := &tfv1.TensorFusionConnection{}
			By("creating the custom resource for the Kind TensorFusionConnection")
			err = k8sClient.Get(ctx, typeNamespacedName, connection)
			if err != nil && errors.IsNotFound(err) {
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
			}
		})

		AfterEach(func() {
			// Clean up connection
			connection := &tfv1.TensorFusionConnection{}
			err := k8sClient.Get(ctx, typeNamespacedName, connection)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the specific resource instance TensorFusionConnection")
			Expect(k8sClient.Delete(ctx, connection)).To(Succeed())

			// Clean up workload
			workload := &tfv1.TensorFusionWorkload{}
			err = k8sClient.Get(ctx, workloadNamespacedName, workload)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the TensorFusionWorkload resource")
			Expect(k8sClient.Delete(ctx, workload)).To(Succeed())
		})

		It("should successfully reconcile the resource", func() {
			By("Reconciling the created resource")

			controllerReconciler := &TensorFusionConnectionReconciler{
				Client:   k8sClient,
				Scheme:   k8sClient.Scheme(),
				Recorder: record.NewFakeRecorder(10),
			}
			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())
		})

		It("should update connection status with worker information", func() {
			By("Reconciling the connection resource")

			controllerReconciler := &TensorFusionConnectionReconciler{
				Client:   k8sClient,
				Scheme:   k8sClient.Scheme(),
				Recorder: record.NewFakeRecorder(10),
			}
			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			By("Verifying the connection status is updated")
			connection := &tfv1.TensorFusionConnection{}
			Eventually(func() bool {
				if err := k8sClient.Get(ctx, typeNamespacedName, connection); err != nil {
					return false
				}
				return connection.Status.WorkerName != "" &&
					connection.Status.Phase == tfv1.WorkerRunning &&
					connection.Status.ConnectionURL != ""
			}, time.Second*5, time.Millisecond*100).Should(BeTrue())

			// Verify specific values
			Expect(connection.Status.WorkerName).To(Equal("test-worker-1"))
			Expect(connection.Status.Phase).To(Equal(tfv1.WorkerRunning))
			Expect(connection.Status.ConnectionURL).To(Equal("native+192.168.1.1+8080+test-worker-1-0"))
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

			By("Reconciling the connection without workload label")

			controllerReconciler := &TensorFusionConnectionReconciler{
				Client:   k8sClient,
				Scheme:   k8sClient.Scheme(),
				Recorder: record.NewFakeRecorder(10),
			}

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      "test-connection-no-label",
					Namespace: "default",
				},
			})

			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("missing workload label"))

			// Clean up the test connection
			Expect(k8sClient.Delete(ctx, connectionNoLabel)).To(Succeed())
		})

		It("should handle worker selection when worker status changes", func() {
			By("Setting up a connection with an already assigned worker")

			controllerReconciler := &TensorFusionConnectionReconciler{
				Client:   k8sClient,
				Scheme:   k8sClient.Scheme(),
				Recorder: record.NewFakeRecorder(10),
			}
			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify initial worker assignment
			connection := &tfv1.TensorFusionConnection{}
			Eventually(func() bool {
				if err := k8sClient.Get(ctx, typeNamespacedName, connection); err != nil {
					return false
				}
				return connection.Status.WorkerName == "test-worker-1"
			}, time.Second*5, time.Millisecond*100).Should(BeTrue())

			By("Updating the workload to mark the worker as failed")
			workload := &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, workloadNamespacedName, workload)).To(Succeed())

			// Add a new worker and mark the existing one as failed
			workload.Status.WorkerStatuses = []tfv1.WorkerStatus{
				{
					WorkerPhase: tfv1.WorkerFailed, // Mark as failed
					WorkerName:  "test-worker-1",
					WorkerIp:    "192.168.1.1",
					WorkerPort:  8080,
				},
				{
					WorkerPhase: tfv1.WorkerRunning,
					WorkerName:  "test-worker-2",
					WorkerIp:    "192.168.1.2",
					WorkerPort:  8081,
				},
			}

			Expect(k8sClient.Status().Update(ctx, workload)).To(Succeed())

			By("Reconciling the connection after worker status change")
			_, err = controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			// Verify worker reselection
			Eventually(func() bool {
				if err := k8sClient.Get(ctx, typeNamespacedName, connection); err != nil {
					return false
				}
				return connection.Status.WorkerName == "test-worker-2" &&
					connection.Status.ConnectionURL == "native+192.168.1.2+8081+test-worker-2-0"
			}, time.Second*5, time.Millisecond*100).Should(BeTrue())
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
			// Update status
			Expect(k8sClient.Status().Update(ctx, failWorkload)).To(Succeed())

			// Verify workload was created properly
			createdWorkload := &tfv1.TensorFusionWorkload{}
			Eventually(func() bool {
				if err := k8sClient.Get(ctx, failWorkloadNamespacedName, createdWorkload); err != nil {
					return false
				}
				return len(createdWorkload.Status.WorkerStatuses) == 0
			}, time.Second*5, time.Millisecond*100).Should(BeTrue())

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

			By("Reconciling the connection to trigger worker selection failure")
			controllerReconciler := &TensorFusionConnectionReconciler{
				Client:   k8sClient,
				Scheme:   k8sClient.Scheme(),
				Recorder: record.NewFakeRecorder(10),
			}

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: failConnectionNamespacedName,
			})
			// We expect an error since worker selection should fail
			Expect(err).To(HaveOccurred())

			By("Verifying the connection status is updated to WorkerPending")
			Eventually(func() bool {
				if err := k8sClient.Get(ctx, failConnectionNamespacedName, failConnection); err != nil {
					return false
				}
				return failConnection.Status.Phase == tfv1.WorkerPending
			}, time.Second*5, time.Millisecond*100).Should(BeTrue())

			By("Cleaning up test resources")
			Expect(k8sClient.Delete(ctx, failConnection)).To(Succeed())
			Expect(k8sClient.Delete(ctx, failWorkload)).To(Succeed())
		})
	})
})
