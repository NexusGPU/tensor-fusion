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

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

var _ = Describe("GPUNode Controller", func() {
	Context("When reconciling gpunodes", func() {
		It("should create the node discovery job and the hypervisor pod", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).
				SetGpuCountPerNode(1).
				Build()
			gpuNode := tfEnv.GetGPUNode(0, 0)

			By("checking that the node discovery job is created")
			Eventually(func(g Gomega) {
				job := &batchv1.Job{}
				g.Expect(k8sClient.Get(ctx, types.NamespacedName{
					Name:      fmt.Sprintf("node-discovery-%s", gpuNode.Name),
					Namespace: utils.CurrentNamespace(),
				}, job)).Should(Succeed())

				g.Expect(job.Spec.TTLSecondsAfterFinished).Should(Equal(ptr.To[int32](3600 * 10)))
			}).Should(Succeed())

			By("checking that the hypervisor pod is created")
			pod := &corev1.Pod{}
			Eventually(func(g Gomega) {
				err := k8sClient.Get(ctx, types.NamespacedName{
					Name:      fmt.Sprintf("hypervisor-%s", gpuNode.Name),
					Namespace: utils.CurrentNamespace(),
				}, pod)
				g.Expect(err).ShouldNot(HaveOccurred())
				g.Expect(pod.Status.Phase).Should(Equal(corev1.PodRunning))
			}).Should(Succeed())

			By("checking that the gpunode status phase should be running")
			Eventually(func(g Gomega) {
				gpunode := tfEnv.GetGPUNode(0, 0)
				g.Expect(gpunode.Status.Phase).Should(Equal(tfv1.TensorFusionGPUNodePhaseRunning))
			}).Should(Succeed())

			By("checking the hypervisor pod should be recreated when enters terminated status")
			pod.Status.Phase = corev1.PodFailed
			Expect(k8sClient.Status().Update(ctx, pod)).Should(Succeed())
			Eventually(func(g Gomega) {
				newPod := &corev1.Pod{}
				err := k8sClient.Get(ctx, types.NamespacedName{
					Name:      fmt.Sprintf("hypervisor-%s", gpuNode.Name),
					Namespace: utils.CurrentNamespace(),
				}, newPod)
				g.Expect(err).ShouldNot(HaveOccurred())
				g.Expect(newPod.UID).ShouldNot(Equal(pod.UID))
			}).Should(Succeed())

			tfEnv.Cleanup()

		})
	})

	Context("Driver probe readiness", func() {
		It("should wait for job success before allowing hypervisor creation", func() {
			scheme := runtime.NewScheme()
			Expect(tfv1.AddToScheme(scheme)).To(Succeed())
			Expect(batchv1.AddToScheme(scheme)).To(Succeed())
			Expect(corev1.AddToScheme(scheme)).To(Succeed())

			node := &tfv1.GPUNode{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node-1",
				},
			}
			gpu := &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "gpu-1",
					Labels: map[string]string{constants.LabelKeyOwner: node.Name},
				},
				Status: tfv1.GPUStatus{
					Vendor: constants.AcceleratorVendorNvidia,
				},
			}
			client := fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(node.DeepCopy(), gpu).
				Build()

			pool := &tfv1.GPUPool{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pool-1",
				},
				Spec: tfv1.GPUPoolSpec{
					ComponentConfig: &tfv1.ComponentConfig{
						Hypervisor: &tfv1.HypervisorConfig{
							Image: "hypervisor:latest",
						},
					},
				},
			}

			reconciler := &GPUNodeReconciler{
				Client:                               client,
				Scheme:                               scheme,
				CompatibleWithNvidiaContainerToolkit: true,
			}

			ctx := context.Background()
			ready, err := reconciler.ensureDriverProbeReady(ctx, node, pool)
			Expect(err).NotTo(HaveOccurred())
			Expect(ready).To(BeFalse())

			job := &batchv1.Job{}
			Expect(client.Get(ctx, ctrlclient.ObjectKey{
				Name:      getDriverProbeJobName(node.Name),
				Namespace: utils.CurrentNamespace(),
			}, job)).To(Succeed())

			job.Status.Succeeded = 1
			Expect(client.Status().Update(ctx, job)).To(Succeed())

			ready, err = reconciler.ensureDriverProbeReady(ctx, node, pool)
			Expect(err).NotTo(HaveOccurred())
			Expect(ready).To(BeTrue())
		})
	})

	Context("Vendor specific handler", func() {
		var (
			scheme *runtime.Scheme
			ctx    context.Context
		)

		BeforeEach(func() {
			scheme = runtime.NewScheme()
			Expect(corev1.AddToScheme(scheme)).To(Succeed())
			ctx = context.Background()
		})

		Describe("nvidiaHandler", func() {
			It("should return nil job when compatible mode is disabled", func() {
				handler := &nvidiaHandler{
					compatibleWithNvidiaContainerToolkit: false,
				}

				node := &tfv1.GPUNode{
					ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
				}
				pool := &tfv1.GPUPool{
					ObjectMeta: metav1.ObjectMeta{Name: "test-pool"},
				}

				job, err := handler.ComposeDriverProbeJob(ctx, node, pool)
				Expect(err).NotTo(HaveOccurred())
				Expect(job).To(BeNil())
			})

			It("should return prerequisites as ready when compatible mode is disabled", func() {
				handler := &nvidiaHandler{
					compatibleWithNvidiaContainerToolkit: false,
				}

				client := fake.NewClientBuilder().WithScheme(scheme).Build()
				reconciler := &GPUNodeReconciler{
					Client: client,
					Scheme: scheme,
				}

				prereqs, err := handler.CheckHypervisorPrerequisites(ctx, reconciler, "test-node")
				Expect(err).NotTo(HaveOccurred())
				Expect(prereqs.Ready).To(BeTrue())
				Expect(prereqs.AdditionalOwners).To(BeNil())
			})

			It("should return prerequisites as not ready when device plugin pod not found", func() {
				handler := &nvidiaHandler{
					compatibleWithNvidiaContainerToolkit: true,
				}

				client := fake.NewClientBuilder().WithScheme(scheme).Build()
				reconciler := &GPUNodeReconciler{
					Client: client,
					Scheme: scheme,
				}

				prereqs, err := handler.CheckHypervisorPrerequisites(ctx, reconciler, "test-node")
				Expect(err).NotTo(HaveOccurred())
				Expect(prereqs.Ready).To(BeFalse())
				Expect(prereqs.AdditionalOwners).To(BeNil())
			})

			It("should return device plugin pod as additional owner when found", func() {
				handler := &nvidiaHandler{
					compatibleWithNvidiaContainerToolkit: true,
				}

				devicePluginPod := &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "nvidia-device-plugin-test",
						Namespace: "default",
						Labels: map[string]string{
							"app": "nvidia-device-plugin-daemonset",
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "test-node",
					},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
					},
				}

				client := fake.NewClientBuilder().
					WithScheme(scheme).
					WithObjects(devicePluginPod).
					Build()

				reconciler := &GPUNodeReconciler{
					Client: client,
					Scheme: scheme,
				}

				prereqs, err := handler.CheckHypervisorPrerequisites(ctx, reconciler, "test-node")
				Expect(err).NotTo(HaveOccurred())
				Expect(prereqs.Ready).To(BeTrue())
				Expect(prereqs.AdditionalOwners).To(HaveLen(1))

				ownerPod, ok := prereqs.AdditionalOwners[0].(*corev1.Pod)
				Expect(ok).To(BeTrue())
				Expect(ownerPod.Name).To(Equal("nvidia-device-plugin-test"))
			})

			It("should prefer running pod over pending pod", func() {
				handler := &nvidiaHandler{
					compatibleWithNvidiaContainerToolkit: true,
				}

				pendingPod := &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "nvidia-device-plugin-pending",
						Namespace: "default",
						Labels: map[string]string{
							"app": "nvidia-device-plugin-daemonset",
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "test-node",
					},
					Status: corev1.PodStatus{
						Phase: corev1.PodPending,
					},
				}

				runningPod := &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "nvidia-device-plugin-running",
						Namespace: "default",
						Labels: map[string]string{
							"app": "nvidia-device-plugin-daemonset",
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "test-node",
					},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
					},
				}

				client := fake.NewClientBuilder().
					WithScheme(scheme).
					WithObjects(pendingPod, runningPod).
					Build()

				reconciler := &GPUNodeReconciler{
					Client: client,
					Scheme: scheme,
				}

				prereqs, err := handler.CheckHypervisorPrerequisites(ctx, reconciler, "test-node")
				Expect(err).NotTo(HaveOccurred())
				Expect(prereqs.Ready).To(BeTrue())
				Expect(prereqs.AdditionalOwners).To(HaveLen(1))

				ownerPod, ok := prereqs.AdditionalOwners[0].(*corev1.Pod)
				Expect(ok).To(BeTrue())
				Expect(ownerPod.Name).To(Equal("nvidia-device-plugin-running"))
			})

			It("should filter pods by node name", func() {
				handler := &nvidiaHandler{
					compatibleWithNvidiaContainerToolkit: true,
				}

				otherNodePod := &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "nvidia-device-plugin-other-node",
						Namespace: "default",
						Labels: map[string]string{
							"app": "nvidia-device-plugin-daemonset",
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "other-node",
					},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
					},
				}

				client := fake.NewClientBuilder().
					WithScheme(scheme).
					WithObjects(otherNodePod).
					Build()

				reconciler := &GPUNodeReconciler{
					Client: client,
					Scheme: scheme,
				}

				prereqs, err := handler.CheckHypervisorPrerequisites(ctx, reconciler, "test-node")
				Expect(err).NotTo(HaveOccurred())
				Expect(prereqs.Ready).To(BeFalse())
				Expect(prereqs.AdditionalOwners).To(BeNil())
			})

			It("should return not ready when only pending pod exists", func() {
				handler := &nvidiaHandler{
					compatibleWithNvidiaContainerToolkit: true,
				}

				pendingPod := &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "nvidia-device-plugin-pending",
						Namespace: "default",
						Labels: map[string]string{
							"app": "nvidia-device-plugin-daemonset",
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "test-node",
					},
					Status: corev1.PodStatus{
						Phase: corev1.PodPending,
					},
				}

				client := fake.NewClientBuilder().
					WithScheme(scheme).
					WithObjects(pendingPod).
					Build()

				reconciler := &GPUNodeReconciler{
					Client: client,
					Scheme: scheme,
				}

				prereqs, err := handler.CheckHypervisorPrerequisites(ctx, reconciler, "test-node")
				Expect(err).NotTo(HaveOccurred())
				Expect(prereqs.Ready).To(BeFalse())
				Expect(prereqs.AdditionalOwners).To(BeNil())
			})

			It("should skip pods with deletion timestamp", func() {
				handler := &nvidiaHandler{
					compatibleWithNvidiaContainerToolkit: true,
				}

				now := metav1.Now()
				deletingPod := &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "nvidia-device-plugin-deleting",
						Namespace:         "default",
						DeletionTimestamp: &now,
						Labels: map[string]string{
							"app": "nvidia-device-plugin-daemonset",
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "test-node",
					},
					Status: corev1.PodStatus{
						Phase: corev1.PodRunning,
					},
				}

				client := fake.NewClientBuilder().
					WithScheme(scheme).
					WithObjects(deletingPod).
					Build()

				reconciler := &GPUNodeReconciler{
					Client: client,
					Scheme: scheme,
				}

				prereqs, err := handler.CheckHypervisorPrerequisites(ctx, reconciler, "test-node")
				Expect(err).NotTo(HaveOccurred())
				Expect(prereqs.Ready).To(BeFalse())
				Expect(prereqs.AdditionalOwners).To(BeNil())
			})

			It("should skip failed pods", func() {
				handler := &nvidiaHandler{
					compatibleWithNvidiaContainerToolkit: true,
				}

				failedPod := &corev1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "nvidia-device-plugin-failed",
						Namespace: "default",
						Labels: map[string]string{
							"app": "nvidia-device-plugin-daemonset",
						},
					},
					Spec: corev1.PodSpec{
						NodeName: "test-node",
					},
					Status: corev1.PodStatus{
						Phase: corev1.PodFailed,
					},
				}

				client := fake.NewClientBuilder().
					WithScheme(scheme).
					WithObjects(failedPod).
					Build()

				reconciler := &GPUNodeReconciler{
					Client: client,
					Scheme: scheme,
				}

				prereqs, err := handler.CheckHypervisorPrerequisites(ctx, reconciler, "test-node")
				Expect(err).NotTo(HaveOccurred())
				Expect(prereqs.Ready).To(BeFalse())
				Expect(prereqs.AdditionalOwners).To(BeNil())
			})
		})

		Describe("getVendorHandler", func() {
			It("should return nvidia handler for NVIDIA vendor", func() {
				reconciler := &GPUNodeReconciler{
					CompatibleWithNvidiaContainerToolkit: true,
				}

				handler := reconciler.getVendorHandler(constants.AcceleratorVendorNvidia)
				Expect(handler).NotTo(BeNil())

				nvidiaHandler, ok := handler.(*nvidiaHandler)
				Expect(ok).To(BeTrue())
				Expect(nvidiaHandler.compatibleWithNvidiaContainerToolkit).To(BeTrue())
			})

			It("should return nil for empty vendor", func() {
				reconciler := &GPUNodeReconciler{
					CompatibleWithNvidiaContainerToolkit: true,
				}

				handler := reconciler.getVendorHandler("")
				Expect(handler).To(BeNil())
			})

			It("should return nil for unknown vendor", func() {
				reconciler := &GPUNodeReconciler{
					CompatibleWithNvidiaContainerToolkit: true,
				}

				handler := reconciler.getVendorHandler("amd")
				Expect(handler).To(BeNil())
			})
		})
	})
})
