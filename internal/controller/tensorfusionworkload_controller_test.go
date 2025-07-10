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
	"bytes"
	"context"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

var _ = Describe("TensorFusionWorkload Controller", func() {
	var tfEnv *TensorFusionEnv
	key := client.ObjectKey{Name: "mock", Namespace: "default"}

	BeforeEach(func() {
		tfEnv = NewTensorFusionEnvBuilder().
			AddPoolWithNodeCount(1).SetGpuCountPerNode(5).
			Build()
		cfg := tfEnv.GetConfig()
		go mockSchedulerLoop(ctx, cfg)
	})
	AfterEach(func() {
		cleanupWorkload(key)
		tfEnv.Cleanup()
	})

	Context("When reconciling a new workload", func() {
		It("Should create worker pods according to replicas", func() {
			pool := tfEnv.GetGPUPool(0)
			By("creating a workload")
			replicas := len(tfEnv.GetPoolGpuList(0).Items)
			workload := createTensorFusionWorkload(pool.Name, key, replicas)

			_ = checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)
		})

		It("Should allocate multiple GPUs per workload when GPUCount > 1", func() {
			pool := tfEnv.GetGPUPool(0)
			By("creating a workload that requests 2 GPUs")
			workload := &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      key.Name,
					Namespace: key.Namespace,
					Labels: map[string]string{
						constants.LabelKeyOwner: pool.Name,
					},
				},
				Spec: tfv1.WorkloadProfileSpec{
					Replicas: ptr.To(int32(1)),
					PoolName: pool.Name,
					GPUCount: 2,
					Resources: tfv1.Resources{
						Requests: tfv1.Resource{
							Tflops: resource.MustParse("10"),
							Vram:   resource.MustParse("8Gi"),
						},
						Limits: tfv1.Resource{
							Tflops: resource.MustParse("20"),
							Vram:   resource.MustParse("16Gi"),
						},
					},
				},
			}

			Expect(k8sClient.Create(ctx, workload)).To(Succeed())

			// Check that pod is created with 2 GPUs
			podList := &corev1.PodList{}
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(key.Namespace),
					client.MatchingLabels{
						constants.WorkloadKey:    key.Name,
						constants.LabelComponent: constants.ComponentWorker,
					})).Should(Succeed())
				g.Expect(podList.Items).Should(HaveLen(1))
				g.Expect(podList.Items[0].Annotations[constants.GpuCountAnnotation]).Should(Equal("2"))
			}).Should(Succeed())
		})
	})

	Context("When scaling up a workload", func() {
		It("Should create additional worker pods", func() {
			pool := tfEnv.GetGPUPool(0)

			workload := createTensorFusionWorkload(pool.Name, key, 1)

			_ = checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)

			workload = &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, key, workload)).To(Succeed())
			workload.Spec.Replicas = ptr.To(int32(3))
			Expect(k8sClient.Update(ctx, workload)).To(Succeed())

			_ = checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)
		})
	})

	Context("When resource limits change in a workload", func() {
		It("Should rebuild all worker pods", func() {
			pool := tfEnv.GetGPUPool(0)

			createTensorFusionWorkload(pool.Name, key, 2)

			podList := &corev1.PodList{}
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(key.Namespace),
					client.MatchingLabels{constants.WorkloadKey: key.Name})).Should(Succeed())
				g.Expect(podList.Items).Should(HaveLen(2))

				// Check if metrics is recorded correctly
				byteWriter := bytes.NewBuffer([]byte{})
				metricsRecorder.RecordMetrics(byteWriter)
				str := byteWriter.String()
				g.Expect(str).Should(MatchRegexp("raw_cost=\\d+"))

			}).Should(Succeed())

			// Store the original pod template hash
			var originalPodNames []string
			var originalPodTemplateHash string
			for _, pod := range podList.Items {
				originalPodNames = append(originalPodNames, pod.Name)
				originalPodTemplateHash = pod.Labels[constants.LabelKeyPodTemplateHash]
			}
			Expect(originalPodTemplateHash).NotTo(BeEmpty())

			workload := &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, key, workload)).To(Succeed())
			workload.Spec.Resources.Limits.Tflops = resource.MustParse("30")
			workload.Spec.Resources.Limits.Vram = resource.MustParse("24Gi")
			Expect(k8sClient.Update(ctx, workload)).To(Succeed())

			Eventually(func(g Gomega) {
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(key.Namespace),
					client.MatchingLabels{constants.WorkloadKey: key.Name})).Should(Succeed())
				g.Expect(podList.Items).Should(HaveLen(2))
				// Verify new pods have different names and pod template hash
				var newPodNames []string
				var newPodTemplateHash string
				for _, pod := range podList.Items {
					newPodNames = append(newPodNames, pod.Name)
					newPodTemplateHash = pod.Labels[constants.LabelKeyPodTemplateHash]
				}
				g.Expect(newPodTemplateHash).NotTo(BeEmpty())
				g.Expect(newPodTemplateHash).NotTo(Equal(originalPodTemplateHash))

				for _, originalName := range originalPodNames {
					g.Expect(newPodNames).NotTo(ContainElement(originalName))
				}
			}).Should(Succeed())

			checkWorkloadStatus(workload)
		})
	})

	Context("When scaling down a workload", func() {
		It("Should delete excess worker pods", func() {
			pool := tfEnv.GetGPUPool(0)

			workload := createTensorFusionWorkload(pool.Name, key, 3)

			_ = checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)

			workload = &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, key, workload)).To(Succeed())
			workload.Spec.Replicas = ptr.To(int32(1))
			Expect(k8sClient.Update(ctx, workload)).To(Succeed())

			_ = checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)
		})
	})

	Context("When handling pods with finalizers", func() {
		It("Should process GPU resource cleanup", func() {
			pool := tfEnv.GetGPUPool(0)

			workload := createTensorFusionWorkload(pool.Name, key, 1)
			_ = checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)

			var updatedGPU tfv1.GPU
			Eventually(func(g Gomega) bool {
				gpuList := tfEnv.GetPoolGpuList(0)
				ok := false
				updatedGPU, ok = lo.Find(gpuList.Items, func(gpu tfv1.GPU) bool {
					return gpu.Status.Available.Tflops.Equal(resource.MustParse("1990")) && gpu.Status.Available.Vram.Equal(resource.MustParse("1992Gi"))
				})
				return ok
			}).Should(BeTrue())

			Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
			workloadCopy := workload.DeepCopy()
			workloadCopy.Spec.Replicas = ptr.To(int32(0))
			Expect(k8sClient.Update(ctx, workloadCopy)).To(Succeed())
			Eventually(func(g Gomega) {
				podList := &corev1.PodList{}
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(key.Namespace),
					client.MatchingLabels{constants.WorkloadKey: key.Name})).To(Succeed())
				g.Expect(podList.Items).Should(BeEmpty())
			}).Should(Succeed())

			Eventually(func(g Gomega) {
				gpu := &tfv1.GPU{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(&updatedGPU), gpu)).NotTo(HaveOccurred())
				g.Expect(gpu.Status.Available.Tflops.Equal(resource.MustParse("2000"))).Should(BeTrue())
				g.Expect(gpu.Status.Available.Vram.Equal(resource.MustParse("2000Gi"))).Should(BeTrue())
			}).Should(Succeed())
		})
	})

	Context("When specifying GPU model in workload", func() {
		It("Should allocate GPUs of the specified model", func() {
			pool := tfEnv.GetGPUPool(0)

			// Create a workload requesting specific GPU model
			workload := createTensorFusionWorkload(pool.Name, key, 1)
			Eventually(func(g Gomega) {
				// Get the latest version of the workload
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(workload), workload)).To(Succeed())
				// Set the GPU model
				workload.Spec.GPUModel = "mock"
				// Update the workload
				g.Expect(k8sClient.Update(ctx, workload)).To(Succeed())
			}).Should(Succeed())

			_ = checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)

			// Verify pods got GPUs of the correct model
			podList := &corev1.PodList{}
			// First make sure the pod exists
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(key.Namespace),
					client.MatchingLabels{constants.WorkloadKey: key.Name})).Should(Succeed())
				// Filter out pods that are being deleted
				podList.Items = lo.Filter(podList.Items, func(pod corev1.Pod, _ int) bool {
					return pod.DeletionTimestamp == nil
				})
				g.Expect(podList.Items).Should(HaveLen(1))
			}).Should(Succeed())

			// Now check if the pod has the correct GPU
			Eventually(func(g Gomega) {
				// Get the latest version of the pod
				pod := &corev1.Pod{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKey{
					Namespace: podList.Items[0].Namespace,
					Name:      podList.Items[0].Name,
				}, pod)).Should(Succeed())
				gpuNames := strings.Split(pod.Annotations[constants.GPUDeviceIDsAnnotation], ",")
				gpuList := tfEnv.GetPoolGpuList(0)
				gpu, ok := lo.Find(gpuList.Items, func(gpu tfv1.GPU) bool {
					return gpu.Name == gpuNames[0]
				})
				g.Expect(ok).To(BeTrue())
				g.Expect(gpu.Status.GPUModel).To(Equal("mock"))
			}).Should(Succeed())
		})
	})

	Context("When deleting workload directly", func() {
		It("Should delete all pods and the workload itself", func() {
			pool := tfEnv.GetGPUPool(0)

			workload := createTensorFusionWorkload(pool.Name, key, 2)
			_ = checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)

			// wait for 2 pods to be created
			Eventually(func(g Gomega) {
				podList := &corev1.PodList{}
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(key.Namespace),
					client.MatchingLabels{constants.WorkloadKey: key.Name})).To(Succeed())
				g.Expect(podList.Items).To(HaveLen(2))
			}).Should(Succeed())

			// delete workload
			Expect(k8sClient.Delete(ctx, workload)).To(Succeed())

			// wait for all pods to be deleted
			Eventually(func(g Gomega) {
				podList := &corev1.PodList{}
				err := k8sClient.List(ctx, podList,
					client.InNamespace(key.Namespace),
					client.MatchingLabels{constants.WorkloadKey: key.Name})
				g.Expect(err).To(Succeed())
				g.Expect(podList.Items).Should(BeEmpty())
			}).Should(Succeed())

			// wait for workload itself to be deleted
			Eventually(func(g Gomega) {
				w := &tfv1.TensorFusionWorkload{}
				err := k8sClient.Get(ctx, key, w)
				g.Expect(err).To(HaveOccurred())
			}).Should(Succeed())
		})
	})

	Context("When GPUPool doesn't exist", func() {
		It("Should not create worker pod when reconciling a workload with non-existent pool", func() {
			workload := createTensorFusionWorkload("non-existent-pool", key, 1)
			podList := &corev1.PodList{}
			Consistently(func(g Gomega) {
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(workload.Namespace),
					client.MatchingLabels{constants.WorkloadKey: workload.Name})).Should(Succeed())
				g.Expect(podList.Items).Should(BeEmpty())
			}, 5*time.Second, 100*time.Millisecond).Should(Succeed())
		})
	})
})

func checkWorkerPodCount(workload *tfv1.TensorFusionWorkload) *corev1.PodList {
	GinkgoHelper()
	podList := &corev1.PodList{}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.List(ctx, podList,
			client.InNamespace(workload.Namespace),
			client.MatchingLabels{constants.WorkloadKey: workload.Name})).Should(Succeed())
		g.Expect(podList.Items).Should(HaveLen(int(*workload.Spec.Replicas)))
	}).Should(Succeed())
	return podList
}

func mockSchedulerLoop(ctx context.Context, cfg *rest.Config) {
	ticker := time.NewTicker(50 * time.Millisecond)
	clientset, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		Expect(err).To(Succeed())
	}
	for range ticker.C {
		select {
		case <-ctx.Done():
			return
		default:
			podList := &corev1.PodList{}
			_ = k8sClient.List(ctx, podList)
			for _, pod := range podList.Items {
				if pod.Spec.NodeName != "" {
					continue
				}
				go scheduleAndStartPod(&pod, clientset)
			}
		}
	}
}

func scheduleAndStartPod(pod *corev1.Pod, clientset *kubernetes.Clientset) {
	// simulate scheduling cycle Filter and Reserve
	allocRequest, _, err := allocator.ComposeAllocationRequest(pod)
	if errors.IsNotFound(err) {
		return
	}
	Expect(err).To(Succeed())
	gpus, err := allocator.Alloc(&allocRequest)
	if err != nil {
		// some test cases are expected to fail, just continue
		return
	}
	Expect(gpus).To(HaveLen(int(allocRequest.Count)))
	allocator.SyncGPUsToK8s()

	// update pod annotation
	Eventually(func(g Gomega) {
		latestPod := &corev1.Pod{}
		err := k8sClient.Get(ctx, types.NamespacedName{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		}, latestPod)
		if errors.IsNotFound(err) {
			return
		}
		g.Expect(err).To(Succeed())

		if latestPod.Annotations == nil {
			latestPod.Annotations = map[string]string{}
		}
		latestPod.Annotations[constants.GPUDeviceIDsAnnotation] = strings.Join(
			lo.Map(gpus, func(gpu *tfv1.GPU, _ int) string {
				return gpu.Name
			}), ",")
		err = k8sClient.Status().Update(ctx, latestPod)
		if errors.IsNotFound(err) {
			return
		}
		g.Expect(err).To(Succeed())

		// update pod node name
		latestPod.Spec.NodeName = gpus[0].Status.NodeSelector[constants.KubernetesHostNameLabel]

		// simulate k8s scheduler binding cycle Bind function
		binding := &corev1.Binding{
			ObjectMeta: metav1.ObjectMeta{
				Name:      pod.Name,
				Namespace: pod.Namespace,
			},
			Target: corev1.ObjectReference{
				Kind: "Node",
				Name: latestPod.Spec.NodeName,
			},
		}

		err = clientset.CoreV1().Pods(latestPod.Namespace).Bind(ctx, binding, metav1.CreateOptions{})
		if errors.IsNotFound(err) {
			return
		}
		g.Expect(err).To(Succeed())
	}).Should(Succeed())

	// simulate kubelet start the pod successfully
	patchPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
	}
	patchPod.Status.Phase = corev1.PodRunning
	patchPod.Status.Conditions = append(patchPod.Status.Conditions, corev1.PodCondition{
		Type:   corev1.PodReady,
		Status: corev1.ConditionTrue,
	})
	err = k8sClient.Status().Patch(ctx, patchPod, client.MergeFrom(&corev1.Pod{}))
	if errors.IsNotFound(err) {
		return
	}
	Expect(err).To(Succeed())
}

func checkWorkloadStatus(in *tfv1.TensorFusionWorkload) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		workload := &tfv1.TensorFusionWorkload{}
		err := k8sClient.Get(ctx, client.ObjectKeyFromObject(in), workload)
		g.Expect(err).To(Succeed())

		// Check basic status
		g.Expect(workload.Status.WorkerCount).Should(Equal(*workload.Spec.Replicas))

		if *workload.Spec.Replicas == 0 {
			return
		}
		// Check phase and conditions
		if workload.Status.WorkerCount == 0 {
			g.Expect(workload.Status.Phase).Should(Equal(tfv1.TensorFusionWorkloadPhasePending))
		} else {
			readyCondition, found := lo.Find(workload.Status.Conditions, func(c metav1.Condition) bool {
				return c.Type == "Ready"
			})
			g.Expect(found).Should(BeTrue())

			if readyCondition.Status == metav1.ConditionTrue {
				g.Expect(workload.Status.Phase).Should(Equal(tfv1.TensorFusionWorkloadPhaseRunning))
				g.Expect(readyCondition.Reason).Should(Equal("WorkloadReady"))
				g.Expect(readyCondition.Message).Should(Equal("All workers are running"))
			} else if readyCondition.Status == metav1.ConditionFalse && readyCondition.Reason == "WorkerFailed" {
				g.Expect(workload.Status.Phase).Should(Equal(tfv1.TensorFusionWorkloadPhaseFailed))
				g.Expect(readyCondition.Message).Should(ContainSubstring("Failed workers:"))
			}
		}
	}).Should(Succeed())
}

func createTensorFusionWorkload(poolName string, key client.ObjectKey, replicas int) *tfv1.TensorFusionWorkload {
	GinkgoHelper()
	tflopsRequests := resource.MustParse("10")
	vramRequests := resource.MustParse("8Gi")
	tflopsLimits := resource.MustParse("20")
	vramLimits := resource.MustParse("16Gi")

	workload := &tfv1.TensorFusionWorkload{
		ObjectMeta: metav1.ObjectMeta{
			Name:      key.Name,
			Namespace: key.Namespace,
			Labels: map[string]string{
				constants.GpuPoolKey: poolName,
			},
		},
		Spec: tfv1.WorkloadProfileSpec{
			Replicas: ptr.To(int32(replicas)),
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
			Qos: constants.QoSLevelMedium,
		},
	}

	Expect(k8sClient.Create(ctx, workload)).To(Succeed())

	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
	}).Should(Succeed())

	return workload
}

func cleanupWorkload(key client.ObjectKey) {
	GinkgoHelper()
	workload := &tfv1.TensorFusionWorkload{}

	if err := k8sClient.Get(ctx, key, workload); err != nil {
		if errors.IsNotFound(err) {
			return
		}
		Expect(err).To(HaveOccurred())
	}

	// Set replicas to 0
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
		workload.Spec.Replicas = ptr.To(int32(0))
		g.Expect(k8sClient.Update(ctx, workload)).To(Succeed())
	}).Should(Succeed())

	Eventually(func(g Gomega) {
		podList := &corev1.PodList{}
		g.Expect(k8sClient.List(ctx, podList,
			client.InNamespace(key.Namespace),
			client.MatchingLabels{constants.WorkloadKey: key.Name})).To(Succeed())
		g.Expect(podList.Items).Should(BeEmpty())
	}).Should(Succeed())

	Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
	Expect(k8sClient.Delete(ctx, workload)).To(Succeed())
	Eventually(func(g Gomega) {
		err := k8sClient.Get(ctx, key, workload)
		g.Expect(err).Should(HaveOccurred())
	}).Should(Succeed())
}
