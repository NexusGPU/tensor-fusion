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
	"time"

	"github.com/aws/smithy-go/ptr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

			checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)
		})
	})

	Context("When scaling up a workload", func() {
		It("Should create additional worker pods", func() {
			pool := tfEnv.GetGPUPool(0)

			workload := createTensorFusionWorkload(pool.Name, key, 1)

			checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)

			workload = &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, key, workload)).To(Succeed())
			workload.Spec.Replicas = ptr.Int32(3)
			Expect(k8sClient.Update(ctx, workload)).To(Succeed())

			checkWorkerPodCount(workload)
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
			}, timeout, interval).Should(Succeed())

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
			}, timeout, interval).Should(Succeed())

			checkWorkloadStatus(workload)
		})
	})

	Context("When scaling down a workload", func() {
		It("Should delete excess worker pods", func() {
			pool := tfEnv.GetGPUPool(0)

			workload := createTensorFusionWorkload(pool.Name, key, 3)

			checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)

			workload = &tfv1.TensorFusionWorkload{}
			Expect(k8sClient.Get(ctx, key, workload)).To(Succeed())
			workload.Spec.Replicas = ptr.Int32(1)
			Expect(k8sClient.Update(ctx, workload)).To(Succeed())

			checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)
		})
	})

	Context("When handling pods with finalizers", func() {
		It("Should process GPU resource cleanup", func() {
			pool := tfEnv.GetGPUPool(0)

			workload := createTensorFusionWorkload(pool.Name, key, 1)
			checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)

			var updatedGPU tfv1.GPU
			Eventually(func(g Gomega) bool {
				gpuList := tfEnv.GetPoolGpuList(0)
				ok := false
				updatedGPU, ok = lo.Find(gpuList.Items, func(gpu tfv1.GPU) bool {
					return gpu.Status.Available.Tflops.Equal(resource.MustParse("1990")) && gpu.Status.Available.Vram.Equal(resource.MustParse("1992Gi"))
				})
				return ok
			}, timeout, interval).Should(BeTrue())

			Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
			workloadCopy := workload.DeepCopy()
			workloadCopy.Spec.Replicas = ptr.Int32(0)
			Expect(k8sClient.Update(ctx, workloadCopy)).To(Succeed())
			Eventually(func(g Gomega) {
				podList := &corev1.PodList{}
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(key.Namespace),
					client.MatchingLabels{constants.WorkloadKey: key.Name})).To(Succeed())
				g.Expect(podList.Items).Should(BeEmpty())
			}, timeout, interval).Should(Succeed())

			Eventually(func(g Gomega) {
				gpu := &tfv1.GPU{}
				g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(&updatedGPU), gpu)).NotTo(HaveOccurred())
				g.Expect(gpu.Status.Available.Tflops.Equal(resource.MustParse("2000"))).Should(BeTrue())
				g.Expect(gpu.Status.Available.Vram.Equal(resource.MustParse("2000Gi"))).Should(BeTrue())
			}, timeout, interval).Should(Succeed())
		})
	})

	Context("When deleting workload directly", func() {
		It("Should delete all pods and the workload itself", func() {
			pool := tfEnv.GetGPUPool(0)

			workload := createTensorFusionWorkload(pool.Name, key, 2)
			checkWorkerPodCount(workload)
			checkWorkloadStatus(workload)

			// wait for 2 pods to be created
			Eventually(func(g Gomega) {
				podList := &corev1.PodList{}
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(key.Namespace),
					client.MatchingLabels{constants.WorkloadKey: key.Name})).To(Succeed())
				g.Expect(podList.Items).To(HaveLen(2))
			}, timeout, interval).Should(Succeed())

			// delete workload
			Expect(k8sClient.Delete(ctx, workload)).To(Succeed())

			// wait for all pods to be deleted
			Eventually(func(g Gomega) {
				podList := &corev1.PodList{}
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(key.Namespace),
					client.MatchingLabels{constants.WorkloadKey: key.Name})).To(Succeed())
				g.Expect(podList.Items).Should(BeEmpty())
			}, timeout, interval).Should(Succeed())

			// wait for workload itself to be deleted
			Eventually(func(g Gomega) {
				w := &tfv1.TensorFusionWorkload{}
				err := k8sClient.Get(ctx, key, w)
				g.Expect(err).To(HaveOccurred())
			}, timeout, interval).Should(Succeed())
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

func checkWorkerPodCount(workload *tfv1.TensorFusionWorkload) {
	GinkgoHelper()
	podList := &corev1.PodList{}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.List(ctx, podList,
			client.InNamespace(workload.Namespace),
			client.MatchingLabels{constants.WorkloadKey: workload.Name})).Should(Succeed())
		g.Expect(podList.Items).Should(HaveLen(int(*workload.Spec.Replicas)))
	}, timeout, interval).Should(Succeed())
}

func checkWorkloadStatus(in *tfv1.TensorFusionWorkload) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		workload := &tfv1.TensorFusionWorkload{}
		g.Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(in), workload)).Should(Succeed())
		g.Expect(workload.Status.Replicas).Should(Equal(*workload.Spec.Replicas))
	}, timeout, interval).Should(Succeed())
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
				constants.LabelKeyOwner: poolName,
			},
		},
		Spec: tfv1.TensorFusionWorkloadSpec{
			Replicas: ptr.Int32(int32(replicas)),
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

	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
	}, timeout, interval).Should(Succeed())

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
		workload.Spec.Replicas = ptr.Int32(0)
		g.Expect(k8sClient.Update(ctx, workload)).To(Succeed())
	}, timeout, interval).Should(Succeed())

	Eventually(func(g Gomega) {
		podList := &corev1.PodList{}
		g.Expect(k8sClient.List(ctx, podList,
			client.InNamespace(key.Namespace),
			client.MatchingLabels{constants.WorkloadKey: key.Name})).To(Succeed())
		g.Expect(podList.Items).Should(BeEmpty())
	}, timeout, interval).Should(Succeed())

	Expect(k8sClient.Get(ctx, key, workload)).Should(Succeed())
	Expect(k8sClient.Delete(ctx, workload)).To(Succeed())
	Eventually(func(g Gomega) {
		err := k8sClient.Get(ctx, key, workload)
		g.Expect(err).Should(HaveOccurred())
	}, timeout, interval).Should(Succeed())
}
