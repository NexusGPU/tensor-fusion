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
	"encoding/json"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var _ = Describe("GPUPool Controller", func() {
	Context("When reconciling a gpupool", func() {
		It("Should update status when nodes ready", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).
				SetGpuCountPerNode(1).
				Build()
			Eventually(func(g Gomega) {
				pool := tfEnv.GetGPUPool(0)
				g.Expect(pool.Status.Phase).Should(Equal(tfv1.TensorFusionPoolPhaseRunning))
			}, timeout, interval).Should(Succeed())
			tfEnv.Cleanup()
		})
	})

	Context("When reconciling hypervisor", func() {
		It("Should update hypervisor status upon configuration changes", func() {
			tfEnv := NewTensorFusionEnvBuilder().AddPoolWithNodeCount(0).Build()
			By("verifying hypervisor status should be initialized when the gpu pool is created")
			pool := tfEnv.GetGPUPool(0)
			oldHash := utils.GetObjectHash(pool.Spec.ComponentConfig.Hypervisor)
			Eventually(func(g Gomega) {
				pool := tfEnv.GetGPUPool(0)
				g.Expect(pool.Status.ComponentStatus.HypervisorVersion).To(Equal(oldHash))
				g.Expect(pool.Status.ComponentStatus.HyperVisorUpdateProgress).To(BeZero())
				g.Expect(pool.Status.ComponentStatus.HypervisorConfigSynced).To(BeFalse())
			}, timeout, interval).Should(Succeed())

			By("verifying hypervisor version should be updated upon configuration changes")
			updateHypervisorConfig(tfEnv)
			Eventually(func(g Gomega) {
				pool := tfEnv.GetGPUPool(0)
				newHash := utils.GetObjectHash(pool.Spec.ComponentConfig.Hypervisor)
				g.Expect(newHash).ShouldNot(Equal(oldHash))
				g.Expect(pool.Status.ComponentStatus.HypervisorVersion).To(Equal(newHash))
				g.Expect(pool.Status.ComponentStatus.HyperVisorUpdateProgress).To(BeZero())
				g.Expect(pool.Status.ComponentStatus.HypervisorConfigSynced).To(BeFalse())
			}, timeout, interval).Should(Succeed())

			tfEnv.Cleanup()
		})

		It("Should not update anything if AutoUpdate is false", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).
				SetGpuCountPerNode(1).
				Build()
			updateRollingUpdatePolicy(tfEnv, false, 100, "3s")
			_, oldHash := triggerHypervisorUpdate(tfEnv)
			verifyAllHypervisorPodHashConsistently(tfEnv, oldHash)
			tfEnv.Cleanup()
		})

		It("Should update according to batch interval", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(2).
				SetGpuCountPerNode(1).
				Build()

			By("configuring a large enougth batch inteval to prevent next update batch")
			updateRollingUpdatePolicy(tfEnv, true, 50, "10m")
			newHash, oldHash := triggerHypervisorUpdate(tfEnv)
			verifyHypervisorPodHash(tfEnv.GetGPUNode(0, 0), newHash)
			verifyHypervisorPodHashConsistently(tfEnv.GetGPUNode(0, 1), oldHash)
			verifyHypervisorUpdateProgressConsistently(tfEnv, 50)

			By("changing the batch inteval to trigger next update batch")
			updateRollingUpdatePolicy(tfEnv, true, 50, "3s")
			verifyHypervisorPodHash(tfEnv.GetGPUNode(0, 1), newHash)
			verifyHypervisorUpdateProgress(tfEnv, 100)

			tfEnv.Cleanup()
		})

		It("Should pause the update according to batch interval", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(2).
				SetGpuCountPerNode(1).
				Build()

			By("configuring a large enougth batch inteval to prevent next update batch")
			updateRollingUpdatePolicy(tfEnv, true, 50, "10m")
			newHash, oldHash := triggerHypervisorUpdate(tfEnv)
			verifyHypervisorPodHash(tfEnv.GetGPUNode(0, 0), newHash)
			verifyHypervisorUpdateProgress(tfEnv, 50)
			verifyHypervisorPodHashConsistently(tfEnv.GetGPUNode(0, 1), oldHash)
			verifyHypervisorUpdateProgressConsistently(tfEnv, 50)

			tfEnv.Cleanup()
		})

		It("Should perform update according to batch percentage", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(2).
				SetGpuCountPerNode(1).
				Build()
			updateRollingUpdatePolicy(tfEnv, true, 50, "3s")
			newHash, _ := triggerHypervisorUpdate(tfEnv)
			verifyAllHypervisorPodHash(tfEnv, newHash)
			verifyHypervisorUpdateProgress(tfEnv, 100)
			tfEnv.Cleanup()
		})

		// It("Should perform update according to non-divisible batch percentage", func() {
		// 	tfEnv := NewTensorFusionEnvBuilder().
		// 		AddPoolWithNodeCount(3).
		// 		SetGpuCountPerNode(1).
		// 		Build()
		// 	updateRollingUpdatePolicy(tfEnv, true, 66, "3s")
		// 	newHash, _ := triggerHypervisorUpdate(tfEnv)
		// 	verifyAllHypervisorPodHash(tfEnv, newHash)
		// 	verifyHypervisorUpdateProgress(tfEnv, 100)
		// 	tfEnv.Cleanup()
		// })

		It("Should update all nodes at once if BatchPercentage is 100", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(3).
				SetGpuCountPerNode(1).
				Build()
			updateRollingUpdatePolicy(tfEnv, true, 100, "3s")
			newHash, _ := triggerHypervisorUpdate(tfEnv)
			verifyAllHypervisorPodHash(tfEnv, newHash)
			verifyHypervisorUpdateProgress(tfEnv, 100)
			tfEnv.Cleanup()
		})
	})

	Context("When reconciling worker", func() {
		It("Should update worker status upon configuration changes", func() {
			tfEnv := NewTensorFusionEnvBuilder().AddPoolWithNodeCount(0).Build()
			By("verifying worker status should be initialized when the gpu pool is created")
			pool := tfEnv.GetGPUPool(0)
			oldHash := utils.GetObjectHash(pool.Spec.ComponentConfig.Worker)
			Eventually(func(g Gomega) {
				pool := tfEnv.GetGPUPool(0)
				g.Expect(pool.Status.ComponentStatus.WorkerVersion).To(Equal(oldHash))
				g.Expect(pool.Status.ComponentStatus.WorkerUpdateProgress).To(BeZero())
				g.Expect(pool.Status.ComponentStatus.WorkerConfigSynced).To(BeFalse())
			}, timeout, interval).Should(Succeed())

			By("verifying worker version should be updated upon configuration changes")
			updateWorkerConfig(tfEnv)
			Eventually(func(g Gomega) {
				pool := tfEnv.GetGPUPool(0)
				newHash := utils.GetObjectHash(pool.Spec.ComponentConfig.Worker)
				g.Expect(newHash).ShouldNot(Equal(oldHash))
				g.Expect(pool.Status.ComponentStatus.WorkerVersion).To(Equal(newHash))
				g.Expect(pool.Status.ComponentStatus.WorkerUpdateProgress).To(BeZero())
				g.Expect(pool.Status.ComponentStatus.WorkerConfigSynced).To(BeFalse())
			}, timeout, interval).Should(Succeed())

			tfEnv.Cleanup()
		})

		It("Should update according to batch interval", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(2).
				SetGpuCountPerNode(1).
				Build()

			By("configuring a large enougth batch inteval to prevent next update batch")
			updateRollingUpdatePolicy(tfEnv, true, 50, "10m")
			createWorkloads(tfEnv, 2)
			triggerWorkerUpdate(tfEnv)
			verifyWorkerPodContainerNameConsistently(1, "tensorfusion-worker")
			verifyWorkerUpdateProgressConsistently(tfEnv, 50)

			By("changing the batch inteval to trigger next update batch")
			updateRollingUpdatePolicy(tfEnv, true, 50, "3s")
			verifyAllWorkerPodContainerName(tfEnv, "updated-name")
			verifyWorkerUpdateProgress(tfEnv, 100)

			deleteWorkloads(2)
			tfEnv.Cleanup()
		})

		It("Should update according to batch percentage", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).
				SetGpuCountPerNode(2).
				Build()
			updateRollingUpdatePolicy(tfEnv, true, 50, "3s")
			createWorkloads(tfEnv, 2)
			triggerWorkerUpdate(tfEnv)
			verifyAllWorkerPodContainerName(tfEnv, "updated-name")
			verifyWorkerUpdateProgress(tfEnv, 100)
			deleteWorkloads(2)
			tfEnv.Cleanup()
		})

		It("Should update all workload at once if BatchPercentage is 100", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).
				SetGpuCountPerNode(2).
				Build()
			updateRollingUpdatePolicy(tfEnv, true, 100, "3s")
			createWorkloads(tfEnv, 2)
			triggerWorkerUpdate(tfEnv)
			verifyAllWorkerPodContainerName(tfEnv, "updated-name")
			verifyWorkerUpdateProgress(tfEnv, 100)
		})
	})

	Context("When reconciling client", func() {
		It("Should update client status upon configuration changes", func() {
			tfEnv := NewTensorFusionEnvBuilder().AddPoolWithNodeCount(0).Build()
			By("verifying client status should be initialized when the gpu pool is created")
			pool := tfEnv.GetGPUPool(0)
			oldHash := utils.GetObjectHash(pool.Spec.ComponentConfig.Client)
			Eventually(func(g Gomega) {
				pool := tfEnv.GetGPUPool(0)
				g.Expect(pool.Status.ComponentStatus.ClientVersion).To(Equal(oldHash))
				g.Expect(pool.Status.ComponentStatus.ClientUpdateProgress).To(BeZero())
				g.Expect(pool.Status.ComponentStatus.ClientConfigSynced).To(BeFalse())
			}, timeout, interval).Should(Succeed())

			By("verifying client version should be updated upon configuration changes")
			updateClientConfig(tfEnv)
			Eventually(func(g Gomega) {
				pool := tfEnv.GetGPUPool(0)
				newHash := utils.GetObjectHash(pool.Spec.ComponentConfig.Client)
				g.Expect(newHash).ShouldNot(Equal(oldHash))
				g.Expect(pool.Status.ComponentStatus.ClientVersion).To(Equal(newHash))
				g.Expect(pool.Status.ComponentStatus.ClientUpdateProgress).To(BeZero())
				g.Expect(pool.Status.ComponentStatus.ClientConfigSynced).To(BeFalse())
			}, timeout, interval).Should(Succeed())

			tfEnv.Cleanup()
		})

		It("Should update according to batch interval", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(2).
				SetGpuCountPerNode(1).
				Build()
			ensureGpuPoolIsRunning(tfEnv)

			createClientPods(tfEnv, 2)

			By("configuring a large enougth batch inteval to prevent next update batch")
			updateRollingUpdatePolicy(tfEnv, true, 50, "10m")
			newHash, oldHash := triggerClientUpdate(tfEnv)

			verifyClientPodWasDeleted(0)
			createClientPodByIndex(tfEnv, 0)
			verifyClientPodHash(0, newHash)

			verifyClientPodHashConsistently(1, oldHash)
			verifyClientUpdateProgressConsistently(tfEnv, 50)

			By("changing the batch inteval to trigger next update batch")
			updateRollingUpdatePolicy(tfEnv, true, 50, "3s")
			verifyClientPodWasDeleted(1)
			createClientPodByIndex(tfEnv, 1)
			verifyClientPodHash(1, newHash)
			verifyClientUpdateProgress(tfEnv, 100)

			cleanupClientPods()
			tfEnv.Cleanup()
		})

		It("Should update all client pods at once if BatchPercentage is 100", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).
				SetGpuCountPerNode(1).
				Build()
			ensureGpuPoolIsRunning(tfEnv)
			updateRollingUpdatePolicy(tfEnv, true, 100, "3s")
			replicas := 2
			createClientPods(tfEnv, replicas)
			updateClientConfig(tfEnv)
			verifyClientPodWasDeleted(0)
			verifyClientPodWasDeleted(1)
			createClientPods(tfEnv, replicas)
			verifyClientUpdateProgress(tfEnv, 100)

			cleanupClientPods()
			tfEnv.Cleanup()
		})
	})
})

func triggerHypervisorUpdate(tfEnv *TensorFusionEnv) (string, string) {
	GinkgoHelper()
	ensureGpuPoolIsRunning(tfEnv)
	oldHash := verifyGpuPoolHypervisorHash(tfEnv, "")
	updateHypervisorConfig(tfEnv)
	newHash := verifyGpuPoolHypervisorHash(tfEnv, oldHash)
	Expect(newHash).ShouldNot(Equal(oldHash))
	return newHash, oldHash
}

func updateHypervisorConfig(tfEnv *TensorFusionEnv) {
	GinkgoHelper()
	tfc := tfEnv.GetCluster()
	hypervisor := tfc.Spec.GPUPools[0].SpecTemplate.ComponentConfig.Hypervisor
	podTmpl := &corev1.PodTemplate{}
	Expect(json.Unmarshal(hypervisor.PodTemplate.Raw, podTmpl)).Should(Succeed())
	podTmpl.Template.Spec.Containers[0].Name = "updated-name"
	hypervisor.PodTemplate.Raw = lo.Must(json.Marshal(podTmpl))
	tfEnv.UpdateCluster(tfc)
}

func updateClientConfig(tfEnv *TensorFusionEnv) {
	GinkgoHelper()
	tfc := tfEnv.GetCluster()
	client := tfc.Spec.GPUPools[0].SpecTemplate.ComponentConfig.Client
	client.OperatorEndpoint = "http://localhost:8081"
	tfEnv.UpdateCluster(tfc)
}

func triggerClientUpdate(tfEnv *TensorFusionEnv) (string, string) {
	GinkgoHelper()
	ensureGpuPoolIsRunning(tfEnv)
	oldHash := verifyGpuPoolClientHash(tfEnv, "")
	updateClientConfig(tfEnv)
	newHash := verifyGpuPoolClientHash(tfEnv, oldHash)
	Expect(newHash).ShouldNot(Equal(oldHash))
	return newHash, oldHash
}

func triggerWorkerUpdate(tfEnv *TensorFusionEnv) {
	GinkgoHelper()
	ensureGpuPoolIsRunning(tfEnv)
	oldHash := verifyGpuPoolWorkerHash(tfEnv, "")
	updateWorkerConfig(tfEnv)
	newHash := verifyGpuPoolWorkerHash(tfEnv, oldHash)
	Expect(newHash).ShouldNot(Equal(oldHash))
}

func updateWorkerConfig(tfEnv *TensorFusionEnv) {
	GinkgoHelper()
	tfc := tfEnv.GetCluster()
	worker := tfc.Spec.GPUPools[0].SpecTemplate.ComponentConfig.Worker
	podTmpl := &corev1.PodTemplate{}
	Expect(json.Unmarshal(worker.PodTemplate.Raw, podTmpl)).Should(Succeed())
	podTmpl.Template.Spec.Containers[0].Name = "updated-name"
	worker.PodTemplate.Raw = lo.Must(json.Marshal(podTmpl))
	tfEnv.UpdateCluster(tfc)
}

func updateRollingUpdatePolicy(tfEnv *TensorFusionEnv, autoUpdate bool, batchPercentage int32, batchInterval string) {
	GinkgoHelper()
	tfc := tfEnv.GetCluster()
	policy := tfc.Spec.GPUPools[0].SpecTemplate.NodeManagerConfig.NodePoolRollingUpdatePolicy
	policy.AutoUpdate = ptr.To(autoUpdate)
	policy.BatchPercentage = batchPercentage
	policy.BatchInterval = batchInterval
	tfEnv.UpdateCluster(tfc)
	Eventually(func(g Gomega) {
		pool := tfEnv.GetGPUPool(0)
		newPolicy := pool.Spec.NodeManagerConfig.NodePoolRollingUpdatePolicy
		g.Expect(newPolicy.AutoUpdate).Should(Equal(policy.AutoUpdate))
		g.Expect(newPolicy.BatchPercentage).Should(Equal(policy.BatchPercentage))
		g.Expect(newPolicy.BatchInterval).Should(Equal(policy.BatchInterval))
	}, timeout, interval).Should(Succeed())
}

func verifyGpuPoolClientHash(tfEnv *TensorFusionEnv, oldHash string) string {
	GinkgoHelper()
	pool := &tfv1.GPUPool{}
	Eventually(func(g Gomega) {
		pool = tfEnv.GetGPUPool(0)
		newHash := utils.GetObjectHash(pool.Spec.ComponentConfig.Client)
		g.Expect(newHash).ShouldNot(Equal(oldHash))
		g.Expect(pool.Status.ComponentStatus.ClientVersion).To(Equal(newHash))
	}, timeout, interval).Should(Succeed())

	return pool.Status.ComponentStatus.ClientVersion
}

func verifyGpuPoolHypervisorHash(tfEnv *TensorFusionEnv, oldHash string) string {
	GinkgoHelper()
	pool := &tfv1.GPUPool{}
	Eventually(func(g Gomega) {
		pool = tfEnv.GetGPUPool(0)
		newHash := utils.GetObjectHash(pool.Spec.ComponentConfig.Hypervisor)
		g.Expect(newHash).ShouldNot(Equal(oldHash))
		g.Expect(pool.Status.ComponentStatus.HypervisorVersion).To(Equal(newHash))
	}, timeout, interval).Should(Succeed())

	return pool.Status.ComponentStatus.HypervisorVersion
}

func verifyGpuPoolWorkerHash(tfEnv *TensorFusionEnv, oldHash string) string {
	GinkgoHelper()
	pool := &tfv1.GPUPool{}
	Eventually(func(g Gomega) {
		pool = tfEnv.GetGPUPool(0)
		newHash := utils.GetObjectHash(pool.Spec.ComponentConfig.Worker)
		g.Expect(newHash).ShouldNot(Equal(oldHash))
		g.Expect(pool.Status.ComponentStatus.WorkerVersion).To(Equal(newHash))
	}, timeout, interval).Should(Succeed())

	return pool.Status.ComponentStatus.WorkerVersion
}

func verifyHypervisorPodHash(gpuNode *tfv1.GPUNode, hash string) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		pod := &corev1.Pod{}
		g.Expect(k8sClient.Get(ctx, client.ObjectKey{
			Name:      fmt.Sprintf("hypervisor-%s", gpuNode.Name),
			Namespace: utils.CurrentNamespace(),
		}, pod)).Should(Succeed())
		g.Expect(pod.Labels[constants.LabelKeyPodTemplateHash]).Should(Equal(hash))
		updatePodPhaseToRunning(pod, hash)
	}, timeout, interval).Should(Succeed())
}

func verifyClientPodHash(index int, hash string) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		pod := &corev1.Pod{}
		key := client.ObjectKey{Namespace: utils.CurrentNamespace(), Name: getClientPodName(index)}
		g.Expect(k8sClient.Get(ctx, key, pod)).Should(Succeed())
		g.Expect(pod.Labels[constants.LabelKeyPodTemplateHash]).Should(Equal(hash))
	}, timeout, interval).Should(Succeed())
}

func verifyClientPodHashConsistently(index int, hash string) {
	GinkgoHelper()
	Consistently(func(g Gomega) {
		pod := &corev1.Pod{}
		key := client.ObjectKey{Namespace: utils.CurrentNamespace(), Name: getClientPodName(index)}
		g.Expect(k8sClient.Get(ctx, key, pod)).Should(Succeed())
		g.Expect(pod.Labels[constants.LabelKeyPodTemplateHash]).Should(Equal(hash))
	}, duration, interval).Should(Succeed())
}

func verifyHypervisorPodHashConsistently(gpuNode *tfv1.GPUNode, hash string) {
	GinkgoHelper()
	Consistently(func(g Gomega) {
		pod := &corev1.Pod{}
		g.Expect(k8sClient.Get(ctx, client.ObjectKey{
			Name:      fmt.Sprintf("hypervisor-%s", gpuNode.Name),
			Namespace: utils.CurrentNamespace(),
		}, pod)).Should(Succeed())
		g.Expect(pod.Labels[constants.LabelKeyPodTemplateHash]).Should(Equal(hash))
		updatePodPhaseToRunning(pod, hash)
	}, duration, interval).Should(Succeed())
}

func verifyClientPodWasDeleted(index int) {
	Eventually(func(g Gomega) {
		pod := &corev1.Pod{}
		key := client.ObjectKey{Namespace: utils.CurrentNamespace(), Name: getClientPodName(index)}
		g.Expect(k8sClient.Get(ctx, key, pod)).ShouldNot(Succeed())
	}, timeout, interval).Should(Succeed())
}

func verifyAllHypervisorPodHash(tfEnv *TensorFusionEnv, hash string) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		nodeList := tfEnv.GetGPUNodeList(0)
		for _, gpuNode := range nodeList.Items {
			pod := &corev1.Pod{}
			g.Expect(k8sClient.Get(ctx, client.ObjectKey{
				Name:      fmt.Sprintf("hypervisor-%s", gpuNode.Name),
				Namespace: utils.CurrentNamespace(),
			}, pod)).Should(Succeed())
			g.Expect(pod.Spec.Containers[0].Name).Should(Equal("updated-name"))
			g.Expect(pod.Labels[constants.LabelKeyPodTemplateHash]).Should(Equal(hash))
			updatePodPhaseToRunning(pod, hash)
		}
	}, timeout, interval).Should(Succeed())
}

// func verifyWorkerPodContainerName(workloadIndex int, name string) {
// 	GinkgoHelper()
// 	Eventually(func(g Gomega) {
// 		podList := &corev1.PodList{}
// 		g.Expect(k8sClient.List(ctx, podList,
// 			client.InNamespace("default"),
// 			client.MatchingLabels{constants.WorkloadKey: getWorkloadName(workloadIndex)})).Should(Succeed())
// 		g.Expect(podList.Items).Should(HaveLen(1))
// 		for _, pod := range podList.Items {
// 			g.Expect(pod.Spec.Containers[0].Name).Should(Equal(name))
// 		}
// 	}, timeout, interval).Should(Succeed())
// }

func verifyWorkerPodContainerNameConsistently(workloadIndex int, name string) {
	GinkgoHelper()
	Consistently(func(g Gomega) {
		podList := &corev1.PodList{}
		g.Expect(k8sClient.List(ctx, podList,
			client.InNamespace("default"),
			client.MatchingLabels{constants.WorkloadKey: getWorkloadName(workloadIndex)})).Should(Succeed())
		g.Expect(podList.Items).Should(HaveLen(1))
		for _, pod := range podList.Items {
			g.Expect(pod.Spec.Containers[0].Name).Should(Equal(name))
		}
	}, duration, interval).Should(Succeed())
}

func verifyAllWorkerPodContainerName(tfEnv *TensorFusionEnv, name string) {
	GinkgoHelper()
	pool := tfEnv.GetGPUPool(0)
	Eventually(func(g Gomega) {
		workloadList := &tfv1.TensorFusionWorkloadList{}
		g.Expect(k8sClient.List(ctx, workloadList, client.MatchingLabels(map[string]string{
			constants.LabelKeyOwner: pool.Name,
		}))).Should(Succeed())
		for _, workload := range workloadList.Items {
			podList := &corev1.PodList{}
			g.Expect(k8sClient.List(ctx, podList,
				client.InNamespace(workload.Namespace),
				client.MatchingLabels{constants.WorkloadKey: workload.Name})).Should(Succeed())
			g.Expect(podList.Items).Should(HaveLen(int(*workload.Spec.Replicas)))
			for _, pod := range podList.Items {
				g.Expect(pod.Spec.Containers[0].Name).Should(Equal(name))
			}
		}

	}, timeout, interval).Should(Succeed())
}

func verifyAllHypervisorPodHashConsistently(tfEnv *TensorFusionEnv, hash string) {
	GinkgoHelper()
	Consistently(func(g Gomega) {
		nodeList := tfEnv.GetGPUNodeList(0)
		for _, gpuNode := range nodeList.Items {
			pod := &corev1.Pod{}
			g.Expect(k8sClient.Get(ctx, client.ObjectKey{
				Name:      fmt.Sprintf("hypervisor-%s", gpuNode.Name),
				Namespace: utils.CurrentNamespace(),
			}, pod)).Should(Succeed())
			g.Expect(pod.Labels[constants.LabelKeyPodTemplateHash]).Should(Equal(hash))
			updatePodPhaseToRunning(pod, hash)
		}
	}, duration, interval).Should(Succeed())
}

// func verifyAllWorkerPodContainerNameConsistently(tfEnv *TensorFusionEnv, name string) {
// 	GinkgoHelper()
// 	pool := tfEnv.GetGPUPool(0)
// 	Consistently(func(g Gomega) {
// 		workloadList := &tfv1.TensorFusionWorkloadList{}
// 		g.Expect(k8sClient.List(ctx, workloadList, client.MatchingLabels(map[string]string{
// 			constants.LabelKeyOwner: pool.Name,
// 		}))).Should(Succeed())
// 		for _, workload := range workloadList.Items {
// 			podList := &corev1.PodList{}
// 			g.Expect(k8sClient.List(ctx, podList,
// 				client.InNamespace(workload.Namespace),
// 				client.MatchingLabels{constants.WorkloadKey: workload.Name})).Should(Succeed())
// 			g.Expect(podList.Items).Should(HaveLen(int(*workload.Spec.Replicas)))
// 			for _, pod := range podList.Items {
// 				g.Expect(pod.Spec.Containers[0].Name).Should(Equal(name))
// 			}
// 		}

// 	}, duration, interval).Should(Succeed())
// }

func verifyHypervisorUpdateProgress(tfEnv *TensorFusionEnv, progress int32) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		pool := tfEnv.GetGPUPool(0)
		g.Expect(pool.Status.ComponentStatus.HyperVisorUpdateProgress).To(Equal(progress))
		if progress == 100 {
			g.Expect(pool.Status.ComponentStatus.HypervisorConfigSynced).To(BeTrue())
		} else {
			g.Expect(pool.Status.ComponentStatus.HypervisorConfigSynced).To(BeFalse())
		}
	}, timeout, interval).Should(Succeed())
}

func verifyWorkerUpdateProgress(tfEnv *TensorFusionEnv, progress int32) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		pool := tfEnv.GetGPUPool(0)
		g.Expect(pool.Status.ComponentStatus.WorkerUpdateProgress).To(Equal(progress))
		if progress == 100 {
			g.Expect(pool.Status.ComponentStatus.WorkerConfigSynced).To(BeTrue())
		} else {
			g.Expect(pool.Status.ComponentStatus.WorkerConfigSynced).To(BeFalse())
		}
	}, timeout, interval).Should(Succeed())
}

func verifyClientUpdateProgress(tfEnv *TensorFusionEnv, progress int32) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		pool := tfEnv.GetGPUPool(0)
		g.Expect(pool.Status.ComponentStatus.ClientUpdateProgress).To(Equal(progress))
		if progress == 100 {
			g.Expect(pool.Status.ComponentStatus.ClientConfigSynced).To(BeTrue())
		} else {
			g.Expect(pool.Status.ComponentStatus.ClientConfigSynced).To(BeFalse())
		}
	}, timeout, interval).Should(Succeed())
}

func verifyClientUpdateProgressConsistently(tfEnv *TensorFusionEnv, progress int32) {
	GinkgoHelper()
	Consistently(func(g Gomega) {
		pool := tfEnv.GetGPUPool(0)
		g.Expect(pool.Status.ComponentStatus.ClientUpdateProgress).To(Equal(progress))
		if progress == 100 {
			g.Expect(pool.Status.ComponentStatus.ClientConfigSynced).To(BeTrue())
		} else {
			g.Expect(pool.Status.ComponentStatus.ClientConfigSynced).To(BeFalse())
		}
	}, duration, interval).Should(Succeed())
}

func verifyHypervisorUpdateProgressConsistently(tfEnv *TensorFusionEnv, progress int32) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		pool := tfEnv.GetGPUPool(0)
		g.Expect(pool.Status.ComponentStatus.HyperVisorUpdateProgress).To(Equal(progress))
		if progress == 100 {
			g.Expect(pool.Status.ComponentStatus.HypervisorConfigSynced).To(BeTrue())
		} else {
			g.Expect(pool.Status.ComponentStatus.HypervisorConfigSynced).To(BeFalse())
		}
	}, timeout, interval).Should(Succeed())
}

func verifyWorkerUpdateProgressConsistently(tfEnv *TensorFusionEnv, progress int32) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		pool := tfEnv.GetGPUPool(0)
		g.Expect(pool.Status.ComponentStatus.WorkerUpdateProgress).To(Equal(progress))
		if progress == 100 {
			g.Expect(pool.Status.ComponentStatus.WorkerConfigSynced).To(BeTrue())
		} else {
			g.Expect(pool.Status.ComponentStatus.WorkerConfigSynced).To(BeFalse())
		}
	}, duration, interval).Should(Succeed())
}

// no pod controller in EnvTest, need to manually update pod status
func updatePodPhaseToRunning(pod *corev1.Pod, hash string) {
	GinkgoHelper()
	if pod.Labels[constants.LabelKeyPodTemplateHash] == hash && pod.Status.Phase != corev1.PodRunning {
		patch := client.MergeFrom(pod.DeepCopy())
		pod.Status.Phase = corev1.PodRunning
		pod.Status.Conditions = append(pod.Status.Conditions, corev1.PodCondition{Type: corev1.PodReady, Status: corev1.ConditionTrue})
		Expect(k8sClient.Status().Patch(ctx, pod, patch)).Should(Succeed())
	}
}

func ensureGpuPoolIsRunning(tfEnv *TensorFusionEnv) {
	GinkgoHelper()
	Eventually(func(g Gomega) {
		pool := tfEnv.GetGPUPool(0)
		g.Expect(pool.Status.Phase).Should(Equal(tfv1.TensorFusionPoolPhaseRunning))
	}, timeout, interval).Should(Succeed())
}

// no RepliaSet like controller in EnvTest, need to create by ourself
func createClientPodByIndex(tfEnv *TensorFusionEnv, index int) {
	GinkgoHelper()
	pool := tfEnv.GetGPUPool(0)
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      getClientPodName(index),
			Namespace: utils.CurrentNamespace(),
			Labels: map[string]string{
				constants.TensorFusionEnabledLabelKey: constants.TrueStringValue,
				constants.GpuPoolKey:                  pool.Name,
				constants.LabelKeyPodTemplateHash:     utils.GetObjectHash(pool.Spec.ComponentConfig.Client),
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "mock",
					Image: "mock",
				},
			},
		},
	}
	Expect(k8sClient.Create(ctx, pod)).Should(Succeed())

	Eventually(func(g Gomega) {
		pod := &corev1.Pod{}
		key := client.ObjectKey{Namespace: utils.CurrentNamespace(), Name: getClientPodName(index)}
		g.Expect(k8sClient.Get(ctx, key, pod)).Should(Succeed())
	}, timeout, interval).Should(Succeed())
}

func createClientPods(tfEnv *TensorFusionEnv, count int) {
	GinkgoHelper()
	pool := tfEnv.GetGPUPool(0)
	for i := range count {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      getClientPodName(i),
				Namespace: utils.CurrentNamespace(),
				Labels: map[string]string{
					constants.TensorFusionEnabledLabelKey: constants.TrueStringValue,
					constants.GpuPoolKey:                  pool.Name,
					constants.LabelKeyPodTemplateHash:     utils.GetObjectHash(pool.Spec.ComponentConfig.Client),
				},
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name:  "mock",
						Image: "mock",
					},
				},
			},
		}
		Expect(k8sClient.Create(ctx, pod)).Should(Succeed())
	}

	Eventually(func(g Gomega) {
		for i := range count {
			pod := &corev1.Pod{}
			key := client.ObjectKey{Namespace: utils.CurrentNamespace(), Name: getClientPodName(i)}
			g.Expect(k8sClient.Get(ctx, key, pod)).Should(Succeed())
		}
	}, timeout, interval).Should(Succeed())
}

func cleanupClientPods() {
	Expect(k8sClient.DeleteAllOf(ctx, &corev1.Pod{}, client.InNamespace(utils.CurrentNamespace()))).Should(Succeed())
}

func createWorkloads(tfEnv *TensorFusionEnv, count int) {
	GinkgoHelper()
	pool := tfEnv.GetGPUPool(0)
	for workloadIndex := range count {
		key := client.ObjectKey{Name: getWorkloadName(workloadIndex), Namespace: "default"}
		replicas := 1
		workload := createTensorFusionWorkload(pool.Name, key, replicas)
		checkWorkerPodCount(workload)
	}
}

func deleteWorkloads(count int) {
	GinkgoHelper()
	for workloadIndex := range count {
		key := client.ObjectKey{Name: getWorkloadName(workloadIndex), Namespace: "default"}
		cleanupWorkload(key)
	}
}

func getWorkloadName(index int) string {
	return fmt.Sprintf("workload-%d", index)
}

func getClientPodName(index int) string {
	return fmt.Sprintf("client-%d", index)
}
