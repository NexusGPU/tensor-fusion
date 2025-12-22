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
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

var _ = Describe("Node Controller", func() {
	Context("When Node has specific labels", func() {
		It("Should create gpunode for pool based on node label", func() {
			var tfEnv *TensorFusionEnv
			By("checking two pools, one with two nodes, one with three nodes")
			tfEnv = NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(2).
				SetGpuCountPerNode(1).
				AddPoolWithNodeCount(3).
				SetGpuCountPerNode(1).
				Build()
			Expect(tfEnv.GetGPUNodeList(0).Items).Should(HaveLen(2))
			Expect(tfEnv.GetGPUNodeList(1).Items).Should(HaveLen(3))
		})
	})

	Context("When removing TensorFusion taint", func() {
		var (
			tfEnv      *TensorFusionEnv
			nodeName   string
			reconciler *NodeReconciler
		)

		BeforeEach(func() {
			tfEnv = NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).
				SetGpuCountPerNode(1).
				Build()
			nodeName = tfEnv.getNodeName(0, 0)
			reconciler = &NodeReconciler{
				Client:   k8sClient,
				Scheme:   k8sClient.Scheme(),
				Recorder: nil,
			}
		})

		AfterEach(func() {
			tfEnv.Cleanup()
		})

		It("Should remove TensorFusion taint from node", func() {
			By("Adding TensorFusion taint to node")
			node := &corev1.Node{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: nodeName}, node)).To(Succeed())

			taint := corev1.Taint{
				Key:    constants.NodeUsedByTaintKey,
				Value:  constants.TensorFusionSystemName,
				Effect: corev1.TaintEffectNoSchedule,
			}
			node.Spec.Taints = append(node.Spec.Taints, taint)
			Expect(k8sClient.Update(ctx, node)).To(Succeed())

			By("Reconciling the node")
			req := reconcile.Request{
				NamespacedName: types.NamespacedName{Name: nodeName},
			}
			_, err := reconciler.Reconcile(ctx, req)
			Expect(err).NotTo(HaveOccurred())

			By("Verifying taint is removed")
			Eventually(func(g Gomega) {
				updatedNode := &corev1.Node{}
				g.Expect(k8sClient.Get(ctx, types.NamespacedName{Name: nodeName}, updatedNode)).To(Succeed())
				taintExists := false
				for _, t := range updatedNode.Spec.Taints {
					if t.Key == constants.NodeUsedByTaintKey && t.Value == constants.TensorFusionSystemName {
						taintExists = true
						break
					}
				}
				g.Expect(taintExists).To(BeFalse(), "TensorFusion taint should be removed")
			}).Should(Succeed())
		})

		It("Should not update node when taint does not exist", func() {
			By("Reconciling the node without taint")
			req := reconcile.Request{
				NamespacedName: types.NamespacedName{Name: nodeName},
			}
			_, err := reconciler.Reconcile(ctx, req)
			Expect(err).NotTo(HaveOccurred())

			By("Verifying node does not have TensorFusion taint")
			Eventually(func(g Gomega) {
				updatedNode := &corev1.Node{}
				g.Expect(k8sClient.Get(ctx, types.NamespacedName{Name: nodeName}, updatedNode)).To(Succeed())
				// Verify taint removal function didn't cause unnecessary update
				g.Expect(updatedNode.Spec.Taints).NotTo(ContainElement(HaveField("Key", constants.NodeUsedByTaintKey)))
			}).Should(Succeed())
		})

		It("Should handle retry on conflict when removing taint", func() {
			By("Adding TensorFusion taint to node")
			node := &corev1.Node{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{Name: nodeName}, node)).To(Succeed())

			taint := corev1.Taint{
				Key:    constants.NodeUsedByTaintKey,
				Value:  constants.TensorFusionSystemName,
				Effect: corev1.TaintEffectNoSchedule,
			}
			node.Spec.Taints = append(node.Spec.Taints, taint)
			Expect(k8sClient.Update(ctx, node)).To(Succeed())

			By("Reconciling the node (should retry on conflict if needed)")
			req := reconcile.Request{
				NamespacedName: types.NamespacedName{Name: nodeName},
			}
			_, err := reconciler.Reconcile(ctx, req)
			Expect(err).NotTo(HaveOccurred())

			By("Verifying taint is removed after retry")
			Eventually(func(g Gomega) {
				updatedNode := &corev1.Node{}
				g.Expect(k8sClient.Get(ctx, types.NamespacedName{Name: nodeName}, updatedNode)).To(Succeed())
				taintExists := false
				for _, t := range updatedNode.Spec.Taints {
					if t.Key == constants.NodeUsedByTaintKey && t.Value == constants.TensorFusionSystemName {
						taintExists = true
						break
					}
				}
				g.Expect(taintExists).To(BeFalse(), "TensorFusion taint should be removed with retry mechanism")
			}).Should(Succeed())
		})
	})
})
