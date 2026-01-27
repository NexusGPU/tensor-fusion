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
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	Context("getMatchedPoolName", func() {
		var (
			node     *corev1.Node
			poolList []tfv1.GPUPool
		)

		BeforeEach(func() {
			node = &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-node",
					Labels: map[string]string{
						"gpu-vendor": "nvidia",
						"gpu-type":   "a100",
					},
				},
			}
		})

		It("should return false when NodeManagerConfig is nil", func() {
			poolList = []tfv1.GPUPool{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pool-nil-config"},
					Spec: tfv1.GPUPoolSpec{
						NodeManagerConfig: nil,
					},
				},
			}
			pool, matched, err := getMatchedPoolName(node, poolList)
			Expect(err).NotTo(HaveOccurred())
			Expect(matched).To(BeFalse())
			Expect(pool).To(BeNil())
		})

		It("should return false when both NodeSelector and MultiVendorNodeSelector are nil", func() {
			poolList = []tfv1.GPUPool{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pool-empty-selectors"},
					Spec: tfv1.GPUPoolSpec{
						NodeManagerConfig: &tfv1.NodeManagerConfig{
							NodeSelector:            nil,
							MultiVendorNodeSelector: nil,
						},
					},
				},
			}
			pool, matched, err := getMatchedPoolName(node, poolList)
			Expect(err).NotTo(HaveOccurred())
			Expect(matched).To(BeFalse())
			Expect(pool).To(BeNil())
		})

		It("should match node using NodeSelector", func() {
			poolList = []tfv1.GPUPool{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pool-node-selector"},
					Spec: tfv1.GPUPoolSpec{
						NodeManagerConfig: &tfv1.NodeManagerConfig{
							NodeSelector: &corev1.NodeSelector{
								NodeSelectorTerms: []corev1.NodeSelectorTerm{
									{
										MatchExpressions: []corev1.NodeSelectorRequirement{
											{
												Key:      "gpu-vendor",
												Operator: corev1.NodeSelectorOpIn,
												Values:   []string{"nvidia"},
											},
										},
									},
								},
							},
						},
					},
				},
			}
			pool, matched, err := getMatchedPoolName(node, poolList)
			Expect(err).NotTo(HaveOccurred())
			Expect(matched).To(BeTrue())
			Expect(pool.Name).To(Equal("pool-node-selector"))
		})

		It("should match node using MultiVendorNodeSelector", func() {
			poolList = []tfv1.GPUPool{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pool-multi-vendor"},
					Spec: tfv1.GPUPoolSpec{
						NodeManagerConfig: &tfv1.NodeManagerConfig{
							MultiVendorNodeSelector: map[string]*corev1.NodeSelector{
								"NVIDIA": {
									NodeSelectorTerms: []corev1.NodeSelectorTerm{
										{
											MatchExpressions: []corev1.NodeSelectorRequirement{
												{
													Key:      "gpu-vendor",
													Operator: corev1.NodeSelectorOpIn,
													Values:   []string{"nvidia"},
												},
											},
										},
									},
								},
								"AMD": {
									NodeSelectorTerms: []corev1.NodeSelectorTerm{
										{
											MatchExpressions: []corev1.NodeSelectorRequirement{
												{
													Key:      "gpu-vendor",
													Operator: corev1.NodeSelectorOpIn,
													Values:   []string{"amd"},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			}
			pool, matched, err := getMatchedPoolName(node, poolList)
			Expect(err).NotTo(HaveOccurred())
			Expect(matched).To(BeTrue())
			Expect(pool.Name).To(Equal("pool-multi-vendor"))
		})

		It("should prioritize MultiVendorNodeSelector over NodeSelector", func() {
			poolList = []tfv1.GPUPool{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pool-both-selectors"},
					Spec: tfv1.GPUPoolSpec{
						NodeManagerConfig: &tfv1.NodeManagerConfig{
							NodeSelector: &corev1.NodeSelector{
								NodeSelectorTerms: []corev1.NodeSelectorTerm{
									{
										MatchExpressions: []corev1.NodeSelectorRequirement{
											{
												Key:      "non-existent-label",
												Operator: corev1.NodeSelectorOpIn,
												Values:   []string{"value"},
											},
										},
									},
								},
							},
							MultiVendorNodeSelector: map[string]*corev1.NodeSelector{
								"NVIDIA": {
									NodeSelectorTerms: []corev1.NodeSelectorTerm{
										{
											MatchExpressions: []corev1.NodeSelectorRequirement{
												{
													Key:      "gpu-vendor",
													Operator: corev1.NodeSelectorOpIn,
													Values:   []string{"nvidia"},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			}
			pool, matched, err := getMatchedPoolName(node, poolList)
			Expect(err).NotTo(HaveOccurred())
			Expect(matched).To(BeTrue())
			Expect(pool.Name).To(Equal("pool-both-selectors"))
		})

		It("should skip nil NodeSelector entries in MultiVendorNodeSelector", func() {
			poolList = []tfv1.GPUPool{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pool-with-nil-vendor"},
					Spec: tfv1.GPUPoolSpec{
						NodeManagerConfig: &tfv1.NodeManagerConfig{
							MultiVendorNodeSelector: map[string]*corev1.NodeSelector{
								"NVIDIA": nil,
								"AMD": {
									NodeSelectorTerms: []corev1.NodeSelectorTerm{
										{
											MatchExpressions: []corev1.NodeSelectorRequirement{
												{
													Key:      "gpu-vendor",
													Operator: corev1.NodeSelectorOpIn,
													Values:   []string{"nvidia"},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			}
			pool, matched, err := getMatchedPoolName(node, poolList)
			Expect(err).NotTo(HaveOccurred())
			Expect(matched).To(BeTrue())
			Expect(pool.Name).To(Equal("pool-with-nil-vendor"))
		})

		It("should return false when node does not match any selector", func() {
			node.Labels = map[string]string{
				"some-other-label": "value",
			}
			poolList = []tfv1.GPUPool{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pool-no-match"},
					Spec: tfv1.GPUPoolSpec{
						NodeManagerConfig: &tfv1.NodeManagerConfig{
							NodeSelector: &corev1.NodeSelector{
								NodeSelectorTerms: []corev1.NodeSelectorTerm{
									{
										MatchExpressions: []corev1.NodeSelectorRequirement{
											{
												Key:      "gpu-vendor",
												Operator: corev1.NodeSelectorOpIn,
												Values:   []string{"nvidia"},
											},
										},
									},
								},
							},
						},
					},
				},
			}
			pool, matched, err := getMatchedPoolName(node, poolList)
			Expect(err).NotTo(HaveOccurred())
			Expect(matched).To(BeFalse())
			Expect(pool).To(BeNil())
		})

		It("should match the first pool when multiple pools match", func() {
			poolList = []tfv1.GPUPool{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pool-first"},
					Spec: tfv1.GPUPoolSpec{
						NodeManagerConfig: &tfv1.NodeManagerConfig{
							NodeSelector: &corev1.NodeSelector{
								NodeSelectorTerms: []corev1.NodeSelectorTerm{
									{
										MatchExpressions: []corev1.NodeSelectorRequirement{
											{
												Key:      "gpu-vendor",
												Operator: corev1.NodeSelectorOpIn,
												Values:   []string{"nvidia"},
											},
										},
									},
								},
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pool-second"},
					Spec: tfv1.GPUPoolSpec{
						NodeManagerConfig: &tfv1.NodeManagerConfig{
							NodeSelector: &corev1.NodeSelector{
								NodeSelectorTerms: []corev1.NodeSelectorTerm{
									{
										MatchExpressions: []corev1.NodeSelectorRequirement{
											{
												Key:      "gpu-vendor",
												Operator: corev1.NodeSelectorOpIn,
												Values:   []string{"nvidia"},
											},
										},
									},
								},
							},
						},
					},
				},
			}
			pool, matched, err := getMatchedPoolName(node, poolList)
			Expect(err).NotTo(HaveOccurred())
			Expect(matched).To(BeTrue())
			Expect(pool.Name).To(Equal("pool-first"))
		})
	})
})
