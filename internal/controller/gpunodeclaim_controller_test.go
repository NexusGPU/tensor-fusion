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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

var _ = Describe("GPUNodeClaim Controller", func() {
	Context("When reconciling pool with Karpenter provisioner", func() {
		It("should successfully create GPU node claim and karpenter node-claim", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).
				SetGpuCountPerNode(1).
				SetProvisioningMode(&tfv1.ComputingVendorConfig{
					Name:   "karpenter-aws",
					Type:   tfv1.ComputingVendorKarpenter,
					Enable: ptr.To(true),
					Params: tfv1.ComputingVendorParams{
						DefaultRegion: "us-east-1",
					},
				}).
				Build()

			// Scale up the pool to meet capacity requirement
			// will create mocked 2x g6.xlarge nodes
			Eventually(func(g Gomega) {
				tfEnv.UpdateHypervisorStatus(false)

				gpuNodeClaimList := &tfv1.GPUNodeClaimList{}
				g.Expect(k8sClient.List(ctx, gpuNodeClaimList)).Should(Succeed())

				gpuNodes := &tfv1.GPUNodeList{}
				g.Expect(k8sClient.List(ctx, gpuNodes)).Should(Succeed())

				// Add mock GPU for the provisioned nodes
				tfEnv.AddMockGPU4ProvisionedNodes(gpuNodeClaimList, gpuNodes)

				k8sNodes := &corev1.NodeList{}
				g.Expect(k8sClient.List(ctx, k8sNodes)).Should(Succeed())

				pool := tfEnv.GetGPUPool(0)
				g.Expect(pool.Status.Phase).Should(Equal(tfv1.TensorFusionPoolPhaseRunning))

				g.Expect(gpuNodeClaimList.Items).Should(HaveLen(2))
				for _, gpuNodeClaim := range gpuNodeClaimList.Items {
					g.Expect(gpuNodeClaim.Status.Phase).Should(Equal(tfv1.GPUNodeClaimBound))
					g.Expect(gpuNodeClaim.Status.InstanceID).ShouldNot(BeEmpty())
				}
			}).Should(Succeed())

			// Scale down the pool, set warm up to 100, should delete 1 g6.xlarge L4 GPU node
			tfc := tfEnv.GetCluster()
			tfc.Spec.GPUPools[0].SpecTemplate.CapacityConfig.WarmResources.TFlops = resource.MustParse("2100")
			tfEnv.UpdateCluster(tfc)
			Eventually(func(g Gomega) {
				gpuNodeClaimList := &tfv1.GPUNodeClaimList{}
				g.Expect(k8sClient.List(ctx, gpuNodeClaimList)).Should(Succeed())

				gpuNodes := &tfv1.GPUNodeList{}
				g.Expect(k8sClient.List(ctx, gpuNodes)).Should(Succeed())

				k8sNodes := &corev1.NodeList{}
				g.Expect(k8sClient.List(ctx, k8sNodes)).Should(Succeed())

				pool := tfEnv.GetGPUPool(0)
				g.Expect(pool.Status.Phase).Should(Equal(tfv1.TensorFusionPoolPhaseRunning))

				g.Expect(gpuNodeClaimList.Items).Should(HaveLen(1))
				g.Expect(gpuNodeClaimList.Items[0].Status.Phase).Should(Equal(tfv1.GPUNodeClaimBound))
				g.Expect(gpuNodeClaimList.Items[0].Status.InstanceID).ShouldNot(BeEmpty())
			}).Should(Succeed())
			tfEnv.Cleanup()
		})

		PIt("should successfully create GPU node claim and aws node", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).
				SetGpuCountPerNode(1).
				SetProvisioningMode(&tfv1.ComputingVendorConfig{
					Name:   "aws",
					Type:   tfv1.ComputingVendorAWS,
					Enable: ptr.To(true),
					Params: tfv1.ComputingVendorParams{
						DefaultRegion: "us-east-1",
					},
				}).
				Build()
			Eventually(func(g Gomega) {
				pool := tfEnv.GetGPUPool(0)
				g.Expect(pool.Status.Phase).Should(Equal(tfv1.TensorFusionPoolPhaseRunning))
			}).Should(Succeed())
			tfEnv.Cleanup()
		})
	})
})
