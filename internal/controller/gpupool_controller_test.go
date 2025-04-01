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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
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
		})
	})

	Context("When pool hypervisor config changed", func() {
		It("Should trigger reconciliation for all gpunodes in the pool", func() {
			By("changing pool hypervisor config")

		})
	})
})
