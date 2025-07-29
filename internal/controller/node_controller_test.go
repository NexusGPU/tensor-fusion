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
})
