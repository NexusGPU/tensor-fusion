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
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("GPU Controller", func() {
	Context("When reconciling GPU", func() {
		It("Should add specific label to gpu", func() {
			tfEnv := NewTensorFusionEnvBuilder().
				AddPoolWithNodeCount(1).SetGpuCountPerNode(2).
				AddPoolWithNodeCount(1).SetGpuCountPerNode(2).
				Build()
			gpuPool := tfEnv.GetGPUPool(0)
			for _, gpu := range tfEnv.GetNodeGpuList(0, 0).Items {
				Expect(gpu.GetLabels()[constants.GpuPoolKey]).Should(Equal(gpuPool.Name))
			}
			gpuPool = tfEnv.GetGPUPool(1)
			for _, gpu := range tfEnv.GetNodeGpuList(1, 0).Items {
				Expect(gpu.GetLabels()[constants.GpuPoolKey]).Should(Equal(gpuPool.Name))
			}
			tfEnv.Cleanup()
		})
	})
})
