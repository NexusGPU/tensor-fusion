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

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"sigs.k8s.io/controller-runtime/pkg/client"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

var _ = Describe("TensorFusionCluster Controller", func() {
	Context("When creating a cluster with a GPUPool", func() {
		It("Should create GPUPool custom resource according to spec", func() {
			ctx := context.Background()
			tfc := getMockCluster(ctx)

			By("checking that the GPUPool is created with specific name and label")
			Eventually(func() string {
				poolList := &tfv1.GPUPoolList{}
				err := k8sClient.List(ctx, poolList, client.MatchingLabels(map[string]string{
					constants.LabelKeyOwner: tfc.GetName(),
				}))
				Expect(err).NotTo(HaveOccurred())
				if len(poolList.Items) > 0 {
					return poolList.Items[0].Name
				}
				return ""
			}, timeout, interval).Should(Equal(tfc.Name + "-" + tfc.Spec.GPUPools[0].Name))
		})
	})
})
