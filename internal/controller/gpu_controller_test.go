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

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

var _ = Describe("GPU Controller", func() {
	Context("When reconciling a GPU resource", func() {
		It("Should add a specific label with pool name", func() {
			ctx := context.Background()
			pool := getMockGPUPool(ctx)
			gpunode := getMockGPUNode(ctx, "mock-node")
			By("creating the custom resource for the Kind GPU")
			key := client.ObjectKey{Name: "mock-gpu", Namespace: "default"}
			gpu := &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{
					Name:      key.Name,
					Namespace: key.Namespace,
				},
			}
			Expect(controllerutil.SetControllerReference(gpunode, gpu, scheme.Scheme)).To(Succeed())
			Expect(k8sClient.Create(ctx, gpu)).To(Succeed())
			By("checking gpu lables")
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.Get(ctx, key, gpu)).Should(Succeed())
				g.Expect(gpu.GetLabels()[constants.GpuPoolKey]).Should(Equal(pool.Name))
			}, timeout, interval).Should(Succeed())

			Expect(k8sClient.Delete(ctx, gpu)).Should(Succeed())
		})
	})
})
