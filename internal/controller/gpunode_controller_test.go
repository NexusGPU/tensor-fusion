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
	"fmt"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
)

var _ = Describe("GPUNode Controller", func() {
	Context("When reconciling a GPUNode", func() {
		It("should create the node discovery job and the hypervisor pod", func() {
			ctx := context.Background()
			gpuNode := getMockGPUNode(ctx, "mock-node")

			By("checking that the node discovery job is created")
			Eventually(func(g Gomega) {
				job := &batchv1.Job{}
				g.Expect(k8sClient.Get(ctx, types.NamespacedName{
					Name:      fmt.Sprintf("node-discovery-%s", gpuNode.Name),
					Namespace: utils.CurrentNamespace(),
				}, job)).Should(Succeed())

				g.Expect(job.Spec.TTLSecondsAfterFinished).Should(Equal(ptr.To[int32](3600 * 10)))
			}, timeout, interval).Should(Succeed())

			By("checking that the hypervisor pod is created")
			pod := &corev1.Pod{}
			Eventually(func() error {
				return k8sClient.Get(ctx, types.NamespacedName{
					Name:      fmt.Sprintf("hypervisor-%s", gpuNode.Name),
					Namespace: utils.CurrentNamespace(),
				}, pod)
			}, timeout, interval).Should(Succeed())

			By("checking that it will recreate terminated hypervisor pod")
			Expect(k8sClient.Delete(ctx, pod)).Should(Succeed())
			Eventually(func() error {
				return k8sClient.Get(ctx, types.NamespacedName{
					Name:      fmt.Sprintf("hypervisor-%s", gpuNode.Name),
					Namespace: utils.CurrentNamespace(),
				}, pod)
			}, timeout, interval).Should(Succeed())

			// TODO: make this test pass when implement rolling udpate
			// By("checking that the hypervisor config changed")
			// tfc := getMockCluster(ctx)
			// hypervisor := tfc.Spec.GPUPools[0].SpecTemplate.ComponentConfig.Hypervisor
			// podTmpl := &corev1.PodTemplate{}
			// err := json.Unmarshal(hypervisor.PodTemplate.Raw, podTmpl)
			// Expect(err).NotTo(HaveOccurred())
			// podTmpl.Template.Spec.Containers[0].Name = "foo"
			// hypervisor.PodTemplate.Raw = lo.Must(json.Marshal(podTmpl))
			// Expect(k8sClient.Update(ctx, tfc)).To(Succeed())
			// Eventually(func() string {
			// 	pod := &corev1.Pod{}
			// 	if err = k8sClient.Get(ctx, types.NamespacedName{
			// 		Name:      fmt.Sprintf("hypervisor-%s", gpuNode.Name),
			// 		Namespace: utils.CurrentNamespace(),
			// 	}, pod); err != nil {
			// 		return ""
			// 	}
			// 	return pod.Spec.Containers[0].Name
			// }, timeout, interval).Should(Equal("foo"))
		})
	})
})
