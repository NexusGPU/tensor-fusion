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

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

var _ = Describe("GPUNode Controller", func() {
	Context("When reconciling a resource", func() {
		const resourceName = "test-resource"

		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name:      resourceName,
			Namespace: "default",
		}
		gpunode := &tfv1.GPUNode{}
		BeforeEach(func() {
			By("creating the custom resource for the Kind GPUNode")
			err := k8sClient.Get(ctx, typeNamespacedName, gpunode)
			if err != nil && errors.IsNotFound(err) {
				resource := &tfv1.GPUNode{
					ObjectMeta: metav1.ObjectMeta{
						Name:      resourceName,
						Namespace: "default",
						Labels: map[string]string{
							fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, "mock"): "true",
						},
					},
				}
				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
				resource.Status.KubernetesNodeName = resource.Name
				resource.Status.Phase = tfv1.TensorFusionGPUNodePhaseRunning
				Expect(k8sClient.Status().Update(ctx, resource)).To(Succeed())
			}
		})

		AfterEach(func() {
			resource := &tfv1.GPUNode{}
			err := k8sClient.Get(ctx, typeNamespacedName, resource)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the specific resource instance GPUNode")
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())
		})

		It("should successfully reconcile the resource", func() {
			By("Reconciling the created resource")
			controllerReconciler := &GPUNodeReconciler{
				Client: k8sClient,
				Scheme: k8sClient.Scheme(),
			}

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

			By("Verify the finalizer is added")
			Expect(k8sClient.Get(ctx, typeNamespacedName, gpunode)).To(Succeed())
			Expect(gpunode.Finalizers).Should(ConsistOf(constants.Finalizer))

			By("Verify the node discovery job is created")
			job := &batchv1.Job{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{
				Name:      fmt.Sprintf("node-discovery-%s", gpunode.Name),
				Namespace: utils.CurrentNamespace(),
			}, job)).To(Succeed())
			Expect(job.Spec.TTLSecondsAfterFinished).Should(Equal(ptr.To[int32](3600 * 10)))

			By("Verify the hypervisor pod is created")
			pod := &corev1.Pod{}
			Expect(k8sClient.Get(ctx, types.NamespacedName{
				Name:      fmt.Sprintf("hypervisor-%s", gpunode.Name),
				Namespace: utils.CurrentNamespace(),
			}, pod)).To(Succeed())
		})
	})
})
