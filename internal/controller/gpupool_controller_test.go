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
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var _ = Describe("GPUPool Controller", func() {
	Context("When reconciling a resource", func() {
		const resourceName = "test-resource"

		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name:      resourceName,
			Namespace: "default",
		}
		nodeNamespacedName := types.NamespacedName{
			Name: "test-node",
		}
		gpupool := &tfv1.GPUPool{}

		BeforeEach(func() {
			By("creating the custom resource for the Kind GPUPool")
			err := k8sClient.Get(ctx, typeNamespacedName, gpupool)
			if err != nil && errors.IsNotFound(err) {
				resource := &tfv1.GPUPool{
					ObjectMeta: metav1.ObjectMeta{
						Name:      resourceName,
						Namespace: "default",
					},
					Spec: *config.MockGPUPoolSpec,
				}
				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}

			node := &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: nodeNamespacedName.Name,
					Labels: map[string]string{
						"mock-label": "true",
					},
				},
				Spec: corev1.NodeSpec{},
			}
			Expect(k8sClient.Create(ctx, node)).To(Succeed())
		})

		AfterEach(func() {
			resource := &tfv1.GPUPool{}
			err := k8sClient.Get(ctx, typeNamespacedName, resource)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the specific resource instance GPUPool")
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())

			Expect(k8sClient.Delete(ctx, &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: nodeNamespacedName.Name,
				},
			})).To(Succeed())

		})

		It("should successfully reconcile the resource", func() {
			By("Reconciling the created resource")
			controllerReconciler := &GPUPoolReconciler{
				Client: k8sClient,
				Scheme: k8sClient.Scheme(),
			}

			_, err := controllerReconciler.Reconcile(ctx, reconcile.Request{
				NamespacedName: typeNamespacedName,
			})
			Expect(err).NotTo(HaveOccurred())

		})
	})
})
