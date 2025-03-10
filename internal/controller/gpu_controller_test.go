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
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

var _ = Describe("GPU Controller", func() {
	Context("When reconciling a resource", func() {
		const resourceName = "test-resource"

		ctx := context.Background()

		typeNamespacedName := types.NamespacedName{
			Name:      resourceName,
			Namespace: "default",
		}
		gpu := &tfv1.GPU{}
		gpunode := &tfv1.GPUNode{}

		BeforeEach(func() {
			By("creating the custom resource for the Kind GPUNode")
			gpunode = &tfv1.GPUNode{
				ObjectMeta: metav1.ObjectMeta{
					Name: resourceName + "-node",
					Labels: map[string]string{
						fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, "mock"): "true",
					},
				},
			}
			Expect(k8sClient.Create(ctx, gpunode)).To(Succeed())

			By("creating the custom resource for the Kind GPU")
			err := k8sClient.Get(ctx, typeNamespacedName, gpu)
			if err != nil && errors.IsNotFound(err) {
				resource := &tfv1.GPU{
					ObjectMeta: metav1.ObjectMeta{
						Name:      resourceName,
						Namespace: "default",
					},
				}
				Expect(controllerutil.SetControllerReference(gpunode, resource, scheme.Scheme)).To(Succeed())
				Expect(k8sClient.Create(ctx, resource)).To(Succeed())
			}
		})

		AfterEach(func() {
			resource := &tfv1.GPU{}
			err := k8sClient.Get(ctx, typeNamespacedName, resource)
			Expect(err).NotTo(HaveOccurred())

			By("Cleanup the specific resource instance GPU")
			Expect(k8sClient.Delete(ctx, resource)).To(Succeed())

			By("Cleanup the specific resource instance GPUNode")
			Expect(k8sClient.Delete(ctx, gpunode)).To(Succeed())
		})
		It("should successfully reconcile the resource", func() {
			By("Reconciling the created resource")
			controllerReconciler := &GPUReconciler{
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
