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
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

var _ = Describe("ProviderConfig controller", func() {
	// Use a unique vendor + node so the reconcile only ever touches this test's
	// hypervisor pod, even when specs run in parallel.
	const vendor = "ProviderConfigTestVendor"

	var (
		reconciler *ProviderConfigReconciler
		pcName     = "test-provider-config"
		nodeName   = "providerconfig-test-node"
		podName    = utils.BuildHypervisorPodName(nodeName)
	)

	reconcileOnce := func() {
		_, err := reconciler.Reconcile(ctx, reconcile.Request{
			NamespacedName: types.NamespacedName{Name: pcName},
		})
		Expect(err).NotTo(HaveOccurred())
	}

	hypervisorPodExists := func() bool {
		pod := &corev1.Pod{}
		err := k8sClient.Get(ctx, types.NamespacedName{
			Namespace: utils.CurrentNamespace(),
			Name:      podName,
		}, pod)
		if err == nil {
			return pod.DeletionTimestamp.IsZero()
		}
		Expect(apierrors.IsNotFound(err)).To(BeTrue())
		return false
	}

	BeforeEach(func() {
		reconciler = &ProviderConfigReconciler{
			Client:          k8sClient,
			Scheme:          k8sClient.Scheme(),
			ProviderManager: provider.NewManager(k8sClient),
		}

		Expect(k8sClient.Create(ctx, &tfv1.ProviderConfig{
			ObjectMeta: metav1.ObjectMeta{Name: pcName},
			Spec:       tfv1.ProviderConfigSpec{Vendor: vendor},
		})).To(Succeed())

		Expect(k8sClient.Create(ctx, &tfv1.GPUNode{
			ObjectMeta: metav1.ObjectMeta{
				Name:   nodeName,
				Labels: map[string]string{constants.AcceleratorLabelVendor: vendor},
			},
		})).To(Succeed())

		Expect(k8sClient.Create(ctx, &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: utils.CurrentNamespace(),
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "hypervisor", Image: "test"}},
			},
		})).To(Succeed())
	})

	AfterEach(func() {
		_ = k8sClient.Delete(ctx, &tfv1.ProviderConfig{ObjectMeta: metav1.ObjectMeta{Name: pcName}})
		_ = k8sClient.Delete(ctx, &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
		_ = k8sClient.Delete(ctx, &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
			Name: podName, Namespace: utils.CurrentNamespace(),
		}}, client.GracePeriodSeconds(0))
	})

	It("does not recreate hypervisor pods on reconcile when the spec is unchanged", func() {
		// First reconcile adopts the current state: it records the spec hash but
		// must NOT delete the already-running hypervisor pod.
		reconcileOnce()
		Expect(hypervisorPodExists()).To(BeTrue(), "pod should survive the initial adopt reconcile")

		pc := &tfv1.ProviderConfig{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: pcName}, pc)).To(Succeed())
		Expect(pc.Annotations).To(HaveKey(constants.ProviderConfigSpecHashAnnotation))

		// A second reconcile with the unchanged spec (mimicking an operator
		// restart / informer resync) must leave the pod alone.
		reconcileOnce()
		Expect(hypervisorPodExists()).To(BeTrue(), "pod should survive a no-op resync reconcile")
	})

	It("recreates hypervisor pods when the spec actually changes", func() {
		reconcileOnce() // adopt + record hash
		Expect(hypervisorPodExists()).To(BeTrue())

		pc := &tfv1.ProviderConfig{}
		Expect(k8sClient.Get(ctx, types.NamespacedName{Name: pcName}, pc)).To(Succeed())
		pc.Spec.InUseResourceNames = append(pc.Spec.InUseResourceNames, "changed-resource")
		Expect(k8sClient.Update(ctx, pc)).To(Succeed())

		reconcileOnce()
		Eventually(hypervisorPodExists).Should(BeFalse(), "pod should be deleted after a real spec change")
	})
})
