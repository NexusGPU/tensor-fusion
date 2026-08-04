package controller

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestPublishProviderRevisionMarksOnlyPoolsThatMayUseVendor(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := tfv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	mixedPool := providerRolloutTestPool("mixed-pool", map[string]*corev1.NodeSelector{
		"NVIDIA": {},
		"AMD":    {},
	})
	amdPool := providerRolloutTestPool("amd-pool", map[string]*corev1.NodeSelector{
		"AMD": {},
	})
	poolLabel := constants.GPUNodePoolIdentifierLabelPrefix + mixedPool.Name
	nvidiaNode := &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{
		Name: "nvidia-node",
		Labels: map[string]string{
			poolLabel:                        "true",
			constants.AcceleratorLabelVendor: "NVIDIA",
		},
	}}

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).
		WithObjects(mixedPool, amdPool, nvidiaNode).Build()
	reconciler := &ProviderConfigReconciler{
		Client:          k8sClient,
		Scheme:          scheme,
		ProviderManager: provider.NewManager(k8sClient),
	}
	if err := reconciler.publishProviderRevision(context.Background(), "NVIDIA", "revision-1"); err != nil {
		t.Fatal(err)
	}

	gotMixed := &tfv1.GPUPool{}
	if err := k8sClient.Get(context.Background(), client.ObjectKey{Name: mixedPool.Name}, gotMixed); err != nil {
		t.Fatal(err)
	}
	if got := utils.ProviderConfigRevision(gotMixed, "nvidia"); got != "revision-1" {
		t.Fatalf("expected mixed pool NVIDIA revision, got %q", got)
	}

	gotAMD := &tfv1.GPUPool{}
	if err := k8sClient.Get(context.Background(), client.ObjectKey{Name: amdPool.Name}, gotAMD); err != nil {
		t.Fatal(err)
	}
	if got := utils.ProviderConfigRevision(gotAMD, "NVIDIA"); got != "" {
		t.Fatalf("AMD-only pool must not receive NVIDIA revision, got %q", got)
	}
}

func providerRolloutTestPool(name string, vendors map[string]*corev1.NodeSelector) *tfv1.GPUPool {
	return &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: tfv1.GPUPoolSpec{
			NodeManagerConfig: &tfv1.NodeManagerConfig{MultiVendorNodeSelector: vendors},
			ComponentConfig:   &tfv1.ComponentConfig{Hypervisor: &tfv1.HypervisorConfig{}},
		},
	}
}
