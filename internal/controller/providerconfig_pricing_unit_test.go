package controller

import (
	"context"
	"errors"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/pricing"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

// The instance pricing tables are derived from ProviderConfig hardware
// metadata. Reconciling a ProviderConfig that is created, corrected or deleted
// after start-up must rebuild them; before, they kept the start-up content
// until the operator was restarted.
func TestProviderConfigReconcileReloadsPricingTables(t *testing.T) {
	const (
		pcName       = "pricing-reload"
		instanceType = "g6.xlarge" // 1x L4 in the embedded instance data
		region       = "us-east-2"
	)
	ctx := context.Background()
	// The pricing tables are process-wide; leave them as the envtest suite
	// of this package loads them.
	t.Cleanup(func() { pricing.SetTflopsMapAndInitGPUPricingInfo(ctx, config.MockGpuInfo()) })

	scheme := runtime.NewScheme()
	if err := tfv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	reconciler := &ProviderConfigReconciler{
		Client:          k8sClient,
		Scheme:          scheme,
		ProviderManager: provider.NewManager(k8sClient),
	}
	reconcileOnce := func() {
		t.Helper()
		if _, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: types.NamespacedName{Name: pcName}}); err != nil {
			t.Fatal(err)
		}
	}
	pricingProvider := pricing.NewStaticPricingProvider()
	expectNoInstanceType := func(when string) {
		t.Helper()
		_, err := pricingProvider.LookupPricing(instanceType, tfv1.CapacityTypeOnDemand, region)
		if !errors.Is(err, pricing.ErrInstanceTypeNotFound) {
			t.Fatalf("%s: expected ErrInstanceTypeNotFound, got %v", when, err)
		}
	}
	expectTflops := func(when string, want float64) {
		t.Helper()
		price, err := pricingProvider.LookupPricing(instanceType, tfv1.CapacityTypeOnDemand, region)
		if err != nil || price <= 0 {
			t.Fatalf("%s: expected a price, got %v (err: %v)", when, price, err)
		}
		info, err := pricingProvider.LookupGPUNodeInstanceTypeInfo(instanceType, region)
		if err != nil {
			t.Fatalf("%s: %v", when, err)
		}
		if info.FP16TFlopsPerGPU != want {
			t.Fatalf("%s: expected %v TFlops per GPU, got %v", when, want, info.FP16TFlopsPerGPU)
		}
	}

	// No ProviderConfig exists: the tables are loaded, but empty.
	reconcileOnce()
	expectNoInstanceType("without ProviderConfig")

	providerConfig := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{Name: pcName},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: "NVIDIA",
			HardwareMetadata: []tfv1.HardwareModelInfo{{
				Model:         "L4",
				FullModelName: "NVIDIA L4",
				Fp16TFlops:    resource.MustParse("121"),
			}},
		},
	}
	if err := k8sClient.Create(ctx, providerConfig); err != nil {
		t.Fatal(err)
	}
	reconcileOnce()
	expectTflops("after the ProviderConfig was created", 121)

	if err := k8sClient.Get(ctx, client.ObjectKey{Name: pcName}, providerConfig); err != nil {
		t.Fatal(err)
	}
	providerConfig.Spec.HardwareMetadata[0].Fp16TFlops = resource.MustParse("242")
	if err := k8sClient.Update(ctx, providerConfig); err != nil {
		t.Fatal(err)
	}
	reconcileOnce()
	expectTflops("after the TFlops value was corrected", 242)

	if err := k8sClient.Delete(ctx, providerConfig); err != nil {
		t.Fatal(err)
	}
	reconcileOnce()
	expectNoInstanceType("after the ProviderConfig was deleted")
}
