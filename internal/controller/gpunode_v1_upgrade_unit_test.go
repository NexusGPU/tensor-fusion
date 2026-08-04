package controller

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestV1HypervisorPodIsReplacedInPlaceByV2(t *testing.T) {
	ctx := context.Background()
	scheme := runtime.NewScheme()
	if err := tfv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	const (
		vendor   = "UpgradeTestVendor"
		oldImage = "hypervisor:v1"
		newImage = "hypervisor:v2"
		nodeName = "upgrade-test-node"
		poolName = "upgrade-test-pool"
	)
	poolSpec := config.MockGPUPoolSpec.DeepCopy()
	poolSpec.NodeManagerConfig.DefaultVendor = vendor
	poolSpec.NodeManagerConfig.NodeSelector = nil
	poolSpec.NodeManagerConfig.MultiVendorNodeSelector = nil
	poolSpec.ComponentConfig.Hypervisor.Image = newImage
	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: poolName},
		Spec:       *poolSpec,
	}
	v1Pool := pool.DeepCopy()
	v1Pool.Spec.ComponentConfig.Hypervisor.Image = oldImage
	oldHash := utils.HypervisorTemplateHash(v1Pool)
	if err := utils.SetProviderConfigRevision(pool, vendor, "v2-provider-revision"); err != nil {
		t.Fatal(err)
	}
	newHash := utils.HypervisorPodTemplateHash(pool, vendor)
	if oldHash == newHash {
		t.Fatal("v2 ProviderConfig revision must change the desired hypervisor hash")
	}

	poolLabel := constants.GPUNodePoolIdentifierLabelPrefix + poolName
	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
			UID:  types.UID("gpu-node-uid"),
			Labels: map[string]string{
				poolLabel:                        "true",
				constants.AcceleratorLabelVendor: vendor,
			},
		},
		Status: tfv1.GPUNodeStatus{Phase: tfv1.TensorFusionGPUNodePhasePending},
	}
	k8sNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: nodeName, UID: types.UID("core-node-uid")},
		Status: corev1.NodeStatus{Conditions: []corev1.NodeCondition{{
			Type: corev1.NodeReady, Status: corev1.ConditionTrue,
		}}},
	}
	providerConfig := &tfv1.ProviderConfig{
		ObjectMeta: metav1.ObjectMeta{Name: "upgrade-test-provider"},
		Spec: tfv1.ProviderConfigSpec{
			Vendor: vendor,
			Images: tfv1.ProviderImages{Middleware: newImage},
		},
	}
	podKey := client.ObjectKey{Name: utils.BuildHypervisorPodName(nodeName), Namespace: utils.CurrentNamespace()}
	v1Pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podKey.Name,
			Namespace: podKey.Namespace,
			Labels: map[string]string{
				constants.LabelComponent:          constants.ComponentHypervisor,
				constants.LabelKeyPodTemplateHash: oldHash,
			},
		},
		Spec: corev1.PodSpec{
			NodeName:   nodeName,
			Containers: []corev1.Container{{Name: constants.TFContainerNameHypervisor, Image: oldImage}},
		},
	}

	k8sClient := fake.NewClientBuilder().WithScheme(scheme).
		WithObjects(pool, gpuNode, k8sNode, providerConfig, v1Pod).Build()
	providerManager := provider.NewManager(k8sClient)
	providerManager.UpdateProvider(providerConfig)
	previousProviderManager := provider.GetManager()
	provider.SetGlobalManagerForTesting(providerManager)
	t.Cleanup(func() { provider.SetGlobalManagerForTesting(previousProviderManager) })

	reconciler := &GPUNodeReconciler{Client: k8sClient, Scheme: scheme}
	name, err := reconciler.reconcileHypervisorPod(ctx, gpuNode, pool, k8sNode)
	if err != nil {
		t.Fatal(err)
	}
	if name != "" {
		t.Fatalf("expected deletion round to return an empty pod name, got %q", name)
	}
	if err := k8sClient.Get(ctx, podKey, &corev1.Pod{}); !apierrors.IsNotFound(err) {
		t.Fatalf("expected v1 hypervisor pod to be deleted before replacement, got %v", err)
	}

	name, err = reconciler.reconcileHypervisorPod(ctx, gpuNode, pool, k8sNode)
	if err != nil {
		t.Fatal(err)
	}
	if name != podKey.Name {
		t.Fatalf("expected v2 replacement to keep name %q, got %q", podKey.Name, name)
	}
	v2Pod := &corev1.Pod{}
	if err := k8sClient.Get(ctx, podKey, v2Pod); err != nil {
		t.Fatal(err)
	}
	if got := v2Pod.Spec.Containers[0].Image; got != newImage {
		t.Fatalf("expected v2 hypervisor image %q, got %q", newImage, got)
	}
	if got := v2Pod.Labels[constants.LabelKeyPodTemplateHash]; got != newHash {
		t.Fatalf("expected v2 hypervisor hash %q, got %q", newHash, got)
	}
	pods := &corev1.PodList{}
	if err := k8sClient.List(ctx, pods, client.InNamespace(podKey.Namespace),
		client.MatchingLabels{constants.LabelComponent: constants.ComponentHypervisor}); err != nil {
		t.Fatal(err)
	}
	if len(pods.Items) != 1 {
		t.Fatalf("expected exactly one hypervisor pod after upgrade, got %d", len(pods.Items))
	}
}
