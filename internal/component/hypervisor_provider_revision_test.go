package component

import (
	"context"
	"fmt"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestHypervisorProviderRevisionOnlySelectsMatchingVendorNodes(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := tfv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: "mixed-pool"},
		Spec: tfv1.GPUPoolSpec{
			ComponentConfig: &tfv1.ComponentConfig{Hypervisor: &tfv1.HypervisorConfig{}},
		},
	}
	baseHash := utils.HypervisorTemplateHash(pool)
	if err := utils.SetProviderConfigRevision(pool, "NVIDIA", "nvidia-revision"); err != nil {
		t.Fatal(err)
	}

	poolLabel := fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name)
	nvidiaNode := &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{
		Name: "nvidia-node",
		Labels: map[string]string{
			poolLabel:                        "true",
			constants.AcceleratorLabelVendor: "NVIDIA",
		},
	}}
	amdNode := &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{
		Name: "amd-node",
		Labels: map[string]string{
			poolLabel:                        "true",
			constants.AcceleratorLabelVendor: "AMD",
		},
	}}
	nvidiaPod := hypervisorHashTestPod(nvidiaNode.Name, baseHash)
	amdPod := hypervisorHashTestPod(amdNode.Name, baseHash)

	client := fake.NewClientBuilder().WithScheme(scheme).
		WithObjects(nvidiaNode, amdNode, nvidiaPod, amdPod).Build()
	h := &Hypervisor{}
	total, updated, recheck, err := h.GetResourcesInfo(
		client, context.Background(), pool, utils.HypervisorTemplateHash(pool),
	)
	if err != nil {
		t.Fatal(err)
	}
	if recheck || total != 2 || updated != 1 {
		t.Fatalf("expected exactly one of two vendor nodes to need update, total=%d updated=%d recheck=%v",
			total, updated, recheck)
	}
	if len(h.nodesToUpdate) != 1 || h.nodesToUpdate[0].Name != nvidiaNode.Name {
		t.Fatalf("expected only NVIDIA node to be selected, got %#v", h.nodesToUpdate)
	}
}

func hypervisorHashTestPod(nodeName, hash string) *corev1.Pod {
	return &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
		Name:      utils.BuildHypervisorPodName(nodeName),
		Namespace: utils.CurrentNamespace(),
		Labels:    map[string]string{constants.LabelKeyPodTemplateHash: hash},
	}}
}

func TestHypervisorProviderRevisionUsesConfiguredBatchSize(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := tfv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: "nvidia-pool"},
		Spec: tfv1.GPUPoolSpec{
			NodeManagerConfig: &tfv1.NodeManagerConfig{
				DefaultVendor: "NVIDIA",
				NodePoolRollingUpdatePolicy: &tfv1.NodeRollingUpdatePolicy{
					AutoUpdateHypervisor: true,
					BatchPercentage:      50,
					BatchInterval:        "0s",
				},
			},
			ComponentConfig: &tfv1.ComponentConfig{Hypervisor: &tfv1.HypervisorConfig{}},
		},
	}
	oldHash := utils.HypervisorTemplateHash(pool)
	pool.Status.ComponentStatus.HypervisorVersion = oldHash
	if err := utils.SetProviderConfigRevision(pool, "NVIDIA", "revision-1"); err != nil {
		t.Fatal(err)
	}

	poolLabel := fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name)
	objects := []runtime.Object{pool}
	for i := 0; i < 4; i++ {
		nodeName := fmt.Sprintf("nvidia-node-%d", i)
		objects = append(objects,
			&tfv1.GPUNode{
				ObjectMeta: metav1.ObjectMeta{
					Name:              nodeName,
					CreationTimestamp: metav1.Unix(int64(i), 0),
					Labels: map[string]string{
						poolLabel:                        "true",
						constants.AcceleratorLabelVendor: "NVIDIA",
					},
				},
				Status: tfv1.GPUNodeStatus{Phase: tfv1.TensorFusionGPUNodePhaseRunning},
			},
			hypervisorHashTestPod(nodeName, oldHash),
		)
	}

	client := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPUPool{}, &tfv1.GPUNode{}).
		WithRuntimeObjects(objects...).Build()
	if _, err := ManageUpdate(context.Background(), client, pool, &Hypervisor{}); err != nil {
		t.Fatal(err)
	}

	nodes := &tfv1.GPUNodeList{}
	if err := client.List(context.Background(), nodes); err != nil {
		t.Fatal(err)
	}
	pending := 0
	for i := range nodes.Items {
		if nodes.Items[i].Status.Phase == tfv1.TensorFusionGPUNodePhasePending {
			pending++
		}
	}
	if pending != 2 {
		t.Fatalf("expected first 50%% batch to mark two of four nodes Pending, got %d", pending)
	}
}
