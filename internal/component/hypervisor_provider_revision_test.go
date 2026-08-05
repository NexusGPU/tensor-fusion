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
	"sigs.k8s.io/controller-runtime/pkg/client"
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
	baseNVIDIAHash := utils.HypervisorPodTemplateHash(pool, "NVIDIA", tfv1.IsolationModeSoft)
	baseAMDHash := utils.HypervisorPodTemplateHash(pool, "AMD", tfv1.IsolationModeSoft)
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
	nvidiaPod := hypervisorHashTestPod(nvidiaNode.Name, baseNVIDIAHash)
	amdPod := hypervisorHashTestPod(amdNode.Name, baseAMDHash)

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

func TestHypervisorIsolationPolicyOnlySelectsNodesWhoseModeChanges(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := tfv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: "isolation-pool"},
		Spec: tfv1.GPUPoolSpec{
			NodeManagerConfig: &tfv1.NodeManagerConfig{DefaultIsolationMode: tfv1.IsolationModeSoft},
			ComponentConfig:   &tfv1.ComponentConfig{Hypervisor: &tfv1.HypervisorConfig{}},
		},
	}
	oldPodHash := utils.HypervisorPodTemplateHash(pool, constants.AcceleratorVendorNvidia, tfv1.IsolationModeSoft)
	pool.Spec.NodeManagerConfig.IsolationModeRules = []tfv1.NodeIsolationModeRule{{
		Mode: tfv1.IsolationModePartitioned,
		Selector: metav1.LabelSelector{MatchExpressions: []metav1.LabelSelectorRequirement{{
			Key:      corev1.LabelInstanceTypeStable,
			Operator: metav1.LabelSelectorOpIn,
			Values:   []string{"p4d.24xlarge", "p5.48xlarge"},
		}}},
	}}

	poolLabel := fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name)
	matchedGPUNode := &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{
		Name:   "partitioned-node",
		Labels: map[string]string{poolLabel: "true"},
	}}
	softGPUNode := &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{
		Name:   "soft-node",
		Labels: map[string]string{poolLabel: "true"},
	}}
	matchedNode := &corev1.Node{ObjectMeta: metav1.ObjectMeta{
		Name: matchedGPUNode.Name,
		Labels: map[string]string{
			corev1.LabelInstanceTypeStable: "p5.48xlarge",
		},
	}}
	softNode := &corev1.Node{ObjectMeta: metav1.ObjectMeta{
		Name: softGPUNode.Name,
		Labels: map[string]string{
			corev1.LabelInstanceTypeStable: "g5.48xlarge",
		},
	}}

	client := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		matchedGPUNode,
		softGPUNode,
		matchedNode,
		softNode,
		hypervisorHashTestPod(matchedGPUNode.Name, oldPodHash),
		hypervisorHashTestPod(softGPUNode.Name, oldPodHash),
	).Build()
	h := &Hypervisor{}
	total, updated, recheck, err := h.GetResourcesInfo(
		client, context.Background(), pool, utils.HypervisorTemplateHash(pool),
	)
	if err != nil {
		t.Fatal(err)
	}
	if recheck || total != 2 || updated != 1 {
		t.Fatalf("expected one affected node, total=%d updated=%d recheck=%v", total, updated, recheck)
	}
	if len(h.nodesToUpdate) != 1 || h.nodesToUpdate[0].Name != matchedGPUNode.Name {
		t.Fatalf("expected only selector-matched node, got %#v", h.nodesToUpdate)
	}
}

func TestHypervisorIsolationPolicyResumesWhenAutoUpdateEnabled(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := tfv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}

	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: "manual-isolation-pool"},
		Spec: tfv1.GPUPoolSpec{
			NodeManagerConfig: &tfv1.NodeManagerConfig{
				DefaultIsolationMode: tfv1.IsolationModeSoft,
				NodePoolRollingUpdatePolicy: &tfv1.NodeRollingUpdatePolicy{
					AutoUpdateHypervisor: false,
					BatchPercentage:      100,
				},
			},
			ComponentConfig: &tfv1.ComponentConfig{Hypervisor: &tfv1.HypervisorConfig{}},
		},
	}
	pool.Status.ComponentStatus.HypervisorVersion = utils.HypervisorTemplateHash(pool)
	pool.Status.ComponentStatus.HypervisorConfigSynced = true
	oldPodHash := utils.HypervisorPodTemplateHash(pool, constants.AcceleratorVendorNvidia, tfv1.IsolationModeSoft)
	pool.Spec.NodeManagerConfig.DefaultIsolationMode = tfv1.IsolationModeHard
	poolLabel := fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name)
	node := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{Name: "manual-node", Labels: map[string]string{poolLabel: "true"}},
		Status:     tfv1.GPUNodeStatus{Phase: tfv1.TensorFusionGPUNodePhaseRunning},
	}
	k8sNode := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: node.Name}}
	pod := hypervisorHashTestPod(node.Name, oldPodHash)

	kubeClient := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPUPool{}, &tfv1.GPUNode{}).
		WithObjects(pool, node, k8sNode, pod).Build()
	if _, err := ManageUpdate(context.Background(), kubeClient, pool, &Hypervisor{}); err != nil {
		t.Fatal(err)
	}

	updated := &tfv1.GPUPool{}
	if err := kubeClient.Get(context.Background(), client.ObjectKey{Name: pool.Name}, updated); err != nil {
		t.Fatal(err)
	}
	if updated.Status.ComponentStatus.HypervisorConfigSynced {
		t.Fatal("policy must remain unsynced while autoUpdateHypervisor is disabled")
	}
	if updated.Annotations[HypervisorUpdateInProgressAnnotation] != "" {
		t.Fatal("disabled auto update must not start a rollout campaign")
	}

	updated.Spec.NodeManagerConfig.NodePoolRollingUpdatePolicy.AutoUpdateHypervisor = true
	if err := kubeClient.Update(context.Background(), updated); err != nil {
		t.Fatal(err)
	}
	if _, err := ManageUpdate(context.Background(), kubeClient, updated, &Hypervisor{}); err != nil {
		t.Fatal(err)
	}
	resumed := &tfv1.GPUPool{}
	if err := kubeClient.Get(context.Background(), client.ObjectKey{Name: pool.Name}, resumed); err != nil {
		t.Fatal(err)
	}
	wantHash := utils.HypervisorTemplateHash(resumed)
	if resumed.Annotations[HypervisorUpdateInProgressAnnotation] != wantHash {
		t.Fatalf("expected rollout %q after enabling auto update, got %q", wantHash, resumed.Annotations[HypervisorUpdateInProgressAnnotation])
	}
	selectedNode := &tfv1.GPUNode{}
	if err := kubeClient.Get(context.Background(), client.ObjectKey{Name: node.Name}, selectedNode); err != nil {
		t.Fatal(err)
	}
	if selectedNode.Status.Phase != tfv1.TensorFusionGPUNodePhasePending {
		t.Fatalf("expected resumed rollout to select node, got phase %q", selectedNode.Status.Phase)
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
	oldCampaignHash := utils.HypervisorTemplateHash(pool)
	oldPodHash := utils.HypervisorPodTemplateHash(pool, "NVIDIA", tfv1.IsolationModeSoft)
	pool.Status.ComponentStatus.HypervisorVersion = oldCampaignHash
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
			hypervisorHashTestPod(nodeName, oldPodHash),
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
