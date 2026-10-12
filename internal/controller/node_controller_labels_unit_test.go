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
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	capacityTypeLabel = "karpenter.sh/capacity-type"
	instanceTypeLabel = "node.kubernetes.io/instance-type"
)

// A GPUNode says how its node is bought and what instance type it is, copied
// from the Node, and follows the Node when that changes.
func TestSyncGPUNodeLabelsFromNodeMirrorsCapacityAndInstanceType(t *testing.T) {
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1", Labels: map[string]string{
		capacityTypeLabel:                "spot",
		instanceTypeLabel:                "g5.2xlarge",
		constants.AcceleratorLabelVendor: "NVIDIA",
		"unrelated":                      "not-copied",
	}}}
	gpuNode := &tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{Name: "node-1", Labels: map[string]string{
		constants.LabelKeyOwner: "pool-1",
	}}}

	require.True(t, syncGPUNodeLabelsFromNode(node, gpuNode))
	require.Equal(t, map[string]string{
		constants.LabelKeyOwner:          "pool-1",
		capacityTypeLabel:                "spot",
		instanceTypeLabel:                "g5.2xlarge",
		constants.AcceleratorLabelVendor: "NVIDIA",
	}, gpuNode.Labels)
	require.False(t, syncGPUNodeLabelsFromNode(node, gpuNode), "nothing to do when already in sync")

	node.Labels[capacityTypeLabel] = "on-demand"
	require.True(t, syncGPUNodeLabelsFromNode(node, gpuNode))
	require.Equal(t, "on-demand", gpuNode.Labels[capacityTypeLabel])

	delete(node.Labels, capacityTypeLabel)
	delete(node.Labels, instanceTypeLabel)
	require.True(t, syncGPUNodeLabelsFromNode(node, gpuNode))
	require.NotContains(t, gpuNode.Labels, capacityTypeLabel)
	require.NotContains(t, gpuNode.Labels, instanceTypeLabel)
	require.Equal(t, "pool-1", gpuNode.Labels[constants.LabelKeyOwner])
}

func TestNodeReconcileCopiesCapacityAndInstanceTypeToGPUNode(t *testing.T) {
	ctx := context.Background()

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-1",
			Labels: map[string]string{
				"gpu":             "true",
				capacityTypeLabel: "spot",
				instanceTypeLabel: "g5.2xlarge",
			},
		},
	}
	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: "pool-1"},
		Spec: tfv1.GPUPoolSpec{
			NodeManagerConfig: &tfv1.NodeManagerConfig{
				ProvisioningMode: tfv1.ProvisioningModeAutoSelect,
				NodeSelector: &corev1.NodeSelector{
					NodeSelectorTerms: []corev1.NodeSelectorTerm{
						{
							MatchExpressions: []corev1.NodeSelectorRequirement{
								{Key: "gpu", Operator: corev1.NodeSelectorOpIn, Values: []string{"true"}},
							},
						},
					},
				},
			},
		},
	}
	// A GPUNode created before the labels were mirrored.
	existing := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-2",
			Labels: map[string]string{
				"gpu":             "true",
				capacityTypeLabel: "on-demand",
				instanceTypeLabel: "p4d.24xlarge",
			},
		},
	}
	existingGPUNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{Name: "node-2", Labels: map[string]string{constants.LabelKeyOwner: "pool-1"}},
	}

	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPUNode{}).
		WithObjects(node, existing, existingGPUNode, pool).
		Build()
	reconciler := &NodeReconciler{Client: kubeClient, Scheme: scheme}

	for _, name := range []string{"node-1", "node-2"} {
		_, err := reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Name: name}})
		require.NoError(t, err)
	}

	created := &tfv1.GPUNode{}
	require.NoError(t, kubeClient.Get(ctx, types.NamespacedName{Name: "node-1"}, created))
	require.Equal(t, "spot", created.Labels[capacityTypeLabel])
	require.Equal(t, "g5.2xlarge", created.Labels[instanceTypeLabel])
	require.Equal(t, "pool-1", created.Labels[constants.LabelKeyOwner])

	updated := &tfv1.GPUNode{}
	require.NoError(t, kubeClient.Get(ctx, types.NamespacedName{Name: "node-2"}, updated))
	require.Equal(t, "on-demand", updated.Labels[capacityTypeLabel])
	require.Equal(t, "p4d.24xlarge", updated.Labels[instanceTypeLabel])

	// The node is rebought as spot: the GPUNode follows.
	latest := &corev1.Node{}
	require.NoError(t, kubeClient.Get(ctx, types.NamespacedName{Name: "node-2"}, latest))
	latest.Labels[capacityTypeLabel] = "spot"
	require.NoError(t, kubeClient.Update(ctx, latest))
	_, err := reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Name: "node-2"}})
	require.NoError(t, err)
	require.NoError(t, kubeClient.Get(ctx, types.NamespacedName{Name: "node-2"}, updated))
	require.Equal(t, "spot", updated.Labels[capacityTypeLabel])
}
