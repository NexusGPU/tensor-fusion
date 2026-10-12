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
	"fmt"
	"strings"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const (
	provisionedTestPool      = "pool-karpenter"
	provisionedTestClaim     = "pool-karpenter-abcdefgh"
	provisionedTestSelectKey = "shared-pool"
)

// longTestNodeName is a legal node name that does not fit a label value.
var longTestNodeName = "kwok-" + provisionedTestClaim + "-" + strings.Repeat("n", 71)

func provisionedTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))
	require.NoError(t, corev1.AddToScheme(scheme))
	return scheme
}

// provisionedTestObjects is what exists when Karpenter has launched the node of
// a GPUNodeClaim: a pool that provisions its nodes and therefore has no node
// selector, a pool that selects nodes by label, the claim, and the node with
// the provisioner label Karpenter copied from the NodeClaim.
func provisionedTestObjects(mode tfv1.ProvisioningMode, nodeLabels map[string]string) []client.Object {
	labels := map[string]string{constants.ProvisionerLabelKey: provisionedTestClaim}
	for k, v := range nodeLabels {
		labels[k] = v
	}
	return []client.Object{
		&tfv1.GPUPool{
			ObjectMeta: metav1.ObjectMeta{Name: provisionedTestPool, UID: "pool-uid"},
			Spec: tfv1.GPUPoolSpec{NodeManagerConfig: &tfv1.NodeManagerConfig{
				ProvisioningMode: mode,
				NodeProvisioner:  &tfv1.NodeProvisioner{},
			}},
		},
		&tfv1.GPUPool{
			ObjectMeta: metav1.ObjectMeta{Name: "pool-select", UID: "select-uid"},
			Spec: tfv1.GPUPoolSpec{NodeManagerConfig: &tfv1.NodeManagerConfig{
				ProvisioningMode: tfv1.ProvisioningModeAutoSelect,
				NodeSelector: &corev1.NodeSelector{NodeSelectorTerms: []corev1.NodeSelectorTerm{{
					MatchExpressions: []corev1.NodeSelectorRequirement{{
						Key: provisionedTestSelectKey, Operator: corev1.NodeSelectorOpExists,
					}},
				}}},
			}},
		},
		&tfv1.GPUNodeClaim{ObjectMeta: metav1.ObjectMeta{
			Name:   provisionedTestClaim,
			UID:    "claim-uid",
			Labels: map[string]string{constants.LabelKeyOwner: provisionedTestPool},
		}},
		&corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: longTestNodeName, Labels: labels}},
	}
}

// A node launched for a GPUNodeClaim must get a GPUNode in the claim's pool
// although that pool has no node selector; without it no hypervisor starts and
// the claim never becomes Bound.
func TestNodeReconcileCreatesGPUNodeForProvisionedNode(t *testing.T) {
	for _, tc := range []struct {
		name       string
		mode       tfv1.ProvisioningMode
		nodeLabels map[string]string
	}{
		{"karpenter pool without node selector", tfv1.ProvisioningModeKarpenter, nil},
		{"provisioned pool without node selector", tfv1.ProvisioningModeProvisioned, nil},
		{"the claim's pool wins over a selector match", tfv1.ProvisioningModeKarpenter,
			map[string]string{provisionedTestSelectKey: constants.TrueStringValue}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			scheme := provisionedTestScheme(t)
			kubeClient := fake.NewClientBuilder().WithScheme(scheme).
				WithStatusSubresource(&tfv1.GPUNode{}).
				WithObjects(provisionedTestObjects(tc.mode, tc.nodeLabels)...).Build()
			reconciler := &NodeReconciler{Client: kubeClient, Scheme: scheme}

			req := ctrl.Request{NamespacedName: types.NamespacedName{Name: longTestNodeName}}
			_, err := reconciler.Reconcile(t.Context(), req)
			if apierrors.IsConflict(err) {
				// Provisioned mode writes the node twice in its first pass (selector
				// hash, then owner) and loses to itself; the requeue succeeds.
				_, err = reconciler.Reconcile(t.Context(), req)
			}
			require.NoError(t, err)

			gpuNode := &tfv1.GPUNode{}
			require.NoError(t, kubeClient.Get(t.Context(), types.NamespacedName{Name: longTestNodeName}, gpuNode),
				"a provisioned node must get a GPUNode")
			require.Equal(t, provisionedTestPool, utils.ExtractPoolNameFromNodeLabel(gpuNode))
			require.Equal(t, provisionedTestPool, gpuNode.Labels[constants.LabelKeyOwner])
			require.Equal(t, provisionedTestClaim, gpuNode.Labels[constants.ProvisionerLabelKey])
			require.Equal(t, tfv1.GPUNodeManageModeProvisioned, gpuNode.Spec.ManageMode)
			require.Equal(t, tfv1.TensorFusionGPUNodePhasePending, gpuNode.Status.Phase)
			for key, value := range gpuNode.Labels {
				require.Empty(t, validation.IsValidLabelValue(value), "label %s=%q", key, value)
			}
		})
	}
}

// Without its claim the provisioner label says nothing about a pool: the node
// is matched by selector only, as before.
func TestNodeReconcileProvisionerLabelWithoutClaim(t *testing.T) {
	scheme := provisionedTestScheme(t)
	var objects []client.Object
	for _, obj := range provisionedTestObjects(tfv1.ProvisioningModeKarpenter, nil) {
		if _, isClaim := obj.(*tfv1.GPUNodeClaim); !isClaim {
			objects = append(objects, obj)
		}
	}
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPUNode{}).WithObjects(objects...).Build()
	reconciler := &NodeReconciler{Client: kubeClient, Scheme: scheme}

	_, err := reconciler.Reconcile(t.Context(),
		ctrl.Request{NamespacedName: types.NamespacedName{Name: longTestNodeName}})
	require.NoError(t, err)
	err = kubeClient.Get(t.Context(), types.NamespacedName{Name: longTestNodeName}, &tfv1.GPUNode{})
	require.True(t, apierrors.IsNotFound(err), "no pool selects this node: %v", err)
}

// The GPUNode controller finds the GPUs of a node whose name does not fit a
// label value, and still those of a short-named node labelled the old way.
func TestFetchOwnedGPUsOfLongNamedNode(t *testing.T) {
	scheme := provisionedTestScheme(t)
	shortNode := "gpu-node-01"
	gpuOf := func(i int, node string, labels map[string]string) *tfv1.GPU {
		return &tfv1.GPU{ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("gpu-%d", i), Labels: labels,
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: tfv1.GroupVersion.String(), Kind: "GPUNode", Name: node, UID: types.UID(node),
			}},
		}}
	}
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		// as the hypervisor writes it now
		gpuOf(0, longTestNodeName, utils.GPUOwnerSelector(longTestNodeName)),
		// as every release so far has written it
		gpuOf(1, shortNode, map[string]string{constants.LabelKeyOwner: shortNode}),
	).Build()
	reconciler := &GPUNodeReconciler{Client: kubeClient, Scheme: scheme}

	for _, node := range []string{longTestNodeName, shortNode} {
		gpus, err := reconciler.fetchAllOwnedGPUDevices(t.Context(),
			&tfv1.GPUNode{ObjectMeta: metav1.ObjectMeta{Name: node}})
		require.NoError(t, err)
		require.Len(t, gpus, 1, "GPUs of node %s", node)
		require.Equal(t, node, utils.GPUNodeNameOfGPU(&gpus[0]))
	}
}
