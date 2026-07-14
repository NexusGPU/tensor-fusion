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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// TestNodeReconcileInitializesGPUNodePhaseToPending covers the real new-node path:
// when the NodeReconciler creates a GPUNode it must initialize the phase to Pending
// so the inflight window never exposes an empty phase to monitoring.
func TestNodeReconcileInitializesGPUNodePhaseToPending(t *testing.T) {
	ctx := context.Background()

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node-1",
			Labels: map[string]string{"gpu": "true"},
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

	scheme := runtime.NewScheme()
	if err := tfv1.AddToScheme(scheme); err != nil {
		t.Fatalf("add TensorFusion scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core/v1 scheme: %v", err)
	}

	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPUNode{}).
		WithObjects(node, pool).
		Build()

	reconciler := &NodeReconciler{
		Client: kubeClient,
		Scheme: scheme,
	}

	if _, err := reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Name: node.Name}}); err != nil {
		t.Fatalf("Reconcile: %v", err)
	}

	gpuNode := &tfv1.GPUNode{}
	if err := kubeClient.Get(ctx, types.NamespacedName{Name: node.Name}, gpuNode); err != nil {
		t.Fatalf("get created GPUNode: %v", err)
	}
	if gpuNode.Status.Phase != tfv1.TensorFusionGPUNodePhasePending {
		t.Fatalf("expected GPUNode phase %q, got %q", tfv1.TensorFusionGPUNodePhasePending, gpuNode.Status.Phase)
	}
}
