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
	"encoding/json"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestGPUNodeReconcileNodeDiscoveryJobIgnoresFailedAttemptsWithoutFailedCondition(t *testing.T) {
	t.Helper()

	ctx := context.Background()
	gpuNode := newNodeDiscoveryTestGPUNode()
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      getDiscoveryJobName(gpuNode.Name),
			Namespace: utils.CurrentNamespace(),
		},
		Status: batchv1.JobStatus{
			Failed: 1,
		},
	}

	reconciler, kubeClient := newNodeDiscoveryTestReconciler(t, gpuNode, job)
	currentNode := &tfv1.GPUNode{}
	if err := kubeClient.Get(ctx, types.NamespacedName{Name: gpuNode.Name}, currentNode); err != nil {
		t.Fatalf("get GPUNode: %v", err)
	}

	if err := reconciler.reconcileNodeDiscoveryJob(ctx, currentNode, newNodeDiscoveryTestPool(t)); err != nil {
		t.Fatalf("reconcileNodeDiscoveryJob: %v", err)
	}

	updatedNode := &tfv1.GPUNode{}
	if err := kubeClient.Get(ctx, types.NamespacedName{Name: gpuNode.Name}, updatedNode); err != nil {
		t.Fatalf("get updated GPUNode: %v", err)
	}

	if updatedNode.Status.Phase != tfv1.TensorFusionGPUNodePhasePending {
		t.Fatalf("expected GPUNode phase %q, got %q", tfv1.TensorFusionGPUNodePhasePending, updatedNode.Status.Phase)
	}
}

func TestGPUNodeReconcileNodeDiscoveryJobMarksNodeFailedOnFailedCondition(t *testing.T) {
	t.Helper()

	ctx := context.Background()
	gpuNode := newNodeDiscoveryTestGPUNode()
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      getDiscoveryJobName(gpuNode.Name),
			Namespace: utils.CurrentNamespace(),
		},
		Status: batchv1.JobStatus{
			Failed: 1,
			Conditions: []batchv1.JobCondition{
				{
					Type:   batchv1.JobFailed,
					Status: corev1.ConditionTrue,
				},
			},
		},
	}

	reconciler, kubeClient := newNodeDiscoveryTestReconciler(t, gpuNode, job)
	currentNode := &tfv1.GPUNode{}
	if err := kubeClient.Get(ctx, types.NamespacedName{Name: gpuNode.Name}, currentNode); err != nil {
		t.Fatalf("get GPUNode: %v", err)
	}

	if err := reconciler.reconcileNodeDiscoveryJob(ctx, currentNode, newNodeDiscoveryTestPool(t)); err != nil {
		t.Fatalf("reconcileNodeDiscoveryJob: %v", err)
	}

	updatedNode := &tfv1.GPUNode{}
	if err := kubeClient.Get(ctx, types.NamespacedName{Name: gpuNode.Name}, updatedNode); err != nil {
		t.Fatalf("get updated GPUNode: %v", err)
	}

	if updatedNode.Status.Phase != tfv1.TensorFusionGPUNodePhaseFailed {
		t.Fatalf("expected GPUNode phase %q, got %q", tfv1.TensorFusionGPUNodePhaseFailed, updatedNode.Status.Phase)
	}
}

func newNodeDiscoveryTestReconciler(t *testing.T, objects ...ctrlclient.Object) (*GPUNodeReconciler, ctrlclient.Client) {
	t.Helper()

	scheme := runtime.NewScheme()
	if err := tfv1.AddToScheme(scheme); err != nil {
		t.Fatalf("add TensorFusion scheme: %v", err)
	}
	if err := batchv1.AddToScheme(scheme); err != nil {
		t.Fatalf("add batch/v1 scheme: %v", err)
	}
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("add core/v1 scheme: %v", err)
	}

	kubeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&tfv1.GPUNode{}, &tfv1.GPU{}).
		WithObjects(objects...).
		Build()

	return &GPUNodeReconciler{
		Client: kubeClient,
		Scheme: scheme,
	}, kubeClient
}

func newNodeDiscoveryTestPool(t *testing.T) *tfv1.GPUPool {
	t.Helper()

	podTemplateRaw, err := json.Marshal(corev1.PodTemplate{
		Template: corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: "node-discovery"},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("marshal node discovery pod template: %v", err)
	}

	return &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pool-1",
		},
		Spec: tfv1.GPUPoolSpec{
			ComponentConfig: &tfv1.ComponentConfig{
				NodeDiscovery: &tfv1.NodeDiscoveryConfig{
					Image: "node-discovery:latest",
					PodTemplate: &runtime.RawExtension{
						Raw: podTemplateRaw,
					},
				},
			},
		},
	}
}

func newNodeDiscoveryTestGPUNode() *tfv1.GPUNode {
	return &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node-1",
		},
		Status: tfv1.GPUNodeStatus{
			Phase: tfv1.TensorFusionGPUNodePhasePending,
		},
	}
}
