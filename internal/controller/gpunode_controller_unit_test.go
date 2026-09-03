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
	"fmt"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestSetHypervisorIsolationEnv(t *testing.T) {
	spec := &corev1.PodSpec{Containers: []corev1.Container{{
		Command: []string{"bash", "-c", "exec /usr/local/bin/hypervisor daemon"},
		Args:    []string{"--custom-arg"},
		Env: []corev1.EnvVar{
			{Name: constants.TFIsolationModeEnv, ValueFrom: &corev1.EnvVarSource{}},
			{Name: constants.IsolationModePolicyEnv, Value: string(tfv1.IsolationModePolicyStatic)},
		},
	}}}

	setHypervisorIsolationEnv(spec, string(tfv1.IsolationModeSoft), string(tfv1.IsolationModePolicyDynamic))

	env := spec.Containers[0].Env
	if len(env) != 2 {
		t.Fatalf("expected two env vars, got %d", len(env))
	}
	if env[0].Value != string(tfv1.IsolationModeSoft) || env[0].ValueFrom != nil {
		t.Fatalf("isolation mode was not replaced: %+v", env[0])
	}
	if env[1].Value != string(tfv1.IsolationModePolicyDynamic) {
		t.Fatalf("isolation policy = %q, want %q", env[1].Value, tfv1.IsolationModePolicyDynamic)
	}
	if got := spec.Containers[0].Command; len(got) != 3 || got[2] != "exec /usr/local/bin/hypervisor daemon" {
		t.Fatalf("command was modified: %v", got)
	}
	if len(spec.Containers[0].Args) != 1 || spec.Containers[0].Args[0] != "--custom-arg" {
		t.Fatalf("args were modified: %v", spec.Containers[0].Args)
	}
}

func TestHypervisorIsolationConfigurationUsesEnvForShellCommand(t *testing.T) {
	pod := &corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{{
		Command: []string{"bash", "-c"},
		Args:    []string{"exec /usr/local/bin/hypervisor daemon", "--isolation-policy=dynamic"},
		Env: []corev1.EnvVar{
			{Name: constants.TFIsolationModeEnv, Value: "soft"},
			{Name: constants.IsolationModePolicyEnv, Value: "dynamic"},
		},
	}}}}

	if !isHypervisorIsolationModeConfigured(pod, string(tfv1.IsolationModeSoft)) {
		t.Fatal("expected shell command to use the isolation mode env")
	}
	if !isHypervisorIsolationPolicyConfigured(pod, string(tfv1.IsolationModePolicyDynamic)) {
		t.Fatal("expected shell command to use the isolation policy env")
	}

	pod.Spec.Containers[0].Env = nil
	if isHypervisorIsolationPolicyConfigured(pod, string(tfv1.IsolationModePolicyDynamic)) {
		t.Fatal("shell command positional args must not be treated as Hypervisor flags")
	}
}

func TestHypervisorIsolationConfigurationUsesArgsForDirectCommand(t *testing.T) {
	pod := &corev1.Pod{Spec: corev1.PodSpec{Containers: []corev1.Container{{
		Args: []string{"--isolation-mode=soft", "--isolation-policy=dynamic"},
	}}}}

	if !isHypervisorIsolationModeConfigured(pod, string(tfv1.IsolationModeSoft)) {
		t.Fatal("expected direct command isolation mode arg")
	}
	if !isHypervisorIsolationPolicyConfigured(pod, string(tfv1.IsolationModePolicyDynamic)) {
		t.Fatal("expected direct command isolation policy arg")
	}
}

// TestGPUNodeReconcileInitializesEmptyPhaseToPending covers the safety net for
// pre-existing GPUNodes with an empty phase: the reconciler must set it to
// Pending before any early-return gate (here the driver-upgrade gate), so the
// inflight window is never invisible to phase-based monitoring.
func TestGPUNodeReconcileInitializesEmptyPhaseToPending(t *testing.T) {
	ctx := context.Background()

	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: "pool-1"},
	}
	// GPUNode in the inflight window: no status set yet (empty phase, no GPUs discovered).
	gpuNode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "node-1",
			Finalizers: []string{constants.Finalizer},
			Labels: map[string]string{
				fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name): "true",
			},
		},
	}
	// Driver-upgrade label makes Reconcile requeue right after the empty-phase
	// safety net, proving the phase is initialized by the safety net itself and
	// not by any later reconcile step.
	coreNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: gpuNode.Name,
			Labels: map[string]string{
				constants.NvidiaGPUDriverUpgradeStateLabel: "upgrade-in-progress",
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
		WithStatusSubresource(&tfv1.GPUNode{}, &tfv1.GPU{}).
		WithObjects(pool, gpuNode, coreNode).
		Build()

	reconciler := &GPUNodeReconciler{
		Client: kubeClient,
		Scheme: scheme,
	}

	if _, err := reconciler.Reconcile(ctx, ctrl.Request{NamespacedName: types.NamespacedName{Name: gpuNode.Name}}); err != nil {
		t.Fatalf("Reconcile: %v", err)
	}

	updatedNode := &tfv1.GPUNode{}
	if err := kubeClient.Get(ctx, types.NamespacedName{Name: gpuNode.Name}, updatedNode); err != nil {
		t.Fatalf("get updated GPUNode: %v", err)
	}
	if updatedNode.Status.Phase != tfv1.TensorFusionGPUNodePhasePending {
		t.Fatalf("expected GPUNode phase %q, got %q", tfv1.TensorFusionGPUNodePhasePending, updatedNode.Status.Phase)
	}
}

// TestIsNodeReady pins the gate signal used by reconcileHypervisorPod: only a
// Ready=True node may get a hypervisor pod. Ready False/Unknown, a missing
// Ready condition, and a nil node must all read as not-ready so no pod is
// created on a dead node. Cordon (spec.Unschedulable) is deliberately not part
// of this signal.
func TestIsNodeReady(t *testing.T) {
	nodeWith := func(status corev1.ConditionStatus) *corev1.Node {
		return &corev1.Node{Status: corev1.NodeStatus{Conditions: []corev1.NodeCondition{
			{Type: corev1.NodeMemoryPressure, Status: corev1.ConditionFalse},
			{Type: corev1.NodeReady, Status: status},
		}}}
	}
	cases := []struct {
		name string
		node *corev1.Node
		want bool
	}{
		{"ready", nodeWith(corev1.ConditionTrue), true},
		{"not-ready", nodeWith(corev1.ConditionFalse), false},
		{"unknown", nodeWith(corev1.ConditionUnknown), false},
		{"no-ready-condition", &corev1.Node{}, false},
		{"nil", nil, false},
		{"cordoned-but-ready", &corev1.Node{
			Spec:   corev1.NodeSpec{Unschedulable: true},
			Status: corev1.NodeStatus{Conditions: []corev1.NodeCondition{{Type: corev1.NodeReady, Status: corev1.ConditionTrue}}},
		}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := isNodeReady(c.node); got != c.want {
				t.Fatalf("isNodeReady(%s) = %v, want %v", c.name, got, c.want)
			}
		})
	}
}
