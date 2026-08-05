/*
Copyright 2026.

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

package utils

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestResolveNodeIsolationMode(t *testing.T) {
	cfg := &tfv1.NodeManagerConfig{
		DefaultIsolationMode: tfv1.IsolationModeSoft,
		IsolationModeRules: []tfv1.NodeIsolationModeRule{
			{
				Mode: tfv1.IsolationModePartitioned,
				Selector: metav1.LabelSelector{MatchExpressions: []metav1.LabelSelectorRequirement{
					{
						Key:      corev1.LabelInstanceTypeStable,
						Operator: metav1.LabelSelectorOpIn,
						Values:   []string{"p4d.24xlarge", "p5.48xlarge"},
					},
				}},
			},
			{
				Mode: tfv1.IsolationModeHard,
				Selector: metav1.LabelSelector{MatchLabels: map[string]string{
					"tensor-fusion.ai/gpu-model": "H100",
				}},
			},
		},
	}

	tests := []struct {
		name   string
		labels map[string]string
		want   tfv1.IsolationModeType
	}{
		{
			name:   "expression matches one of multiple instance types",
			labels: map[string]string{corev1.LabelInstanceTypeStable: "p5.48xlarge"},
			want:   tfv1.IsolationModePartitioned,
		},
		{
			name: "first matching rule wins",
			labels: map[string]string{
				corev1.LabelInstanceTypeStable: "p4d.24xlarge",
				"tensor-fusion.ai/gpu-model":   "H100",
			},
			want: tfv1.IsolationModePartitioned,
		},
		{
			name:   "falls back to configured default",
			labels: map[string]string{"unrelated": "true"},
			want:   tfv1.IsolationModeSoft,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Labels: tt.labels}}
			got, err := ResolveNodeIsolationMode(node, cfg)
			if err != nil {
				t.Fatalf("ResolveNodeIsolationMode() error = %v", err)
			}
			if got != tt.want {
				t.Fatalf("ResolveNodeIsolationMode() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestResolveNodeIsolationModeDefaultsToSoft(t *testing.T) {
	got, err := ResolveNodeIsolationMode(&corev1.Node{}, nil)
	if err != nil {
		t.Fatalf("ResolveNodeIsolationMode() error = %v", err)
	}
	if got != tfv1.IsolationModeSoft {
		t.Fatalf("ResolveNodeIsolationMode() = %q, want %q", got, tfv1.IsolationModeSoft)
	}
}

func TestNodeIsolationPolicyHash(t *testing.T) {
	base := &tfv1.NodeManagerConfig{DefaultIsolationMode: tfv1.IsolationModeSoft}
	sameEffectiveDefault := &tfv1.NodeManagerConfig{}
	hard := &tfv1.NodeManagerConfig{DefaultIsolationMode: tfv1.IsolationModeHard}

	if NodeIsolationPolicyHash(base) != NodeIsolationPolicyHash(sameEffectiveDefault) {
		t.Fatal("empty and explicit soft defaults must have the same policy hash")
	}
	if NodeIsolationPolicyHash(base) == NodeIsolationPolicyHash(hard) {
		t.Fatal("different defaults must have different policy hashes")
	}
}
