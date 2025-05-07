package config

import (
	"encoding/json"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
)

// This is for unit testing
var MockGPUPoolSpec = &tfv1.GPUPoolSpec{
	CapacityConfig: &tfv1.CapacityConfig{
		Oversubscription: &tfv1.Oversubscription{
			TFlopsOversellRatio: 2000,
		},
	},
	NodeManagerConfig: &tfv1.NodeManagerConfig{
		NodeSelector: &corev1.NodeSelector{
			NodeSelectorTerms: []corev1.NodeSelectorTerm{
				{
					MatchExpressions: []corev1.NodeSelectorRequirement{
						{
							Key:      "mock-label",
							Operator: "In",
							Values:   []string{"true"},
						},
					},
				},
			},
		},
		NodePoolRollingUpdatePolicy: &tfv1.NodeRollingUpdatePolicy{
			AutoUpdate:        ptr.To(false),
			BatchPercentage:   25,
			BatchInterval:     "10m",
			MaxDuration:       "10m",
			MaintenanceWindow: tfv1.MaintenanceWindow{},
		},
	},
	ComponentConfig: &tfv1.ComponentConfig{
		Hypervisor: &tfv1.HypervisorConfig{
			PodTemplate: &runtime.RawExtension{
				Raw: lo.Must(json.Marshal(
					corev1.PodTemplate{
						Template: corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								RestartPolicy: corev1.RestartPolicyOnFailure,
								Containers: []corev1.Container{
									{
										Name:    "tensorfusion-hypervisor",
										Image:   "busybox:stable-glibc",
										Command: []string{"sleep", "infinity"},
									},
								},
							},
						},
					},
				)),
			},
		},
		NodeDiscovery: &tfv1.NodeDiscoveryConfig{
			PodTemplate: &runtime.RawExtension{
				Raw: lo.Must(json.Marshal(
					corev1.PodTemplate{
						Template: corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								RestartPolicy:                 corev1.RestartPolicyOnFailure,
								TerminationGracePeriodSeconds: ptr.To[int64](0),
								Containers: []corev1.Container{
									{
										Name:    "tensorfusion-node-discovery",
										Image:   "busybox:stable-glibc",
										Command: []string{"sleep", "infinity"},
									},
								},
							},
						},
					},
				)),
			},
		},
		Worker: &tfv1.WorkerConfig{
			PodTemplate: &runtime.RawExtension{
				Raw: lo.Must(json.Marshal(
					corev1.PodTemplate{
						Template: corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
								TerminationGracePeriodSeconds: ptr.To[int64](0),
								Containers: []corev1.Container{
									{
										Name:    "tensorfusion-worker",
										Image:   "busybox:stable-glibc",
										Command: []string{"sleep", "infinity"},
									},
								},
							},
						},
					},
				)),
			},
		},
		Client: &tfv1.ClientConfig{
			OperatorEndpoint: "http://localhost:8080",
			PatchToPod: &runtime.RawExtension{
				Raw: lo.Must(json.Marshal(map[string]any{
					"spec": map[string]any{
						"initContainers": []corev1.Container{
							{
								Name:  "inject-lib",
								Image: "busybox:stable-glibc",
							},
						},
					},
				})),
			},
			PatchToContainer: &runtime.RawExtension{
				Raw: lo.Must(json.Marshal(map[string]any{
					"env": []corev1.EnvVar{
						{
							Name:  "LD_PRELOAD",
							Value: "tensorfusion.so",
						},
					},
				})),
			},
		},
	},
}
