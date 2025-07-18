package config

import (
	"encoding/json"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
)

// This is for unit testing
var MockGPUPoolSpec = &tfv1.GPUPoolSpec{
	CapacityConfig: &tfv1.CapacityConfig{
		Oversubscription: &tfv1.Oversubscription{
			TFlopsOversellRatio: 2000,
		},
		MinResources: &tfv1.GPUOrCPUResourceUnit{
			TFlops: resource.MustParse("100"),
			VRAM:   resource.MustParse("10Gi"),
		},
		MaxResources: &tfv1.GPUOrCPUResourceUnit{
			TFlops: resource.MustParse("3000"),
			VRAM:   resource.MustParse("3000Gi"),
		},
		WarmResources: &tfv1.GPUOrCPUResourceUnit{
			TFlops: resource.MustParse("2200"),
			VRAM:   resource.MustParse("2020Gi"),
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
		ProvisioningMode: tfv1.ProvisioningModeAutoSelect,
	},
	ComponentConfig: &tfv1.ComponentConfig{
		Hypervisor: &tfv1.HypervisorConfig{
			Image:       "hypervisor",
			VectorImage: "vector",
			PodTemplate: &runtime.RawExtension{
				Raw: lo.Must(json.Marshal(
					corev1.PodTemplate{
						Template: corev1.PodTemplateSpec{
							Spec: corev1.PodSpec{
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
			Image: "node-discovery",
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
			Image: "worker",
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
			RemoteModeImage:   "client",
			EmbeddedModeImage: "ngpu",
			OperatorEndpoint:  "http://localhost:8080",
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
	QosConfig: &tfv1.QosConfig{
		Definitions: []tfv1.QosDefinition{
			{
				Name: constants.QoSLevelMedium,
			},
			{
				Name: constants.QoSLevelHigh,
			},
		},
		DefaultQoS: constants.QoSLevelMedium,
		Pricing: []tfv1.QosPricing{
			{
				Qos: constants.QoSLevelMedium,
				Requests: tfv1.GPUResourcePricingUnit{
					PerFP16TFlopsPerHour: "2",
					PerGBOfVRAMPerHour:   "1",
				},
				LimitsOverRequestsChargingRatio: "0.5",
			},
			{
				Qos: constants.QoSLevelHigh,
				Requests: tfv1.GPUResourcePricingUnit{
					PerFP16TFlopsPerHour: "2",
					PerGBOfVRAMPerHour:   "1",
				},
				LimitsOverRequestsChargingRatio: "0.8",
			},
		},
	},
}
