package config

import (
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

var MockDeployment = &appsv1.Deployment{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "pytorch-example",
		Namespace: "tensor-fusion",
		Labels: map[string]string{
			"app":                      "pytorch-example",
			"tensor-fusion.ai/enabled": "true",
		},
	},
	Spec: appsv1.DeploymentSpec{
		Replicas: ptr.To[int32](1),
		Selector: &metav1.LabelSelector{
			MatchLabels: map[string]string{
				"app": "pytorch-example",
			},
		},
		Template: corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					"app":                      "pytorch-example",
					"tensor-fusion.ai/enabled": "true",
				},
				Annotations: map[string]string{
					"tensor-fusion.ai/generate-workload": "true",
					"tensor-fusion.ai/gpupool":           "mock",
					"tensor-fusion.ai/inject-container":  "python",
					"tensor-fusion.ai/replicas":          "1",
					"tensor-fusion.ai/tflops-limit":      "10",
					"tensor-fusion.ai/tflops-request":    "10",
					"tensor-fusion.ai/vram-limit":        "1Gi",
					"tensor-fusion.ai/vram-request":      "1Gi",
					"tensor-fusion.ai/workload":          "pytorch-example",
				},
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name:    "python",
						Image:   "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime",
						Command: []string{"sh", "-c", "sleep", "1d"},
					},
				},
			},
		},
	},
}
