package worker

import (
	"testing"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestWorkerPort(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected int
		wantErr  bool
	}{
		{
			name: "port found in env vars",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Env: []corev1.EnvVar{
								{
									Name:  constants.WorkerPortEnv,
									Value: "8080",
								},
							},
						},
					},
				},
			},
			expected: 8080,
			wantErr:  false,
		},
		{
			name: "port not found in env vars",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Env: []corev1.EnvVar{
								{
									Name:  "OTHER_ENV",
									Value: "value",
								},
							},
						},
					},
				},
			},
			expected: 0,
			wantErr:  true,
		},
	}

	wg := &WorkerGenerator{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			port, err := wg.WorkerPort(tt.pod)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, port)
			}
		})
	}
}
