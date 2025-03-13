package worker

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
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

// TestSelectWorker tests the SelectWorker function
func TestSelectWorker(t *testing.T) {
	// Define test cases
	tests := []struct {
		name           string
		maxSkew        int32
		workload       *tfv1.TensorFusionWorkload
		connections    []tfv1.TensorFusionConnection
		expectedWorker string
		expectError    bool
		errorSubstring string
	}{
		{
			name:    "no workers available",
			maxSkew: 1,
			workload: &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-workload",
					Namespace: "default",
				},
				Status: tfv1.TensorFusionWorkloadStatus{
					WorkerStatuses: []tfv1.WorkerStatus{},
				},
			},
			connections:    []tfv1.TensorFusionConnection{},
			expectedWorker: "",
			expectError:    true,
			errorSubstring: "no available worker",
		},
		{
			name:    "one worker with no connections",
			maxSkew: 1,
			workload: &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-workload",
					Namespace: "default",
				},
				Status: tfv1.TensorFusionWorkloadStatus{
					WorkerStatuses: []tfv1.WorkerStatus{
						{
							WorkerName:  "worker-1",
							WorkerPhase: tfv1.WorkerRunning,
						},
					},
				},
			},
			connections:    []tfv1.TensorFusionConnection{},
			expectedWorker: "worker-1",
			expectError:    false,
		},
		{
			name:    "two workers with balanced load",
			maxSkew: 1,
			workload: &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-workload",
					Namespace: "default",
				},
				Status: tfv1.TensorFusionWorkloadStatus{
					WorkerStatuses: []tfv1.WorkerStatus{
						{
							WorkerName:  "worker-1",
							WorkerPhase: tfv1.WorkerRunning,
						},
						{
							WorkerName:  "worker-2",
							WorkerPhase: tfv1.WorkerRunning,
						},
					},
				},
			},
			connections: []tfv1.TensorFusionConnection{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "conn-1",
						Namespace: "default",
						Labels: map[string]string{
							constants.WorkloadKey: "test-workload",
						},
					},
					Status: tfv1.TensorFusionConnectionStatus{
						WorkerName: "worker-1",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "conn-2",
						Namespace: "default",
						Labels: map[string]string{
							constants.WorkloadKey: "test-workload",
						},
					},
					Status: tfv1.TensorFusionConnectionStatus{
						WorkerName: "worker-2",
					},
				},
			},
			expectedWorker: "worker-1", // Both have equal load, should select worker-1 as it's first in list
			expectError:    false,
		},
		{
			name:    "three workers with uneven load, maxSkew=1",
			maxSkew: 1,
			workload: &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-workload",
					Namespace: "default",
				},
				Status: tfv1.TensorFusionWorkloadStatus{
					WorkerStatuses: []tfv1.WorkerStatus{
						{
							WorkerName:  "worker-1",
							WorkerPhase: tfv1.WorkerRunning,
						},
						{
							WorkerName:  "worker-2",
							WorkerPhase: tfv1.WorkerRunning,
						},
						{
							WorkerName:  "worker-3",
							WorkerPhase: tfv1.WorkerRunning,
						},
					},
				},
			},
			connections: []tfv1.TensorFusionConnection{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "conn-1",
						Namespace: "default",
						Labels: map[string]string{
							constants.WorkloadKey: "test-workload",
						},
					},
					Status: tfv1.TensorFusionConnectionStatus{
						WorkerName: "worker-1",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "conn-2",
						Namespace: "default",
						Labels: map[string]string{
							constants.WorkloadKey: "test-workload",
						},
					},
					Status: tfv1.TensorFusionConnectionStatus{
						WorkerName: "worker-1",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "conn-3",
						Namespace: "default",
						Labels: map[string]string{
							constants.WorkloadKey: "test-workload",
						},
					},
					Status: tfv1.TensorFusionConnectionStatus{
						WorkerName: "worker-2",
					},
				},
			},
			expectedWorker: "worker-3", // Has zero connections, should be selected
			expectError:    false,
		},
		{
			name:    "worker with failed status should be skipped",
			maxSkew: 1,
			workload: &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-workload",
					Namespace: "default",
				},
				Status: tfv1.TensorFusionWorkloadStatus{
					WorkerStatuses: []tfv1.WorkerStatus{
						{
							WorkerName:  "worker-1",
							WorkerPhase: tfv1.WorkerFailed,
						},
						{
							WorkerName:  "worker-2",
							WorkerPhase: tfv1.WorkerRunning,
						},
					},
				},
			},
			connections: []tfv1.TensorFusionConnection{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "conn-1",
						Namespace: "default",
						Labels: map[string]string{
							constants.WorkloadKey: "test-workload",
						},
					},
					Status: tfv1.TensorFusionConnectionStatus{
						WorkerName: "worker-1", // Even though it has a connection, it's failed so should be skipped
					},
				},
			},
			expectedWorker: "worker-2",
			expectError:    false,
		},
		{
			name:    "maxSkew=0 should select worker with minimum usage",
			maxSkew: 0,
			workload: &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-workload",
					Namespace: "default",
				},
				Status: tfv1.TensorFusionWorkloadStatus{
					WorkerStatuses: []tfv1.WorkerStatus{
						{
							WorkerName:  "worker-1",
							WorkerPhase: tfv1.WorkerRunning,
						},
						{
							WorkerName:  "worker-2",
							WorkerPhase: tfv1.WorkerRunning,
						},
						{
							WorkerName:  "worker-3",
							WorkerPhase: tfv1.WorkerRunning,
						},
					},
				},
			},
			connections: []tfv1.TensorFusionConnection{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "conn-1",
						Namespace: "default",
						Labels: map[string]string{
							constants.WorkloadKey: "test-workload",
						},
					},
					Status: tfv1.TensorFusionConnectionStatus{
						WorkerName: "worker-1",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "conn-2",
						Namespace: "default",
						Labels: map[string]string{
							constants.WorkloadKey: "test-workload",
						},
					},
					Status: tfv1.TensorFusionConnectionStatus{
						WorkerName: "worker-2",
					},
				},
			},
			expectedWorker: "worker-3", // Has 0 connections, the other two have 1 each
			expectError:    false,
		},
		{
			name:    "maxSkew=2 should allow selection from wider range",
			maxSkew: 2,
			workload: &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-workload",
					Namespace: "default",
				},
				Status: tfv1.TensorFusionWorkloadStatus{
					WorkerStatuses: []tfv1.WorkerStatus{
						{
							WorkerName:  "worker-1",
							WorkerPhase: tfv1.WorkerRunning,
						},
						{
							WorkerName:  "worker-2",
							WorkerPhase: tfv1.WorkerRunning,
						},
						{
							WorkerName:  "worker-3",
							WorkerPhase: tfv1.WorkerRunning,
						},
					},
				},
			},
			connections: []tfv1.TensorFusionConnection{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "conn-1",
						Namespace: "default",
						Labels: map[string]string{
							constants.WorkloadKey: "test-workload",
						},
					},
					Status: tfv1.TensorFusionConnectionStatus{
						WorkerName: "worker-1",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "conn-2",
						Namespace: "default",
						Labels: map[string]string{
							constants.WorkloadKey: "test-workload",
						},
					},
					Status: tfv1.TensorFusionConnectionStatus{
						WorkerName: "worker-1",
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "conn-3",
						Namespace: "default",
						Labels: map[string]string{
							constants.WorkloadKey: "test-workload",
						},
					},
					Status: tfv1.TensorFusionConnectionStatus{
						WorkerName: "worker-2",
					},
				},
			},
			expectedWorker: "worker-3", // Worker-3 has 0, Worker-2 has 1, Worker-1 has 2, all within maxSkew=2
			expectError:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a new scheme and add the types that we need to register
			scheme := runtime.NewScheme()
			_ = tfv1.AddToScheme(scheme)

			// Create a list of connection objects to be returned when List is called
			connectionList := &tfv1.TensorFusionConnectionList{
				Items: tt.connections,
			}

			// Create a fake client that returns our connection list
			client := fake.NewClientBuilder().
				WithScheme(scheme).
				WithLists(connectionList).
				Build()

			// Call the function under test
			worker, err := SelectWorker(context.Background(), client, tt.workload, tt.maxSkew)

			// Check the error condition
			if tt.expectError {
				assert.Error(t, err)
				if tt.errorSubstring != "" {
					assert.Contains(t, err.Error(), tt.errorSubstring)
				}
				assert.Nil(t, worker)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, worker)
				assert.Equal(t, tt.expectedWorker, worker.WorkerName)
			}
		})
	}
}
