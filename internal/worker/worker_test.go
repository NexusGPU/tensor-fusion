package worker

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

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
		workerStatuses []tfv1.WorkerStatus
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
			},
			workerStatuses: []tfv1.WorkerStatus{},
			connections:    []tfv1.TensorFusionConnection{},
			expectedWorker: "",
			expectError:    true,
			errorSubstring: "no available worker",
		},
		{
			name:    "one worker with no connections from dynamic replicas",
			maxSkew: 1,
			workload: &tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-workload",
					Namespace: "default",
				},
				Spec: tfv1.WorkloadProfileSpec{
					Replicas: ptr.To(int32(1)),
				},
			},
			workerStatuses: []tfv1.WorkerStatus{
				{
					WorkerName:  "worker-1",
					WorkerPhase: tfv1.WorkerRunning,
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
				Spec: tfv1.WorkloadProfileSpec{
					Replicas: ptr.To(int32(2)),
				},
			},
			workerStatuses: []tfv1.WorkerStatus{
				{
					WorkerName:  "worker-1",
					WorkerPhase: tfv1.WorkerRunning,
				},
				{
					WorkerName:  "worker-2",
					WorkerPhase: tfv1.WorkerRunning,
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
			},
			workerStatuses: []tfv1.WorkerStatus{
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
				Spec: tfv1.WorkloadProfileSpec{
					Replicas: ptr.To(int32(1)),
				},
			},
			workerStatuses: []tfv1.WorkerStatus{
				{
					WorkerName:  "worker-1",
					WorkerPhase: tfv1.WorkerFailed,
				},
				{
					WorkerName:  "worker-2",
					WorkerPhase: tfv1.WorkerRunning,
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
			},
			workerStatuses: []tfv1.WorkerStatus{
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
			},
			workerStatuses: []tfv1.WorkerStatus{
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
			_ = v1.AddToScheme(scheme)

			// Create a list of connection objects to be returned when List is called
			connectionList := &tfv1.TensorFusionConnectionList{
				Items: tt.connections,
			}

			// Create a fake client that returns our connection list
			client := fake.NewClientBuilder().
				WithScheme(scheme).
				WithLists(connectionList).
				WithLists(generateWorkerPodList(tt.workerStatuses)).
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

func generateWorkerPodList(workloadStatus []tfv1.WorkerStatus) *v1.PodList {
	return &v1.PodList{
		Items: lo.Map(workloadStatus, func(status tfv1.WorkerStatus, _ int) v1.Pod {
			return v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: status.WorkerName,
					Labels: map[string]string{
						constants.WorkloadKey: "test-workload",
					},
				},
				Status: v1.PodStatus{
					Phase: v1.PodPhase(status.WorkerPhase),
				},
			}
		}),
	}
}
