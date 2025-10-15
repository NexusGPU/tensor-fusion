package worker

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
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

// TestMergePodTemplateSpec tests the mergePodTemplateSpec function
func TestMergePodTemplateSpec(t *testing.T) {
	tests := []struct {
		name     string
		base     *v1.PodTemplateSpec
		override *v1.PodTemplateSpec
		validate func(t *testing.T, merged *v1.PodTemplateSpec)
	}{
		{
			name: "merge labels",
			base: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": "base",
					},
				},
			},
			override: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"env": "prod",
					},
				},
			},
			validate: func(t *testing.T, merged *v1.PodTemplateSpec) {
				assert.Equal(t, "base", merged.Labels["app"])
				assert.Equal(t, "prod", merged.Labels["env"])
			},
		},
		{
			name: "override existing labels",
			base: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": "base",
						"env": "dev",
					},
				},
			},
			override: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"env": "prod",
					},
				},
			},
			validate: func(t *testing.T, merged *v1.PodTemplateSpec) {
				assert.Equal(t, "base", merged.Labels["app"])
				assert.Equal(t, "prod", merged.Labels["env"])
			},
		},
		{
			name: "merge annotations",
			base: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"base-annotation": "value1",
					},
				},
			},
			override: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Annotations: map[string]string{
						"override-annotation": "value2",
					},
				},
			},
			validate: func(t *testing.T, merged *v1.PodTemplateSpec) {
				assert.Equal(t, "value1", merged.Annotations["base-annotation"])
				assert.Equal(t, "value2", merged.Annotations["override-annotation"])
			},
		},
		{
			name: "merge container env vars",
			base: &v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "worker",
							Image: "base-image:v1",
							Env: []v1.EnvVar{
								{Name: "BASE_VAR", Value: "base"},
							},
						},
					},
				},
			},
			override: &v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "worker",
							Env: []v1.EnvVar{
								{Name: "OVERRIDE_VAR", Value: "override"},
							},
						},
					},
				},
			},
			validate: func(t *testing.T, merged *v1.PodTemplateSpec) {
				assert.Len(t, merged.Spec.Containers, 1)
				assert.Equal(t, "base-image:v1", merged.Spec.Containers[0].Image)
				assert.Len(t, merged.Spec.Containers[0].Env, 2)
				envMap := make(map[string]string)
				for _, env := range merged.Spec.Containers[0].Env {
					envMap[env.Name] = env.Value
				}
				assert.Equal(t, "base", envMap["BASE_VAR"])
				assert.Equal(t, "override", envMap["OVERRIDE_VAR"])
			},
		},
		{
			name: "override container image",
			base: &v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "worker",
							Image: "base-image:v1",
						},
					},
				},
			},
			override: &v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "worker",
							Image: "override-image:v2",
						},
					},
				},
			},
			validate: func(t *testing.T, merged *v1.PodTemplateSpec) {
				assert.Len(t, merged.Spec.Containers, 1)
				assert.Equal(t, "override-image:v2", merged.Spec.Containers[0].Image)
			},
		},
		{
			name: "merge resource requests",
			base: &v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "worker",
							Image: "base-image:v1",
						},
					},
				},
			},
			override: &v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "worker",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceMemory: *ptr.To(resource.MustParse("2Gi")),
								},
							},
						},
					},
				},
			},
			validate: func(t *testing.T, merged *v1.PodTemplateSpec) {
				assert.Len(t, merged.Spec.Containers, 1)
				memRequest := merged.Spec.Containers[0].Resources.Requests[v1.ResourceMemory]
				assert.Equal(t, "2Gi", memRequest.String())
			},
		},
		{
			name: "add new container",
			base: &v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "worker",
							Image: "base-image:v1",
						},
					},
				},
			},
			override: &v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "sidecar",
							Image: "sidecar-image:v1",
						},
					},
				},
			},
			validate: func(t *testing.T, merged *v1.PodTemplateSpec) {
				assert.Len(t, merged.Spec.Containers, 2)
				containerNames := []string{merged.Spec.Containers[0].Name, merged.Spec.Containers[1].Name}
				assert.Contains(t, containerNames, "worker")
				assert.Contains(t, containerNames, "sidecar")
			},
		},
		{
			name: "merge volumes",
			base: &v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "base-volume",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{},
							},
						},
					},
				},
			},
			override: &v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "override-volume",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{},
							},
						},
					},
				},
			},
			validate: func(t *testing.T, merged *v1.PodTemplateSpec) {
				assert.Len(t, merged.Spec.Volumes, 2)
				volumeNames := []string{merged.Spec.Volumes[0].Name, merged.Spec.Volumes[1].Name}
				assert.Contains(t, volumeNames, "base-volume")
				assert.Contains(t, volumeNames, "override-volume")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := mergePodTemplateSpec(tt.base, tt.override)
			assert.NoError(t, err)
			tt.validate(t, tt.base)
		})
	}
}
