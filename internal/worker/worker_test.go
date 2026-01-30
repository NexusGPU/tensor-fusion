package worker

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

var _ = Describe("SelectWorker", func() {
	It("should return error when no workers available", func() {
		scheme := runtime.NewScheme()
		_ = tfv1.AddToScheme(scheme)
		_ = v1.AddToScheme(scheme)

		connectionList := &tfv1.TensorFusionConnectionList{
			Items: []tfv1.TensorFusionConnection{},
		}

		client := fake.NewClientBuilder().
			WithScheme(scheme).
			WithLists(connectionList).
			WithLists(generateWorkerPodList([]tfv1.WorkerStatus{})).
			Build()

		workload := &tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-workload",
				Namespace: "default",
			},
		}

		worker, err := SelectWorker(context.Background(), client, workload, 1)

		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("no available worker"))
		Expect(worker).To(BeNil())
	})

	It("should select one worker with no connections from dynamic replicas", func() {
		scheme := runtime.NewScheme()
		_ = tfv1.AddToScheme(scheme)
		_ = v1.AddToScheme(scheme)

		connectionList := &tfv1.TensorFusionConnectionList{
			Items: []tfv1.TensorFusionConnection{},
		}

		workerStatuses := []tfv1.WorkerStatus{
			{
				WorkerName:  "worker-1",
				WorkerPhase: tfv1.WorkerRunning,
			},
		}

		client := fake.NewClientBuilder().
			WithScheme(scheme).
			WithLists(connectionList).
			WithLists(generateWorkerPodList(workerStatuses)).
			Build()

		workload := &tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-workload",
				Namespace: "default",
			},
			Spec: tfv1.WorkloadProfileSpec{
				Replicas: ptr.To(int32(1)),
			},
		}

		worker, err := SelectWorker(context.Background(), client, workload, 1)

		Expect(err).NotTo(HaveOccurred())
		Expect(worker).NotTo(BeNil())
		Expect(worker.WorkerName).To(Equal("worker-1"))
	})

	It("should select worker from two workers with balanced load", func() {
		scheme := runtime.NewScheme()
		_ = tfv1.AddToScheme(scheme)
		_ = v1.AddToScheme(scheme)

		connections := []tfv1.TensorFusionConnection{
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
		}

		connectionList := &tfv1.TensorFusionConnectionList{
			Items: connections,
		}

		workerStatuses := []tfv1.WorkerStatus{
			{
				WorkerName:  "worker-1",
				WorkerPhase: tfv1.WorkerRunning,
			},
			{
				WorkerName:  "worker-2",
				WorkerPhase: tfv1.WorkerRunning,
			},
		}

		client := fake.NewClientBuilder().
			WithScheme(scheme).
			WithLists(connectionList).
			WithLists(generateWorkerPodList(workerStatuses)).
			Build()

		workload := &tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-workload",
				Namespace: "default",
			},
			Spec: tfv1.WorkloadProfileSpec{
				Replicas: ptr.To(int32(2)),
			},
		}

		worker, err := SelectWorker(context.Background(), client, workload, 1)

		Expect(err).NotTo(HaveOccurred())
		Expect(worker).NotTo(BeNil())
		Expect(worker.WorkerName).To(Equal("worker-1"))
	})

	It("should select worker with zero connections when three workers have uneven load with maxSkew=1", func() {
		scheme := runtime.NewScheme()
		_ = tfv1.AddToScheme(scheme)
		_ = v1.AddToScheme(scheme)

		connections := []tfv1.TensorFusionConnection{
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
		}

		connectionList := &tfv1.TensorFusionConnectionList{
			Items: connections,
		}

		workerStatuses := []tfv1.WorkerStatus{
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
		}

		client := fake.NewClientBuilder().
			WithScheme(scheme).
			WithLists(connectionList).
			WithLists(generateWorkerPodList(workerStatuses)).
			Build()

		workload := &tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-workload",
				Namespace: "default",
			},
		}

		worker, err := SelectWorker(context.Background(), client, workload, 1)

		Expect(err).NotTo(HaveOccurred())
		Expect(worker).NotTo(BeNil())
		Expect(worker.WorkerName).To(Equal("worker-3"))
	})

	It("should skip worker with failed status", func() {
		scheme := runtime.NewScheme()
		_ = tfv1.AddToScheme(scheme)
		_ = v1.AddToScheme(scheme)

		connections := []tfv1.TensorFusionConnection{
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
		}

		connectionList := &tfv1.TensorFusionConnectionList{
			Items: connections,
		}

		workerStatuses := []tfv1.WorkerStatus{
			{
				WorkerName:  "worker-1",
				WorkerPhase: tfv1.WorkerFailed,
			},
			{
				WorkerName:  "worker-2",
				WorkerPhase: tfv1.WorkerRunning,
			},
		}

		client := fake.NewClientBuilder().
			WithScheme(scheme).
			WithLists(connectionList).
			WithLists(generateWorkerPodList(workerStatuses)).
			Build()

		workload := &tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-workload",
				Namespace: "default",
			},
			Spec: tfv1.WorkloadProfileSpec{
				Replicas: ptr.To(int32(1)),
			},
		}

		worker, err := SelectWorker(context.Background(), client, workload, 1)

		Expect(err).NotTo(HaveOccurred())
		Expect(worker).NotTo(BeNil())
		Expect(worker.WorkerName).To(Equal("worker-2"))
	})

	It("should select worker with minimum usage when maxSkew=0", func() {
		scheme := runtime.NewScheme()
		_ = tfv1.AddToScheme(scheme)
		_ = v1.AddToScheme(scheme)

		connections := []tfv1.TensorFusionConnection{
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
		}

		connectionList := &tfv1.TensorFusionConnectionList{
			Items: connections,
		}

		workerStatuses := []tfv1.WorkerStatus{
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
		}

		client := fake.NewClientBuilder().
			WithScheme(scheme).
			WithLists(connectionList).
			WithLists(generateWorkerPodList(workerStatuses)).
			Build()

		workload := &tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-workload",
				Namespace: "default",
			},
		}

		worker, err := SelectWorker(context.Background(), client, workload, 0)

		Expect(err).NotTo(HaveOccurred())
		Expect(worker).NotTo(BeNil())
		Expect(worker.WorkerName).To(Equal("worker-3"))
	})

	It("should allow selection from wider range when maxSkew=2", func() {
		scheme := runtime.NewScheme()
		_ = tfv1.AddToScheme(scheme)
		_ = v1.AddToScheme(scheme)

		connections := []tfv1.TensorFusionConnection{
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
		}

		connectionList := &tfv1.TensorFusionConnectionList{
			Items: connections,
		}

		workerStatuses := []tfv1.WorkerStatus{
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
		}

		client := fake.NewClientBuilder().
			WithScheme(scheme).
			WithLists(connectionList).
			WithLists(generateWorkerPodList(workerStatuses)).
			Build()

		workload := &tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-workload",
				Namespace: "default",
			},
		}

		worker, err := SelectWorker(context.Background(), client, workload, 2)

		Expect(err).NotTo(HaveOccurred())
		Expect(worker).NotTo(BeNil())
		Expect(worker.WorkerName).To(Equal("worker-3"))
	})
})

var _ = Describe("mergePodTemplateSpec", func() {
	It("should merge labels", func() {
		base := &v1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					"app": "base",
				},
			},
		}
		override := &v1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					"env": "prod",
				},
			},
		}

		err := mergePodTemplateSpec(base, override)
		Expect(err).NotTo(HaveOccurred())
		Expect(base.Labels["app"]).To(Equal("base"))
		Expect(base.Labels["env"]).To(Equal("prod"))
	})

	It("should override existing labels", func() {
		base := &v1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					"app": "base",
					"env": "dev",
				},
			},
		}
		override := &v1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					"env": "prod",
				},
			},
		}

		err := mergePodTemplateSpec(base, override)
		Expect(err).NotTo(HaveOccurred())
		Expect(base.Labels["app"]).To(Equal("base"))
		Expect(base.Labels["env"]).To(Equal("prod"))
	})

	It("should merge annotations", func() {
		base := &v1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					"base-annotation": "value1",
				},
			},
		}
		override := &v1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					"override-annotation": "value2",
				},
			},
		}

		err := mergePodTemplateSpec(base, override)
		Expect(err).NotTo(HaveOccurred())
		Expect(base.Annotations["base-annotation"]).To(Equal("value1"))
		Expect(base.Annotations["override-annotation"]).To(Equal("value2"))
	})

	It("should merge container env vars", func() {
		base := &v1.PodTemplateSpec{
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
		}
		override := &v1.PodTemplateSpec{
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
		}

		err := mergePodTemplateSpec(base, override)
		Expect(err).NotTo(HaveOccurred())
		Expect(base.Spec.Containers).To(HaveLen(1))
		Expect(base.Spec.Containers[0].Image).To(Equal("base-image:v1"))
		Expect(base.Spec.Containers[0].Env).To(HaveLen(2))
		envMap := make(map[string]string)
		for _, env := range base.Spec.Containers[0].Env {
			envMap[env.Name] = env.Value
		}
		Expect(envMap["BASE_VAR"]).To(Equal("base"))
		Expect(envMap["OVERRIDE_VAR"]).To(Equal("override"))
	})

	It("should override container image", func() {
		base := &v1.PodTemplateSpec{
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "worker",
						Image: "base-image:v1",
					},
				},
			},
		}
		override := &v1.PodTemplateSpec{
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "worker",
						Image: "override-image:v2",
					},
				},
			},
		}

		err := mergePodTemplateSpec(base, override)
		Expect(err).NotTo(HaveOccurred())
		Expect(base.Spec.Containers).To(HaveLen(1))
		Expect(base.Spec.Containers[0].Image).To(Equal("override-image:v2"))
	})

	It("should merge resource requests", func() {
		base := &v1.PodTemplateSpec{
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "worker",
						Image: "base-image:v1",
					},
				},
			},
		}
		override := &v1.PodTemplateSpec{
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
		}

		err := mergePodTemplateSpec(base, override)
		Expect(err).NotTo(HaveOccurred())
		Expect(base.Spec.Containers).To(HaveLen(1))
		memRequest := base.Spec.Containers[0].Resources.Requests[v1.ResourceMemory]
		Expect(memRequest.String()).To(Equal("2Gi"))
	})

	It("should add new container", func() {
		base := &v1.PodTemplateSpec{
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "worker",
						Image: "base-image:v1",
					},
				},
			},
		}
		override := &v1.PodTemplateSpec{
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "sidecar",
						Image: "sidecar-image:v1",
					},
				},
			},
		}

		err := mergePodTemplateSpec(base, override)
		Expect(err).NotTo(HaveOccurred())
		Expect(base.Spec.Containers).To(HaveLen(2))
		containerNames := []string{base.Spec.Containers[0].Name, base.Spec.Containers[1].Name}
		Expect(containerNames).To(ContainElement("worker"))
		Expect(containerNames).To(ContainElement("sidecar"))
	})

	It("should merge volumes", func() {
		base := &v1.PodTemplateSpec{
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
		}
		override := &v1.PodTemplateSpec{
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
		}

		err := mergePodTemplateSpec(base, override)
		Expect(err).NotTo(HaveOccurred())
		Expect(base.Spec.Volumes).To(HaveLen(2))
		volumeNames := []string{base.Spec.Volumes[0].Name, base.Spec.Volumes[1].Name}
		Expect(volumeNames).To(ContainElement("base-volume"))
		Expect(volumeNames).To(ContainElement("override-volume"))
	})
})

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
