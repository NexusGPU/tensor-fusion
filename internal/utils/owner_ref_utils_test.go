package utils_test

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

var _ = Describe("Owner Reference Utils", func() {
	var sch *runtime.Scheme

	BeforeEach(func() {
		sch = runtime.NewScheme()
		Expect(corev1.AddToScheme(sch)).To(Succeed())
		Expect(appsv1.AddToScheme(sch)).To(Succeed())
		Expect(batchv1.AddToScheme(sch)).To(Succeed())
	})

	Describe("FindRootOwnerReference", func() {
		It("should return nil when pod has no owner", func() {
			pod := &corev1.Pod{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "Pod",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mypod",
					Namespace: "default",
					UID:       "uid-pod",
				},
			}

			c := fake.NewClientBuilder().WithScheme(sch).WithObjects(pod).Build()

			rootRef, err := utils.FindRootOwnerReference(context.TODO(), c, "default", pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(rootRef).To(BeNil())
		})

		It("should return deployment when hierarchy exists", func() {
			controller := true
			deployment := &appsv1.Deployment{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mydeploy",
					Namespace: "default",
					UID:       "uid-deploy",
				},
			}

			rs := &appsv1.ReplicaSet{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "apps/v1",
					Kind:       "ReplicaSet",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myrs",
					Namespace: "default",
					UID:       "uid-rs",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "Deployment",
							Name:       "mydeploy",
							UID:        deployment.UID,
							Controller: &controller,
						},
					},
				},
			}

			pod := &corev1.Pod{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "Pod",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mypod",
					Namespace: "default",
					UID:       "uid-pod",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "ReplicaSet",
							Name:       "myrs",
							UID:        rs.UID,
							Controller: &controller,
						},
					},
				},
			}

			c := fake.NewClientBuilder().WithScheme(sch).WithObjects(pod, rs, deployment).Build()

			rootRef, err := utils.FindRootOwnerReference(context.TODO(), c, "default", pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(rootRef).NotTo(BeNil())
			Expect(rootRef.Name).To(Equal("mydeploy"))
			Expect(rootRef.Kind).To(Equal("Deployment"))
		})

		It("should return ownerRef when owner is missing", func() {
			controller := true
			pod := &corev1.Pod{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "Pod",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mypod",
					Namespace: "default",
					UID:       "uid-pod",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "ReplicaSet",
							Name:       "missing-rs",
							UID:        "uid-missing",
							Controller: &controller,
						},
					},
				},
			}

			c := fake.NewClientBuilder().WithScheme(sch).WithObjects(pod).Build()

			rootRef, err := utils.FindRootOwnerReference(context.TODO(), c, "default", pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(rootRef).NotTo(BeNil())
			Expect(rootRef.Name).To(Equal("missing-rs"))
			Expect(rootRef.Kind).To(Equal("ReplicaSet"))
		})
	})

	Describe("FindRootControllerRef", func() {
		It("should return nil when pod has no controller", func() {
			pod := &corev1.Pod{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "Pod",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mypod",
					Namespace: "default",
					UID:       "uid-pod",
				},
			}

			c := fake.NewClientBuilder().WithScheme(sch).WithObjects(pod).Build()

			rootRef, err := utils.FindRootControllerRef(context.TODO(), c, pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(rootRef).To(BeNil())
		})

		It("should return deployment when hierarchy exists", func() {
			controller := true
			deployment := &appsv1.Deployment{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mydeploy",
					Namespace: "default",
					UID:       "uid-deploy",
				},
			}

			rs := &appsv1.ReplicaSet{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "apps/v1",
					Kind:       "ReplicaSet",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myrs",
					Namespace: "default",
					UID:       "uid-rs",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "Deployment",
							Name:       "mydeploy",
							UID:        deployment.UID,
							Controller: &controller,
						},
					},
				},
			}

			pod := &corev1.Pod{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "Pod",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mypod",
					Namespace: "default",
					UID:       "uid-pod",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "ReplicaSet",
							Name:       "myrs",
							UID:        rs.UID,
							Controller: &controller,
						},
					},
				},
			}

			c := fake.NewClientBuilder().WithScheme(sch).WithObjects(pod, rs, deployment).Build()

			rootRef, err := utils.FindRootControllerRef(context.TODO(), c, pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(rootRef).NotTo(BeNil())
			Expect(rootRef.Name).To(Equal("mydeploy"))
			Expect(rootRef.Kind).To(Equal("Deployment"))
		})

		It("should return last found ref when controller is missing", func() {
			controller := true
			pod := &corev1.Pod{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "Pod",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mypod",
					Namespace: "default",
					UID:       "uid-pod",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "ReplicaSet",
							Name:       "missing-rs",
							UID:        "uid-missing",
							Controller: &controller,
						},
					},
				},
			}

			c := fake.NewClientBuilder().WithScheme(sch).WithObjects(pod).Build()

			rootRef, err := utils.FindRootControllerRef(context.TODO(), c, pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(rootRef).NotTo(BeNil())
			Expect(rootRef.Name).To(Equal("missing-rs"))
			Expect(rootRef.Kind).To(Equal("ReplicaSet"))
		})
	})

	Describe("GetPodControllerRef", func() {
		It("should return nil when pod has no controller", func() {
			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mypod",
					Namespace: "default",
				},
			}

			c := fake.NewClientBuilder().WithScheme(sch).WithObjects(pod).Build()

			ref, err := utils.GetPodControllerRef(context.TODO(), c, pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(ref).To(BeNil())
		})

		It("should return deployment ref when pod is owned by replicaset owned by deployment", func() {
			controller := true
			deployment := &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mydeploy",
					Namespace: "default",
					UID:       "uid-deploy",
				},
			}

			rs := &appsv1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myrs",
					Namespace: "default",
					UID:       "uid-rs",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "Deployment",
							Name:       "mydeploy",
							UID:        deployment.UID,
							Controller: &controller,
						},
					},
				},
			}

			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mypod",
					Namespace: "default",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "apps/v1",
							Kind:       "ReplicaSet",
							Name:       "myrs",
							UID:        rs.UID,
							Controller: &controller,
						},
					},
				},
			}

			c := fake.NewClientBuilder().WithScheme(sch).WithObjects(pod, rs, deployment).Build()

			ref, err := utils.GetPodControllerRef(context.TODO(), c, pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(ref).NotTo(BeNil())
			Expect(ref.Name).To(Equal("mydeploy"))
			Expect(ref.Kind).To(Equal("Deployment"))
		})

		It("should return cronjob ref when pod is owned by job owned by cronjob", func() {
			controller := true
			cronjob := &batchv1.CronJob{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mycronjob",
					Namespace: "default",
					UID:       "uid-cronjob",
				},
			}

			job := &batchv1.Job{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "myjob",
					Namespace: "default",
					UID:       "uid-job",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "batch/v1",
							Kind:       "CronJob",
							Name:       "mycronjob",
							UID:        cronjob.UID,
							Controller: &controller,
						},
					},
				},
			}

			pod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "mypod",
					Namespace: "default",
					OwnerReferences: []metav1.OwnerReference{
						{
							APIVersion: "batch/v1",
							Kind:       "Job",
							Name:       "myjob",
							UID:        job.UID,
							Controller: &controller,
						},
					},
				},
			}

			c := fake.NewClientBuilder().WithScheme(sch).WithObjects(pod, job, cronjob).Build()

			ref, err := utils.GetPodControllerRef(context.TODO(), c, pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(ref).NotTo(BeNil())
			Expect(ref.Name).To(Equal("mycronjob"))
			Expect(ref.Kind).To(Equal("CronJob"))
		})
	})
})
