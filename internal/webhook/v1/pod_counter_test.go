package v1

import (
	"context"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var _ = Describe("TensorFusionPodCounter", func() {
	var (
		counter *TensorFusionPodCounter
		ctx     context.Context
		pod     *corev1.Pod
		owner   *appsv1.Deployment
	)

	BeforeEach(func() {
		ctx = context.Background()
		counter = &TensorFusionPodCounter{Client: k8sClient}
		pod = &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test-pod",
				Namespace: "default",
				Annotations: map[string]string{
					constants.TensorFusionPodCounterKeyAnnotation: "my-key",
				},
				Labels: map[string]string{
					"pod-template-hash": "hash123",
				},
				OwnerReferences: []metav1.OwnerReference{{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       "owner",
					Controller: ptr.To(true),
				}},
			},
		}
		owner = &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "owner",
				Namespace:   "default",
				Annotations: map[string]string{},
			},
			Spec: appsv1.DeploymentSpec{
				Selector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"app": "dummy"},
				},
				Template: corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{"app": "dummy"},
					},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:    "dummy",
							Image:   "busybox",
							Command: []string{"sleep", "3600"},
						}},
					},
				},
			},
		}
		Expect(k8sClient.Create(ctx, owner)).To(Succeed())
	})

	AfterEach(func() {
		Expect(k8sClient.Delete(ctx, owner)).To(Succeed())
	})

	It("should get 0 if annotation not set", func() {
		val, _, err := counter.Get(ctx, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(val).To(Equal(int32(0)))
	})

	It("should increase and get the counter", func() {
		Expect(counter.Increase(ctx, pod)).To(Succeed())
		val, _, err := counter.Get(ctx, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(val).To(Equal(int32(1)))
	})

	It("should increase twice and get the correct value", func() {
		Expect(counter.Increase(ctx, pod)).To(Succeed())
		Expect(counter.Increase(ctx, pod)).To(Succeed())
		val, _, err := counter.Get(ctx, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(val).To(Equal(int32(2)))
	})

	It("should decrease the counter", func() {
		Expect(counter.Increase(ctx, pod)).To(Succeed())
		Expect(counter.Decrease(ctx, pod)).To(Succeed())
		val, _, err := counter.Get(ctx, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(val).To(Equal(int32(0)))
	})

	It("should not go below zero", func() {
		Expect(counter.Decrease(ctx, pod)).To(Succeed())
		val, _, err := counter.Get(ctx, pod)
		Expect(err).NotTo(HaveOccurred())
		Expect(val).To(Equal(int32(0)))
	})

	It("should return error if owner not found", func() {
		pod.OwnerReferences[0].Name = "notfound"
		_, _, err := counter.Get(ctx, pod)
		Expect(err).To(HaveOccurred())
	})

	It("should delete annotation key when count reaches zero", func() {
		// Increase
		Expect(counter.Increase(ctx, pod)).To(Succeed())
		// Decrease to 0
		Expect(counter.Decrease(ctx, pod)).To(Succeed())

		// Get owner object
		ownerRef := getControllerOwnerRef(pod)
		ownerObj := &unstructured.Unstructured{}
		ownerObj.SetAPIVersion(ownerRef.APIVersion)
		ownerObj.SetKind(ownerRef.Kind)
		objKey := client.ObjectKey{Name: ownerRef.Name, Namespace: pod.Namespace}
		Expect(counter.Client.Get(ctx, objKey, ownerObj)).To(Succeed())
		annotations := ownerObj.GetAnnotations()
		key := getOrGenerateKey(pod)
		_, exists := annotations[key]
		Expect(exists).To(BeFalse())
	})
})
