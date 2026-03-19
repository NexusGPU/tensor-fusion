package indexallocator

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

var _ = Describe("IndexAllocator", func() {
	var (
		allocator *IndexAllocator
		ctx       context.Context
	)

	BeforeEach(func() {
		scheme := runtime.NewScheme()
		Expect(v1.AddToScheme(scheme)).To(Succeed())
		client := fake.NewClientBuilder().WithScheme(scheme).Build()

		ctx = context.Background()
		var err error
		allocator, err = NewIndexAllocator(ctx, client)
		Expect(err).NotTo(HaveOccurred())
		allocator.IsLeader = true
	})

	Describe("AssignIndex", func() {
		It("should assign first index as 1", func() {
			index1, err := allocator.AssignIndex("pod-1")
			Expect(err).NotTo(HaveOccurred())
			Expect(index1).To(Equal(1))
		})

		It("should assign second index as 2", func() {
			_, err := allocator.AssignIndex("pod-1")
			Expect(err).NotTo(HaveOccurred())

			index2, err := allocator.AssignIndex("pod-2")
			Expect(err).NotTo(HaveOccurred())
			Expect(index2).To(Equal(2))
		})

		It("should assign indices in ascending order", func() {
			for i := 1; i <= 10; i++ {
				index, err := allocator.AssignIndex("pod-" + string(rune(i)))
				Expect(err).NotTo(HaveOccurred())
				Expect(index).To(Equal(i), "index should be assigned in ascending order")
			}
		})
	})

	Describe("AssignIndex incremental order", func() {
		It("should assign indices in incremental order (1, 2, 3, ...)", func() {
			expectedIndex := 1
			for i := 0; i < 20; i++ {
				index, err := allocator.AssignIndex("pod-" + string(rune(i)))
				Expect(err).NotTo(HaveOccurred())
				Expect(index).To(Equal(expectedIndex), "index should be assigned in ascending order")
				expectedIndex++
			}
		})
	})

	Describe("WrapAround", func() {
		It("should wrap around from max index to 1", func() {
			// Max index is IndexModLength * IndexKeyLength = 8 * 16 = 128
			maxIndex := constants.IndexModLength * constants.IndexKeyLength
			// Assign indices until we reach maxIndex (128)
			for i := 1; i <= maxIndex; i++ {
				index, err := allocator.AssignIndex("pod-" + string(rune(i)))
				Expect(err).NotTo(HaveOccurred())
				Expect(index).To(Equal(i))
			}

			// Next assignment should wrap around to 1
			index, err := allocator.AssignIndex("pod-wrap")
			Expect(err).NotTo(HaveOccurred())
			Expect(index).To(Equal(1), "index should wrap around from 128 to 1")
		})
	})

	Describe("AsyncCheckNodeIndexAvailableAndAssign", func() {
		It("should patch the pod index annotation when annotations already exist", func() {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "worker-pod",
					Namespace: "default",
					Annotations: map[string]string{
						"existing": "value",
					},
				},
				Spec: v1.PodSpec{
					NodeName: "node-a",
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			}
			Expect(allocator.Client.Create(ctx, pod)).To(Succeed())

			allocator.SetReady()
			allocator.AsyncCheckNodeIndexAvailableAndAssign(pod, 3)

			Eventually(func(g Gomega) {
				updated := &v1.Pod{}
				g.Expect(allocator.Client.Get(ctx, types.NamespacedName{
					Namespace: pod.Namespace,
					Name:      pod.Name,
				}, updated)).To(Succeed())
				g.Expect(updated.Annotations).To(HaveKeyWithValue(constants.PodIndexAnnotation, "3"))
				g.Expect(updated.Annotations).To(HaveKeyWithValue("existing", "value"))
			}).Should(Succeed())
		})

		It("should patch the pod index annotation when annotations are nil", func() {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "worker-pod-no-annotations",
					Namespace: "default",
				},
				Spec: v1.PodSpec{
					NodeName: "node-a",
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			}
			Expect(allocator.Client.Create(ctx, pod)).To(Succeed())

			allocator.SetReady()
			allocator.AsyncCheckNodeIndexAvailableAndAssign(pod, 7)

			Eventually(func(g Gomega) {
				updated := &v1.Pod{}
				g.Expect(allocator.Client.Get(ctx, types.NamespacedName{
					Namespace: pod.Namespace,
					Name:      pod.Name,
				}, updated)).To(Succeed())
				g.Expect(updated.Annotations).To(HaveKeyWithValue(constants.PodIndexAnnotation, "7"))
			}).Should(Succeed())
		})
	})

	Describe("Index occupancy tracking", func() {
		It("should remove an occupied index after the pod reaches running state", func() {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "worker-pod-running",
					Namespace: "default",
					Labels: map[string]string{
						constants.LabelComponent: constants.ComponentWorker,
					},
				},
				Spec: v1.PodSpec{
					NodeName: "node-a",
					Containers: []v1.Container{{
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceName(constants.PodIndexAnnotation + constants.PodIndexDelimiter + "0"): resourceMustParse("4"),
							},
						},
					}},
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			}

			allocator.SetReady()
			Expect(allocator.CheckNodeIndexAndTryOccupy(pod, 4)).To(BeTrue())

			pod.Status.Phase = v1.PodRunning
			allocator.ReconcileLockState(pod)

			allocator.storeMutex.RLock()
			defer allocator.storeMutex.RUnlock()
			Expect(allocator.nodeIndexQueue["node-a"]).NotTo(HaveKey(4))
			Expect(allocator.podIndexMap).NotTo(HaveKey(types.NamespacedName{Namespace: "default", Name: "worker-pod-running"}))
		})

		It("should reclaim a stale occupied index for a new pending pod", func() {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "worker-pod-stale",
					Namespace: "default",
					Labels: map[string]string{
						constants.LabelComponent: constants.ComponentWorker,
					},
				},
				Spec: v1.PodSpec{
					NodeName: "node-a",
					Containers: []v1.Container{{
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceName(constants.PodIndexAnnotation + constants.PodIndexDelimiter + "0"): resourceMustParse("3"),
							},
						},
					}},
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			}
			Expect(allocator.Client.Create(ctx, pod)).To(Succeed())

			stalePod := types.NamespacedName{Namespace: "default", Name: "stale-worker"}
			allocator.nodeIndexQueue["node-a"] = map[int]types.NamespacedName{3: stalePod}
			allocator.podIndexMap[stalePod] = indexIdentifier{nodeName: "node-a", index: 3}

			allocator.SetReady()
			allocator.ReconcileLockState(pod)

			Eventually(func(g Gomega) {
				updated := &v1.Pod{}
				g.Expect(allocator.Client.Get(ctx, types.NamespacedName{
					Namespace: pod.Namespace,
					Name:      pod.Name,
				}, updated)).To(Succeed())
				g.Expect(updated.Annotations).To(HaveKeyWithValue(constants.PodIndexAnnotation, "3"))
			}).Should(Succeed())
		})

		It("should reclaim an index left behind by a running pod", func() {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "worker-pod-after-running-occupant",
					Namespace: "default",
					Labels: map[string]string{
						constants.LabelComponent: constants.ComponentWorker,
					},
				},
				Spec: v1.PodSpec{
					NodeName: "node-a",
					Containers: []v1.Container{{
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceName(constants.PodIndexAnnotation + constants.PodIndexDelimiter + "0"): resourceMustParse("6"),
							},
						},
					}},
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			}
			Expect(allocator.Client.Create(ctx, pod)).To(Succeed())

			runningOccupant := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "running-worker",
					Namespace: "default",
				},
				Spec: v1.PodSpec{
					NodeName: "node-a",
				},
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
				},
			}
			Expect(allocator.Client.Create(ctx, runningOccupant)).To(Succeed())

			runningMeta := types.NamespacedName{Namespace: runningOccupant.Namespace, Name: runningOccupant.Name}
			allocator.nodeIndexQueue["node-a"] = map[int]types.NamespacedName{6: runningMeta}
			allocator.podIndexMap[runningMeta] = indexIdentifier{nodeName: "node-a", index: 6}

			allocator.SetReady()
			allocator.ReconcileLockState(pod)

			Eventually(func(g Gomega) {
				updated := &v1.Pod{}
				g.Expect(allocator.Client.Get(ctx, types.NamespacedName{
					Namespace: pod.Namespace,
					Name:      pod.Name,
				}, updated)).To(Succeed())
				g.Expect(updated.Annotations).To(HaveKeyWithValue(constants.PodIndexAnnotation, "6"))
			}).Should(Succeed())
		})

		It("should start async index patching when queue already points to the same pending pod", func() {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "worker-pod-existing-queue",
					Namespace: "default",
					Labels: map[string]string{
						constants.LabelComponent: constants.ComponentWorker,
					},
				},
				Spec: v1.PodSpec{
					NodeName: "node-a",
					Containers: []v1.Container{{
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceName(constants.PodIndexAnnotation + constants.PodIndexDelimiter + "0"): resourceMustParse("5"),
							},
						},
					}},
				},
				Status: v1.PodStatus{
					Phase: v1.PodPending,
				},
			}
			Expect(allocator.Client.Create(ctx, pod)).To(Succeed())

			podMeta := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
			allocator.nodeIndexQueue["node-a"] = map[int]types.NamespacedName{5: podMeta}

			allocator.SetReady()
			allocator.ReconcileLockState(pod)

			Eventually(func(g Gomega) {
				updated := &v1.Pod{}
				g.Expect(allocator.Client.Get(ctx, podMeta, updated)).To(Succeed())
				g.Expect(updated.Annotations).To(HaveKeyWithValue(constants.PodIndexAnnotation, "5"))
			}).Should(Succeed())
		})
	})
})

func resourceMustParse(value string) resource.Quantity {
	return resource.MustParse(value)
}
