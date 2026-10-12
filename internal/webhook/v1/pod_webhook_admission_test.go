package v1

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/indexallocator"
	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	admissionv1 "k8s.io/api/admission/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// These specs go through the API server and the registered webhook
// configuration, so they see what a client of the cluster sees.

func newAdmissionTestNamespace(ctx context.Context, prefix string) string {
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{GenerateName: prefix + "-"}}
	Expect(k8sClient.Create(ctx, ns)).To(Succeed())
	return ns.Name
}

func newAdmissionTestReplicaSet(ctx context.Context, namespace, name string) *appsv1.ReplicaSet {
	rs := &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name},
		Spec: appsv1.ReplicaSetSpec{
			// envtest runs no kube-controller-manager, so nothing creates pods
			// for this ReplicaSet except the spec itself.
			Replicas: ptr.To(int32(0)),
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": name}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": name}},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: "main", Image: "test-image"}},
				},
			},
		},
	}
	Expect(k8sClient.Create(ctx, rs)).To(Succeed())
	return rs
}

func newAdmissionTestPod(namespace string, owner *appsv1.ReplicaSet, enabledReplicas string) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Labels: map[string]string{
				constants.TensorFusionEnabledLabelKey: constants.TrueStringValue,
			},
			Annotations: map[string]string{
				constants.GpuPoolKey:              "mock",
				constants.TFLOPSRequestAnnotation: "10",
				constants.VRAMRequestAnnotation:   "1Gi",
				// keep the pod free of a priority class the test cluster lacks
				constants.QoSLevelAnnotation: string(tfv1.QoSMedium),
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{Name: "main", Image: "test-image"}},
		},
	}
	if owner == nil {
		pod.Name = "standalone"
		return pod
	}
	pod.GenerateName = owner.Name + "-"
	pod.Labels["app"] = owner.Name
	pod.Labels["pod-template-hash"] = "abc123"
	pod.OwnerReferences = []metav1.OwnerReference{{
		APIVersion: "apps/v1",
		Kind:       "ReplicaSet",
		Name:       owner.Name,
		UID:        owner.UID,
		Controller: ptr.To(true),
	}}
	if enabledReplicas != "" {
		pod.Annotations[constants.TensorFusionEnabledReplicasAnnotation] = enabledReplicas
	}
	return pod
}

func listAdmissionTestWorkloads(ctx context.Context, namespace string) []string {
	list := &tfv1.TensorFusionWorkloadList{}
	Expect(k8sClient.List(ctx, list, client.InNamespace(namespace))).To(Succeed())
	names := make([]string, 0, len(list.Items))
	for i := range list.Items {
		names = append(names, list.Items[i].Name)
	}
	return names
}

func ownerCounterAnnotations(ctx context.Context, rs *appsv1.ReplicaSet) map[string]string {
	current := &appsv1.ReplicaSet{}
	Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(rs), current)).To(Succeed())
	counters := map[string]string{}
	for k, v := range current.Annotations {
		// The time of the last reservation is kept next to the counters; it is
		// not one of them.
		if k == PodCounterLastReserveAnnotation {
			continue
		}
		counters[k] = v
	}
	return counters
}

var _ = Describe("Pod webhook server-side dry-run", func() {
	It("returns the mutated pod without creating a workload or counting the pod", func(ctx SpecContext) {
		ns := newAdmissionTestNamespace(ctx, "dryrun")
		rs := newAdmissionTestReplicaSet(ctx, ns, "dry")

		By("dry-run create of a standalone pod")
		standalone := newAdmissionTestPod(ns, nil, "")
		Expect(k8sClient.Create(ctx, standalone, client.DryRunAll)).To(Succeed())
		Expect(standalone.Labels).To(HaveKey(constants.LabelComponent), "dry-run returns the mutated pod")
		Expect(standalone.Labels).To(HaveKeyWithValue(constants.GpuPoolKey, "mock"))
		Expect(standalone.Annotations).To(HaveKeyWithValue(constants.SetPendingOwnedWorkloadAnnotation, "standalone"))

		By("dry-run create of a pod limited by enabled-replicas")
		limited := newAdmissionTestPod(ns, rs, "1")
		Expect(k8sClient.Create(ctx, limited, client.DryRunAll)).To(Succeed())
		Expect(limited.Labels).To(HaveKey(constants.LabelComponent))
		Expect(limited.Annotations).To(HaveKey(constants.TensorFusionPodCounterKeyAnnotation))

		Consistently(func() []string {
			return listAdmissionTestWorkloads(ctx, ns)
		}, 2*time.Second, 200*time.Millisecond).Should(BeEmpty(), "dry-run must not create a TensorFusionWorkload")
		Expect(ownerCounterAnnotations(ctx, rs)).To(BeEmpty(), "dry-run must not count the pod against enabled-replicas")

		By("the same creates without dry-run")
		Expect(k8sClient.Create(ctx, newAdmissionTestPod(ns, nil, ""))).To(Succeed())
		real := newAdmissionTestPod(ns, rs, "1")
		Expect(k8sClient.Create(ctx, real)).To(Succeed())
		Expect(real.Labels).To(HaveKey(constants.LabelComponent))
		Expect(listAdmissionTestWorkloads(ctx, ns)).To(ConsistOf("standalone", "dry"))
		key := real.Annotations[constants.TensorFusionPodCounterKeyAnnotation]
		Expect(key).NotTo(BeEmpty())
		Expect(ownerCounterAnnotations(ctx, rs)).To(HaveKeyWithValue(key, "1"))
	})

	It("leaves port, index and counter state untouched when the handler runs with dryRun", func(ctx SpecContext) {
		ns := newAdmissionTestNamespace(ctx, "dryrun-alloc")
		rs := newAdmissionTestReplicaSet(ctx, ns, "alloc")

		indexAllocator, err := indexallocator.NewIndexAllocator(ctx, k8sClient)
		Expect(err).NotTo(HaveOccurred())
		indexAllocator.IsLeader = true
		portAllocator := &portallocator.PortAllocator{
			IsLeader:              true,
			PortRangeStartCluster: 42000,
			PortRangeEndCluster:   62000,
			BitmapCluster:         make([]uint64, (62000-42000)/64+1),
		}
		scheme := runtime.NewScheme()
		Expect(corev1.AddToScheme(scheme)).To(Succeed())
		mutator := &TensorFusionPodMutator{
			Client:         k8sClient,
			decoder:        admission.NewDecoder(scheme),
			portAllocator:  portAllocator,
			indexAllocator: indexAllocator,
		}

		pod := newAdmissionTestPod(ns, rs, "1")
		pod.Annotations[constants.IsLocalGPUAnnotation] = constants.TrueStringValue
		pod.Labels[constants.GenHostPortLabel] = constants.GenHostPortLabelValue
		pod.Labels[constants.GenHostPortNameLabel] = "http"
		pod.Spec.Containers[0].Ports = []corev1.ContainerPort{{Name: "http", ContainerPort: 8080}}
		raw, err := json.Marshal(pod)
		Expect(err).NotTo(HaveOccurred())
		request := func(dryRun bool) admission.Request {
			return admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
				Object:    runtime.RawExtension{Raw: raw},
				Operation: admissionv1.Create,
				Namespace: ns,
				DryRun:    ptr.To(dryRun),
			}}
		}
		usedPorts := func() int {
			used := 0
			for _, word := range portAllocator.BitmapCluster {
				for ; word != 0; word &= word - 1 {
					used++
				}
			}
			return used
		}
		patchPaths := func(resp admission.Response) []string {
			paths := make([]string, 0, len(resp.Patches))
			for _, p := range resp.Patches {
				paths = append(paths, p.Path)
			}
			return paths
		}
		counterPath := "/metadata/annotations/" + utils.EscapeJSONPointer(constants.TensorFusionPodCounterKeyAnnotation)
		portPath := "/metadata/annotations/" + utils.EscapeJSONPointer(constants.GenPortNumberAnnotation)

		dry := mutator.Handle(ctx, request(true))
		Expect(dry.Allowed).To(BeTrue(), "%v", dry.Result)
		Expect(patchPaths(dry)).To(ContainElement(counterPath), "dry-run reports the injection it would do")
		Expect(patchPaths(dry)).NotTo(ContainElement(portPath), "no host port is reserved for a dry-run")
		Expect(usedPorts()).To(BeZero())
		Expect(listAdmissionTestWorkloads(ctx, ns)).To(BeEmpty())
		Expect(ownerCounterAnnotations(ctx, rs)).To(BeEmpty())

		real := mutator.Handle(ctx, request(false))
		Expect(real.Allowed).To(BeTrue(), "%v", real.Result)
		Expect(patchPaths(real)).To(ContainElements(counterPath, portPath))
		Expect(usedPorts()).To(Equal(1))
		Expect(listAdmissionTestWorkloads(ctx, ns)).To(ConsistOf("alloc"))
		Expect(ownerCounterAnnotations(ctx, rs)).To(HaveLen(1))
		// The real admission took the first index, so the dry-run before it
		// did not consume one.
		next, err := indexAllocator.AssignIndex("probe")
		Expect(err).NotTo(HaveOccurred())
		Expect(next).To(Equal(2))

		By("a dry-run against a full owner reports that the pod would be left alone")
		full := mutator.Handle(ctx, request(true))
		Expect(full.Allowed).To(BeTrue())
		Expect(full.Patches).To(BeEmpty())
	})
})

var _ = Describe("Pod webhook enabled-replicas under concurrent admission", func() {
	It("gives the slot back when the pod is not injected after all", func(ctx SpecContext) {
		ns := newAdmissionTestNamespace(ctx, "gray-release")
		rs := newAdmissionTestReplicaSet(ctx, ns, "release")

		// Rejected after the slot was taken: the pod asks for no GPU resources.
		rejected := newAdmissionTestPod(ns, rs, "1")
		delete(rejected.Annotations, constants.TFLOPSRequestAnnotation)
		delete(rejected.Annotations, constants.VRAMRequestAnnotation)
		err := k8sClient.Create(ctx, rejected)
		Expect(err).To(MatchError(ContainSubstring("tflops request is not set")))
		Expect(ownerCounterAnnotations(ctx, rs)).To(BeEmpty(), "a pod that was not injected must not keep a slot")

		injected := newAdmissionTestPod(ns, rs, "1")
		Expect(k8sClient.Create(ctx, injected)).To(Succeed())
		Expect(injected.Labels).To(HaveKey(constants.LabelComponent), "the slot is still there for the next pod")
	})

	It("injects exactly enabled-replicas pods of one owner", func(ctx SpecContext) {
		const (
			rounds  = 6
			pods    = 12
			enabled = 3
		)
		ns := newAdmissionTestNamespace(ctx, "gray")
		for round := range rounds {
			rs := newAdmissionTestReplicaSet(ctx, ns, fmt.Sprintf("gray-%d", round))

			var wg sync.WaitGroup
			errs := make([]error, pods)
			start := make(chan struct{})
			for i := range pods {
				wg.Add(1)
				go func() {
					defer wg.Done()
					defer GinkgoRecover()
					<-start
					errs[i] = k8sClient.Create(ctx, newAdmissionTestPod(ns, rs, fmt.Sprint(enabled)))
				}()
			}
			close(start)
			wg.Wait()
			for i := range errs {
				Expect(errs[i]).NotTo(HaveOccurred(), "round %d pod %d", round, i)
			}

			list := &corev1.PodList{}
			Expect(k8sClient.List(ctx, list, client.InNamespace(ns), client.MatchingLabels{"app": rs.Name})).To(Succeed())
			Expect(list.Items).To(HaveLen(pods))
			injected := 0
			var key string
			for i := range list.Items {
				if _, ok := list.Items[i].Labels[constants.LabelComponent]; ok {
					injected++
					key = list.Items[i].Annotations[constants.TensorFusionPodCounterKeyAnnotation]
				}
			}
			Expect(injected).To(Equal(enabled), "round %d: injected pods must equal enabled-replicas", round)
			Expect(ownerCounterAnnotations(ctx, rs)).To(HaveKeyWithValue(key, fmt.Sprint(enabled)), "round %d", round)
		}
	})
})
