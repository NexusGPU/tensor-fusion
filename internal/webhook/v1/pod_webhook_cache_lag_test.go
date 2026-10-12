package v1

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// The webhook reads TensorFusionWorkloads through the manager's informer
// cache, which is a moment behind the API server in both directions: a
// workload that was just created is not in it yet, and one that was just
// deleted still is. These specs run against the real API server and the real
// manager client, and hold the cache back for one workload so that the moment
// does not depend on timing.

// staleWorkloadCache is the manager's client with two switches for its reads
// of TensorFusionWorkloads. Everything else, writes included, goes to the
// wrapped client unchanged.
type staleWorkloadCache struct {
	client.Client

	mu sync.Mutex
	// notYetReceived are workloads the cache does not hold yet although they
	// may exist: reading one is a miss.
	notYetReceived map[client.ObjectKey]bool
	misses         map[client.ObjectKey]int
	// stillHeld are workloads the cache holds although they are gone. One is
	// served once; after that the cache has caught up.
	stillHeld map[client.ObjectKey]*tfv1.TensorFusionWorkload
}

func newStaleWorkloadCache(cached client.Client) *staleWorkloadCache {
	return &staleWorkloadCache{
		Client:         cached,
		notYetReceived: map[client.ObjectKey]bool{},
		misses:         map[client.ObjectKey]int{},
		stillHeld:      map[client.ObjectKey]*tfv1.TensorFusionWorkload{},
	}
}

func (c *staleWorkloadCache) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	workload, ok := obj.(*tfv1.TensorFusionWorkload)
	if !ok {
		return c.Client.Get(ctx, key, obj, opts...)
	}
	c.mu.Lock()
	held := c.stillHeld[key]
	delete(c.stillHeld, key)
	missing := c.notYetReceived[key]
	if held == nil && missing {
		c.misses[key]++
	}
	c.mu.Unlock()

	switch {
	case held != nil:
		held.DeepCopyInto(workload)
		return nil
	case missing:
		return apierrors.NewNotFound(tfv1.GroupVersion.WithResource("tensorfusionworkloads").GroupResource(), key.Name)
	}
	return c.Client.Get(ctx, key, obj, opts...)
}

func (c *staleWorkloadCache) holdBack(key client.ObjectKey) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.notYetReceived[key] = true
}

func (c *staleWorkloadCache) missesOf(key client.ObjectKey) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.misses[key]
}

func (c *staleWorkloadCache) keepOnce(workload *tfv1.TensorFusionWorkload) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.stillHeld[client.ObjectKeyFromObject(workload)] = workload.DeepCopy()
}

// podCreateAnswer is what the API server answered one pod create with.
type podCreateAnswer struct {
	code       int
	retryAfter string
}

// podCreateRecorder notes every answer to a pod create, including the ones the
// client repeats by itself and never shows to its caller.
type podCreateRecorder struct {
	next http.RoundTripper

	mu      sync.Mutex
	answers []podCreateAnswer
}

func (r *podCreateRecorder) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := r.next.RoundTrip(req)
	if err == nil && req.Method == http.MethodPost && strings.HasSuffix(req.URL.Path, "/pods") {
		r.mu.Lock()
		r.answers = append(r.answers, podCreateAnswer{code: resp.StatusCode, retryAfter: resp.Header.Get("Retry-After")})
		r.mu.Unlock()
	}
	return resp, err
}

func (r *podCreateRecorder) seen() []podCreateAnswer {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]podCreateAnswer{}, r.answers...)
}

// newRecordingClient is a client like k8sClient whose pod creates are recorded.
func newRecordingClient() (client.Client, *podCreateRecorder) {
	recorder := &podCreateRecorder{}
	config := rest.CopyConfig(cfg)
	config.Wrap(func(next http.RoundTripper) http.RoundTripper {
		recorder.next = next
		return recorder
	})
	c, err := client.New(config, client.Options{Scheme: k8sClient.Scheme()})
	Expect(err).NotTo(HaveOccurred())
	return c, recorder
}

func webhookTestWorkload(namespace, name string, owner *metav1.OwnerReference) *tfv1.TensorFusionWorkload {
	workload := &tfv1.TensorFusionWorkload{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:   namespace,
			Name:        name,
			Labels:      map[string]string{constants.GpuPoolKey: "mock"},
			Annotations: map[string]string{constants.WorkloadModeAnnotation: constants.WorkloadModeDynamic},
		},
		Spec: tfv1.WorkloadProfileSpec{PoolName: "mock"},
	}
	if owner != nil {
		workload.OwnerReferences = []metav1.OwnerReference{*owner}
	}
	return workload
}

func controllerOf(ctx context.Context, key client.ObjectKey) *metav1.OwnerReference {
	workload := &tfv1.TensorFusionWorkload{}
	Expect(k8sClient.Get(ctx, key, workload)).To(Succeed())
	return metav1.GetControllerOf(workload)
}

var _ = Describe("Pod webhook and a workload the cache is wrong about", func() {
	It("admits pods created together, and has a create repeated while the workload is on its way out", func(ctx SpecContext) {
		By("admitting all pods of one owner while the cache has not received the workload the first one created")
		const pods = 8
		ns := newAdmissionTestNamespace(ctx, "together")
		rs := newAdmissionTestReplicaSet(ctx, ns, "together")
		key := client.ObjectKey{Namespace: ns, Name: rs.Name}
		managerClient.holdBack(key)

		var wg sync.WaitGroup
		errs := make([]error, pods)
		admitted := make([]*corev1.Pod, pods)
		start := make(chan struct{})
		for i := range pods {
			wg.Add(1)
			go func() {
				defer wg.Done()
				defer GinkgoRecover()
				<-start
				admitted[i] = newAdmissionTestPod(ns, rs, "")
				errs[i] = k8sClient.Create(ctx, admitted[i])
			}()
		}
		close(start)
		wg.Wait()
		for i := range errs {
			Expect(errs[i]).NotTo(HaveOccurred(), "pod %d", i)
			Expect(admitted[i].Labels).To(HaveKey(constants.LabelComponent), "pod %d", i)
		}
		Expect(listAdmissionTestWorkloads(ctx, ns)).To(ConsistOf(rs.Name))
		Expect(controllerOf(ctx, key).UID).To(Equal(rs.UID))
		// Every admission missed the workload in the cache exactly once: the
		// read that follows the refused create did not go through the cache.
		Expect(managerClient.missesOf(key)).To(Equal(pods))

		By("refusing a pod, with a status its creator repeats by itself, while the workload belongs to a replaced owner")
		ns = newAdmissionTestNamespace(ctx, "replaced-owner")
		rs = newAdmissionTestReplicaSet(ctx, ns, "replaced")
		key = client.ObjectKey{Namespace: ns, Name: rs.Name}
		// The workload of an owner of the same kind and name that has been
		// deleted: the garbage collector, which envtest does not run, is about
		// to remove it.
		previous := webhookTestWorkload(ns, rs.Name, &metav1.OwnerReference{
			APIVersion: "apps/v1", Kind: "ReplicaSet", Name: rs.Name, UID: "uid-of-the-previous-owner",
			Controller: ptr.To(true),
		})
		Expect(k8sClient.Create(ctx, previous)).To(Succeed())

		creator, recorder := newRecordingClient()
		pod := newAdmissionTestPod(ns, rs, "")
		created := make(chan error, 1)
		go func() {
			defer GinkgoRecover()
			created <- creator.Create(ctx, pod)
		}()
		Eventually(recorder.seen).WithTimeout(10 * time.Second).ShouldNot(BeEmpty())
		Expect(recorder.seen()[0]).To(Equal(podCreateAnswer{code: http.StatusServiceUnavailable, retryAfter: "1"}))
		Expect(controllerOf(ctx, key).UID).To(BeEquivalentTo("uid-of-the-previous-owner"), "the old workload is left alone")

		By("admitting the repeated create once the old workload is gone")
		Expect(k8sClient.Delete(ctx, previous)).To(Succeed())
		Eventually(created).WithTimeout(30 * time.Second).Should(Receive(Succeed()))
		answers := recorder.seen()
		Expect(answers[len(answers)-1].code).To(Equal(http.StatusCreated))
		Expect(pod.Labels).To(HaveKey(constants.LabelComponent))
		Expect(controllerOf(ctx, key).UID).To(Equal(rs.UID), "the pod got a workload of its own owner")

		By("having the create repeated when the workload is gone by the time it is written")
		ns = newAdmissionTestNamespace(ctx, "vanished")
		// The workload of an earlier pod of the same name, deleted a moment ago:
		// the API server no longer has it, the cache still does.
		gone := webhookTestWorkload(ns, "standalone", nil)
		gone.UID = "uid-of-the-deleted-workload"
		gone.ResourceVersion = "1"
		managerClient.keepOnce(gone)

		creator, recorder = newRecordingClient()
		pod = newAdmissionTestPod(ns, nil, "")
		Expect(creator.Create(ctx, pod)).To(Succeed(), "the creator sees no failure: its client repeated the request")
		answers = recorder.seen()
		Expect(answers).To(HaveLen(2))
		Expect(answers[0]).To(Equal(podCreateAnswer{code: http.StatusServiceUnavailable, retryAfter: "1"}))
		Expect(answers[1].code).To(Equal(http.StatusCreated))
		Expect(pod.Labels).To(HaveKey(constants.LabelComponent))
		Expect(listAdmissionTestWorkloads(ctx, ns)).To(ConsistOf("standalone"), "the repeated create made a new workload")
	})
})
