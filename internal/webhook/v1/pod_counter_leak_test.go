package v1

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

type leakTestClock struct{ at time.Time }

func (c *leakTestClock) now() time.Time { return c.at }

// newLeakTestClient holds the owner "rs" with the given annotations and the
// given pods.
func newLeakTestClient(
	t *testing.T, ownerAnnotations map[string]string, funcs *interceptor.Funcs, pods ...*corev1.Pod,
) client.Client {
	t.Helper()
	scheme := k8sruntime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, appsv1.AddToScheme(scheme))
	rs := &appsv1.ReplicaSet{ObjectMeta: metav1.ObjectMeta{
		Namespace: "ns", Name: "rs", UID: "rs-uid", Annotations: ownerAnnotations,
	}}
	builder := fake.NewClientBuilder().WithScheme(scheme).WithObjects(rs)
	for _, pod := range pods {
		builder = builder.WithObjects(pod)
	}
	if funcs != nil {
		builder = builder.WithInterceptorFuncs(*funcs)
	}
	return builder.Build()
}

// makeSlotHolder is a pod as the webhook leaves it after injecting it for the
// given counter key.
func makeSlotHolder(name, key string) *corev1.Pod {
	pod := makeCounterPod(name)
	pod.Labels = map[string]string{constants.LabelComponent: constants.ComponentClient}
	pod.Annotations[constants.TensorFusionPodCounterKeyAnnotation] = key
	return pod
}

func leakTestOwnerAnnotations(t *testing.T, c client.Client) map[string]string {
	t.Helper()
	got := &appsv1.ReplicaSet{}
	require.NoError(t, c.Get(context.Background(), client.ObjectKey{Namespace: "ns", Name: "rs"}, got))
	return got.Annotations
}

func longAgo() string {
	return formatReserveTime(time.Now().Add(-time.Hour))
}

// The gap: a pod is admitted, its slot is taken, and the API server never
// stores it. No pod exists whose deletion would give the slot back.
func TestReleaseLeakedSlotsAfterGracePeriodAndNotBefore(t *testing.T) {
	ctx := context.Background()
	reservedAt := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
	clock := &leakTestClock{at: reservedAt}
	c := newLeakTestClient(t, nil, nil)
	counter := &TensorFusionPodCounter{Client: c, now: clock.now}

	// Three pods are admitted, two of them are stored.
	for range 3 {
		ok, _, err := counter.Reserve(ctx, makeCounterPod("admitted"), 3)
		require.NoError(t, err)
		require.True(t, ok)
	}
	stored := makeSlotHolder("pod-0", reserveTestKey)
	require.NoError(t, c.Create(ctx, stored))
	require.NoError(t, c.Create(ctx, makeSlotHolder("pod-1", reserveTestKey)))
	assert.Equal(t, formatReserveTime(reservedAt), leakTestOwnerAnnotations(t, c)[PodCounterLastReserveAnnotation],
		"Reserve records when the newest slot was taken")
	ok, _, err := counter.Reserve(ctx, makeCounterPod("replacement"), 3)
	require.NoError(t, err)
	assert.False(t, ok, "the slot of the pod that was never stored is still taken")

	clock.at = reservedAt.Add(time.Minute)
	recheck, err := counter.ReleaseLeakedSlots(ctx, stored)
	require.NoError(t, err)
	assert.Equal(t, time.Minute+leakRecheckSlack, recheck, "asks to look again when the grace period is over")
	assert.Equal(t, "3", reserveTestCount(t, c), "nothing is released inside the grace period")

	clock.at = reservedAt.Add(PodCounterLeakGracePeriod)
	recheck, err = counter.ReleaseLeakedSlots(ctx, stored)
	require.NoError(t, err)
	assert.Equal(t, leakRecheckSlack, recheck)
	assert.Equal(t, "3", reserveTestCount(t, c), "nothing is released at the very end of the grace period")

	clock.at = reservedAt.Add(PodCounterLeakGracePeriod + leakRecheckSlack)
	recheck, err = counter.ReleaseLeakedSlots(ctx, stored)
	require.NoError(t, err)
	assert.Zero(t, recheck)
	assert.Equal(t, "2", reserveTestCount(t, c), "the count is lowered to the pods that exist")

	// The slot is usable again. The pod it is taken for is not stored yet: a
	// correction that runs now sees a count above the pods, and must wait.
	ok, _, err = counter.Reserve(ctx, makeCounterPod("replacement"), 3)
	require.NoError(t, err)
	require.True(t, ok, "the released slot can be taken")
	for _, later := range []time.Duration{0, time.Minute, PodCounterLeakGracePeriod} {
		clock.at = reservedAt.Add(PodCounterLeakGracePeriod + leakRecheckSlack + later)
		recheck, err = counter.ReleaseLeakedSlots(ctx, stored)
		require.NoError(t, err)
		assert.Equal(t, PodCounterLeakGracePeriod-later+leakRecheckSlack, recheck)
		assert.Equal(t, "3", reserveTestCount(t, c), "a fresh reservation is not released, %s after it", later)
	}
}

// A Reserve of another operator replica lands while the correction is under
// way: after the pods were listed, or between the correction's read of the
// owner and its write. The pod of that reservation is not stored yet, so the
// list does not contain it; only the reservation time protects its slot.
func TestReleaseLeakedSlotsKeepsAReservationMadeDuringTheCorrection(t *testing.T) {
	const limit = 4
	reserveOnce := func(t *testing.T, done *atomic.Bool, c client.Client) {
		if !done.CompareAndSwap(false, true) {
			return
		}
		other := &TensorFusionPodCounter{Client: c, locks: &ownerLocks{}}
		ok, _, err := other.Reserve(context.Background(), makeCounterPod("in-flight"), limit)
		require.NoError(t, err)
		require.True(t, ok)
	}
	cases := map[string]func(t *testing.T, done *atomic.Bool) interceptor.Funcs{
		"after the pods were listed": func(t *testing.T, done *atomic.Bool) interceptor.Funcs {
			return interceptor.Funcs{
				List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					err := c.List(ctx, list, opts...)
					if _, live := list.(*unstructured.UnstructuredList); live {
						reserveOnce(t, done, c)
					}
					return err
				},
			}
		},
		"between the read of the owner and the write": func(t *testing.T, done *atomic.Bool) interceptor.Funcs {
			return interceptor.Funcs{
				Update: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.UpdateOption) error {
					reserveOnce(t, done, c)
					return c.Update(ctx, obj, opts...)
				},
			}
		},
	}
	for name, funcs := range cases {
		t.Run(name, func(t *testing.T) {
			var done atomic.Bool
			f := funcs(t, &done)
			holder := makeSlotHolder("pod-0", reserveTestKey)
			// Three slots taken long ago, one pod: two slots have leaked.
			c := newLeakTestClient(t, map[string]string{
				reserveTestKey:                  "3",
				PodCounterLastReserveAnnotation: longAgo(),
			}, &f, holder)
			counter := &TensorFusionPodCounter{Client: c}

			recheck, err := counter.ReleaseLeakedSlots(context.Background(), holder)
			require.NoError(t, err)
			require.True(t, done.Load(), "the reservation happened during the correction")
			assert.Equal(t, "4", reserveTestCount(t, c), "the reservation made meanwhile is kept, and nothing is released")
			assert.Equal(t, PodCounterLeakGracePeriod+leakRecheckSlack, recheck)
		})
	}
}

// The correction and admissions of the same owner run at the same time. The
// owner has limit slots taken and fewer pods; admissions keep trying while the
// corrections run and are never stored during the test, which is the worst
// case for the correction: every slot they take has no pod yet.
func TestReleaseLeakedSlotsRacingReserveNeverExceedsLimit(t *testing.T) {
	const limit, held, reservers, correctors = 4, 2, 8, 4
	for _, mode := range []string{"one process", "several replicas"} {
		t.Run(mode, func(t *testing.T) {
			for round := range 25 {
				holders := []*corev1.Pod{
					makeSlotHolder("pod-0", reserveTestKey),
					makeSlotHolder("pod-1", reserveTestKey),
				}
				c := newLeakTestClient(t, map[string]string{
					reserveTestKey:                  "4",
					PodCounterLastReserveAnnotation: longAgo(),
				}, nil, holders...)
				processLocks := &ownerLocks{}
				newCounter := func() *TensorFusionPodCounter {
					if mode == "one process" {
						return &TensorFusionPodCounter{Client: c, locks: processLocks}
					}
					return &TensorFusionPodCounter{Client: c, locks: &ownerLocks{}}
				}

				var reserved, correcting atomic.Int32
				correcting.Store(correctors)
				start := make(chan struct{})
				var wg sync.WaitGroup
				for range correctors {
					wg.Add(1)
					go func() {
						defer wg.Done()
						defer correcting.Add(-1)
						counter := newCounter()
						<-start
						for range 3 {
							_, err := counter.ReleaseLeakedSlots(context.Background(), holders[0])
							assert.NoError(t, err)
						}
					}()
				}
				for range reservers {
					wg.Add(1)
					go func() {
						defer wg.Done()
						counter := newCounter()
						<-start
						for {
							// One more attempt after the corrections are over.
							last := correcting.Load() == 0
							ok, _, err := counter.Reserve(context.Background(), makeCounterPod("in-flight"), limit)
							if !assert.NoError(t, err) {
								return
							}
							if ok {
								reserved.Add(1)
								return
							}
							if last {
								return
							}
							runtime.Gosched()
						}
					}()
				}
				close(start)
				wg.Wait()

				assert.Equal(t, int32(limit-held), reserved.Load(),
					"round %d: the leaked slots, and only those, were handed out again", round)
				assert.Equal(t, "4", reserveTestCount(t, c),
					"round %d: every slot handed out is still counted", round)
			}
		})
	}
}

func TestReleaseLeakedSlotsCountsOnlyPodsHoldingASlotOfTheKey(t *testing.T) {
	ctx := context.Background()
	const keyA, keyB = "counter-a", "counter-b"

	leftAlone := makeSlotHolder("a-left-alone", keyA)
	leftAlone.Labels = nil // the key came with the pod template; the pod was not injected
	otherOwner := makeSlotHolder("a-other-owner", keyA)
	otherOwner.OwnerReferences[0].UID = "another-rs-uid"
	terminating := makeSlotHolder("a-terminating", keyA)
	terminating.Finalizers = []string{constants.Finalizer}
	a0 := makeSlotHolder("a-0", keyA)
	b0, b1 := makeSlotHolder("b-0", keyB), makeSlotHolder("b-1", keyB)
	b0.Labels[constants.LabelComponent] = constants.ComponentWorker

	c := newLeakTestClient(t, map[string]string{
		keyA:                            "3",
		keyB:                            "2",
		PodCounterLastReserveAnnotation: longAgo(),
	}, nil, a0, terminating, leftAlone, otherOwner, b0, b1)
	// Its finalizer has not been handled: Decrease has not run for it.
	require.NoError(t, c.Delete(ctx, terminating))
	counter := &TensorFusionPodCounter{Client: c}

	recheck, err := counter.ReleaseLeakedSlots(ctx, a0)
	require.NoError(t, err)
	assert.Zero(t, recheck)
	annotations := leakTestOwnerAnnotations(t, c)
	assert.Equal(t, "2", annotations[keyA], "the running and the terminating pod hold a slot, the other two do not")
	assert.Equal(t, "2", annotations[keyB], "another key of the same owner is not touched")

	_, err = counter.ReleaseLeakedSlots(ctx, b0)
	require.NoError(t, err)
	assert.Equal(t, "2", leakTestOwnerAnnotations(t, c)[keyB], "every slot of this key has its pod")

	// A pod disappears without the pod controller having given its slot back.
	require.NoError(t, c.Delete(ctx, b1))
	_, err = counter.ReleaseLeakedSlots(ctx, b0)
	require.NoError(t, err)
	annotations = leakTestOwnerAnnotations(t, c)
	assert.Equal(t, "1", annotations[keyB])
	assert.Equal(t, "2", annotations[keyA])

	// The last one too; the correction is asked with what is left of it.
	require.NoError(t, c.Delete(ctx, b0))
	_, err = counter.ReleaseLeakedSlots(ctx, b0)
	require.NoError(t, err)
	annotations = leakTestOwnerAnnotations(t, c)
	assert.NotContains(t, annotations, keyB, "a key without slots is removed, as Decrease does")
	assert.Equal(t, "2", annotations[keyA])
}

// A reservation for one key of the owner postpones the correction of its
// other keys: the reservation time is kept per owner.
func TestReleaseLeakedSlotsWaitsForAReservationOfAnotherKey(t *testing.T) {
	ctx := context.Background()
	const keyA, keyB = "counter-a", "counter-b"
	a0 := makeSlotHolder("a-0", keyA)
	c := newLeakTestClient(t, map[string]string{keyA: "2", PodCounterLastReserveAnnotation: longAgo()}, nil, a0)
	counter := &TensorFusionPodCounter{Client: c}

	ok, key, err := counter.Reserve(ctx, makeSlotHolder("b-in-flight", keyB), 1)
	require.NoError(t, err)
	require.True(t, ok)
	require.Equal(t, keyB, key)

	recheck, err := counter.ReleaseLeakedSlots(ctx, a0)
	require.NoError(t, err)
	assert.Positive(t, recheck)
	annotations := leakTestOwnerAnnotations(t, c)
	assert.Equal(t, "2", annotations[keyA])
	assert.Equal(t, "1", annotations[keyB])
}

// A count without a usable reservation time (written by an earlier version, or
// by hand) says nothing about how old its slots are.
func TestReleaseLeakedSlotsStartsTheGracePeriodWhenTheReservationTimeIsUnknown(t *testing.T) {
	for name, stamp := range map[string]string{"missing": "", "unreadable": "yesterday"} {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			seenAt := time.Date(2026, 1, 1, 12, 0, 0, 0, time.UTC)
			clock := &leakTestClock{at: seenAt}
			ownerAnnotations := map[string]string{reserveTestKey: "3"}
			if stamp != "" {
				ownerAnnotations[PodCounterLastReserveAnnotation] = stamp
			}
			holder := makeSlotHolder("pod-0", reserveTestKey)
			c := newLeakTestClient(t, ownerAnnotations, nil, holder)
			counter := &TensorFusionPodCounter{Client: c, now: clock.now}

			recheck, err := counter.ReleaseLeakedSlots(ctx, holder)
			require.NoError(t, err)
			assert.Equal(t, PodCounterLeakGracePeriod+leakRecheckSlack, recheck)
			annotations := leakTestOwnerAnnotations(t, c)
			assert.Equal(t, "3", annotations[reserveTestKey], "an unknown age is not an old one")
			assert.Equal(t, formatReserveTime(seenAt), annotations[PodCounterLastReserveAnnotation])

			clock.at = seenAt.Add(PodCounterLeakGracePeriod)
			_, err = counter.ReleaseLeakedSlots(ctx, holder)
			require.NoError(t, err)
			assert.Equal(t, "3", reserveTestCount(t, c))

			clock.at = seenAt.Add(PodCounterLeakGracePeriod + leakRecheckSlack)
			recheck, err = counter.ReleaseLeakedSlots(ctx, holder)
			require.NoError(t, err)
			assert.Zero(t, recheck)
			assert.Equal(t, "1", reserveTestCount(t, c))
		})
	}
}

func TestReleaseLeakedSlotsWhenInDoubtReleasesNothing(t *testing.T) {
	ctx := context.Background()
	leaked := func() map[string]string {
		return map[string]string{reserveTestKey: "3", PodCounterLastReserveAnnotation: longAgo()}
	}

	t.Run("the cache shows fewer pods than the API server", func(t *testing.T) {
		pods := []*corev1.Pod{
			makeSlotHolder("pod-0", reserveTestKey),
			makeSlotHolder("pod-1", reserveTestKey),
			makeSlotHolder("pod-2", reserveTestKey),
		}
		// Typed lists are what the manager's client answers from its cache.
		staleCache := interceptor.Funcs{
			List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
				if _, cached := list.(*corev1.PodList); cached {
					return nil
				}
				return c.List(ctx, list, opts...)
			},
		}
		c := newLeakTestClient(t, leaked(), &staleCache, pods...)
		recheck, err := (&TensorFusionPodCounter{Client: c}).ReleaseLeakedSlots(ctx, pods[0])
		require.NoError(t, err)
		assert.Zero(t, recheck)
		assert.Equal(t, "3", reserveTestCount(t, c))
	})

	t.Run("the reservation time is ahead of the clock", func(t *testing.T) {
		holder := makeSlotHolder("pod-0", reserveTestKey)
		annotations := leaked()
		annotations[PodCounterLastReserveAnnotation] = formatReserveTime(time.Now().Add(24 * time.Hour))
		c := newLeakTestClient(t, annotations, nil, holder)
		recheck, err := (&TensorFusionPodCounter{Client: c}).ReleaseLeakedSlots(ctx, holder)
		require.NoError(t, err)
		assert.Equal(t, PodCounterLeakGracePeriod+leakRecheckSlack, recheck, "keeps looking at the usual pace")
		assert.Equal(t, "3", reserveTestCount(t, c))
	})

	t.Run("the owner was replaced under the same name", func(t *testing.T) {
		holder := makeSlotHolder("pod-0", reserveTestKey)
		c := newLeakTestClient(t, leaked(), nil, holder)
		ofFormerOwner := makeSlotHolder("pod-old", reserveTestKey)
		ofFormerOwner.OwnerReferences[0].UID = "former-rs-uid"
		recheck, err := (&TensorFusionPodCounter{Client: c}).ReleaseLeakedSlots(ctx, ofFormerOwner)
		require.NoError(t, err)
		assert.Zero(t, recheck)
		assert.Equal(t, "3", reserveTestCount(t, c))
	})

	t.Run("the pod was not counted or has no owner to ask", func(t *testing.T) {
		holder := makeSlotHolder("pod-0", reserveTestKey)
		c := newLeakTestClient(t, leaked(), nil, holder)
		counter := &TensorFusionPodCounter{Client: c}

		uncounted := makeSlotHolder("pod-uncounted", reserveTestKey)
		delete(uncounted.Annotations, constants.TensorFusionPodCounterKeyAnnotation)
		ownerGone := makeSlotHolder("pod-orphan", reserveTestKey)
		ownerGone.OwnerReferences[0].Name = "rs-gone"
		bare := makeSlotHolder("pod-bare", reserveTestKey)
		bare.OwnerReferences = nil
		for _, pod := range []*corev1.Pod{uncounted, ownerGone, bare} {
			recheck, err := counter.ReleaseLeakedSlots(ctx, pod)
			require.NoError(t, err, pod.Name)
			assert.Zero(t, recheck, pod.Name)
		}
		assert.Equal(t, "3", reserveTestCount(t, c))
	})

	t.Run("the count is not a number", func(t *testing.T) {
		holder := makeSlotHolder("pod-0", reserveTestKey)
		annotations := leaked()
		annotations[reserveTestKey] = "many"
		c := newLeakTestClient(t, annotations, nil, holder)
		recheck, err := (&TensorFusionPodCounter{Client: c}).ReleaseLeakedSlots(ctx, holder)
		require.NoError(t, err)
		assert.Zero(t, recheck)
		assert.Equal(t, "many", reserveTestCount(t, c))
	})
}
