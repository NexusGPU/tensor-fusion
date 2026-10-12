package v1

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

const reserveTestKey = "counter-key"

func newReserveTestClient(t *testing.T) client.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, appsv1.AddToScheme(scheme))
	rs := &appsv1.ReplicaSet{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "rs", UID: "rs-uid"}}
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(rs).Build()
}

func reserveTestCount(t *testing.T, c client.Client) string {
	t.Helper()
	got := &appsv1.ReplicaSet{}
	require.NoError(t, c.Get(context.Background(), client.ObjectKey{Namespace: "ns", Name: "rs"}, got))
	return got.Annotations[reserveTestKey]
}

// reserveConcurrently lets every counter try once, all at the same moment, and
// returns how many got a slot.
func reserveConcurrently(t *testing.T, counters []*TensorFusionPodCounter, limit int32) int32 {
	t.Helper()
	var reserved atomic.Int32
	var wg sync.WaitGroup
	errs := make([]error, len(counters))
	start := make(chan struct{})
	for i, counter := range counters {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			ok, key, err := counter.Reserve(context.Background(), makeCounterPod("pod"), limit)
			errs[i] = err
			if err == nil && ok {
				assert.Equal(t, reserveTestKey, key)
				reserved.Add(1)
			}
		}()
	}
	close(start)
	wg.Wait()
	for i := range errs {
		assert.NoError(t, errs[i], "admission %d", i)
	}
	return reserved.Load()
}

// The defect: admission compared the count with the limit, built the patch and
// only then increased the count, so pods admitted together all passed the
// comparison. TestGetThenIncreaseOverAdmits keeps that sequence to show what
// Reserve replaces; the other tests run the same load through Reserve.
func TestGetThenIncreaseOverAdmits(t *testing.T) {
	const admissions, limit = 16, 3
	c := newReserveTestClient(t)
	counter := &TensorFusionPodCounter{Client: c}

	var wg sync.WaitGroup
	var admitted atomic.Int32
	compared := make(chan struct{})
	var comparing sync.WaitGroup
	comparing.Add(admissions)
	for range admissions {
		wg.Add(1)
		go func() {
			defer wg.Done()
			pod := makeCounterPod("pod")
			count, key, err := counter.Get(context.Background(), pod)
			comparing.Done()
			// every admission has compared before any of them counts itself
			<-compared
			if err != nil || count >= limit {
				return
			}
			if counter.IncreaseWithKey(context.Background(), pod, key) == nil {
				admitted.Add(1)
			}
		}()
	}
	comparing.Wait()
	close(compared)
	wg.Wait()
	assert.Greater(t, admitted.Load(), int32(limit), "comparing and counting in two steps admits more than the limit")
}

func TestReserveConcurrentAdmissionsOfOneOwnerTakeExactlyLimitSlots(t *testing.T) {
	const admissions, limit = 64, 5
	for round := range 20 {
		c := newReserveTestClient(t)
		// One operator process: all admissions share its per-owner lock.
		counters := make([]*TensorFusionPodCounter, admissions)
		for i := range counters {
			counters[i] = &TensorFusionPodCounter{Client: c}
		}
		assert.Equal(t, int32(limit), reserveConcurrently(t, counters, limit), "round %d", round)
		assert.Equal(t, "5", reserveTestCount(t, c), "round %d", round)
	}
}

// Several operator replicas share nothing but the API server. Each counter
// gets a lock set of its own here, so only the conditional write on the owner
// keeps them apart.
func TestReserveAcrossReplicasTakesExactlyLimitSlots(t *testing.T) {
	const admissions, limit = 48, 4
	for round := range 20 {
		c := newReserveTestClient(t)
		counters := make([]*TensorFusionPodCounter, admissions)
		for i := range counters {
			counters[i] = &TensorFusionPodCounter{Client: c, locks: &ownerLocks{}}
		}
		assert.Equal(t, int32(limit), reserveConcurrently(t, counters, limit), "round %d", round)
		assert.Equal(t, "4", reserveTestCount(t, c), "round %d", round)
	}
}

func TestReserveAndReleaseOfASlot(t *testing.T) {
	c := newReserveTestClient(t)
	counter := &TensorFusionPodCounter{Client: c}
	ctx := context.Background()
	pod := makeCounterPod("pod")

	ok, _, err := counter.Reserve(ctx, pod, 0)
	require.NoError(t, err)
	assert.False(t, ok, "a limit of zero admits nothing")
	assert.Empty(t, reserveTestCount(t, c), "a refused admission writes nothing")

	ok, key, err := counter.Reserve(ctx, pod, 1)
	require.NoError(t, err)
	require.True(t, ok)
	ok, _, err = counter.Reserve(ctx, pod, 1)
	require.NoError(t, err)
	assert.False(t, ok, "the only slot is taken")
	assert.Equal(t, "1", reserveTestCount(t, c))

	require.NoError(t, counter.Release(ctx, pod, key))
	assert.Empty(t, reserveTestCount(t, c))
	ok, _, err = counter.Reserve(ctx, pod, 1)
	require.NoError(t, err)
	assert.True(t, ok, "a released slot can be taken again")

	_, _, err = counter.Reserve(ctx, &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "bare"}}, 1)
	assert.ErrorContains(t, err, "no controller owner reference found")
}

func TestOwnerLocksForgetIdleOwners(t *testing.T) {
	locks := &ownerLocks{}
	var wg sync.WaitGroup
	inside := 0
	for range 32 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			unlock := locks.lock("ns/ReplicaSet/rs/key")
			inside++ // the race detector fails the test if the lock does not exclude
			unlock()
		}()
	}
	wg.Wait()
	assert.Equal(t, 32, inside)
	assert.Empty(t, locks.locks, "no entry is kept for an owner nobody is admitting pods for")
}
