package v1

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type TensorFusionPodCounter struct {
	Client client.Client

	// locks serialises counter writes per owner inside this process; nil means
	// the process-wide set. Tests give each simulated replica its own.
	locks *ownerLocks

	// now is the clock behind the last-reserve time; nil means time.Now. Tests
	// move it instead of waiting out the grace period.
	now func() time.Time
}

func (c *TensorFusionPodCounter) clock() time.Time {
	if c.now != nil {
		return c.now()
	}
	return time.Now()
}

// PodCounterLastReserveAnnotation is written on the owner by Reserve, in the
// same conditional write as the count: the time the newest slot of this owner
// was taken, for any of its counter keys. ReleaseLeakedSlots needs it to tell a
// slot whose pod may still be on its way into the API server from a slot whose
// pod never arrived. It is one annotation per owner rather than one per counter
// key so that its name cannot outgrow the annotation name limit when the key is
// supplied by the user; the cost is that a reservation for one key postpones
// the correction of the owner's other keys, which is the safe direction.
var PodCounterLastReserveAnnotation = constants.Domain + "/pod-counter-last-reserve"

// PodCounterLeakGracePeriod is how long after the owner's last reservation a
// slot without a pod is still assumed to belong to an admission in flight.
//
// A slot is taken inside the webhook call, which is part of the API server's
// handling of the pod's create request. After the webhook has answered (it has
// timeoutSeconds, at most 30s, to do so) the API server still runs the
// remaining admission steps and stores the pod, all inside its request timeout
// (--request-timeout, 60s by default). So about a minute after Reserve the pod
// is either stored or never will be. ReleaseLeakedSlots confirms the number of
// pods with a read from the API server, not from the informer cache, so cache
// lag is not part of the budget. The second minute is margin: for clock
// differences between the operator replica that served the admission and the
// one that runs the correction, and for a write that reaches the store just
// after the request deadline. A cluster whose API server runs with a request
// timeout close to or above two minutes needs a larger value here.
const PodCounterLeakGracePeriod = 2 * time.Minute

// leakRecheckSlack is added to the delay ReleaseLeakedSlots asks for, so the
// next look happens after the grace period has passed, not exactly at its end.
const leakRecheckSlack = time.Second

// counterWriteBackoff bounds how long a counter write keeps retrying after a
// conflict. The owner object is also written by other operator replicas and by
// its own controller (status updates), so conflicts are expected; about 1.3s
// in total stays well inside the admission timeout.
var counterWriteBackoff = wait.Backoff{
	Steps:    8,
	Duration: 10 * time.Millisecond,
	Factor:   2.0,
	Jitter:   0.2,
}

// ownerLocks hands out one mutex per owner counter. Entries are dropped when
// nobody holds or waits for them, so the map does not grow with the number of
// owners ever seen.
type ownerLocks struct {
	mu    sync.Mutex
	locks map[string]*ownerLock
}

type ownerLock struct {
	mu   sync.Mutex
	refs int
}

var processOwnerLocks = &ownerLocks{}

func (l *ownerLocks) lock(id string) (unlock func()) {
	l.mu.Lock()
	if l.locks == nil {
		l.locks = map[string]*ownerLock{}
	}
	entry, ok := l.locks[id]
	if !ok {
		entry = &ownerLock{}
		l.locks[id] = entry
	}
	entry.refs++
	l.mu.Unlock()

	entry.mu.Lock()
	return func() {
		entry.mu.Unlock()
		l.mu.Lock()
		entry.refs--
		if entry.refs == 0 {
			delete(l.locks, id)
		}
		l.mu.Unlock()
	}
}

func (c *TensorFusionPodCounter) lockOwner(namespace string, ownerRef *metav1.OwnerReference, key string) (unlock func()) {
	locks := c.locks
	if locks == nil {
		locks = processOwnerLocks
	}
	return locks.lock(namespace + "/" + ownerRef.Kind + "/" + ownerRef.Name + "/" + key)
}

// getOrGenerateKey returns the pod's counter key from annotation if present, otherwise generates one from pod template labels (e.g. pod-template-hash or fallback to object hash)
func getOrGenerateKey(pod *corev1.Pod) string {
	if pod.Annotations != nil {
		if key, ok := pod.Annotations[constants.TensorFusionPodCounterKeyAnnotation]; ok && key != "" {
			return key
		}
	}
	// Try to use pod-template-hash if present
	if hash, ok := pod.Labels["pod-template-hash"]; ok && hash != "" {
		return fmt.Sprintf("%s/tf-counter-%s", constants.Domain, hash)
	}

	// Fallback to object hash
	return fmt.Sprintf("%s/tf-counter-%s", constants.Domain, utils.GetObjectHash(pod))
}

// Get gets the counter value from the owner annotation by key
func (c *TensorFusionPodCounter) Get(ctx context.Context, pod *corev1.Pod) (int32, string, error) {
	ownerRef := getControllerOwnerRef(pod)
	if ownerRef == nil {
		return 0, "", fmt.Errorf("no controller owner reference found for pod %s/%s", pod.Namespace, pod.Name)
	}
	key := getOrGenerateKey(pod)
	ownerObj := &unstructured.Unstructured{}
	ownerObj.SetAPIVersion(ownerRef.APIVersion)
	ownerObj.SetKind(ownerRef.Kind)
	objKey := client.ObjectKey{Name: ownerRef.Name, Namespace: pod.Namespace}
	if err := c.Client.Get(ctx, objKey, ownerObj); err != nil {
		return 0, "", fmt.Errorf("failed to get owner object: %w", err)
	}
	annotations := ownerObj.GetAnnotations()
	if annotations == nil {
		return 0, key, nil
	}
	val, ok := annotations[key]
	if !ok || val == "" {
		return 0, key, nil
	}
	count, err := strconv.ParseInt(val, 10, 32)
	if err != nil {
		return 0, "", fmt.Errorf("invalid count annotation: %s, err: %w", val, err)
	}
	return int32(count), key, nil
}

// Increase increases the counter in owner annotation by key. Wrapped in
// retry.RetryOnConflict so concurrent admission of multiple pods sharing the
// same owner (e.g. a Deployment scaling up) does not surface as a 500 to the
// API server — each concurrent webhook would otherwise read the same
// ResourceVersion, both compute count+1, and only the first Update succeeds.
// Mirrors Decrease's retry strategy so admission-side and finalizer-side
// bookkeeping stay symmetric.
func (c *TensorFusionPodCounter) Increase(ctx context.Context, pod *corev1.Pod) error {
	return c.IncreaseWithKey(ctx, pod, getOrGenerateKey(pod))
}

// IncreaseWithKey increments a previously resolved counter key. Admission
// mutates the Pod between Get and Increase; callers that resolved a fallback
// object-hash key before mutation must reuse that key or the increment would
// land in a different owner annotation.
func (c *TensorFusionPodCounter) IncreaseWithKey(ctx context.Context, pod *corev1.Pod, key string) error {
	ownerRef := getControllerOwnerRef(pod)
	if ownerRef == nil {
		return fmt.Errorf("no controller owner reference found for pod %s/%s", pod.Namespace, pod.Name)
	}
	objKey := client.ObjectKey{Name: ownerRef.Name, Namespace: pod.Namespace}

	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		ownerObj := &unstructured.Unstructured{}
		ownerObj.SetAPIVersion(ownerRef.APIVersion)
		ownerObj.SetKind(ownerRef.Kind)
		if err := c.Client.Get(ctx, objKey, ownerObj); err != nil {
			return fmt.Errorf("failed to get owner object: %w", err)
		}
		annotations := ownerObj.GetAnnotations()
		if annotations == nil {
			annotations = map[string]string{}
		}
		val := annotations[key]
		if val == "" {
			val = "0"
		}
		count, err := strconv.ParseInt(val, 10, 32)
		if err != nil {
			return fmt.Errorf("invalid count annotation: %s, err: %w", val, err)
		}
		count++
		annotations[key] = fmt.Sprintf("%d", count)
		annotations[PodCounterLastReserveAnnotation] = formatReserveTime(c.clock())
		ownerObj.SetAnnotations(annotations)
		if err := c.Client.Update(ctx, ownerObj); err != nil {
			return fmt.Errorf("failed to update owner annotation: %w", err)
		}
		return nil
	})
}

// Reserve takes one of the limit slots of the pod's owner, and reports whether
// it got one. It is the only way admission may count a pod: the comparison
// with the limit and the increment are one conditional write on the owner, so
// a slot cannot be handed out twice.
//
// Reading the count first and increasing it later, as admission used to do,
// let every admission that ran in between pass the same comparison: N pods of
// one ReplicaSet created together could all be counted against the same free
// slot.
//
// What makes it exact:
//   - the owner is updated with the resourceVersion it was read at, so of all
//     writers that saw the same count (other admissions here, other operator
//     replicas, the pod controller's Decrease) only one succeeds and the rest
//     read again and compare again. The count therefore never exceeds limit,
//     with any number of operator replicas;
//   - the owner is read as unstructured, which the manager's client does not
//     serve from its cache. Even a stale read could only fail the conditional
//     write, never over-admit;
//   - a per-owner lock in this process keeps concurrent admissions of one
//     owner from colliding with each other, so the retries are left for the
//     writers outside this process.
//
// What remains inexact, always towards fewer injected pods than the limit:
//   - a slot is taken when the pod is admitted, not when it is stored. If the
//     API server drops the pod afterwards (a later admission step or quota
//     rejects it, the request times out, the operator stops before answering)
//     no pod exists whose deletion would give the slot back. Every code path of
//     the handler that does not inject the pod releases its slot, but those
//     cases are outside the handler. ReleaseLeakedSlots gives such a slot back
//     once PodCounterLeakGracePeriod has passed since the owner's last
//     reservation (the time Reserve writes next to the count). It runs when the
//     pod controller reconciles an injected pod of the same owner, so an owner
//     none of whose injected pods exists or changes keeps the slot until one
//     does, or until the owner is replaced;
//   - the webhook is registered with the default reinvocationPolicy (Never). If
//     it were re-invoked for one pod, each invocation would take a slot;
//   - a slot is given back when the pod controller handles the deletion of an
//     injected pod. A replacement pod admitted before that sees the slot still
//     taken and is left alone.
//
// If the conditional write keeps conflicting past counterWriteBackoff the
// error is returned and the pod is not admitted; its controller creates it
// again.
func (c *TensorFusionPodCounter) Reserve(ctx context.Context, pod *corev1.Pod, limit int32) (bool, string, error) {
	ownerRef := getControllerOwnerRef(pod)
	if ownerRef == nil {
		return false, "", fmt.Errorf("no controller owner reference found for pod %s/%s", pod.Namespace, pod.Name)
	}
	key := getOrGenerateKey(pod)
	objKey := client.ObjectKey{Name: ownerRef.Name, Namespace: pod.Namespace}

	unlock := c.lockOwner(pod.Namespace, ownerRef, key)
	defer unlock()

	reserved := false
	err := retry.RetryOnConflict(counterWriteBackoff, func() error {
		reserved = false
		ownerObj := &unstructured.Unstructured{}
		ownerObj.SetAPIVersion(ownerRef.APIVersion)
		ownerObj.SetKind(ownerRef.Kind)
		if err := c.Client.Get(ctx, objKey, ownerObj); err != nil {
			return fmt.Errorf("failed to get owner object: %w", err)
		}
		annotations := ownerObj.GetAnnotations()
		if annotations == nil {
			annotations = map[string]string{}
		}
		val := annotations[key]
		if val == "" {
			val = "0"
		}
		count, err := strconv.ParseInt(val, 10, 32)
		if err != nil {
			return fmt.Errorf("invalid count annotation: %s, err: %w", val, err)
		}
		if count >= int64(limit) {
			return nil
		}
		annotations[key] = strconv.FormatInt(count+1, 10)
		// Same write as the count: ReleaseLeakedSlots must never see the new
		// count together with an older reservation time.
		annotations[PodCounterLastReserveAnnotation] = formatReserveTime(c.clock())
		ownerObj.SetAnnotations(annotations)
		// Update carries the resourceVersion of the read above: it fails with
		// a conflict if anything changed the owner in between.
		if err := c.Client.Update(ctx, ownerObj); err != nil {
			return fmt.Errorf("failed to update owner annotation: %w", err)
		}
		reserved = true
		return nil
	})
	if err != nil {
		return false, "", err
	}
	return reserved, key, nil
}

// Release gives back a slot taken by Reserve for a pod that is not injected
// after all. key is the one Reserve returned: admission mutates the pod in
// between, and a key derived from the mutated pod could differ.
func (c *TensorFusionPodCounter) Release(ctx context.Context, pod *corev1.Pod, key string) error {
	return c.decreaseWithKey(ctx, pod, key)
}

// Decrease decreases the counter in owner annotation by key. Wraps the
// Get+modify+Update in retry.RetryOnConflict so a transient ResourceVersion
// race with the workload controller (or another finalizer hook on the same
// owner) does not surface as a failure to the caller. Genuine errors
// (RBAC, network, persistent conflict after retries) propagate so the pod
// reconciler can requeue and the count stays consistent with admission-side
// bookkeeping — silently swallowing them would leak the owner annotation
// past the pod's lifetime and skew future admission decisions.
func (c *TensorFusionPodCounter) Decrease(ctx context.Context, pod *corev1.Pod) error {
	return c.decreaseWithKey(ctx, pod, getOrGenerateKey(pod))
}

func (c *TensorFusionPodCounter) decreaseWithKey(ctx context.Context, pod *corev1.Pod, key string) error {
	log := log.FromContext(ctx)
	ownerRef := getControllerOwnerRef(pod)
	if ownerRef == nil {
		log.Error(nil, "no controller owner reference found for pod", "namespace", pod.Namespace, "name", pod.Name)
		return nil
	}
	objKey := client.ObjectKey{Name: ownerRef.Name, Namespace: pod.Namespace}

	unlock := c.lockOwner(pod.Namespace, ownerRef, key)
	defer unlock()

	return retry.RetryOnConflict(counterWriteBackoff, func() error {
		ownerObj := &unstructured.Unstructured{}
		ownerObj.SetAPIVersion(ownerRef.APIVersion)
		ownerObj.SetKind(ownerRef.Kind)
		if err := c.Client.Get(ctx, objKey, ownerObj); err != nil {
			// owner already gone — there is no counter to decrement.
			if errors.IsNotFound(err) {
				return nil
			}
			return fmt.Errorf("failed to get owner object: %w", err)
		}
		annotations := ownerObj.GetAnnotations()
		if annotations == nil {
			annotations = map[string]string{}
		}
		val := annotations[key]
		if val == "" {
			val = "0"
		}
		count, err := strconv.ParseInt(val, 10, 32)
		if err != nil {
			// Corrupt annotation — give up silently rather than spin on retries.
			log.Error(err, "invalid count annotation", "namespace", pod.Namespace, "name", pod.Name)
			return nil
		}
		count--
		if count <= 0 {
			delete(annotations, key)
		} else {
			annotations[key] = fmt.Sprintf("%d", count)
		}
		ownerObj.SetAnnotations(annotations)
		if err := c.Client.Update(ctx, ownerObj); err != nil {
			return fmt.Errorf("failed to update owner annotation: %w", err)
		}
		return nil
	})
}

// ReleaseLeakedSlots lowers the count of the pod's counter key on the pod's
// owner to the number of pods that hold a slot, when the count is above that
// number and the owner's last reservation is older than
// PodCounterLeakGracePeriod. pod is any injected pod of the owner; it only
// names the owner and the key. The returned duration, when not zero, is when
// to look again.
//
// It closes the gap described on Reserve (a slot taken for a pod the API
// server never stored) and equally a slot whose pod was deleted without
// Decrease having run for it.
//
// A pod holds a slot while it exists, whatever its phase and also while it is
// terminating: Decrease runs when its finalizer is handled, so the slot is
// either still taken or the pod is about to disappear. Counting a pod that
// has already given its slot back only postpones the correction.
//
// Why it cannot take away a slot that is in use, or let the limit be exceeded:
//   - the count is only ever lowered, and only to a number of pods read from
//     the API server (an unstructured list, which the manager's client does
//     not serve from its cache) after the grace period was found to have
//     passed. A pod admitted before the last reservation time is, by then,
//     stored and in that list, or it will never exist;
//   - the write carries the resourceVersion of a read made after the list,
//     and is made only if that read still shows a last reservation older
//     than the grace period, measured at the moment before the list. A
//     Reserve that ran in between either changed the owner (the write
//     conflicts and everything is evaluated again against the new time) or
//     is visible in that read; in both cases nothing is released. A Decrease
//     in between only makes the count smaller than what is compared here;
//   - a missing or unreadable reservation time (a count written by an older
//     version, or edited by hand) is not taken as old: the time is set to now
//     and the grace period starts from there;
//   - the pod cache is used only to skip the work when it already shows as
//     many pods as slots; it never justifies a release.
//
// Not covered: an operator replica of a version that counts without writing
// the reservation time, serving admissions while this runs (possible for a
// moment during the upgrade to this version), can have a fresh slot released.
func (c *TensorFusionPodCounter) ReleaseLeakedSlots(ctx context.Context, pod *corev1.Pod) (time.Duration, error) {
	ownerRef := getControllerOwnerRef(pod)
	key := pod.Annotations[constants.TensorFusionPodCounterKeyAnnotation]
	if ownerRef == nil || key == "" {
		return 0, nil
	}

	owner, err := c.readOwner(ctx, pod.Namespace, ownerRef)
	if err != nil || owner == nil {
		return 0, err
	}
	count, ok := slotCount(owner.GetAnnotations(), key)
	if !ok || count <= 0 {
		return 0, nil
	}
	// The usual case ends here, with one read of the owner and a list from
	// the cache: every slot has its pod.
	cached, err := c.countSlotHolders(ctx, &corev1.PodList{}, pod.Namespace, owner.GetUID(), key)
	if err != nil {
		return 0, err
	}
	if count <= cached {
		return 0, nil
	}

	now := c.clock()
	last, known := lastReserveTime(owner.GetAnnotations())
	if !known {
		err := c.writeOwnerAnnotations(ctx, pod.Namespace, ownerRef, key, func(annotations map[string]string) bool {
			if _, known := lastReserveTime(annotations); known {
				return false
			}
			annotations[PodCounterLastReserveAnnotation] = formatReserveTime(c.clock())
			return true
		})
		return PodCounterLeakGracePeriod + leakRecheckSlack, err
	}
	if wait := last.Add(PodCounterLeakGracePeriod).Sub(now); wait >= 0 {
		// A time further ahead than the grace period comes from a clock that
		// is off; keep looking at the usual pace until it has passed.
		return min(wait, PodCounterLeakGracePeriod) + leakRecheckSlack, nil
	}

	livePods := &unstructured.UnstructuredList{}
	livePods.SetAPIVersion("v1")
	livePods.SetKind("PodList")
	live, err := c.countSlotHolders(ctx, livePods, pod.Namespace, owner.GetUID(), key)
	if err != nil {
		return 0, err
	}
	if count <= live {
		return 0, nil
	}

	var before int64
	var recheck time.Duration
	err = c.writeOwnerAnnotations(ctx, pod.Namespace, ownerRef, key, func(annotations map[string]string) bool {
		before, recheck = 0, 0
		current, ok := slotCount(annotations, key)
		if !ok || current <= live {
			return false
		}
		last, known := lastReserveTime(annotations)
		if !known || !last.Add(PodCounterLeakGracePeriod).Before(now) {
			// A slot was taken after the pods were listed.
			recheck = PodCounterLeakGracePeriod + leakRecheckSlack
			return false
		}
		before = current
		if live <= 0 {
			delete(annotations, key)
		} else {
			annotations[key] = strconv.FormatInt(live, 10)
		}
		return true
	})
	if err != nil {
		return 0, err
	}
	if before > 0 {
		log.FromContext(ctx).Info("released tf pod count slots that no pod holds",
			"namespace", pod.Namespace, "ownerKind", ownerRef.Kind, "owner", ownerRef.Name,
			"key", key, "from", before, "to", live)
	}
	return recheck, nil
}

// readOwner reads the owner from the API server. It returns nil when the owner
// is gone or another object has taken its name: that object has a counter of
// its own.
func (c *TensorFusionPodCounter) readOwner(
	ctx context.Context, namespace string, ownerRef *metav1.OwnerReference,
) (*unstructured.Unstructured, error) {
	ownerObj := &unstructured.Unstructured{}
	ownerObj.SetAPIVersion(ownerRef.APIVersion)
	ownerObj.SetKind(ownerRef.Kind)
	if err := c.Client.Get(ctx, client.ObjectKey{Name: ownerRef.Name, Namespace: namespace}, ownerObj); err != nil {
		if errors.IsNotFound(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to get owner object: %w", err)
	}
	if ownerObj.GetUID() != ownerRef.UID {
		return nil, nil
	}
	return ownerObj, nil
}

// writeOwnerAnnotations is the write discipline of Reserve for the correction:
// under the per-owner lock, read the owner, let mutate change its annotations
// and update with the resourceVersion of that read, evaluating again from the
// read on a conflict. mutate returns false to leave the owner as it is.
func (c *TensorFusionPodCounter) writeOwnerAnnotations(
	ctx context.Context, namespace string, ownerRef *metav1.OwnerReference, key string,
	mutate func(annotations map[string]string) bool,
) error {
	unlock := c.lockOwner(namespace, ownerRef, key)
	defer unlock()

	return retry.RetryOnConflict(counterWriteBackoff, func() error {
		ownerObj, err := c.readOwner(ctx, namespace, ownerRef)
		if err != nil || ownerObj == nil {
			return err
		}
		annotations := ownerObj.GetAnnotations()
		if annotations == nil {
			annotations = map[string]string{}
		}
		if !mutate(annotations) {
			return nil
		}
		ownerObj.SetAnnotations(annotations)
		if err := c.Client.Update(ctx, ownerObj); err != nil {
			return fmt.Errorf("failed to update owner annotation: %w", err)
		}
		return nil
	})
}

// countSlotHolders lists the pods of the namespace into list and counts those
// holding a slot of the owner's counter key. A typed list is answered from the
// manager's cache, an unstructured one by the API server.
func (c *TensorFusionPodCounter) countSlotHolders(
	ctx context.Context, list client.ObjectList, namespace string, ownerUID types.UID, key string,
) (int64, error) {
	err := c.Client.List(ctx, list, client.InNamespace(namespace), client.HasLabels{constants.LabelComponent})
	if err != nil {
		return 0, fmt.Errorf("failed to list pods of the owner: %w", err)
	}
	var holders int64
	err = meta.EachListItem(list, func(obj runtime.Object) error {
		pod, err := meta.Accessor(obj)
		if err != nil {
			return err
		}
		if holdsSlot(pod, ownerUID, key) {
			holders++
		}
		return nil
	})
	return holders, err
}

// holdsSlot reports whether the pod was injected for the given counter key of
// the owner. The webhook stamps both marks in the patch that injects a pod: the
// component label and the counter key annotation. The annotation alone is not
// enough, a pod template may carry it and then so do the pods left alone.
func holdsSlot(pod metav1.Object, ownerUID types.UID, key string) bool {
	component := pod.GetLabels()[constants.LabelComponent]
	if component != constants.ComponentClient && component != constants.ComponentWorker {
		return false
	}
	if pod.GetAnnotations()[constants.TensorFusionPodCounterKeyAnnotation] != key {
		return false
	}
	ref := metav1.GetControllerOfNoCopy(pod)
	return ref != nil && ref.UID == ownerUID
}

// slotCount reads the count of a counter key; false when it is not a number.
func slotCount(annotations map[string]string, key string) (int64, bool) {
	val := annotations[key]
	if val == "" {
		return 0, true
	}
	count, err := strconv.ParseInt(val, 10, 32)
	return count, err == nil
}

func formatReserveTime(t time.Time) string {
	return t.UTC().Format(time.RFC3339Nano)
}

func lastReserveTime(annotations map[string]string) (time.Time, bool) {
	last, err := time.Parse(time.RFC3339Nano, annotations[PodCounterLastReserveAnnotation])
	return last, err == nil
}

// getControllerOwnerRef returns the controller owner reference of a pod
func getControllerOwnerRef(pod *corev1.Pod) *metav1.OwnerReference {
	for i, ref := range pod.OwnerReferences {
		if ref.Controller != nil && *ref.Controller {
			return &pod.OwnerReferences[i]
		}
	}
	return nil
}
