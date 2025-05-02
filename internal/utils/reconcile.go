package utils

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"hash/fnv"
	"math"
	"math/rand/v2"
	"os"
	"sync"
	"time"

	constants "github.com/NexusGPU/tensor-fusion/internal/constants"
	"k8s.io/apimachinery/pkg/types"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// ErrNextLoop is not a real error. It forces the current reconciliation loop to stop
// and return the associated Result object
var ErrNextLoop = errors.New("stop this loop and return the associated Result object")

// ErrTerminateLoop is not a real error. It forces the current reconciliation loop to stop
var ErrTerminateLoop = errors.New("stop this loop and do not requeue")

// HandleFinalizer ensures proper finalizer management for Kubernetes resources.
// It automatically adds the finalizer when needed, and removes it after successful cleanup.
// Returns (shouldReturn, err):
//   - shouldReturn: true if the caller should immediately return and wait for the next reconcile.
//   - err: any error encountered during update or deleteHook.
func HandleFinalizer[T client.Object](
	ctx context.Context,
	obj T,
	r client.Client,
	deleteHook func(context.Context, T) (bool, error),
) (shouldReturn bool, err error) {
	// If the object is being deleted, handle finalizer removal
	if !obj.GetDeletionTimestamp().IsZero() {
		if controllerutil.ContainsFinalizer(obj, constants.Finalizer) {
			// Run custom deletion logic before removing the finalizer
			var canBeDeleted bool
			canBeDeleted, err = deleteHook(ctx, obj)
			if err != nil {
				// Error during deletion hook, requeue for next reconcile
				shouldReturn = true
				return shouldReturn, err
			}
			if canBeDeleted {
				controllerutil.RemoveFinalizer(obj, constants.Finalizer)
				err = r.Update(ctx, obj)
				if err != nil {
					// Failed to update object, requeue for next reconcile
					return shouldReturn, err
				}
				// Finalizer removed, wait for next reconcile
				shouldReturn = true
				return shouldReturn, err
			}
		}
		// continue with deletion logic
		shouldReturn = false
		return shouldReturn, err
	}

	// If the object is not being deleted, ensure the finalizer is present
	if !controllerutil.ContainsFinalizer(obj, constants.Finalizer) {
		controllerutil.AddFinalizer(obj, constants.Finalizer)
		err = r.Update(ctx, obj)
		if err != nil {
			// Failed to update object, requeue for next reconcile
			return shouldReturn, err
		}
		// Finalizer added, wait for next reconcile
		shouldReturn = true
		return shouldReturn, err
	}

	// Finalizer already present, continue with business logic
	shouldReturn = false
	return shouldReturn, err
}

func CalculateExponentialBackoffWithJitter(retryCount int64) time.Duration {
	const (
		baseDelay  = 3 * time.Second
		maxDelay   = 60 * time.Second
		factor     = 2.0
		maxRetries = 10
	)

	if retryCount > maxRetries {
		retryCount = maxRetries
	}

	backoff := float64(baseDelay) * math.Pow(factor, float64(retryCount))

	jitter := rand.Float64() * backoff

	totalDelay := time.Duration(jitter)
	if totalDelay < baseDelay {
		totalDelay = baseDelay
	}
	if totalDelay > maxDelay {
		totalDelay = maxDelay
	}

	return totalDelay
}

func CurrentNamespace() string {
	namespace := constants.NamespaceDefaultVal
	envNamespace := os.Getenv(constants.NamespaceEnv)
	if envNamespace != "" {
		namespace = envNamespace
	}
	return namespace
}

// GetObjectHash generates a shorter FNV-1a hash for one or more objects
func GetObjectHash(objs ...any) string {
	hasher := fnv.New64a()

	for _, obj := range objs {
		jsonBytes, err := json.Marshal(obj)
		if err != nil {
			panic(err)
		}
		// Add length prefix to prevent collisions when combining multiple objects
		hasher.Write(fmt.Appendf(nil, "%d:", len(jsonBytes)))
		hasher.Write(jsonBytes)
	}

	return fmt.Sprintf("%x", hasher.Sum(nil))
}

func CompareAndGetObjectHash(hash string, obj ...any) (bool, string) {
	newHash := GetObjectHash(obj...)
	return hash != newHash, newHash
}

const DebounceKeySuffix = ":in_queue"

func DebouncedReconcileCheck(ctx context.Context, lastProcessedItems *sync.Map, name types.NamespacedName) (runNow bool, alreadyQueued bool, waitTime time.Duration) {
	const (
		// Minimum time between reconciliations for the same object
		debounceInterval = 3 * time.Second
	)
	now := time.Now()
	key := name.String()
	inQueueKey := key + DebounceKeySuffix

	if val, exists := lastProcessedItems.Load(key); exists {
		if lastProcessed, ok := val.(time.Time); ok {
			elapsed := now.Sub(lastProcessed)
			if elapsed < debounceInterval {
				wait := debounceInterval - elapsed

				if _, loaded := lastProcessedItems.LoadOrStore(inQueueKey, now); loaded {
					return false, true, wait
				}
				return false, false, wait
			}
		}
	}

	lastProcessedItems.Delete(inQueueKey)
	lastProcessedItems.Store(key, now)
	return true, false, 0
}

func IsPodConditionTrue(conditions []corev1.PodCondition, conditionType corev1.PodConditionType) bool {
	for _, condition := range conditions {
		if condition.Type == conditionType {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

func IsPodTerminated(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodFailed || pod.Status.Phase == corev1.PodSucceeded
}
