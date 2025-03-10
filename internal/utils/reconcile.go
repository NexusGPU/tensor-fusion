package utils

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
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

func HandleFinalizer[T client.Object](ctx context.Context, obj T, r client.Client, deleteHook func(context.Context, T) (bool, error)) (bool, error) {
	// Check if object is being deleted
	deleted := !obj.GetDeletionTimestamp().IsZero()
	if deleted {
		// Object is being deleted - process finalizer
		if controllerutil.ContainsFinalizer(obj, constants.Finalizer) {
			// Run custom deletion hook
			canBeDeleted, err := deleteHook(ctx, obj)
			if err != nil {
				return false, err
			}

			// Remove finalizer once cleanup is done
			if canBeDeleted {
				controllerutil.RemoveFinalizer(obj, constants.Finalizer)
				if err := r.Update(ctx, obj); err != nil {
					return false, err
				}
			}
		}
	} else {
		// Object is not being deleted - add finalizer if not present
		if !controllerutil.ContainsFinalizer(obj, constants.Finalizer) {
			controllerutil.AddFinalizer(obj, constants.Finalizer)
			if err := r.Update(ctx, obj); err != nil {
				return false, err
			}
		}
	}
	return deleted, nil
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

func GetObjectHash(obj any) string {
	hasher := sha256.New()
	jsonBytes, err := json.Marshal(obj)
	if err != nil {
		panic(err)
	}
	str := string(jsonBytes)
	hasher.Write([]byte(str))
	return hex.EncodeToString(hasher.Sum(nil))
}

const DebounceKeySuffix = ":in_queue"

func DebouncedReconcileCheck(ctx context.Context, lastProcessedItems *sync.Map, name types.NamespacedName) (runNow bool, alreadyQueued bool, waitTime time.Duration) {
	const (
		// Minimum time between reconciliations for the same object
		debounceInterval = 5 * time.Second
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
