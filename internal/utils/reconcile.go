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
	"strings"
	"time"

	constants "github.com/NexusGPU/tensor-fusion/internal/constants"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// ErrNextLoop is not a real error. It forces the current reconciliation loop to stop
// and return the associated Result object
var ErrNextLoop = errors.New("stop this loop and return the associated Result object")

// ErrTerminateLoop is not a real error. It forces the current reconciliation loop to stop
var ErrTerminateLoop = errors.New("stop this loop and do not requeue")

var IsTestMode = false

func init() {
	// in unit testing mode, debounce should be very short
	if (len(os.Args) > 1 && os.Args[1] == "-test.run") ||
		os.Getenv("GO_TESTING") == "true" {
		IsTestMode = true
		constants.PendingRequeueDuration = time.Millisecond * 150
		constants.StatusCheckInterval = time.Millisecond * 200
		constants.GracefulPeriodSeconds = ptr.To(int64(0))
		constants.UnschedQueueBufferDuration = time.Millisecond * 50
	}
}

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
	totalDelay := min(max(time.Duration(jitter), baseDelay), maxDelay)

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
		if objStr, ok := obj.(string); ok {
			hasher.Write([]byte(objStr))
		} else if objBytes, ok := obj.([]byte); ok {
			hasher.Write(objBytes)
		} else {
			jsonBytes, err := json.Marshal(obj)
			if err != nil {
				hasher.Write([]byte(err.Error()))
			}
			hasher.Write(jsonBytes)
		}
	}
	return fmt.Sprintf("%x", hasher.Sum(nil))
}

func CompareAndGetObjectHash(hash string, obj ...any) (bool, string) {
	newHash := GetObjectHash(obj...)
	return hash != newHash, newHash
}

func IsPodConditionTrue(conditions []corev1.PodCondition, conditionType corev1.PodConditionType) bool {
	for _, condition := range conditions {
		if condition.Type == conditionType {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

func IsPodStopped(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodFailed || pod.Status.Phase == corev1.PodSucceeded
}

func IsPodRunning(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodRunning
}

func IsPodPending(pod *corev1.Pod) bool {
	return pod.Status.Phase == corev1.PodPending && pod.DeletionTimestamp.IsZero()
}

func ExtractPoolNameFromNodeLabel(node *tfv1.GPUNode) string {
	var poolName string
	for labelKey := range node.Labels {
		after, ok := strings.CutPrefix(labelKey, constants.GPUNodePoolIdentifierLabelPrefix)
		if ok {
			poolName = after
			break
		}
	}
	return poolName
}

func EqualConditionsDisregardTransitionTime(a, b []metav1.Condition) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Type != b[i].Type ||
			a[i].Status != b[i].Status ||
			a[i].Reason != b[i].Reason ||
			a[i].Message != b[i].Message {
			return false
		}
	}
	return true
}

func HasGPUResourceRequest(pod *corev1.Pod) bool {
	for _, container := range pod.Spec.Containers {
		if container.Resources.Requests != nil {
			if containsGPUResources(container.Resources.Requests) {
				return true
			}
			if containsGPUResources(container.Resources.Limits) {
				return true
			}
		}
	}
	return false
}

func IsTensorFusionPod(pod *corev1.Pod) bool {
	return pod.Labels != nil && pod.Labels[constants.TensorFusionEnabledLabelKey] == constants.TrueStringValue
}

func IsTensorFusionPodDisabled(pod *corev1.Pod) bool {
	return pod.Labels != nil && pod.Labels[constants.TensorFusionEnabledLabelKey] == constants.FalseStringValue
}
func IsTensorFusionWorker(pod *corev1.Pod) bool {
	return pod.Labels[constants.LabelComponent] == constants.ComponentWorker
}

func IsDesignatedNodePod(pod *corev1.Pod) bool {
	if pod.Spec.NodeName != "" {
		return true
	}
	if pod.Spec.NodeSelector != nil && pod.Spec.NodeSelector[constants.KubernetesHostNameLabel] != "" {
		return true
	}
	if pod.Spec.Affinity != nil && pod.Spec.Affinity.NodeAffinity != nil {
		if pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
			terms := pod.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
			for _, term := range terms {
				for _, field := range term.MatchFields {
					if field.Key == constants.ResourceNameFieldRef && field.Operator == corev1.NodeSelectorOpIn {
						return true
					}
				}
				for _, expr := range term.MatchExpressions {
					if expr.Key == constants.KubernetesHostNameLabel && expr.Operator == corev1.NodeSelectorOpIn {
						return true
					}
				}
			}
		}
	}
	return false
}

func GetInitialGPUNodeSelector() []string {
	selector := os.Getenv("INITIAL_GPU_NODE_LABEL_SELECTOR")
	if selector == "" {
		return nil
	}
	selectors := strings.Split(selector, "=")
	if len(selectors) != 2 {
		return nil
	}
	return selectors
}

var GPUResourceNames = []corev1.ResourceName{
	constants.NvidiaGPUKey,
	constants.AmdGPUKey,
}

func containsGPUResources(res corev1.ResourceList) bool {
	for _, gpuResourceName := range GPUResourceNames {
		quantity, ok := res[gpuResourceName]
		if ok && quantity.Sign() > 0 {
			return true
		}
	}
	return false
}

// AppendEnvVarsIfNotExists appends environment variables to the slice only if they don't already exist (by name).
// It returns the updated slice with new env vars appended.
func AppendEnvVarsIfNotExists(envVars []corev1.EnvVar, newEnvVars ...corev1.EnvVar) []corev1.EnvVar {
	existingNames := make(map[string]bool)
	for _, env := range envVars {
		existingNames[env.Name] = true
	}

	for _, newEnv := range newEnvVars {
		if !existingNames[newEnv.Name] {
			envVars = append(envVars, newEnv)
			existingNames[newEnv.Name] = true
		}
	}

	return envVars
}
