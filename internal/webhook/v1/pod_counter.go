package v1

import (
	"context"
	"fmt"
	"strconv"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type TensorFusionPodCounter struct {
	Client client.Client
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

// Increase increases the counter in owner annotation by key
func (c *TensorFusionPodCounter) Increase(ctx context.Context, pod *corev1.Pod) error {
	ownerRef := getControllerOwnerRef(pod)
	if ownerRef == nil {
		return fmt.Errorf("no controller owner reference found for pod %s/%s", pod.Namespace, pod.Name)
	}
	key := getOrGenerateKey(pod)
	ownerObj := &unstructured.Unstructured{}
	ownerObj.SetAPIVersion(ownerRef.APIVersion)
	ownerObj.SetKind(ownerRef.Kind)
	objKey := client.ObjectKey{Name: ownerRef.Name, Namespace: pod.Namespace}
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
	ownerObj.SetAnnotations(annotations)
	if err := c.Client.Update(ctx, ownerObj); err != nil {
		return fmt.Errorf("failed to update owner annotation: %w", err)
	}
	return nil
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
	log := log.FromContext(ctx)
	ownerRef := getControllerOwnerRef(pod)
	if ownerRef == nil {
		log.Error(nil, "no controller owner reference found for pod", "namespace", pod.Namespace, "name", pod.Name)
		return nil
	}
	key := getOrGenerateKey(pod)
	objKey := client.ObjectKey{Name: ownerRef.Name, Namespace: pod.Namespace}

	return retry.RetryOnConflict(retry.DefaultBackoff, func() error {
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

// getControllerOwnerRef returns the controller owner reference of a pod
func getControllerOwnerRef(pod *corev1.Pod) *metav1.OwnerReference {
	for i, ref := range pod.OwnerReferences {
		if ref.Controller != nil && *ref.Controller {
			return &pod.OwnerReferences[i]
		}
	}
	return nil
}
