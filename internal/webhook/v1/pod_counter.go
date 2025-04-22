package v1

import (
	"context"
	"fmt"
	"strconv"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
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

// Decrease decreases the counter in owner annotation by key
func (c *TensorFusionPodCounter) Decrease(ctx context.Context, pod *corev1.Pod) error {
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
