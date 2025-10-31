package utils

import (
	"context"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// FindRootOwnerReference recursively finds the root owner reference for a given object (e.g. Pod).
func FindRootOwnerReference(ctx context.Context, c client.Client, namespace string, obj metav1.Object) (*metav1.OwnerReference, error) {
	owners := obj.GetOwnerReferences()
	if len(owners) == 0 {
		return nil, nil
	}
	current := obj
	for {
		owners := current.GetOwnerReferences()
		// if no owner, return self
		if len(owners) == 0 {
			var apiVersion, kind string
			if rObj, ok := current.(runtime.Object); ok {
				gvk := rObj.GetObjectKind().GroupVersionKind()
				apiVersion = gvk.GroupVersion().String()
				kind = gvk.Kind
			}

			selfRef := metav1.OwnerReference{
				APIVersion: apiVersion,
				Kind:       kind,
				Name:       current.GetName(),
				UID:        current.GetUID(),
			}
			return &selfRef, nil
		}

		// prefer ownerRef with controller=true
		var ownerRef metav1.OwnerReference
		foundController := false
		for _, ref := range owners {
			if ref.Controller != nil && *ref.Controller {
				ownerRef = ref
				foundController = true
				break
			}
		}
		if !foundController {
			ownerRef = owners[0]
		}

		unObj := &unstructured.Unstructured{}
		unObj.SetAPIVersion(ownerRef.APIVersion)
		unObj.SetKind(ownerRef.Kind)
		key := client.ObjectKey{Name: ownerRef.Name, Namespace: namespace}
		err := c.Get(ctx, key, unObj)
		if err != nil {
			// if not found, return ownerRef as root
			if errors.IsNotFound(err) {
				return &ownerRef, nil
			}
			return nil, fmt.Errorf("get owner object: %w", err)
		}

		// Cast back to metav1.Object if possible
		if metaObj, ok := any(unObj).(metav1.Object); ok {
			current = metaObj
		} else {
			return nil, fmt.Errorf("unexpected type for owner object %s/%s", ownerRef.Kind, ownerRef.Name)
		}
	}
}

// FindFirstLevelOwnerReference recursively finds the root owner reference for a given object (e.g. Pod).
func FindFirstLevelOwnerReference(obj metav1.Object) *metav1.OwnerReference {
	owners := obj.GetOwnerReferences()
	if len(owners) == 0 {
		if obj.GetUID() == "" {
			return nil
		}
		return &metav1.OwnerReference{
			APIVersion: "v1",
			Kind:       "Pod",
			Name:       obj.GetName(),
			UID:        obj.GetUID(),
			Controller: ptr.To(true),
		}
	}
	ownerRef := owners[0]
	if ownerRef.UID == "" {
		return nil
	}
	return &ownerRef
}

// FindRootControllerRef recursively finds the root controller reference for a given object (e.g. Pod).
func FindRootControllerRef(ctx context.Context, c client.Client, obj metav1.Object) (*metav1.OwnerReference, error) {
	if metav1.GetControllerOfNoCopy(obj) == nil {
		return nil, nil
	}

	namespace := obj.GetNamespace()
	current := obj
	for {
		controllerRef := metav1.GetControllerOf(current)
		if controllerRef == nil {
			if rObj, ok := current.(runtime.Object); ok {
				gvk := rObj.GetObjectKind().GroupVersionKind()
				return metav1.NewControllerRef(current, gvk), nil
			} else {
				return nil, fmt.Errorf("not a runtime.Object")
			}
		}

		unObj := &unstructured.Unstructured{}
		unObj.SetAPIVersion(controllerRef.APIVersion)
		unObj.SetKind(controllerRef.Kind)
		err := c.Get(ctx, client.ObjectKey{Name: controllerRef.Name, Namespace: namespace}, unObj)
		if err != nil {
			// if not found, return controllerRef as root
			if errors.IsNotFound(err) {
				return controllerRef, nil
			}
			return nil, fmt.Errorf("get controller object: %w", err)
		}

		// Cast back to metav1.Object if possible
		if metaObj, ok := any(unObj).(metav1.Object); ok {
			current = metaObj
		} else {
			return nil, fmt.Errorf("unexpected type for controller object %s/%s", controllerRef.Kind, controllerRef.Name)
		}
	}
}

// GetPodControllerRef returns the controller reference for a Pod.
// For Pods that are indirectly controlled (e.g., by a Deployment or CronJob), return the indirect controller.
// For other cases, it returns the direct controller reference of the Pod.
// If the Pod has no controller reference, it returns nil.
func GetPodControllerRef(ctx context.Context, c client.Client, pod *corev1.Pod) (*metav1.OwnerReference, error) {
	podControllerRef := metav1.GetControllerOf(pod)
	if podControllerRef == nil {
		return nil, nil
	}

	getControllerRef := func(obj client.Object) (*metav1.OwnerReference, error) {
		if err := c.Get(ctx, client.ObjectKey{
			Namespace: pod.Namespace,
			Name:      podControllerRef.Name,
		}, obj); err != nil {
			if errors.IsNotFound(err) {
				return podControllerRef, nil
			}
			return nil, fmt.Errorf("failed to get %T: %w", obj, err)
		}
		return metav1.GetControllerOf(obj), nil
	}

	switch podControllerRef.Kind {
	case "ReplicaSet":
		if parentRef, err := getControllerRef(&appsv1.ReplicaSet{}); err != nil {
			return nil, err
		} else if parentRef != nil && parentRef.Kind == "Deployment" {
			return parentRef, nil
		}

	case "Job":
		if parentRef, err := getControllerRef(&batchv1.Job{}); err != nil {
			return nil, err
		} else if parentRef != nil && parentRef.Kind == "CronJob" {
			return parentRef, nil
		}
	}

	return podControllerRef, nil
}
