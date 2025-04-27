package utils

import (
	context "context"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// FindRootOwnerReference recursively finds the root owner reference for a given object (e.g. Pod).
func FindRootOwnerReference(ctx context.Context, c client.Client, namespace string, obj metav1.Object) (*metav1.OwnerReference, error) {
	current := obj
	for {
		owners := current.GetOwnerReferences()
		if len(owners) == 0 {
			return nil, nil // no owner, this is root
		}
		ownerRef := owners[0]
		// Try to get the owner object as unstructured
		unObj := &unstructured.Unstructured{}
		unObj.SetAPIVersion(ownerRef.APIVersion)
		unObj.SetKind(ownerRef.Kind)
		key := client.ObjectKey{Name: ownerRef.Name, Namespace: namespace}
		err := c.Get(ctx, key, unObj)
		if err != nil {
			// If not found, treat this ownerRef as root
			return &ownerRef, nil
		}
		// Cast back to metav1.Object if possible
		if metaObj, ok := any(unObj).(metav1.Object); ok {
			current = metaObj
		} else {
			return &ownerRef, nil
		}
	}
}
