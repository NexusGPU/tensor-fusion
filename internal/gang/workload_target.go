package gang

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// GangEnabledForWorkload returns true only when gang scheduling is explicitly enabled
// and the workload currently needs at least two members.
func GangEnabledForWorkload(workload *tfv1.TensorFusionWorkload, desiredMembers int32) bool {
	if workload == nil || workload.Spec.GangScheduling == nil {
		return false
	}
	return desiredMembers >= 2
}

// ResolveRequiredMembers returns the effective gang quorum.
// When gang scheduling is enabled and MinMembers is omitted/zero,
// the workload's desired replicas are used as the quorum.
func ResolveRequiredMembers(workload *tfv1.TensorFusionWorkload, desiredMembers int32) int32 {
	if workload == nil || workload.Spec.GangScheduling == nil || desiredMembers < 2 {
		return 0
	}
	if workload.Spec.GangScheduling.MinMembers >= 2 {
		return workload.Spec.GangScheduling.MinMembers
	}
	return desiredMembers
}

// ResolveDesiredMembers returns the workload's current desired gang size.
// It prefers the workload spec, then the owner controller's desired replicas,
// and finally falls back to the observed worker count.
func ResolveDesiredMembers(ctx context.Context, c client.Client, workload *tfv1.TensorFusionWorkload) int32 {
	if workload == nil {
		return 0
	}
	if workload.Spec.Replicas != nil {
		return nonNegative(*workload.Spec.Replicas)
	}
	if c != nil {
		if replicas, ok := resolveDesiredMembersFromOwner(ctx, c, workload); ok {
			return replicas
		}
	}
	return nonNegative(workload.Status.WorkerCount)
}

func resolveDesiredMembersFromOwner(ctx context.Context, c client.Client, workload *tfv1.TensorFusionWorkload) (int32, bool) {
	ownerRef := metav1.GetControllerOfNoCopy(workload)
	if ownerRef == nil && len(workload.OwnerReferences) > 0 {
		ownerRef = &workload.OwnerReferences[0]
	}
	if ownerRef == nil {
		return 0, false
	}

	owner := &unstructured.Unstructured{}
	owner.SetAPIVersion(ownerRef.APIVersion)
	owner.SetKind(ownerRef.Kind)
	if err := c.Get(ctx, client.ObjectKey{Namespace: workload.Namespace, Name: ownerRef.Name}, owner); err != nil {
		return 0, false
	}

	if replicas, found, err := unstructured.NestedInt64(owner.Object, "spec", "replicas"); err == nil && found {
		return nonNegative(int32(replicas)), true
	}
	if parallelism, found, err := unstructured.NestedInt64(owner.Object, "spec", "parallelism"); err == nil && found {
		return nonNegative(int32(parallelism)), true
	}
	if completions, found, err := unstructured.NestedInt64(owner.Object, "spec", "completions"); err == nil && found {
		return nonNegative(int32(completions)), true
	}
	return 0, false
}

func nonNegative(v int32) int32 {
	if v < 0 {
		return 0
	}
	return v
}
