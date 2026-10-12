package expander

import (
	"context"
	"strings"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	errors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	karpv1 "sigs.k8s.io/karpenter/pkg/apis/v1"
)

// constants.Domain is a variable, so these are too; they are never assigned.
var (
	// expansionProtectedUntilAnnotation marks a NodeClaim the expander created
	// and shields from Karpenter's voluntary disruption, and holds the RFC3339
	// time at which the shield ends at the latest. Karpenter copies a
	// NodeClaim's annotations to its Node, so the Node carries the marker too.
	expansionProtectedUntilAnnotation = constants.Domain + "/expansion-protected-until"
	// expansionProtectedForAnnotation names the pod the node was requested
	// for, as <namespace>/<name>/<uid>.
	expansionProtectedForAnnotation = constants.Domain + "/expansion-protected-for"
)

// expansionProtectionTTL bounds how long a node the expander asked for stays
// shielded when the pod it was requested for never lands. A variable so tests
// can shorten it.
var expansionProtectionTTL = 15 * time.Minute

// protectionEndedPodGone is the reason expansionProtectionEnded gives when the
// pod the node was requested for no longer exists.
const protectionEndedPodGone = "pod gone"

// protectExpansionNodeClaim shields a NodeClaim the expander is about to
// create from Karpenter's consolidation.
//
// The node is requested for a pod that Karpenter does not know about: the pod
// is not nominated to the NodeClaim, so from the moment the node is
// initialized until the pod is bound Karpenter sees an empty node and removes
// it once the NodePool's consolidateAfter has passed. The shield is
// karpenter.sh/do-not-disrupt; the two marker annotations record that the
// expander set it, for which pod, and until when, so that it can be lifted by
// any operator instance, including one started later.
func protectExpansionNodeClaim(nodeClaim *karpv1.NodeClaim, pod *corev1.Pod, now time.Time) {
	if nodeClaim.Annotations == nil {
		nodeClaim.Annotations = make(map[string]string, 3)
	}
	nodeClaim.Annotations[karpv1.DoNotDisruptAnnotationKey] = constants.TrueStringValue
	nodeClaim.Annotations[expansionProtectedUntilAnnotation] = now.Add(expansionProtectionTTL).UTC().Format(time.RFC3339)
	nodeClaim.Annotations[expansionProtectedForAnnotation] = pod.Namespace + "/" + pod.Name + "/" + string(pod.UID)
}

// releaseExpansionProtection lifts the shield from every NodeClaim and Node
// whose pod has been bound or is gone, or whose time is up. Everything it
// decides on is read from the annotations, so nothing is lost when the
// operator restarts. boundPodUID names a pod that is known to be bound even if
// the client's cache does not show it yet; it may be empty.
func (e *NodeExpander) releaseExpansionProtection(ctx context.Context, boundPodUID types.UID) {
	now := time.Now()

	// NodeClaims first: a Node is only left alone by Karpenter once the
	// NodeClaim it is derived from no longer carries the shield.
	nodeClaims := &karpv1.NodeClaimList{}
	if err := e.client.List(ctx, nodeClaims, client.HasLabels{constants.KarpenterExpansionLabel}); err != nil {
		if !meta.IsNoMatchError(err) {
			e.logger.Error(err, "failed to list Karpenter NodeClaims for expansion protection release")
		}
	}
	for i := range nodeClaims.Items {
		e.releaseExpansionProtectionOf(ctx, &nodeClaims.Items[i], boundPodUID, now)
	}

	nodes := &corev1.NodeList{}
	if err := e.client.List(ctx, nodes, client.HasLabels{constants.KarpenterExpansionLabel}); err != nil {
		e.logger.Error(err, "failed to list nodes for expansion protection release")
	}
	for i := range nodes.Items {
		e.releaseExpansionProtectionOf(ctx, &nodes.Items[i], boundPodUID, now)
	}
}

func (e *NodeExpander) releaseExpansionProtectionOf(
	ctx context.Context, obj client.Object, boundPodUID types.UID, now time.Time,
) {
	annotations := obj.GetAnnotations()
	if _, protected := annotations[expansionProtectedUntilAnnotation]; !protected {
		// Not shielded by the expander. A do-not-disrupt annotation that
		// someone else put there is theirs to remove.
		return
	}
	reason := e.expansionProtectionEnded(ctx, annotations, boundPodUID, now)
	if reason == "" {
		return
	}

	patch := client.MergeFrom(obj.DeepCopyObject().(client.Object))
	remaining := make(map[string]string, len(annotations))
	for key, value := range annotations {
		switch key {
		case karpv1.DoNotDisruptAnnotationKey, expansionProtectedUntilAnnotation, expansionProtectedForAnnotation:
		default:
			remaining[key] = value
		}
	}
	obj.SetAnnotations(remaining)
	if err := e.client.Patch(ctx, obj, patch); err != nil {
		if !errors.IsNotFound(err) {
			// The markers are still on the object; the next pass retries.
			e.logger.Error(err, "failed to release expansion protection",
				"kind", obj.GetObjectKind().GroupVersionKind().Kind, "name", obj.GetName())
		}
		return
	}
	e.logger.Info("released expansion protection", "name", obj.GetName(),
		"pod", annotations[expansionProtectedForAnnotation], "reason", reason)
}

// expansionProtectionEnded returns why the shield described by the annotations
// is over, or "" while it still holds.
func (e *NodeExpander) expansionProtectionEnded(
	ctx context.Context, annotations map[string]string, boundPodUID types.UID, now time.Time,
) string {
	until, err := time.Parse(time.RFC3339, annotations[expansionProtectedUntilAnnotation])
	if err != nil {
		// An unreadable deadline must not shield the node for ever.
		return "unreadable deadline"
	}
	if !now.Before(until) {
		return "deadline passed"
	}

	parts := strings.Split(annotations[expansionProtectedForAnnotation], "/")
	if len(parts) != 3 {
		return "unreadable pod reference"
	}
	namespace, name, uid := parts[0], parts[1], types.UID(parts[2])
	if boundPodUID != "" && uid == boundPodUID {
		return "pod bound"
	}
	pod := &corev1.Pod{}
	if err := e.client.Get(ctx, client.ObjectKey{Namespace: namespace, Name: name}, pod); err != nil {
		if errors.IsNotFound(err) {
			return protectionEndedPodGone
		}
		// Unknown: keep the shield, the deadline bounds it.
		return ""
	}
	switch {
	case pod.UID != uid:
		return protectionEndedPodGone
	case pod.Spec.NodeName != "":
		return "pod bound"
	case !pod.DeletionTimestamp.IsZero():
		return protectionEndedPodGone
	}
	return ""
}
