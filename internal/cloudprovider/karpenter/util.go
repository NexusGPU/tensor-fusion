package karpenter

import (
	"context"
	"fmt"
	"maps"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/rand"
	client "sigs.k8s.io/controller-runtime/pkg/client"
	karpv1 "sigs.k8s.io/karpenter/pkg/apis/v1"
)

func CreateNodeClaimFromNode(ctx context.Context, k8sClient client.Client, node *corev1.Node) (*karpv1.NodeClaim, error) {
	// Find the parent NodeClaim for this node
	nodeClaims := &karpv1.NodeClaimList{}
	if err := k8sClient.List(ctx, nodeClaims); err != nil {
		return nil, fmt.Errorf("failed to list NodeClaims: %w", err)
	}

	var parentNodeClaim *karpv1.NodeClaim
	for _, nc := range nodeClaims.Items {
		if nc.Status.NodeName == node.Name {
			parentNodeClaim = &nc
			break
		}
	}

	if parentNodeClaim == nil {
		return nil, fmt.Errorf("parent NodeClaim not found for node %s", node.Name)
	}

	// Clone the NodeClaim with a new name
	newNodeClaim := &karpv1.NodeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:        generateNodeClaimName(parentNodeClaim.Name),
			Namespace:   parentNodeClaim.Namespace,
			Labels:      maps.Clone(parentNodeClaim.Labels),
			Annotations: maps.Clone(parentNodeClaim.Annotations),
		},
		Spec: karpv1.NodeClaimSpec{
			NodeClassRef: parentNodeClaim.Spec.NodeClassRef,
			Requirements: parentNodeClaim.Spec.Requirements,
			Taints:       parentNodeClaim.Spec.Taints,
		},
	}

	// Create the new NodeClaim
	if err := k8sClient.Create(ctx, newNodeClaim); err != nil {
		return nil, fmt.Errorf("failed to create NodeClaim: %w", err)
	}

	return newNodeClaim, nil
}

func generateNodeClaimName(originalName string) string {
	// Find the last "-" and replace what comes after it with a random string
	lastDashIndex := strings.LastIndex(originalName, "-")
	if lastDashIndex == -1 {
		// No dash found, append random string
		return fmt.Sprintf("%s-%s", originalName, rand.String(8))
	}
	prefix := originalName[:lastDashIndex]
	return fmt.Sprintf("%s-%s", prefix, rand.String(8))
}
