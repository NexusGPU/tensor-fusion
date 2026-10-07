package kubernetes

import (
	"context"
	"fmt"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	authv1 "k8s.io/api/authentication/v1"
)

// AuthenticatePodToken validates the signature, expiry, audience and bound Pod
// with the API server. Only the verified Pod UID is used for allocation lookup.
func (b *KubeletBackend) AuthenticatePodToken(ctx context.Context, token string) (string, error) {
	review := &authv1.TokenReview{Spec: authv1.TokenReviewSpec{Token: token}}
	if err := b.apiClient.client.Create(ctx, review); err != nil {
		return "", fmt.Errorf("review Pod token: %w", err)
	}
	uids := review.Status.User.Extra[constants.ExtraVerificationInfoPodIDKey]
	if !review.Status.Authenticated || review.Status.Error != "" || len(uids) != 1 || uids[0] == "" {
		return "", fmt.Errorf("token is not authenticated as a bound Pod")
	}
	return uids[0], nil
}
