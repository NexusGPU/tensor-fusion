package kubernetes

import (
	"context"
	"errors"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	authv1 "k8s.io/api/authentication/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestAuthenticatePodTokenRequiresVerifiedBoundUID(t *testing.T) {
	for _, tt := range []struct {
		name          string
		authenticated bool
		uids          []string
		reviewError   string
		apiError      error
		wantUID       string
	}{
		{name: "verified Pod", authenticated: true, uids: []string{"pod-uid"}, wantUID: "pod-uid"},
		{name: "invalid signature or expired", uids: []string{"forged-uid"}},
		{name: "unbound service account", authenticated: true},
		{name: "empty UID", authenticated: true, uids: []string{""}},
		{name: "ambiguous UID", authenticated: true, uids: []string{"a", "b"}},
		{name: "review rejected", authenticated: true, uids: []string{"pod-uid"}, reviewError: "rejected"},
		{name: "API unavailable", apiError: errors.New("unavailable")},
	} {
		t.Run(tt.name, func(t *testing.T) {
			reviewed := false
			c := fake.NewClientBuilder().WithScheme(scheme).WithInterceptorFuncs(interceptor.Funcs{
				Create: func(_ context.Context, _ client.WithWatch, obj client.Object, _ ...client.CreateOption) error {
					reviewed = true
					review := obj.(*authv1.TokenReview)
					require.Equal(t, "opaque-token", review.Spec.Token)
					if tt.apiError != nil {
						return tt.apiError
					}
					review.Status.Authenticated = tt.authenticated
					review.Status.Error = tt.reviewError
					review.Status.User.Extra = map[string]authv1.ExtraValue{constants.ExtraVerificationInfoPodIDKey: tt.uids}
					return nil
				},
			}).Build()
			backend := &KubeletBackend{apiClient: NewAPIClient(t.Context(), c)}
			uid, err := backend.AuthenticatePodToken(t.Context(), "opaque-token")
			require.True(t, reviewed)
			require.Equal(t, tt.wantUID, uid)
			if tt.wantUID != "" {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
			}
		})
	}
}
