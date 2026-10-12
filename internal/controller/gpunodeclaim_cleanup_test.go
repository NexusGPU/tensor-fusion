package controller

import (
	"context"
	"errors"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	providertypes "github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

type cleanupNodeProvider struct {
	providertypes.GPUNodeProvider
	terminated []string
	err        error
}

func (p *cleanupNodeProvider) TerminateNode(_ context.Context, param *providertypes.NodeIdentityParam) error {
	p.terminated = append(p.terminated, param.InstanceID)
	return p.err
}

func TestGPUNodeClaimCleanupTerminatesUnregisteredInstances(t *testing.T) {
	for _, name := range []string{"never registered", "node terminating", "termination failed"} {
		t.Run(name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			var objects []client.Object
			if name == "node terminating" {
				now := metav1.Now()
				objects = append(objects, &corev1.Node{ObjectMeta: metav1.ObjectMeta{
					Name: "node", Labels: map[string]string{constants.ProvisionerLabelKey: "claim"},
					DeletionTimestamp: &now, Finalizers: []string{"test-finalizer"},
				}})
			}
			r := &GPUNodeClaimReconciler{Client: fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).Build()}
			provider := &cleanupNodeProvider{}
			if name == "termination failed" {
				provider.err = errors.New("cloud API unavailable")
			}
			claim := &tfv1.GPUNodeClaim{ObjectMeta: metav1.ObjectMeta{Name: "claim"}}
			claim.Status.InstanceID = "i-created"
			canDelete, err := r.finalizeCloudVendorNode(t.Context(), claim, provider)
			require.Equal(t, []string{"i-created"}, provider.terminated)
			if provider.err != nil {
				require.ErrorIs(t, err, provider.err)
				require.False(t, canDelete, "a failed termination must retain the finalizer")
			} else {
				require.NoError(t, err)
				require.Equal(t, name == "never registered", canDelete)
			}
		})
	}
}

// A pool being deleted must stay until its GPUNodeClaims are gone, also when
// no GPUNode is left: the claim's finalizer needs the pool to terminate the
// cloud node, and is never removed once the pool has disappeared.
func TestPoolCleanupWaitsForGPUNodeClaims(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, tfv1.AddToScheme(scheme))
	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{Name: "pool"},
		Status:     tfv1.GPUPoolStatus{Phase: tfv1.TensorFusionPoolPhaseDestroying},
	}
	claim := &tfv1.GPUNodeClaim{ObjectMeta: metav1.ObjectMeta{
		Name:       "pool-claim",
		Labels:     map[string]string{constants.LabelKeyOwner: pool.Name},
		Finalizers: []string{constants.Finalizer},
	}}
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(pool, claim).Build()
	r := &GPUPoolReconciler{Client: kubeClient, Scheme: scheme}

	done, err := r.cleanUpPool(t.Context(), pool)
	require.NoError(t, err)
	require.False(t, done, "the claim still exists (its finalizer has not run)")
	current := &tfv1.GPUNodeClaim{}
	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(claim), current))
	require.False(t, current.DeletionTimestamp.IsZero(), "the pool asks for the claim's deletion")

	// The claim controller finishes: finalizer removed, object gone.
	current.Finalizers = nil
	require.NoError(t, kubeClient.Update(t.Context(), current))
	done, err = r.cleanUpPool(t.Context(), pool)
	require.NoError(t, err)
	require.True(t, done)
}
