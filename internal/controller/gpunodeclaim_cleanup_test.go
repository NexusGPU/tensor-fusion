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
