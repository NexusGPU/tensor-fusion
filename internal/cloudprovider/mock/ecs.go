package mock

import (
	"context"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	types "github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
)

type MockGPUNodeProvider struct {
	nodeClass *tfv1.GPUNodeClass
}

func NewMockGPUNodeProvider(config tfv1.ComputingVendorConfig, nodeClass *tfv1.GPUNodeClass) (MockGPUNodeProvider, error) {
	var provider MockGPUNodeProvider
	provider.nodeClass = nodeClass
	return provider, nil
}

func (p MockGPUNodeProvider) TestConnection() error {
	return nil
}

func (p MockGPUNodeProvider) CreateNode(ctx context.Context, param *tfv1.GPUNodeClaim) (*types.GPUNodeStatus, error) {
	// TODO: Mock a Kubernetes node for e2e testing
	return &types.GPUNodeStatus{
		InstanceID: param.Spec.NodeName + "-Mock",
		CreatedAt:  time.Now(),
	}, nil
}

func (p MockGPUNodeProvider) TerminateNode(ctx context.Context, param *types.NodeIdentityParam) error {
	return nil
}

func (p MockGPUNodeProvider) GetNodeStatus(ctx context.Context, param *types.NodeIdentityParam) (*types.GPUNodeStatus, error) {
	status := &types.GPUNodeStatus{
		InstanceID: param.InstanceID,
		CreatedAt:  time.Now(),
		PrivateIP:  "10.0.0.1",
		PublicIP:   "1.2.3.4",
	}
	return status, nil
}
