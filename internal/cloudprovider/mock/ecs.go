package alibaba

import (
	"context"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	types "github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
)

type MockGPUNodeProvider struct {
}

func NewMockGPUNodeProvider(config tfv1.ComputingVendorConfig) (MockGPUNodeProvider, error) {
	var provider MockGPUNodeProvider
	return provider, nil
}

func (p MockGPUNodeProvider) TestConnection() error {
	return nil
}

func (p MockGPUNodeProvider) CreateNode(ctx context.Context, param *types.NodeCreationParam) (*types.GPUNodeStatus, error) {
	// TODO: Mock a Kubernetes node for e2e testing
	return &types.GPUNodeStatus{
		InstanceID: param.NodeName + "-Mock",
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
