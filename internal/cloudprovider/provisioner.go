package cloudprovider

import (
	"context"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	alibaba "github.com/NexusGPU/tensor-fusion/internal/cloudprovider/alibaba"
	aws "github.com/NexusGPU/tensor-fusion/internal/cloudprovider/aws"
	karpenter "github.com/NexusGPU/tensor-fusion/internal/cloudprovider/karpenter"
	mock "github.com/NexusGPU/tensor-fusion/internal/cloudprovider/mock"
)

func GetProvider(ctx context.Context, config tfv1.ComputingVendorConfig, k8sClient client.Client, nodeManagerConfig *tfv1.NodeManagerConfig) (types.GPUNodeProvider, error) {
	var err error
	var provider types.GPUNodeProvider
	if config.Type == "" {
		return nil, fmt.Errorf("cloud provider type is empty")
	}
	if config.Type == tfv1.ComputingVendorKarpenter {
		return karpenter.NewKarpenterGPUNodeProvider(ctx, config, k8sClient, nodeManagerConfig)
	}

	// for none karpenter vendors, get tensor fusion node class and pass to provider
	nodeClass := &tfv1.GPUNodeClass{}
	err = k8sClient.Get(ctx, client.ObjectKey{
		Name: nodeManagerConfig.NodeProvisioner.NodeClass,
	}, nodeClass)
	if err != nil {
		return nil, fmt.Errorf("failed to get node class %s", nodeManagerConfig.NodeProvisioner.NodeClass)
	}

	switch config.Type {
	case "aws":
		provider, err = aws.NewAWSGPUNodeProvider(ctx, config, nodeClass)
	case "alibaba":
		provider, err = alibaba.NewAlibabaGPUNodeProvider(ctx, config, nodeClass)
	case "mock":
		provider, err = mock.NewMockGPUNodeProvider(config, nodeClass)
	default:
		return nil, fmt.Errorf("unsupported cloud provider: %s", config.Type)
	}
	return provider, err

}
