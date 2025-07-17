package aws

import (
	"context"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/ec2"
	ec2Types "github.com/aws/aws-sdk-go-v2/service/ec2/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/pricing"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
)

type AWSGPUNodeProvider struct {
	ec2Client       *ec2.Client
	pricingProvider *pricing.StaticPricingProvider
	nodeClass       *tfv1.GPUNodeClass
	ctx             context.Context
}

func NewAWSGPUNodeProvider(ctx context.Context, cfg tfv1.ComputingVendorConfig, nodeClass *tfv1.GPUNodeClass) (AWSGPUNodeProvider, error) {
	// TODO only support IAM role at first, need to assume role if role set with custom role
	awsCfg := aws.Config{
		Region: cfg.Params.DefaultRegion,
	}
	ec2Client := ec2.NewFromConfig(awsCfg)
	pricingProvider := pricing.NewStaticPricingProvider()

	provider := AWSGPUNodeProvider{
		ec2Client:       ec2Client,
		pricingProvider: pricingProvider,
		nodeClass:       nodeClass,
		ctx:             ctx,
	}
	return provider, nil
}

func (p AWSGPUNodeProvider) TestConnection() error {
	_, err := p.ec2Client.DescribeInstances(context.Background(), &ec2.DescribeInstancesInput{})
	return err
}

func (p AWSGPUNodeProvider) CreateNode(ctx context.Context, param *tfv1.GPUNodeClaim) (*types.GPUNodeStatus, error) {
	awsTags := []ec2Types.Tag{
		{Key: aws.String("managed-by"), Value: aws.String("tensor-fusion.ai")},
		{Key: aws.String("tensor-fusion.ai/node-name"), Value: aws.String(param.Spec.NodeName)},
		{Key: aws.String("tensor-fusion.ai/node-class"), Value: aws.String(p.nodeClass.Name)},
	}
	nodeClass := p.nodeClass.Spec

	for k, v := range nodeClass.Tags {
		awsTags = append(awsTags, ec2Types.Tag{
			Key:   aws.String(k),
			Value: aws.String(v),
		})
	}

	// Get OS image ID
	if len(nodeClass.OSImageSelectorTerms) == 0 || nodeClass.OSImageSelectorTerms[0].ID == "" {
		// TODO: should support other definition types such as name/tags and choose one from selector terms
		// currently only support ID
		return nil, fmt.Errorf("no OS image selector terms found")
	}

	input := &ec2.RunInstancesInput{
		ImageId:      &nodeClass.OSImageSelectorTerms[0].ID,
		InstanceType: ec2Types.InstanceType(param.Spec.InstanceType),
		MinCount:     aws.Int32(1),
		MaxCount:     aws.Int32(1),
		TagSpecifications: []ec2Types.TagSpecification{
			{
				ResourceType: ec2Types.ResourceTypeInstance,
				Tags:         awsTags,
			},
		},
	}
	_, err := p.ec2Client.RunInstances(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to create instance: %w", err)
	}
	return &types.GPUNodeStatus{}, nil
}

func (p AWSGPUNodeProvider) TerminateNode(ctx context.Context, param *types.NodeIdentityParam) error {
	input := &ec2.TerminateInstancesInput{
		InstanceIds: []string{param.InstanceID},
	}
	_, err := p.ec2Client.TerminateInstances(ctx, input)
	if err != nil {
		return fmt.Errorf("failed to terminate instance: %w", err)
	}
	return nil
}

func (p AWSGPUNodeProvider) GetNodeStatus(ctx context.Context, param *types.NodeIdentityParam) (*types.GPUNodeStatus, error) {
	// Fetch instance status using DescribeInstances API
	input := &ec2.DescribeInstancesInput{
		InstanceIds: []string{param.InstanceID},
	}
	output, err := p.ec2Client.DescribeInstances(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("failed to describe instance: %w", err)
	}
	if len(output.Reservations) == 0 || len(output.Reservations[0].Instances) == 0 {
		return nil, fmt.Errorf("instance not found")
	}

	instance := output.Reservations[0].Instances[0]

	status := &types.GPUNodeStatus{
		InstanceID: *instance.InstanceId,
		CreatedAt:  *instance.LaunchTime,

		PrivateIP: *instance.PrivateIpAddress,
		PublicIP:  *instance.PublicIpAddress,
	}

	return status, nil
}

func (p AWSGPUNodeProvider) GetInstancePricing(instanceType string, capacityType tfv1.CapacityTypeEnum, region string) (float64, error) {
	if price, exists := p.pricingProvider.GetPricing(instanceType, capacityType, region); exists {
		return price, nil
	}
	return 0, nil
}

func (p AWSGPUNodeProvider) GetGPUNodeInstanceTypeInfo(region string) []types.GPUNodeInstanceInfo {
	instanceTypes, exists := p.pricingProvider.GetRegionalGPUNodeInstanceTypes(region)
	if !exists {
		log.FromContext(p.ctx).Error(nil, "no instance type info found for region", "region", region)
		return []types.GPUNodeInstanceInfo{}
	}
	return instanceTypes
}
