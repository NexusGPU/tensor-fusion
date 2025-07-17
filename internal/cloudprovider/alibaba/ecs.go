package alibaba

import (
	"context"
	"encoding/base64"
	"fmt"
	"strconv"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	common "github.com/NexusGPU/tensor-fusion/internal/cloudprovider/common"
	types "github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/aliyun/alibaba-cloud-sdk-go/sdk/requests"
	"github.com/aliyun/alibaba-cloud-sdk-go/services/ecs"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

var cachedClient *ecs.Client

type AlibabaGPUNodeProvider struct {
	client    *ecs.Client
	nodeClass *tfv1.GPUNodeClass
	ctx       context.Context
}

func NewAlibabaGPUNodeProvider(ctx context.Context, config tfv1.ComputingVendorConfig, nodeClass *tfv1.GPUNodeClass) (AlibabaGPUNodeProvider, error) {

	var provider AlibabaGPUNodeProvider

	if cachedClient != nil {
		provider.client = cachedClient
		return provider, nil
	}

	if config.AuthType != tfv1.AuthTypeAccessKey {
		return provider, fmt.Errorf("unsupported auth type for alibaba cloud: %s", config.AuthType)
	}
	ak, err := common.GetAccessKeyOrSecretFromPath(
		config.Params.AccessKeyPath,
	)
	if err != nil {
		return provider, err
	}
	sk, err := common.GetAccessKeyOrSecretFromPath(
		config.Params.SecretKeyPath,
	)

	if ak == "" || sk == "" {
		return provider, fmt.Errorf("empty access key or secret key, can not create alicloud provider")
	}

	if err != nil {
		return provider, err
	}

	client, err := ecs.NewClientWithAccessKey(config.Params.DefaultRegion, ak, sk)
	if err != nil {
		return provider, fmt.Errorf("failed to create ECS client: %w", err)
	}

	provider.client = client

	if err := provider.TestConnection(); err != nil {
		return provider, err
	}

	cachedClient = client

	provider.nodeClass = nodeClass
	provider.ctx = ctx
	return provider, nil
}

func (p AlibabaGPUNodeProvider) TestConnection() error {
	request := ecs.CreateDescribeRegionsRequest()
	_, err := p.client.DescribeRegions(request)
	if err != nil {
		return fmt.Errorf("can not connect to Aliyun ECS API: %v", err)
	}
	log.FromContext(p.ctx).Info("Successfully connected to Aliyun ECS. Available regions got")
	return nil
}

func (p AlibabaGPUNodeProvider) CreateNode(ctx context.Context, claim *tfv1.GPUNodeClaim) (*types.GPUNodeStatus, error) {
	param := claim.Spec
	nodeClass := p.nodeClass.Spec
	request := ecs.CreateRunInstancesRequest()
	request.LaunchTemplateId = nodeClass.LaunchTemplate.ID
	request.ClientToken = param.NodeName

	if len(nodeClass.OSImageSelectorTerms) > 0 {
		// TODO: should support other query types not only ID
		request.ImageId = nodeClass.OSImageSelectorTerms[0].ID
	}
	request.InstanceType = param.InstanceType
	request.InstanceName = param.NodeName
	request.RegionId = param.Region
	request.Amount = "1"

	if err := p.handleNodeClassAndExtraParams(request, &param); err != nil {
		return nil, err
	}

	tag := []ecs.RunInstancesTag{
		{Key: "managed-by", Value: "tensor-fusion.ai"},
		{Key: "tensor-fusion.ai/node-name", Value: param.NodeName},
		{Key: "tensor-fusion.ai/node-class", Value: p.nodeClass.Name},
	}
	for k, v := range nodeClass.Tags {
		tag = append(tag, ecs.RunInstancesTag{
			Key:   k,
			Value: v,
		})
	}
	request.Tag = &tag

	// TODO: handle other extra params which contains OS constraints, charging type, security group, volume type, user data, vpc/subnet etc.

	response, err := p.client.RunInstances(request)
	if err != nil {
		return nil, fmt.Errorf("failed to create instance: %w", err)
	}
	if !response.IsSuccess() {
		return nil, fmt.Errorf("instance creation failed: %s", response.String())
	}

	return &types.GPUNodeStatus{
		InstanceID: response.InstanceIdSets.InstanceIdSet[0],
		CreatedAt:  time.Now(),
	}, nil
}

func (p AlibabaGPUNodeProvider) TerminateNode(ctx context.Context, param *types.NodeIdentityParam) error {
	request := ecs.CreateDeleteInstanceRequest()
	request.InstanceId = param.InstanceID
	request.RegionId = param.Region
	request.Force = requests.NewBoolean(true)
	response, err := p.client.DeleteInstance(request)
	if err != nil {
		return fmt.Errorf("failed to terminate instance: %w", err)
	}
	if !response.IsSuccess() {
		return fmt.Errorf("instance termination failed: %s", response.String())
	}
	return nil
}

func (p AlibabaGPUNodeProvider) GetNodeStatus(ctx context.Context, param *types.NodeIdentityParam) (*types.GPUNodeStatus, error) {
	request := ecs.CreateDescribeInstancesRequest()
	request.InstanceIds = fmt.Sprintf("[\"%s\"]", param.InstanceID)
	response, err := p.client.DescribeInstances(request)
	if err != nil {
		return nil, fmt.Errorf("failed to describe instance: %w", err)
	}
	if len(response.Instances.Instance) == 0 {
		return nil, fmt.Errorf("instance not found")
	}
	instance := response.Instances.Instance[0]
	createTime, _ := time.Parse(time.RFC3339, instance.CreationTime)

	privateIP := ""
	publicIP := ""
	if len(instance.PublicIpAddress.IpAddress) > 0 {
		publicIP = instance.PublicIpAddress.IpAddress[0]
	}
	if len(instance.VpcAttributes.PrivateIpAddress.IpAddress) > 0 {
		privateIP = instance.VpcAttributes.PrivateIpAddress.IpAddress[0]
	}

	status := &types.GPUNodeStatus{
		InstanceID: instance.InstanceId,
		CreatedAt:  createTime,
		PrivateIP:  privateIP,
		PublicIP:   publicIP,
	}
	return status, nil
}

func (p AlibabaGPUNodeProvider) handleNodeClassAndExtraParams(request *ecs.RunInstancesRequest, param *tfv1.GPUNodeClaimSpec) error {
	nodeClass := p.nodeClass.Spec
	if len(nodeClass.SecurityGroupSelectorTerms) > 0 {
		request.SecurityGroupId = nodeClass.SecurityGroupSelectorTerms[0].ID
	}
	if len(nodeClass.SubnetSelectorTerms) > 0 {
		request.VSwitchId = nodeClass.SubnetSelectorTerms[0].ID
	}

	if len(nodeClass.BlockDeviceMappings) > 0 {
		perfLevel := "PL0"
		if param.ExtraParams["dataDiskPerformanceLevel"] != "" {
			perfLevel = param.ExtraParams["dataDiskPerformanceLevel"]
		}
		dataDisks := []ecs.RunInstancesDataDisk{}
		for _, ebsMapping := range nodeClass.BlockDeviceMappings {
			dataDisks = append(dataDisks, ecs.RunInstancesDataDisk{
				DiskName:           ebsMapping.DeviceName,
				Size:               ebsMapping.EBS.VolumeSize,
				Category:           ebsMapping.EBS.VolumeType,
				PerformanceLevel:   perfLevel,
				DeleteWithInstance: strconv.FormatBool(ebsMapping.EBS.DeleteOnTermination),
				Encrypted:          strconv.FormatBool(ebsMapping.EBS.Encrypted),
			})
		}
		request.DataDisk = &dataDisks
	}

	// Add best practices
	request.InternetMaxBandwidthOut = requests.NewInteger(100)
	request.InternetChargeType = "PayByTraffic"
	request.Description = "GPU node managed by TensorFusion NodeClass: " + p.nodeClass.Name

	// Add user data, replace placeholder is very important, so that to build the mapping between GPUNode and real Kubernetes node
	request.UserData = base64.StdEncoding.EncodeToString([]byte(strings.ReplaceAll(nodeClass.UserData, constants.ProvisionerNamePlaceholder, param.NodeName)))

	// Handle extra params
	capacityType := param.CapacityType
	if capacityType != "" && capacityType != tfv1.CapacityTypeOnDemand {
		// Convert from Spot/OnDemand to each cloud vendor's equivalent
		if param.ExtraParams["spotPriceLimit"] != "" {
			priceLimit, err := strconv.ParseFloat(param.ExtraParams["spotPriceLimit"], 64)
			if err != nil {
				return err
			}
			request.SpotPriceLimit = requests.NewFloat(priceLimit)
			request.SpotStrategy = "SpotWithPriceLimit"
		} else {
			request.SpotStrategy = "SpotAsPriceGo"

		}
		if param.ExtraParams["spotDuration"] != "" {
			duration, err := strconv.Atoi(param.ExtraParams["spotDuration"])
			if err != nil {
				return err
			}
			request.SpotDuration = requests.NewInteger(duration)
		} else {
			request.SpotDuration = requests.NewInteger(0)
		}

		// Could be Stop or Terminate in alicloud
		if param.ExtraParams["spotInterruptionBehavior"] != "" {
			request.SpotInterruptionBehavior = param.ExtraParams["spotInterruptionBehavior"]
		}
	}
	if param.ExtraParams["keyPairName"] != "" {
		request.KeyPairName = param.ExtraParams["keyPairName"]
	}
	if param.ExtraParams["systemDiskCategory"] != "" {
		request.SystemDiskCategory = param.ExtraParams["systemDiskCategory"]
	} else {
		request.SystemDiskCategory = "cloud_essd"
	}
	if param.ExtraParams["systemDiskSize"] != "" {
		request.SystemDiskSize = param.ExtraParams["systemDiskSize"]
	} else {
		// 40G is enough for most cases
		request.SystemDiskSize = "40"
	}
	return nil
}
