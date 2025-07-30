package karpenter

import (
	"context"
	"fmt"
	"maps"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/pricing"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/mitchellh/mapstructure"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	karpv1 "sigs.k8s.io/karpenter/pkg/apis/v1"
)

var (
	initSchemaOnce sync.Once

	KarpenterGroup         = "karpenter.sh"
	KarpenterVersion       = "v1"
	kindNodeClaim          = "NodeClaim"
	DefaultGPUResourceName = "nvidia.com/gpu"
	EC2NodeClassGroup      = "karpenter.k8s.aws"
	// AWS capacity type: https://karpenter.sh/docs/concepts/scheduling/#well-known-labels
	AWSOnDemandType = "on-demand"
	AWSReservedType = "reserved"
	AWSSpotType     = "spot"

	// CapacityTypeMapping maps the CapacityTypeEnum to the corresponding Karpenter capacity type
	CapacityTypeMapping = map[tfv1.CapacityTypeEnum]string{
		tfv1.CapacityTypeOnDemand: AWSOnDemandType,
		tfv1.CapacityTypeSpot:     AWSReservedType,
		tfv1.CapacityTypeReserved: AWSSpotType,
	}
)

// KarpenterExtraConfig holds Karpenter-specific configuration parsed from ExtraParams
type KarpenterExtraConfig struct {
	// NodeClaim configuration
	NodeClaim struct {
		TerminationGracePeriod string `mapstructure:"terminationGracePeriod"`
	} `mapstructure:"nodeClaim"`
	// Additional Karpenter settings
	// GPU resource name, default to "nvidia.com/gpu"
	GPUResourceName corev1.ResourceName `mapstructure:"gpuResource"`
}

type KarpenterGPUNodeProvider struct {
	client            client.Client
	config            tfv1.ComputingVendorConfig
	nodeManagerConfig *tfv1.NodeManagerConfig
	pricingProvider   *pricing.StaticPricingProvider
	ctx               context.Context
}

func NewKarpenterGPUNodeProvider(ctx context.Context, cfg tfv1.ComputingVendorConfig, client client.Client, nodeManagerConfig *tfv1.NodeManagerConfig) (KarpenterGPUNodeProvider, error) {
	// Validate configuration
	if cfg.Type != tfv1.ComputingVendorKarpenter {
		return KarpenterGPUNodeProvider{}, fmt.Errorf("invalid computing vendor type: expected 'karpenter', got '%s'", cfg.Type)
	}
	if client == nil {
		return KarpenterGPUNodeProvider{}, fmt.Errorf("kubernetes client cannot be nil")
	}

	initSchemaOnce.Do(func() {
		scheme := client.Scheme()
		// Add Karpenter v1 types manually
		gv := schema.GroupVersion{Group: KarpenterGroup, Version: KarpenterVersion}
		if !scheme.Recognizes(gv.WithKind(kindNodeClaim)) {
			scheme.AddKnownTypes(gv,
				&karpv1.NodeClaim{}, &karpv1.NodeClaimList{},
				&karpv1.NodePool{}, &karpv1.NodePoolList{},
			)
			metav1.AddToGroupVersion(scheme, gv)
		}
	})

	pricingProvider := pricing.NewStaticPricingProvider()
	// Initialize the Karpenter GPU Node Provider with the provided client
	return KarpenterGPUNodeProvider{
		client:            client,
		config:            cfg,
		nodeManagerConfig: nodeManagerConfig,
		pricingProvider:   pricingProvider,
		ctx:               ctx,
	}, nil
}

func (p KarpenterGPUNodeProvider) TestConnection() error {
	if p.client == nil {
		return fmt.Errorf("kubernetes client is not initialized")
	}
	// check if NodeClaim CRD exists
	if err := p.client.List(context.Background(), &karpv1.NodeClaimList{}); err != nil {
		log.FromContext(p.ctx).Error(err, "karpenter NodeClaim CRD not found.")
		return fmt.Errorf("karpenter nodeclaim CRD not found or not accessible")
	}
	log.FromContext(p.ctx).Info("Test connection to Karpenter nodeclaim succeeded.")
	return nil
}

func (p KarpenterGPUNodeProvider) CreateNode(ctx context.Context, claim *tfv1.GPUNodeClaim) (*types.GPUNodeStatus, error) {
	if claim == nil || p.client == nil {
		return nil, fmt.Errorf("NodeCreationParam cannot be nil")
	}
	param := &claim.Spec
	// Build NodeClaim from the creation parameters
	nodeClaim, err := p.buildNodeClaim(ctx, param)

	if err != nil {
		return nil, fmt.Errorf("failed to build NodeClaim for node %s: %v", param.NodeName, err)
	}
	_ = controllerutil.SetControllerReference(claim, nodeClaim, p.client.Scheme())

	// Create the NodeClaim using the Karpenter client
	err = p.client.Create(ctx, nodeClaim)
	if err != nil {
		return nil, fmt.Errorf("failed to create NodeClaim %s: %v", nodeClaim.Name, err)
	}

	// Log the NodeClaim creation
	log.FromContext(ctx).Info("Created NodeClaim", "nodeClaim", nodeClaim.Name, "instanceType",
		param.InstanceType, "region", param.Region)

	// Return the initial status - actual node creation will be handled by Karpenter
	status := &types.GPUNodeStatus{
		InstanceID: param.NodeName, // Use nodeName as instance ID
		CreatedAt:  metav1.Now().Time,
		PrivateIP:  "", // Will be populated once node is actually provisioned
		PublicIP:   "", // Will be populated once node is actually provisioned
	}
	return status, nil
}

func (p KarpenterGPUNodeProvider) TerminateNode(ctx context.Context, param *types.NodeIdentityParam) error {
	if param == nil {
		return fmt.Errorf("terminate NodeClaim failed, NodeIdentityParam cannot be nil")
	}

	if p.client == nil {
		return fmt.Errorf("terminate NodeClaim failed, kubernetes client is not initialized")
	}
	// Delete the NodeClaim, Karpenter will handle the node termination
	nodeClaim := &karpv1.NodeClaim{}
	err := p.client.Get(ctx, client.ObjectKey{
		Name: param.InstanceID, // Using instance ID as NodeClaim name
	}, nodeClaim)
	if err != nil {
		return fmt.Errorf("failed to find NodeClaim for instance %s: %v", param.InstanceID, err)
	}

	err = p.client.Delete(ctx, nodeClaim)
	if err != nil {
		return fmt.Errorf("failed to delete NodeClaim for instance %s: %v", param.InstanceID, err)
	}
	log.FromContext(ctx).Info("Terminated NodeClaim", "instanceID", param.InstanceID, "region", param.Region)

	return nil
}

func (p KarpenterGPUNodeProvider) GetNodeStatus(ctx context.Context, param *types.NodeIdentityParam) (*types.GPUNodeStatus, error) {
	if param == nil {
		return nil, fmt.Errorf("NodeIdentityParam cannot be nil")
	}

	if p.client == nil {
		return nil, fmt.Errorf("kubernetes client is not initialized")
	}

	// Get the NodeClaim status
	nodeClaim := &karpv1.NodeClaim{}
	err := p.client.Get(ctx, client.ObjectKey{
		Name: param.InstanceID,
	}, nodeClaim)
	if err != nil {
		return nil, fmt.Errorf("failed to get NodeClaim for instance %s: %v", param.InstanceID, err)
	}

	// Convert NodeClaim status to GPUNodeStatus
	status := &types.GPUNodeStatus{
		InstanceID: param.InstanceID,
		CreatedAt:  nodeClaim.CreationTimestamp.Time,
		PrivateIP:  "",
		PublicIP:   "",
	}

	k8sNode := &corev1.Node{}
	if err := p.client.Get(ctx, client.ObjectKey{Name: param.InstanceID}, k8sNode); err == nil {
		// Node exists, fill in IP, status, and architecture
		for _, a := range k8sNode.Status.Addresses {
			switch a.Type {
			case corev1.NodeInternalIP:
				status.PrivateIP = a.Address
			case corev1.NodeExternalIP:
				status.PublicIP = a.Address
			}
		}
	} else if !apierrors.IsNotFound(err) {
		return nil, fmt.Errorf("get Node %q failed: %w", param.InstanceID, err)
	} else {
		// Node not registered
		log.FromContext(ctx).Info("Node not registered in cluster", "instanceID", param.InstanceID)
	}
	return status, nil
}

func (p KarpenterGPUNodeProvider) GetInstancePricing(instanceType string, capacityType tfv1.CapacityTypeEnum, region string) (float64, error) {
	// Use the static pricing provider for calculations
	if price, exists := p.pricingProvider.GetPricing(instanceType, capacityType, region); exists {
		return price, nil
	}
	return 0.0, fmt.Errorf("no on-demand pricing found for instance type %s in region %s", instanceType, region)
}

func (p KarpenterGPUNodeProvider) GetGPUNodeInstanceTypeInfo(region string) []types.GPUNodeInstanceInfo {
	instanceTypes, exists := p.pricingProvider.GetRegionalGPUNodeInstanceTypes(region)
	if !exists {
		log.FromContext(p.ctx).Error(nil, "no instance type info found for region %s", region)
		return []types.GPUNodeInstanceInfo{} // avoid panic
	}
	return instanceTypes
}

// parseKarpenterConfig extracts Karpenter-specific configuration from ExtraParams
func (p KarpenterGPUNodeProvider) parseKarpenterConfig(param *tfv1.GPUNodeClaimSpec) *KarpenterExtraConfig {
	karpenterConfig := &KarpenterExtraConfig{
		GPUResourceName: corev1.ResourceName(DefaultGPUResourceName),
	}
	extraParams := param.ExtraParams
	if extraParams == nil {
		return karpenterConfig
	}
	// map -> structure
	karpenterData := make(map[string]any)

	for key, value := range extraParams {
		if !strings.HasPrefix(key, "karpenter.") {
			continue
		}

		// remove "karpenter." prefix
		karpenterKey := strings.TrimPrefix(key, "karpenter.")

		// build nested map structure
		setNestedValue(karpenterData, karpenterKey, value)
	}

	// decode
	decoder, err := mapstructure.NewDecoder(&mapstructure.DecoderConfig{
		Result:      karpenterConfig,
		TagName:     "mapstructure",
		ErrorUnused: false, // allow unused fields
	})
	if err != nil {
		// if decoder creation failed, return empty config
		return &KarpenterExtraConfig{}
	}

	if err := decoder.Decode(karpenterData); err != nil {
		// if decode failed, return empty config
		return &KarpenterExtraConfig{}
	}

	if karpenterConfig.GPUResourceName == "" {
		karpenterConfig.GPUResourceName = corev1.ResourceName(DefaultGPUResourceName)
	}

	return karpenterConfig
}

// setNestedValue set value in nested map, support dot-separated path
func setNestedValue(data map[string]any, path string, value string) {
	parts := strings.Split(path, ".")
	current := data

	// traverse to the second last layer
	for i := 0; i < len(parts)-1; i++ {
		part := parts[i]
		if _, exists := current[part]; !exists {
			current[part] = make(map[string]any)
		}
		if nextMap, ok := current[part].(map[string]any); ok {
			current = nextMap
		} else {
			// if not map, cannot continue nested
			return
		}
	}
	// set final value
	current[parts[len(parts)-1]] = value
}

func (p KarpenterGPUNodeProvider) buildNodeClaim(ctx context.Context, param *tfv1.GPUNodeClaimSpec) (*karpv1.NodeClaim, error) {
	if param == nil {
		return nil, fmt.Errorf("create NodeClaim failed, NodeCreationParam cannot be nil")
	}
	if p.nodeManagerConfig == nil {
		return nil, fmt.Errorf("create NodeClaim failed, NodeManagerConfig cannot be nil")
	}

	// Parse Karpenter configuration from ExtraParams
	karpenterConfig := p.parseKarpenterConfig(param)

	// Build resource requirements - only include resources that Karpenter understands
	resourceRequests := corev1.ResourceList{}

	// Add GPU resources if specified (Karpenter supports nvidia.com/gpu)
	if param.GPUDeviceOffered > 0 {
		resourceRequests[karpenterConfig.GPUResourceName] = resource.MustParse(fmt.Sprintf("%d", param.GPUDeviceOffered))
	}

	// query nodeClass and build NodeClassRef
	nodeClassRef, err := p.queryAndBuildNodeClassRef(ctx, param)
	if err != nil {
		return nil, fmt.Errorf("failed to query NodeClass %s: %v", param.NodeClassRef.Name, err)
	}

	// Create NodeClaim using Karpenter's official type
	nodeClaim := &karpv1.NodeClaim{
		TypeMeta: metav1.TypeMeta{
			APIVersion: fmt.Sprintf("%s/%s", KarpenterGroup, KarpenterVersion),
			Kind:       kindNodeClaim,
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: param.NodeName,
			Labels: map[string]string{
				// pass through provisioner label to discover its owner
				constants.ProvisionerLabelKey: param.NodeName,
			},
			Annotations: map[string]string{
				// Protection annotation to prevent unexpected deletion by Karpenter
				"karpenter.sh/do-not-disrupt": "true",
			},
		},
		Spec: karpv1.NodeClaimSpec{
			NodeClassRef: nodeClassRef,
			Resources: karpv1.ResourceRequirements{
				Requests: resourceRequests,
			},
		},
	}
	// build requirements
	p.buildRequirements(nodeClaim, param)
	// build custom labels
	p.buildCustomLabels(nodeClaim)
	// build custom taints
	p.buildCustomTaints(nodeClaim)
	// build custom termination grace period
	p.buildTerminationGracePeriod(nodeClaim, *karpenterConfig)
	// build custom annotations
	p.buildCustomAnnotations(nodeClaim)
	return nodeClaim, nil
}

func (p KarpenterGPUNodeProvider) queryAndBuildNodeClassRef(ctx context.Context, param *tfv1.GPUNodeClaimSpec) (*karpv1.NodeClassReference, error) {
	gvk := schema.GroupVersionKind{
		Group:   param.NodeClassRef.Group,
		Kind:    param.NodeClassRef.Kind,
		Version: param.NodeClassRef.Version,
	}
	nodeClass := &unstructured.Unstructured{}
	nodeClass.SetGroupVersionKind(gvk)
	err := p.client.Get(ctx, client.ObjectKey{Name: param.NodeClassRef.Name}, nodeClass)
	if err == nil {
		return &karpv1.NodeClassReference{
			Group: gvk.Group,
			Kind:  gvk.Kind,
			Name:  param.NodeClassRef.Name,
		}, nil
	} else if !apierrors.IsNotFound(err) {
		return nil, fmt.Errorf("error querying %s/%s %s: %v", gvk.Group, gvk.Kind, param.NodeClassRef.Name, err)
	}
	return nil, fmt.Errorf("NodeClass %s %s %s %s not found", gvk.Group, gvk.Version, gvk.Kind, param.NodeClassRef.Name)
}

func (p KarpenterGPUNodeProvider) buildRequirements(nodeClaim *karpv1.NodeClaim, param *tfv1.GPUNodeClaimSpec) {
	// Build node selector requirements using Karpenter's NodeSelectorRequirement
	requirements := []karpv1.NodeSelectorRequirementWithMinValues{}
	seen := make(map[string]struct{})
	// 1. instance type
	if param.InstanceType != "" {
		key := string(tfv1.NodeRequirementKeyInstanceType)
		requirements = append(requirements, karpv1.NodeSelectorRequirementWithMinValues{
			NodeSelectorRequirement: corev1.NodeSelectorRequirement{
				Key:      key,
				Operator: corev1.NodeSelectorOpIn,
				Values:   []string{param.InstanceType},
			},
		})
		seen[key] = struct{}{}
	}
	// 2. zone
	if param.Zone != "" {
		key := string(tfv1.NodeRequirementKeyZone)
		requirements = append(requirements, karpv1.NodeSelectorRequirementWithMinValues{
			NodeSelectorRequirement: corev1.NodeSelectorRequirement{
				Key:      key,
				Operator: corev1.NodeSelectorOpIn,
				Values:   []string{param.Zone},
			},
		})
		seen[key] = struct{}{}
	}

	// 3. region
	if param.Region != "" {
		key := string(tfv1.NodeRequirementKeyRegion)
		requirements = append(requirements, karpv1.NodeSelectorRequirementWithMinValues{
			NodeSelectorRequirement: corev1.NodeSelectorRequirement{
				Key:      key,
				Operator: corev1.NodeSelectorOpIn,
				Values:   []string{param.Region},
			},
		})
		seen[key] = struct{}{}
	}
	// 4. capacity type
	if param.CapacityType != "" {
		key := string(tfv1.NodeRequirementKeyCapacityType)
		value := string(param.CapacityType)
		// if capacity type is not in the mapping, use the original value
		// otherwise, use the mapping value
		// https://github.com/kubernetes-sigs/karpenter/blob/f8da711d7e72b678e77f4758bc73a34ba34286d2/pkg/apis/v1/labels.go#L35
		if _, exists := CapacityTypeMapping[param.CapacityType]; exists {
			value = CapacityTypeMapping[param.CapacityType]
		}

		requirements = append(requirements, karpv1.NodeSelectorRequirementWithMinValues{
			NodeSelectorRequirement: corev1.NodeSelectorRequirement{
				Key:      key,
				Operator: corev1.NodeSelectorOpIn,
				Values:   []string{value},
			},
		})
		seen[key] = struct{}{}
	}

	// 4. custom GPU requirements
	if p.nodeManagerConfig.NodeProvisioner.GPURequirements != nil {
		for _, requirement := range p.nodeManagerConfig.NodeProvisioner.GPURequirements {
			key := string(requirement.Key)
			// remove duplicate requirements
			if _, exists := seen[key]; exists {
				continue
			}
			seen[key] = struct{}{}
			requirements = append(requirements, karpv1.NodeSelectorRequirementWithMinValues{
				NodeSelectorRequirement: corev1.NodeSelectorRequirement{
					Key:      key,
					Operator: requirement.Operator,
					Values:   requirement.Values,
				},
			})
		}
	}
	if len(requirements) > 0 {
		nodeClaim.Spec.Requirements = requirements
	}
}

func (p KarpenterGPUNodeProvider) buildCustomLabels(nodeClaim *karpv1.NodeClaim) {
	if p.nodeManagerConfig.NodeProvisioner.GPULabels == nil {
		return
	}
	if nodeClaim.Labels == nil {
		nodeClaim.Labels = make(map[string]string)
	}
	maps.Copy(nodeClaim.Labels, p.nodeManagerConfig.NodeProvisioner.GPULabels)
}

func (p KarpenterGPUNodeProvider) buildCustomTaints(nodeClaim *karpv1.NodeClaim) {
	if p.nodeManagerConfig.NodeProvisioner.GPUTaints == nil {
		return
	}
	if nodeClaim.Spec.Taints == nil {
		nodeClaim.Spec.Taints = make([]corev1.Taint, 0)
	}
	for _, taint := range p.nodeManagerConfig.NodeProvisioner.GPUTaints {
		nodeClaim.Spec.Taints = append(nodeClaim.Spec.Taints, corev1.Taint{
			Key:    taint.Key,
			Value:  taint.Value,
			Effect: taint.Effect,
		})
	}
}

func (p KarpenterGPUNodeProvider) buildTerminationGracePeriod(nodeClaim *karpv1.NodeClaim, karpenterConfig KarpenterExtraConfig) {
	if karpenterConfig.NodeClaim.TerminationGracePeriod == "" {
		return
	}
	duration, err := time.ParseDuration(karpenterConfig.NodeClaim.TerminationGracePeriod)
	if err != nil {
		// Log error and skip setting termination grace period
		log.FromContext(p.ctx).Error(err, "Failed to parse termination grace period", "terminationGracePeriod", karpenterConfig.NodeClaim.TerminationGracePeriod)
		return
	}
	nodeClaim.Spec.TerminationGracePeriod = &metav1.Duration{
		Duration: duration,
	}
}

func (p KarpenterGPUNodeProvider) buildCustomAnnotations(nodeClaim *karpv1.NodeClaim) {
	if p.nodeManagerConfig.NodeProvisioner.GPUAnnotation == nil {
		return
	}
	if nodeClaim.Annotations == nil {
		nodeClaim.Annotations = make(map[string]string)
	}
	maps.Copy(nodeClaim.Annotations, p.nodeManagerConfig.NodeProvisioner.GPUAnnotation)
}
