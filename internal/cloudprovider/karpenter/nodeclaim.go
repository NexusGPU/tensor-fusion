package karpenter

import (
	"context"
	"fmt"
	"log"
	"maps"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/pricing"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	"github.com/mitchellh/mapstructure"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	karpv1 "sigs.k8s.io/karpenter/pkg/apis/v1"
)

// KarpenterConfig holds Karpenter-specific configuration parsed from ExtraParams
type KarpenterConfig struct {
	NodeClassRef struct {
		Name    string `mapstructure:"name"`
		Group   string `mapstructure:"group"`
		Version string `mapstructure:"version"`
		Kind    string `mapstructure:"kind"`
	} `mapstructure:"nodeclassref"`

	// NodeClaim configuration
	NodeClaim struct {
		TerminationGracePeriod string `mapstructure:"terminationgraceperiod"`
	} `mapstructure:"nodeclaim"`
	// Additional Karpenter settings
	// GPU resource name, default to "nvidia.com/gpu"
	GPUResourceName corev1.ResourceName `mapstructure:"gpuresource"`
}

type KarpenterGPUNodeProvider struct {
	client            client.Client
	config            tfv1.ComputingVendorConfig
	nodeManagerConfig *tfv1.NodeManagerConfig
	pricingProvider   *pricing.StaticPricingProvider
}

func NewKarpenterGPUNodeProvider(cfg tfv1.ComputingVendorConfig, client client.Client, nodeManagerConfig tfv1.NodeManagerConfig) (KarpenterGPUNodeProvider, error) {
	// Validate configuration
	if cfg.Type != tfv1.ComputingVendorKarpenter {
		return KarpenterGPUNodeProvider{}, fmt.Errorf("invalid computing vendor type: expected 'karpenter', got '%s'", cfg.Type)
	}
	if client == nil {
		return KarpenterGPUNodeProvider{}, fmt.Errorf("kubernetes client cannot be nil")
	}
	pricingProvider := pricing.NewStaticPricingProvider()
	// Initialize the Karpenter GPU Node Provider with the provided client
	return KarpenterGPUNodeProvider{
		client:          client,
		config:          cfg,
		pricingProvider: pricingProvider,
	}, nil
}

func (p KarpenterGPUNodeProvider) TestConnection() error {
	if p.client == nil {
		return fmt.Errorf("kubernetes client is not initialized")
	}
	exist := true
	// check if NodeClaim CRD exists
	if err := p.client.List(context.Background(), &karpv1.NodeClaimList{}); err != nil {
		log.Printf("karpenter NodeClaim CRD not found.")
		exist = false
	}
	if exist {
		return nil
	}
	// check if NodePool CRD exists
	if err := p.client.List(context.Background(), &karpv1.NodePoolList{}); err != nil {
		log.Printf("karpenter NodePool CRD not found.")
		return fmt.Errorf("karpenter CRD not found or not accessible")
	}
	return nil
}

func (p KarpenterGPUNodeProvider) CreateNode(ctx context.Context, param *types.NodeCreationParam) (*types.GPUNodeStatus, error) {
	if param == nil || p.client == nil {
		return nil, fmt.Errorf("NodeCreationParam cannot be nil")
	}
	// Build NodeClaim from the creation parameters
	nodeClaim, err := p.buildNodeClaim(ctx, param)
	if err != nil {
		return nil, fmt.Errorf("failed to build NodeClaim for node %s: %v", param.NodeName, err)
	}
	// Create the NodeClaim using the Karpenter client
	err = p.client.Create(ctx, nodeClaim)
	if err != nil {
		return nil, fmt.Errorf("failed to create NodeClaim %s: %v", nodeClaim.Name, err)
	}

	// Log the NodeClaim creation
	fmt.Printf("Created NodeClaim: %s with instance type: %s in region: %s\n",
		nodeClaim.Name, param.InstanceType, param.Region)

	// Return the initial status - actual node creation will be handled by Karpenter
	status := &types.GPUNodeStatus{
		InstanceID: param.NodeName, // Use nodename as instance ID
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
	fmt.Printf("Terminated NodeClaim for instance: %s in region: %s\n", param.InstanceID, param.Region)

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
		// Node not register
		log.Printf("Node %q not registered in cluster", param.InstanceID)
	}
	return status, nil
}

func (p KarpenterGPUNodeProvider) GetInstancePricing(instanceType string, region string, capacityType types.CapacityTypeEnum) (float64, error) {
	// Use the static pricing provider for calculations
	if price, exists := p.pricingProvider.GetPringcing(instanceType, capacityType); exists {
		return price, nil
	}
	return 0.0, fmt.Errorf("no on-demand pricing found for instance type %s in region %s", instanceType, region)
}

func (p KarpenterGPUNodeProvider) GetGPUNodeInstanceTypeInfo(region string) []types.GPUNodeInstanceInfo {
	instanceTypes, exists := p.pricingProvider.GetGPUNodeInstanceTypeInfo(region)
	if !exists {
		log.Printf("no instance type info found for region %s", region)
		return []types.GPUNodeInstanceInfo{} // avoid panic
	}
	return instanceTypes
}

// parseKarpenterConfig extracts Karpenter-specific configuration from ExtraParams
func (p KarpenterGPUNodeProvider) parseKarpenterConfig(param *types.NodeCreationParam) *KarpenterConfig {
	karpenterConfig := &KarpenterConfig{}
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
		return &KarpenterConfig{}
	}

	if err := decoder.Decode(karpenterData); err != nil {
		// if decode failed, return empty config
		return &KarpenterConfig{}
	}

	if karpenterConfig.GPUResourceName == "" {
		karpenterConfig.GPUResourceName = "nvidia.com/gpu"
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

func (p KarpenterGPUNodeProvider) buildNodeClaim(ctx context.Context, param *types.NodeCreationParam) (*karpv1.NodeClaim, error) {
	if param == nil {
		return nil, fmt.Errorf("create NodeClaim failed, NodeCreationParam cannot be nil")
	}
	if p.nodeManagerConfig == nil {
		return nil, fmt.Errorf("create NodeClaim failed, NodeManagerConfig cannot be nil")
	}

	// Parse Karpenter configuration from ExtraParams
	karpenterConfig := p.parseKarpenterConfig(param)

	if karpenterConfig.NodeClassRef.Name == "" {
		return nil, fmt.Errorf("NodeClass name is required")
	}

	// Build resource requirements - only include resources that Karpenter understands
	resourceRequests := corev1.ResourceList{}

	// Add GPU resources if specified (Karpenter supports nvidia.com/gpu)
	if param.GPUDeviceOffered > 0 {
		resourceRequests[karpenterConfig.GPUResourceName] = resource.MustParse(fmt.Sprintf("%d", param.GPUDeviceOffered))
	}

	// query nodeClass and build NodeClassRef
	nodeClassRef, err := p.queryAndBuildNodeClassRef(ctx, *karpenterConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to query NodeClass %s: %v", karpenterConfig.NodeClassRef.Name, err)
	}

	// Create NodeClaim using Karpenter's official type
	nodeClaim := &karpv1.NodeClaim{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "karpenter.sh/v1",
			Kind:       "NodeClaim",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: param.NodeName,
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
	// build customer labels
	p.buildCustomerLabels(nodeClaim)
	// build customer taints
	p.buildCustomerTaints(nodeClaim)
	// build termination grace period
	p.buildTerminationGracePeriod(nodeClaim, *karpenterConfig)
	return nodeClaim, nil
}

func (p KarpenterGPUNodeProvider) queryAndBuildNodeClassRef(ctx context.Context, karpenterConfig KarpenterConfig) (*karpv1.NodeClassReference, error) {
	gvk := schema.GroupVersionKind{
		Group:   karpenterConfig.NodeClassRef.Group,
		Version: karpenterConfig.NodeClassRef.Version,
		Kind:    karpenterConfig.NodeClassRef.Kind,
	}
	nodeClass := &unstructured.Unstructured{}
	nodeClass.SetGroupVersionKind(gvk)
	err := p.client.Get(ctx, client.ObjectKey{Name: karpenterConfig.NodeClassRef.Name}, nodeClass)
	if err == nil {
		return &karpv1.NodeClassReference{
			Group: gvk.Group,
			Kind:  gvk.Kind,
			Name:  karpenterConfig.NodeClassRef.Name,
		}, nil
	} else if !apierrors.IsNotFound(err) {
		return nil, fmt.Errorf("error querying %s/%s %s: %v", gvk.Group, gvk.Kind, karpenterConfig.NodeClassRef.Name, err)
	}
	return nil, fmt.Errorf("NodeClass %s %s %s %s not found", gvk.Group, gvk.Version, gvk.Kind, karpenterConfig.NodeClassRef.Name)
}

func (p KarpenterGPUNodeProvider) buildRequirements(nodeClaim *karpv1.NodeClaim, param *types.NodeCreationParam) {
	// Build node selector requirements using Karpenter's NodeSelectorRequirement
	requirements := []karpv1.NodeSelectorRequirementWithMinValues{}
	// 1. instance type
	if param.InstanceType != "" {
		requirements = append(requirements, karpv1.NodeSelectorRequirementWithMinValues{
			NodeSelectorRequirement: corev1.NodeSelectorRequirement{
				Key:      string(tfv1.NodeRequirementKeyInstanceType),
				Operator: corev1.NodeSelectorOpIn,
				Values:   []string{param.InstanceType},
			},
		})
	}
	// 2. zone
	if param.Zone != "" {
		requirements = append(requirements, karpv1.NodeSelectorRequirementWithMinValues{
			NodeSelectorRequirement: corev1.NodeSelectorRequirement{
				Key:      string(tfv1.NodeRequirementKeyZone),
				Operator: corev1.NodeSelectorOpIn,
				Values:   []string{param.Zone},
			},
		})
	}

	// 3. region
	if param.Region != "" {
		requirements = append(requirements, karpv1.NodeSelectorRequirementWithMinValues{
			NodeSelectorRequirement: corev1.NodeSelectorRequirement{
				Key:      string(tfv1.NodeRequirementKeyRegion),
				Operator: corev1.NodeSelectorOpIn,
				Values:   []string{param.Region},
			},
		})
	}

	// 4. customer GPU requirements
	if p.nodeManagerConfig.NodeProvisioner.GPURequirements != nil {
		for _, requirement := range p.nodeManagerConfig.NodeProvisioner.GPURequirements {
			requirements = append(requirements, karpv1.NodeSelectorRequirementWithMinValues{
				NodeSelectorRequirement: corev1.NodeSelectorRequirement{
					Key:      string(requirement.Key),
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

func (p KarpenterGPUNodeProvider) buildCustomerLabels(nodeClaim *karpv1.NodeClaim) {
	if p.nodeManagerConfig.NodeProvisioner.GPULabels == nil {
		return
	}
	if nodeClaim.Labels == nil {
		nodeClaim.Labels = make(map[string]string)
	}
	maps.Copy(nodeClaim.Labels, p.nodeManagerConfig.NodeProvisioner.GPULabels)
}

func (p KarpenterGPUNodeProvider) buildCustomerTaints(nodeClaim *karpv1.NodeClaim) {
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

func (p KarpenterGPUNodeProvider) buildTerminationGracePeriod(nodeClaim *karpv1.NodeClaim, karpenterConfig KarpenterConfig) {
	if karpenterConfig.NodeClaim.TerminationGracePeriod == "" {
		return
	}
	duration, err := time.ParseDuration(karpenterConfig.NodeClaim.TerminationGracePeriod)
	if err != nil {
		// Log error and skip setting termination grace period
		log.Printf("Failed to parse termination grace period %s: %v", karpenterConfig.NodeClaim.TerminationGracePeriod, err)
		return
	}
	nodeClaim.Spec.TerminationGracePeriod = &metav1.Duration{
		Duration: duration,
	}
}
