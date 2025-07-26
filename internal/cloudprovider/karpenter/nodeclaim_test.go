package karpenter

import (
	"context"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/pricing"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/yaml"
)

func TestNewKarpenterGPUNodeProvider(t *testing.T) {
	tests := []struct {
		name          string
		cfg           tfv1.ComputingVendorConfig
		client        client.Client
		nodeConfig    tfv1.NodeManagerConfig
		expectError   bool
		errorContains string
	}{
		{
			name: "valid configuration",
			cfg: tfv1.ComputingVendorConfig{
				Type: tfv1.ComputingVendorKarpenter,
			},
			client:        nil, // Will trigger error in actual implementation
			nodeConfig:    tfv1.NodeManagerConfig{},
			expectError:   true,
			errorContains: "kubernetes client cannot be nil",
		},
		{
			name: "invalid vendor type",
			cfg: tfv1.ComputingVendorConfig{
				Type: "invalid",
			},
			client:        nil,
			nodeConfig:    tfv1.NodeManagerConfig{},
			expectError:   true,
			errorContains: "invalid computing vendor type",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := t.Context()
			provider, err := NewKarpenterGPUNodeProvider(ctx, tt.cfg, tt.client, &tt.nodeConfig)
			if tt.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorContains)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, provider)
				assert.Equal(t, tt.cfg, provider.config)
				assert.Equal(t, tt.client, provider.client)
			}
		})
	}
}

func TestKarpenterGPUNodeProvider_TestConnection(t *testing.T) {
	tests := []struct {
		name          string
		client        client.Client
		expectError   bool
		errorContains string
	}{
		{
			name:          "nil client",
			client:        nil,
			expectError:   true,
			errorContains: "kubernetes client is not initialized",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider := KarpenterGPUNodeProvider{
				client: tt.client,
			}
			err := provider.TestConnection()
			if tt.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errorContains)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestKarpenterGPUNodeProvider_CreateNode(t *testing.T) {
	// Create a simple fake kubernetes client
	scheme := runtime.NewScheme()
	// Add core types to scheme
	scheme.AddKnownTypes(corev1.SchemeGroupVersion, &corev1.Node{}, &corev1.NodeList{})

	// Create a simple fake NodeClass using unstructured with basic fields only
	nodeClass := &unstructured.Unstructured{}
	nodeClass.SetAPIVersion("karpenter.k8s.aws/v1")
	nodeClass.SetKind("EC2NodeClass")
	nodeClass.SetName("test-ec2-node-class")

	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(nodeClass).
		Build()

	// Create node manager config
	nodeManagerConfig := tfv1.NodeManagerConfig{
		NodeProvisioner: &tfv1.NodeProvisioner{
			GPURequirements: []tfv1.Requirement{
				{
					Key:      "karpenter.sh/capacity-type",
					Operator: corev1.NodeSelectorOpIn,
					Values:   []string{string(tfv1.CapacityTypeOnDemand)},
				},
				{
					Key:      "node.kubernetes.io/instance-type",
					Operator: corev1.NodeSelectorOpIn,
					Values:   []string{"x6g.large"},
				},
			},
			GPULabels: map[string]string{
				"tensor-fusion.ai/gpu-type":  "nvidia",
				"tensor-fusion.ai/node-pool": "test-pool",
			},
			GPUTaints: []tfv1.Taint{
				{
					Key:    "tensor-fusion.ai/gpu-node",
					Value:  "true",
					Effect: corev1.TaintEffectNoSchedule,
				},
			},
		},
	}

	// Create provider
	provider := KarpenterGPUNodeProvider{
		client:            fakeClient,
		nodeManagerConfig: &nodeManagerConfig,
		pricingProvider:   pricing.NewStaticPricingProvider(),
	}

	tests := []struct {
		name        string
		param       *tfv1.GPUNodeClaimSpec
		expectError bool
	}{
		{
			name: "successful node creation with GPU",
			param: &tfv1.GPUNodeClaimSpec{
				NodeName:     "test-gpu-node",
				Region:       "us-west-2",
				Zone:         "us-west-2a",
				InstanceType: "p3.8xlarge",
				CapacityType: tfv1.CapacityTypeOnDemand,
				NodeClassRef: tfv1.GroupKindName{
					Name:    "test-ec2-node-class",
					Group:   "karpenter.k8s.aws",
					Kind:    "EC2NodeClass",
					Version: "v1",
				},
				TFlopsOffered:    resource.MustParse("125"),
				VRAMOffered:      resource.MustParse("64Gi"),
				GPUDeviceOffered: 4,
				ExtraParams:      map[string]string{},
			},
			expectError: false,
		},
		{
			name: "successful node creation with GPU",
			param: &tfv1.GPUNodeClaimSpec{
				NodeName:     "test-gpu-node",
				Region:       "us-west-2",
				Zone:         "us-west-2a",
				InstanceType: "p3.8xlarge",
				CapacityType: tfv1.CapacityTypeReserved,
				NodeClassRef: tfv1.GroupKindName{
					Name:    "test-ec2-node-class",
					Group:   "karpenter.k8s.aws",
					Kind:    "EC2NodeClass",
					Version: "v1",
				},
				TFlopsOffered:    resource.MustParse("125"),
				VRAMOffered:      resource.MustParse("64Gi"),
				GPUDeviceOffered: 4,
				ExtraParams:      map[string]string{},
			},
			expectError: false,
		},
		{
			name: "successful node creation with GPU",
			param: &tfv1.GPUNodeClaimSpec{
				NodeName:     "test-gpu-node",
				Region:       "us-west-2",
				Zone:         "us-west-2a",
				InstanceType: "p3.8xlarge",
				CapacityType: tfv1.CapacityTypeSpot,
				NodeClassRef: tfv1.GroupKindName{
					Name:    "test-ec2-node-class",
					Group:   "karpenter.k8s.aws",
					Kind:    "EC2NodeClass",
					Version: "v1",
				},
				TFlopsOffered:    resource.MustParse("125"),
				VRAMOffered:      resource.MustParse("64Gi"),
				GPUDeviceOffered: 4,
				ExtraParams:      map[string]string{},
			},
			expectError: false,
		},
		{
			name: "successful node creation with GPU",
			param: &tfv1.GPUNodeClaimSpec{
				NodeName:     "test-gpu-node",
				Region:       "us-west-2",
				Zone:         "us-west-2a",
				InstanceType: "p3.8xlarge",
				CapacityType: "unexpect-capacity-type",
				NodeClassRef: tfv1.GroupKindName{
					Name:    "test-ec2-node-class",
					Group:   "karpenter.k8s.aws",
					Kind:    "EC2NodeClass",
					Version: "v1",
				},
				TFlopsOffered:    resource.MustParse("125"),
				VRAMOffered:      resource.MustParse("64Gi"),
				GPUDeviceOffered: 4,
				ExtraParams:      map[string]string{},
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()

			// Test the buildNodeClaim method directly to get the generated structure
			// This bypasses the client creation issue while still testing the core logic
			nodeClaim, err := provider.buildNodeClaim(ctx, tt.param)

			if tt.expectError {
				assert.Error(t, err)
				assert.Nil(t, nodeClaim)
				return
			}

			require.NoError(t, err)
			require.NotNil(t, nodeClaim)

			if yamlBytes, err := yaml.Marshal(nodeClaim); err == nil {
				t.Logf("Generated NodeClaim YAML:\n%s", string(yamlBytes))
			} else {
				t.Logf("YAML marshal error: %v", err)
			}

			// Verify annotations
			assert.Equal(t, "true", nodeClaim.Annotations["karpenter.sh/do-not-disrupt"])

			// Verify NodeClassRef
			assert.Equal(t, "test-ec2-node-class", nodeClaim.Spec.NodeClassRef.Name)
			assert.Equal(t, "karpenter.k8s.aws", nodeClaim.Spec.NodeClassRef.Group)
			assert.Equal(t, "EC2NodeClass", nodeClaim.Spec.NodeClassRef.Kind)

			// Verify requirements
			assert.GreaterOrEqual(t, len(nodeClaim.Spec.Requirements), 3) // instance-type, zone, region

			// Find and verify specific requirements
			var foundInstanceType, foundZone, foundRegion bool
			for _, req := range nodeClaim.Spec.Requirements {
				switch req.Key {
				case string(tfv1.NodeRequirementKeyInstanceType):
					foundInstanceType = true
					assert.Equal(t, corev1.NodeSelectorOpIn, req.Operator)
					assert.Contains(t, req.Values, tt.param.InstanceType)
				case string(tfv1.NodeRequirementKeyZone):
					foundZone = true
					assert.Equal(t, corev1.NodeSelectorOpIn, req.Operator)
					assert.Contains(t, req.Values, tt.param.Zone)
				case string(tfv1.NodeRequirementKeyRegion):
					foundRegion = true
					assert.Equal(t, corev1.NodeSelectorOpIn, req.Operator)
					assert.Contains(t, req.Values, tt.param.Region)
				}
			}
			assert.True(t, foundInstanceType, "instance-type requirement not found")
			assert.True(t, foundZone, "zone requirement not found")
			assert.True(t, foundRegion, "region requirement not found")

			// Verify GPU resources if specified
			if tt.param.GPUDeviceOffered > 0 {
				gpuResource := nodeClaim.Spec.Resources.Requests["nvidia.com/gpu"]
				assert.Equal(t, resource.MustParse("4"), gpuResource)
			}

			// Verify labels
			assert.Equal(t, "nvidia", nodeClaim.Labels["tensor-fusion.ai/gpu-type"])
			assert.Equal(t, "test-pool", nodeClaim.Labels["tensor-fusion.ai/node-pool"])

			// Verify taints
			assert.Len(t, nodeClaim.Spec.Taints, 1)
			assert.Equal(t, "tensor-fusion.ai/gpu-node", nodeClaim.Spec.Taints[0].Key)
			assert.Equal(t, "true", nodeClaim.Spec.Taints[0].Value)
			assert.Equal(t, corev1.TaintEffectNoSchedule, nodeClaim.Spec.Taints[0].Effect)

			// Verify termination grace period
			if tt.param.ExtraParams["karpenter.nodeClaim.terminationGracePeriod"] != "" {
				assert.NotNil(t, nodeClaim.Spec.TerminationGracePeriod)
			}
		})
	}
}

func TestKarpenterGPUNodeProvider_TerminateNode(t *testing.T) {
	provider := KarpenterGPUNodeProvider{
		client: nil,
	}

	t.Run("nil parameter", func(t *testing.T) {
		ctx := context.Background()
		err := provider.TerminateNode(ctx, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "NodeIdentityParam cannot be nil")
	})

	t.Run("nil client", func(t *testing.T) {
		ctx := context.Background()
		param := &types.NodeIdentityParam{
			InstanceID: "test-instance",
		}
		err := provider.TerminateNode(ctx, param)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "kubernetes client is not initialized")
	})
}

func TestKarpenterGPUNodeProvider_GetNodeStatus(t *testing.T) {
	provider := KarpenterGPUNodeProvider{
		client: nil,
	}

	t.Run("nil parameter", func(t *testing.T) {
		ctx := context.Background()
		status, err := provider.GetNodeStatus(ctx, nil)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "NodeIdentityParam cannot be nil")
		assert.Nil(t, status)
	})

	t.Run("nil client", func(t *testing.T) {
		ctx := context.Background()
		param := &types.NodeIdentityParam{
			InstanceID: "test-instance",
		}
		status, err := provider.GetNodeStatus(ctx, param)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "kubernetes client is not initialized")
		assert.Nil(t, status)
	})
}

func TestKarpenterGPUNodeProvider_parseKarpenterConfig(t *testing.T) {
	provider := KarpenterGPUNodeProvider{}

	tests := []struct {
		name     string
		param    *tfv1.GPUNodeClaimSpec
		expected string // Just test the GPU resource name for simplicity
	}{
		{
			name: "nil extra params",
			param: &tfv1.GPUNodeClaimSpec{
				ExtraParams: nil,
			},
			expected: "nvidia.com/gpu", // Early return when extraParams is nil
		},
		{
			name: "empty extra params",
			param: &tfv1.GPUNodeClaimSpec{
				ExtraParams: map[string]string{
					"karpenter.nodeClaim.terminationGracePeriod": "30s",
				},
			},
			expected: "30s", // Default value should be set
		},
		{
			name: "empty extra params",
			param: &tfv1.GPUNodeClaimSpec{
				ExtraParams: map[string]string{},
			},
			expected: "nvidia.com/gpu", // Default value should be set
		},
		{
			name:     "empty extra params",
			param:    &tfv1.GPUNodeClaimSpec{},
			expected: "nvidia.com/gpu", // Default value should be set
		},
	}

	t.Run(tests[0].name, func(t *testing.T) {
		result := provider.parseKarpenterConfig(tests[0].param)
		assert.Equal(t, tests[0].expected, string(result.GPUResourceName))
	})

	t.Run(tests[1].name, func(t *testing.T) {
		result := provider.parseKarpenterConfig(tests[1].param)
		assert.Equal(t, tests[1].expected, result.NodeClaim.TerminationGracePeriod)
	})

	t.Run(tests[2].name, func(t *testing.T) {
		result := provider.parseKarpenterConfig(tests[2].param)
		assert.Equal(t, tests[2].expected, string(result.GPUResourceName))
	})
	t.Run(tests[3].name, func(t *testing.T) {
		result := provider.parseKarpenterConfig(tests[3].param)
		assert.Equal(t, tests[3].expected, string(result.GPUResourceName))
	})
}

func TestSetNestedValue(t *testing.T) {
	tests := []struct {
		name     string
		path     string
		value    string
		expected map[string]any
	}{
		{
			name:  "simple path",
			path:  "key",
			value: "value",
			expected: map[string]any{
				"key": "value",
			},
		},
		{
			name:  "nested path",
			path:  "parent.child",
			value: "value",
			expected: map[string]any{
				"parent": map[string]any{
					"child": "value",
				},
			},
		},
		{
			name:  "deep nested path",
			path:  "a.b.c.d",
			value: "value",
			expected: map[string]any{
				"a": map[string]any{
					"b": map[string]any{
						"c": map[string]any{
							"d": "value",
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := make(map[string]any)
			setNestedValue(data, tt.path, tt.value)
			assert.Equal(t, tt.expected, data)
		})
	}
}
