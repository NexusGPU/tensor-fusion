package types

import (
	"context"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

type NodeIdentityParam struct {
	InstanceID string
	Region     string
}

type GPUNodeStatus struct {
	InstanceID string
	CreatedAt  time.Time

	PrivateIP string
	PublicIP  string
}

type GPUNodeProvider interface {
	TestConnection() error

	CreateNode(ctx context.Context, param *tfv1.GPUNodeClaim) (*GPUNodeStatus, error)
	TerminateNode(ctx context.Context, param *NodeIdentityParam) error
	GetNodeStatus(ctx context.Context, param *NodeIdentityParam) (*GPUNodeStatus, error)

	GetInstancePricing(instanceType string, capacityType tfv1.CapacityTypeEnum, region string) (float64, error)

	GetGPUNodeInstanceTypeInfo(region string) []GPUNodeInstanceInfo
}

type GPUNodeInstanceInfo struct {
	InstanceType string
	CPUs         int32
	MemoryGiB    int32

	// In TFlops and GB for every GPU
	FP16TFlopsPerGPU    float64
	VRAMGigabytesPerGPU int32

	GPUModel string
	GPUCount int32

	CPUArchitecture CPUArchitectureEnum
	GPUVendor       GPUVendorEnum
}

type CPUArchitectureEnum string

const (
	CPUArchitectureAMD64 CPUArchitectureEnum = "amd64"
	CPUArchitectureARM64 CPUArchitectureEnum = "arm64" // namely aarch64
)

type GPUVendorEnum string

const (
	// GPU manufacturer
	GPUVendorNvidia GPUVendorEnum = "NVIDIA"
	GPUVendorAMD    GPUVendorEnum = "AMD"
	GPUVendorIntel  GPUVendorEnum = "Intel"
	GPUVendorAWS    GPUVendorEnum = "AWS"
)
