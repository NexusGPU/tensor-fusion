package types

import (
	"context"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

type NodeCreationParam struct {
	NodeName     string
	Region       string
	Zone         string
	InstanceType string
	NodeClass    *tfv1.GPUNodeClass
	CapacityType CapacityTypeEnum

	TFlopsOffered    resource.Quantity
	VRAMOffered      resource.Quantity
	GPUDeviceOffered int32

	ExtraParams map[string]string
}

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

	CreateNode(ctx context.Context, param *NodeCreationParam) (*GPUNodeStatus, error)
	TerminateNode(ctx context.Context, param *NodeIdentityParam) error
	GetNodeStatus(ctx context.Context, param *NodeIdentityParam) (*GPUNodeStatus, error)

	GetInstancePricing(instanceType string, region string, capacityType CapacityTypeEnum) (float64, error)

	GetGPUNodeInstanceTypeInfo(region string) []GPUNodeInstanceInfo
}

type CapacityTypeEnum string

const (
	CapacityTypeOnDemand CapacityTypeEnum = "OnDemand"

	CapacityTypeReserved CapacityTypeEnum = "Reserved"

	// Spot and Preemptive are aliases of each other, used by different providers
	CapacityTypeSpot CapacityTypeEnum = "Spot"
)

type GPUNodeInstanceInfo struct {
	InstanceType string
	CostPerHour  float64

	CPUs      int32
	MemoryGiB int32

	// In TFlops and GB for every GPU
	FP16TFlopsPerGPU    int32
	VRAMGigabytesPerGPU int32

	GPUModel string
	GPUCount int32

	CPUArchitecture CPUArchitectureEnum
	GPUArchitecture GPUArchitectureEnum
}

type CPUArchitectureEnum string

const (
	CPUArchitectureAMD64 CPUArchitectureEnum = "amd64"
	CPUArchitectureARM64 CPUArchitectureEnum = "arm64" // namely aarch64
)

type GPUArchitectureEnum string

const (
	// Nvidia's new architectures from
	GPUArchitectureNvidiaVolta       GPUArchitectureEnum = "NVIDIA_Volta"       // introduced in 2017
	GPUArchitectureNvidiaTuring      GPUArchitectureEnum = "NVIDIA_Turing"      // introduced in 2018
	GPUArchitectureNvidiaAmpere      GPUArchitectureEnum = "NVIDIA_Ampere"      // introduced in 2020
	GPUArchitectureNvidiaAdaLovelace GPUArchitectureEnum = "NVIDIA_AdaLovelace" // introduced in 2022
	GPUArchitectureNvidiaHopper      GPUArchitectureEnum = "NVIDIA_Hopper"      // introduced in 2023
	GPUArchitectureNvidiaBlackwell   GPUArchitectureEnum = "NVIDIA_Blackwell"   // introduced in 2024

	// CDNA is Compute focused, offers HBM GPU memory, build for AI and ML workloads
	GPUArchitectureAMDCDNA1 GPUArchitectureEnum = "AMD_CDNA1"
	GPUArchitectureAMDCDNA2 GPUArchitectureEnum = "AMD_CDNA2"
	GPUArchitectureAMDCDNA3 GPUArchitectureEnum = "AMD_CDNA3"
	GPUArchitectureAMDCDNA4 GPUArchitectureEnum = "AMD_CDNA4"

	// RDNA is Graphics focused, offers DDR GPU memory, performance is not as good as CDNA
	GPUArchitectureAMDRDNA1 GPUArchitectureEnum = "AMD_RDNA1"
	GPUArchitectureAMDRDNA2 GPUArchitectureEnum = "AMD_RDNA2"
	GPUArchitectureAMDRDNA3 GPUArchitectureEnum = "AMD_RDNA3"
	GPUArchitectureAMDRDNA4 GPUArchitectureEnum = "AMD_RDNA4"
)
