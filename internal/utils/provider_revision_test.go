package utils

import (
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

func TestProviderRevisionProducesVendorScopedHypervisorHashes(t *testing.T) {
	pool := &tfv1.GPUPool{
		Spec: tfv1.GPUPoolSpec{
			ComponentConfig: &tfv1.ComponentConfig{
				Hypervisor: &tfv1.HypervisorConfig{},
			},
		},
	}

	baseCampaign := HypervisorTemplateHash(pool)
	baseNVIDIA := HypervisorPodTemplateHash(pool, constants.AcceleratorVendorNvidia)
	baseAMD := HypervisorPodTemplateHash(pool, "AMD")
	if baseCampaign != baseNVIDIA || baseCampaign != baseAMD {
		t.Fatalf("hashes without provider revisions must preserve the legacy hash")
	}

	if err := SetProviderConfigRevision(pool, "NVIDIA", "revision-1"); err != nil {
		t.Fatalf("set provider revision: %v", err)
	}
	campaignV1 := HypervisorTemplateHash(pool)
	nvidiaV1 := HypervisorPodTemplateHash(pool, "nvidia")
	amdV1 := HypervisorPodTemplateHash(pool, "AMD")
	if campaignV1 == baseCampaign {
		t.Fatalf("provider revision must change the pool rollout hash")
	}
	if nvidiaV1 == baseNVIDIA {
		t.Fatalf("provider revision must change the matching vendor pod hash")
	}
	if amdV1 != baseAMD {
		t.Fatalf("provider revision must not change another vendor pod hash")
	}

	if err := SetProviderConfigRevision(pool, "NVIDIA", "revision-2"); err != nil {
		t.Fatalf("update provider revision: %v", err)
	}
	if HypervisorPodTemplateHash(pool, "NVIDIA") == nvidiaV1 {
		t.Fatalf("new provider revision must produce a new matching pod hash")
	}
	if HypervisorPodTemplateHash(pool, "AMD") != amdV1 {
		t.Fatalf("new provider revision must leave another vendor unchanged")
	}

	if err := SetProviderConfigRevision(pool, "NVIDIA", ""); err != nil {
		t.Fatalf("remove provider revision: %v", err)
	}
	if _, ok := pool.Annotations[constants.ProviderConfigRevisionsAnnotation]; ok {
		t.Fatalf("empty revision map annotation should be removed")
	}
	if HypervisorTemplateHash(pool) != baseCampaign {
		t.Fatalf("removing the revision must restore the base rollout hash")
	}
}
