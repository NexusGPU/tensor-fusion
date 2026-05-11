package computing

import (
	"os"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	workerstate "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/worker/state"
)

func TestComputeDesiredRateIncreasesWhenUnderTarget(t *testing.T) {
	state := &erlState{currentRate: 100, initialized: true}

	got := computeDesiredRate(state.currentRate, 0.7, 0.35, erlUpdateInterval.Seconds(), state, defaultERLConfig())
	if got <= 100 {
		t.Fatalf("expected rate increase, got %v", got)
	}
}

func TestComputeDesiredRateDecreasesWhenOverTarget(t *testing.T) {
	state := &erlState{currentRate: 1000, initialized: true}

	got := computeDesiredRate(state.currentRate, 0.5, 0.9, erlUpdateInterval.Seconds(), state, defaultERLConfig())
	if got >= 1000 {
		t.Fatalf("expected rate decrease, got %v", got)
	}
}

func TestRebalanceTokenBucketSmoothlyDrainsExcessTokens(t *testing.T) {
	deviceInfo := workerstate.NewSharedDeviceInfoV2(1024, 50, 1<<30)
	deviceInfo.SetERLCurrentTokens(200)
	deviceInfo.SetERLTokenCapacity(100)
	deviceInfo.SetERLLastTokenUpdate(10)

	got := rebalanceTokenBucket(deviceInfo, 10.5, 50, 100, 0.5, 0.8)
	if got >= 100 || got <= 35 {
		t.Fatalf("expected smooth drain into the reserve window, got %v", got)
	}
}

func TestLoadERLConfigFromEnv(t *testing.T) {
	t.Setenv(constants.HypervisorSchedulingConfigEnv, `{
		"elasticRateLimitParameters":{
			"maxRefillRate":"1234",
			"minRefillRate":"12",
			"filterAlpha":"0.4",
			"kp":"1.2",
			"ki":"0.5",
			"kd":"0.2",
			"burstWindow":"0.8",
			"capacityMin":"321",
			"capacityMax":"4321",
			"integralDecayFactor":"0.9"
		}
	}`)

	cfg := loadERLConfigFromEnv()
	switch {
	case cfg.rateMax != 1234,
		cfg.rateMin != 12,
		cfg.utilAlpha != 0.4,
		cfg.kp != 1.2,
		cfg.ki != 0.5,
		cfg.kd != 0.2,
		cfg.burstWindow != 0.8,
		cfg.capacityMin != 321,
		cfg.capacityMax != 4321,
		cfg.integralDecayFactor != 0.9:
		t.Fatalf("unexpected config loaded: %+v", cfg)
	}

	_ = os.Unsetenv(constants.HypervisorSchedulingConfigEnv)
}
