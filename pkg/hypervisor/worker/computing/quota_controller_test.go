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

	got, added := rebalanceTokenBucket(deviceInfo, 10.5, 50, 100, 0.8)
	if got >= 100 || got <= 35 {
		t.Fatalf("expected smooth drain into the reserve window, got %v", got)
	}
	// What was removed is accounted for, so it is not mistaken for spending.
	if added != got-200 {
		t.Fatalf("net added = %v, want %v", added, got-200)
	}
}

// Overrides of the control loop's parameters are accepted within the range
// that is known to work, and replace the compiled defaults.
func TestLoadERLConfigFromEnv(t *testing.T) {
	t.Setenv(constants.HypervisorSchedulingConfigEnv, `{
		"elasticRateLimitParameters":{
			"maxRefillRate":"50000000",
			"minRefillRate":"1",
			"filterAlpha":"0.4",
			"kp":"1.2",
			"ki":"0.5",
			"kd":"0.2",
			"burstWindow":"0.2",
			"capacityMin":"10",
			"capacityMax":"20000000",
			"integralDecayFactor":"0.9"
		}
	}`)

	cfg := loadERLConfigFromEnv()
	want := erlConfig{
		burstWindow: 0.2, rateMin: 1, rateMax: 5e7, capacityMin: 10, capacityMax: 2e7,
		utilAlpha: 0.4, kp: 1.2, ki: 0.5, kd: 0.2, integralDecayFactor: 0.9,
	}
	if cfg != want {
		t.Fatalf("config loaded = %+v, want %+v", cfg, want)
	}

	// Without overrides, and with an environment that cannot be read, the
	// compiled defaults are the configuration.
	for _, raw := range []string{"", "{", `{"elasticRateLimitParameters":{}}`} {
		t.Setenv(constants.HypervisorSchedulingConfigEnv, raw)
		if got := loadERLConfigFromEnv(); got != defaultERLConfig() {
			t.Errorf("config loaded from %q = %+v, want the compiled defaults", raw, got)
		}
	}
	_ = os.Unsetenv(constants.HypervisorSchedulingConfigEnv)
}

// rebalanceTokenBucket does to the bucket of a worker with a limit of 50% what
// one control interval does: it refills it up to nowSecs and bleeds surplus
// burst budget. It returns the resulting balance and the net amount of tokens
// it added (negative when it removed more than it added).
func rebalanceTokenBucket(
	deviceInfo *workerstate.SharedDeviceInfoV2, nowSecs, refillRate, capacity, smoothedUtil float64,
) (currentTokens, added float64) {
	const targetUtil = 0.5
	added = deviceInfo.RefillERLTokens(nowSecs, refillRate, capacity, maxTokenUpdateGap)
	currentTokens, removed := bleedTokenBucket(deviceInfo, capacity, capacity, targetUtil, smoothedUtil)
	return currentTokens, added - removed
}
