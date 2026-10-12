package api

import (
	"bytes"
	"io"
	"strings"
	"testing"

	"k8s.io/klog/v2"
)

// A DeviceInfo whose capacity was never resolved still has one for every
// consumer: the provider's figure, said once. A resolved capacity is never
// second-guessed, also when it is zero.
func TestEffectiveTflopsCapacityOfAnUnresolvedDevice(t *testing.T) {
	var warnings bytes.Buffer
	klog.LogToStderr(false)
	klog.SetOutput(io.Discard)
	klog.SetOutputBySeverity("WARNING", &warnings)
	t.Cleanup(func() {
		klog.Flush()
		klog.SetOutput(io.Discard)
		klog.LogToStderr(true)
	})

	var none *DeviceInfo
	if got := none.EffectiveTflopsCapacity(); got != 0 {
		t.Errorf("nil device: %v", got)
	}
	if got := (&DeviceInfo{UUID: "gpu-empty"}).EffectiveTflopsCapacity(); got != 0 {
		t.Errorf("device without any figure: %v", got)
	}

	// Built the way code written before TflopsCapacity builds one.
	legacy := &DeviceInfo{UUID: "gpu-legacy", Model: "model", MaxTflops: 44.08} //nolint:staticcheck // the alias under test
	for range 3 {
		if got := legacy.EffectiveTflopsCapacity(); got != 44.08 {
			t.Fatalf("unresolved device with the legacy field: %v, want 44.08", got)
		}
	}
	estimated := &DeviceInfo{UUID: "gpu-estimated", ProviderEstimatedTflops: 19.49184}
	if got := estimated.EffectiveTflopsCapacity(); got != NormalizeTflops(19.49184) {
		t.Errorf("unresolved device with an estimate: %v", got)
	}
	if got, ok := TflopsInUse(50, legacy); !ok || got != 22.04 {
		t.Errorf("TFLOPS in use of an unresolved device: %v %v, want 22.04", got, ok)
	}

	// Resolved: the resolved figure, whatever the estimate says.
	listed := &DeviceInfo{UUID: "gpu-listed", ProviderEstimatedTflops: 44.08}
	listed.ResolveTflopsCapacity(71.2, true)
	if got := listed.EffectiveTflopsCapacity(); got != 71.2 {
		t.Errorf("resolved device: %v, want 71.2", got)
	}
	unknown := &DeviceInfo{UUID: "gpu-unknown"}
	unknown.ResolveTflopsCapacity(0, false)
	if got := unknown.EffectiveTflopsCapacity(); got != 0 {
		t.Errorf("device resolved to unknown: %v, want 0", got)
	}

	klog.Flush()
	logged := warnings.String()
	if got := strings.Count(logged, "gpu-legacy"); got != 1 {
		t.Errorf("fallback of one device logged %d times, want once:\n%s", got, logged)
	}
	if strings.Contains(logged, "gpu-listed") || strings.Contains(logged, "gpu-unknown") || strings.Contains(logged, "gpu-empty") {
		t.Errorf("nothing to say about devices that did not fall back:\n%s", logged)
	}
}
