package utils

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

// The annotation is a Go duration between zero and the maximum. A value that
// cannot be used as written comes back with the reason and with the nearest
// value that can, for the hypervisor, which cannot refuse a Pod that runs.
func TestComputeBurstWindow(t *testing.T) {
	for _, tc := range []struct {
		raw     string
		want    time.Duration
		wantSet bool
		wantErr string
	}{
		{raw: "", wantSet: false},
		{raw: "  ", wantSet: false},
		{raw: "500ms", want: 500 * time.Millisecond, wantSet: true},
		{raw: " 2s ", want: 2 * time.Second, wantSet: true},
		{raw: "0", want: 0, wantSet: true},
		{raw: "0s", want: 0, wantSet: true},
		{raw: "1.5s", want: 1500 * time.Millisecond, wantSet: true},
		{raw: "10s", want: constants.ComputeBurstWindowMax, wantSet: true},
		{raw: "10001ms", want: constants.ComputeBurstWindowMax, wantSet: true, wantErr: "above the maximum of 10s"},
		{raw: "1h", want: constants.ComputeBurstWindowMax, wantSet: true, wantErr: "above the maximum of 10s"},
		{raw: "-1ms", want: 0, wantSet: true, wantErr: "cannot be negative"},
		{raw: "0.5", wantErr: "not a duration"},
		{raw: "fast", wantErr: "not a duration"},
		{raw: "NaN", wantErr: "not a duration"},
		{raw: "1e3s", wantErr: "not a duration"},
	} {
		window, set, err := ComputeBurstWindow(map[string]string{constants.ComputeBurstWindowAnnotation: tc.raw})
		if window != tc.want || set != tc.wantSet {
			t.Errorf("%q: window = %v, set = %t, want %v, %t", tc.raw, window, set, tc.want, tc.wantSet)
		}
		switch {
		case tc.wantErr == "" && err != nil:
			t.Errorf("%q: unexpected error %v", tc.raw, err)
		case tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr) ||
			!strings.Contains(err.Error(), constants.ComputeBurstWindowAnnotation)):
			t.Errorf("%q: error = %v, want one naming the annotation and saying %q", tc.raw, err, tc.wantErr)
		}
	}
	if _, set, err := ComputeBurstWindow(nil); set || err != nil {
		t.Errorf("no annotations: set = %t, err = %v", set, err)
	}
}

// The burst window of the earlier charts was 0.5 s. A user who wants 0.5 s
// sets that one parameter: the block is then not the chart's, and the
// hypervisor receives exactly that parameter. Only the chart's complete block,
// all ten values, is left out; with a different burst window it is the user's
// and is passed on whole.
func TestBurstWindowIsPassedToTheHypervisor(t *testing.T) {
	received := func(parameters tfv1.ElasticRateLimitParameters) (map[string]string, bool) {
		t.Helper()
		effective, ignored := HypervisorSchedulingForHypervisor(&tfv1.HypervisorScheduling{
			ElasticRateLimitParameters: parameters,
		})
		raw, err := json.Marshal(effective)
		if err != nil {
			t.Fatal(err)
		}
		var decoded struct {
			ElasticRateLimitParameters map[string]string `json:"elasticRateLimitParameters"`
		}
		if err := json.Unmarshal(raw, &decoded); err != nil {
			t.Fatal(err)
		}
		return decoded.ElasticRateLimitParameters, ignored
	}

	for _, window := range []string{"0.5", "0.50", "0", "0.1", "2", "10"} {
		got, ignored := received(tfv1.ElasticRateLimitParameters{BurstWindow: window})
		if ignored || len(got) != 1 || got["burstWindow"] != window {
			t.Errorf("burstWindow %q alone: the hypervisor receives %v (left out: %t)", window, got, ignored)
		}
	}

	if got, ignored := received(legacyChartERLParameters); !ignored || len(got) != 0 {
		t.Errorf("the earlier charts' block: the hypervisor receives %v (left out: %t)", got, ignored)
	}

	changed := legacyChartERLParameters
	changed.BurstWindow = "2"
	if got, ignored := received(changed); ignored || len(got) != 10 || got["burstWindow"] != "2" {
		t.Errorf("the earlier charts' block with another burst window: the hypervisor receives %v (left out: %t)", got, ignored)
	}

	// Nine of the chart's values without its burst window are not the chart's
	// block either: the window then stays the hypervisor's own.
	without := legacyChartERLParameters
	without.BurstWindow = ""
	if got, ignored := received(without); ignored || len(got) != 9 {
		t.Errorf("the earlier charts' block without a burst window: the hypervisor receives %v (left out: %t)", got, ignored)
	}
}
