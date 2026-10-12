package utils

import (
	"encoding/json"
	"reflect"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
)

func legacyChartScheduling() *tfv1.HypervisorScheduling {
	return &tfv1.HypervisorScheduling{
		AutoFreezeAndResume: tfv1.AutoFreezeAndResume{
			AutoFreeze: []tfv1.AutoFreeze{{Qos: tfv1.QoSLow, FreezeToMemTTL: "1m"}},
		},
		ElasticRateLimitParameters: legacyChartERLParameters,
	}
}

// The rate limit parameters of earlier charts are not handed to a hypervisor;
// the rest of the configuration is.
func TestLegacyChartERLParametersAreNotPassedToTheHypervisor(t *testing.T) {
	scheduling := legacyChartScheduling()
	effective, ignored := HypervisorSchedulingForHypervisor(scheduling)
	if !ignored {
		t.Fatal("the parameters of the earlier charts were not recognised")
	}
	if effective.ElasticRateLimitParameters != (tfv1.ElasticRateLimitParameters{}) {
		t.Errorf("parameters are still passed on: %+v", effective.ElasticRateLimitParameters)
	}
	if !reflect.DeepEqual(effective.AutoFreezeAndResume, scheduling.AutoFreezeAndResume) {
		t.Errorf("the rest of the configuration changed: %+v", effective.AutoFreezeAndResume)
	}
	if scheduling.ElasticRateLimitParameters != legacyChartERLParameters {
		t.Errorf("the template object was modified: %+v", scheduling.ElasticRateLimitParameters)
	}

	// What the hypervisor receives carries no parameter at all.
	raw, err := json.Marshal(effective)
	if err != nil {
		t.Fatal(err)
	}
	var received struct {
		ElasticRateLimitParameters map[string]string `json:"elasticRateLimitParameters"`
	}
	if err := json.Unmarshal(raw, &received); err != nil {
		t.Fatal(err)
	}
	if len(received.ElasticRateLimitParameters) != 0 {
		t.Errorf("the hypervisor would receive %v", received.ElasticRateLimitParameters)
	}

	// The same numbers written differently are the same values.
	rewritten := legacyChartScheduling()
	rewritten.ElasticRateLimitParameters.FilterAlpha = "0.3"
	rewritten.ElasticRateLimitParameters.Kd = "1"
	rewritten.ElasticRateLimitParameters.CapacityMin = " 150.0 "
	if _, ignored := HypervisorSchedulingForHypervisor(rewritten); !ignored {
		t.Error("the parameters of the earlier charts were not recognised when written differently")
	}
}

// Everything else is the user's setting and reaches the hypervisor unchanged,
// whatever the values are.
func TestOtherERLParametersArePassedToTheHypervisorUnchanged(t *testing.T) {
	cases := map[string]func(*tfv1.ElasticRateLimitParameters){
		"nothing set":   func(p *tfv1.ElasticRateLimitParameters) { *p = tfv1.ElasticRateLimitParameters{} },
		"one value set": func(p *tfv1.ElasticRateLimitParameters) { *p = tfv1.ElasticRateLimitParameters{BurstWindow: "0.5"} },
		"not a number":  func(p *tfv1.ElasticRateLimitParameters) { p.Kp = "fast" },
	}
	// The chart's block with any single field changed or removed.
	fields := reflect.TypeOf(tfv1.ElasticRateLimitParameters{})
	for i := 0; i < fields.NumField(); i++ {
		cases[fields.Field(i).Name+" changed"] = func(p *tfv1.ElasticRateLimitParameters) {
			reflect.ValueOf(p).Elem().Field(i).SetString("0.25")
		}
		cases[fields.Field(i).Name+" removed"] = func(p *tfv1.ElasticRateLimitParameters) {
			reflect.ValueOf(p).Elem().Field(i).SetString("")
		}
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			scheduling := legacyChartScheduling()
			change(&scheduling.ElasticRateLimitParameters)
			want := *scheduling.DeepCopy()
			effective, ignored := HypervisorSchedulingForHypervisor(scheduling)
			if ignored {
				t.Fatalf("parameters the user set were left out: %+v", scheduling.ElasticRateLimitParameters)
			}
			if !reflect.DeepEqual(*effective, want) {
				t.Errorf("configuration changed on the way: got %+v, want %+v", *effective, want)
			}
		})
	}

	if effective, ignored := HypervisorSchedulingForHypervisor(nil); effective != nil || ignored {
		t.Errorf("no configuration: got %+v, %t", effective, ignored)
	}
}

// Every parameter takes part in the comparison. A parameter added to the API
// later is not part of what the earlier charts wrote: decide here whether a
// block that also sets it is still the chart's.
func TestLegacyChartERLParametersCoverEveryField(t *testing.T) {
	legacy := reflect.ValueOf(legacyChartERLParameters)
	for i := 0; i < legacy.NumField(); i++ {
		if legacy.Field(i).Kind() != reflect.String || legacy.Field(i).String() == "" {
			t.Errorf("field %s of ElasticRateLimitParameters is not compared with the earlier charts' value",
				legacy.Type().Field(i).Name)
		}
	}
}
