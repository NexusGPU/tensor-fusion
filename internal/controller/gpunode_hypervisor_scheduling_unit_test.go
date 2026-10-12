package controller

import (
	"encoding/json"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// The elasticRateLimitParameters block of the Helm charts up to 1.8.8.
func earlierChartERLParameters() tfv1.ElasticRateLimitParameters {
	return tfv1.ElasticRateLimitParameters{
		Kp: "0.12", Ki: "0.4", Kd: "1.0", FilterAlpha: "0.30", BurstWindow: "0.5",
		IntegralDecayFactor: "0.85", CapacityMin: "150", CapacityMax: "6000",
		MinRefillRate: "80", MaxRefillRate: "12000",
	}
}

func schedulingTemplate(generation int64, parameters tfv1.ElasticRateLimitParameters) *tfv1.SchedulingConfigTemplate {
	return &tfv1.SchedulingConfigTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "tf-scheduling-config", Generation: generation},
		Spec: tfv1.SchedulingConfigTemplateSpec{Hypervisor: &tfv1.HypervisorScheduling{
			AutoFreezeAndResume: tfv1.AutoFreezeAndResume{
				AutoFreeze: []tfv1.AutoFreeze{{Qos: tfv1.QoSLow, FreezeToMemTTL: "1m"}},
			},
			ElasticRateLimitParameters: parameters,
		}},
	}
}

// What createHypervisorPod puts into TF_HYPERVISOR_SCHEDULING_CONFIG.
func schedulingEnvValue(t *testing.T, lines *[]string, pool string, template *tfv1.SchedulingConfigTemplate) tfv1.HypervisorScheduling {
	t.Helper()
	log := funcr.New(func(_, args string) { *lines = append(*lines, args) }, funcr.Options{})
	raw, err := json.Marshal(hypervisorSchedulingConfig(log,
		&tfv1.GPUPool{ObjectMeta: metav1.ObjectMeta{Name: pool}}, template))
	require.NoError(t, err)
	var received tfv1.HypervisorScheduling
	require.NoError(t, json.Unmarshal(raw, &received))
	return received
}

// A template that still carries the rate limit parameters of an earlier chart
// does not make the hypervisors run them, and says so once per pool.
func TestHypervisorSchedulingEnvLeavesOutEarlierChartERLParameters(t *testing.T) {
	var lines []string
	template := schedulingTemplate(1, earlierChartERLParameters())

	for range 3 { // three nodes of the same pool
		received := schedulingEnvValue(t, &lines, "erl-pool-a", template)
		assert.Equal(t, tfv1.ElasticRateLimitParameters{}, received.ElasticRateLimitParameters)
		assert.Equal(t, template.Spec.Hypervisor.AutoFreezeAndResume, received.AutoFreezeAndResume)
	}
	require.Len(t, lines, 1, "one line per pool, not one per hypervisor pod")
	assert.Contains(t, lines[0], "erl-pool-a")
	assert.Contains(t, lines[0], "tf-scheduling-config")
	assert.Equal(t, earlierChartERLParameters(), template.Spec.Hypervisor.ElasticRateLimitParameters,
		"the template object must not be modified")

	schedulingEnvValue(t, &lines, "erl-pool-b", template)
	assert.Len(t, lines, 2, "another pool is reported separately")

	// Still the earlier chart's values after the template was edited: said again.
	schedulingEnvValue(t, &lines, "erl-pool-a", schedulingTemplate(2, earlierChartERLParameters()))
	assert.Len(t, lines, 3)
}

// Parameters the user set reach the hypervisor as they are, without a remark.
func TestHypervisorSchedulingEnvPassesOnOtherERLParameters(t *testing.T) {
	var lines []string
	custom := earlierChartERLParameters()
	custom.BurstWindow = "0.2"
	for name, parameters := range map[string]tfv1.ElasticRateLimitParameters{
		"one value differs": custom,
		"a few values":      {Kp: "0.5", MaxRefillRate: "50000"},
		"none":              {},
	} {
		t.Run(name, func(t *testing.T) {
			template := schedulingTemplate(1, parameters)
			received := schedulingEnvValue(t, &lines, "erl-pool-c", template)
			assert.Equal(t, parameters, received.ElasticRateLimitParameters)
			assert.Equal(t, template.Spec.Hypervisor.AutoFreezeAndResume, received.AutoFreezeAndResume)
		})
	}
	assert.Empty(t, lines)
}
