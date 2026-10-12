package utils

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

// legacyChartERLParameters is the elasticRateLimitParameters block that the
// Helm chart wrote into the SchedulingConfigTemplate it creates, up to chart
// 1.8.8. It described the limiter of those releases. The chart no longer sets
// the block; the hypervisor's built-in parameters are the ones that are
// developed and tested.
var legacyChartERLParameters = tfv1.ElasticRateLimitParameters{
	Kp:                  "0.12",
	Ki:                  "0.4",
	Kd:                  "1.0",
	FilterAlpha:         "0.30",
	BurstWindow:         "0.5",
	IntegralDecayFactor: "0.85",
	CapacityMin:         "150",
	CapacityMax:         "6000",
	MinRefillRate:       "80",
	MaxRefillRate:       "12000",
}

// HypervisorSchedulingForHypervisor returns the scheduling configuration that
// is handed to a hypervisor, and whether the rate limit parameters were left
// out of it.
//
// A SchedulingConfigTemplate that an earlier chart created keeps that chart's
// elasticRateLimitParameters when it is not managed by Helm any more (it was
// applied by hand, copied, or edited so that Helm no longer owns the fields).
// Nobody chose those values, so a block that equals them in every field is not
// passed on and the hypervisor uses its built-in parameters. Any other block,
// including one that differs from it in a single value, is a setting of the
// user and is passed on unchanged.
//
// The earlier charts' burst window was 0.5 s. A user who wants a burst window
// of 0.5 s with otherwise built-in behaviour sets burstWindow alone: a block
// that does not carry all ten of the chart's values is never the chart's.
func HypervisorSchedulingForHypervisor(scheduling *tfv1.HypervisorScheduling) (*tfv1.HypervisorScheduling, bool) {
	if scheduling == nil || !isLegacyChartERLParameters(scheduling.ElasticRateLimitParameters) {
		return scheduling, false
	}
	effective := scheduling.DeepCopy()
	effective.ElasticRateLimitParameters = tfv1.ElasticRateLimitParameters{}
	return effective, true
}

func isLegacyChartERLParameters(parameters tfv1.ElasticRateLimitParameters) bool {
	legacy := legacyChartERLParameters
	for _, pair := range [][2]string{
		{parameters.Kp, legacy.Kp},
		{parameters.Ki, legacy.Ki},
		{parameters.Kd, legacy.Kd},
		{parameters.FilterAlpha, legacy.FilterAlpha},
		{parameters.BurstWindow, legacy.BurstWindow},
		{parameters.IntegralDecayFactor, legacy.IntegralDecayFactor},
		{parameters.CapacityMin, legacy.CapacityMin},
		{parameters.CapacityMax, legacy.CapacityMax},
		{parameters.MinRefillRate, legacy.MinRefillRate},
		{parameters.MaxRefillRate, legacy.MaxRefillRate},
	} {
		// Compared as numbers: the same value written as 0.3 or '0.30' is
		// still the chart's.
		got, err := strconv.ParseFloat(strings.TrimSpace(pair[0]), 64)
		if err != nil {
			return false
		}
		if want, _ := strconv.ParseFloat(pair[1], 64); got != want {
			return false
		}
	}
	return true
}

// ComputeBurstWindow reads the compute burst window a workload asks for with
// constants.ComputeBurstWindowAnnotation. set is false when it asks for none
// and the pool's burst window applies. A value that cannot be used as written
// comes back with an error that says why, together with the nearest value that
// can: none for one that is not a duration, zero for a negative one, the
// maximum for one above it. The admission webhook rejects on the error; the
// hypervisor, which can meet an annotation that was changed after admission,
// applies the returned value and reports the error.
func ComputeBurstWindow(annotations map[string]string) (window time.Duration, set bool, err error) {
	raw := strings.TrimSpace(annotations[constants.ComputeBurstWindowAnnotation])
	if raw == "" {
		return 0, false, nil
	}
	window, err = time.ParseDuration(raw)
	switch {
	case err != nil:
		return 0, false, fmt.Errorf("invalid %s=%q: not a duration such as 500ms or 2s",
			constants.ComputeBurstWindowAnnotation, raw)
	case window < 0:
		return 0, true, fmt.Errorf("invalid %s=%q: a burst window cannot be negative",
			constants.ComputeBurstWindowAnnotation, raw)
	case window > constants.ComputeBurstWindowMax:
		return constants.ComputeBurstWindowMax, true, fmt.Errorf("invalid %s=%q: above the maximum of %s",
			constants.ComputeBurstWindowAnnotation, raw, constants.ComputeBurstWindowMax)
	}
	return window, true, nil
}
