package sched

import (
	"bytes"
	"testing"

	"github.com/NexusGPU/tensor-fusion/internal/scheduler/gpuresources"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/textlogger"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/profile"
)

func gpuFitSet() config.PluginSet {
	return config.PluginSet{Enabled: []config.Plugin{{Name: "SomeOtherPlugin"}, {Name: gpuresources.Name}}}
}

func TestGPUFitExtensionPointsEnabled(t *testing.T) {
	all := &config.Plugins{
		PreEnqueue: gpuFitSet(), PreFilter: gpuFitSet(), Filter: gpuFitSet(), PostFilter: gpuFitSet(),
		Score: gpuFitSet(), Reserve: gpuFitSet(), Permit: gpuFitSet(), PreBind: gpuFitSet(), PostBind: gpuFitSet(),
	}
	enabled, missing := gpuFitExtensionPointsEnabled(all)
	require.Len(t, enabled, len(gpuFitExtensionPoints))
	require.Empty(t, missing)

	// The scheduler config of the v1.65 chart.
	v1 := &config.Plugins{
		PreFilter: gpuFitSet(), Filter: gpuFitSet(), Score: gpuFitSet(), Reserve: gpuFitSet(), PostBind: gpuFitSet(),
		PreBind: config.PluginSet{Enabled: []config.Plugin{{Name: "VolumeBinding"}}},
	}
	enabled, missing = gpuFitExtensionPointsEnabled(v1)
	require.Equal(t, []string{"preFilter", "filter", "score", "reserve", "postBind"}, enabled)
	require.Equal(t, []string{"preEnqueue", "postFilter", "permit", "preBind"}, missing)
}

func TestCheckGPUFitPluginsLogsAndRejects(t *testing.T) {
	var buf bytes.Buffer
	logger := textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(&buf)))
	v1 := &config.Plugins{
		PreFilter: gpuFitSet(), Filter: gpuFitSet(), Score: gpuFitSet(), Reserve: gpuFitSet(), PostBind: gpuFitSet(),
	}

	err := checkGPUFitPlugins(logger, v1, "/etc/tensor-fusion/scheduler-config.yaml")
	require.ErrorContains(t, err, "does not enable GPUResourcesFit at preEnqueue, postFilter, permit, preBind")
	require.Contains(t, buf.String(), `"GPUResourcesFit extension points"`)
	require.Contains(t, buf.String(), `enabled="preFilter,filter,score,reserve,postBind"`)
	require.Contains(t, buf.String(), `missing="preEnqueue,postFilter,permit,preBind"`)
}

func TestCheckGPUFitExtensionPointsWithoutProfile(t *testing.T) {
	err := checkGPUFitExtensionPoints(klog.Background(), profile.Map{}, "/etc/tensor-fusion/scheduler-config.yaml")
	require.ErrorContains(t, err, `no "tensor-fusion-scheduler" profile`)
}
