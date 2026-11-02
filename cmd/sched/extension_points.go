package sched

import (
	"fmt"
	"slices"
	"strings"

	"github.com/NexusGPU/tensor-fusion/internal/scheduler/gpuresources"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/profile"
)

// gpuFitExtensionPoints are the extension points GPUResourcesFit implements.
// The scheduler starts with any subset of them enabled, so a scheduler config
// left over from an older chart goes unnoticed: without preBind, for one, pods
// are bound without their GPU allocation and the device plugin waits for it
// until kubelet gives up.
var gpuFitExtensionPoints = []struct {
	name string
	set  func(*config.Plugins) config.PluginSet
}{
	{"preEnqueue", func(p *config.Plugins) config.PluginSet { return p.PreEnqueue }},
	{"preFilter", func(p *config.Plugins) config.PluginSet { return p.PreFilter }},
	{"filter", func(p *config.Plugins) config.PluginSet { return p.Filter }},
	{"postFilter", func(p *config.Plugins) config.PluginSet { return p.PostFilter }},
	{"score", func(p *config.Plugins) config.PluginSet { return p.Score }},
	{"reserve", func(p *config.Plugins) config.PluginSet { return p.Reserve }},
	{"permit", func(p *config.Plugins) config.PluginSet { return p.Permit }},
	{"preBind", func(p *config.Plugins) config.PluginSet { return p.PreBind }},
	{"postBind", func(p *config.Plugins) config.PluginSet { return p.PostBind }},
}

// gpuFitExtensionPointsEnabled splits the extension points GPUResourcesFit
// needs into those the effective plugin set enables and those it does not.
func gpuFitExtensionPointsEnabled(plugins *config.Plugins) (enabled, missing []string) {
	for _, point := range gpuFitExtensionPoints {
		if slices.ContainsFunc(point.set(plugins).Enabled, func(p config.Plugin) bool {
			return p.Name == gpuresources.Name
		}) {
			enabled = append(enabled, point.name)
		} else {
			missing = append(missing, point.name)
		}
	}
	return enabled, missing
}

// checkGPUFitExtensionPoints logs which of the extension points GPUResourcesFit
// needs the tensor-fusion scheduler profile enables, and refuses to start when
// any is missing.
func checkGPUFitExtensionPoints(logger klog.Logger, profiles profile.Map, configPath string) error {
	fwk, ok := profiles[constants.SchedulerName]
	if !ok {
		return fmt.Errorf("scheduler config %s has no %q profile", configPath, constants.SchedulerName)
	}
	return checkGPUFitPlugins(logger, fwk.ListPlugins(), configPath)
}

func checkGPUFitPlugins(logger klog.Logger, plugins *config.Plugins, configPath string) error {
	enabled, missing := gpuFitExtensionPointsEnabled(plugins)
	logger.Info("GPUResourcesFit extension points",
		"profile", constants.SchedulerName, "config", configPath,
		"enabled", strings.Join(enabled, ","), "missing", strings.Join(missing, ","))
	if len(missing) > 0 {
		return fmt.Errorf("scheduler config %s does not enable %s at %s in profile %q; "+
			"it is probably the scheduler config of an older chart: use the schedulerConfig of this chart version",
			configPath, gpuresources.Name, strings.Join(missing, ", "), constants.SchedulerName)
	}
	return nil
}
