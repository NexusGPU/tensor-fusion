package controller

import (
	"testing"

	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
)

func TestHypervisorPodCrashExceeded(t *testing.T) {
	defaultCfg := config.GetGlobalConfig()
	t.Cleanup(func() { config.SetGlobalConfig(defaultCfg) })

	pod := func(initRestart, mainRestart int32) *corev1.Pod {
		return &corev1.Pod{
			Status: corev1.PodStatus{
				InitContainerStatuses: []corev1.ContainerStatus{
					{Name: "init", RestartCount: initRestart},
				},
				ContainerStatuses: []corev1.ContainerStatus{
					{Name: "main", RestartCount: mainRestart},
				},
			},
		}
	}

	t.Run("default threshold, restart count below threshold => not exceeded", func(t *testing.T) {
		config.SetGlobalConfig(&config.GlobalConfig{})
		_, _, threshold, exceeded := hypervisorPodCrashExceeded(pod(0, config.DefaultHypervisorMaxCrashCount))
		assert.False(t, exceeded, "RestartCount == threshold should not trigger (strict greater-than)")
		assert.Equal(t, config.DefaultHypervisorMaxCrashCount, threshold)
	})

	t.Run("default threshold, restart count above threshold => exceeded", func(t *testing.T) {
		config.SetGlobalConfig(&config.GlobalConfig{})
		name, count, threshold, exceeded := hypervisorPodCrashExceeded(pod(0, config.DefaultHypervisorMaxCrashCount+1))
		assert.True(t, exceeded)
		assert.Equal(t, "main", name)
		assert.Equal(t, config.DefaultHypervisorMaxCrashCount+1, count)
		assert.Equal(t, config.DefaultHypervisorMaxCrashCount, threshold)
	})

	t.Run("init container exceeds threshold is detected first", func(t *testing.T) {
		config.SetGlobalConfig(&config.GlobalConfig{})
		name, count, _, exceeded := hypervisorPodCrashExceeded(pod(config.DefaultHypervisorMaxCrashCount+1, 0))
		assert.True(t, exceeded)
		assert.Equal(t, "init", name)
		assert.Equal(t, config.DefaultHypervisorMaxCrashCount+1, count)
	})

	t.Run("threshold = 0 disables the mechanism", func(t *testing.T) {
		zero := int32(0)
		config.SetGlobalConfig(&config.GlobalConfig{HypervisorMaxCrashCount: &zero})
		_, _, _, exceeded := hypervisorPodCrashExceeded(pod(100, 100))
		assert.False(t, exceeded, "threshold <= 0 must disable the check entirely")
	})

	t.Run("negative threshold disables the mechanism", func(t *testing.T) {
		neg := int32(-1)
		config.SetGlobalConfig(&config.GlobalConfig{HypervisorMaxCrashCount: &neg})
		_, _, _, exceeded := hypervisorPodCrashExceeded(pod(50, 50))
		assert.False(t, exceeded)
	})

	t.Run("custom threshold from config is honored", func(t *testing.T) {
		custom := int32(2)
		config.SetGlobalConfig(&config.GlobalConfig{HypervisorMaxCrashCount: &custom})
		_, _, threshold, exceeded := hypervisorPodCrashExceeded(pod(0, 2))
		assert.False(t, exceeded)
		assert.Equal(t, int32(2), threshold)

		_, _, _, exceeded = hypervisorPodCrashExceeded(pod(0, 3))
		assert.True(t, exceeded, "RestartCount 3 > threshold 2 should trigger")
	})

	t.Run("nil GlobalConfig falls back to default", func(t *testing.T) {
		config.SetGlobalConfig(nil)
		_, _, threshold, _ := hypervisorPodCrashExceeded(pod(0, 0))
		assert.Equal(t, config.DefaultHypervisorMaxCrashCount, threshold)
	})
}
