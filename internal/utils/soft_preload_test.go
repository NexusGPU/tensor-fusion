package utils

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func TestSoftPreloadPreservesContainerConfiguration(t *testing.T) {
	originalMount := v1.VolumeMount{Name: "preload-config", MountPath: constants.LdPreloadFile, SubPath: "original", ReadOnly: true}
	spec := v1.PodSpec{Containers: []v1.Container{
		{Name: "asr", Image: "asr:latest", ImagePullPolicy: v1.PullAlways,
			Env:             []v1.EnvVar{{Name: "LD_PRELOAD", Value: "/lib/libjemalloc.so"}},
			EnvFrom:         []v1.EnvFromSource{{ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "settings"}}}},
			SecurityContext: &v1.SecurityContext{RunAsUser: ptr.To(int64(1000))},
			VolumeMounts:    []v1.VolumeMount{originalMount}},
		{Name: "second", Image: "second:latest"},
	}}
	original := spec.Containers[0].DeepCopy()
	addSoftLimiterPreload(&spec, 0, "libcuda_limiter.so")
	addSoftLimiterPreload(&spec, 1, "libascend_limiter.so")
	require.Len(t, spec.InitContainers, 2)
	init := spec.InitContainers[0]
	require.Equal(t, original.Image, init.Image)
	require.Equal(t, original.ImagePullPolicy, init.ImagePullPolicy)
	require.Equal(t, original.SecurityContext, init.SecurityContext)
	require.Equal(t, original.Env, init.Env)
	require.Equal(t, original.EnvFrom, init.EnvFrom)
	require.Contains(t, init.VolumeMounts, originalMount)
	require.Equal(t, original.Env, spec.Containers[0].Env)
	require.Len(t, spec.Containers[0].VolumeMounts, 1)
	require.Equal(t, constants.LdPreloadFile, spec.Containers[0].VolumeMounts[0].MountPath)
	require.True(t, spec.Containers[0].VolumeMounts[0].ReadOnly)
	require.NotEqual(t, spec.Containers[0].VolumeMounts[0].Name, spec.Containers[1].VolumeMounts[0].Name)
	require.Equal(t, "/tensor-fusion-limiter/libascend_limiter.so", spec.InitContainers[1].Command[5])
}

func TestSoftPreloadMergeScript(t *testing.T) {
	for _, tc := range []struct {
		name     string
		existing *string
	}{
		{name: "absent"},
		{name: "empty", existing: ptr.To("")},
		{name: "existing without trailing newline", existing: ptr.To("/opt/business/libtrace.so")},
		{name: "existing with comment", existing: ptr.To("# business preload\n/opt/business/libtrace.so\n")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			source := filepath.Join(dir, "original")
			target := filepath.Join(dir, "merged")
			if tc.existing != nil {
				require.NoError(t, os.WriteFile(source, []byte(*tc.existing), 0600))
			}
			spec := v1.PodSpec{Containers: []v1.Container{{Name: "asr", Image: "asr:latest"}}}
			addSoftLimiterPreload(&spec, 0, "libcuda_limiter.so")
			command := spec.InitContainers[0].Command
			script := strings.ReplaceAll(command[2], constants.LdPreloadFile, source)
			out, err := exec.Command(command[0], command[1], script, command[3], target, command[5]).CombinedOutput()
			require.NoError(t, err, "%s", out)
			data, err := os.ReadFile(target)
			require.NoError(t, err)
			expected := "\n" + constants.LdPreloadSoftLimiter + "\n"
			if tc.existing != nil {
				expected = *tc.existing + expected
			}
			require.Equal(t, expected, string(data))
		})
	}
}
