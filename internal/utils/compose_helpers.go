package utils

import (
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	constants "github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func appendEnvIfMissing(envList []v1.EnvVar, envs ...v1.EnvVar) []v1.EnvVar {
	for _, env := range envs {
		if lo.ContainsBy(envList, func(item v1.EnvVar) bool {
			return item.Name == env.Name
		}) {
			continue
		}
		envList = append(envList, env)
	}
	return envList
}

// findMountAtPath returns the first volumeMount at path, if any.
func findMountAtPath(mounts []v1.VolumeMount, path string) (v1.VolumeMount, bool) {
	for _, mount := range mounts {
		if mount.MountPath == path {
			return mount, true
		}
	}
	return v1.VolumeMount{}, false
}

// hasMountAtPath reports whether any volumeMount targets path.
func hasMountAtPath(mounts []v1.VolumeMount, path string) bool {
	_, ok := findMountAtPath(mounts, path)
	return ok
}

func setEnv(envList []v1.EnvVar, env v1.EnvVar) []v1.EnvVar {
	result := envList[:0]
	for _, item := range envList {
		if item.Name != env.Name {
			result = append(result, item)
		}
	}
	return append(result, env)
}

func convertDisabledFeaturesToEnvs(disabledFeatures string, envList []v1.EnvVar) []v1.EnvVar {
	disabledFeaturesList := strings.SplitSeq(disabledFeatures, ",")
	for feature := range disabledFeaturesList {
		if feat, ok := featureShortcutMap[feature]; ok {
			if !lo.ContainsBy(envList, func(item v1.EnvVar) bool {
				return item.Name == feat.EnvName
			}) {
				envList = append(envList, v1.EnvVar{
					Name:  feat.EnvName,
					Value: feat.EnvValue,
				})
			}
		}
	}
	return envList
}

func getProviderConfig(vendor string) (*tfv1.ProviderConfig, bool) {
	mgr := provider.GetManager()
	if mgr == nil {
		return nil, false
	}
	return mgr.GetProvider(vendor)
}

func applyProviderHostPathMounts(spec *v1.PodSpec, containerIndex int, mounts []tfv1.ProviderHypervisorHostPathMount) {
	if len(mounts) == 0 {
		return
	}

	existingMounts := make(map[string]bool)
	for _, mount := range spec.Containers[containerIndex].VolumeMounts {
		existingMounts[mount.Name] = true
	}
	existingVolumes := make(map[string]bool)
	for _, volume := range spec.Volumes {
		existingVolumes[volume.Name] = true
	}

	for _, mount := range mounts {
		if mount.Name == "" || mount.HostPath == "" || mount.MountPath == "" {
			continue
		}
		if !existingMounts[mount.Name] {
			spec.Containers[containerIndex].VolumeMounts = append(spec.Containers[containerIndex].VolumeMounts, v1.VolumeMount{
				Name:      mount.Name,
				MountPath: mount.MountPath,
				ReadOnly:  mount.ReadOnly,
			})
			existingMounts[mount.Name] = true
		}
		if !existingVolumes[mount.Name] {
			spec.Volumes = append(spec.Volumes, v1.Volume{
				Name: mount.Name,
				VolumeSource: v1.VolumeSource{
					HostPath: &v1.HostPathVolumeSource{
						Path: mount.HostPath,
						Type: ptr.To(v1.HostPathDirectory),
					},
				},
			})
			existingVolumes[mount.Name] = true
		}
	}
}

func appendLDLibraryPath(envs []v1.EnvVar, extra string) []v1.EnvVar {
	ldSet := false
	for i := range envs {
		if envs[i].Name == "LD_LIBRARY_PATH" {
			if !strings.Contains(envs[i].Value, extra) {
				if envs[i].Value == "" {
					envs[i].Value = extra
				} else {
					envs[i].Value += ":" + extra
				}
			}
			ldSet = true
			break
		}
	}
	if !ldSet {
		envs = append(envs, v1.EnvVar{
			Name:  "LD_LIBRARY_PATH",
			Value: extra,
		})
	}
	return envs
}

func getHypervisorPortNumber(hypervisorConfig *tfv1.HypervisorConfig) int32 {
	port := constants.HypervisorDefaultPortNumber
	if hypervisorConfig == nil {
		return port
	}

	if hypervisorConfig.PortNumber != nil {
		port = *hypervisorConfig.PortNumber
	}
	return port
}

// GetClientImage returns the provider-specific remote client image for a vendor.
// Falls back to defaultImage if Provider Manager is not initialized or vendor not found.
func GetClientImage(vendor string, defaultImage string) string {
	if mgr := provider.GetManager(); mgr != nil {
		return mgr.GetRemoteClientImage(vendor, defaultImage)
	}
	return defaultImage
}

// GetWorkerImage returns the worker image for a vendor using Provider Manager
// Falls back to defaultImage if Provider Manager is not initialized or vendor not found
func GetWorkerImage(vendor string, defaultImage string) string {
	if mgr := provider.GetManager(); mgr != nil {
		return mgr.GetRemoteWorkerImage(vendor, defaultImage)
	}
	return defaultImage
}

// GetMiddlewareImage returns the middleware/hypervisor image for a vendor using Provider Manager
// Falls back to defaultImage if Provider Manager is not initialized or vendor not found
func GetMiddlewareImage(vendor string, defaultImage string) string {
	if mgr := provider.GetManager(); mgr != nil {
		return mgr.GetMiddlewareImage(vendor, defaultImage)
	}
	return defaultImage
}

func shouldInjectNvidiaVisibleDevices(vendor string) bool {
	vendor = strings.TrimSpace(vendor)
	if vendor == "" {
		// Keep historical behavior for legacy workloads without explicit vendor.
		return true
	}
	return strings.EqualFold(vendor, constants.AcceleratorVendorNvidia)
}

func applyAscendRuntimeClassIfNeeded(spec *v1.PodSpec, profile *tfv1.WorkloadProfileSpec) {
	if spec == nil || profile == nil {
		return
	}
	if spec.RuntimeClassName != nil && *spec.RuntimeClassName != "" {
		return
	}
	if !strings.EqualFold(strings.TrimSpace(profile.GPUVendor), constants.AcceleratorVendorHuaweiAscendNPU) {
		return
	}
	if !profile.IsLocalGPU || profile.Isolation == tfv1.IsolationModePartitioned || UseLocalWorkerSidecar(profile) {
		spec.RuntimeClassName = ptr.To(constants.AscendRuntimeClassName)
	}
}
