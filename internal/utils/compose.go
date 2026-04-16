package utils

import (
	context "context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"os"
	"strconv"
	"strings"
	"sync/atomic"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	constants "github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

var injectLibResource v1.ResourceList = v1.ResourceList{
	v1.ResourceCPU:    resource.MustParse("20m"),
	v1.ResourceMemory: resource.MustParse("256Mi"),
}

var hypervisorDefaultRequests v1.ResourceList = v1.ResourceList{
	v1.ResourceCPU:    resource.MustParse("50m"),
	v1.ResourceMemory: resource.MustParse("128Mi"),
}
var hypervisorDefaultLimits v1.ResourceList = v1.ResourceList{
	v1.ResourceCPU:    resource.MustParse("1000m"),
	v1.ResourceMemory: resource.MustParse("256Mi"),
}

var vectorDefaultRequests v1.ResourceList = v1.ResourceList{
	v1.ResourceCPU:    resource.MustParse("20m"),
	v1.ResourceMemory: resource.MustParse("64Mi"),
}
var vectorDefaultLimits v1.ResourceList = v1.ResourceList{
	v1.ResourceCPU:    resource.MustParse("1000m"),
	v1.ResourceMemory: resource.MustParse("256Mi"),
}

// TODO GPU workload varies, user should specify worker CPU/Memory when using remote CUDA
// By default, only set very low requests for each worker and allow burst to full GPU CPU/Memory
var workerDefaultRequests v1.ResourceList = v1.ResourceList{
	v1.ResourceCPU:    resource.MustParse("50m"),
	v1.ResourceMemory: resource.MustParse("128Mi"),
}

var workerPodIndexCounter uint32
var featureShortcutMap = map[string]struct {
	EnvName  string
	EnvValue string
}{
	constants.BuiltInFeaturesGpuLimiter: {
		EnvName:  constants.DisableGpuLimiterEnv,
		EnvValue: constants.TrueStringValue,
	},
	constants.BuiltInFeaturesGpuOpt: {
		EnvName:  constants.DisableCudaOptimizationEnv,
		EnvValue: constants.DisableWorkerFeatureEnvVal,
	},
	constants.BuiltInFeaturesMemManager: {
		EnvName:  constants.DisableVRAMManagerEnv,
		EnvValue: constants.DisableWorkerFeatureEnvVal,
	},
}

type TensorFusionInfo struct {
	Profile          *tfv1.WorkloadProfileSpec
	DynamicReplicas  bool
	EnabledReplicas  *int32
	WorkloadName     string
	PodControllerRef *metav1.OwnerReference
	ContainerNames   []string
}

func UseLocalWorkerSidecar(profile *tfv1.WorkloadProfileSpec) bool {
	if profile == nil || !profile.IsLocalGPU {
		return false
	}
	if profile.SidecarWorker {
		return true
	}
	// Soft isolation does NOT need a worker sidecar — the C limiter is injected
	// directly into the business container via LD_PRELOAD.
	if profile.Isolation == tfv1.IsolationModeSoft {
		return false
	}
	return profile.Isolation == "" || profile.Isolation == tfv1.IsolationModeHard
}

// useLocalSoftIsolation returns true when the pod should use direct limiter injection
// into the business container (no worker sidecar).
func useLocalSoftIsolation(profile *tfv1.WorkloadProfileSpec) bool {
	if profile == nil || !profile.IsLocalGPU {
		return false
	}
	return profile.Isolation == tfv1.IsolationModeSoft
}

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

func appendNvidiaLibraryPathEnvs(envList []v1.EnvVar) []v1.EnvVar {
	return appendEnvIfMissing(envList,
		v1.EnvVar{
			Name:  constants.RealCUDALibPathEnv,
			Value: constants.RealCUDALibPathValue,
		},
		v1.EnvVar{
			Name:  constants.RealNvmlLibPathEnv,
			Value: constants.RealNvmlLibPathValue,
		},
	)
}

func shouldInjectClientBootstrap(profile *tfv1.WorkloadProfileSpec) bool {
	if profile == nil {
		return true
	}
	if !profile.IsLocalGPU {
		return true
	}
	return UseLocalWorkerSidecar(profile)
}

func shouldDisableCudaHooksForLocalSidecarClient(profile *tfv1.WorkloadProfileSpec) bool {
	if profile == nil {
		return false
	}
	if !profile.IsLocalGPU || !UseLocalWorkerSidecar(profile) {
		return false
	}
	return shouldInjectNvidiaVisibleDevices(profile.GPUVendor)
}

func AddOrOverrideTFClientMissingAnnotationsBeforePatch(pod *v1.Pod, tfInfo TensorFusionInfo) {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	// When it's worker, set workload key to label for triggering workload reconcile
	if tfInfo.Profile.IsLocalGPU {
		pod.Labels[constants.WorkloadKey] = tfInfo.WorkloadName
	} else {
		pod.Annotations[constants.SelectedWorkloadAnnotation] = tfInfo.WorkloadName
	}

	// add full annotations
	if !tfInfo.Profile.Resources.Limits.Tflops.IsZero() {
		pod.Annotations[constants.TFLOPSLimitAnnotation] = tfInfo.Profile.Resources.Limits.Tflops.String()
	}
	if !tfInfo.Profile.Resources.Limits.Vram.IsZero() {
		pod.Annotations[constants.VRAMLimitAnnotation] = tfInfo.Profile.Resources.Limits.Vram.String()
	}
	if !tfInfo.Profile.Resources.Requests.Tflops.IsZero() {
		pod.Annotations[constants.TFLOPSRequestAnnotation] = tfInfo.Profile.Resources.Requests.Tflops.String()
	}
	if !tfInfo.Profile.Resources.Requests.Vram.IsZero() {
		pod.Annotations[constants.VRAMRequestAnnotation] = tfInfo.Profile.Resources.Requests.Vram.String()
	}
	if !tfInfo.Profile.Resources.Requests.ComputePercent.IsZero() {
		pod.Annotations[constants.ComputeRequestAnnotation] = tfInfo.Profile.Resources.Requests.ComputePercent.String()
	}
	if !tfInfo.Profile.Resources.Limits.ComputePercent.IsZero() {
		pod.Annotations[constants.ComputeLimitAnnotation] = tfInfo.Profile.Resources.Limits.ComputePercent.String()
	}
	if tfInfo.Profile.Qos == "" {
		pod.Annotations[constants.QoSLevelAnnotation] = string(tfv1.QoSMedium)
	} else {
		pod.Annotations[constants.QoSLevelAnnotation] = string(tfInfo.Profile.Qos)
	}
	pod.Annotations[constants.GpuCountAnnotation] = fmt.Sprintf("%d", tfInfo.Profile.GPUCount)
	pod.Annotations[constants.GpuPoolKey] = tfInfo.Profile.PoolName
	if tfInfo.Profile.GPUModel != "" {
		pod.Annotations[constants.GPUModelAnnotation] = tfInfo.Profile.GPUModel
	}
	if tfInfo.Profile.GPUVendor != "" {
		pod.Annotations[constants.GpuVendorAnnotation] = tfInfo.Profile.GPUVendor
	}
	pod.Annotations[constants.IsLocalGPUAnnotation] = strconv.FormatBool(tfInfo.Profile.IsLocalGPU)
	pod.Annotations[constants.SidecarWorkerAnnotation] = strconv.FormatBool(UseLocalWorkerSidecar(tfInfo.Profile))
	// add inject container annotation for client Pod, in case user doesn't specify it
	pod.Annotations[constants.InjectContainerAnnotation] = strings.Join(tfInfo.ContainerNames, ",")
	pod.Annotations[constants.IsolationModeAnnotation] = string(tfInfo.Profile.Isolation)
	// add partition template ID if in partitioned mode
	if tfInfo.Profile.Isolation == tfv1.IsolationModePartitioned && tfInfo.Profile.PartitionTemplateID != "" {
		pod.Annotations[constants.PartitionTemplateIDAnnotation] = tfInfo.Profile.PartitionTemplateID
	}
}

// AppendTFWorkerLabelsAndAnnotationsAfterTemplate builds worker pod labels and
// annotations from the workload spec. desiredMembers is the pre-resolved gang
// size; pass 0 when gang scheduling is not enabled.
func AppendTFWorkerLabelsAndAnnotationsAfterTemplate(
	podTmpl *v1.PodTemplate,
	workload *tfv1.TensorFusionWorkload,
	containerName string,
	desiredMembers int32,
) (map[string]string, map[string]string) {
	labels := maps.Clone(podTmpl.Template.Labels)
	if labels == nil {
		labels = map[string]string{}
	}
	labels[constants.LabelComponent] = constants.ComponentWorker

	annotations := maps.Clone(podTmpl.Template.Annotations)
	if annotations == nil {
		annotations = map[string]string{}
	}
	res := workload.Spec.Resources

	// TFLOPs and compute percent are mutually exclusive, if TFLOPs is set, compute percent will be ignored
	if !res.Limits.Tflops.IsZero() {
		annotations[constants.TFLOPSLimitAnnotation] = res.Limits.Tflops.String()
	}
	if !res.Requests.Tflops.IsZero() {
		annotations[constants.TFLOPSRequestAnnotation] = res.Requests.Tflops.String()
	}
	if !res.Requests.ComputePercent.IsZero() {
		annotations[constants.ComputeRequestAnnotation] = res.Requests.ComputePercent.String()
	}
	if !res.Limits.ComputePercent.IsZero() {
		annotations[constants.ComputeLimitAnnotation] = res.Limits.ComputePercent.String()
	}

	annotations[constants.VRAMLimitAnnotation] = res.Limits.Vram.String()
	annotations[constants.VRAMRequestAnnotation] = res.Requests.Vram.String()

	annotations[constants.InjectContainerAnnotation] = containerName
	if workload.Spec.Qos == "" {
		annotations[constants.QoSLevelAnnotation] = string(tfv1.QoSMedium)
	} else {
		annotations[constants.QoSLevelAnnotation] = string(workload.Spec.Qos)
	}

	if workload.Spec.GPUCount > 0 {
		annotations[constants.GpuCountAnnotation] = fmt.Sprintf("%d", workload.Spec.GPUCount)
	} else {
		annotations[constants.GpuCountAnnotation] = fmt.Sprintf("%d", 1)
	}
	annotations[constants.GpuPoolKey] = workload.Spec.PoolName
	if workload.Spec.GPUModel != "" {
		annotations[constants.GPUModelAnnotation] = workload.Spec.GPUModel
	}
	if workload.Spec.GPUVendor != "" {
		annotations[constants.GpuVendorAnnotation] = workload.Spec.GPUVendor
	}
	if len(workload.Spec.GPUIndices) > 0 {
		annotations[constants.GpuIndicesAnnotation] = strings.Join(lo.Map(workload.Spec.GPUIndices, func(index int32, _ int) string {
			return strconv.Itoa(int(index))
		}), ",")
	}
	annotations[constants.IsolationModeAnnotation] = string(workload.Spec.Isolation)
	// add partition template ID if in partitioned mode
	if workload.Spec.Isolation == tfv1.IsolationModePartitioned && workload.Spec.PartitionTemplateID != "" {
		annotations[constants.PartitionTemplateIDAnnotation] = workload.Spec.PartitionTemplateID
	}

	// Pass through container-gpu-count annotation from workload to worker pod
	// This preserves per-container GPU count information for multi-container scenarios
	if workload.Annotations != nil {
		if containerGPUCount, ok := workload.Annotations[constants.ContainerGPUCountAnnotation]; ok && containerGPUCount != "" {
			annotations[constants.ContainerGPUCountAnnotation] = containerGPUCount
		}
	}

	// Add gang scheduling annotations if configured.
	// Stamp all gang decision inputs so the scheduler only reads pod annotations.
	if workload.Spec.GangScheduling != nil {
		annotations[constants.GangEnabledAnnotation] = constants.TrueStringValue
		if workload.Spec.GangScheduling.MinMembers > 0 {
			annotations[constants.GangMinMembersAnnotation] = strconv.Itoa(int(workload.Spec.GangScheduling.MinMembers))
		}
		if workload.Spec.GangScheduling.Timeout != nil && workload.Spec.GangScheduling.Timeout.Duration > 0 {
			annotations[constants.GangTimeoutAnnotation] = workload.Spec.GangScheduling.Timeout.Duration.String()
		}

		if desiredMembers >= 2 {
			requiredMembers := desiredMembers
			if workload.Spec.GangScheduling.MinMembers >= 2 {
				requiredMembers = workload.Spec.GangScheduling.MinMembers
			}
			annotations[constants.GangDesiredMembersAnnotation] = strconv.Itoa(int(desiredMembers))
			annotations[constants.GangRequiredMembersAnnotation] = strconv.Itoa(int(requiredMembers))
			annotations[constants.GangGroupKeyAnnotation] = workload.Namespace + "/" + workload.Name
		}
	}

	return labels, annotations
}

func AddTFDefaultClientConfBeforePatch(
	ctx context.Context,
	pod *v1.Pod,
	pool *tfv1.GPUPool,
	tfInfo TensorFusionInfo,
	injectContainerIndices []int,
) {
	applyAscendRuntimeClassIfNeeded(&pod.Spec, tfInfo.Profile)

	clientConfig := pool.Spec.ComponentConfig.Client
	useLocalWorkerSidecar := UseLocalWorkerSidecar(tfInfo.Profile)
	useInjectLib := shouldInjectClientBootstrap(tfInfo.Profile)
	if useInjectLib {
		// Any mode that needs the client initContainer should prefer the provider-specific
		// client image. Local shared/embedded mode skips this path entirely.
		image := GetClientImage(tfInfo.Profile.GPUVendor, clientConfig.Image)
		pod.Spec.InitContainers = append(pod.Spec.InitContainers, v1.Container{
			Name:  constants.TFContainerNameClient,
			Image: image,
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      constants.TFLibsVolumeName,
					MountPath: constants.TFLibsVolumeMountPath,
				},
				{
					// Current inject-lib images write ld.so.preload under /tensor-fusion.
					// Mirror /tensor-fusion-conf onto the same volume so either path stays compatible.
					Name:      constants.TFLibsVolumeName,
					MountPath: constants.TFConfVolumeMountPath,
				},
			},
			Resources: v1.ResourceRequirements{
				Requests: injectLibResource,
				Limits:   injectLibResource,
			},
			Env: configureFeatures4InjectLib(tfInfo.Profile.IsLocalGPU, pod.Annotations[constants.DisableFeaturesAnnotation]),
		})
		pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
			Name: constants.TFLibsVolumeName,
			VolumeSource: v1.VolumeSource{
				EmptyDir: &v1.EmptyDirVolumeSource{},
			},
		})

		for _, injectContainerIndex := range injectContainerIndices {
			pod.Spec.Containers[injectContainerIndex].Env = append(pod.Spec.Containers[injectContainerIndex].Env, v1.EnvVar{
				Name:  constants.PrependPathEnv,
				Value: constants.TFLibsVolumeMountPath,
			}, v1.EnvVar{
				Name:  constants.PrependLibPathEnv,
				Value: constants.TFLibsVolumeMountPath,
			})

			// Known issue: glibc ldd config style, does NOT support musl, fortunately, musl rarely used in AI workloads
			pod.Spec.Containers[injectContainerIndex].VolumeMounts = append(
				pod.Spec.Containers[injectContainerIndex].VolumeMounts,
				v1.VolumeMount{
					// Mount the generated preload file from the libs volume, which is where
					// the current inject-lib images actually write it.
					Name:      constants.TFLibsVolumeName,
					MountPath: constants.LdPreloadFile,
					SubPath:   constants.LdPreloadFileName,
					ReadOnly:  true,
				}, v1.VolumeMount{
					Name:      constants.TFLibsVolumeName,
					MountPath: constants.TFLibsVolumeMountPath,
				})
		}
	}

	if tfInfo.Profile.IsLocalGPU {
		// Local mode always mounts shared data path for worker-hypervisor communication.
		pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
			Name: constants.DataVolumeName,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: constants.TFDataPath,
					Type: ptr.To(v1.HostPathDirectoryOrCreate),
				},
			},
		})

		if useLocalSoftIsolation(tfInfo.Profile) {
			// Soft isolation: inject C limiter directly into business container via middleware image.
			// No worker sidecar needed — the limiter intercepts CUDA calls in-process.
			middlewareImage := GetMiddlewareImage(tfInfo.Profile.GPUVendor, pool.Spec.ComponentConfig.Hypervisor.Image)
			pod.Spec.InitContainers = append(pod.Spec.InitContainers, v1.Container{
				Name:  constants.TFSoftLimiterInitContainerName,
				Image: middlewareImage,
				Command: []string{
					"sh", "-c",
					"cp /build/* " + constants.TFSoftLimiterVolumeMountPath + "/ 2>/dev/null; true",
				},
				VolumeMounts: []v1.VolumeMount{
					{
						Name:      constants.TFSoftLimiterVolumeName,
						MountPath: constants.TFSoftLimiterVolumeMountPath,
					},
				},
				Resources: v1.ResourceRequirements{
					Requests: injectLibResource,
					Limits:   injectLibResource,
				},
			})
			pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
				Name: constants.TFSoftLimiterVolumeName,
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{},
				},
			})

			for _, injectContainerIndex := range injectContainerIndices {
				pod.Spec.Containers[injectContainerIndex].VolumeMounts = append(
					pod.Spec.Containers[injectContainerIndex].VolumeMounts,
					v1.VolumeMount{
						Name:      constants.TFSoftLimiterVolumeName,
						MountPath: constants.TFSoftLimiterVolumeMountPath,
						ReadOnly:  true,
					},
					v1.VolumeMount{
						Name:      constants.TFSoftLimiterVolumeName,
						MountPath: "/usr/local/bin/nvidia-smi",
						SubPath:   "nvidia-smi",
						ReadOnly:  true,
					},
					v1.VolumeMount{
						Name:             constants.DataVolumeName,
						MountPath:        constants.TFDataPath + constants.SharedMemMountSubPath,
						SubPathExpr:      constants.TFDataPathWorkerExpr,
						MountPropagation: ptr.To(v1.MountPropagationHostToContainer),
					},
				)

				envList := pod.Spec.Containers[injectContainerIndex].Env
				envList = append(envList, v1.EnvVar{
					Name:  constants.LdPreloadEnv,
					Value: constants.LdPreloadSoftLimiter,
				}, v1.EnvVar{
					Name:  constants.TFIsolationModeEnv,
					Value: string(tfv1.IsolationModeSoft),
				}, v1.EnvVar{
					Name:  constants.TFShmPathEnv,
					Value: constants.TFShmPathValueInPod,
				})
				// Do NOT set NVIDIA_VISIBLE_DEVICES=all here.
				// Device plugin's Allocate sets it to the assigned GPU UUID.
				if shouldInjectNvidiaVisibleDevices(tfInfo.Profile.GPUVendor) {
					envList = appendNvidiaLibraryPathEnvs(envList)
				}
				envList = appendEnvIfMissing(envList,
					v1.EnvVar{Name: constants.PodNamespaceEnv, ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{FieldPath: constants.NamespaceFieldRef}}},
					v1.EnvVar{Name: constants.PodNameEnv, ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{FieldPath: constants.ResourceNameFieldRef}}},
					v1.EnvVar{Name: constants.ContainerNameEnv, Value: pod.Spec.Containers[injectContainerIndex].Name},
					v1.EnvVar{Name: constants.HypervisorIPEnv, ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{FieldPath: constants.HostIPFieldRef}}},
					v1.EnvVar{Name: constants.HypervisorPortEnv, Value: strconv.Itoa(int(getHypervisorPortNumber(pool.Spec.ComponentConfig.Hypervisor)))},
				)
				pod.Spec.Containers[injectContainerIndex].Env = envList
			}
		} else if useLocalWorkerSidecar {
			// Local hard modes run the TensorFusion worker in a sibling container and use /dev/shm for transport.
			pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
				Name: constants.TransportShmVolumeName,
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{
						Medium: v1.StorageMediumMemory,
					},
				},
			})

			pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
				Name: constants.TFContainerNameWorker,
				VolumeMounts: []v1.VolumeMount{
					{
						Name:      constants.TransportShmVolumeName,
						MountPath: constants.TransportShmPath,
					},
				},
			})

			workerContainerIndex := len(pod.Spec.Containers) - 1
			SetWorkerContainerSpec(
				&pod.Spec.Containers[workerContainerIndex],
				tfInfo.Profile,
				pool.Spec.ComponentConfig.Worker,
				pool.Spec.ComponentConfig.Hypervisor,
				pod.Annotations[constants.DisableFeaturesAnnotation],
				true,
			)
			applyProviderRemoteWorkerConfigToContainerIndex(&pod.Spec, tfInfo.Profile.GPUVendor, workerContainerIndex)
		}

		for _, injectContainerIndex := range injectContainerIndices {
			if useLocalSoftIsolation(tfInfo.Profile) {
				// Soft isolation already handled volumes and env in its own block above.
				continue
			}
			if useLocalWorkerSidecar {
				pod.Spec.Containers[injectContainerIndex].VolumeMounts = append(
					pod.Spec.Containers[injectContainerIndex].VolumeMounts,
					v1.VolumeMount{
						Name:      constants.TransportShmVolumeName,
						MountPath: constants.TransportShmPath,
					},
					// Local sidecar clients still need the host-backed TF shared-memory state for
					// NVML/memory hooks, even though client-worker RPC uses /dev/shm transport.
					v1.VolumeMount{
						Name:             constants.DataVolumeName,
						MountPath:        constants.TFDataPath + constants.SharedMemMountSubPath,
						SubPathExpr:      constants.TFDataPathWorkerExpr,
						MountPropagation: ptr.To(v1.MountPropagationHostToContainer),
					},
				)
			} else {
				// add ngpu spec, client is the same as worker, in same process
				pod.Spec.Containers[injectContainerIndex].VolumeMounts = append(
					pod.Spec.Containers[injectContainerIndex].VolumeMounts,
					v1.VolumeMount{
						Name:             constants.DataVolumeName,
						MountPath:        constants.TFDataPath + constants.SharedMemMountSubPath,
						SubPathExpr:      constants.TFDataPathWorkerExpr,
						MountPropagation: ptr.To(v1.MountPropagationHostToContainer),
					})
			}

			envList := pod.Spec.Containers[injectContainerIndex].Env
			if !lo.ContainsBy(envList, func(env v1.EnvVar) bool {
				return env.Name == constants.PodNamespaceEnv
			}) {
				envList = append(envList, v1.EnvVar{
					Name: constants.PodNamespaceEnv,
					ValueFrom: &v1.EnvVarSource{
						FieldRef: &v1.ObjectFieldSelector{
							FieldPath: constants.NamespaceFieldRef,
						},
					},
				})
			}
			if !lo.ContainsBy(envList, func(env v1.EnvVar) bool {
				return env.Name == constants.PodNameEnv
			}) {
				envList = append(envList, v1.EnvVar{
					Name: constants.PodNameEnv,
					ValueFrom: &v1.EnvVarSource{
						FieldRef: &v1.ObjectFieldSelector{
							FieldPath: constants.ResourceNameFieldRef,
						},
					},
				})
			}
			if !lo.ContainsBy(envList, func(env v1.EnvVar) bool {
				return env.Name == constants.ContainerNameEnv
			}) {
				envList = append(envList, v1.EnvVar{
					Name:  constants.ContainerNameEnv,
					Value: pod.Spec.Containers[injectContainerIndex].Name,
				})
			}

			if useInjectLib && shouldInjectNvidiaVisibleDevices(tfInfo.Profile.GPUVendor) && !lo.ContainsBy(envList, func(env v1.EnvVar) bool {
				return env.Name == constants.NvidiaVisibleAllDeviceEnv
			}) {
				envList = append(envList, v1.EnvVar{
					Name:  constants.NvidiaVisibleAllDeviceEnv,
					Value: constants.NvidiaVisibleAllDeviceValue,
				})
			}
			if useInjectLib && shouldInjectNvidiaVisibleDevices(tfInfo.Profile.GPUVendor) {
				envList = appendNvidiaLibraryPathEnvs(envList)
			}
			if useInjectLib && shouldDisableCudaHooksForLocalSidecarClient(tfInfo.Profile) {
				envList = appendEnvIfMissing(envList, v1.EnvVar{
					Name:  constants.EnableCudaHooksEnv,
					Value: "false",
				})
			}

			envList = append(envList, v1.EnvVar{
				Name: constants.HypervisorIPEnv,
				ValueFrom: &v1.EnvVarSource{
					FieldRef: &v1.ObjectFieldSelector{
						FieldPath: constants.HostIPFieldRef,
					},
				},
			}, v1.EnvVar{
				Name:  constants.HypervisorPortEnv,
				Value: strconv.Itoa(int(getHypervisorPortNumber(pool.Spec.ComponentConfig.Hypervisor))),
			})

			if IsLicensed() {
				envList = append(envList, v1.EnvVar{
					Name:  constants.NGPUPathEnv,
					Value: constants.NGPUPathValue,
				})
			}

			// disable GPU limiter killer switch
			if pod.Annotations[constants.DisableFeaturesAnnotation] != "" {
				envList = convertDisabledFeaturesToEnvs(pod.Annotations[constants.DisableFeaturesAnnotation], envList)
			}

			pod.Spec.Containers[injectContainerIndex].Env = envList
		}
	}
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

func configureFeatures4InjectLib(isLocalGPU bool, disabledFeatures string) []v1.EnvVar {
	envList := make([]v1.EnvVar, 0, 1)
	if isLocalGPU {
		// when tensor-fusion client already in GPU node, nvidia-smi and cuda are available, no need to copy
		// for remote mode, should copy nvidia-smi since we don't know if nvidia-container-runtime is installed
		return append(envList, v1.EnvVar{
			Name:  constants.RunInsideGPUEnv,
			Value: constants.TrueStringValue,
		})
	}
	if disabledFeatures == "" {
		return envList
	}
	disabledFeaturesList := strings.SplitSeq(disabledFeatures, ",")

	// GPU limiter by-pass take effect in bootstrap stage, add special handling here
	for feature := range disabledFeaturesList {
		if feature == constants.BuiltInFeaturesGpuLimiter {
			envList = append(envList, v1.EnvVar{
				Name:  featureShortcutMap[feature].EnvName,
				Value: featureShortcutMap[feature].EnvValue,
			})
		}
	}
	return envList
}

func AddTFHypervisorConfAfterTemplate(ctx context.Context, spec *v1.PodSpec, pool *tfv1.GPUPool, vendor string, compatibleWithNvidiaContainerToolkit bool) {
	// Hypervisor needs to read /proc to map pod with processID
	spec.HostPID = true
	spec.TerminationGracePeriodSeconds = constants.GracefulPeriodSeconds
	spec.PriorityClassName = constants.NodeCriticalPriorityClassName

	enableVector := pool.Spec.ComponentConfig.Hypervisor != nil && pool.Spec.ComponentConfig.Hypervisor.EnableVector

	// when no config or config is not valid, reset hypervisor&vector container
	if enableVector && len(spec.Containers) != 2 {
		spec.Containers = []v1.Container{
			{
				Name: constants.TFContainerNameHypervisor,
			},
			{
				Name: constants.TFContainerVector,
			},
		}
	}
	if !enableVector && len(spec.Containers) != 1 {
		spec.Containers = []v1.Container{
			{
				Name: constants.TFContainerNameHypervisor,
			},
		}
	}

	// add volumes of vector and configs
	spec.Volumes = append(spec.Volumes, v1.Volume{
		Name: constants.DataVolumeName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: constants.TFDataPath,
				Type: ptr.To(v1.HostPathDirectoryOrCreate),
			},
		},
	}, v1.Volume{
		Name: constants.TensorFusionVectorConfigVolumeName,
		VolumeSource: v1.VolumeSource{
			ConfigMap: &v1.ConfigMapVolumeSource{
				LocalObjectReference: v1.LocalObjectReference{
					Name: constants.TensorFusionVectorConfigName,
				},
			},
		},
	}, v1.Volume{
		Name: constants.LogsVolumeName,
		VolumeSource: v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		},
	}, v1.Volume{
		Name: constants.KubernetesLogsVolumeName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: constants.KubernetesLogsPath,
				Type: ptr.To(v1.HostPathDirectoryOrCreate),
			},
		},
	}, v1.Volume{
		Name: constants.TensorFusionGPUInfoConfigVolumeName,
		VolumeSource: v1.VolumeSource{
			ConfigMap: &v1.ConfigMapVolumeSource{
				LocalObjectReference: v1.LocalObjectReference{
					Name: constants.TensorFusionGPUInfoConfigName,
				},
			},
		},
	}, v1.Volume{
		Name: constants.KubeletDevicePluginVolumeName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: constants.KubeletDevicePluginPath,
				Type: ptr.To(v1.HostPathDirectoryOrCreate),
			},
		},
	}, v1.Volume{
		Name: constants.KubeletPodResourcesVolumeName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: constants.KubeletPodResourcesPath,
				Type: ptr.To(v1.HostPathDirectoryOrCreate),
			},
		},
	})

	composeHypervisorInitContainer(ctx, spec, pool, vendor, compatibleWithNvidiaContainerToolkit)
	composeHypervisorContainer(spec, pool, vendor, enableVector)

	if enableVector {
		composeVectorContainer(spec, pool)
	}
}

func composeHypervisorInitContainer(
	ctx context.Context, spec *v1.PodSpec, pool *tfv1.GPUPool,
	vendor string, compatibleWithNvidiaContainerToolkit bool,
) {
	hypervisorConfig := pool.Spec.ComponentConfig.Hypervisor
	if hypervisorConfig == nil {
		log.FromContext(ctx).Error(errors.New("hypervisor config is nil"),
			"hypervisor config is nil, can not add init container", "pool", pool.Name)
		return
	}
	spec.InitContainers = append(spec.InitContainers, v1.Container{
		Name:  "init-shm",
		Image: hypervisorConfig.Image,
		Command: []string{
			constants.ComponentHypervisor,
			constants.MountShmSubcommand,
			"--mount-point", constants.TFDataPath + constants.SharedMemMountSubPath,
			"--size", constants.ConnectionSharedMemSize,
		},
		SecurityContext: &v1.SecurityContext{
			Privileged: ptr.To(true),
			RunAsUser:  ptr.To[int64](0),
			RunAsGroup: ptr.To[int64](0),
		},
		VolumeMounts: []v1.VolumeMount{
			{
				Name:             constants.DataVolumeName,
				ReadOnly:         false,
				MountPath:        constants.TFDataPath,
				MountPropagation: ptr.To(v1.MountPropagationBidirectional),
			},
		},
	}, v1.Container{
		Name:  "init-runtime",
		Image: GetMiddlewareImage(vendor, hypervisorConfig.Image),
		Command: []string{
			"sh",
			"-c",
			"if [ -d /build ]; then cp -r /build/. " + constants.TFDataPath + "; fi",
		},
		SecurityContext: &v1.SecurityContext{
			Privileged: ptr.To(true),
			RunAsUser:  ptr.To[int64](0),
			RunAsGroup: ptr.To[int64](0),
		},
		VolumeMounts: []v1.VolumeMount{
			{
				Name:             constants.DataVolumeName,
				ReadOnly:         false,
				MountPath:        constants.TFDataPath,
				MountPropagation: ptr.To(v1.MountPropagationBidirectional),
			},
		},
	})

	// Add initContainer to wait for NVIDIA Container Toolkit toolkit-ready validation
	if compatibleWithNvidiaContainerToolkit {
		initContainerImage := pool.Spec.ComponentConfig.Hypervisor.Image
		if initContainerImage == "" {
			// Use the same image as the main container if not specified
			if len(spec.Containers) > 0 {
				initContainerImage = spec.Containers[0].Image
			}
		}

		initContainer := v1.Container{
			Name:    "toolkit-validation",
			Image:   initContainerImage,
			Command: []string{"sh", "-c"},
			Args: []string{
				"until [ -f /run/nvidia/validations/toolkit-ready ]; do echo waiting for nvidia container stack to be setup; sleep 5; done",
			},
			SecurityContext: &v1.SecurityContext{
				Privileged: ptr.To(true),
			},
			VolumeMounts: []v1.VolumeMount{
				{
					Name:             "run-nvidia-validations",
					MountPath:        "/run/nvidia/validations",
					MountPropagation: ptr.To(v1.MountPropagationHostToContainer),
				},
			},
		}

		spec.InitContainers = append(spec.InitContainers, initContainer)

		// Add volume for NVIDIA validations
		spec.Volumes = append(spec.Volumes, v1.Volume{
			Name: "run-nvidia-validations",
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: "/run/nvidia/validations",
					Type: ptr.To(v1.HostPathDirectoryOrCreate),
				},
			},
		})
	}
}

func composeHypervisorContainer(spec *v1.PodSpec, pool *tfv1.GPUPool, vendor string, enableVector bool) {
	spec.HostNetwork = true
	spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts, v1.VolumeMount{
		Name:      constants.DataVolumeName,
		ReadOnly:  false,
		MountPath: constants.TFDataPath,
	}, v1.VolumeMount{
		Name:      constants.TensorFusionGPUInfoConfigVolumeName,
		MountPath: constants.TensorFusionGPUInfoConfigMountPath,
		SubPath:   constants.TensorFusionGPUInfoConfigSubPath,
	}, v1.VolumeMount{
		Name:      constants.KubeletDevicePluginVolumeName,
		MountPath: constants.KubeletDevicePluginPath,
	}, v1.VolumeMount{
		Name:      constants.KubeletPodResourcesVolumeName,
		MountPath: constants.KubeletPodResourcesPath,
	})
	if enableVector {
		spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts, v1.VolumeMount{
			Name:      constants.LogsVolumeName,
			MountPath: constants.TensorFusionLogPath,
		})
	}

	spec.Containers[0].SecurityContext = &v1.SecurityContext{
		Capabilities: &v1.Capabilities{
			Add: []v1.Capability{
				constants.SystemPtraceCapability,
			},
		},
	}
	applyProviderHypervisorConfig(spec, vendor)

	// When k8s version >= 1.30, avoid AppArmor level limit of writing shared memory and reading /proc
	minorVersionStr := os.Getenv(constants.KubeApiVersionMinorEnv)
	if minorVersionStr != "" {
		minorVersion, err := strconv.Atoi(minorVersionStr)
		if err != nil || minorVersion >= 30 {
			spec.Containers[0].SecurityContext.AppArmorProfile = &v1.AppArmorProfile{
				Type: v1.AppArmorProfileTypeUnconfined,
			}
		}
	}

	port := getHypervisorPortNumber(pool.Spec.ComponentConfig.Hypervisor)
	spec.ServiceAccountName = constants.HypervisorServiceAccountName
	spec.Containers[0].Env = append(spec.Containers[0].Env, v1.EnvVar{
		Name:  constants.HypervisorPoolNameEnv,
		Value: pool.Name,
	}, v1.EnvVar{
		Name:  constants.TensorFusionGPUInfoEnvVar,
		Value: constants.TensorFusionGPUInfoConfigMountPath,
	}, v1.EnvVar{
		Name:  constants.HypervisorListenAddrEnv,
		Value: fmt.Sprintf("%s:%d", constants.DefaultHttpBindIP, port),
	}, v1.EnvVar{
		Name: constants.PodNameEnv,
		ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: constants.ResourceNameFieldRef,
			},
		},
	}, v1.EnvVar{
		Name: constants.HypervisorGPUNodeNameEnv,
		ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: constants.NodeNameFieldRef,
			},
		},
	}, v1.EnvVar{
		Name:  constants.HypervisorDetectUsedGPUEnv,
		Value: fmt.Sprintf("%t", IsProgressiveMigration()),
	}, v1.EnvVar{
		Name:  constants.TFProductNameEnv,
		Value: constants.ProductNameTensorFusionOSS,
	})
	if shouldInjectNvidiaVisibleDevices(vendor) {
		spec.Containers[0].Env = append(spec.Containers[0].Env, v1.EnvVar{
			Name:  constants.NvidiaVisibleAllDeviceEnv,
			Value: constants.NvidiaVisibleAllDeviceValue,
		})
	}

	if pool.Spec.ComponentConfig.Hypervisor.Image != "" {
		spec.Containers[0].Image = pool.Spec.ComponentConfig.Hypervisor.Image
	}

	spec.Containers[0].Env = append(spec.Containers[0].Env, v1.EnvVar{
		Name:  constants.HypervisorDevicePluginPathEnv,
		Value: constants.KubeletDevicePluginPath,
	})

	if len(spec.Containers[0].Resources.Requests) == 0 {
		spec.Containers[0].Resources.Requests = hypervisorDefaultRequests
	}
	if len(spec.Containers[0].Resources.Limits) == 0 {
		spec.Containers[0].Resources.Limits = hypervisorDefaultLimits
	}

	if spec.Containers[0].LivenessProbe == nil {
		spec.Containers[0].LivenessProbe = &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				HTTPGet: &v1.HTTPGetAction{
					Path: "/healthz",
					Port: intstr.FromInt(int(port)),
				},
			},
			InitialDelaySeconds: 15,
			PeriodSeconds:       20,
			TimeoutSeconds:      5,
			FailureThreshold:    5,
		}
	}
	if spec.Containers[0].ReadinessProbe == nil {
		spec.Containers[0].ReadinessProbe = &v1.Probe{
			ProbeHandler: v1.ProbeHandler{
				HTTPGet: &v1.HTTPGetAction{
					Path: "/readyz",
					Port: intstr.FromInt(int(port)),
				},
			},
			InitialDelaySeconds: 5,
			PeriodSeconds:       15,
			TimeoutSeconds:      5,
			FailureThreshold:    2,
		}
	}

	// TODO HypervisorVerifyServiceAccountEnabledEnvVar and Public Key
}

func applyProviderHypervisorConfig(spec *v1.PodSpec, vendor string) {
	providerCfg, ok := getProviderConfig(vendor)
	if !ok || providerCfg.Spec.Hypervisor == nil {
		return
	}

	hypervisorCfg := providerCfg.Spec.Hypervisor
	if hypervisorCfg.LDLibraryPath != "" {
		spec.Containers[0].Env = appendLDLibraryPath(spec.Containers[0].Env, hypervisorCfg.LDLibraryPath)
	}

	if hypervisorCfg.PrivilegedHypervisor {
		if spec.Containers[0].SecurityContext == nil {
			spec.Containers[0].SecurityContext = &v1.SecurityContext{}
		}
		spec.Containers[0].SecurityContext.Privileged = ptr.To(true)
	}

	applyProviderHostPathMounts(spec, 0, hypervisorCfg.HostPathMounts)
}

func applyProviderRemoteWorkerConfig(spec *v1.PodSpec, vendor string) {
	applyProviderRemoteWorkerConfigToContainerIndex(spec, vendor, 0)
}

func applyProviderRemoteWorkerConfigToContainerIndex(spec *v1.PodSpec, vendor string, containerIndex int) {
	if !strings.EqualFold(strings.TrimSpace(vendor), constants.AcceleratorVendorHuaweiAscendNPU) {
		return
	}
	if spec == nil || containerIndex < 0 || containerIndex >= len(spec.Containers) {
		return
	}

	providerCfg, ok := getProviderConfig(vendor)
	if !ok || providerCfg.Spec.Hypervisor == nil {
		return
	}

	hypervisorCfg := providerCfg.Spec.Hypervisor
	if hypervisorCfg.LDLibraryPath != "" {
		spec.Containers[containerIndex].Env = appendLDLibraryPath(spec.Containers[containerIndex].Env, hypervisorCfg.LDLibraryPath)
	}
	applyProviderHostPathMounts(spec, containerIndex, hypervisorCfg.HostPathMounts)
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

func composeVectorContainer(spec *v1.PodSpec, pool *tfv1.GPUPool) {
	if pool.Spec.ComponentConfig.Hypervisor.VectorImage != "" {
		spec.Containers[1].Image = pool.Spec.ComponentConfig.Hypervisor.VectorImage
	}

	spec.Containers[1].VolumeMounts = append(spec.Containers[1].VolumeMounts, v1.VolumeMount{
		Name:      constants.TensorFusionVectorConfigVolumeName,
		ReadOnly:  true,
		MountPath: constants.TensorFusionVectorConfigMountPath,
		SubPath:   constants.TensorFusionVectorConfigSubPath,
	}, v1.VolumeMount{
		Name:      constants.LogsVolumeName,
		MountPath: constants.TensorFusionLogPath,
	})

	spec.Containers[1].Env = append(spec.Containers[1].Env, v1.EnvVar{
		Name: constants.VectorPodNodeNameEnv,
		ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: constants.NodeNameFieldRef,
			},
		},
	})

	if len(spec.Containers[1].Resources.Requests) == 0 {
		spec.Containers[1].Resources.Requests = vectorDefaultRequests
	}
	if len(spec.Containers[1].Resources.Limits) == 0 {
		spec.Containers[1].Resources.Limits = vectorDefaultLimits
	}
}

// SetWorkerContainerSpec configures the worker container with required settings
func SetWorkerContainerSpec(
	container *v1.Container,
	workloadProfile *tfv1.WorkloadProfileSpec,
	workerConfig *tfv1.WorkerConfig,
	hypervisorConfig *tfv1.HypervisorConfig,
	disabledFeatures string,
	sharedMemMode bool,
) {
	if workloadProfile == nil {
		workloadProfile = &tfv1.WorkloadProfileSpec{}
	}
	if workerConfig == nil {
		workerConfig = &tfv1.WorkerConfig{}
	}

	// NOTE: need to set environment variable to make all GPUs visible to the worker,
	// vgpu.rs limiter will limit to specific devices after Pod started
	container.Name = constants.TFContainerNameWorker
	container.Image = GetWorkerImage(workloadProfile.GPUVendor, workerConfig.Image)
	container.VolumeMounts = append(
		container.VolumeMounts,
		v1.VolumeMount{
			Name:             constants.DataVolumeName,
			MountPath:        constants.TFDataPath + constants.SharedMemMountSubPath,
			SubPathExpr:      constants.TFDataPathWorkerExpr,
			MountPropagation: ptr.To(v1.MountPropagationHostToContainer),
		})
	container.Env = append(container.Env, v1.EnvVar{
		Name: constants.HypervisorIPEnv,
		ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: constants.HostIPFieldRef,
			},
		},
	}, v1.EnvVar{
		Name:  constants.HypervisorPortEnv,
		Value: strconv.Itoa(int(getHypervisorPortNumber(hypervisorConfig))),
	}, v1.EnvVar{
		Name: constants.PodNameEnv,
		ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: constants.ResourceNameFieldRef,
			},
		},
	}, v1.EnvVar{
		Name:  constants.ContainerNameEnv,
		Value: constants.TFContainerNameWorker,
	}, v1.EnvVar{
		Name:  constants.EnableWorkerLogEnv,
		Value: constants.EnableWorkerLogValue,
	}, v1.EnvVar{
		Name: constants.PodNamespaceEnv,
		ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: constants.NamespaceFieldRef,
			},
		},
	})
	if shouldInjectNvidiaVisibleDevices(workloadProfile.GPUVendor) {
		// Soft / hard isolation: let device plugin set NVIDIA_VISIBLE_DEVICES to the allocated GPU UUID.
		// Partitioned / shared: expose all GPUs (no per-device restriction at this layer).
		if workloadProfile.Isolation != tfv1.IsolationModeSoft &&
			workloadProfile.Isolation != tfv1.IsolationModeHard {
			container.Env = append(container.Env, v1.EnvVar{
				Name:  constants.NvidiaVisibleAllDeviceEnv,
				Value: constants.NvidiaVisibleAllDeviceValue,
			})
		}
		container.Env = appendNvidiaLibraryPathEnvs(container.Env)
	}

	// Only soft isolation preloads the open-source vgpu.rs cuda_limiter (reads limits from shm).
	// Hard mode relies on the closed-source hard limiter that reads TF_CUDA_SM_PERCENT_LIMIT /
	// TF_CUDA_MEMORY_LIMIT env vars, and per the HardMemLimiterEnv comment in pkg/constants/env.go
	// it "only take effect ... when open source vgpu.rs gpu-limiter is disabled".
	if workloadProfile.Isolation == tfv1.IsolationModeSoft &&
		shouldInjectCudaLimiter(workloadProfile.GPUVendor) &&
		!strings.Contains(disabledFeatures, constants.BuiltInFeaturesGpuLimiter) {
		container.Env = append(container.Env, v1.EnvVar{
			Name:  constants.LdPreloadEnv,
			Value: constants.LdPreloadLimiter,
		})
	}

	if disabledFeatures != "" {
		container.Env = convertDisabledFeaturesToEnvs(disabledFeatures, container.Env)
	}

	// TODO should calculate and set by hypervisor before container created
	// when compute isolation mode is hard-isolation, memory limit also change to hard-mode
	// open source vgpu.rs memory limiter is feedback-loop based, potentially cause resource contention
	if workloadProfile.Isolation == tfv1.IsolationModeHard {
		container.Env = append(container.Env, v1.EnvVar{
			Name:  constants.HardSMLimiterEnv,
			Value: workloadProfile.Resources.Limits.ComputePercent.String(),
		}, v1.EnvVar{
			Name:  constants.HardMemLimiterEnv,
			Value: strconv.FormatInt(workloadProfile.Resources.Limits.Vram.Value()/(1024*1024), 10),
		})
	}

	// TODO support hostNetwork mode and InfiniBand for higher performance
	container.Ports = append(container.Ports, v1.ContainerPort{
		ContainerPort: constants.TensorFusionRemoteWorkerPortNumber,
		Name:          constants.TensorFusionRemoteWorkerPortName,
		Protocol:      v1.ProtocolTCP,
	})

	if len(container.Command) == 0 {
		if strings.Contains(disabledFeatures, constants.BuiltInFeatureStartWorker) {
			container.Command = []string{
				"sleep",
				"infinity",
			}
		} else {
			if sharedMemMode {
				shmPath := constants.TransportShmPath + "/" + constants.ConnectionSharedMemName
				container.Command = []string{
					"/bin/bash",
					"-c",
					"touch " + shmPath + " && chmod 666 " + shmPath + " && exec ./tensor-fusion-worker -n shmem -m " + constants.ConnectionSharedMemName + " -M " + constants.ConnectionSharedMemSize,
				}
			} else {
				container.Command = []string{
					"./tensor-fusion-worker",
					"-p",
					strconv.Itoa(int(constants.TensorFusionRemoteWorkerPortNumber)),
				}
			}
		}
	}

	if len(container.Resources.Requests) == 0 {
		container.Resources.Requests = workerDefaultRequests
	}
}

// HypervisorTemplateHash computes hash of the full hypervisor pod template
// including code-level defaults from AddTFHypervisorConfAfterTemplate.
// Vendor-agnostic: uses empty vendor and compatibleWithNvidiaContainerToolkit=false
// so the hash stays stable across nodes — vendor-specific extras are applied
// per-node at pod creation time.
func HypervisorTemplateHash(pool *tfv1.GPUPool) string {
	podTmpl := &v1.PodTemplate{}
	if pool.Spec.ComponentConfig.Hypervisor != nil && pool.Spec.ComponentConfig.Hypervisor.PodTemplate != nil {
		json.Unmarshal(pool.Spec.ComponentConfig.Hypervisor.PodTemplate.Raw, podTmpl) //nolint:errcheck
	}
	spec := podTmpl.Template.Spec
	AddTFHypervisorConfAfterTemplate(context.Background(), &spec, pool, "", false)
	return GetObjectHash(spec)
}

// WorkerTemplateHash computes hash of the base worker pod template
// including code-level defaults from SetWorkerContainerSpec.
// Only the hypervisor port number is included — other hypervisor config
// changes should not cascade to worker pod recreation.
func WorkerTemplateHash(workerConfig *tfv1.WorkerConfig, hypervisorConfig *tfv1.HypervisorConfig) string {
	podTmpl := &v1.PodTemplate{}
	if workerConfig != nil && workerConfig.PodTemplate != nil {
		json.Unmarshal(workerConfig.PodTemplate.Raw, podTmpl) //nolint:errcheck
	}
	spec := podTmpl.Template.Spec
	if len(spec.Containers) == 0 {
		spec.Containers = []v1.Container{{}}
	}
	// Use a minimal HypervisorConfig containing only the port number,
	// so that unrelated hypervisor changes don't affect the worker hash.
	portOnly := &tfv1.HypervisorConfig{}
	if hypervisorConfig != nil && hypervisorConfig.PortNumber != nil {
		portOnly.PortNumber = hypervisorConfig.PortNumber
	}
	SetWorkerContainerSpec(&spec.Containers[0], &tfv1.WorkloadProfileSpec{}, workerConfig, portOnly, "", false)
	spec.TerminationGracePeriodSeconds = constants.GracefulPeriodSeconds
	return GetObjectHash(spec)
}

// ClientTemplateHash computes hash of the base client injection template
// including code-level defaults from AddTFDefaultClientConfBeforePatch,
// plus webhook-stage fields (OperatorEndpoint, PatchToContainer, etc.)
// that also affect the final injected pod spec.
func ClientTemplateHash(pool *tfv1.GPUPool) string {
	pod := &v1.Pod{Spec: v1.PodSpec{Containers: []v1.Container{{Name: "baseline"}}}}
	tfInfo := TensorFusionInfo{Profile: &tfv1.WorkloadProfileSpec{}}
	AddTFDefaultClientConfBeforePatch(context.Background(), pod, pool, tfInfo, []int{0})
	clientConfig := pool.Spec.ComponentConfig.Client
	return GetObjectHash(pod.Spec, clientConfig.OperatorEndpoint, clientConfig.PatchToPod, clientConfig.PatchToContainer)
}

func AddWorkerConfAfterTemplate(
	ctx context.Context, spec *v1.PodSpec, workloadProfile *tfv1.WorkloadProfileSpec, workerConfig *tfv1.WorkerConfig,
	hypervisorConfig *tfv1.HypervisorConfig, workload *tfv1.TensorFusionWorkload,
) string {
	disabledFeatures := workload.Annotations[constants.DisableFeaturesAnnotation]

	// Configure worker container
	SetWorkerContainerSpec(&spec.Containers[0], workloadProfile, workerConfig, hypervisorConfig, disabledFeatures, false)
	assignWorkerPodIndexResource(&spec.Containers[0])
	if workloadProfile != nil {
		applyProviderRemoteWorkerConfig(spec, workloadProfile.GPUVendor)
	}

	// Add volume from host for CUDA hot migration and snapshot
	spec.Volumes = append(spec.Volumes, v1.Volume{
		Name: constants.DataVolumeName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: constants.TFDataPath,
				Type: ptr.To(v1.HostPathDirectoryOrCreate),
			},
		},
	})

	spec.TerminationGracePeriodSeconds = constants.GracefulPeriodSeconds
	applyAscendRuntimeClassIfNeeded(spec, workloadProfile)

	// For soft isolation, inject an init container that copies the C limiter from
	// the middleware image, overwriting the Rust limiter bundled in the worker image.
	if workloadProfile != nil && workloadProfile.Isolation == tfv1.IsolationModeSoft {
		middlewareImage := GetMiddlewareImage(workloadProfile.GPUVendor, hypervisorConfig.Image)
		spec.InitContainers = append(spec.InitContainers, v1.Container{
			Name:  constants.TFSoftLimiterInitContainerName,
			Image: middlewareImage,
			Command: []string{
				"sh", "-c",
				"cp /build/* /soft-limiter/ 2>/dev/null; true",
			},
			VolumeMounts: []v1.VolumeMount{
				{Name: "soft-limiter", MountPath: "/soft-limiter"},
			},
			Resources: v1.ResourceRequirements{
				Requests: injectLibResource,
				Limits:   injectLibResource,
			},
		})
		spec.Volumes = append(spec.Volumes, v1.Volume{
			Name:         "soft-limiter",
			VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
		})
		// Mount the C limiter over the Rust limiter path in the worker container
		spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts,
			v1.VolumeMount{
				Name:      "soft-limiter",
				MountPath: constants.LdPreloadLimiter, // /home/app/libcuda_limiter.so
				SubPath:   constants.TFSoftLimiterLibName,
				ReadOnly:  true,
			},
			v1.VolumeMount{
				Name:      "soft-limiter",
				MountPath: "/usr/local/bin/nvidia-smi",
				SubPath:   "nvidia-smi",
				ReadOnly:  true,
			},
		)
		// Set env vars for C limiter activation
		spec.Containers[0].Env = append(spec.Containers[0].Env,
			v1.EnvVar{Name: constants.TFIsolationModeEnv, Value: string(tfv1.IsolationModeSoft)},
			v1.EnvVar{Name: constants.TFShmPathEnv, Value: constants.TFShmPathValueInPod},
		)
	}

	return spec.Containers[0].Name
}

func assignWorkerPodIndexResource(container *v1.Container) {
	if container == nil || hasPodIndexResourceClaim(container) {
		return
	}
	if container.Resources.Limits == nil {
		container.Resources.Limits = make(v1.ResourceList)
	}

	index := nextWorkerPodIndex()
	indexKey := fmt.Sprintf("%s%s%x", constants.PodIndexAnnotation, constants.PodIndexDelimiter, index/constants.IndexModLength)
	indexQuantity := resource.MustParse(strconv.Itoa((index % constants.IndexModLength) + 1))
	container.Resources.Limits[v1.ResourceName(indexKey)] = indexQuantity
}

func hasPodIndexResourceClaim(container *v1.Container) bool {
	if container == nil || container.Resources.Limits == nil {
		return false
	}
	for key := range container.Resources.Limits {
		if strings.HasPrefix(string(key), constants.PodIndexAnnotation+constants.PodIndexDelimiter) {
			return true
		}
	}
	return false
}

func nextWorkerPodIndex() int {
	maxIndex := uint32(constants.IndexKeyLength * constants.IndexModLength)
	next := atomic.AddUint32(&workerPodIndexCounter, 1)
	return int((next - 1) % maxIndex)
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

func shouldInjectCudaLimiter(vendor string) bool {
	return shouldInjectNvidiaVisibleDevices(vendor)
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
