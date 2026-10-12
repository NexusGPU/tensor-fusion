package utils

import (
	context "context"
	"fmt"
	"maps"
	"strconv"
	"strings"
	"sync/atomic"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	constants "github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

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

func applyProviderRemoteWorkerConfig(spec *v1.PodSpec, vendor string) {
	applyProviderRemoteWorkerConfigToContainerIndex(spec, vendor, 0)
}

func applyProviderRemoteWorkerConfigToContainerIndex(spec *v1.PodSpec, vendor string, containerIndex int) {
	// Ascend mounts host driver libs; PPU mounts the host NVML/CUDA runtime tree
	// (/usr/local/ppu, libnvidia-ml.so) so the in-process libppu_limiter.so can
	// resolve them in soft-iso user pods that request GPUs via the non-plugin path.
	v := strings.TrimSpace(vendor)
	if !strings.EqualFold(v, constants.AcceleratorVendorHuaweiAscendNPU) &&
		!strings.EqualFold(v, constants.AcceleratorVendorAlibabaPPU) {
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

func workerTemplateContainer(containers []v1.Container) *v1.Container {
	for i := range containers {
		if containers[i].Name == constants.TFContainerNameWorker {
			return &containers[i]
		}
	}
	for i := range containers {
		if containers[i].Name == legacyTFContainerNameWorker {
			return &containers[i]
		}
	}
	// v1 always used the first template container. Preserve that behavior for
	// single-container templates without making multi-container matching ambiguous.
	if len(containers) == 1 {
		return &containers[0]
	}
	return nil
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

	// NVIDIA_VISIBLE_DEVICES is intentionally not set here — the hypervisor's
	// AllocateWorkerDevices in pkg/hypervisor/worker/allocation.go is the single
	// source of truth (canonical GPU-<hex> for non-partitioned NVIDIA, MIG-<uuid>
	// kept untouched for partitioned). Pre-setting any value at pod-spec level
	// would win over device-plugin envs at kubelet merge time and silently
	// clobber the per-card or per-MIG boundary.
	container.Name = constants.TFContainerNameWorker
	container.Image = GetWorkerImage(workloadProfile.GPUVendor, workerConfig.Image)
	// Shared mode (whole-card dedicated) has no limiter and no per-pod
	// quota state for the worker to read or write, so the host-backed
	// /run/tensor-fusion shm is dead weight for it. Skip the mount so the
	// worker pod's spec stays clean.
	if workloadProfile.Isolation != tfv1.IsolationModeShared {
		container.VolumeMounts = append(
			container.VolumeMounts,
			v1.VolumeMount{
				Name:             constants.DataVolumeName,
				MountPath:        constants.TFDataPath + constants.SharedMemMountSubPath,
				SubPathExpr:      constants.TFDataPathWorkerExpr,
				MountPropagation: ptr.To(v1.MountPropagationHostToContainer),
			})
	}
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
	if workloadProfile.Isolation != tfv1.IsolationModeShared {
		// The mount above must resolve to this pod's own directory whatever
		// the worker template defines.
		container.Env = setWorkerDataPathEnv(container.Env)
	}
	// Intentionally do NOT pre-set NVIDIA_VISIBLE_DEVICES on the worker container.
	// The hypervisor's AllocateWorkerDevices (pkg/hypervisor/worker/allocation.go)
	// is the single source of truth: it writes the canonical GPU-<hex> UUID list
	// for soft / hard / shared isolation, and intentionally lets the provider's
	// MIG-<uuid> from PartitionResult.envVars stick for partitioned mode. Pod-spec
	// env wins over device-plugin envs at kubelet merge time, so any pre-set
	// value here would silently clobber the per-card / per-MIG boundary —
	// including breaking MIG isolation by exposing the parent card.

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
		memLimitMB := strconv.FormatInt(workloadProfile.Resources.Limits.Vram.Value()/(1024*1024), 10)
		// Keep an explicit compute-percent in the Pod spec. For an absolute
		// TFLOPS limit, the percentage depends on the GPU selected later by the
		// scheduler, so the hypervisor device plugin injects this env at Allocate.
		if !workloadProfile.Resources.Limits.ComputePercent.IsZero() {
			container.Env = append(container.Env, v1.EnvVar{
				Name:  constants.HardSMLimiterEnv,
				Value: workloadProfile.Resources.Limits.ComputePercent.String(),
			})
		}
		container.Env = append(container.Env, v1.EnvVar{
			Name:  constants.HardMemLimiterEnv,
			Value: memLimitMB,
		}, v1.EnvVar{
			Name:  constants.LegacyHardMemLimiterEnv,
			Value: memLimitMB,
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

	// Add volume from host for CUDA hot migration and snapshot.
	// Shared (whole-card) workers don't read or write this shm, so skip the
	// volume to avoid leaking /run/tensor-fusion into the pod spec.
	if workloadProfile == nil || workloadProfile.Isolation != tfv1.IsolationModeShared {
		spec.Volumes = append(spec.Volumes, v1.Volume{
			Name: constants.DataVolumeName,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: constants.TFDataPath,
					Type: ptr.To(v1.HostPathDirectoryOrCreate),
				},
			},
		})
	}

	spec.TerminationGracePeriodSeconds = constants.GracefulPeriodSeconds
	applyAscendRuntimeClassIfNeeded(spec, workloadProfile)

	// For soft isolation, inject an init container that copies the C limiter from
	// the middleware image, overwriting the Rust limiter bundled in the worker image.
	if workloadProfile != nil && workloadProfile.Isolation == tfv1.IsolationModeSoft {
		middlewareImage := GetMiddlewareImage(workloadProfile.GPUVendor, hypervisorConfig.Image)
		softLimiterLib := constants.GetSoftLimiterLibName(workloadProfile.GPUVendor)
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
		// Mount the C limiter over the Rust limiter path in the worker container.
		// SubPath is the per-vendor filename produced by `cp /build/*` above.
		// MountPath stays as the NVIDIA path: the Rust limiter only ships in
		// the NVIDIA worker image, so non-NVIDIA vendors just see a new file
		// at this path (harmless — those workers don't LD_PRELOAD it).
		spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts,
			v1.VolumeMount{
				Name:      "soft-limiter",
				MountPath: constants.LdPreloadLimiter, // /home/app/libcuda_limiter.so
				SubPath:   softLimiterLib,
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
		if key == v1.ResourceName(constants.PodIndexAnnotation) ||
			strings.HasPrefix(string(key), constants.PodIndexAnnotation+constants.PodIndexDelimiter) {
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
