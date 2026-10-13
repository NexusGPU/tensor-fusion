package utils

import (
	context "context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	constants "github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

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

func appendRemoteVendorToolMounts(volumeMounts []v1.VolumeMount, vendor, volumeName string) []v1.VolumeMount {
	switch {
	case shouldInjectNvidiaVisibleDevices(vendor):
		return appendRemoteNvidiaToolMounts(volumeMounts, volumeName)
	default:
		return volumeMounts
	}
}

func appendRemoteNvidiaToolMounts(volumeMounts []v1.VolumeMount, volumeName string) []v1.VolumeMount {
	return append(volumeMounts, v1.VolumeMount{
		// Keep nvidia-smi available from the default PATH for interactive exec shells.
		// This avoids relying on TF_PREPEND_PATH/add-path behavior when users exec directly into bash.
		Name:      volumeName,
		MountPath: "/usr/local/bin/nvidia-smi",
		SubPath:   "nvidia-smi",
		ReadOnly:  true,
	})
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

func configureLocalTransportShm(ctx context.Context, pod *v1.Pod, injectContainerIndices []int) (string, string) {
	shmVolumeName := constants.TransportShmVolumeName
	shmSubPath := ""
	// Local hard modes run the TensorFusion worker in a sibling container and use /dev/shm for transport.
	//
	// If a business container already declares its own /dev/shm mount
	// (a common pattern: emptyDir with medium: Memory and a sizeLimit,
	// e.g. for PyTorch DataLoader/NCCL), reuse that volume for the
	// transport instead of appending a second mount at the same path.
	// This keeps the user's sizeLimit in control and avoids the
	// duplicate mount path that kubelet rejects. The transport only
	// needs a shared tmpfs file, so any shared pod volume works as
	// long as both the business container and the worker mount it.
	for _, containerIndex := range injectContainerIndices {
		userMount, found := findMountAtPath(pod.Spec.Containers[containerIndex].VolumeMounts, constants.TransportShmPath)
		if !found {
			continue
		}
		if shmVolumeName == constants.TransportShmVolumeName {
			shmVolumeName = userMount.Name
			shmSubPath = userMount.SubPath
		} else if userMount.Name != shmVolumeName {
			log.FromContext(ctx).Info(
				"container /dev/shm volume differs from the reused transport volume; it will not share the tensor-fusion transport",
				"container", pod.Spec.Containers[containerIndex].Name,
				"volume", userMount.Name,
				"sharedVolume", shmVolumeName)
		}
	}
	if shmVolumeName == constants.TransportShmVolumeName {
		pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
			Name: constants.TransportShmVolumeName,
			VolumeSource: v1.VolumeSource{
				EmptyDir: &v1.EmptyDirVolumeSource{
					Medium: v1.StorageMediumMemory,
				},
			},
		})
	} else {
		log.FromContext(ctx).Info(
			"reusing user-declared /dev/shm volume for local hard sidecar transport",
			"volume", shmVolumeName)
	}
	return shmVolumeName, shmSubPath
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
	// Transport shared-memory volume for client-worker RPC in local hard
	// sidecar mode. Defaults to a TF-managed emptyDir; when the workload
	// already declares its own /dev/shm mount, that volume is reused instead
	// so we never emit a duplicate mount path, which kubelet rejects
	// ("mountPath ... must be unique").
	shmVolumeName := constants.TransportShmVolumeName
	shmSubPath := ""
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
			// The local soft path adds its own nvidia-smi wrapper from the
			// soft-limiter volume below. Do not also add the remote client
			// wrapper, since Kubernetes requires mountPath values to be unique.
			if !useLocalSoftIsolation(tfInfo.Profile) {
				pod.Spec.Containers[injectContainerIndex].VolumeMounts = appendRemoteVendorToolMounts(
					pod.Spec.Containers[injectContainerIndex].VolumeMounts,
					tfInfo.Profile.GPUVendor,
					constants.TFLibsVolumeName,
				)
			}
		}
	}

	if tfInfo.Profile.IsLocalGPU {
		// Local mode mounts shared data path for worker-hypervisor communication.
		// Partitioned (MIG) and shared (whole-card) don't need it — neither has
		// a limiter or worker that reads this hostPath. Skip the volume so the
		// pod spec stays clean and we don't leak /run/tensor-fusion into pods
		// with no consumer.
		if tfInfo.Profile.Isolation != tfv1.IsolationModePartitioned &&
			tfInfo.Profile.Isolation != tfv1.IsolationModeShared {
			pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
				Name: constants.DataVolumeName,
				VolumeSource: v1.VolumeSource{
					HostPath: &v1.HostPathVolumeSource{
						Path: constants.TFDataPath,
						Type: ptr.To(v1.HostPathDirectoryOrCreate),
					},
				},
			})
			// The volume exists in this pod from here on. Whatever mounts and
			// env the rest of this function adds, finish by restricting the
			// volume to the pod's own directory in every container.
			defer enforceWorkerDataMounts(&pod.Spec)
		}

		if useLocalSoftIsolation(tfInfo.Profile) {
			// Soft isolation: inject C limiter directly into business container via middleware image.
			// No worker sidecar needed — the limiter intercepts CUDA / aclrt / musa calls in-process.
			// Limiter filename varies by vendor: libcuda_limiter.so (NVIDIA),
			// libascend_limiter.so (Ascend), libmusa_limiter.so (MThreads).
			// Each vendor's middleware image ships its own under /build/.
			middlewareImage := GetMiddlewareImage(tfInfo.Profile.GPUVendor, pool.Spec.ComponentConfig.Hypervisor.Image)
			softLimiterLib := constants.GetSoftLimiterLibName(tfInfo.Profile.GPUVendor)
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
				addSoftLimiterPreload(&pod.Spec, injectContainerIndex, softLimiterLib)
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
					Name:  constants.TFIsolationModeEnv,
					Value: string(tfv1.IsolationModeSoft),
				}, v1.EnvVar{
					Name:  constants.TFShmPathEnv,
					Value: constants.TFShmPathValueInPod,
				})
				// Do NOT set NVIDIA_VISIBLE_DEVICES=all here.
				// Device plugin's Allocate sets it to the assigned GPU UUID.
				// The vgpu-provider limiter (libcuda_limiter.so) uses dlopen("libcuda.so")
				// which finds the right library via ldconfig on any architecture or
				// distro — no need to inject TF_CUDA_LIB_PATH / TF_NVML_LIB_PATH.
				// Users who need a custom path can still set those env vars in their
				// pod spec, and vgpu-provider will honor them.
				envList = appendEnvIfMissing(envList,
					v1.EnvVar{Name: constants.PodNamespaceEnv, ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{FieldPath: constants.NamespaceFieldRef}}},
					v1.EnvVar{Name: constants.PodNameEnv, ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{FieldPath: constants.ResourceNameFieldRef}}},
					v1.EnvVar{Name: constants.ContainerNameEnv, Value: pod.Spec.Containers[injectContainerIndex].Name},
					v1.EnvVar{Name: constants.HypervisorIPEnv, ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{FieldPath: constants.HostIPFieldRef}}},
					v1.EnvVar{Name: constants.HypervisorPortEnv, Value: strconv.Itoa(int(getHypervisorPortNumber(pool.Spec.ComponentConfig.Hypervisor)))},
				)
				pod.Spec.Containers[injectContainerIndex].Env = envList

				// Belt-and-suspenders: device-plugin's Allocate response
				// already mounts vendor lib paths via AccelGetVendorMountLibs,
				// but workloads that request our resource via a non-plugin
				// path (e.g. raw GPUIndices) wouldn't get them. Apply the
				// ProviderConfig HostPathMounts the same way the remote
				// worker sidecar does — currently Ascend-gated inside the
				// helper, so non-Ascend vendors are a no-op.
				applyProviderRemoteWorkerConfigToContainerIndex(&pod.Spec, tfInfo.Profile.GPUVendor, injectContainerIndex)
			}
		} else if useLocalWorkerSidecar {
			shmVolumeName, shmSubPath = configureLocalTransportShm(ctx, pod, injectContainerIndices)

			// Seed the sidecar from worker.podTemplate so user-defined env/resources
			// (e.g. TF_LICENSE) reach the sidecar like they do for remote worker pods.
			sidecar := v1.Container{
				Name: constants.TFContainerNameWorker,
				VolumeMounts: []v1.VolumeMount{
					{
						Name:      shmVolumeName,
						MountPath: constants.TransportShmPath,
						SubPath:   shmSubPath,
					},
				},
			}
			if wc := pool.Spec.ComponentConfig.Worker; wc != nil && wc.PodTemplate != nil {
				var tmpl v1.PodTemplate
				if err := json.Unmarshal(wc.PodTemplate.Raw, &tmpl); err == nil {
					if c := workerTemplateContainer(tmpl.Template.Spec.Containers); c != nil {
						sidecar.Env = append(sidecar.Env, c.Env...)
						if c.Resources.Requests != nil || c.Resources.Limits != nil {
							sidecar.Resources = c.Resources
						}
					}
				}
			}
			pod.Spec.Containers = append(pod.Spec.Containers, sidecar)

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
			moveWorkerToNativeSidecar(pod, workerContainerIndex)
		}

		for _, injectContainerIndex := range injectContainerIndices {
			if useLocalSoftIsolation(tfInfo.Profile) {
				// Soft isolation already handled volumes and env in its own block above.
				continue
			}
			if tfInfo.Profile.Isolation == tfv1.IsolationModePartitioned ||
				tfInfo.Profile.Isolation == tfv1.IsolationModeShared {
				// Partitioned (MIG/vNPU) relies on hardware isolation; shared
				// (whole-card) has no isolation at all. Neither needs the
				// limiter / worker shm or the hypervisor RPC env. Device
				// plugin or container runtime already exposes the right
				// device(s) to the pod. Skipping the worker-shm mount also
				// avoids leaking /run/tensor-fusion into a container that
				// has no consumer.
				continue
			}
			if useLocalWorkerSidecar {
				if !hasMountAtPath(pod.Spec.Containers[injectContainerIndex].VolumeMounts, constants.TransportShmPath) {
					pod.Spec.Containers[injectContainerIndex].VolumeMounts = append(
						pod.Spec.Containers[injectContainerIndex].VolumeMounts,
						v1.VolumeMount{
							Name:      shmVolumeName,
							MountPath: constants.TransportShmPath,
							SubPath:   shmSubPath,
						},
					)
				}
				// Local sidecar clients still need the host-backed TF shared-memory state for
				// NVML/memory hooks, even though client-worker RPC uses /dev/shm transport.
				pod.Spec.Containers[injectContainerIndex].VolumeMounts = append(
					pod.Spec.Containers[injectContainerIndex].VolumeMounts,
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

			// We deliberately do NOT pre-set NVIDIA_VISIBLE_DEVICES=all here. After
			// the soft/partitioned/shared `continue`s above, the only case left in
			// this loop is `hard local sidecar`. For that flow the sidecar worker
			// holds the real GPU; the client container talks to it over /dev/shm
			// and does not need direct device access. Pre-setting =all would
			// override the device plugin's per-card UUID at kubelet merge time
			// and let the client physically see (and bypass the limiter to use)
			// every GPU on the node. Letting the device plugin Allocate response
			// be the single source of truth for NVIDIA_VISIBLE_DEVICES keeps the
			// per-card boundary intact.
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

// moveWorkerToNativeSidecar makes the worker of a pod that is meant to finish
// (restartPolicy Never or OnFailure: a Job, a one-off pod) a native sidecar: an
// init container with restartPolicy Always. As an ordinary container it keeps
// running after the workload has exited, and the pod never completes. The
// kubelet starts a native sidecar before the workload's containers and stops it
// when they are done. Pods that restart their containers anyway (Deployments,
// StatefulSets) keep the worker as an ordinary container.
func moveWorkerToNativeSidecar(pod *v1.Pod, workerContainerIndex int) {
	if pod.Spec.RestartPolicy != v1.RestartPolicyNever && pod.Spec.RestartPolicy != v1.RestartPolicyOnFailure {
		return
	}
	worker := pod.Spec.Containers[workerContainerIndex]
	worker.RestartPolicy = ptr.To(v1.ContainerRestartPolicyAlways)
	pod.Spec.Containers = append(pod.Spec.Containers[:workerContainerIndex], pod.Spec.Containers[workerContainerIndex+1:]...)
	pod.Spec.InitContainers = append(pod.Spec.InitContainers, worker)
}
