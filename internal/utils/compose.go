package utils

import (
	context "context"
	"fmt"
	"maps"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	constants "github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

var nodeDiscoveryDefaultRequests v1.ResourceList = v1.ResourceList{
	v1.ResourceCPU:    resource.MustParse("20m"),
	v1.ResourceMemory: resource.MustParse("64Mi"),
}
var nodeDiscoveryDefaultLimits v1.ResourceList = v1.ResourceList{
	v1.ResourceCPU:    resource.MustParse("500m"),
	v1.ResourceMemory: resource.MustParse("128Mi"),
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
	Profile         *tfv1.WorkloadProfileSpec
	DynamicReplicas bool
	EnabledReplicas *int32
	WorkloadName    string
	ContainerNames  []string
	GenWorkload     bool

	// Pod mutating webhook can not get Pod UID sometimes,
	// thus need pod controller to set the owner reference
	PendingSetPodAsOwner bool
}

func AddOrOverrideTFClientMissingAnnotationsBeforePatch(pod *v1.Pod, tfInfo TensorFusionInfo) {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	// When it's worker, set workload key to label for triggering workload reconcile
	if tfInfo.Profile.IsLocalGPU {
		if pod.Labels == nil {
			pod.Labels = map[string]string{}
		}
		pod.Labels[constants.WorkloadKey] = tfInfo.WorkloadName
	}

	// add full annotations
	pod.Annotations[constants.TFLOPSLimitAnnotation] = tfInfo.Profile.Resources.Limits.Tflops.String()
	pod.Annotations[constants.VRAMLimitAnnotation] = tfInfo.Profile.Resources.Limits.Vram.String()
	if tfInfo.Profile.Qos == "" {
		pod.Annotations[constants.QoSLevelAnnotation] = string(tfv1.QoSMedium)
	} else {
		pod.Annotations[constants.QoSLevelAnnotation] = string(tfInfo.Profile.Qos)
	}
	pod.Annotations[constants.TFLOPSRequestAnnotation] = tfInfo.Profile.Resources.Requests.Tflops.String()
	pod.Annotations[constants.VRAMRequestAnnotation] = tfInfo.Profile.Resources.Requests.Vram.String()
	pod.Annotations[constants.GpuCountAnnotation] = fmt.Sprintf("%d", tfInfo.Profile.GPUCount)
	pod.Annotations[constants.GpuPoolKey] = tfInfo.Profile.PoolName
	if tfInfo.Profile.GPUModel != "" {
		pod.Annotations[constants.GPUModelAnnotation] = tfInfo.Profile.GPUModel
	}
	pod.Annotations[constants.IsLocalGPUAnnotation] = strconv.FormatBool(tfInfo.Profile.IsLocalGPU)
	// add inject container annotation for client Pod, in case user doesn't specify it
	pod.Annotations[constants.InjectContainerAnnotation] = strings.Join(tfInfo.ContainerNames, ",")
}

func AppendTFWorkerLabelsAndAnnotationsAfterTemplate(
	podTmpl *v1.PodTemplate,
	workload *tfv1.TensorFusionWorkload,
	containerName string,
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
	annotations[constants.TFLOPSLimitAnnotation] = res.Limits.Tflops.String()
	annotations[constants.VRAMLimitAnnotation] = res.Limits.Vram.String()
	annotations[constants.TFLOPSRequestAnnotation] = res.Requests.Tflops.String()
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
	return labels, annotations
}

func AddTFDefaultClientConfBeforePatch(
	ctx context.Context,
	pod *v1.Pod,
	pool *tfv1.GPUPool,
	tfInfo TensorFusionInfo,
	injectContainerIndices []int,
) {
	clientConfig := pool.Spec.ComponentConfig.Client
	image := clientConfig.RemoteModeImage
	if tfInfo.Profile.IsLocalGPU {
		image = clientConfig.EmbeddedModeImage
	}
	pod.Spec.InitContainers = append(pod.Spec.InitContainers, v1.Container{
		Name:  constants.TFContainerNameClient,
		Image: image,
		VolumeMounts: []v1.VolumeMount{
			{
				Name:      constants.TFLibsVolumeName,
				MountPath: constants.TFLibsVolumeMountPath,
			},
		},
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
			Name:  constants.PrependLDLibraryPathEnv,
			Value: constants.TFLibsVolumeMountPath,
		})

		pod.Spec.Containers[injectContainerIndex].VolumeMounts = append(
			pod.Spec.Containers[injectContainerIndex].VolumeMounts,
			v1.VolumeMount{
				Name:      constants.TFLibsVolumeName,
				MountPath: constants.LdPreloadFile,
				SubPath:   constants.LdPreloadFileName,
				ReadOnly:  true,
			}, v1.VolumeMount{
				Name:      constants.TFLibsVolumeName,
				MountPath: constants.TFLibsVolumeMountPath,
			})
	}

	if tfInfo.Profile.IsLocalGPU {
		pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
			Name: constants.DataVolumeName,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: constants.TFDataPath,
					Type: ptr.To(v1.HostPathDirectoryOrCreate),
				},
			},
		})

		for _, injectContainerIndex := range injectContainerIndices {
			pod.Spec.Containers[injectContainerIndex].VolumeMounts = append(
				pod.Spec.Containers[injectContainerIndex].VolumeMounts,
				v1.VolumeMount{
					Name:      constants.DataVolumeName,
					MountPath: constants.SharedMemDeviceName,
					SubPath:   constants.SharedMemMountSubPath,
					//  + constants.TFLibsVolumeMountPath, SubPathExpr:      constants.TFDataPathWorkerExpr,
					MountPropagation: ptr.To(v1.MountPropagationHostToContainer),
				})

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

			if !lo.ContainsBy(envList, func(env v1.EnvVar) bool {
				return env.Name == constants.NvidiaVisibleAllDeviceEnv
			}) {
				envList = append(envList, v1.EnvVar{
					Name:  constants.NvidiaVisibleAllDeviceEnv,
					Value: constants.NvidiaVisibleAllDeviceValue,
				})
			}

			envList = append(envList, v1.EnvVar{
				Name:  constants.RealNvmlLibPathEnv,
				Value: constants.RealNvmlLibPathValue,
			}, v1.EnvVar{
				Name:  constants.RealCUDALibPathEnv,
				Value: constants.RealCUDALibPathValue,
			}, v1.EnvVar{
				Name: constants.HypervisorIPEnv,
				ValueFrom: &v1.EnvVarSource{
					FieldRef: &v1.ObjectFieldSelector{
						FieldPath: constants.HostIPFieldRef,
					},
				},
			}, v1.EnvVar{
				Name:  constants.HypervisorPortEnv,
				Value: strconv.Itoa(int(getHypervisorPortNumber(pool.Spec.ComponentConfig.Hypervisor))),
			}, v1.EnvVar{
				Name:  constants.NGPUPathEnv,
				Value: constants.NGPUPathValue,
			})

			// disable GPU limiter killer switch
			if pod.Annotations[constants.DisableFeaturesAnnotation] != "" {
				envList = convertDisabledFeaturesToEnvs(pod.Annotations[constants.DisableFeaturesAnnotation], envList)
			}

			pod.Spec.Containers[injectContainerIndex].Env = envList
		}
	}
}

func convertDisabledFeaturesToEnvs(disabledFeatures string, envList []v1.EnvVar) []v1.EnvVar {
	disabledFeaturesList := strings.Split(disabledFeatures, ",")
	for _, feature := range disabledFeaturesList {
		if feat, ok := featureShortcutMap[feature]; ok {
			envList = append(envList, v1.EnvVar{
				Name:  feat.EnvName,
				Value: feat.EnvValue,
			})
		}
	}
	return envList
}

func AddTFHypervisorConfAfterTemplate(ctx context.Context, spec *v1.PodSpec, pool *tfv1.GPUPool) {
	// Hypervisor needs to read /proc to map pod with processID
	spec.HostPID = true
	spec.TerminationGracePeriodSeconds = constants.GracefulPeriodSeconds

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
	})

	composeHypervisorInitContainer(spec, pool)
	composeHypervisorContainer(spec, pool, enableVector)

	if enableVector {
		composeVectorContainer(spec, pool)
	}
}

func composeHypervisorInitContainer(spec *v1.PodSpec, pool *tfv1.GPUPool) {
	spec.InitContainers = append(spec.InitContainers, v1.Container{
		Name:    "init-shm",
		Image:   pool.Spec.ComponentConfig.Hypervisor.Image,
		Command: []string{"hypervisor", "mount-shm"},
		SecurityContext: &v1.SecurityContext{
			Privileged: ptr.To(true),
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
}

func composeHypervisorContainer(spec *v1.PodSpec, pool *tfv1.GPUPool, enableVector bool) {
	spec.HostNetwork = true
	spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts, v1.VolumeMount{
		Name:      constants.DataVolumeName,
		ReadOnly:  false,
		MountPath: constants.SharedMemDeviceName,
		SubPath:   constants.SharedMemMountSubPath,
	}, v1.VolumeMount{
		Name:      constants.TensorFusionGPUInfoConfigVolumeName,
		MountPath: constants.TensorFusionGPUInfoConfigMountPath,
		SubPath:   constants.TensorFusionGPUInfoConfigSubPath,
	}, v1.VolumeMount{
		Name:      constants.KubeletDevicePluginVolumeName,
		MountPath: constants.KubeletDevicePluginPath,
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

	port := getHypervisorPortNumber(pool.Spec.ComponentConfig.Hypervisor)
	spec.ServiceAccountName = constants.HypervisorServiceAccountName
	spec.Containers[0].Env = append(spec.Containers[0].Env, v1.EnvVar{
		Name:  constants.HypervisorPoolNameEnv,
		Value: pool.Name,
	}, v1.EnvVar{
		Name:  constants.NvidiaVisibleAllDeviceEnv,
		Value: constants.NvidiaVisibleAllDeviceValue,
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
	})

	if pool.Spec.ComponentConfig.Hypervisor.Image != "" {
		spec.Containers[0].Image = pool.Spec.ComponentConfig.Hypervisor.Image
	}

	if len(spec.Containers[0].Resources.Requests) == 0 {
		spec.Containers[0].Resources.Requests = hypervisorDefaultRequests
	}
	if len(spec.Containers[0].Resources.Limits) == 0 {
		spec.Containers[0].Resources.Limits = hypervisorDefaultLimits
	}

	// TODO HypervisorVerifyServiceAccountEnabledEnvVar and Public Key
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

func AddTFNodeDiscoveryConfAfterTemplate(ctx context.Context, tmpl *v1.PodTemplateSpec, pool *tfv1.GPUPool, gpuNodeName string) {
	tmpl.Spec.RestartPolicy = v1.RestartPolicyOnFailure
	serviceAccountName := GetSelfServiceAccountNameShort()
	if serviceAccountName == "" {
		serviceAccountName = constants.NamespaceDefaultVal
	}
	tmpl.Spec.ServiceAccountName = serviceAccountName
	tmpl.Spec.TerminationGracePeriodSeconds = constants.GracefulPeriodSeconds

	if len(tmpl.Spec.Containers) == 0 {
		tmpl.Spec.Containers = []v1.Container{
			{
				Name: constants.TFContainerNameNodeDiscovery,
			},
		}
	}

	if pool.Spec.ComponentConfig.NodeDiscovery.Image != "" {
		tmpl.Spec.Containers[0].Image = pool.Spec.ComponentConfig.NodeDiscovery.Image
	}

	tmpl.Spec.Containers[0].Env = append(tmpl.Spec.Containers[0].Env, v1.EnvVar{
		Name:  constants.NodeDiscoveryReportGPUNodeEnvName,
		Value: gpuNodeName,
	}, v1.EnvVar{
		Name: constants.NodeDiscoveryHostNameEnv,
		ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: constants.NodeNameFieldRef,
			},
		},
	}, v1.EnvVar{
		Name:  constants.NvidiaVisibleAllDeviceEnv,
		Value: constants.NvidiaVisibleAllDeviceValue,
	})

	tmpl.Spec.Containers[0].VolumeMounts = append(tmpl.Spec.Containers[0].VolumeMounts, v1.VolumeMount{
		Name:      constants.TensorFusionGPUInfoConfigVolumeName,
		MountPath: constants.TensorFusionGPUInfoConfigMountPath,
		SubPath:   constants.TensorFusionGPUInfoConfigSubPath,
	})

	tmpl.Spec.Volumes = append(tmpl.Spec.Volumes, v1.Volume{
		Name: constants.TensorFusionGPUInfoConfigVolumeName,
		VolumeSource: v1.VolumeSource{
			ConfigMap: &v1.ConfigMapVolumeSource{
				LocalObjectReference: v1.LocalObjectReference{
					Name: constants.TensorFusionGPUInfoConfigName,
				},
			},
		},
	})

	if len(tmpl.Spec.Containers[0].Resources.Limits) == 0 {
		tmpl.Spec.Containers[0].Resources.Limits = nodeDiscoveryDefaultLimits
	}
	if len(tmpl.Spec.Containers[0].Resources.Requests) == 0 {
		tmpl.Spec.Containers[0].Resources.Requests = nodeDiscoveryDefaultRequests
	}
}

func AddWorkerConfAfterTemplate(ctx context.Context, spec *v1.PodSpec, workerConfig *tfv1.WorkerConfig, hypervisorConfig *tfv1.HypervisorConfig, workload *tfv1.TensorFusionWorkload) string {
	// NOTE: need to set environment variable to make all GPUs visible to the worker,
	// vgpu.rs limiter will limit to specific devices after Pod started
	spec.Containers[0].Name = constants.TFContainerNameWorker
	if workerConfig.Image != "" {
		spec.Containers[0].Image = workerConfig.Image
	}
	spec.Containers[0].VolumeMounts = append(
		spec.Containers[0].VolumeMounts,
		v1.VolumeMount{
			Name:      constants.DataVolumeName,
			MountPath: constants.SharedMemDeviceName,
			// TODO not working.
			// + constants.TFLibsVolumeMountPath
			// SubPathExpr: constants.TFDataPathWorkerExpr,
			SubPath:          constants.SharedMemMountSubPath,
			MountPropagation: ptr.To(v1.MountPropagationHostToContainer),
		})
	spec.Containers[0].Env = append(spec.Containers[0].Env, v1.EnvVar{
		Name:  constants.NvidiaVisibleAllDeviceEnv,
		Value: constants.NvidiaVisibleAllDeviceValue,
	}, v1.EnvVar{
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
		Name:  constants.LdPreloadEnv,
		Value: constants.LdPreloadLimiter,
	}, v1.EnvVar{
		Name: constants.PodNamespaceEnv,
		ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{
				FieldPath: constants.NamespaceFieldRef,
			},
		},
	})

	disabledFeatures := workload.Annotations[constants.DisableFeaturesAnnotation]
	if disabledFeatures != "" {
		spec.Containers[0].Env = convertDisabledFeaturesToEnvs(disabledFeatures, spec.Containers[0].Env)
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

	// TODO support hostNetwork mode and InfiniBand for higher performance
	spec.Containers[0].Ports = append(spec.Containers[0].Ports, v1.ContainerPort{
		ContainerPort: constants.TensorFusionRemoteWorkerPortNumber,
		Name:          constants.TensorFusionRemoteWorkerPortName,
		Protocol:      v1.ProtocolTCP,
	})
	spec.TerminationGracePeriodSeconds = constants.GracefulPeriodSeconds

	if len(spec.Containers[0].Command) == 0 {
		if strings.Contains(disabledFeatures, constants.BuiltInFeatureStartWorker) {
			spec.Containers[0].Command = []string{
				"sleep",
				"infinity",
			}
		} else {
			spec.Containers[0].Command = []string{
				"./tensor-fusion-worker",
				"-p",
				strconv.Itoa(int(constants.TensorFusionRemoteWorkerPortNumber)),
			}
		}
	}

	if len(spec.Containers[0].Resources.Requests) == 0 {
		spec.Containers[0].Resources.Requests = workerDefaultRequests
	}

	return spec.Containers[0].Name
}
