package utils

import (
	context "context"
	"errors"
	"fmt"
	"maps"
	"os"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	constants "github.com/NexusGPU/tensor-fusion/internal/constants"
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
	v1.ResourceMemory: resource.MustParse("64Mi"),
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
var sharedMemMaxSize = resource.NewQuantity(512*1024*1024, resource.DecimalSI)

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
	pod.Annotations[constants.SidecarWorkerAnnotation] = strconv.FormatBool(tfInfo.Profile.SidecarWorker)
	// add inject container annotation for client Pod, in case user doesn't specify it
	pod.Annotations[constants.InjectContainerAnnotation] = strings.Join(tfInfo.ContainerNames, ",")
	pod.Annotations[constants.IsolationModeAnnotation] = string(tfInfo.Profile.Isolation)
	// add partition template ID if in partitioned mode
	if tfInfo.Profile.Isolation == tfv1.IsolationModePartitioned && tfInfo.Profile.PartitionTemplateID != "" {
		pod.Annotations[constants.PartitionTemplateIDAnnotation] = tfInfo.Profile.PartitionTemplateID
	}
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

	// Add gang scheduling annotations if configured
	if workload.Spec.GangScheduling != nil && workload.Spec.GangScheduling.MinMembers > 0 {
		annotations[constants.GangMinMembersAnnotation] = strconv.Itoa(int(workload.Spec.GangScheduling.MinMembers))
		if workload.Spec.GangScheduling.Timeout != nil && workload.Spec.GangScheduling.Timeout.Duration > 0 {
			annotations[constants.GangTimeoutAnnotation] = workload.Spec.GangScheduling.Timeout.Duration.String()
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
	clientConfig := pool.Spec.ComponentConfig.Client
	image := getProviderImageOrDefault(tfInfo.Profile.GPUVendor, clientConfig.ProviderImage, clientConfig.Image)
	pod.Spec.InitContainers = append(pod.Spec.InitContainers, v1.Container{
		Name:  constants.TFContainerNameClient,
		Image: image,
		VolumeMounts: []v1.VolumeMount{
			{
				Name:      constants.TFLibsVolumeName,
				MountPath: constants.TFLibsVolumeMountPath,
			},
			{
				Name:      constants.TFConfVolumeName,
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
	pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
		Name: constants.TFConfVolumeName,
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
				Name:      constants.TFConfVolumeName,
				MountPath: constants.LdPreloadFile,
				SubPath:   constants.LdPreloadFileName,
				ReadOnly:  true,
			}, v1.VolumeMount{
				Name:      constants.TFConfVolumeName,
				MountPath: constants.LdLibraryPathFile,
				SubPath:   constants.LdLibraryPathFileName,
				ReadOnly:  true,
			}, v1.VolumeMount{
				Name:      constants.TFLibsVolumeName,
				MountPath: constants.TFLibsVolumeMountPath,
			})
	}

	if tfInfo.Profile.IsLocalGPU {
		// shm to communicate between worker and hypervisor
		pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
			Name: constants.DataVolumeName,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: constants.TFDataPath,
					Type: ptr.To(v1.HostPathDirectoryOrCreate),
				},
			},
		})

		if tfInfo.Profile.SidecarWorker {
			// Add shared memory for worker-client communication
			pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
				Name: constants.TransportShmVolumeName,
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{
						SizeLimit: sharedMemMaxSize,
						Medium:    v1.StorageMediumMemory,
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

			lastContainer := &pod.Spec.Containers[len(pod.Spec.Containers)-1]
			SetWorkerContainerSpec(lastContainer, tfInfo.Profile,
				pool.Spec.ComponentConfig.Worker, pool.Spec.ComponentConfig.Hypervisor,
				pod.Annotations[constants.DisableFeaturesAnnotation], true)
		}

		for _, injectContainerIndex := range injectContainerIndices {
			if tfInfo.Profile.SidecarWorker {
				// add transport shm for client container to communicate with sidecar worker
				pod.Spec.Containers[injectContainerIndex].VolumeMounts = append(
					pod.Spec.Containers[injectContainerIndex].VolumeMounts,
					v1.VolumeMount{
						Name:      constants.TransportShmVolumeName,
						MountPath: constants.TransportShmPath,
					})
				continue
			}

			// add ngpu spec, client is the same as worker, in same process
			pod.Spec.Containers[injectContainerIndex].VolumeMounts = append(
				pod.Spec.Containers[injectContainerIndex].VolumeMounts,
				v1.VolumeMount{
					Name:             constants.DataVolumeName,
					MountPath:        constants.TFDataPath + constants.SharedMemMountSubPath,
					SubPathExpr:      constants.TFDataPathWorkerExpr,
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
	composeHypervisorContainer(spec, pool, enableVector)

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
		Name:    "init-shm",
		Image:   hypervisorConfig.Image,
		Command: []string{constants.ComponentHypervisor, constants.MountShmSubcommand},
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
	}, v1.Container{
		Name:    "init-runtime",
		Image:   getProviderImageOrDefault(vendor, hypervisorConfig.ProviderImage, hypervisorConfig.Image),
		Command: []string{"cp", "-r", "/build/**", constants.TFDataPath},
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

func composeHypervisorContainer(spec *v1.PodSpec, pool *tfv1.GPUPool, enableVector bool) {
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
	// NOTE: need to set environment variable to make all GPUs visible to the worker,
	// vgpu.rs limiter will limit to specific devices after Pod started
	container.Name = constants.TFContainerNameWorker
	container.Image = getProviderImageOrDefault(workloadProfile.GPUVendor, workerConfig.ProviderImage, workerConfig.Image)
	container.VolumeMounts = append(
		container.VolumeMounts,
		v1.VolumeMount{
			Name:             constants.DataVolumeName,
			MountPath:        constants.TFDataPath + constants.SharedMemMountSubPath,
			SubPathExpr:      constants.TFDataPathWorkerExpr,
			MountPropagation: ptr.To(v1.MountPropagationHostToContainer),
		})
	container.Env = append(container.Env, v1.EnvVar{
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

	if !strings.Contains(disabledFeatures, constants.BuiltInFeaturesGpuLimiter) {
		// TODO: In hard isolation mode, current implementation relies on limiter to set CUDA_VISIBLE_DEVICES env.
		// In next hypervisor versions, device allocation will be handled by device-plugin, so LD_PRELOAD should be removed.
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

func AddWorkerConfAfterTemplate(
	ctx context.Context, spec *v1.PodSpec, workloadProfile *tfv1.WorkloadProfileSpec, workerConfig *tfv1.WorkerConfig,
	hypervisorConfig *tfv1.HypervisorConfig, workload *tfv1.TensorFusionWorkload,
) string {
	disabledFeatures := workload.Annotations[constants.DisableFeaturesAnnotation]

	// Configure worker container
	SetWorkerContainerSpec(&spec.Containers[0], workloadProfile, workerConfig, hypervisorConfig, disabledFeatures, false)

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

	return spec.Containers[0].Name
}

func getProviderImageOrDefault(vendor string, providerImageMap map[string]string, defaultImage string) string {
	if providerImageMap != nil && providerImageMap[vendor] != "" {
		return providerImageMap[vendor]
	}
	return defaultImage
}
