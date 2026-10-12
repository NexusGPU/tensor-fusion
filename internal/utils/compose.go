package utils

import (
	context "context"
	"errors"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
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

const legacyTFContainerNameWorker = "tensor-fusion-worker"

// hypervisorMetricsOutputPathFlag is the hypervisor flag (cmd/hypervisor) that
// sets the metrics file path.
const hypervisorMetricsOutputPathFlag = "metrics-output-path"

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

// The same limits as the controller's Vector sidecar has in the chart.
var vectorDefaultLimits v1.ResourceList = v1.ResourceList{
	v1.ResourceCPU:    resource.MustParse("1000m"),
	v1.ResourceMemory: resource.MustParse("512Mi"),
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

func AddTFHypervisorConfAfterTemplate(ctx context.Context, spec *v1.PodSpec, pool *tfv1.GPUPool, vendor string, compatibleWithNvidiaContainerToolkit bool) {
	// Hypervisor needs to read /proc to map pod with processID
	spec.HostPID = true
	spec.TerminationGracePeriodSeconds = constants.GracefulPeriodSeconds
	spec.PriorityClassName = constants.NodeCriticalPriorityClassName

	enableVector := pool.Spec.ComponentConfig.Hypervisor != nil && pool.Spec.ComponentConfig.Hypervisor.EnableVector
	// Opt-in: the kubelet pod-resources proxy (DCGM exporter integration) is
	// off by default. When enabled, the operator injects the pod-resources-tf
	// hostPath volume/mount and propagates the same env into the hypervisor
	// container so the backend agrees with what the operator decided here.
	enablePodResourcesProxy := os.Getenv(constants.HypervisorPodResourcesProxyEnabledEnv) == constants.TrueStringValue

	// when no config or config is not valid, reset hypervisor&vector container
	if enableVector && len(spec.Containers) == 1 {
		// Template only declares the hypervisor container: keep it (env, resources,
		// securityContext) and add the vector sidecar next to it.
		spec.Containers = append(spec.Containers, v1.Container{
			Name: constants.TFContainerVector,
		})
	}
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
	spec.Volumes = appendVolumesIfNotExists(spec.Volumes, v1.Volume{
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
				// Only the Vector sidecar reads it. Without the ConfigMap
				// (an installation that does not use the chart) the pod
				// must still start: the hypervisor does not depend on it.
				Optional: ptr.To(true),
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

	if enablePodResourcesProxy {
		// Sibling dir for the pod-resources proxy socket exposed to DCGM exporter.
		spec.Volumes = appendVolumesIfNotExists(spec.Volumes, v1.Volume{
			Name: constants.KubeletPodResourcesProxyVolumeName,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: constants.KubeletPodResourcesProxyPath,
					Type: ptr.To(v1.HostPathDirectoryOrCreate),
				},
			},
		})
	}

	composeHypervisorInitContainer(ctx, spec, pool, vendor, compatibleWithNvidiaContainerToolkit)
	composeHypervisorContainer(spec, pool, vendor, enableVector, enablePodResourcesProxy)
	if compatibleWithNvidiaContainerToolkit && strings.EqualFold(vendor, constants.AcceleratorVendorNvidia) {
		addNvidiaDriverContractMounts(spec)
	}

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
					Name:             constants.NvidiaValidationsVolumeName,
					MountPath:        constants.NvidiaValidationsHostPath,
					MountPropagation: ptr.To(v1.MountPropagationHostToContainer),
				},
			},
		}

		spec.InitContainers = append(spec.InitContainers, initContainer)

		// Add volume for NVIDIA validations
		spec.Volumes = appendVolumesIfNotExists(spec.Volumes, v1.Volume{
			Name: constants.NvidiaValidationsVolumeName,
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{
					Path: constants.NvidiaValidationsHostPath,
					Type: ptr.To(v1.HostPathDirectoryOrCreate),
				},
			},
		})
	}
}

func addNvidiaDriverContractMounts(spec *v1.PodSpec) {
	if spec == nil || len(spec.Containers) == 0 {
		return
	}

	bindings := []struct {
		volume v1.Volume
		mount  v1.VolumeMount
	}{
		{
			volume: v1.Volume{
				Name: constants.NvidiaValidationsVolumeName,
				VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{
					Path: constants.NvidiaValidationsHostPath,
					Type: ptr.To(v1.HostPathDirectoryOrCreate),
				}},
			},
			mount: v1.VolumeMount{
				Name: constants.NvidiaValidationsVolumeName, MountPath: constants.NvidiaValidationsHostPath, ReadOnly: true,
			},
		},
		{
			volume: v1.Volume{
				Name: constants.NvidiaHostRootVolumeName,
				VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{
					Path: constants.NvidiaHostRootHostPath,
					Type: ptr.To(v1.HostPathDirectory),
				}},
			},
			mount: v1.VolumeMount{
				Name: constants.NvidiaHostRootVolumeName, MountPath: constants.NvidiaHostRootMountPath, ReadOnly: true,
			},
		},
		{
			volume: v1.Volume{
				Name: constants.NvidiaDriverRootVolumeName,
				VolumeSource: v1.VolumeSource{HostPath: &v1.HostPathVolumeSource{
					Path: constants.NvidiaDriverRootHostPath,
					Type: ptr.To(v1.HostPathDirectoryOrCreate),
				}},
			},
			mount: v1.VolumeMount{
				Name: constants.NvidiaDriverRootVolumeName, MountPath: constants.NvidiaDriverRootMountPath, ReadOnly: true,
			},
		},
	}
	for _, binding := range bindings {
		if slices.ContainsFunc(spec.Containers[0].VolumeMounts, func(existing v1.VolumeMount) bool {
			return existing.Name == binding.mount.Name || existing.MountPath == binding.mount.MountPath
		}) {
			continue
		}
		spec.Volumes = appendVolumesIfNotExists(spec.Volumes, binding.volume)
		spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts, binding.mount)
	}
}

// appendVolumesIfNotExists merges operator-provided volumes into a pod
// template using Volume.name as the key. Volumes explicitly supplied by the
// template take precedence.
func appendVolumesIfNotExists(volumes []v1.Volume, newVolumes ...v1.Volume) []v1.Volume {
	existingNames := make(map[string]struct{}, len(volumes)+len(newVolumes))
	for _, volume := range volumes {
		existingNames[volume.Name] = struct{}{}
	}
	for _, volume := range newVolumes {
		if _, exists := existingNames[volume.Name]; exists {
			continue
		}
		volumes = append(volumes, volume)
		existingNames[volume.Name] = struct{}{}
	}
	return volumes
}

func composeHypervisorContainer(spec *v1.PodSpec, pool *tfv1.GPUPool, vendor string, enableVector bool, enablePodResourcesProxy bool) {
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
	if enablePodResourcesProxy {
		spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts, v1.VolumeMount{
			Name:      constants.KubeletPodResourcesProxyVolumeName,
			MountPath: constants.KubeletPodResourcesProxyPath,
		})
	}
	if enableVector {
		spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts, v1.VolumeMount{
			Name:      constants.LogsVolumeName,
			MountPath: constants.TensorFusionLogPath,
		})
		// The hypervisor defaults to ./metrics.log, outside the shared logs volume;
		// write where the vector sidecar tails.
		if !lo.ContainsBy(spec.Containers[0].Args, func(arg string) bool {
			return strings.Contains(arg, hypervisorMetricsOutputPathFlag)
		}) {
			spec.Containers[0].Args = append(spec.Containers[0].Args,
				"-"+hypervisorMetricsOutputPathFlag+"="+constants.TensorFusionLogPath+"/metrics.log")
		}
	}

	if spec.Containers[0].SecurityContext == nil {
		spec.Containers[0].SecurityContext = &v1.SecurityContext{}
	}
	if spec.Containers[0].SecurityContext.Capabilities == nil {
		spec.Containers[0].SecurityContext.Capabilities = &v1.Capabilities{}
	}
	if !lo.Contains(spec.Containers[0].SecurityContext.Capabilities.Add, constants.SystemPtraceCapability) {
		spec.Containers[0].SecurityContext.Capabilities.Add = append(
			spec.Containers[0].SecurityContext.Capabilities.Add,
			constants.SystemPtraceCapability,
		)
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
	spec.Containers[0].Env = setEnv(spec.Containers[0].Env, v1.EnvVar{
		Name:  constants.HypervisorListenAddrEnv,
		Value: fmt.Sprintf("%s:%d", constants.DefaultHttpBindIP, port),
	})
	spec.Containers[0].Env = setEnv(spec.Containers[0].Env, v1.EnvVar{
		Name:  constants.HypervisorPortEnv,
		Value: strconv.Itoa(int(port)),
	})
	spec.Containers[0].Env = append(spec.Containers[0].Env, v1.EnvVar{
		Name:  constants.HypervisorPoolNameEnv,
		Value: pool.Name,
	}, v1.EnvVar{
		Name:  constants.TensorFusionGPUInfoEnvVar,
		Value: constants.TensorFusionGPUInfoConfigMountPath,
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

	if enablePodResourcesProxy {
		spec.Containers[0].Env = append(spec.Containers[0].Env, v1.EnvVar{
			Name:  constants.HypervisorPodResourcesProxyEnabledEnv,
			Value: constants.TrueStringValue,
		})
	}

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

		// MIG GpuInstance creation succeeds with just `privileged: true`, but the
		// follow-up nvmlGpuInstanceGetComputeInstanceProfileInfo call needs the
		// per-GI access files under /proc/driver/nvidia/capabilities/. Those are
		// injected by NVIDIA Container Runtime only when these two env vars are
		// set; otherwise NVML returns NVML_ERROR_INSUFFICIENT_PERMISSIONS for
		// every CI profile query and partition assignment fails.
		if strings.EqualFold(strings.TrimSpace(vendor), constants.AcceleratorVendorNvidia) {
			spec.Containers[0].Env = AppendEnvVarsIfNotExists(spec.Containers[0].Env,
				v1.EnvVar{Name: "NVIDIA_MIG_CONFIG_DEVICES", Value: "all"},
				v1.EnvVar{Name: "NVIDIA_MIG_MONITOR_DEVICES", Value: "all"},
			)
		}
	}

	applyProviderHostPathMounts(spec, 0, hypervisorCfg.HostPathMounts)
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

	// A database password reaches Vector through its environment, from the
	// Secret the operator was told about; the Vector configuration only names
	// the variable.
	if secretName := os.Getenv(constants.VectorGreptimePasswordSecretEnv); secretName != "" {
		spec.Containers[1].Env = appendEnvIfMissing(spec.Containers[1].Env, v1.EnvVar{
			Name: constants.VectorGreptimePasswordEnv,
			ValueFrom: &v1.EnvVarSource{
				SecretKeyRef: &v1.SecretKeySelector{
					LocalObjectReference: v1.LocalObjectReference{Name: secretName},
					Key:                  constants.GreptimePasswordSecretKey,
				},
			},
		})
	}

	if len(spec.Containers[1].Resources.Requests) == 0 {
		spec.Containers[1].Resources.Requests = vectorDefaultRequests
	}
	if len(spec.Containers[1].Resources.Limits) == 0 {
		spec.Containers[1].Resources.Limits = vectorDefaultLimits
	}
}
