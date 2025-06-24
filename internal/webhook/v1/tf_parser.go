package v1

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type TFResource struct {
	ContainerName       string
	ConnectionName      string
	ConnectionNamespace string
	TflopsRequest       resource.Quantity
	VramRequest         resource.Quantity
	TflopsLimit         resource.Quantity
	VramLimit           resource.Quantity
	GPUModel            string // Required GPU model (e.g., A100, H100)
}

type TensorFusionInfo struct {
	Profile         *tfv1.WorkloadProfileSpec
	Replicas        int32
	EnabledReplicas *int32
	WorkloadName    string
	ContainerNames  []string
	GenWorkload     bool
}

func ParseTensorFusionInfo(
	ctx context.Context,
	k8sClient client.Client,
	pod *corev1.Pod,
) (TensorFusionInfo, error) {
	var info TensorFusionInfo
	if pod.Annotations == nil {
		return info, fmt.Errorf("no annotations found")
	}
	enabledReplicas, specifiedPool := pod.Annotations[constants.TensorFusionEnabledReplicasAnnotation]
	if !specifiedPool {
		info.EnabledReplicas = nil
	} else {
		val, err := strconv.ParseInt(enabledReplicas, 10, 32)
		if err != nil {
			return info, fmt.Errorf("invalid enabledReplicas value: %s, err: %w", enabledReplicas, err)
		}
		val32 := int32(val)
		info.EnabledReplicas = &val32
	}

	workloadName, specifiedPool := pod.Annotations[constants.WorkloadKey]
	if !specifiedPool {
		return info, fmt.Errorf("workload key not found")
	}
	info.WorkloadName = workloadName
	genWorkload, specifiedPool := pod.Annotations[constants.GenWorkloadAnnotation]
	info.GenWorkload = (specifiedPool && genWorkload == constants.TrueStringValue)

	replicas, specifiedPool := pod.Annotations[constants.ReplicasAnnotation]

	if !specifiedPool {
		info.Replicas = 1
	} else {
		val, err := strconv.ParseInt(replicas, 10, 32)
		if err != nil {
			return info, fmt.Errorf("invalid replicas value: %w", err)
		}
		info.Replicas = int32(val)
	}

	workloadProfileName, specifiedPool := pod.Annotations[constants.WorkloadProfileAnnotation]
	workloadProfile := &tfv1.WorkloadProfile{}
	if specifiedPool {
		if err := k8sClient.Get(ctx, client.ObjectKey{Name: workloadProfileName, Namespace: pod.Namespace}, workloadProfile); err != nil {
			return info, fmt.Errorf("get workload profile(%s) : %w", workloadProfileName, err)
		}
	}

	gpuPoolList := &tfv1.GPUPoolList{}
	if err := k8sClient.List(ctx, gpuPoolList); err != nil {
		return info, fmt.Errorf("list gpu pools: %w", err)
	}

	poolName, specifiedPool := pod.Annotations[constants.GpuPoolKey]
	validPool := false
	// verify gpu pool name or assign default pool when not specified
	for _, gpuPool := range gpuPoolList.Items {
		if !specifiedPool && gpuPool.Annotations != nil &&
			gpuPool.Annotations[constants.TensorFusionDefaultPoolKeyAnnotation] == constants.TrueStringValue {
			poolName = gpuPool.Name
			validPool = true
			break
		}
		if specifiedPool && gpuPool.Name == poolName {
			validPool = true
			break
		}
	}
	if !validPool {
		return info, fmt.Errorf("gpu pool not found")
	}

	workloadProfile.Spec.PoolName = poolName

	tflopsRequest, specifiedPool := pod.Annotations[constants.TFLOPSRequestAnnotation]
	if specifiedPool {
		workloadProfile.Spec.Resources.Requests.Tflops = resource.MustParse(tflopsRequest)
	}
	vramRequest, specifiedPool := pod.Annotations[constants.VRAMRequestAnnotation]
	if specifiedPool {
		workloadProfile.Spec.Resources.Requests.Vram = resource.MustParse(vramRequest)
	}
	tflopsLimit, specifiedPool := pod.Annotations[constants.TFLOPSLimitAnnotation]
	if specifiedPool {
		workloadProfile.Spec.Resources.Limits.Tflops = resource.MustParse(tflopsLimit)
	}
	vramLimit, specifiedPool := pod.Annotations[constants.VRAMLimitAnnotation]
	if specifiedPool {
		workloadProfile.Spec.Resources.Limits.Vram = resource.MustParse(vramLimit)
	}
	gpuCount, specifiedPool := pod.Annotations[constants.GpuCountKey]
	if specifiedPool {
		val, err := strconv.ParseInt(gpuCount, 10, 32)
		if err != nil {
			return info, fmt.Errorf("invalid gpuCount value: %w", err)
		}
		workloadProfile.Spec.GPUCount = uint(val)
	}

	localGPU, specifiedPool := pod.Annotations[constants.IsLocalGPUAnnotation]
	if specifiedPool && localGPU == constants.TrueStringValue {
		workloadProfile.Spec.IsLocalGPU = true
		// default to no-standalone-worker, namely embedded worker when it's local GPU mode
		standaloneWorkerMode := pod.Annotations[constants.StandaloneWorkerModeAnnotation]
		workloadProfile.Spec.StandaloneWorkerMode = standaloneWorkerMode == constants.TrueStringValue
	} else {
		// default to standalone worker mode, ignore standalone-worker-mode annotation when it's not local GPU mode
		workloadProfile.Spec.StandaloneWorkerMode = true
	}

	// Parse auto-scaling annotations
	autoLimits, specifiedPool := pod.Annotations[constants.AutoScaleLimitsAnnotation]
	if specifiedPool && autoLimits == constants.TrueStringValue {
		workloadProfile.Spec.AutoScalingConfig.AutoSetLimits.Enable = true
	}
	autoRequests, specifiedPool := pod.Annotations[constants.AutoScaleRequestsAnnotation]
	if specifiedPool && autoRequests == constants.TrueStringValue {
		workloadProfile.Spec.AutoScalingConfig.AutoSetRequests.Enable = true
	}
	autoReplicas, specifiedPool := pod.Annotations[constants.AutoScaleReplicasAnnotation]
	if specifiedPool && autoReplicas == constants.TrueStringValue {
		workloadProfile.Spec.AutoScalingConfig.AutoSetReplicas.Enable = true
	}

	injectContainer, specifiedPool := pod.Annotations[constants.InjectContainerAnnotation]
	containerNames := strings.Split(injectContainer, ",")
	if len(pod.Spec.Containers) > 1 {
		if !specifiedPool || len(containerNames) == 0 {
			return info, fmt.Errorf("inject container has to be specified when Pod containers > 1")
		}
	} else {
		// assign default container name when annotation not specified
		containerNames = []string{pod.Spec.Containers[0].Name}
	}

	gpuModel, specifiedPool := pod.Annotations[constants.GPUModelAnnotation]
	if specifiedPool {
		workloadProfile.Spec.GPUModel = gpuModel
	}

	info.Profile = &workloadProfile.Spec
	info.ContainerNames = containerNames
	return info, nil
}
