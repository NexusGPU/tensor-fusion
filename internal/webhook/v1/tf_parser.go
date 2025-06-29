package v1

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
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
	DynamicReplicas bool
	EnabledReplicas *int32
	WorkloadName    string
	ContainerNames  []string
	GenWorkload     bool

	// Pod mutating webhook can not get Pod UID sometimes,
	// thus need pod controller to set the owner reference
	PendingSetPodAsOwner bool
}

func ParseTensorFusionInfo(
	ctx context.Context,
	k8sClient client.Client,
	pod *corev1.Pod,
) (TensorFusionInfo, error) {
	var info TensorFusionInfo
	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}
	enabledReplicas, grayMigration := pod.Annotations[constants.TensorFusionEnabledReplicasAnnotation]
	if !grayMigration {
		info.EnabledReplicas = nil
	} else {
		val, err := strconv.ParseInt(enabledReplicas, 10, 32)
		if err != nil {
			return info, fmt.Errorf("invalid enabledReplicas value: %s, err: %w", enabledReplicas, err)
		}
		val32 := int32(val)
		info.EnabledReplicas = &val32
	}

	workloadName, ok := pod.Annotations[constants.WorkloadKey]
	if !ok {
		// auto generate a workload with owner name
		info.GenWorkload = true
		owner := utils.FindFirstLevelOwnerReference(pod)
		if owner == nil {
			if pod.Name == "" {
				info.WorkloadName = pod.GenerateName + "-" + utils.NewShortID(8)
			} else {
				info.WorkloadName = pod.Name
			}
			info.PendingSetPodAsOwner = true
		} else {
			info.WorkloadName = owner.Name
		}
	} else {
		// when workload is manually created, user can specify workload's replicas
		// it remotely connects to lease connection worker when SelectWorker
		info.WorkloadName = workloadName
	}

	workloadProfileName, ok := pod.Annotations[constants.WorkloadProfileAnnotation]
	workloadProfile := &tfv1.WorkloadProfile{}
	if ok {
		if err := k8sClient.Get(ctx, client.ObjectKey{Name: workloadProfileName, Namespace: pod.Namespace}, workloadProfile); err != nil {
			return info, fmt.Errorf("get workload profile(%s) : %w", workloadProfileName, err)
		}
	}

	localGPU, ok := pod.Annotations[constants.IsLocalGPUAnnotation]
	if ok && localGPU == constants.TrueStringValue {
		workloadProfile.Spec.IsLocalGPU = true
	}

	if poolName, err := getGPUPoolNameAndVerify(ctx, k8sClient, pod); err != nil {
		return info, err
	} else {
		workloadProfile.Spec.PoolName = poolName
	}

	nsQuotas := &tfv1.GPUResourceQuotaList{}
	nsQuotasErr := k8sClient.List(ctx, nsQuotas, client.InNamespace(pod.Namespace))
	if nsQuotasErr == nil && len(nsQuotas.Items) > 0 {
		setDefaultQuotasIfExists(workloadProfile, nsQuotas.Items[0].Spec.Single)
	}

	err := parseGPUResourcesAnnotations(pod, workloadProfile)
	if err != nil {
		return info, err
	}

	parseAutoScalingAnnotations(pod, workloadProfile)

	injectContainer, ok := pod.Annotations[constants.InjectContainerAnnotation]
	containerNames := strings.Split(injectContainer, ",")
	if len(pod.Spec.Containers) > 1 {
		if !ok || len(containerNames) == 0 {
			return info, fmt.Errorf("inject container has to be specified when Pod containers > 1")
		}
	} else {
		// assign default container name when annotation not specified
		containerNames = []string{pod.Spec.Containers[0].Name}
	}

	gpuModel, ok := pod.Annotations[constants.GPUModelAnnotation]
	if ok {
		workloadProfile.Spec.GPUModel = gpuModel
	}

	info.Profile = &workloadProfile.Spec
	info.ContainerNames = containerNames
	return info, nil
}

func parseAutoScalingAnnotations(pod *corev1.Pod, workloadProfile *tfv1.WorkloadProfile) {
	autoLimits, ok := pod.Annotations[constants.AutoScaleLimitsAnnotation]
	if ok && autoLimits == constants.TrueStringValue {
		workloadProfile.Spec.AutoScalingConfig.AutoSetLimits.Enable = true
	}
	autoRequests, ok := pod.Annotations[constants.AutoScaleRequestsAnnotation]
	if ok && autoRequests == constants.TrueStringValue {
		workloadProfile.Spec.AutoScalingConfig.AutoSetRequests.Enable = true
	}
	autoReplicas, ok := pod.Annotations[constants.AutoScaleReplicasAnnotation]
	if ok && autoReplicas == constants.TrueStringValue {
		workloadProfile.Spec.AutoScalingConfig.AutoSetReplicas.Enable = true
	}
}

func parseGPUResourcesAnnotations(pod *corev1.Pod, workloadProfile *tfv1.WorkloadProfile) error {
	if tflopsRequest, hasValue := parseResourceQuantity(pod, constants.TFLOPSRequestAnnotation); hasValue {
		workloadProfile.Spec.Resources.Requests.Tflops = tflopsRequest
	}
	if vramRequest, hasValue := parseResourceQuantity(pod, constants.VRAMRequestAnnotation); hasValue {
		workloadProfile.Spec.Resources.Requests.Vram = vramRequest
	}
	if tflopsLimit, hasValue := parseResourceQuantity(pod, constants.TFLOPSLimitAnnotation); hasValue {
		workloadProfile.Spec.Resources.Limits.Tflops = tflopsLimit
	}
	if vramLimit, hasValue := parseResourceQuantity(pod, constants.VRAMLimitAnnotation); hasValue {
		workloadProfile.Spec.Resources.Limits.Vram = vramLimit
	}

	gpuCount, hasValue := pod.Annotations[constants.GpuCountAnnotation]
	if hasValue {
		val, err := strconv.ParseInt(gpuCount, 10, 32)
		if err != nil {
			return fmt.Errorf("invalid gpuCount value: %w", err)
		}
		workloadProfile.Spec.GPUCount = uint32(val)
	} else if workloadProfile.Spec.GPUCount == 0 {
		workloadProfile.Spec.GPUCount = 1
	}
	return nil
}

func parseResourceQuantity(pod *corev1.Pod, annotationKey string) (resource.Quantity, bool) {
	if pod.Annotations == nil || pod.Annotations[annotationKey] == "" {
		return resource.Quantity{}, false
	}
	quantity, err := resource.ParseQuantity(pod.Annotations[annotationKey])
	if err != nil {
		// not valid quantity, return empty quantity, handle error in scheduler
		return resource.Quantity{}, false
	}
	return quantity, true
}

func getGPUPoolNameAndVerify(ctx context.Context, k8sClient client.Client, pod *corev1.Pod) (string, error) {
	gpuPoolList := &tfv1.GPUPoolList{}
	if err := k8sClient.List(ctx, gpuPoolList); err != nil {
		return "", fmt.Errorf("list gpu pools: %w", err)
	}

	poolName, ok := pod.Annotations[constants.GpuPoolKey]
	validPool := false
	// verify gpu pool name or assign default pool when not specified
	for _, gpuPool := range gpuPoolList.Items {
		if !ok && gpuPool.Annotations != nil &&
			gpuPool.Annotations[constants.TensorFusionDefaultPoolKeyAnnotation] == constants.TrueStringValue {
			poolName = gpuPool.Name
			validPool = true
			break
		}
		if ok && gpuPool.Name == poolName {
			validPool = true
			break
		}
	}
	if !validPool {
		return "", fmt.Errorf("gpu pool not found")
	}
	return poolName, nil
}

func setDefaultQuotasIfExists(workloadProfile *tfv1.WorkloadProfile, single tfv1.GPUResourceQuotaSingle) {
	defaultReq := single.DefaultRequests.DeepCopy()
	if defaultReq != nil {
		if workloadProfile.Spec.Resources.Requests.Tflops.IsZero() {
			workloadProfile.Spec.Resources.Requests.Tflops = defaultReq.Tflops
		}
		if workloadProfile.Spec.Resources.Requests.Vram.IsZero() {
			workloadProfile.Spec.Resources.Requests.Vram = defaultReq.Vram
		}
	}

	defaultLimit := single.DefaultLimits.DeepCopy()
	if defaultLimit != nil {
		if workloadProfile.Spec.Resources.Limits.Tflops.IsZero() {
			workloadProfile.Spec.Resources.Limits.Tflops = defaultLimit.Tflops
		}
		if workloadProfile.Spec.Resources.Limits.Vram.IsZero() {
			workloadProfile.Spec.Resources.Limits.Vram = defaultLimit.Vram
		}
	}
}
