package v1

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
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

func ParseTensorFusionInfo(
	ctx context.Context,
	k8sClient client.Client,
	pod *corev1.Pod,
) (utils.TensorFusionInfo, error) {
	var info utils.TensorFusionInfo
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

	// Generate the workload name
	if controllerRef, err := utils.GetPodControllerRef(ctx, k8sClient, pod); err == nil {
		if controllerRef != nil {
			info.WorkloadName = controllerRef.Name
			info.PodControllerRef = controllerRef
		} else {
			if pod.Name == "" {
				info.WorkloadName = pod.GenerateName + "-" + utils.NewShortID(8)
			} else {
				info.WorkloadName = pod.Name
			}
		}
	} else {
		return info, err
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

	// Handle dedicated GPU logic
	err = handleDedicatedGPU(pod, workloadProfile)
	if err != nil {
		return info, fmt.Errorf("handle dedicated GPU: %w", err)
	}

	info.Profile = &workloadProfile.Spec
	info.ContainerNames = containerNames
	return info, nil
}

func parseAutoScalingAnnotations(pod *corev1.Pod, workloadProfile *tfv1.WorkloadProfile) {
	autoResources, ok := pod.Annotations[constants.AutoScaleResourcesAnnotation]
	if ok && autoResources == constants.TrueStringValue {
		workloadProfile.Spec.AutoScalingConfig.AutoSetResources.Enable = true
	}
	targetResource, ok := pod.Annotations[constants.AutoScaleTargetResourceAnnotation]
	if ok {
		workloadProfile.Spec.AutoScalingConfig.AutoSetResources.TargetResource = targetResource
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

	qosLevel, hasValue := pod.Annotations[constants.QoSLevelAnnotation]
	if hasValue {
		workloadProfile.Spec.Qos = tfv1.QoSLevel(qosLevel)
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

// handleDedicatedGPU handles dedicated GPU annotation by setting full GPU capacity
func handleDedicatedGPU(pod *corev1.Pod, workloadProfile *tfv1.WorkloadProfile) error {
	dedicatedGPU, ok := pod.Annotations[constants.DedicatedGPUAnnotation]
	if !ok || dedicatedGPU != constants.TrueStringValue {
		return nil // Not a dedicated GPU request
	}

	// Must have GPU model specified for dedicated GPU
	if workloadProfile.Spec.GPUModel == "" {
		return fmt.Errorf("dedicated GPU requires gpu-model annotation to be specified")
	}

	// Get full GPU capacity from pricing provider
	resource, found := gpuallocator.GPUCapacityMap[workloadProfile.Spec.GPUModel]
	if !found {
		return fmt.Errorf("could not find capacity information for GPU model: %s", workloadProfile.Spec.GPUModel)
	}

	// Set full capacity for both requests and limits
	workloadProfile.Spec.Resources.Requests.Tflops = resource.Tflops
	workloadProfile.Spec.Resources.Requests.Vram = resource.Vram
	workloadProfile.Spec.Resources.Limits.Tflops = resource.Tflops
	workloadProfile.Spec.Resources.Limits.Vram = resource.Vram
	return nil
}
