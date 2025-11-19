package v1

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"
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

// nolint:gocyclo
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

	pool, err := checkAndGetValidGPUPool(ctx, k8sClient, pod)
	if err != nil {
		return info, err
	}
	workloadProfile.Spec.PoolName = pool.Name

	localGPU, ok := pod.Annotations[constants.IsLocalGPUAnnotation]
	if ok && localGPU == constants.TrueStringValue {
		workloadProfile.Spec.IsLocalGPU = true
	} else if ok && localGPU == constants.FalseStringValue {
		workloadProfile.Spec.IsLocalGPU = false
	} else {
		workloadProfile.Spec.IsLocalGPU = pool.Spec.DefaultUsingLocalGPU == nil || *pool.Spec.DefaultUsingLocalGPU
	}

	// check if its sidecar worker mode
	sidecarWorker, ok := pod.Annotations[constants.SidecarWorkerAnnotation]
	if ok && sidecarWorker == constants.TrueStringValue {
		workloadProfile.Spec.IsLocalGPU = true
		workloadProfile.Spec.SidecarWorker = true
	}

	// check if its compute isolation mode
	computeIsolation, ok := pod.Annotations[constants.IsolationModeAnnotation]
	if ok {
		workloadProfile.Spec.Isolation = tfv1.IsolationModeType(computeIsolation)
	} else {
		workloadProfile.Spec.Isolation = tfv1.IsolationModeSoft
	}

	// Read partition template ID annotation if in partitioned mode
	if workloadProfile.Spec.Isolation == tfv1.IsolationModePartitioned {
		if partitionTemplateID, ok := pod.Annotations[constants.PartitionTemplateIDAnnotation]; ok && partitionTemplateID != "" {
			// Store in a custom field or annotation for later use in ComposeAllocateRequest
			// We'll need to add this to WorkloadProfile or pass it through annotations
			// For now, we'll store it in pod annotations and read it in ComposeAllocateRequest
		}
	}

	workerPodTemplate, ok := pod.Annotations[constants.WorkerPodTemplateAnnotation]
	if ok && workerPodTemplate != "" {
		if workloadProfile.Spec.IsLocalGPU {
			return info, fmt.Errorf("worker pod template is not supported in localGPU mode")
		} else if workloadProfile.Spec.SidecarWorker {
			return info, fmt.Errorf("worker pod template is not supported in SidecarWorker mode")
		}
		workerPodTemplateSpec := &corev1.PodTemplateSpec{}
		if err := yaml.Unmarshal([]byte(workerPodTemplate), workerPodTemplateSpec); err != nil {
			return info, fmt.Errorf("unmarshal worker pod template from annotation: %w", err)
		}
		// Marshal to JSON and store as RawExtension to keep generic object while preserving content
		raw, err := json.Marshal(workerPodTemplateSpec)
		if err != nil {
			return info, fmt.Errorf("marshal worker pod template to json: %w", err)
		}
		workloadProfile.Spec.WorkerPodTemplate = &runtime.RawExtension{Raw: raw}
	}

	nsQuotas := &tfv1.GPUResourceQuotaList{}
	nsQuotasErr := k8sClient.List(ctx, nsQuotas, client.InNamespace(pod.Namespace))
	if nsQuotasErr == nil && len(nsQuotas.Items) > 0 {
		setDefaultQuotasIfExists(workloadProfile, nsQuotas.Items[0].Spec.Single)
	}

	err = parseGPUResourcesAnnotations(pod, workloadProfile)
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
	// extract any containers has GPU count limits and set to annotation
	isMigratedFromContainerLimits := false
	gpuCount, hasValue := pod.Annotations[constants.GpuCountAnnotation]
	if hasValue {
		val, err := strconv.ParseInt(gpuCount, 10, 32)
		if err != nil {
			return fmt.Errorf("invalid gpuCount value: %w", err)
		}
		workloadProfile.Spec.GPUCount = uint32(val)
	} else if workloadProfile.Spec.GPUCount == 0 {
		for _, container := range pod.Spec.Containers {
			if quantity, ok := container.Resources.Limits[constants.NvidiaGPUKey]; ok {
				gpuNumber, err := strconv.Atoi(quantity.String())
				if err != nil || gpuNumber <= 0 {
					ctrl.Log.Error(err, "unrecognized nvidia.com/gpu in resources, not a valid number", "pod", pod.Name, "container", container.Name)
				} else {
					workloadProfile.Spec.GPUCount = uint32(gpuNumber)
					// For seamless migration with only one tensor-fusion.ai/enabled label
					// and one tensor-fusion.ai/vram-limit annotation, convert this to 100% computing-percent
					workloadProfile.Spec.Resources.Limits.ComputePercent = resource.MustParse("100")
					isMigratedFromContainerLimits = true
					// convert limits containers to annotation for inject container when not specified
					if pod.Annotations[constants.InjectContainerAnnotation] == "" {
						pod.Annotations[constants.InjectContainerAnnotation] = container.Name
					}
					break
				}
			}
		}
	}

	if tflopsLimit, hasValue := parseResourceQuantity(pod, constants.TFLOPSLimitAnnotation); hasValue {
		workloadProfile.Spec.Resources.Limits.Tflops = tflopsLimit
		// clean compute percent limit when tflops limit is set in annotation
		if isMigratedFromContainerLimits {
			workloadProfile.Spec.Resources.Limits.ComputePercent = resource.Quantity{}
		}
	}
	if vramLimit, hasValue := parseResourceQuantity(pod, constants.VRAMLimitAnnotation); hasValue {
		workloadProfile.Spec.Resources.Limits.Vram = vramLimit
	}

	if tflopsRequest, hasValue := parseResourceQuantity(pod, constants.TFLOPSRequestAnnotation); hasValue {
		workloadProfile.Spec.Resources.Requests.Tflops = tflopsRequest
	} else if workloadProfile.Spec.Resources.Requests.Tflops.IsZero() {
		workloadProfile.Spec.Resources.Requests.Tflops = workloadProfile.Spec.Resources.Limits.Tflops
	}
	if vramRequest, hasValue := parseResourceQuantity(pod, constants.VRAMRequestAnnotation); hasValue {
		workloadProfile.Spec.Resources.Requests.Vram = vramRequest
	} else if workloadProfile.Spec.Resources.Requests.Vram.IsZero() {
		workloadProfile.Spec.Resources.Requests.Vram = workloadProfile.Spec.Resources.Limits.Vram
	}

	// Percentage way to specify GPU resource request, not recommended, should use TFLOPs instead
	computeLimit, hasValue := parseResourceQuantity(pod, constants.ComputeLimitAnnotation)
	if hasValue {
		workloadProfile.Spec.Resources.Limits.ComputePercent = computeLimit
	}
	computeRequest, hasValue := parseResourceQuantity(pod, constants.ComputeRequestAnnotation)
	if hasValue {
		workloadProfile.Spec.Resources.Requests.ComputePercent = computeRequest
	} else if workloadProfile.Spec.Resources.Requests.Tflops.IsZero() && workloadProfile.Spec.Resources.Requests.ComputePercent.IsZero() {
		workloadProfile.Spec.Resources.Requests.ComputePercent = workloadProfile.Spec.Resources.Limits.ComputePercent
	}

	// tflops - computePercent are mutually exclusive
	if !workloadProfile.Spec.Resources.Requests.Tflops.IsZero() && !workloadProfile.Spec.Resources.Requests.ComputePercent.IsZero() {
		return fmt.Errorf("tflops- and computePercent request are mutually exclusive, please specify only one")
	}

	qosLevel, hasValue := pod.Annotations[constants.QoSLevelAnnotation]
	if hasValue {
		workloadProfile.Spec.Qos = tfv1.QoSLevel(qosLevel)
	}

	gpuVendor, hasValue := pod.Annotations[constants.GpuVendorAnnotation]
	if hasValue {
		workloadProfile.Spec.GPUVendor = gpuVendor
	}

	// parse GPU indices
	gpuIndices, hasError := utils.ParseIndicesAnnotation(pod.Annotations[constants.GpuIndicesAnnotation])
	if hasError {
		return fmt.Errorf("tensor-fusion.ai/gpu-indices annotation is not valid for Pod %s", pod.Name)
	}
	workloadProfile.Spec.GPUIndices = gpuIndices
	if len(gpuIndices) > 0 {
		workloadProfile.Spec.GPUCount = uint32(len(gpuIndices))
	}

	// finally add default GPU count when not specified
	if workloadProfile.Spec.GPUCount == 0 {
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

func checkAndGetValidGPUPool(ctx context.Context, k8sClient client.Client, pod *corev1.Pod) (*tfv1.GPUPool, error) {
	gpuPoolList := &tfv1.GPUPoolList{}
	if err := k8sClient.List(ctx, gpuPoolList); err != nil {
		return nil, fmt.Errorf("list gpu pools: %w", err)
	}

	poolName, poolSpecified := pod.Annotations[constants.GpuPoolKey]
	validPool := tfv1.GPUPool{}
	// verify gpu pool name or assign default pool when not specified
	for _, gpuPool := range gpuPoolList.Items {
		if !poolSpecified && gpuPool.Annotations != nil &&
			gpuPool.Annotations[constants.TensorFusionDefaultPoolKeyAnnotation] == constants.TrueStringValue {
			validPool = gpuPool
			break
		}
		if poolSpecified && gpuPool.Name == poolName {
			validPool = gpuPool
			break
		}
	}
	if validPool.Name == "" {
		return nil, fmt.Errorf("gpu pool not found")
	}
	return &validPool, nil
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
