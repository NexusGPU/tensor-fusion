package v1

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
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
	if err := validateIsolationAndExecutionMode(&workloadProfile.Spec); err != nil {
		return info, err
	}

	// Read partition template ID annotation if in partitioned mode
	if workloadProfile.Spec.Isolation == tfv1.IsolationModePartitioned {
		if partitionTemplateID, ok := pod.Annotations[constants.PartitionTemplateIDAnnotation]; ok && partitionTemplateID != "" {
			workloadProfile.Spec.PartitionTemplateID = partitionTemplateID
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

	// Apply pool-level vertical scaling rules if SchedulingConfigTemplate is configured
	applyVerticalScalingRules(ctx, k8sClient, pod, pool, workloadProfile)

	injectContainer, ok := pod.Annotations[constants.InjectContainerAnnotation]
	containerNames := []string{}
	if ok && injectContainer != "" {
		containerNames = strings.Split(injectContainer, ",")
		for i := range containerNames {
			containerNames[i] = strings.TrimSpace(containerNames[i])
		}
	}
	if len(pod.Spec.Containers) > 1 {
		if len(containerNames) == 0 {
			return info, fmt.Errorf("inject container has to be specified when Pod containers > 1")
		}
	} else if len(containerNames) == 0 {
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

	// Parse gang scheduling config from annotations
	if err := parseGangSchedulingAnnotations(pod, workloadProfile); err != nil {
		return info, err
	}
	if err := validateGangSchedulingConfig(ctx, k8sClient, pod, workloadProfile, info.EnabledReplicas, info.PodControllerRef, info.WorkloadName); err != nil {
		return info, err
	}

	info.Profile = &workloadProfile.Spec
	info.ContainerNames = containerNames
	return info, nil
}

func validateIsolationAndExecutionMode(profile *tfv1.WorkloadProfileSpec) error {
	if profile == nil {
		return nil
	}
	// Local hard/soft modes run through a sibling worker container. Keep the old sidecar flag as a compatibility hint.
	if profile.Isolation == tfv1.IsolationModeHard && profile.IsLocalGPU && !utils.UseLocalWorkerSidecar(profile) {
		return fmt.Errorf("hard isolation is only supported in remote mode or local sidecar-worker mode")
	}
	return nil
}

// parseGangSchedulingAnnotations parses gang scheduling configuration from pod annotations
func parseGangSchedulingAnnotations(pod *corev1.Pod, workloadProfile *tfv1.WorkloadProfile) error {
	if pod.Annotations[constants.GangEnabledAnnotation] != constants.TrueStringValue {
		return nil
	}

	if workloadProfile.Spec.GangScheduling == nil {
		workloadProfile.Spec.GangScheduling = &tfv1.GangSchedulingConfig{}
	}

	if minMembersStr := pod.Annotations[constants.GangMinMembersAnnotation]; minMembersStr != "" {
		minMembers, err := strconv.ParseInt(minMembersStr, 10, 32)
		if err != nil {
			return fmt.Errorf("invalid %s value %q: %w", constants.GangMinMembersAnnotation, minMembersStr, err)
		}
		if minMembers < 0 {
			return fmt.Errorf("invalid %s=%d: must be >= 0", constants.GangMinMembersAnnotation, minMembers)
		}
		workloadProfile.Spec.GangScheduling.MinMembers = int32(minMembers)
	}

	// Parse timeout (optional)
	timeoutStr, hasTimeout := pod.Annotations[constants.GangTimeoutAnnotation]
	if hasTimeout && timeoutStr != "" && timeoutStr != "0" && timeoutStr != "0s" {
		timeout, err := time.ParseDuration(timeoutStr)
		if err != nil {
			return fmt.Errorf("invalid %s value %q: %w", constants.GangTimeoutAnnotation, timeoutStr, err)
		}
		workloadProfile.Spec.GangScheduling.Timeout = &metav1.Duration{Duration: timeout}
	}
	return nil
}

func validateGangSchedulingConfig(
	ctx context.Context,
	k8sClient client.Client,
	pod *corev1.Pod,
	workloadProfile *tfv1.WorkloadProfile,
	enabledReplicas *int32,
	controllerRef *metav1.OwnerReference,
	workloadName string,
) error {
	if workloadProfile == nil || workloadProfile.Spec.GangScheduling == nil {
		return nil
	}

	minMembers := workloadProfile.Spec.GangScheduling.MinMembers
	if minMembers == 1 {
		return fmt.Errorf("invalid %s=%d: explicit gang minMembers must be >= 2", constants.GangMinMembersAnnotation, minMembers)
	}

	desiredMembers, known := resolveDesiredGangMembers(ctx, k8sClient, pod, workloadProfile, enabledReplicas, controllerRef)
	if known && desiredMembers < 2 {
		return fmt.Errorf("invalid %s=true: gang scheduling requires at least 2 replicas, got %d", constants.GangEnabledAnnotation, desiredMembers)
	}
	if minMembers >= 2 && known && minMembers > desiredMembers {
		return fmt.Errorf("invalid %s=%d: exceeds desired replicas %d", constants.GangMinMembersAnnotation, minMembers, desiredMembers)
	}

	// Stamp resolved gang quorum onto the pod so the scheduler can read it
	// directly from annotations without fetching the TensorFusionWorkload CR.
	//
	// When desiredMembers cannot be resolved (owner not yet visible, transient
	// API error, etc.) but the user provided an explicit minMembers >= 2, use
	// minMembers as the fallback for both desired and required so the pod is
	// still scheduled with gang semantics (fail-closed for gang-enabled).
	if !known && minMembers >= 2 {
		desiredMembers = minMembers
		known = true
	}
	if !known {
		return fmt.Errorf("invalid %s=true: cannot determine gang size — set %s or spec.replicas on the workload",
			constants.GangEnabledAnnotation, constants.GangMinMembersAnnotation)
	}
	if desiredMembers >= 2 {
		requiredMembers := desiredMembers
		if minMembers >= 2 {
			requiredMembers = minMembers
		}
		pod.Annotations[constants.GangDesiredMembersAnnotation] = strconv.FormatInt(int64(desiredMembers), 10)
		pod.Annotations[constants.GangRequiredMembersAnnotation] = strconv.FormatInt(int64(requiredMembers), 10)
		pod.Annotations[constants.GangGroupKeyAnnotation] = pod.Namespace + "/" + workloadName
	}
	return nil
}

func resolveDesiredGangMembers(
	ctx context.Context,
	k8sClient client.Client,
	pod *corev1.Pod,
	workloadProfile *tfv1.WorkloadProfile,
	enabledReplicas *int32,
	controllerRef *metav1.OwnerReference,
) (int32, bool) {
	if enabledReplicas != nil {
		return nonNegativeInt32(*enabledReplicas), true
	}
	if workloadProfile != nil && workloadProfile.Spec.Replicas != nil {
		return nonNegativeInt32(*workloadProfile.Spec.Replicas), true
	}
	if controllerRef == nil || k8sClient == nil {
		return 0, false
	}

	owner := &unstructured.Unstructured{}
	owner.SetAPIVersion(controllerRef.APIVersion)
	owner.SetKind(controllerRef.Kind)
	if err := k8sClient.Get(ctx, client.ObjectKey{Namespace: pod.Namespace, Name: controllerRef.Name}, owner); err != nil {
		return 0, false
	}

	if replicas, found, err := unstructured.NestedInt64(owner.Object, "spec", "replicas"); err == nil && found {
		return nonNegativeInt32(int32(replicas)), true
	}
	if parallelism, found, err := unstructured.NestedInt64(owner.Object, "spec", "parallelism"); err == nil && found {
		return nonNegativeInt32(int32(parallelism)), true
	}
	if completions, found, err := unstructured.NestedInt64(owner.Object, "spec", "completions"); err == nil && found {
		return nonNegativeInt32(int32(completions)), true
	}
	return 0, false
}

func nonNegativeInt32(v int32) int32 {
	if v < 0 {
		return 0
	}
	return v
}

func parseAutoScalingAnnotations(pod *corev1.Pod, workloadProfile *tfv1.WorkloadProfile) {
	autoResources, ok := pod.Annotations[constants.AutoScaleResourcesAnnotation]
	if ok && autoResources == constants.TrueStringValue {
		if workloadProfile.Spec.AutoScalingConfig.AutoSetResources == nil {
			workloadProfile.Spec.AutoScalingConfig.AutoSetResources = &tfv1.AutoSetResources{}
		}
		workloadProfile.Spec.AutoScalingConfig.AutoSetResources.Enable = true

		targetResource, ok := pod.Annotations[constants.AutoScaleTargetResourceAnnotation]
		if ok {
			workloadProfile.Spec.AutoScalingConfig.AutoSetResources.TargetResource = tfv1.ScalingTargetResource(targetResource)
		} else {
			workloadProfile.Spec.AutoScalingConfig.AutoSetResources.TargetResource = tfv1.ScalingTargetResourceAll
		}
	}
}

// applyVerticalScalingRules applies pool-level vertical scaling rules from SchedulingConfigTemplate
// to the workload profile if the pod matches any rule's selector
func applyVerticalScalingRules(ctx context.Context, k8sClient client.Client, pod *corev1.Pod, pool *tfv1.GPUPool, workloadProfile *tfv1.WorkloadProfile) {
	if pool.Spec.SchedulingConfigTemplate == nil || *pool.Spec.SchedulingConfigTemplate == "" {
		return
	}

	schedulingConfigTemplate := &tfv1.SchedulingConfigTemplate{}
	if err := k8sClient.Get(ctx, client.ObjectKey{Name: *pool.Spec.SchedulingConfigTemplate}, schedulingConfigTemplate); err != nil {
		// If template not found, just skip
		return
	}

	// Check if pod matches any vertical scaling rule
	for _, rule := range schedulingConfigTemplate.Spec.VerticalScalingRules {
		if rule.Rule == nil {
			continue
		}

		selector, err := metav1.LabelSelectorAsSelector(&rule.Selector)
		if err != nil {
			continue
		}

		if selector.Matches(labels.Set(pod.Labels)) {
			// Merge the rule's AutoScalingConfig into workload profile
			mergeAutoScalingConfig(workloadProfile, rule.Rule)
			break // Apply first matching rule
		}
	}
}

// mergeAutoScalingConfig merges the rule's AutoScalingConfig into workload profile
func mergeAutoScalingConfig(workloadProfile *tfv1.WorkloadProfile, ruleConfig *tfv1.AutoScalingConfig) {
	if ruleConfig.AutoSetResources != nil {
		if workloadProfile.Spec.AutoScalingConfig.AutoSetResources == nil {
			workloadProfile.Spec.AutoScalingConfig.AutoSetResources = &tfv1.AutoSetResources{}
		}
		utils.MergeStructFields(workloadProfile.Spec.AutoScalingConfig.AutoSetResources, ruleConfig.AutoSetResources)
	}

	// Merge CronScalingRules
	if len(ruleConfig.CronScalingRules) > 0 {
		workloadProfile.Spec.AutoScalingConfig.CronScalingRules = append(workloadProfile.Spec.AutoScalingConfig.CronScalingRules, ruleConfig.CronScalingRules...)
	}

	// Merge ExternalScaler
	if ruleConfig.ExternalScaler != nil {
		workloadProfile.Spec.AutoScalingConfig.ExternalScaler = ruleConfig.ExternalScaler
	}
}

//nolint:gocyclo
func parseGPUResourcesAnnotations(pod *corev1.Pod, workloadProfile *tfv1.WorkloadProfile) error {
	// extract any containers has GPU count limits and set to annotation
	isMigratedFromContainerLimits := false
	gpuCount, hasValue := pod.Annotations[constants.GpuCountAnnotation]
	if hasValue {
		val, err := strconv.ParseInt(gpuCount, 10, 32)
		if err != nil {
			return fmt.Errorf("invalid gpuCount value: %w", err)
		}
		// Reject negative / zero / absurdly large values up front. A negative
		// value cast to uint32 silently becomes ~4G, a 0 is a no-op that
		// downstream code does not handle, and anything past a sane upper
		// bound is operator error.
		if val < 1 || val > 128 {
			return fmt.Errorf("invalid gpuCount value %d: must be in [1, 128]", val)
		}
		workloadProfile.Spec.GPUCount = uint32(val)
	} else if workloadProfile.Spec.GPUCount == 0 {
		// Map to track GPU count per container: containerName -> gpuCount
		containerGPUCounts := make(map[string]int)
		injectContainerNames := []string{}

		// Check all GPU resource names from all providers
		resourceNames := utils.GetGPUResourceNames()
		for _, container := range pod.Spec.Containers {
			for _, resourceName := range resourceNames {
				if quantity, ok := container.Resources.Limits[resourceName]; ok {
					gpuNumber, err := strconv.Atoi(quantity.String())
					if err != nil || gpuNumber <= 0 || gpuNumber > int(math.MaxUint32) {
						ctrl.Log.Error(err, "unrecognized GPU resource in limits, not a valid number", "pod", pod.Name, "container", container.Name, "resource", resourceName)
					} else {
						// Track GPU count per container
						containerGPUCounts[container.Name] = gpuNumber
						// Accumulate GPU count from all containers
						workloadProfile.Spec.GPUCount += uint32(gpuNumber)
						// For seamless migration with only one tensor-fusion.ai/enabled label
						// and one tensor-fusion.ai/vram-limit annotation, convert this to 100% computing-percent
						if !isMigratedFromContainerLimits {
							workloadProfile.Spec.Resources.Limits.ComputePercent = resource.MustParse("100")
							isMigratedFromContainerLimits = true
						}
						// Collect container names for inject annotation
						injectContainerNames = append(injectContainerNames, container.Name)
						// Set GPU vendor from resource name
						if vendor := getVendorFromResourceName(resourceName); vendor != "" && workloadProfile.Spec.GPUVendor == "" {
							workloadProfile.Spec.GPUVendor = vendor
						}
					}
				}
			}
		}

		// Convert limits containers to annotation for inject container when not specified
		if len(injectContainerNames) > 0 && pod.Annotations[constants.InjectContainerAnnotation] == "" {
			pod.Annotations[constants.InjectContainerAnnotation] = strings.Join(injectContainerNames, ",")
		}

		// Save per-container GPU count as annotation for later GPU allocation
		// Format: JSON key-value map, e.g., {"container1": 1, "container2": 2}
		if len(containerGPUCounts) > 0 {
			containerGPUCountJSON, err := json.Marshal(containerGPUCounts)
			if err != nil {
				return fmt.Errorf("failed to marshal container GPU counts: %w", err)
			}
			pod.Annotations[constants.ContainerGPUCountAnnotation] = string(containerGPUCountJSON)
		}
	}

	// Parse TFLOPs annotations first (higher priority than compute-percent)
	hasTflopsLimit := false
	if tflopsLimit, hasValue := parseResourceQuantity(pod, constants.TFLOPSLimitAnnotation); hasValue {
		workloadProfile.Spec.Resources.Limits.Tflops = tflopsLimit
		hasTflopsLimit = true
		// clean compute percent limit when tflops limit is set in annotation
		if isMigratedFromContainerLimits {
			workloadProfile.Spec.Resources.Limits.ComputePercent = resource.Quantity{}
		}
	}
	if vramLimit, hasValue := parseResourceQuantity(pod, constants.VRAMLimitAnnotation); hasValue {
		workloadProfile.Spec.Resources.Limits.Vram = vramLimit
	}

	hasTflopsRequest := false
	if tflopsRequest, hasValue := parseResourceQuantity(pod, constants.TFLOPSRequestAnnotation); hasValue {
		workloadProfile.Spec.Resources.Requests.Tflops = tflopsRequest
		hasTflopsRequest = true
		// TFLOPs has higher priority: clear compute percent request when tflops request is set
		workloadProfile.Spec.Resources.Requests.ComputePercent = resource.Quantity{}
		// If user explicitly uses TFLOPs for request but limit is still using compute-percent from migration,
		// clear the migrated compute-percent limit to maintain consistency (user should explicitly set limit)
		if isMigratedFromContainerLimits && !hasTflopsLimit {
			workloadProfile.Spec.Resources.Limits.ComputePercent = resource.Quantity{}
		}
	} else if workloadProfile.Spec.Resources.Requests.Tflops.IsZero() {
		workloadProfile.Spec.Resources.Requests.Tflops = workloadProfile.Spec.Resources.Limits.Tflops
		// If request inherits from limit and limit is TFLOPs, ensure compute percent is cleared
		if hasTflopsLimit {
			workloadProfile.Spec.Resources.Requests.ComputePercent = resource.Quantity{}
		}
	}
	if vramRequest, hasValue := parseResourceQuantity(pod, constants.VRAMRequestAnnotation); hasValue {
		workloadProfile.Spec.Resources.Requests.Vram = vramRequest
	} else if workloadProfile.Spec.Resources.Requests.Vram.IsZero() {
		workloadProfile.Spec.Resources.Requests.Vram = workloadProfile.Spec.Resources.Limits.Vram
	}

	// Percentage way to specify GPU resource request, not recommended, should use TFLOPs instead
	// Only process compute-percent if TFLOPs is not set (TFLOPs has higher priority)
	computeLimit, hasComputeLimit := parseResourceQuantity(pod, constants.ComputeLimitAnnotation)
	if hasComputeLimit {
		if hasTflopsLimit {
			// TFLOPs limit has higher priority, ignore compute-percent-limit
			workloadProfile.Spec.Resources.Limits.ComputePercent = resource.Quantity{}
		} else {
			workloadProfile.Spec.Resources.Limits.ComputePercent = computeLimit
		}
	}
	computeRequest, hasComputeRequest := parseResourceQuantity(pod, constants.ComputeRequestAnnotation)
	if hasComputeRequest {
		if hasTflopsRequest || hasTflopsLimit {
			// TFLOPs has higher priority, ignore compute-percent-request
			workloadProfile.Spec.Resources.Requests.ComputePercent = resource.Quantity{}
		} else {
			workloadProfile.Spec.Resources.Requests.ComputePercent = computeRequest
		}
	} else if workloadProfile.Spec.Resources.Requests.Tflops.IsZero() && workloadProfile.Spec.Resources.Requests.ComputePercent.IsZero() {
		// Only inherit compute percent if TFLOPs is not set
		if !hasTflopsLimit && !workloadProfile.Spec.Resources.Limits.ComputePercent.IsZero() {
			workloadProfile.Spec.Resources.Requests.ComputePercent = workloadProfile.Spec.Resources.Limits.ComputePercent
		}
	}

	// tflops - computePercent are mutually exclusive
	if !workloadProfile.Spec.Resources.Requests.Tflops.IsZero() && !workloadProfile.Spec.Resources.Requests.ComputePercent.IsZero() {
		return fmt.Errorf("tflops- and computePercent request are mutually exclusive, please specify only one")
	}
	if !workloadProfile.Spec.Resources.Limits.Tflops.IsZero() && !workloadProfile.Spec.Resources.Limits.ComputePercent.IsZero() {
		return fmt.Errorf("tflops- and computePercent limit are mutually exclusive, please specify only one")
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
	// Final guard: regardless of whether GPUCount was sourced from the
	// pod annotation, the WorkloadProfile spec, container limits, or the
	// gpu-indices annotation, the resulting value must fit in [1, 128].
	// Catches the case where a user creates a WorkloadProfile CR with
	// `spec.gpuCount: 0` or `200` and references it from a pod that has
	// no `tensor-fusion.ai/gpu-count` annotation of its own — the
	// annotation-only check above would never fire.
	if workloadProfile.Spec.GPUCount < 1 || workloadProfile.Spec.GPUCount > 128 {
		return fmt.Errorf("invalid gpuCount %d: must be in [1, 128]", workloadProfile.Spec.GPUCount)
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

// ParseContainerGPUCounts parses the container GPU count annotation (JSON format)
// Returns a map of container name to GPU count
func ParseContainerGPUCounts(pod *corev1.Pod) (map[string]int, error) {
	if pod.Annotations == nil {
		return nil, nil
	}

	annotationValue := pod.Annotations[constants.ContainerGPUCountAnnotation]
	if annotationValue == "" {
		return nil, nil
	}

	var containerGPUCounts map[string]int
	if err := json.Unmarshal([]byte(annotationValue), &containerGPUCounts); err != nil {
		return nil, fmt.Errorf("failed to parse container GPU count annotation: %w", err)
	}

	return containerGPUCounts, nil
}

// getVendorFromResourceName finds the vendor that owns a given resource name
func getVendorFromResourceName(resourceName corev1.ResourceName) string {
	mgr := provider.GetManager()
	if mgr == nil {
		return ""
	}

	return mgr.GetVendorFromResourceName(resourceName)
}
