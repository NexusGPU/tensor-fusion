/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"

	"gomodules.xyz/jsonpatch/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/pricing"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/indexallocator"
	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var httpClient = &http.Client{Timeout: 10 * time.Second}
var operatorPort = "8080"

func init() {
	if port := os.Getenv("PORT"); port != "" {
		operatorPort = port
	}
}

// SetupPodWebhookWithManager registers the webhook for Pod in the manager.
func SetupPodWebhookWithManager(mgr ctrl.Manager, portAllocator *portallocator.PortAllocator, indexAllocator *indexallocator.IndexAllocator, pricingProvider pricing.PricingProvider) error {
	webhookServer := mgr.GetWebhookServer()

	webhookServer.Register("/mutate-v1-pod",
		&admission.Webhook{
			Handler: &TensorFusionPodMutator{
				decoder:        admission.NewDecoder(mgr.GetScheme()),
				Client:         mgr.GetClient(),
				portAllocator:  portAllocator,
				indexAllocator: indexAllocator,
			},
		})
	return nil
}

type TensorFusionPodMutator struct {
	Client         client.Client
	decoder        admission.Decoder
	portAllocator  *portallocator.PortAllocator
	indexAllocator *indexallocator.IndexAllocator
}

// Handle implements admission.Handler interface.
//
//nolint:gocyclo
func (m *TensorFusionPodMutator) Handle(ctx context.Context, req admission.Request) admission.Response {
	pod := &corev1.Pod{}
	if err := m.decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}

	if len(pod.Namespace) == 0 {
		// Using req.Namespace, as pod.Namespace appears to be unset.
		pod.Namespace = req.Namespace
	}

	log := log.FromContext(ctx)
	log.Info("Mutating pod", "generateName", pod.GenerateName, "namespace", pod.Namespace)

	var currentBytes []byte

	// for non tensor fusion pod, check if there are any GPU resource request,
	// when there is, set scheduler to tensor-fusion-scheduler to trigger proxied scheduling
	// this is to ensure that non tensor fusion pod can be scheduled to nodes not conflict with tensor fusion
	if !utils.IsTensorFusionPod(pod) {
		hasGPUResource := utils.HasGPUResourceRequest(pod)
		if !hasGPUResource {
			return admission.Allowed("non tensor fusion pod nor GPU resource request, skipped")
		}

		shouldMigrate, err := ShouldAutoMigrateGPUPod(ctx, m.Client, pod)
		if err != nil {
			log.Error(err, "failed to check auto migration rules", "pod", pod.Name, "namespace", pod.Namespace)
			return admission.Allowed("non tensor fusion pod and invalid migration rule, skipped")
		}
		if shouldMigrate {
			// set original Pod bytes before mutate it, so that patch compare can be done correctly
			currentBytes, err = json.Marshal(pod)
			// Add tensor-fusion.ai/enabled label to mark pod for TensorFusion injection
			if pod.Labels == nil {
				pod.Labels = make(map[string]string)
			}
			if err != nil {
				return admission.Errored(http.StatusBadRequest, fmt.Errorf("failed to marshal current pod: %w", err))
			}
			pod.Labels[constants.TensorFusionEnabledLabelKey] = constants.TrueStringValue
		} else {
			if utils.IsProgressiveMigration() {
				return admission.Patched("set scheduler to tensor-fusion-scheduler", jsonpatch.JsonPatchOperation{
					Operation: "replace",
					Path:      "/spec/schedulerName",
					Value:     constants.SchedulerName,
				})
			}
			return admission.Allowed("GPU pod found, skip since NVIDIA_OPERATOR_PROGRESSIVE_MIGRATION not set")
		}
	} else {
		currentBytesOriginal, err := json.Marshal(pod)
		if err != nil {
			return admission.Errored(http.StatusBadRequest, fmt.Errorf("failed to marshal current pod: %w", err))
		}
		currentBytes = currentBytesOriginal
	}

	tfInfo, err := ParseTensorFusionInfo(ctx, m.Client, pod)
	if err != nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("parse tf resources: %w", err))
	}

	counter := &TensorFusionPodCounter{Client: m.Client}
	enabledReplicas := tfInfo.EnabledReplicas

	var podCounterAnnotationKey string
	if enabledReplicas != nil {
		// Get `tf-pod-count` by querying the owner's annotation
		// and then decide whether to patch the current pod
		podCount, podCounterKey, err := counter.Get(ctx, pod)
		if err != nil {
			return admission.Errored(http.StatusInternalServerError, fmt.Errorf("get tf pod count: %w", err))
		}
		if podCount >= *enabledReplicas {
			return admission.Allowed("tensor fusion pod count reached, keep original Pod for tensor fusion grey releasing")
		}
		podCounterAnnotationKey = podCounterKey
	}

	pool := &tfv1.GPUPool{}
	if err := m.Client.Get(ctx, client.ObjectKey{Name: tfInfo.Profile.PoolName}, pool); err != nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("gpu pool(%s) does not exist", tfInfo.Profile.PoolName))
	}
	tfInfo.Profile.Qos = calculateQoSLevel(tfInfo.Profile, pool)

	// Set GPU vendor from pool's DefaultVendor if not explicitly specified by user
	// This allows pods to inherit the vendor from their selected pool
	if tfInfo.Profile.GPUVendor == "" && pool.Spec.NodeManagerConfig != nil {
		tfInfo.Profile.GPUVendor = pool.Spec.NodeManagerConfig.DefaultVendor
		log.Info("Auto-detected GPU vendor from pool", "vendor", tfInfo.Profile.GPUVendor, "pool", pool.Name)
	}

	workload, err := m.createOrUpdateWorkload(ctx, pod, &tfInfo)
	if err != nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("create tf workload: %w", err))
	}

	// Pod mutating webhook can not get Pod UID,
	// thus need pod controller to set the controller reference
	if controllerRef := metav1.GetControllerOfNoCopy(workload); controllerRef == nil {
		pod.Annotations[constants.SetPendingOwnedWorkloadAnnotation] = tfInfo.WorkloadName
	}

	// Task 5: If workload already exists and has autoscaling enabled, set recommended annotations
	m.applyRecommendedAnnotations(pod, workload)

	// make sure required Pod info has been changed before generating patches
	if tfInfo.Profile.IsLocalGPU {
		// only patch scheduler when using local-gpu mode
		// for remote vGPU mode, start worker with tensor-fusion scheduler
		pod.Spec.SchedulerName = constants.SchedulerName
	}

	// find container index
	containerIndices := []int{}
	for _, name := range tfInfo.ContainerNames {
		for i := range pod.Spec.Containers {
			if pod.Spec.Containers[i].Name == name {
				containerIndices = append(containerIndices, i)
				break
			}
		}
	}

	if len(containerIndices) == 0 {
		return admission.Allowed("no valid container to inject tensor-fusion, skipped")
	}

	// Check if final profile is valid and contains valid GPU resource requests
	if tfInfo.Profile.Resources.Requests.Tflops.IsZero() &&
		tfInfo.Profile.Resources.Requests.ComputePercent.IsZero() &&
		tfInfo.Profile.Resources.Requests.Vram.IsZero() {
		return admission.Errored(http.StatusInternalServerError,
			fmt.Errorf("tflops request is not set, please set tensor-fusion.ai/tflops-request or/and tensor-fusion.ai/compute-percent-request"+
				" or/and tensor-fusion.ai/vram-request annotation on Pod"))
	}

	// Add defaults and tensor-fusion injection logic
	utils.AddOrOverrideTFClientMissingAnnotationsBeforePatch(pod, tfInfo)
	utils.AddTFDefaultClientConfBeforePatch(ctx, pod, pool, tfInfo, containerIndices)

	// Add priorityClass if contains higher QoS level and Pod priority class not specified
	if pod.Spec.PriorityClassName == "" &&
		(tfInfo.Profile.Qos == tfv1.QoSHigh || tfInfo.Profile.Qos == tfv1.QoSCritical) {
		pod.Spec.PriorityClassName = fmt.Sprintf("%s-%s",
			constants.TensorFusionSystemName, string(tfInfo.Profile.Qos))
		// Remove priority field if PriorityClassName is set, as Kubernetes Priority admission controller
		// will compute priority from PriorityClassName and doesn't allow both fields to be set
		pod.Spec.Priority = nil
	}

	// Inject initContainer and env variables
	patches, err := m.patchTFClient(
		ctx, pod, pool, tfInfo.Profile.IsLocalGPU, currentBytes, containerIndices, tfInfo.Profile.SidecarWorker,
	)
	if err != nil {
		log.Error(err, "failed to patch tf client", "pod", req.Name, "namespace", req.Namespace)
		return admission.Errored(http.StatusInternalServerError, err)
	}

	if podCounterAnnotationKey != "" {
		if err := counter.Increase(ctx, pod); err != nil {
			return admission.Errored(http.StatusInternalServerError, fmt.Errorf("increase tf pod count: %w", err))
		}
		// Patch annotation for pod counter
		patch := jsonpatch.JsonPatchOperation{
			Operation: "add",
			Path:      "/metadata/annotations/" + utils.EscapeJSONPointer(constants.TensorFusionPodCounterKeyAnnotation),
			Value:     podCounterAnnotationKey,
		}
		patches = append(patches, patch)
	}

	return admission.Patched("tensor fusion component patched", patches...)
}

// InjectDecoder injects the decoder.
func (m *TensorFusionPodMutator) InjectDecoder(d admission.Decoder) error {
	m.decoder = d
	return nil
}

func (m *TensorFusionPodMutator) createOrUpdateWorkload(
	ctx context.Context,
	pod *corev1.Pod,
	tfInfo *utils.TensorFusionInfo,
) (*tfv1.TensorFusionWorkload, error) {
	// Create the desired spec for comparison
	desiredSpec := *tfInfo.Profile.DeepCopy()

	workload := &tfv1.TensorFusionWorkload{}
	err := m.Client.Get(ctx, client.ObjectKey{Name: tfInfo.WorkloadName, Namespace: pod.Namespace}, workload)
	if err != nil {
		if !errors.IsNotFound(err) {
			return nil, fmt.Errorf("failed to get workload: %w", err)
		}

		// Create a new workload
		workload = &tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      tfInfo.WorkloadName,
				Namespace: pod.Namespace,
				Labels: map[string]string{
					constants.GpuPoolKey: tfInfo.Profile.PoolName,
				},
				Annotations: map[string]string{
					constants.WorkloadModeAnnotation: constants.WorkloadModeDynamic,
				},
			},
			Spec: desiredSpec,
		}

		// Pass through disable features annotation
		if pod.Labels[constants.DisableFeaturesAnnotation] != "" {
			workload.Annotations[constants.DisableFeaturesAnnotation] = pod.Labels[constants.DisableFeaturesAnnotation]
		}

		if tfInfo.PodControllerRef != nil {
			workload.OwnerReferences = []metav1.OwnerReference{*tfInfo.PodControllerRef}
		}

		if err := m.Client.Create(ctx, workload); err != nil {
			return nil, fmt.Errorf("failed to create workload: %w", err)
		}
		return workload, nil
	}

	if !equality.Semantic.DeepEqual(workload.Spec, desiredSpec) {
		patch := client.MergeFrom(workload.DeepCopy())
		workload.Spec = desiredSpec
		if err := m.Client.Patch(ctx, workload, patch); err != nil {
			return nil, fmt.Errorf("failed to patch workload: %w", err)
		}
	}
	return workload, nil
}

// applyRecommendedAnnotations applies recommended resource annotations to the pod
// if the workload already exists and has autoscaling enabled with a recommendation
func (m *TensorFusionPodMutator) applyRecommendedAnnotations(
	pod *corev1.Pod,
	workload *tfv1.TensorFusionWorkload,
) {
	// Only apply if autoscaling is enabled
	asr := workload.Spec.AutoScalingConfig.AutoSetResources
	if asr == nil || !asr.Enable {
		return
	}

	// Only apply if there's a recommendation
	if workload.Status.Recommendation == nil {
		return
	}

	recommendation := workload.Status.Recommendation

	// Set recommended annotations similar to VPA logic
	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}

	// Apply compute (TFlops) recommendations if target includes compute
	targetResource := asr.TargetResource
	if targetResource == "" || targetResource == tfv1.ScalingTargetResourceAll || targetResource == tfv1.ScalingTargetResourceCompute {
		if !recommendation.Requests.Tflops.IsZero() {
			pod.Annotations[constants.TFLOPSRequestAnnotation] = recommendation.Requests.Tflops.String()
		}
		if !recommendation.Limits.Tflops.IsZero() {
			pod.Annotations[constants.TFLOPSLimitAnnotation] = recommendation.Limits.Tflops.String()
		}
	}

	// Apply VRAM recommendations if target includes vram
	if targetResource == "" || targetResource == tfv1.ScalingTargetResourceAll || targetResource == tfv1.ScalingTargetResourceVRAM {
		if !recommendation.Requests.Vram.IsZero() {
			pod.Annotations[constants.VRAMRequestAnnotation] = recommendation.Requests.Vram.String()
		}
		if !recommendation.Limits.Vram.IsZero() {
			pod.Annotations[constants.VRAMLimitAnnotation] = recommendation.Limits.Vram.String()
		}
	}
}

func (m *TensorFusionPodMutator) patchTFClient(
	ctx context.Context,
	pod *corev1.Pod,
	pool *tfv1.GPUPool,
	isLocalGPU bool,
	currentBytes []byte,
	containerIndices []int,
	isSidecarWorker bool,
) ([]jsonpatch.JsonPatchOperation, error) {
	clientConfig := pool.Spec.ComponentConfig.Client

	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	pod.Labels[constants.LabelKeyPodTemplateHash] = utils.GetObjectHash(clientConfig)

	assignPodLabelsAndAnnotations(isLocalGPU, pod, pool)

	// Index allocation only for worker pods
	// Index is used for Device Plugin communication to match Pod with CDI device
	var index int
	if pod.Labels[constants.LabelComponent] == constants.ComponentWorker {
		// Assign index once per pod (before processing containers)
		// Index must be assigned in webhook stage since scheduler cannot modify Pod
		// This is a special index resource (1-32), not a real device resource
		// Index is assigned in ascending order (1, 2, 3, ...) via distributed lock (leader election)
		index = m.assignDeviceAllocationIndex(ctx, pod)
	}
	// clean annotation if exists, must be assigned by scheduler to ensure lock of certain index on one node
	delete(pod.Annotations, constants.PodIndexAnnotation)

	for _, containerIndex := range containerIndices {
		container := &pod.Spec.Containers[containerIndex]
		containerJSON, err := json.Marshal(container)
		if err != nil {
			return nil, fmt.Errorf("marshal container: %w", err)
		}

		var patchJSON []byte
		patchJSON, err = serializeContainerInjectionPatchJson(clientConfig, patchJSON, isLocalGPU)
		if err != nil {
			return nil, err
		}

		patchedJSON, err := strategicpatch.StrategicMergePatch(containerJSON, patchJSON, corev1.Container{})
		if err != nil {
			return nil, fmt.Errorf("apply strategic merge patch to container: %w", err)
		}

		// validate if container decoded successfully after merge patch
		container = &corev1.Container{}
		if err := json.Unmarshal(patchedJSON, container); err != nil {
			return nil, fmt.Errorf("unmarshal patched container, invalid container patch: %w", err)
		}

		removeNativeGPULimits(container)

		// Inject tensor-fusion.ai/index resource for Device Plugin communication (worker pods only)
		// This is a special index resource (not a real device), used for Pod-to-DevicePlugin communication
		if pod.Labels[constants.LabelComponent] == constants.ComponentWorker {
			if container.Resources.Limits == nil {
				container.Resources.Limits = make(corev1.ResourceList)
			}
			// Limit is set to actual index value (1-128) for Device Plugin to match Pod
			// ResourceFit of dummy device already ignored in TF scheduler
			indexQuantity := resource.MustParse(strconv.Itoa((index % constants.IndexModLength) + 1))
			indexKey := fmt.Sprintf("%s%s%x", constants.PodIndexAnnotation, constants.PodIndexDelimiter, index/constants.IndexModLength)
			container.Resources.Limits[corev1.ResourceName(indexKey)] = indexQuantity
		}

		if !isLocalGPU {
			addConnectionForRemoteFixedReplicaVirtualGPU(pod, container, clientConfig)
		} else if isSidecarWorker {
			// Hard-isolation mode in container, use tensor-fusion worker as sidecar and communicate thru /dev/shm/tf_shm
			container.Env = append(container.Env, corev1.EnvVar{
				Name: constants.ConnectionInfoEnv,
				// protocol+identifier+size+initVersion
				Value: fmt.Sprintf("shmem+%s+%s+1",
					constants.ConnectionSharedMemName, constants.ConnectionSharedMemSize),
			}, corev1.EnvVar{
				Name:  constants.DisableVMSharedMemEnv,
				Value: "0",
			})
		}

		pod.Spec.Containers[containerIndex] = *container
	}

	// Patch hostPort allocation
	if pod.Labels[constants.GenHostPortLabel] == constants.GenHostPortLabelValue {
		// TODO/FIXME potential bug, when it's deployment created Pod rather than standalone Pod, pod.Name is empty
		if err := m.generateHostPort(pod, pod.Labels[constants.GenHostPortNameLabel]); err != nil {
			return nil, fmt.Errorf("can not generate host port: %w", err)
		}
	}

	containerPatchedJSON, err := json.Marshal(pod)
	if err != nil {
		return nil, fmt.Errorf("marshal current pod: %w", err)
	}
	patches, err := jsonpatch.CreatePatch(currentBytes, containerPatchedJSON)
	if err != nil {
		return nil, fmt.Errorf("patch to container: %w", err)
	}

	// Additional pod level patch
	strategicpatches, err := calculatePodPatch(currentBytes, pod, clientConfig, isLocalGPU)
	if err != nil {
		return nil, fmt.Errorf("calculate pod patch: %w", err)
	}
	patches = append(patches, strategicpatches...)
	return patches, nil
}

//nolint:unused
func (m *TensorFusionPodMutator) assignDeviceAllocationIndex(ctx context.Context, pod *corev1.Pod) int {
	var index int
	var indexErr error
	podIdentifier := pod.Name
	if podIdentifier == "" {
		// For Deployment/StatefulSet created pods, Name might be empty, use GenerateName + UID(maybe empty)
		podIdentifier = pod.GenerateName + string(pod.UID)
	}

	if m.indexAllocator != nil && m.indexAllocator.IsLeader {
		index, indexErr = m.indexAllocator.AssignIndex(podIdentifier)
		if indexErr != nil {
			log := log.FromContext(ctx)
			log.Error(indexErr, "failed to assign index for pod", "pod", podIdentifier)
			index = 0
		}
	} else if m.indexAllocator != nil && !m.indexAllocator.IsLeader {
		// If not leader, get index from leader via HTTP API (similar to port allocation)
		// This ensures global increment across distributed webhook instances
		index, indexErr = m.assignIndexFromLeader(ctx, pod)
		if indexErr != nil {
			log := log.FromContext(ctx)
			log.Error(indexErr, "failed to assign index from leader", "pod", podIdentifier)
			index = 0
		}
	} else {
		// No allocator available, use 0 as fallback
		index = 0
	}
	return index
}

// Convert the strategic merge patch to JSON
func calculatePodPatch(currentBytes []byte, pod *corev1.Pod, clientConfig *tfv1.ClientConfig, isLocalGPU bool) ([]jsonpatch.JsonPatchOperation, error) {
	var patchBytes []byte
	var err error
	if !isLocalGPU {
		patchBytes, err = json.Marshal(clientConfig.PatchToPod)
	}
	if err != nil {
		return nil, fmt.Errorf("marshal patch: %w", err)
	}

	// Apply the strategic merge patch
	resultBytes, err := strategicpatch.StrategicMergePatch(currentBytes, patchBytes, corev1.Pod{})
	if err != nil {
		return nil, fmt.Errorf("apply strategic merge patch: %w", err)
	}
	// Generate JSON patch operations by comparing original and patched pod
	strategicpatches, err := jsonpatch.CreatePatch(currentBytes, resultBytes)
	if err != nil {
		return nil, fmt.Errorf("create json patch: %w", err)
	}
	// Unmarshal the result back into the pod
	if err := json.Unmarshal(resultBytes, pod); err != nil {
		return nil, fmt.Errorf("unmarshal patched pod: %w", err)
	}
	return strategicpatches, nil
}

func assignPodLabelsAndAnnotations(isLocalGPU bool, pod *corev1.Pod, pool *tfv1.GPUPool) {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	if isLocalGPU {
		pod.Labels[constants.LabelComponent] = constants.ComponentWorker
		pod.Annotations[constants.EmbeddedWorkerAnnotation] = constants.TrueStringValue
		// no need to add port in local gpu mode, communication is done through shared memory in the same process

		// Add toleration for TensorFusion nodes
		pod.Spec.Tolerations = append(pod.Spec.Tolerations, corev1.Toleration{
			Key:      constants.NodeUsedByTaintKey,
			Operator: corev1.TolerationOpEqual,
			Value:    constants.TensorFusionSystemName,
			Effect:   corev1.TaintEffectPreferNoSchedule,
		})
	} else {
		pod.Labels[constants.LabelComponent] = constants.ComponentClient
	}
	pod.Labels[constants.GpuPoolKey] = pool.Name
}

func addConnectionForRemoteFixedReplicaVirtualGPU(pod *corev1.Pod, container *corev1.Container, clientConfig *tfv1.ClientConfig) {
	var prefix string
	if pod.GenerateName == "" && pod.Name != "" {
		prefix = pod.Name + constants.TFConnectionNamePrefix
	} else {
		prefix = pod.GenerateName + constants.TFConnectionNameNoPrefix
	}
	connectionName := fmt.Sprintf("%s%s", prefix, utils.NewShortID(10))
	connectionNamespace := pod.Namespace

	// metadata TF_POD_NAME and TF_CONNECTION_NAMESPACE
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  constants.ConnectionNameEnv,
		Value: connectionName,
	})
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  constants.ConnectionNamespaceEnv,
		Value: connectionNamespace,
	})
	// operator k8s serviceURL ? namespace
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  constants.GetConnectionURLEnv,
		Value: fmt.Sprintf("%s/api/connection?name=%s&namespace=%s", clientConfig.OperatorEndpoint, connectionName, connectionNamespace),
	})
}

func removeNativeGPULimits(container *corev1.Container) {
	if container.Resources.Requests != nil {
		delete(container.Resources.Requests, constants.NvidiaGPUKey)
		delete(container.Resources.Requests, constants.AmdGPUKey)
	}
	if container.Resources.Limits != nil {
		delete(container.Resources.Limits, constants.NvidiaGPUKey)
		delete(container.Resources.Limits, constants.AmdGPUKey)
	}
}

func serializeContainerInjectionPatchJson(clientConfig *tfv1.ClientConfig, patchJSON []byte, isLocalGPU bool) ([]byte, error) {
	var err error
	if !isLocalGPU && clientConfig.PatchToContainer != nil {
		patchJSON, err = json.Marshal(clientConfig.PatchToContainer)
		if err != nil {
			return nil, fmt.Errorf("marshal patchToContainer: %w", err)
		}
	}
	return patchJSON, nil
}

func (m *TensorFusionPodMutator) generateHostPort(pod *corev1.Pod, portName string) error {

	portNameFound := false
	containerIndex := -1
	portIndex := -1
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		for j := range container.Ports {
			port := &container.Ports[j]
			if port.Name == portName {
				portNameFound = true
				containerIndex = i
				portIndex = j
			}
		}
	}
	if !portNameFound {
		return fmt.Errorf("port name %s not found, can not assign host port for pod %s", portName, pod.Name)
	}

	if !m.portAllocator.IsLeader {
		port, err := m.assignClusterHostPortFromLeader(pod)
		if err != nil {
			return fmt.Errorf("can not assign cluster host port from leader: %w", err)
		}
		pod.Annotations[constants.GenPortNumberAnnotation] = strconv.Itoa(port)
	} else {
		port, err := m.portAllocator.AssignClusterLevelHostPort(pod.Name)
		if err != nil {
			return fmt.Errorf("can not assign cluster level host port: %w", err)
		}
		pod.Annotations[constants.GenPortNumberAnnotation] = strconv.Itoa(port)
	}

	pod.Spec.Containers[containerIndex].Ports[portIndex].HostPort = int32(m.getPortNumber(pod))
	return nil
}

func (m *TensorFusionPodMutator) getPortNumber(pod *corev1.Pod) int {
	portNumber, _ := strconv.Atoi(pod.Annotations[constants.GenPortNumberAnnotation])
	return portNumber
}

func (m *TensorFusionPodMutator) assignClusterHostPortFromLeader(pod *corev1.Pod) (int, error) {
	leaderIP := utils.GetLeaderIP(m.Client)
	if leaderIP == "" {
		return 0, fmt.Errorf("operator leader IP not found")
	}

	urlStr := fmt.Sprintf("http://%s:%s/api/assign-host-port?podName=%s", leaderIP, operatorPort, pod.Name)
	req, err := http.NewRequest("GET", urlStr, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set(constants.AuthorizationHeader, "Bearer "+utils.ReadServiceAccountToken())
	resp, err := httpClient.Do(req)
	if err != nil {
		return 0, fmt.Errorf("failed to assign host port: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("host port allocation failed: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("failed to read allocation response: %w", err)
	}

	return strconv.Atoi(string(body))
}

//nolint:unused
func (m *TensorFusionPodMutator) assignIndexFromLeader(ctx context.Context, pod *corev1.Pod) (int, error) {
	leaderIP := utils.GetLeaderIP(m.Client)
	if leaderIP == "" {
		return 0, fmt.Errorf("operator leader IP not found")
	}

	podIdentifier := pod.Name
	if podIdentifier == "" {
		podIdentifier = pod.GenerateName + string(pod.UID)
	}
	urlStr := fmt.Sprintf("http://%s:%s/api/assign-index?podName=%s", leaderIP, operatorPort, podIdentifier)
	req, err := http.NewRequestWithContext(ctx, "GET", urlStr, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set(constants.AuthorizationHeader, "Bearer "+utils.ReadServiceAccountToken())
	resp, err := httpClient.Do(req)
	if err != nil {
		return 0, fmt.Errorf("failed to assign index: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("index allocation failed: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("failed to read allocation response: %w", err)
	}

	return strconv.Atoi(string(body))
}

func calculateQoSLevel(profile *tfv1.WorkloadProfileSpec, pool *tfv1.GPUPool) tfv1.QoSLevel {
	// when not set, assign default QoS
	if profile.Qos == "" {
		sameReqLimits := profile.Resources.Limits.Tflops.Cmp(profile.Resources.Requests.Tflops) == 0 &&
			profile.Resources.Limits.Vram.Cmp(profile.Resources.Requests.Vram) == 0

		// set to high if req == limits, same logic as Kubernetes QoS
		// critical QoS can preempt other pods, have to be set manually
		if sameReqLimits {
			return constants.QoSLevelHigh
		}

		if pool.Spec.QosConfig == nil || pool.Spec.QosConfig.DefaultQoS == "" {
			return constants.QoSLevelMedium
		}
		return pool.Spec.QosConfig.DefaultQoS
	}
	return profile.Qos
}
