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
	"strconv"
	"time"

	"gomodules.xyz/jsonpatch/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/pricing"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var httpClient = &http.Client{Timeout: 10 * time.Second}

// SetupPodWebhookWithManager registers the webhook for Pod in the manager.
func SetupPodWebhookWithManager(mgr ctrl.Manager, portAllocator *portallocator.PortAllocator, pricingProvider pricing.PricingProvider) error {
	webhookServer := mgr.GetWebhookServer()

	webhookServer.Register("/mutate-v1-pod",
		&admission.Webhook{
			Handler: &TensorFusionPodMutator{
				decoder:       admission.NewDecoder(mgr.GetScheme()),
				Client:        mgr.GetClient(),
				portAllocator: portAllocator,
			},
		})
	return nil
}

type TensorFusionPodMutator struct {
	Client        client.Client
	decoder       admission.Decoder
	portAllocator *portallocator.PortAllocator
}

// Handle implements admission.Handler interface.
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

	// for non tensor fusion pod, check if there are any GPU resource request,
	// when there is, set scheduler to tensor-fusion-scheduler to trigger proxied scheduling
	// this is to ensure that non tensor fusion pod can be scheduled to nodes not conflict with tensor fusion
	if !utils.IsTensorFusionPod(pod) {
		if utils.IsProgressiveMigration() && utils.HasGPUResourceRequest(pod) {
			return admission.Patched("set scheduler to tensor-fusion-scheduler", jsonpatch.JsonPatchOperation{
				Operation: "replace",
				Path:      "/spec/schedulerName",
				Value:     constants.SchedulerName,
			})
		}
		return admission.Allowed("non tensor fusion pod nor GPU resource request, skipped")
	}

	currentBytes, err := json.Marshal(pod)
	if err != nil {
		return admission.Errored(http.StatusBadRequest, fmt.Errorf("failed to marshal current pod: %w", err))
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

	if workload, err := m.createOrUpdateWorkload(ctx, pod, &tfInfo, pool); err != nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("create tf workload: %w", err))
	} else {
		// Pod mutating webhook can not get Pod UID,
		// thus need pod controller to set the controller reference
		if controllerRef := metav1.GetControllerOfNoCopy(workload); controllerRef == nil {
			pod.Annotations[constants.SetPendingOwnedWorkloadAnnotation] = tfInfo.WorkloadName
		}
	}

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

	// Add defaults and tensor-fusion injection logic
	utils.AddOrOverrideTFClientMissingAnnotationsBeforePatch(pod, tfInfo)
	utils.AddTFDefaultClientConfBeforePatch(ctx, pod, pool, tfInfo, containerIndices)

	// Add priorityClass if contains higher QoS level and Pod priority class not specified
	if pod.Spec.PriorityClassName == "" &&
		(tfInfo.Profile.Qos == tfv1.QoSHigh || tfInfo.Profile.Qos == tfv1.QoSCritical) {
		pod.Spec.PriorityClassName = constants.TensorFusionSystemName + string(tfInfo.Profile.Qos)
	}

	// Inject initContainer and env variables
	patches, err := m.patchTFClient(
		pod, pool, tfInfo.Profile.IsLocalGPU, currentBytes, containerIndices, tfInfo.Profile.SidecarWorker,
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
	pool *tfv1.GPUPool) (*tfv1.TensorFusionWorkload, error) {
	// Create the desired spec for comparison
	desiredSpec := tfv1.WorkloadProfileSpec{
		Replicas:          nil,
		PoolName:          tfInfo.Profile.PoolName,
		Resources:         tfInfo.Profile.Resources,
		Qos:               calculateQoSLevel(tfInfo.Profile, pool),
		IsLocalGPU:        tfInfo.Profile.IsLocalGPU,
		GPUCount:          tfInfo.Profile.GPUCount,
		GPUModel:          tfInfo.Profile.GPUModel,
		AutoScalingConfig: tfInfo.Profile.AutoScalingConfig,
	}

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

func (m *TensorFusionPodMutator) patchTFClient(
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

		removeNativeGPULimitsAndAddCountToAnnotation(pod, container)

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

// Convert the strategic merge patch to JSON
func calculatePodPatch(currentBytes []byte, pod *corev1.Pod, clientConfig *tfv1.ClientConfig, isLocalGPU bool) ([]jsonpatch.JsonPatchOperation, error) {
	var patchBytes []byte
	var err error
	if isLocalGPU {
		patchBytes, err = json.Marshal(clientConfig.PatchEmbeddedWorkerToPod)
	} else {
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

// remove nvidia.com/gpu in resources, add the GPU number into annotation
func removeNativeGPULimitsAndAddCountToAnnotation(pod *corev1.Pod, container *corev1.Container) {
	if container.Resources.Requests != nil {
		delete(container.Resources.Requests, constants.NvidiaGPUKey)
	}
	if container.Resources.Limits != nil {
		if _, ok := container.Resources.Limits[constants.NvidiaGPUKey]; ok {
			// parse the gpu number to annotation
			quantity := container.Resources.Limits[constants.NvidiaGPUKey]
			gpuNumber, err := strconv.Atoi(quantity.String())
			if err != nil || gpuNumber <= 0 {
				ctrl.Log.Error(err, "unrecognized nvidia.com/gpu in resources, not a valid number", "pod", pod.Name, "container", container.Name)
			} else {
				pod.Annotations[constants.GpuCountAnnotation] = strconv.Itoa(gpuNumber)
			}
			delete(container.Resources.Limits, constants.NvidiaGPUKey)
		}
	}
}

func serializeContainerInjectionPatchJson(clientConfig *tfv1.ClientConfig, patchJSON []byte, isLocalGPU bool) ([]byte, error) {
	var err error
	if !isLocalGPU && clientConfig.PatchToContainer != nil {
		patchJSON, err = json.Marshal(clientConfig.PatchToContainer)
		if err != nil {
			return nil, fmt.Errorf("marshal patchToContainer: %w", err)
		}
	} else if isLocalGPU && clientConfig.PatchToEmbeddedWorkerContainer != nil {
		patchJSON, err = json.Marshal(clientConfig.PatchToEmbeddedWorkerContainer)
		if err != nil {
			return nil, fmt.Errorf("marshal patchToEmbeddedWorkerContainer: %w", err)
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

	leaderIP := m.portAllocator.GetLeaderIP()
	if leaderIP == "" {
		return 0, fmt.Errorf("operator leader IP not found")
	}

	urlStr := fmt.Sprintf("http://%s:8080/assign-host-port?podName=%s", leaderIP, pod.Name)
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
