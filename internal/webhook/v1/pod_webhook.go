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
	goErrors "errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"gomodules.xyz/jsonpatch/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	"al.essio.dev/pkg/shellescape"
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/internal/worker"
	"github.com/lithammer/shortuuid/v4"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var httpClient = &http.Client{Timeout: 10 * time.Second}

// SetupPodWebhookWithManager registers the webhook for Pod in the manager.
func SetupPodWebhookWithManager(mgr ctrl.Manager, portAllocator *portallocator.PortAllocator) error {
	webhookServer := mgr.GetWebhookServer()

	webhookServer.Register("/mutate-v1-pod",
		&admission.Webhook{
			Handler: &TensorFusionPodMutator{
				decoder:       admission.NewDecoder(runtime.NewScheme()),
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
			return admission.Allowed("tf pod count exceeds enabled replicas")
		}
		podCounterAnnotationKey = podCounterKey
	}

	pool := &tfv1.GPUPool{}
	if err := m.Client.Get(ctx, client.ObjectKey{Name: tfInfo.Profile.PoolName}, pool); err != nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("gpu pool(%s) does not exist", tfInfo.Profile.PoolName))
	}

	workload := &tfv1.TensorFusionWorkload{}
	if tfInfo.GenWorkload {
		if err := m.createOrUpdateWorkload(ctx, pod, &tfInfo, workload, pool); err != nil {
			return admission.Errored(http.StatusInternalServerError, fmt.Errorf("create tf workload: %w", err))
		}
	}

	var nodeSelector map[string]string
	if tfInfo.Profile.IsLocalGPU {
		if !tfInfo.GenWorkload {
			if err := m.Client.Get(ctx, client.ObjectKey{Name: tfInfo.WorkloadName, Namespace: pod.Namespace}, workload); err != nil {
				return admission.Errored(http.StatusInternalServerError, fmt.Errorf("workload(%s) does not exist", tfInfo.WorkloadName))
			}
		}

		workerFound := false
		for i := 0; i < 25; i++ {
			workloadStatus, err := worker.SelectWorker(ctx, m.Client, workload, 1)
			if err != nil {
				if goErrors.Is(err, worker.ErrNoAvailableWorker) {
					time.Sleep(time.Second)
					continue
				}
				log.Error(err, "failed to select worker for pod", "pod", req.Name, "namespace", req.Namespace)
				return admission.Errored(http.StatusInternalServerError, fmt.Errorf("select worker: %w", err))
			}
			nodeSelector = workloadStatus.NodeSelector
			workerFound = true
			break
		}

		if !workerFound {
			return admission.Errored(http.StatusInternalServerError, fmt.Errorf("no available worker for pod: %s", req.Name))
		}
	}

	// Inject initContainer and env variables
	patches, err := m.patchTFClient(pod, pool, tfInfo.ContainerNames, nodeSelector)
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

func (m *TensorFusionPodMutator) createOrUpdateWorkload(ctx context.Context, pod *corev1.Pod, tfInfo *TensorFusionInfo, workload *tfv1.TensorFusionWorkload, pool *tfv1.GPUPool) error {
	// Check if workload exists
	err := m.Client.Get(ctx, client.ObjectKey{Name: tfInfo.WorkloadName, Namespace: pod.Namespace}, workload)

	qos := calculateQoSLevel(tfInfo.Profile, pool)

	if err != nil {
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to get workload: %w", err)
		}
		// find root owner references of pod
		rootOwnerRef, err := utils.FindRootOwnerReference(ctx, m.Client, pod.Namespace, pod)
		if err != nil {
			return fmt.Errorf("failed to find root owner reference: %w", err)
		}

		// Create a new workload
		replicas := tfInfo.Replicas
		workload = &tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      tfInfo.WorkloadName,
				Namespace: pod.Namespace,
				Labels: map[string]string{
					constants.GpuPoolKey: tfInfo.Profile.PoolName,
				},
			},
			Spec: tfv1.WorkloadProfileSpec{
				Replicas:   &replicas,
				PoolName:   tfInfo.Profile.PoolName,
				Resources:  tfInfo.Profile.Resources,
				GPUCount:   tfInfo.Profile.GPUCount,
				Qos:        qos,
				GPUModel:   tfInfo.Profile.GPUModel,
				IsLocalGPU: tfInfo.Profile.IsLocalGPU,
			},
		}

		if rootOwnerRef != nil {
			workload.OwnerReferences = []metav1.OwnerReference{*rootOwnerRef}
		}

		if err := m.Client.Create(ctx, workload); err != nil {
			return fmt.Errorf("failed to create workload: %w", err)
		}
		return nil
	}

	// Create the desired spec for comparison
	replicas := tfInfo.Replicas
	desiredSpec := tfv1.WorkloadProfileSpec{
		Replicas:   &replicas,
		PoolName:   tfInfo.Profile.PoolName,
		Resources:  tfInfo.Profile.Resources,
		Qos:        qos,
		IsLocalGPU: tfInfo.Profile.IsLocalGPU,
		GPUCount:   tfInfo.Profile.GPUCount,
		GPUModel:   tfInfo.Profile.GPUModel,
	}

	// Compare the entire spec at once
	if !equality.Semantic.DeepEqual(workload.Spec, desiredSpec) {
		workload.Spec = desiredSpec
		// TODO retry on conflict
		if err := m.Client.Update(ctx, workload); err != nil {
			return fmt.Errorf("failed to update workload: %w", err)
		}
	}
	return nil
}

func (m *TensorFusionPodMutator) patchTFClient(
	pod *corev1.Pod,
	pool *tfv1.GPUPool,
	containerNames []string,
	nodeSelector map[string]string,
) ([]jsonpatch.JsonPatchOperation, error) {
	// Convert the current pod to JSON
	currentBytes, err := json.Marshal(pod)
	if err != nil {
		return nil, fmt.Errorf("marshal current pod: %w", err)
	}

	if nodeSelector != nil {
		// Local GPU Mode
		if pod.Spec.Affinity == nil {
			pod.Spec.Affinity = &corev1.Affinity{}
		}
		pod.Spec.Affinity.NodeAffinity = &corev1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
				NodeSelectorTerms: []corev1.NodeSelectorTerm{
					{
						MatchExpressions: []corev1.NodeSelectorRequirement{
							{
								Key:      "kubernetes.io/hostname",
								Operator: corev1.NodeSelectorOpIn,
								Values:   []string{nodeSelector["kubernetes.io/hostname"]},
							},
						},
					},
				},
			},
		}
	}

	clientConfig := pool.Spec.ComponentConfig.Client

	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	pod.Labels[constants.LabelKeyPodTemplateHash] = utils.GetObjectHash(clientConfig)
	pod.Labels[constants.LabelComponent] = constants.ComponentClient
	pod.Labels[constants.GpuPoolKey] = pool.Name

	// Patch hostPort allocation
	if pod.Labels[constants.GenHostPortLabel] == constants.GenHostPortLabelValue {
		if err := m.generateHostPort(pod, pod.Labels[constants.GenHostPortNameLabel]); err != nil {
			return nil, fmt.Errorf("can not generate host port: %w", err)
		}
	}

	containerPatched := false
	// Patch to Container
	for _, name := range containerNames {
		for i := range pod.Spec.Containers {
			container := &pod.Spec.Containers[i]
			if container.Name == name {
				// fix for issue https://github.com/NexusGPU/tensor-fusion/issues/164:
				//	transform bash/zsh -c commands
				if len(container.Command) >= 3 {
					shell := container.Command[0]
					if (shell == "bash" || shell == "zsh") && container.Command[1] == "-c" {
						originalCommand := container.Command[2]
						comment := "# [TensorFusion Patch] This command is wrapped by sh -c to improve compatibility with certain container environments."
						safeCommand := shellescape.Quote(fmt.Sprintf("%s\n%s", comment, originalCommand))
						container.Command = []string{"sh", "-c", fmt.Sprintf("%s -c %s", shell, safeCommand)}
					}
				}
				// patch from config
				containerJSON, err := json.Marshal(container)
				if err != nil {
					return nil, fmt.Errorf("marshal container: %w", err)
				}
				patchJSON, err := json.Marshal(clientConfig.PatchToContainer)
				if err != nil {
					return nil, fmt.Errorf("marshal patchToContainer: %w", err)
				}

				patchedJSON, err := strategicpatch.StrategicMergePatch(containerJSON, patchJSON, corev1.Container{})
				if err != nil {
					return nil, fmt.Errorf("apply strategic merge patch to container: %w", err)
				}
				container = &corev1.Container{}
				if err := json.Unmarshal(patchedJSON, container); err != nil {
					return nil, fmt.Errorf("unmarshal patched container: %w", err)
				}

				// remove nvidia.com/gpu in resources
				if container.Resources.Requests != nil {
					delete(container.Resources.Requests, constants.NvidiaGPUKey)
				}
				if container.Resources.Limits != nil {
					delete(container.Resources.Limits, constants.NvidiaGPUKey)
				}

				// add connection env
				connectionName := fmt.Sprintf("%s%s", pod.GenerateName, shortuuid.NewWithAlphabet("123456789abcdefghijkmnopqrstuvwxy"))
				connectionNamespace := pod.Namespace

				container.Env = append(container.Env, corev1.EnvVar{
					Name:  constants.ConnectionNameEnv,
					Value: connectionName,
				})
				container.Env = append(container.Env, corev1.EnvVar{
					Name:  constants.ConnectionNamespaceEnv,
					Value: connectionNamespace,
				})
				container.Env = append(container.Env, corev1.EnvVar{
					Name:  constants.GetConnectionURLEnv,
					Value: fmt.Sprintf("%s/api/connection?name=%s&namespace=%s", clientConfig.OperatorEndpoint, connectionName, connectionNamespace),
				})
				containerPatched = true
			}
			pod.Spec.Containers[i] = *container
		}
	}

	if !containerPatched {
		return nil, fmt.Errorf("no container found that needs tf-client injection")
	}

	containerPatchedJSON, err := json.Marshal(pod)
	if err != nil {
		return nil, fmt.Errorf("marshal current pod: %w", err)
	}
	patches, err := jsonpatch.CreatePatch(currentBytes, containerPatchedJSON)
	if err != nil {
		return nil, fmt.Errorf("patch to container: %w", err)
	}

	// Convert the strategic merge patch to JSON
	patchBytes, err := json.Marshal(clientConfig.PatchToPod)

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

	patches = append(patches, strategicpatches...)
	return patches, nil
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

	url := fmt.Sprintf("http://%s:8080/assign-host-port?podName=%s", leaderIP, pod.Name)
	resp, err := httpClient.Get(url)
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
	sameReqLimits := profile.Resources.Limits.Tflops.Value() == profile.Resources.Requests.Tflops.Value() &&
		profile.Resources.Limits.Vram.Value() == profile.Resources.Requests.Vram.Value()

	// set to critical if req == limits, same logic as Kubernetes QoS
	if sameReqLimits {
		return constants.QoSLevelCritical
	}

	// when not set, assign default QoS
	if profile.Qos == "" {
		if pool.Spec.QosConfig == nil || pool.Spec.QosConfig.DefaultQoS == "" {
			return constants.QoSLevelMedium
		}
		return pool.Spec.QosConfig.DefaultQoS
	}
	return profile.Qos
}
