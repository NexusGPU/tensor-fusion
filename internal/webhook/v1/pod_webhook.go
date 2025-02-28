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
	"net/http"
	"strings"

	"gomodules.xyz/jsonpatch/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
	"github.com/NexusGPU/tensor-fusion-operator/internal/scheduler"
	"github.com/lithammer/shortuuid/v4"
)

// SetupPodWebhookWithManager registers the webhook for Pod in the manager.
func SetupPodWebhookWithManager(mgr ctrl.Manager, scheduler scheduler.Scheduler) error {
	webhookServer := mgr.GetWebhookServer()

	webhookServer.Register("/mutate-v1-pod",
		&admission.Webhook{
			Handler: &TensorFusionPodMutator{
				decoder:   admission.NewDecoder(runtime.NewScheme()),
				Client:    mgr.GetClient(),
				scheduler: scheduler,
			},
		})
	return nil
}

type TensorFusionPodMutator struct {
	Client    client.Client
	decoder   admission.Decoder
	scheduler scheduler.Scheduler
}

// Handle implements admission.Handler interface.
func (m *TensorFusionPodMutator) Handle(ctx context.Context, req admission.Request) admission.Response {
	pod := &corev1.Pod{}
	if err := m.decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}

	log := log.FromContext(ctx)
	log.Info("Mutating pod", "generateName", pod.GenerateName, "namespace", pod.Namespace)

	profile, containerNames, err := ParseTFResources(ctx, m.Client, pod)
	if err != nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("parse tf resources: %w", err))
	}

	pool := &tfv1.GPUPool{}
	if err := m.Client.Get(ctx, client.ObjectKey{Name: profile.PoolName}, pool); err != nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("gpu pool(%s) does not exist", profile.PoolName))
	}

	var gpu *tfv1.GPU = nil
	if profile.IsLocalGPU {
		gpu, err = m.scheduler.Schedule(ctx, profile.PoolName, profile.Resources.Requests)
		if err != nil {
			log.Error(err, "failed to schedule gpu for pod", "pod", req.Name, "namespace", req.Namespace)
			return admission.Errored(http.StatusInternalServerError, fmt.Errorf("schedule gpu: %w", err))
		}
	}

	// Inject initContainer and env variables
	patches, err := m.patchTFClient(pod, pool.Spec.ComponentConfig.Client, containerNames, gpu)
	if err != nil {
		if gpu != nil {
			retryErr := retry.OnError(retry.DefaultRetry, func(err error) bool {
				// Retry on all errors
				return true
			}, func() error {
				return m.scheduler.Release(ctx, profile.Resources.Requests, gpu)
			})
			if retryErr != nil {
				log.Error(retryErr, "failed to release gpu after multiple retries", "pod", req.Name, "namespace", req.Namespace)
			}
		}
		return admission.Errored(http.StatusInternalServerError, err)
	}

	return admission.Patched("tensor fusion component patched", patches...)
}

// InjectDecoder injects the decoder.
func (m *TensorFusionPodMutator) InjectDecoder(d admission.Decoder) error {
	m.decoder = d
	return nil
}

type TFResource struct {
	ContainerName       string
	ConnectionName      string
	ConnectionNamespace string
	TflopsRequest       resource.Quantity
	VramRequest         resource.Quantity
	TflopsLimit         resource.Quantity
	VramLimit           resource.Quantity
}

func ParseTFResources(ctx context.Context, k8sclient client.Client, pod *corev1.Pod) (profile *tfv1.ClientProfileSpec, containerNames []string, err error) {
	if pod.Annotations == nil {
		return nil, nil, fmt.Errorf("no annotations found")
	}
	clientProfileName, ok := pod.Annotations[constants.ClientProfileAnnotation]
	clientProfile := &tfv1.ClientProfile{}
	if ok {
		if err := k8sclient.Get(ctx, client.ObjectKey{Name: clientProfileName, Namespace: pod.Namespace}, clientProfile); err != nil {
			return nil, nil, fmt.Errorf("get client profile(%s) : %w", clientProfileName, err)
		}
	}

	poolName, ok := pod.Annotations[constants.GpuPoolKey]
	if !ok {
		// TODO: select default pool
		return nil, nil, fmt.Errorf("gpu pool not found")
	}
	clientProfile.Spec.PoolName = poolName

	tflopsRequest, ok := pod.Annotations[constants.TFLOPSRequestAnnotation]
	if ok {
		clientProfile.Spec.Resources.Requests.Tflops = resource.MustParse(tflopsRequest)
	}
	vramRequest, ok := pod.Annotations[constants.VRAMRequestAnnotation]
	if ok {
		clientProfile.Spec.Resources.Requests.Vram = resource.MustParse(vramRequest)
	}
	tflopsLimit, ok := pod.Annotations[constants.TFLOPSLimitAnnotation]
	if ok {
		clientProfile.Spec.Resources.Limits.Tflops = resource.MustParse(tflopsLimit)
	}
	vramLimit, ok := pod.Annotations[constants.VRAMLimitAnnotation]
	if ok {
		clientProfile.Spec.Resources.Limits.Vram = resource.MustParse(vramLimit)
	}

	injectContainer, ok := pod.Annotations[constants.InjectContainerAnnotation]
	containerNames = strings.Split(injectContainer, ",")
	if !ok || len(containerNames) == 0 {
		return nil, nil, fmt.Errorf("inject container not found")
	}
	return &clientProfile.Spec, containerNames, nil
}

func (m *TensorFusionPodMutator) patchTFClient(pod *corev1.Pod, clientConfig *tfv1.ClientConfig, containerNames []string, gpu *tfv1.GPU) ([]jsonpatch.JsonPatchOperation, error) {
	// Convert the current pod to JSON
	currentBytes, err := json.Marshal(pod)
	if err != nil {
		return nil, fmt.Errorf("marshal current pod: %w", err)
	}

	if gpu != nil {
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
								Values:   []string{gpu.Status.NodeSelector["kubernetes.io/hostname"]},
							},
						},
					},
				},
			},
		}
		// Add GPU annotation
		if pod.Annotations == nil {
			pod.Annotations = make(map[string]string)
		}
		pod.Annotations[constants.GPUAnnotation] = gpu.Name
	}

	// Patch to Container
	for _, name := range containerNames {
		for i := range pod.Spec.Containers {
			container := &pod.Spec.Containers[i]
			if container.Name == name {
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

				// add connection env
				connectionName := fmt.Sprintf("%s-tf-worker-%s", pod.GenerateName, shortuuid.NewWithAlphabet("123456789abcdefghijkmnopqrstuvwxy"))
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
			}
			pod.Spec.Containers[i] = *container
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
