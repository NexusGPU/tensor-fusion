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

	"gomodules.xyz/jsonpatch/v2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
	"github.com/lithammer/shortuuid/v4"
	"github.com/samber/lo"
)

// SetupPodWebhookWithManager registers the webhook for Pod in the manager.
func SetupPodWebhookWithManager(mgr ctrl.Manager) error {
	webhookServer := mgr.GetWebhookServer()

	webhookServer.Register("/mutate-v1-pod",
		&admission.Webhook{
			Handler: &TensorFusionPodMutator{
				decoder: admission.NewDecoder(runtime.NewScheme()),
				Client:  mgr.GetClient(),
			},
		})
	return nil
}

type TensorFusionPodMutator struct {
	Client  client.Client
	decoder admission.Decoder
}

// Handle implements admission.Handler interface.
func (m *TensorFusionPodMutator) Handle(ctx context.Context, req admission.Request) admission.Response {
	pod := &corev1.Pod{}
	if err := m.decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}

	log := log.FromContext(ctx)
	log.Info("Mutating pod", "generateName", pod.GenerateName, "namespace", pod.Namespace)

	poolName, resources := ParseTFResources(pod)
	if len(resources) == 0 {
		log.Info("No tensor fusion requirements found for pod", "podName", pod.Name, "namespace: ", pod.Namespace)
		return admission.Allowed("no tensor fusion requirements found")
	}

	pool := &tfv1.GPUPool{}
	if err := m.Client.Get(ctx, client.ObjectKey{Name: poolName}, pool); err != nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("gpu pool(%s) does not exist", poolName))
	}

	// 1. Inject initContainer and env variables
	patches, err := m.patchTFClient(pod, pool.Spec.ComponentConfig.Client, resources)
	if err != nil {
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

func ParseTFResources(pod *corev1.Pod) (poolName string, resources []TFResource) {
	if pod.Annotations == nil {
		return "", nil
	}

	resources = make([]TFResource, 0, len(pod.Spec.Containers))
	poolName, ok := pod.Annotations[constants.GpuPoolKey]
	if !ok {
		return "", nil
	}

	for _, container := range pod.Spec.Containers {
		containerName := container.Name

		// Check if TF requirements exist for this container
		tflopsReqKey := fmt.Sprintf(constants.TFLOPSRequestAnnotationFormat, containerName)
		vramReqKey := fmt.Sprintf(constants.VRAMRequestAnnotationFormat, containerName)
		tflopsLimitKey := fmt.Sprintf(constants.TFLOPSLimitAnnotationFormat, containerName)
		vramLimitKey := fmt.Sprintf(constants.VRAMLimitAnnotationFormat, containerName)

		tflopsReqStr, hasTflopsReq := pod.Annotations[tflopsReqKey]
		vramReqStr, hasVramReq := pod.Annotations[vramReqKey]

		tflopsLimitStr, hasTflopsLimit := pod.Annotations[tflopsLimitKey]
		vramLimitStr, hasVramLimit := pod.Annotations[vramLimitKey]

		if !hasTflopsReq && !hasVramReq && !hasTflopsLimit && !hasVramLimit {
			continue
		}

		res := TFResource{
			ContainerName: containerName,
		}
		connectionNameEnv, ok := lo.Find(container.Env, func(e corev1.EnvVar) bool {
			return e.Name == constants.ConnectionNameEnv
		})
		if ok {
			res.ConnectionName = connectionNameEnv.Value
		}
		connectionNamespaceEnv, ok := lo.Find(container.Env, func(e corev1.EnvVar) bool {
			return e.Name == constants.ConnectionNamespaceEnv
		})
		if ok {
			res.ConnectionNamespace = connectionNamespaceEnv.Value
		}
		// Parse TFLOPS request
		if hasTflopsReq {
			tflops, err := resource.ParseQuantity(tflopsReqStr)
			if err == nil {
				res.TflopsRequest = tflops
			}
		}

		// Parse VRAM request
		if hasVramReq {
			vram, err := resource.ParseQuantity(vramReqStr)
			if err == nil {
				res.VramRequest = vram
			}
		}

		// Parse TFLOPS limit
		if hasTflopsReq {
			tflops, err := resource.ParseQuantity(tflopsLimitStr)
			if err == nil {
				res.TflopsLimit = tflops
			}
		}

		// Parse VRAM limit
		if hasVramReq {
			vram, err := resource.ParseQuantity(vramLimitStr)
			if err == nil {
				res.VramLimit = vram
			}
		}

		resources = append(resources, res)
	}

	return poolName, resources
}

func (m *TensorFusionPodMutator) patchTFClient(pod *corev1.Pod, clientConfig *tfv1.ClientConfig, tfResources []TFResource) ([]jsonpatch.JsonPatchOperation, error) {
	// Convert the current pod to JSON
	currentBytes, err := json.Marshal(pod)
	if err != nil {
		return nil, fmt.Errorf("marshal current pod: %w", err)
	}

	// Patch to Container
	for _, res := range tfResources {
		for i := range pod.Spec.Containers {
			container := &pod.Spec.Containers[i]
			if container.Name == res.ContainerName {
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
				connectionName := fmt.Sprintf("%s-tf-worker-%s", pod.GenerateName+container.Name, shortuuid.NewWithAlphabet("123456789abcdefghijkmnopqrstuvwxy"))
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
