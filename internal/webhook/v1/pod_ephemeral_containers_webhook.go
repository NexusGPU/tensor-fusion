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
	"errors"
	"fmt"
	"net/http"
	"path"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

const (
	// podEphemeralContainersValidatePath is where the API server sends updates
	// of the pods/ephemeralcontainers subresource. It sends them for every pod:
	// the registration has no selector, see dataVolumeNames.
	podEphemeralContainersValidatePath = "/validate-v1-pod-ephemeralcontainers"

	ephemeralContainersSubResource = "ephemeralcontainers"
)

// reservedDataPathEnvs are the variables that constants.TFDataPathWorkerExpr
// expands, with the downward API field each one is always set from. They are
// what binds the per-pod mount of the shared data volume to the pod's own
// namespace and name.
var reservedDataPathEnvs = []struct {
	name      string
	fieldPath string
}{
	{constants.TFShmPodNamespaceEnv, constants.NamespaceFieldRef},
	{constants.TFShmPodNameEnv, constants.ResourceNameFieldRef},
}

// isUnderDataRoot reports whether hostPath is the node data directory or a
// path below it. The path is cleaned first, so a trailing or doubled slash
// makes no difference, and a sibling such as /run/tensor-fusion-other does not
// count.
func isUnderDataRoot(hostPath string) bool {
	cleaned := path.Clean(hostPath)
	return cleaned == constants.TFDataPath || strings.HasPrefix(cleaned, constants.TFDataPath+"/")
}

// isDataVolume reports whether volume gives a container access to the node
// data directory. The volume the pod webhook injects is recognised by its
// name, and any hostPath volume at or below the data root by what it points
// to, whatever it is called.
func isDataVolume(volume *corev1.Volume) bool {
	if volume.Name == constants.DataVolumeName {
		return true
	}
	return volume.HostPath != nil && isUnderDataRoot(volume.HostPath.Path)
}

// dataVolumeNames returns the names of the data volumes of pod. A pod that has
// at least one is subject to the ephemeral container rule, any other pod is
// not.
//
// The rule is selected by the volumes and not by a label or an annotation
// because the volumes of a pod cannot be changed after it is created, while
// its metadata can. For the same reason the volumes in an update of the
// ephemeral containers subresource are always the stored ones: the API server
// takes nothing but the ephemeral containers from such a request.
func dataVolumeNames(pod *corev1.Pod) map[string]struct{} {
	var names map[string]struct{}
	for i := range pod.Spec.Volumes {
		if !isDataVolume(&pod.Spec.Volumes[i]) {
			continue
		}
		if names == nil {
			names = map[string]struct{}{}
		}
		names[pod.Spec.Volumes[i].Name] = struct{}{}
	}
	return names
}

// workerDataMount is the only mount of the shared data volume the pod webhook
// composes for a workload container: the pod's own directory of that volume.
// It mirrors the mount built in utils.AddTFDefaultClientConfBeforePatch.
func workerDataMount() corev1.VolumeMount {
	return corev1.VolumeMount{
		Name:             constants.DataVolumeName,
		MountPath:        constants.TFDataPath + constants.SharedMemMountSubPath,
		SubPathExpr:      constants.TFDataPathWorkerExpr,
		MountPropagation: ptr.To(corev1.MountPropagationHostToContainer),
	}
}

// isReservedDataPathEnv reports whether env sets a reserved variable the way
// the pod webhook does: from the given downward API field and nothing else.
func isReservedDataPathEnv(env corev1.EnvVar, fieldPath string) bool {
	if env.Value != "" || env.ValueFrom == nil {
		return false
	}
	want := &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: fieldPath}}
	got := env.ValueFrom.DeepCopy()
	if got.FieldRef != nil && got.FieldRef.APIVersion == "v1" {
		// The API server defaults the field reference version.
		got.FieldRef.APIVersion = ""
	}
	return equality.Semantic.DeepEqual(got, want)
}

// ephemeralContainerViolations returns why an ephemeral container may not be
// added to a pod with the given data volumes, or nothing when it may.
//
// The pod webhook only sees pod creation, where it limits every container to
// the pod's own directory of the shared data volume. Ephemeral containers are
// added later through their own subresource, so the same limit is applied to
// them here: such a container either leaves the data volumes and the reserved
// variables alone, or uses them exactly as the pod webhook composes them. The
// composed mount names the injected volume, so a data volume under any other
// name cannot be mounted at all.
func ephemeralContainerViolations(container *corev1.EphemeralContainer, dataVolumes map[string]struct{}) []string {
	var violations []string

	dataMounts := 0
	for _, mount := range container.VolumeMounts {
		if _, found := dataVolumes[mount.Name]; !found {
			continue
		}
		dataMounts++
		if dataMounts > 1 || !equality.Semantic.DeepEqual(mount, workerDataMount()) {
			violations = append(violations, fmt.Sprintf(
				"volume %q holds TensorFusion node data and cannot be mounted at %q by an ephemeral container",
				mount.Name, mount.MountPath))
		}
	}
	for _, device := range container.VolumeDevices {
		if _, found := dataVolumes[device.Name]; found {
			violations = append(violations, fmt.Sprintf(
				"volume %q holds TensorFusion node data and cannot be attached as a device by an ephemeral container",
				device.Name))
		}
	}

	for _, reserved := range reservedDataPathEnvs {
		defined := false
		for _, env := range container.Env {
			if env.Name != reserved.name {
				continue
			}
			defined = true
			if !isReservedDataPathEnv(env, reserved.fieldPath) {
				violations = append(violations, fmt.Sprintf(
					"environment variable %q is reserved for TensorFusion and cannot be set by an ephemeral container",
					reserved.name))
			}
		}
		// The mount resolves through these variables. Left undefined here they
		// could still come from envFrom, which is not visible at admission.
		if dataMounts > 0 && !defined {
			violations = append(violations, fmt.Sprintf(
				"environment variable %q must be set from the downward API field %q to mount volume %q",
				reserved.name, reserved.fieldPath, constants.DataVolumeName))
		}
	}
	return violations
}

// validateEphemeralContainers checks the ephemeral containers that an update
// adds to pod. Existing ones are immutable and were admitted earlier, so they
// are not judged again. A pod without a data volume is accepted unchanged.
func validateEphemeralContainers(pod, oldPod *corev1.Pod) error {
	dataVolumes := dataVolumeNames(pod)
	if len(dataVolumes) == 0 {
		return nil
	}
	existing := map[string]struct{}{}
	if oldPod != nil {
		for i := range oldPod.Spec.EphemeralContainers {
			existing[oldPod.Spec.EphemeralContainers[i].Name] = struct{}{}
		}
	}

	var violations []string
	for i := range pod.Spec.EphemeralContainers {
		container := &pod.Spec.EphemeralContainers[i]
		if _, found := existing[container.Name]; found {
			continue
		}
		for _, violation := range ephemeralContainerViolations(container, dataVolumes) {
			violations = append(violations, fmt.Sprintf("ephemeral container %q: %s", container.Name, violation))
		}
	}
	if len(violations) == 0 {
		return nil
	}
	return errors.New(strings.Join(violations, "; "))
}

// EphemeralContainerValidator admits updates of the pods/ephemeralcontainers
// subresource. It receives them for every pod in the cluster and applies its
// rule to the pods that carry a data volume.
type EphemeralContainerValidator struct {
	decoder admission.Decoder
}

// Handle implements admission.Handler interface.
func (v *EphemeralContainerValidator) Handle(ctx context.Context, req admission.Request) admission.Response {
	if req.SubResource != ephemeralContainersSubResource {
		return admission.Allowed("not an ephemeral containers update, skipped")
	}

	pod := &corev1.Pod{}
	if err := v.decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}
	// Most pods have no data volume. Answer for them before anything else is
	// decoded or compared: this is on the path of every debugging session.
	if len(dataVolumeNames(pod)) == 0 {
		return admission.Allowed("pod has no TensorFusion data volume, skipped")
	}
	var oldPod *corev1.Pod
	if len(req.OldObject.Raw) > 0 {
		oldPod = &corev1.Pod{}
		if err := v.decoder.DecodeRaw(req.OldObject, oldPod); err != nil {
			return admission.Errored(http.StatusBadRequest, err)
		}
	}

	if err := validateEphemeralContainers(pod, oldPod); err != nil {
		log.FromContext(ctx).Info("ephemeral container refused", "reason", err.Error())
		return admission.Denied(err.Error())
	}
	return admission.Allowed("")
}
