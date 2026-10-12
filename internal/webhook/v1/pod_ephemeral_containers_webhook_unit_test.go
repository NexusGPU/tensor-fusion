package v1

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

const (
	debugContainerName = "debugger"
	debugMountPath     = "/data"
	otherPodName       = "another-pod"
	scratchVolumeName  = "scratch"
	// renamedVolumeName is a volume that points into the data root without
	// carrying the name of the injected volume.
	renamedVolumeName = "node-state"
)

// injectedDataVolume is the data volume as the pod webhook adds it.
func injectedDataVolume() corev1.Volume {
	return hostPathVolume(constants.DataVolumeName, constants.TFDataPath)
}

func hostPathVolume(name, path string) corev1.Volume {
	return corev1.Volume{Name: name, VolumeSource: corev1.VolumeSource{
		HostPath: &corev1.HostPathVolumeSource{Path: path},
	}}
}

func emptyDirVolume(name string) corev1.Volume {
	return corev1.Volume{Name: name, VolumeSource: corev1.VolumeSource{
		EmptyDir: &corev1.EmptyDirVolumeSource{},
	}}
}

func downwardEnv(name, fieldPath string) corev1.EnvVar {
	return corev1.EnvVar{Name: name, ValueFrom: &corev1.EnvVarSource{
		FieldRef: &corev1.ObjectFieldSelector{FieldPath: fieldPath},
	}}
}

func composedDataPathEnvs() []corev1.EnvVar {
	return []corev1.EnvVar{
		downwardEnv(constants.TFShmPodNamespaceEnv, constants.NamespaceFieldRef),
		downwardEnv(constants.TFShmPodNameEnv, constants.ResourceNameFieldRef),
	}
}

func ephemeralContainer(name string, mounts []corev1.VolumeMount, envs []corev1.EnvVar) corev1.EphemeralContainer {
	return corev1.EphemeralContainer{EphemeralContainerCommon: corev1.EphemeralContainerCommon{
		Name: name, Image: "debug-image", VolumeMounts: mounts, Env: envs,
	}}
}

// podWithVolumes builds a pod with exactly the given volumes. Only labeled
// pods carry the label the pod webhook is selected by.
func podWithVolumes(
	labeled bool, volumes []corev1.Volume, containers ...corev1.EphemeralContainer,
) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "workload"},
		Spec: corev1.PodSpec{
			Containers:          []corev1.Container{{Name: "main", Image: "test-image"}},
			Volumes:             volumes,
			EphemeralContainers: containers,
		},
	}
	if labeled {
		pod.Labels = map[string]string{constants.TensorFusionEnabledLabelKey: constants.TrueStringValue}
	}
	return pod
}

// podWithEphemeralContainers builds a pod that carries the injected data
// volume and, on purpose, no label: the rule must not depend on one.
func podWithEphemeralContainers(containers ...corev1.EphemeralContainer) *corev1.Pod {
	volumes := []corev1.Volume{injectedDataVolume(), emptyDirVolume(scratchVolumeName)}
	return podWithVolumes(false, volumes, containers...)
}

// expectVerdict checks the outcome for the container named debugContainerName:
// admitted when refused is empty, otherwise refused with a reason that names
// the container and everything in refused.
func expectVerdict(t *testing.T, err error, refused []string) {
	t.Helper()
	if len(refused) == 0 {
		if err != nil {
			t.Fatalf("expected the ephemeral container to be admitted, got: %v", err)
		}
		return
	}
	if err == nil {
		t.Fatal("expected the ephemeral container to be refused")
	}
	for _, want := range append([]string{debugContainerName}, refused...) {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("refusal %q does not name %s", err.Error(), want)
		}
	}
}

func TestIsUnderDataRoot(t *testing.T) {
	tests := []struct {
		path string
		want bool
	}{
		{constants.TFDataPath, true},
		{constants.TFDataPath + "/", true},
		{constants.TFDataPath + "/shm", true},
		{constants.TFDataPath + "/shm/default/workload", true},
		{"/run//tensor-fusion", true},
		{"/run/./tensor-fusion/shm/", true},
		{"/run/other/../tensor-fusion", true},
		{constants.TFDataPath + "-other", false},
		{constants.TFDataPath + "-other/shm", false},
		{constants.TFDataPath + "/../other", false},
		{"/run/tensor-fusio", false},
		{"/data" + constants.TFDataPath, false},
		{"/var/lib/kubelet", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := isUnderDataRoot(tt.path); got != tt.want {
			t.Errorf("isUnderDataRoot(%q) = %v, want %v", tt.path, got, tt.want)
		}
	}
}

// The pods the rule applies to are found by what their volumes are. Labels
// and annotations play no part, since they can change after creation.
func TestDataVolumeNames(t *testing.T) {
	tests := []struct {
		name    string
		labeled bool
		volumes []corev1.Volume
		want    []string
	}{
		{name: "pod without volumes"},
		{name: "labeled pod without a data volume", labeled: true, volumes: []corev1.Volume{
			emptyDirVolume(scratchVolumeName), hostPathVolume("logs", "/var/log"),
		}},
		{name: "injected volume on a pod without the label",
			volumes: []corev1.Volume{emptyDirVolume(scratchVolumeName), injectedDataVolume()},
			want:    []string{constants.DataVolumeName}},
		{name: "injected volume on a labeled pod", labeled: true,
			volumes: []corev1.Volume{injectedDataVolume()},
			want:    []string{constants.DataVolumeName}},
		{name: "the reserved volume name with another source",
			volumes: []corev1.Volume{emptyDirVolume(constants.DataVolumeName)},
			want:    []string{constants.DataVolumeName}},
		{name: "host path at the data root under another name",
			volumes: []corev1.Volume{hostPathVolume(renamedVolumeName, constants.TFDataPath)},
			want:    []string{renamedVolumeName}},
		{name: "host path below the data root under another name",
			volumes: []corev1.Volume{hostPathVolume(renamedVolumeName, constants.TFDataPath+"/shm/")},
			want:    []string{renamedVolumeName}},
		{name: "host path next to the data root",
			volumes: []corev1.Volume{hostPathVolume(renamedVolumeName, constants.TFDataPath+"-other")}},
		{name: "every data volume is reported",
			volumes: []corev1.Volume{
				injectedDataVolume(),
				hostPathVolume(renamedVolumeName, constants.TFDataPath+"/shm"),
				hostPathVolume("sibling", constants.TFDataPath+"-other"),
			},
			want: []string{constants.DataVolumeName, renamedVolumeName}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := dataVolumeNames(podWithVolumes(tt.labeled, tt.volumes))
			if len(got) != len(tt.want) {
				t.Fatalf("expected data volumes %v, got %v", tt.want, got)
			}
			for _, name := range tt.want {
				if _, found := got[name]; !found {
					t.Errorf("expected %q among the data volumes, got %v", name, got)
				}
			}
		})
	}
}

func TestValidateEphemeralContainers(t *testing.T) {
	composed := workerDataMount()
	modified := func(change func(*corev1.VolumeMount)) corev1.VolumeMount {
		mount := workerDataMount()
		change(&mount)
		return mount
	}

	tests := []struct {
		name   string
		mounts []corev1.VolumeMount
		envs   []corev1.EnvVar
		// refused lists what the refusal has to name; empty means admitted.
		refused []string
	}{
		{name: "plain debug container is admitted"},
		{
			name:   "mount of another volume is admitted",
			mounts: []corev1.VolumeMount{{Name: scratchVolumeName, MountPath: "/scratch"}},
			envs:   []corev1.EnvVar{{Name: "DEBUG", Value: "1"}},
		},
		{
			name:   "data mount and reserved variables exactly as composed are admitted",
			mounts: []corev1.VolumeMount{composed},
			envs:   composedDataPathEnvs(),
		},
		{
			name: "reserved variables with the defaulted field reference version are admitted",
			envs: []corev1.EnvVar{{Name: constants.TFShmPodNameEnv, ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{APIVersion: "v1", FieldPath: constants.ResourceNameFieldRef},
			}}},
		},
		{
			name:    "data volume without the per-pod sub-path is refused",
			mounts:  []corev1.VolumeMount{{Name: constants.DataVolumeName, MountPath: debugMountPath}},
			refused: []string{constants.DataVolumeName},
		},
		{
			name: "data volume with another sub-path expression is refused",
			mounts: []corev1.VolumeMount{modified(func(mount *corev1.VolumeMount) {
				mount.SubPathExpr = "shm/$(OTHER)"
			})},
			envs:    composedDataPathEnvs(),
			refused: []string{constants.DataVolumeName},
		},
		{
			name: "data volume with a literal sub-path is refused",
			mounts: []corev1.VolumeMount{modified(func(mount *corev1.VolumeMount) {
				mount.SubPathExpr = ""
				mount.SubPath = "shm"
			})},
			envs:    composedDataPathEnvs(),
			refused: []string{constants.DataVolumeName},
		},
		{
			name: "data volume at another mount path is refused",
			mounts: []corev1.VolumeMount{modified(func(mount *corev1.VolumeMount) {
				mount.MountPath = debugMountPath
			})},
			envs:    composedDataPathEnvs(),
			refused: []string{constants.DataVolumeName},
		},
		{
			name: "data volume with another mount propagation is refused",
			mounts: []corev1.VolumeMount{modified(func(mount *corev1.VolumeMount) {
				mount.MountPropagation = ptr.To(corev1.MountPropagationBidirectional)
			})},
			envs:    composedDataPathEnvs(),
			refused: []string{constants.DataVolumeName},
		},
		{
			name: "second mount of the data volume is refused",
			mounts: []corev1.VolumeMount{composed, modified(func(mount *corev1.VolumeMount) {
				mount.MountPath = debugMountPath
			})},
			envs:    composedDataPathEnvs(),
			refused: []string{constants.DataVolumeName},
		},
		{
			name:    "composed data mount without the reserved variables is refused",
			mounts:  []corev1.VolumeMount{composed},
			refused: []string{constants.TFShmPodNamespaceEnv, constants.TFShmPodNameEnv},
		},
		{
			name:    "reserved variable with a literal value is refused",
			envs:    []corev1.EnvVar{{Name: constants.TFShmPodNameEnv, Value: otherPodName}},
			refused: []string{constants.TFShmPodNameEnv},
		},
		{
			name:    "reserved variable with an empty value is refused",
			envs:    []corev1.EnvVar{{Name: constants.TFShmPodNamespaceEnv}},
			refused: []string{constants.TFShmPodNamespaceEnv},
		},
		{
			name: "reserved variable from another field is refused",
			envs: []corev1.EnvVar{
				downwardEnv(constants.TFShmPodNamespaceEnv, "metadata.labels['team']"),
			},
			refused: []string{constants.TFShmPodNamespaceEnv},
		},
		{
			name: "reserved variable from another source is refused",
			envs: []corev1.EnvVar{{Name: constants.TFShmPodNameEnv, ValueFrom: &corev1.EnvVarSource{
				ConfigMapKeyRef: &corev1.ConfigMapKeySelector{
					LocalObjectReference: corev1.LocalObjectReference{Name: "settings"}, Key: "name",
				},
			}}},
			refused: []string{constants.TFShmPodNameEnv},
		},
		{
			name: "reserved variable defined again with a literal value is refused",
			envs: append(composedDataPathEnvs(),
				corev1.EnvVar{Name: constants.TFShmPodNameEnv, Value: otherPodName}),
			mounts:  []corev1.VolumeMount{composed},
			refused: []string{constants.TFShmPodNameEnv},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			oldPod := podWithEphemeralContainers()
			pod := podWithEphemeralContainers(ephemeralContainer(debugContainerName, tt.mounts, tt.envs))

			err := validateEphemeralContainers(pod, oldPod)
			expectVerdict(t, err, tt.refused)
		})
	}
}

// Whether the rule applies follows from the pod's volumes alone. The cases
// are the ones the label used to decide: a pod with the data volume is held to
// the rule without the label, and a pod without one is not held to it with the
// label.
func TestValidateEphemeralContainersSelectsPodsByDataVolume(t *testing.T) {
	mount := func(volume string) []corev1.VolumeMount {
		return []corev1.VolumeMount{{Name: volume, MountPath: debugMountPath}}
	}
	literalPodName := []corev1.EnvVar{{Name: constants.TFShmPodNameEnv, Value: otherPodName}}
	renamedAsComposed := workerDataMount()
	renamedAsComposed.Name = renamedVolumeName

	tests := []struct {
		name    string
		labeled bool
		volumes []corev1.Volume
		mounts  []corev1.VolumeMount
		envs    []corev1.EnvVar
		// refused lists what the refusal has to name; empty means admitted.
		refused []string
	}{
		{
			name:    "data volume without the label: mount is refused",
			volumes: []corev1.Volume{injectedDataVolume()},
			mounts:  mount(constants.DataVolumeName),
			refused: []string{constants.DataVolumeName},
		},
		{
			name:    "data volume without the label: reserved variable is refused",
			volumes: []corev1.Volume{injectedDataVolume()},
			envs:    literalPodName,
			refused: []string{constants.TFShmPodNameEnv},
		},
		{
			name:    "data volume without the label: plain debug container is admitted",
			volumes: []corev1.Volume{injectedDataVolume()},
		},
		{
			name:    "label without a data volume: any mount and variable is admitted",
			labeled: true,
			volumes: []corev1.Volume{emptyDirVolume(scratchVolumeName)},
			mounts:  mount(scratchVolumeName),
			envs:    literalPodName,
		},
		{
			name:    "label without a data volume: plain debug container is admitted",
			labeled: true,
		},
		{
			name:    "host path below the data root under another name: mount is refused",
			volumes: []corev1.Volume{hostPathVolume(renamedVolumeName, constants.TFDataPath+"/shm")},
			mounts:  mount(renamedVolumeName),
			refused: []string{renamedVolumeName},
		},
		{
			name:    "host path at the data root under another name: the composed form is refused too",
			volumes: []corev1.Volume{hostPathVolume(renamedVolumeName, constants.TFDataPath)},
			mounts:  []corev1.VolumeMount{renamedAsComposed},
			envs:    composedDataPathEnvs(),
			refused: []string{renamedVolumeName},
		},
		{
			name:    "host path below the data root under another name: reserved variable is refused",
			volumes: []corev1.Volume{hostPathVolume(renamedVolumeName, constants.TFDataPath+"/shm")},
			envs:    literalPodName,
			refused: []string{constants.TFShmPodNameEnv},
		},
		{
			name:    "host path below the data root under another name: plain debug container is admitted",
			volumes: []corev1.Volume{hostPathVolume(renamedVolumeName, constants.TFDataPath+"/shm")},
		},
		{
			name: "second data volume next to the injected one: mount is refused",
			volumes: []corev1.Volume{
				injectedDataVolume(), hostPathVolume(renamedVolumeName, constants.TFDataPath),
			},
			mounts:  append([]corev1.VolumeMount{workerDataMount()}, mount(renamedVolumeName)...),
			envs:    composedDataPathEnvs(),
			refused: []string{renamedVolumeName},
		},
		{
			name:    "the reserved volume name with another source: mount is refused",
			volumes: []corev1.Volume{emptyDirVolume(constants.DataVolumeName)},
			mounts:  mount(constants.DataVolumeName),
			refused: []string{constants.DataVolumeName},
		},
		{
			name:    "host path next to the data root: any mount and variable is admitted",
			volumes: []corev1.Volume{hostPathVolume(renamedVolumeName, constants.TFDataPath+"-other")},
			mounts:  mount(renamedVolumeName),
			envs:    literalPodName,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			oldPod := podWithVolumes(tt.labeled, tt.volumes)
			pod := podWithVolumes(tt.labeled, tt.volumes, ephemeralContainer(debugContainerName, tt.mounts, tt.envs))

			err := validateEphemeralContainers(pod, oldPod)
			expectVerdict(t, err, tt.refused)
		})
	}
}

// Ephemeral containers cannot be changed once added, so only the ones an
// update adds are judged: an earlier one must not block later debugging.
func TestValidateEphemeralContainersJudgesOnlyAddedContainers(t *testing.T) {
	wholeVolume := []corev1.VolumeMount{{Name: constants.DataVolumeName, MountPath: debugMountPath}}
	earlier := ephemeralContainer("earlier", wholeVolume, nil)

	oldPod := podWithEphemeralContainers(earlier)
	pod := podWithEphemeralContainers(earlier, ephemeralContainer(debugContainerName, nil, nil))
	if err := validateEphemeralContainers(pod, oldPod); err != nil {
		t.Fatalf("expected a plain container to be admitted next to an earlier one, got: %v", err)
	}

	pod = podWithEphemeralContainers(earlier, ephemeralContainer("added", wholeVolume, nil))
	err := validateEphemeralContainers(pod, oldPod)
	if err == nil {
		t.Fatal("expected the added container to be refused")
	}
	if !strings.Contains(err.Error(), `"added"`) || strings.Contains(err.Error(), `"earlier"`) {
		t.Fatalf("refusal should name only the added container, got: %v", err)
	}

	// Without the previous object every ephemeral container counts as added.
	if err := validateEphemeralContainers(podWithEphemeralContainers(earlier), nil); err == nil {
		t.Fatal("expected the container to be refused when the previous pod is unknown")
	}
}

func TestEphemeralContainerValidatorHandle(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	validator := &EphemeralContainerValidator{decoder: admission.NewDecoder(scheme)}

	request := func(subResource string, pod, oldPod *corev1.Pod) admission.Request {
		t.Helper()
		raw, err := json.Marshal(pod)
		if err != nil {
			t.Fatal(err)
		}
		oldRaw, err := json.Marshal(oldPod)
		if err != nil {
			t.Fatal(err)
		}
		return admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
			Operation:   admissionv1.Update,
			SubResource: subResource,
			Namespace:   pod.Namespace,
			Name:        pod.Name,
			Object:      runtime.RawExtension{Raw: raw},
			OldObject:   runtime.RawExtension{Raw: oldRaw},
		}}
	}
	wholeVolume := ephemeralContainer(debugContainerName,
		[]corev1.VolumeMount{{Name: constants.DataVolumeName, MountPath: debugMountPath}}, nil)

	t.Run("refuses with a forbidden status that names the container", func(t *testing.T) {
		resp := validator.Handle(t.Context(), request(ephemeralContainersSubResource,
			podWithEphemeralContainers(wholeVolume), podWithEphemeralContainers()))
		if resp.Allowed {
			t.Fatal("expected the update to be refused")
		}
		if resp.Result == nil || resp.Result.Code != http.StatusForbidden {
			t.Fatalf("expected a forbidden status, got: %+v", resp.Result)
		}
		if !strings.Contains(resp.Result.Message, debugContainerName) {
			t.Fatalf("refusal should name the container, got: %q", resp.Result.Message)
		}
	})

	t.Run("admits a plain debug container", func(t *testing.T) {
		resp := validator.Handle(t.Context(), request(ephemeralContainersSubResource,
			podWithEphemeralContainers(ephemeralContainer(debugContainerName, nil, nil)),
			podWithEphemeralContainers()))
		if !resp.Allowed {
			t.Fatalf("expected the update to be admitted, got: %+v", resp.Result)
		}
	})

	t.Run("refuses for a pod that lost its label", func(t *testing.T) {
		labeled := podWithVolumes(true, []corev1.Volume{injectedDataVolume()})
		unlabeled := podWithVolumes(false, []corev1.Volume{injectedDataVolume()}, wholeVolume)
		resp := validator.Handle(t.Context(), request(ephemeralContainersSubResource, unlabeled, labeled))
		if resp.Allowed {
			t.Fatal("expected the update to be refused")
		}
	})

	t.Run("admits a labeled pod without a data volume", func(t *testing.T) {
		volumes := []corev1.Volume{emptyDirVolume(scratchVolumeName)}
		container := ephemeralContainer(debugContainerName,
			[]corev1.VolumeMount{{Name: scratchVolumeName, MountPath: debugMountPath}},
			[]corev1.EnvVar{{Name: constants.TFShmPodNameEnv, Value: otherPodName}})
		resp := validator.Handle(t.Context(), request(ephemeralContainersSubResource,
			podWithVolumes(true, volumes, container), podWithVolumes(true, volumes)))
		if !resp.Allowed {
			t.Fatalf("expected the update to be admitted, got: %+v", resp.Result)
		}
	})

	// The handler is called for every pod in the cluster. For one without a
	// data volume it answers from the submitted pod alone and does not look
	// at the previous object, which the same request with a data volume needs.
	t.Run("answers for a pod without a data volume before reading the previous object", func(t *testing.T) {
		container := ephemeralContainer(debugContainerName, nil, nil)
		unreadable := runtime.RawExtension{Raw: []byte("{")}

		req := request(ephemeralContainersSubResource, podWithVolumes(false, nil, container), podWithVolumes(false, nil))
		req.OldObject = unreadable
		if resp := validator.Handle(t.Context(), req); !resp.Allowed {
			t.Fatalf("expected the update to be admitted, got: %+v", resp.Result)
		}

		req = request(ephemeralContainersSubResource, podWithEphemeralContainers(container), podWithEphemeralContainers())
		req.OldObject = unreadable
		resp := validator.Handle(t.Context(), req)
		if resp.Allowed || resp.Result == nil || resp.Result.Code != http.StatusBadRequest {
			t.Fatalf("expected a bad request for a pod with a data volume, got: %+v", resp)
		}
	})

	t.Run("leaves other pod requests alone", func(t *testing.T) {
		resp := validator.Handle(t.Context(), request("",
			podWithEphemeralContainers(wholeVolume), podWithEphemeralContainers()))
		if !resp.Allowed {
			t.Fatalf("expected requests for other resources to be admitted, got: %+v", resp.Result)
		}
	})
}
