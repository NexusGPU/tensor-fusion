package utils

import (
	"fmt"
	"slices"

	constants "github.com/NexusGPU/tensor-fusion/pkg/constants"
	v1 "k8s.io/api/core/v1"
)

// setWorkerDataPathEnv makes the variables expanded by TFDataPathWorkerExpr
// resolve to the pod's own namespace and name. The kubelet expands subPathExpr
// from the container's environment, where the last definition of a name wins
// and an explicit definition overrides envFrom, so every other definition of
// these reserved names is dropped and the downward API ones are added.
func setWorkerDataPathEnv(envList []v1.EnvVar) []v1.EnvVar {
	result := make([]v1.EnvVar, 0, len(envList)+2)
	for _, env := range envList {
		if env.Name != constants.TFShmPodNamespaceEnv && env.Name != constants.TFShmPodNameEnv {
			result = append(result, env)
		}
	}
	return append(result,
		v1.EnvVar{Name: constants.TFShmPodNamespaceEnv, ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{FieldPath: constants.NamespaceFieldRef},
		}},
		v1.EnvVar{Name: constants.TFShmPodNameEnv, ValueFrom: &v1.EnvVarSource{
			FieldRef: &v1.ObjectFieldSelector{FieldPath: constants.ResourceNameFieldRef},
		}},
	)
}

// isWorkerDataMount reports whether mount is the per-pod mount of the shared
// data volume, the only form in which that volume may enter a workload.
func isWorkerDataMount(mount v1.VolumeMount) bool {
	return mount.Name == constants.DataVolumeName &&
		mount.MountPath == constants.TFDataPath+constants.SharedMemMountSubPath &&
		mount.SubPath == "" && mount.SubPathExpr == constants.TFDataPathWorkerExpr
}

// enforceWorkerDataMounts is the last word on the shared data volume, a
// hostPath that holds the state of every pod on the node. A container, init
// containers included, keeps at most the per-pod mount of it, and that mount
// can only resolve to the pod's own directory. Mounts of the volume that the
// pod spec declared itself are removed: the volume only exists because it was
// injected, so no workload can have a legitimate mount of its own.
func enforceWorkerDataMounts(spec *v1.PodSpec) {
	enforce := func(container *v1.Container) {
		mounted := false
		mounts := make([]v1.VolumeMount, 0, len(container.VolumeMounts))
		for _, mount := range container.VolumeMounts {
			if mount.Name == constants.DataVolumeName {
				if mounted || !isWorkerDataMount(mount) {
					continue
				}
				mounted = true
			}
			mounts = append(mounts, mount)
		}
		if len(mounts) != len(container.VolumeMounts) {
			container.VolumeMounts = mounts
		}
		if mounted {
			container.Env = setWorkerDataPathEnv(container.Env)
		}
	}
	for i := range spec.InitContainers {
		enforce(&spec.InitContainers[i])
	}
	for i := range spec.Containers {
		enforce(&spec.Containers[i])
	}
}

// Use the workload image and mounts to preserve its existing preload file. A
// separate file per container also keeps container-specific libraries isolated.
func addSoftLimiterPreload(spec *v1.PodSpec, containerIndex int, limiterLib string) {
	container := &spec.Containers[containerIndex]
	original := container.DeepCopy()
	volumeName := fmt.Sprintf("tf-soft-preload-%d", containerIndex)
	const outputDir = "/tensor-fusion-preload"
	spec.Volumes = append(spec.Volumes, v1.Volume{
		Name:         volumeName,
		VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
	})
	spec.InitContainers = append(spec.InitContainers, v1.Container{
		Name:            volumeName,
		Image:           original.Image,
		ImagePullPolicy: original.ImagePullPolicy,
		SecurityContext: original.SecurityContext,
		Env:             original.Env,
		EnvFrom:         original.EnvFrom,
		WorkingDir:      original.WorkingDir,
		Command: []string{"/bin/sh", "-ec", `
if [ -e /etc/ld.so.preload ]; then
    cat /etc/ld.so.preload > "$1"
else
    : > "$1"
fi
printf '\n%s\n' "$2" >> "$1"
chmod 644 "$1"
`, "tf-soft-preload", outputDir + "/" + constants.LdPreloadFileName,
			constants.TFSoftLimiterVolumeMountPath + "/" + limiterLib},
		VolumeMounts: append(original.VolumeMounts, v1.VolumeMount{
			Name: volumeName, MountPath: outputDir,
		}),
		Resources: v1.ResourceRequirements{Requests: injectLibResource, Limits: injectLibResource},
	})
	// The init container retains the original mount; replace it only in the
	// business container so Kubernetes does not reject duplicate mount paths.
	container.VolumeMounts = slices.DeleteFunc(container.VolumeMounts, func(m v1.VolumeMount) bool {
		return m.MountPath == constants.LdPreloadFile
	})
	container.VolumeMounts = append(container.VolumeMounts, v1.VolumeMount{
		Name: volumeName, MountPath: constants.LdPreloadFile,
		SubPath: constants.LdPreloadFileName, ReadOnly: true,
	})
}

func shouldInjectCudaLimiter(vendor string) bool {
	return shouldInjectNvidiaVisibleDevices(vendor)
}
