package utils_test

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

const (
	ownNamespace   = "team-a"
	ownPodName     = "train-0"
	otherNamespace = "team-b"
	otherPodName   = "serve-0"
)

var envReference = regexp.MustCompile(`\$\(([A-Za-z_][A-Za-z0-9_.-]*)\)`)

// workerDataPathVariables are the variables the kubelet substitutes into the
// per-pod data mount.
func workerDataPathVariables() []string {
	matches := envReference.FindAllStringSubmatch(constants.TFDataPathWorkerExpr, -1)
	names := make([]string, 0, len(matches))
	for _, match := range matches {
		names = append(names, match[1])
	}
	return names
}

func podFieldValue(pod *corev1.Pod, fieldPath string) string {
	switch {
	case fieldPath == "metadata.name":
		return pod.Name
	case fieldPath == "metadata.namespace":
		return pod.Namespace
	case fieldPath == "metadata.uid":
		return string(pod.UID)
	case strings.HasPrefix(fieldPath, "metadata.labels['"):
		return pod.Labels[strings.TrimSuffix(strings.TrimPrefix(fieldPath, "metadata.labels['"), "']")]
	}
	return ""
}

// kubeletEnvironment resolves a container's environment the way the kubelet
// does before it expands subPathExpr: definitions are applied in order, a later
// definition of a name replaces an earlier one, and $(VAR) refers to variables
// defined so far.
func kubeletEnvironment(pod *corev1.Pod, container corev1.Container) map[string]string {
	resolved := map[string]string{}
	for _, env := range container.Env {
		value := envReference.ReplaceAllStringFunc(env.Value, func(reference string) string {
			if known, ok := resolved[envReference.FindStringSubmatch(reference)[1]]; ok {
				return known
			}
			return reference
		})
		if env.ValueFrom != nil && env.ValueFrom.FieldRef != nil {
			value = podFieldValue(pod, env.ValueFrom.FieldRef.FieldPath)
		}
		resolved[env.Name] = value
	}
	return resolved
}

// kubeletSubPath expands subPathExpr like the kubelet; an undefined or empty
// variable fails container creation.
func kubeletSubPath(pod *corev1.Pod, container corev1.Container, mount corev1.VolumeMount) (string, error) {
	env := kubeletEnvironment(pod, container)
	var missing []string
	expanded := envReference.ReplaceAllStringFunc(mount.SubPathExpr, func(reference string) string {
		name := envReference.FindStringSubmatch(reference)[1]
		if env[name] == "" {
			missing = append(missing, name)
		}
		return env[name]
	})
	if len(missing) > 0 {
		return "", fmt.Errorf("missing value for %s", strings.Join(missing, ", "))
	}
	return expanded, nil
}

// expectOnlyOwnWorkerData asserts that no container can reach the shared data
// volume other than through the mount of the pod's own directory, and returns
// how many containers have that mount.
func expectOnlyOwnWorkerData(pod *corev1.Pod) int {
	GinkgoHelper()
	mounted := 0
	containers := append(append([]corev1.Container{}, pod.Spec.InitContainers...), pod.Spec.Containers...)
	for _, container := range containers {
		paths := map[string]int{}
		for _, mount := range container.VolumeMounts {
			paths[mount.MountPath]++
			if mount.Name != constants.DataVolumeName {
				continue
			}
			mounted++
			Expect(mount.MountPath).To(Equal(constants.TFDataPath+constants.SharedMemMountSubPath), container.Name)
			Expect(mount.SubPath).To(BeEmpty(), container.Name)
			subPath, err := kubeletSubPath(pod, container, mount)
			Expect(err).NotTo(HaveOccurred(), container.Name)
			Expect(subPath).To(Equal("shm/"+ownNamespace+"/"+ownPodName), container.Name)
		}
		for path, count := range paths {
			Expect(count).To(Equal(1), "container %s mounts %s more than once", container.Name, path)
		}
	}
	return mounted
}

func literalEnv(names []string, value func(name string) string) []corev1.EnvVar {
	envs := make([]corev1.EnvVar, 0, len(names))
	for _, name := range names {
		envs = append(envs, corev1.EnvVar{Name: name, Value: value(name)})
	}
	return envs
}

func fieldEnv(name, fieldPath string) corev1.EnvVar {
	return corev1.EnvVar{Name: name, ValueFrom: &corev1.EnvVarSource{
		FieldRef: &corev1.ObjectFieldSelector{FieldPath: fieldPath},
	}}
}

// identityVariables are all the names a workload might define to describe
// "its" pod: the conventional ones and the ones the data mount expands.
func identityVariables() []string {
	names := []string{constants.PodNamespaceEnv, constants.PodNameEnv}
	for _, name := range workerDataPathVariables() {
		if name != constants.PodNamespaceEnv && name != constants.PodNameEnv {
			names = append(names, name)
		}
	}
	return names
}

func otherPodValue(name string) string {
	if strings.HasSuffix(name, "NAMESPACE") {
		return otherNamespace
	}
	return otherPodName
}

var _ = Describe("Per-pod data mount", func() {
	newPool := func() *tfv1.GPUPool {
		return &tfv1.GPUPool{
			Spec: tfv1.GPUPoolSpec{
				ComponentConfig: &tfv1.ComponentConfig{
					Client:     &tfv1.ClientConfig{Image: "client:latest"},
					Worker:     &tfv1.WorkerConfig{Image: "worker:latest"},
					Hypervisor: &tfv1.HypervisorConfig{Image: "hypervisor:latest"},
				},
			},
		}
	}
	newPod := func() *corev1.Pod {
		return &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace:   ownNamespace,
				Name:        ownPodName,
				UID:         types.UID("7b0d9a5e-6f0c-4d57-9b1c-0c3f0f1a2b3c"),
				Labels:      map[string]string{"neighbour": otherPodName},
				Annotations: map[string]string{},
			},
			Spec: corev1.PodSpec{
				InitContainers: []corev1.Container{{Name: "setup"}},
				Containers:     []corev1.Container{{Name: "main"}, {Name: "helper"}, {Name: "logger"}},
			},
		}
	}
	// main and helper are injected; logger and the init container are not.
	inject := func(pod *corev1.Pod, isolation tfv1.IsolationModeType) {
		utils.AddTFDefaultClientConfBeforePatch(context.Background(), pod, newPool(), utils.TensorFusionInfo{
			Profile: &tfv1.WorkloadProfileSpec{
				IsLocalGPU: true,
				Isolation:  isolation,
				GPUVendor:  constants.AcceleratorVendorNvidia,
			},
		}, []int{0, 1})
	}
	eachContainer := func(pod *corev1.Pod, mutate func(container *corev1.Container)) {
		for i := range pod.Spec.InitContainers {
			mutate(&pod.Spec.InitContainers[i])
		}
		for i := range pod.Spec.Containers {
			mutate(&pod.Spec.Containers[i])
		}
	}
	dataMount := func(mount corev1.VolumeMount) corev1.VolumeMount {
		mount.Name = constants.DataVolumeName
		return mount
	}

	type workloadSpec func(pod *corev1.Pod)
	variants := []TableEntry{
		Entry("no identity variables defined by the workload", workloadSpec(func(*corev1.Pod) {})),
		Entry("identity variables defined as literal values", workloadSpec(func(pod *corev1.Pod) {
			eachContainer(pod, func(container *corev1.Container) {
				container.Env = literalEnv(identityVariables(), otherPodValue)
			})
		})),
		Entry("empty values", workloadSpec(func(pod *corev1.Pod) {
			eachContainer(pod, func(container *corev1.Container) {
				container.Env = literalEnv(identityVariables(), func(string) string { return "" })
			})
		})),
		Entry("identity variables defined as other field references", workloadSpec(func(pod *corev1.Pod) {
			eachContainer(pod, func(container *corev1.Container) {
				for _, name := range identityVariables() {
					fieldPath := "metadata.labels['neighbour']"
					if strings.HasSuffix(name, "NAMESPACE") {
						fieldPath = "metadata.uid"
					}
					container.Env = append(container.Env, fieldEnv(name, fieldPath))
				}
			})
		})),
		Entry("namespace and name field references swapped", workloadSpec(func(pod *corev1.Pod) {
			eachContainer(pod, func(container *corev1.Container) {
				for _, name := range identityVariables() {
					fieldPath := constants.NamespaceFieldRef
					if strings.HasSuffix(name, "NAMESPACE") {
						fieldPath = constants.ResourceNameFieldRef
					}
					container.Env = append(container.Env, fieldEnv(name, fieldPath))
				}
			})
		})),
		Entry("repeated definitions", workloadSpec(func(pod *corev1.Pod) {
			eachContainer(pod, func(container *corev1.Container) {
				container.Env = append(literalEnv(identityVariables(), func(string) string { return "first" }),
					literalEnv(identityVariables(), otherPodValue)...)
			})
		})),
		Entry("values composed from other variables", workloadSpec(func(pod *corev1.Pod) {
			eachContainer(pod, func(container *corev1.Container) {
				container.Env = append([]corev1.EnvVar{{Name: "TARGET", Value: otherPodName}},
					literalEnv(identityVariables(), func(string) string { return "$(TARGET)" })...)
			})
		})),
		Entry("mounts of the data volume declared by the workload", workloadSpec(func(pod *corev1.Pod) {
			eachContainer(pod, func(container *corev1.Container) {
				container.Env = literalEnv(identityVariables(), otherPodValue)
				container.VolumeMounts = append(container.VolumeMounts,
					dataMount(corev1.VolumeMount{MountPath: "/all"}),
					dataMount(corev1.VolumeMount{MountPath: "/fixed", SubPath: "shm/" + otherNamespace + "/" + otherPodName}),
					dataMount(corev1.VolumeMount{MountPath: "/expr", SubPathExpr: "shm/$(POD_NAMESPACE)/$(POD_NAME)"}),
					dataMount(corev1.VolumeMount{MountPath: "/parent", SubPathExpr: "shm/$(POD_NAMESPACE)"}),
				)
			})
		})),
		Entry("a mount declared at the data path itself", workloadSpec(func(pod *corev1.Pod) {
			eachContainer(pod, func(container *corev1.Container) {
				container.VolumeMounts = append(container.VolumeMounts,
					dataMount(corev1.VolumeMount{MountPath: constants.TFDataPath + constants.SharedMemMountSubPath}))
			})
		})),
	}

	DescribeTable("resolves only to the pod's own directory with soft isolation",
		func(prepare workloadSpec) {
			pod := newPod()
			prepare(pod)
			inject(pod, tfv1.IsolationModeSoft)
			Expect(expectOnlyOwnWorkerData(pod)).To(Equal(2), "both injected containers keep their own mount")
		}, variants)

	DescribeTable("resolves only to the pod's own directory with a local worker sidecar",
		func(prepare workloadSpec) {
			pod := newPod()
			prepare(pod)
			inject(pod, tfv1.IsolationModeHard)
			Expect(pod.Spec.Containers[len(pod.Spec.Containers)-1].Name).To(Equal(constants.TFContainerNameWorker))
			Expect(expectOnlyOwnWorkerData(pod)).To(Equal(3), "both injected containers and the sidecar keep their own mount")
		}, variants)

	It("leaves the workload's own POD_NAME and POD_NAMESPACE definitions alone", func() {
		pod := newPod()
		pod.Spec.Containers[0].Env = []corev1.EnvVar{
			{Name: constants.PodNameEnv, Value: "logical-name"},
			{Name: constants.PodNamespaceEnv, Value: "logical-namespace"},
			{Name: "GREETING", Value: "hello $(POD_NAME)"},
		}
		inject(pod, tfv1.IsolationModeSoft)

		env := kubeletEnvironment(pod, pod.Spec.Containers[0])
		Expect(env[constants.PodNameEnv]).To(Equal("logical-name"))
		Expect(env[constants.PodNamespaceEnv]).To(Equal("logical-namespace"))
		Expect(env["GREETING"]).To(Equal("hello logical-name"))
		Expect(countEnv(pod.Spec.Containers[0].Env, constants.PodNameEnv)).To(Equal(1))
		Expect(expectOnlyOwnWorkerData(pod)).To(Equal(2))
	})

	It("keeps a volume of the same name that belongs to a remote-mode workload", func() {
		pod := newPod()
		pod.Spec.Volumes = []corev1.Volume{{
			Name:         constants.DataVolumeName,
			VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
		}}
		pod.Spec.Containers[0].VolumeMounts = []corev1.VolumeMount{dataMount(corev1.VolumeMount{MountPath: "/data"})}
		utils.AddTFDefaultClientConfBeforePatch(context.Background(), pod, newPool(), utils.TensorFusionInfo{
			Profile: &tfv1.WorkloadProfileSpec{GPUVendor: constants.AcceleratorVendorNvidia},
		}, []int{0})

		Expect(hasVolumeMount(pod.Spec.Containers[0].VolumeMounts, constants.DataVolumeName, "/data")).To(BeTrue())
		Expect(hasHostPathVolume(pod.Spec.Volumes, constants.DataVolumeName, constants.TFDataPath)).To(BeFalse())
	})

	It("resolves only to the worker pod's own directory whatever the worker template defines", func() {
		pod := newPod()
		container := &corev1.Container{Env: literalEnv(identityVariables(), otherPodValue)}
		utils.SetWorkerContainerSpec(container,
			&tfv1.WorkloadProfileSpec{GPUVendor: constants.AcceleratorVendorNvidia, Isolation: tfv1.IsolationModeSoft},
			&tfv1.WorkerConfig{}, &tfv1.HypervisorConfig{}, "", false)
		pod.Spec.InitContainers, pod.Spec.Containers = nil, []corev1.Container{*container}

		Expect(expectOnlyOwnWorkerData(pod)).To(Equal(1))
	})
})
