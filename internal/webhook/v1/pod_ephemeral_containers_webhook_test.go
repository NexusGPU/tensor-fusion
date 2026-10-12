package v1

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var _ = Describe("EphemeralContainerValidator", func() {
	// The name the rule is registered under in config/webhook/manifests.yaml.
	// A refusal that carries it was decided by this webhook and not by the
	// API server's own validation.
	const ruleName = "vpod-ephemeralcontainers-v1.kb.io"

	hasDataVolume := func(pod *corev1.Pod) bool {
		return lo.ContainsBy(pod.Spec.Volumes, func(volume corev1.Volume) bool {
			return volume.Name == constants.DataVolumeName && volume.HostPath != nil
		})
	}

	// createLabeledPod submits a workload pod with the label the pod webhook is
	// selected by and returns it as stored, that is after the pod webhook
	// composed it for the given isolation mode.
	createLabeledPod := func(isolation tfv1.IsolationModeType) *corev1.Pod {
		submitted := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace:    "default",
				GenerateName: "ephemeral-managed-",
				Labels: map[string]string{
					constants.TensorFusionEnabledLabelKey: constants.TrueStringValue,
				},
				Annotations: map[string]string{
					constants.GpuPoolKey:                "mock",
					constants.IsLocalGPUAnnotation:      "true",
					constants.IsolationModeAnnotation:   string(isolation),
					constants.InjectContainerAnnotation: "main",
					constants.TFLOPSRequestAnnotation:   "10",
					constants.VRAMRequestAnnotation:     "1Gi",
					constants.TFLOPSLimitAnnotation:     "20",
					constants.VRAMLimitAnnotation:       "2Gi",
				},
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "main", Image: "test-image"}},
				Volumes: []corev1.Volume{{
					Name:         scratchVolumeName,
					VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
				}},
			},
		}
		Expect(k8sClient.Create(ctx, submitted)).To(Succeed())
		DeferCleanup(func() { _ = k8sClient.Delete(ctx, submitted) })
		pod := &corev1.Pod{}
		Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(submitted), pod)).To(Succeed())
		Expect(pod.Labels).To(HaveKeyWithValue(constants.TensorFusionEnabledLabelKey, constants.TrueStringValue))
		return pod
	}

	// createManagedPod returns a stored pod that carries the data volume
	// because the pod webhook added it.
	createManagedPod := func(isolation tfv1.IsolationModeType) *corev1.Pod {
		pod := createLabeledPod(isolation)
		Expect(hasDataVolume(pod)).To(BeTrue(), "the pod webhook adds the data volume")
		return pod
	}

	// createUnlabeledPod stores a pod with exactly the given volumes and no
	// TensorFusion label, so the pod webhook never sees it.
	createUnlabeledPod := func(volumes ...corev1.Volume) *corev1.Pod {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Namespace: "default", GenerateName: "ephemeral-unlabeled-"},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: "main", Image: "test-image"}},
				Volumes:    volumes,
			},
		}
		Expect(k8sClient.Create(ctx, pod)).To(Succeed())
		DeferCleanup(func() { _ = k8sClient.Delete(ctx, pod) })
		Expect(pod.Labels).NotTo(HaveKey(constants.TensorFusionEnabledLabelKey))
		return pod
	}

	wholeVolume := func(volume string) corev1.EphemeralContainer {
		return ephemeralContainer(debugContainerName,
			[]corev1.VolumeMount{{Name: volume, MountPath: debugMountPath}}, nil)
	}
	literalPodName := corev1.EnvVar{Name: constants.TFShmPodNameEnv, Value: otherPodName}

	// addEphemeralContainer updates the pods/ephemeralcontainers subresource
	// the way a debugging client does.
	addEphemeralContainer := func(pod *corev1.Pod, container corev1.EphemeralContainer) error {
		latest := &corev1.Pod{}
		Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(pod), latest)).To(Succeed())
		latest.Spec.EphemeralContainers = append(latest.Spec.EphemeralContainers, container)
		return k8sClient.SubResource(ephemeralContainersSubResource).Update(ctx, latest)
	}

	ephemeralContainerNames := func(pod *corev1.Pod) []string {
		stored := &corev1.Pod{}
		Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(pod), stored)).To(Succeed())
		return lo.Map(stored.Spec.EphemeralContainers, func(container corev1.EphemeralContainer, _ int) string {
			return container.Name
		})
	}

	expectRefused := func(pod *corev1.Pod, err error, named string) {
		GinkgoHelper()
		Expect(err).To(HaveOccurred())
		Expect(apierrors.IsForbidden(err)).To(BeTrue(), err.Error())
		Expect(err.Error()).To(ContainSubstring(ruleName))
		Expect(err.Error()).To(ContainSubstring(named))
		Expect(ephemeralContainerNames(pod)).To(BeEmpty())
	}

	Context("when adding an ephemeral container through the API server", func() {
		// The rule compares against its own description of the composed mount
		// and variables, so that description has to match what the pod webhook
		// really stores for the pod's regular containers.
		DescribeTable("should describe the data mount exactly as the pod webhook composes it",
			func(isolation tfv1.IsolationModeType) {
				pod := createManagedPod(isolation)
				mounted := 0
				for _, container := range pod.Spec.Containers {
					for _, mount := range container.VolumeMounts {
						if mount.Name != constants.DataVolumeName {
							continue
						}
						mounted++
						Expect(mount).To(Equal(workerDataMount()), container.Name)
						for _, reserved := range reservedDataPathEnvs {
							definitions := lo.Filter(container.Env, func(env corev1.EnvVar, _ int) bool {
								return env.Name == reserved.name
							})
							Expect(definitions).To(HaveLen(1), "%s defines %s once", container.Name, reserved.name)
							Expect(isReservedDataPathEnv(definitions[0], reserved.fieldPath)).To(BeTrue(), container.Name)
						}
					}
				}
				Expect(mounted).To(BeNumerically(">", 0))
			},
			Entry("soft isolation", tfv1.IsolationModeType(tfv1.IsolationModeSoft)),
			Entry("worker sidecar", tfv1.IsolationModeType(tfv1.IsolationModeHard)),
		)

		DescribeTable("should refuse an ephemeral container that mounts the data volume other than as composed",
			func(mount corev1.VolumeMount) {
				pod := createManagedPod(tfv1.IsolationModeSoft)
				err := addEphemeralContainer(pod, ephemeralContainer(debugContainerName, []corev1.VolumeMount{mount}, nil))
				expectRefused(pod, err, constants.DataVolumeName)
			},
			Entry("at another path", corev1.VolumeMount{
				Name: constants.DataVolumeName, MountPath: debugMountPath,
			}),
			Entry("at the composed path without the per-pod sub-path", corev1.VolumeMount{
				Name: constants.DataVolumeName, MountPath: constants.TFDataPath + constants.SharedMemMountSubPath,
			}),
			Entry("read-only", corev1.VolumeMount{
				Name: constants.DataVolumeName, MountPath: debugMountPath, ReadOnly: true,
			}),
		)

		DescribeTable("should refuse an ephemeral container that sets a reserved variable",
			func(env corev1.EnvVar) {
				pod := createManagedPod(tfv1.IsolationModeSoft)
				err := addEphemeralContainer(pod, ephemeralContainer(debugContainerName, nil, []corev1.EnvVar{env}))
				expectRefused(pod, err, env.Name)
			},
			Entry("pod name, literal value", corev1.EnvVar{Name: constants.TFShmPodNameEnv, Value: otherPodName}),
			Entry("pod namespace, literal value", corev1.EnvVar{Name: constants.TFShmPodNamespaceEnv, Value: otherPodName}),
			Entry("pod name, another field reference",
				downwardEnv(constants.TFShmPodNameEnv, "metadata.labels['"+constants.TensorFusionEnabledLabelKey+"']")),
		)

		It("should admit a plain debug container for a managed pod", func() {
			pod := createManagedPod(tfv1.IsolationModeSoft)
			container := ephemeralContainer(debugContainerName,
				[]corev1.VolumeMount{{Name: scratchVolumeName, MountPath: "/scratch"}},
				[]corev1.EnvVar{{Name: "DEBUG", Value: "1"}})
			container.TargetContainerName = "main"
			Expect(addEphemeralContainer(pod, container)).To(Succeed())
			Expect(ephemeralContainerNames(pod)).To(ConsistOf(debugContainerName))

			By("admitting another one next to it")
			Expect(addEphemeralContainer(pod, ephemeralContainer("second", nil, nil))).To(Succeed())
			Expect(ephemeralContainerNames(pod)).To(ConsistOf(debugContainerName, "second"))
		})

		// The label is what selects a pod for the pod webhook, and it can be
		// changed afterwards. The volumes cannot, so the rule follows them.
		It("should keep refusing after the label is removed from a managed pod", func() {
			pod := createManagedPod(tfv1.IsolationModeSoft)
			unlabeled := pod.DeepCopy()
			delete(unlabeled.Labels, constants.TensorFusionEnabledLabelKey)
			Expect(k8sClient.Patch(ctx, unlabeled, client.MergeFrom(pod))).To(Succeed())
			stored := &corev1.Pod{}
			Expect(k8sClient.Get(ctx, client.ObjectKeyFromObject(pod), stored)).To(Succeed())
			Expect(stored.Labels).NotTo(HaveKey(constants.TensorFusionEnabledLabelKey))

			expectRefused(pod, addEphemeralContainer(pod, wholeVolume(constants.DataVolumeName)), constants.DataVolumeName)
			err := addEphemeralContainer(pod, ephemeralContainer(debugContainerName, nil, []corev1.EnvVar{literalPodName}))
			expectRefused(pod, err, literalPodName.Name)

			By("still admitting a plain debug container")
			Expect(addEphemeralContainer(pod, ephemeralContainer(debugContainerName, nil, nil))).To(Succeed())
			Expect(ephemeralContainerNames(pod)).To(ConsistOf(debugContainerName))
		})

		DescribeTable("should apply the rule to a pod without the label that carries a data volume",
			func(volume corev1.Volume) {
				pod := createUnlabeledPod(volume, emptyDirVolume(scratchVolumeName))

				expectRefused(pod, addEphemeralContainer(pod, wholeVolume(volume.Name)), volume.Name)
				err := addEphemeralContainer(pod, ephemeralContainer(debugContainerName, nil, []corev1.EnvVar{literalPodName}))
				expectRefused(pod, err, literalPodName.Name)

				By("still admitting a plain debug container")
				container := ephemeralContainer(debugContainerName,
					[]corev1.VolumeMount{{Name: scratchVolumeName, MountPath: "/scratch"}},
					[]corev1.EnvVar{{Name: "DEBUG", Value: "1"}})
				Expect(addEphemeralContainer(pod, container)).To(Succeed())
				Expect(ephemeralContainerNames(pod)).To(ConsistOf(debugContainerName))
			},
			Entry("the data volume as the pod webhook adds it", injectedDataVolume()),
			Entry("a host path at the data root under another name",
				hostPathVolume(renamedVolumeName, constants.TFDataPath)),
			Entry("a host path below the data root under another name",
				hostPathVolume(renamedVolumeName, constants.TFDataPath+"/shm/")),
			Entry("the reserved volume name with another source", emptyDirVolume(constants.DataVolumeName)),
		)

		It("should admit any ephemeral container for a labeled pod without the data volume", func() {
			// Shared isolation has no consumer of the data volume, so the
			// pod webhook does not add it.
			pod := createLabeledPod(tfv1.IsolationModeShared)
			Expect(pod.Spec.Volumes).NotTo(ContainElement(Satisfy(func(volume corev1.Volume) bool {
				return isDataVolume(&volume)
			})), "the pod webhook adds no data volume in this mode")

			Expect(addEphemeralContainer(pod, ephemeralContainer(debugContainerName,
				[]corev1.VolumeMount{{Name: scratchVolumeName, MountPath: debugMountPath}},
				[]corev1.EnvVar{literalPodName, {Name: constants.TFShmPodNamespaceEnv, Value: otherPodName}},
			))).To(Succeed())
			Expect(ephemeralContainerNames(pod)).To(ConsistOf(debugContainerName))
		})

		DescribeTable("should leave a pod without a data volume unaffected",
			func(volumes []corev1.Volume) {
				pod := createUnlabeledPod(volumes...)
				container := ephemeralContainer(debugContainerName, nil, []corev1.EnvVar{literalPodName})
				for _, volume := range volumes {
					container.VolumeMounts = append(container.VolumeMounts,
						corev1.VolumeMount{Name: volume.Name, MountPath: "/mnt/" + volume.Name})
				}
				Expect(addEphemeralContainer(pod, container)).To(Succeed())
				Expect(ephemeralContainerNames(pod)).To(ConsistOf(debugContainerName))
			},
			Entry("no volumes", []corev1.Volume(nil)),
			Entry("an emptyDir volume", []corev1.Volume{emptyDirVolume(scratchVolumeName)}),
			Entry("a host path just outside the data root",
				[]corev1.Volume{hostPathVolume(renamedVolumeName, constants.TFDataPath+"-other")}),
			Entry("a host path elsewhere on the node",
				[]corev1.Volume{hostPathVolume(renamedVolumeName, "/var/log")}),
		)
	})
})
