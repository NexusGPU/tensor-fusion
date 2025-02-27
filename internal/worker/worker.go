package worker

import (
	"encoding/json"
	"fmt"
	"path"
	"strconv"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
	"github.com/samber/lo"
	"golang.org/x/exp/rand"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func init() {
	rand.Seed(uint64(time.Now().UnixNano()))
}

type WorkerGenerator struct {
	WorkerConfig *tfv1.WorkerConfig
}

func (wg *WorkerGenerator) GenerateConnectionURL(connection *tfv1.TensorFusionConnection, pod *corev1.Pod) (string, error) {
	port, ok := lo.Find(pod.Spec.Containers[0].Env, func(env corev1.EnvVar) bool {
		return env.Name == constants.WorkerPortEnv
	})

	if !ok {
		return "", fmt.Errorf("worker port not found in pod %s", pod.Name)
	}
	return fmt.Sprintf("native+%s+%s", pod.Status.PodIP, port.Value), nil
}

func (wg *WorkerGenerator) AllocPort() int {
	min := 30000
	max := 65535
	return rand.Intn(max-min+1) + min
}

func (wg *WorkerGenerator) GenerateWorkerPod(
	gpu *tfv1.GPU,
	connection *tfv1.TensorFusionConnection,
	namespacedName types.NamespacedName,
	port int,
) (*corev1.Pod, error) {
	podTmpl := &corev1.PodTemplate{}
	err := json.Unmarshal(wg.WorkerConfig.PodTemplate.Raw, podTmpl)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal pod template: %w", err)
	}
	spec := podTmpl.Template.Spec
	if spec.NodeSelector == nil {
		spec.NodeSelector = make(map[string]string)
	}
	spec.NodeSelector = gpu.Status.NodeSelector
	spec.Volumes = append(spec.Volumes, corev1.Volume{
		Name: constants.DataVolumeName,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{
				Path: path.Join(constants.TFDataPath, namespacedName.Name),
			},
		},
	})
	spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts, corev1.VolumeMount{
		Name:      constants.DataVolumeName,
		MountPath: constants.TFDataPath,
	})

	spec.Containers[0].Env = append(spec.Containers[0].Env, corev1.EnvVar{
		Name:  "NVIDIA_VISIBLE_DEVICES",
		Value: gpu.Status.UUID,
	}, corev1.EnvVar{
		Name:  constants.WorkerPortEnv,
		Value: strconv.Itoa(port),
	})

	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      namespacedName.Name,
			Namespace: namespacedName.Namespace,
		},
		Spec: spec,
	}, nil
}
