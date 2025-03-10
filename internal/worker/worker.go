package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"path"
	"strconv"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/samber/lo"
	"golang.org/x/exp/rand"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func init() {
	rand.Seed(uint64(time.Now().UnixNano()))
}

type WorkerGenerator struct {
	WorkerConfig *tfv1.WorkerConfig
}

func (wg *WorkerGenerator) WorkerPort(pod *corev1.Pod) (int, error) {
	port, ok := lo.Find(pod.Spec.Containers[0].Env, func(env corev1.EnvVar) bool {
		return env.Name == constants.WorkerPortEnv
	})

	if !ok {
		return 0, fmt.Errorf("worker port not found in pod %s", pod.Name)
	}

	return strconv.Atoi(port.Value)
}

func (wg *WorkerGenerator) AllocPort() int {
	min := 30000
	max := 65535
	return rand.Intn(max-min+1) + min
}

func (wg *WorkerGenerator) GenerateWorkerPod(
	gpu *tfv1.GPU,
	namespacedName types.NamespacedName,
	port int,
	limits tfv1.Resource,
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
	}, corev1.EnvVar{
		Name: constants.WokerCudaUpLimitEnv,
		// TODO: convert tflops to percent
		Value: "100",
	}, corev1.EnvVar{
		Name: constants.WokerCudaMemLimitEnv,
		// bytesize
		Value: strconv.FormatInt(limits.Vram.Value(), 10),
	})

	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      namespacedName.Name,
			Namespace: namespacedName.Namespace,
		},
		Spec: spec,
	}, nil
}

func SelectWorker(ctx context.Context, k8sClient client.Client, workloadName string, workerStatuses []tfv1.WorkerStatus) (*tfv1.WorkerStatus, error) {
	if len(workerStatuses) == 0 {
		return nil, fmt.Errorf("no available worker")
	}
	usageMapping := make(map[string]int, len(workerStatuses))
	for _, workerStatus := range workerStatuses {
		usageMapping[workerStatus.WorkerName] = 0
	}

	connectionList := tfv1.TensorFusionConnectionList{}
	if err := k8sClient.List(ctx, &connectionList, client.MatchingLabels{constants.WorkloadKey: workloadName}); err != nil {
		return nil, fmt.Errorf("list TensorFusionConnection: %w", err)
	}

	for _, connection := range connectionList.Items {
		if connection.Status.WorkerName != "" {
			continue
		}
		usageMapping[connection.Status.WorkerName]++
	}

	var minUsageWorker *tfv1.WorkerStatus
	// Initialize with max int value
	minUsage := int(^uint(0) >> 1)
	for _, workerStatus := range workerStatuses {
		if workerStatus.WorkerPhase == tfv1.WorkerFailed {
			continue
		}
		usage := usageMapping[workerStatus.WorkerName]
		if usage < minUsage {
			minUsage = usage
			minUsageWorker = &workerStatus
		}
	}
	if minUsageWorker == nil {
		return nil, fmt.Errorf("no available worker")
	}
	return minUsageWorker, nil
}
