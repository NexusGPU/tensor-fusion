package worker

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/samber/lo"
	"golang.org/x/exp/rand"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func init() {
	rand.Seed(uint64(time.Now().UnixNano()))
}

type WorkerGenerator struct {
	GpuInfos     *[]config.GpuInfo
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

func (wg *WorkerGenerator) PodTemplateHash(limits tfv1.Resource) (string, error) {
	podTmpl := &corev1.PodTemplate{}
	err := json.Unmarshal(wg.WorkerConfig.PodTemplate.Raw, podTmpl)
	if err != nil {
		return "", fmt.Errorf("failed to unmarshal pod template: %w", err)
	}
	return utils.GetObjectHash(podTmpl, limits), nil
}

func (wg *WorkerGenerator) GenerateWorkerPod(
	gpu *tfv1.GPU,
	generateName string,
	namespace string,
	port int,
	limits tfv1.Resource,
) (*corev1.Pod, string, error) {
	podTmpl := &corev1.PodTemplate{}
	err := json.Unmarshal(wg.WorkerConfig.PodTemplate.Raw, podTmpl)
	if err != nil {
		return nil, "", fmt.Errorf("failed to unmarshal pod template: %w", err)
	}
	podTemplateHash := utils.GetObjectHash(podTmpl, limits)
	spec := podTmpl.Template.Spec
	if spec.NodeSelector == nil {
		spec.NodeSelector = make(map[string]string)
	}
	spec.NodeSelector = gpu.Status.NodeSelector
	spec.Volumes = append(spec.Volumes, corev1.Volume{
		Name: constants.DataVolumeName,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{
				Path: constants.TFDataPath,
			},
		},
	})
	spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts, corev1.VolumeMount{
		Name:        constants.DataVolumeName,
		MountPath:   constants.TFDataPath,
		SubPathExpr: fmt.Sprintf("${%s}", constants.WorkerPodNameEnv),
	})

	info, ok := lo.Find(*wg.GpuInfos, func(info config.GpuInfo) bool {
		return info.FullModelName == gpu.Status.GPUModel
	})
	if !ok {
		return nil, "", fmt.Errorf("gpu info(%s) not found", gpu.Status.GPUModel)
	}

	spec.Containers[0].Env = append(spec.Containers[0].Env, corev1.EnvVar{
		Name:  "NVIDIA_VISIBLE_DEVICES",
		Value: gpu.Status.UUID,
	}, corev1.EnvVar{
		Name:  constants.WorkerPortEnv,
		Value: strconv.Itoa(port),
	}, corev1.EnvVar{
		Name:  constants.WorkerCudaUpLimitTflopsEnv,
		Value: strconv.FormatInt(limits.Tflops.Value(), 10),
	}, corev1.EnvVar{
		Name:  constants.WorkerCudaUpLimitEnv,
		Value: strconv.FormatInt(int64(math.Ceil(float64(limits.Tflops.Value())/float64(info.Fp16TFlops.Value())*100)), 10),
	}, corev1.EnvVar{
		Name: constants.WorkerCudaMemLimitEnv,
		// bytesize
		Value: strconv.FormatInt(limits.Vram.Value(), 10),
	}, corev1.EnvVar{
		Name: constants.WorkerPodNameEnv,
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{
				FieldPath: "metadata.name",
			},
		},
	})
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: generateName,
			Namespace:    namespace,
		},
		Spec: spec,
	}, podTemplateHash, nil
}

func SelectWorker(
	ctx context.Context,
	k8sClient client.Client,
	workload *tfv1.TensorFusionWorkload,
	maxSkew int32,
) (*tfv1.WorkerStatus, error) {
	if len(workload.Status.WorkerStatuses) == 0 {
		return nil, fmt.Errorf("no available worker")
	}
	usageMapping := make(map[string]int, len(workload.Status.WorkerStatuses))
	for _, workerStatus := range workload.Status.WorkerStatuses {
		usageMapping[workerStatus.WorkerName] = 0
	}

	connectionList := tfv1.TensorFusionConnectionList{}
	if err := k8sClient.List(ctx, &connectionList, client.MatchingLabels{constants.WorkloadKey: workload.Name}); err != nil {
		return nil, fmt.Errorf("list TensorFusionConnection: %w", err)
	}

	for _, connection := range connectionList.Items {
		if connection.Status.WorkerName != "" {
			usageMapping[connection.Status.WorkerName]++
		}
	}

	// First find the minimum usage
	minUsage := int(^uint(0) >> 1)
	// Initialize with max int value
	for _, workerStatus := range workload.Status.WorkerStatuses {
		if workerStatus.WorkerPhase == tfv1.WorkerFailed {
			continue
		}
		usage := usageMapping[workerStatus.WorkerName]
		if usage < minUsage {
			minUsage = usage
		}
	}

	// Collect all eligible workers that are within maxSkew of the minimum usage
	var eligibleWorkers []*tfv1.WorkerStatus
	for _, workerStatus := range workload.Status.WorkerStatuses {
		if workerStatus.WorkerPhase == tfv1.WorkerFailed {
			continue
		}
		usage := usageMapping[workerStatus.WorkerName]
		// Worker is eligible if its usage is within maxSkew of the minimum usage
		if usage <= minUsage+int(maxSkew) {
			eligibleWorkers = append(eligibleWorkers, &workerStatus)
		}
	}

	if len(eligibleWorkers) == 0 {
		return nil, fmt.Errorf("no available worker")
	}

	// Choose the worker with the minimum usage among eligible workers
	selectedWorker := eligibleWorkers[0]
	selectedUsage := usageMapping[selectedWorker.WorkerName]
	for i := 1; i < len(eligibleWorkers); i++ {
		worker := eligibleWorkers[i]
		usage := usageMapping[worker.WorkerName]
		if usage < selectedUsage {
			selectedWorker = worker
			selectedUsage = usage
		}
	}

	return selectedWorker, nil
}
