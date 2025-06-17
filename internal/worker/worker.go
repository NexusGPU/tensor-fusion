package worker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type WorkerGenerator struct {
	GpuInfos     *[]config.GpuInfo
	WorkerConfig *tfv1.WorkerConfig
}

var ErrNoAvailableWorker = errors.New("no available worker")

func (wg *WorkerGenerator) WorkerPort(pod *corev1.Pod) (int, error) {
	portAnnotation, ok := pod.Annotations[constants.GenPortNumberAnnotation]
	if ok {
		return strconv.Atoi(portAnnotation)
	}

	// Compatible with old version in which no annotation in worker Pod
	portEnv, ok := lo.Find(pod.Spec.Containers[0].Env, func(env corev1.EnvVar) bool {
		return env.Name == constants.WorkerPortEnv
	})

	if !ok {
		return 0, fmt.Errorf("worker port not found in pod %s", pod.Name)
	}

	return strconv.Atoi(portEnv.Value)
}

func (wg *WorkerGenerator) PodTemplateHash(workloadSpec any) (string, error) {
	podTmpl := &corev1.PodTemplate{}
	err := json.Unmarshal(wg.WorkerConfig.PodTemplate.Raw, podTmpl)
	if err != nil {
		return "", fmt.Errorf("failed to unmarshal pod template: %w", err)
	}
	return utils.GetObjectHash(podTmpl, workloadSpec), nil
}

func (wg *WorkerGenerator) GenerateWorkerPod(
	gpus []*tfv1.GPU,
	workloadName string,
	namespace string,
	port int,
	requests tfv1.Resource,
	limits tfv1.Resource,
	podTemplateHash string,
) (*corev1.Pod, string, error) {
	podTmpl := &corev1.PodTemplate{}
	err := json.Unmarshal(wg.WorkerConfig.PodTemplate.Raw, podTmpl)
	if err != nil {
		return nil, "", fmt.Errorf("failed to unmarshal pod template: %w", err)
	}
	spec := podTmpl.Template.Spec

	// all the gpus are on the same node
	spec.NodeSelector = gpus[0].Status.NodeSelector

	spec.Volumes = append(spec.Volumes, corev1.Volume{
		Name: constants.DataVolumeName,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{
				Path: constants.TFDataPath,
				Type: ptr.To(corev1.HostPathDirectoryOrCreate),
			},
		},
	})

	// performance optimization, service link will cause high CPU usage when service number is large
	spec.EnableServiceLinks = ptr.To(false)

	spec.Containers[0].VolumeMounts = append(spec.Containers[0].VolumeMounts, corev1.VolumeMount{
		Name:        constants.DataVolumeName,
		MountPath:   constants.TFDataPath,
		SubPathExpr: fmt.Sprintf("${%s}", constants.PodNameEnv),
	})

	firstGPU := gpus[0]
	info, ok := lo.Find(*wg.GpuInfos, func(info config.GpuInfo) bool {
		return info.FullModelName == firstGPU.Status.GPUModel
	})
	if !ok {
		return nil, "", fmt.Errorf("gpu info(%s) not found", firstGPU.Status.GPUModel)
	}

	gpuUUIDs := lo.Map(gpus, func(gpu *tfv1.GPU, _ int) string {
		return gpu.Status.UUID
	})

	spec.Containers[0].Env = append(spec.Containers[0].Env, corev1.EnvVar{
		Name:  "NVIDIA_VISIBLE_DEVICES",
		Value: strings.Join(gpuUUIDs, ","),
	}, corev1.EnvVar{
		Name:  constants.WorkerPortEnv,
		Value: strconv.Itoa(port),
	}, corev1.EnvVar{
		Name: constants.WorkerCudaUpLimitTflopsEnv,
		Value: func() string {
			tflopsMap := make(map[string]int64)
			for _, gpu := range gpus {
				tflopsMap[gpu.Status.UUID] = limits.Tflops.Value()
			}
			jsonBytes, _ := json.Marshal(tflopsMap)
			return string(jsonBytes)
		}(),
	}, corev1.EnvVar{
		Name: constants.WorkerCudaUpLimitEnv,
		Value: func() string {
			upLimitMap := make(map[string]int64)
			for _, gpu := range gpus {
				upLimitMap[gpu.Status.UUID] = int64(math.Ceil(float64(limits.Tflops.Value()) / float64(info.Fp16TFlops.Value()) * 100))
			}
			jsonBytes, _ := json.Marshal(upLimitMap)
			return string(jsonBytes)
		}(),
	}, corev1.EnvVar{
		Name: constants.WorkerCudaMemLimitEnv,
		// bytesize
		Value: func() string {
			memLimitMap := make(map[string]int64)
			for _, gpu := range gpus {
				memLimitMap[gpu.Status.UUID] = limits.Vram.Value()
			}
			jsonBytes, _ := json.Marshal(memLimitMap)
			return string(jsonBytes)
		}(),
	}, corev1.EnvVar{
		Name: constants.PodNameEnv,
		ValueFrom: &corev1.EnvVarSource{
			FieldRef: &corev1.ObjectFieldSelector{
				FieldPath: "metadata.name",
			},
		},
	}, corev1.EnvVar{
		Name:  constants.WorkloadNameEnv,
		Value: workloadName,
	})
	workerLabels := map[string]string{
		constants.LabelComponent: constants.ComponentWorker,
	}
	if podTmpl.Template.Labels != nil {
		maps.Copy(workerLabels, podTmpl.Template.Labels)
	}
	workerAnnotations := map[string]string{
		constants.TFLOPSRequestAnnotation: requests.Tflops.String(),
		constants.TFLOPSLimitAnnotation:   limits.Tflops.String(),
		constants.VRAMRequestAnnotation:   requests.Vram.String(),
		constants.VRAMLimitAnnotation:     limits.Vram.String(),
		constants.GenPortNumberAnnotation: strconv.Itoa(port),
	}
	if podTmpl.Template.Annotations != nil {
		for k, v := range podTmpl.Template.Annotations {
			workerAnnotations[k] = v
		}
	}
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("%s-tf-worker-", workloadName),
			Namespace:    namespace,
			Labels:       workerLabels,
			Annotations:  workerAnnotations,
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
		return nil, ErrNoAvailableWorker
	}

	usageMapping := lo.SliceToMap(workload.Status.WorkerStatuses, func(status tfv1.WorkerStatus) (string, int) {
		return status.WorkerName, 0
	})

	connectionList := tfv1.TensorFusionConnectionList{}
	if err := k8sClient.List(ctx, &connectionList, client.MatchingLabels{constants.WorkloadKey: workload.Name}); err != nil {
		return nil, fmt.Errorf("list TensorFusionConnection: %w", err)
	}

	lo.ForEach(connectionList.Items, func(conn tfv1.TensorFusionConnection, _ int) {
		if conn.Status.WorkerName != "" {
			usageMapping[conn.Status.WorkerName]++
		}
	})

	// filter out failed workers and get the usage of available workers
	activeWorkers := lo.Filter(workload.Status.WorkerStatuses, func(status tfv1.WorkerStatus, _ int) bool {
		return status.WorkerPhase != tfv1.WorkerFailed
	})

	if len(activeWorkers) == 0 {
		return nil, fmt.Errorf("no available worker")
	}

	// find the worker with the minimum usage
	minUsage := lo.MinBy(activeWorkers, func(a, b tfv1.WorkerStatus) bool {
		return usageMapping[a.WorkerName] < usageMapping[b.WorkerName]
	})
	minUsageValue := usageMapping[minUsage.WorkerName]

	// collect all workers within the minimum usage plus maxSkew range
	eligibleWorkers := lo.Filter(activeWorkers, func(status tfv1.WorkerStatus, _ int) bool {
		return usageMapping[status.WorkerName] <= minUsageValue+int(maxSkew)
	})

	// select the worker with the minimum usage among eligible workers
	selectedWorker := lo.MinBy(eligibleWorkers, func(a, b tfv1.WorkerStatus) bool {
		return usageMapping[a.WorkerName] < usageMapping[b.WorkerName]
	})

	return &selectedWorker, nil
}
