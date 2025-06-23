package worker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type WorkerGenerator struct {
	GpuInfos     *[]config.GpuInfo
	WorkerConfig *tfv1.WorkerConfig
}

var ErrNoAvailableWorker = errors.New("no available worker")

func WorkerPort(pod *corev1.Pod) (int, error) {
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
	workload *tfv1.TensorFusionWorkload,
	podTemplateHash string,
) (*corev1.Pod, string, error) {
	podTmpl := &corev1.PodTemplate{}
	err := json.Unmarshal(wg.WorkerConfig.PodTemplate.Raw, podTmpl)
	if err != nil {
		return nil, "", fmt.Errorf("failed to unmarshal pod template: %w", err)
	}
	spec := podTmpl.Template.Spec
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

	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("%s-tf-worker-", workload.Name),
			Namespace:    workload.Namespace,
			Labels:       podTmpl.Template.Labels,
			Annotations:  map[string]string{},
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

	workerList := v1.PodList{}
	k8sClient.List(ctx, &workerList, client.MatchingLabels{constants.WorkloadKey: workload.Name})
	usageMapping := lo.SliceToMap(workerList.Items, func(pod v1.Pod) (string, int) {
		return pod.Name, 0
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
	activeWorkers := lo.Filter(workerList.Items, func(pod v1.Pod, _ int) bool {
		return pod.Status.Phase != v1.PodFailed && pod.Status.Phase != v1.PodUnknown && pod.Status.Phase != v1.PodPending
	})

	if len(activeWorkers) == 0 {
		return nil, fmt.Errorf("no available worker")
	}

	// find the worker with the minimum usage
	minUsage := lo.MinBy(activeWorkers, func(a, b v1.Pod) bool {
		return usageMapping[a.Name] < usageMapping[b.Name]
	})
	minUsageValue := usageMapping[minUsage.Name]

	// collect all workers within the minimum usage plus maxSkew range
	eligibleWorkers := lo.Filter(activeWorkers, func(pod v1.Pod, _ int) bool {
		return usageMapping[pod.Name] <= minUsageValue+int(maxSkew)
	})

	// select the worker with the minimum usage among eligible workers
	selectedWorker := lo.MinBy(eligibleWorkers, func(a, b v1.Pod) bool {
		return usageMapping[a.Name] < usageMapping[b.Name]
	})

	workerPort, err := WorkerPort(&selectedWorker)
	if err != nil {
		return nil, fmt.Errorf("failed to get worker port: %w", err)
	}

	return &tfv1.WorkerStatus{
		WorkerName:      selectedWorker.Name,
		WorkerIp:        selectedWorker.Status.PodIP,
		WorkerPort:      workerPort,
		ResourceVersion: selectedWorker.ResourceVersion,
	}, nil
}
