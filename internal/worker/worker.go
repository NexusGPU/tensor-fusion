package worker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/samber/lo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type WorkerGenerator struct {
	WorkerConfig     *tfv1.WorkerConfig
	HypervisorConfig *tfv1.HypervisorConfig
}

var ErrNoAvailableWorker = errors.New("no available worker")

func (wg *WorkerGenerator) PodTemplateHash(workloadSpec any) (string, error) {
	return utils.GetObjectHash(
		wg.WorkerConfig,
		workloadSpec,
	), nil
}

func (wg *WorkerGenerator) GenerateWorkerPod(
	ctx context.Context,
	workload *tfv1.TensorFusionWorkload,
) (*v1.Pod, error) {
	podTmpl := &v1.PodTemplate{}
	err := json.Unmarshal(wg.WorkerConfig.PodTemplate.Raw, podTmpl)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal pod template: %w", err)
	}
	spec := podTmpl.Template.Spec

	utils.AddWorkerConfAfterTemplate(ctx, &spec, wg.WorkerConfig, wg.HypervisorConfig, workload)

	// performance optimization, service link will cause high CPU usage when service number is large
	spec.EnableServiceLinks = ptr.To(false)
	spec.SchedulerName = constants.SchedulerName

	// Add labels to identify this pod as part of the workload
	labels, annotations := utils.AppendTFWorkerLabelsAndAnnotationsAfterTemplate(podTmpl, workload)

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("%s-tf-worker-", workload.Name),
			Namespace:    workload.Namespace,
			Labels:       labels,
			Annotations:  annotations,
		},
		Spec: spec,
	}, nil
}

func SelectWorker(
	ctx context.Context,
	k8sClient client.Client,
	workload *tfv1.TensorFusionWorkload,
	maxSkew int32,
) (*tfv1.WorkerStatus, error) {
	workerList := &v1.PodList{}
	if err := k8sClient.List(ctx, workerList, client.MatchingLabels{constants.WorkloadKey: workload.Name}); err != nil {
		return nil, fmt.Errorf("can not list worker pods: %w", err)
	}
	usageMapping := lo.SliceToMap(workerList.Items, func(pod v1.Pod) (string, int) {
		return pod.Name, 0
	})

	connectionList := &tfv1.TensorFusionConnectionList{}
	if err := k8sClient.List(ctx, connectionList, client.MatchingLabels{constants.WorkloadKey: workload.Name}); err != nil {
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
		return nil, ErrNoAvailableWorker
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

	return &tfv1.WorkerStatus{
		WorkerName:      selectedWorker.Name,
		WorkerIp:        selectedWorker.Status.PodIP,
		ResourceVersion: selectedWorker.ResourceVersion,
	}, nil
}
