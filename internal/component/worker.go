package component

import (
	"context"
	"fmt"
	"sort"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/internal/worker"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	WorkerUpdateInProgressAnnotation    = constants.Domain + "/worker-update-in-progress"
	WorkerBatchUpdateLastTimeAnnotation = constants.Domain + "/worker-batch-update-last-time"
)

type Worker struct {
	workloadsToUpdate []*tfv1.TensorFusionWorkload
}

func (w *Worker) GetName() string {
	return "worker"
}

func (w *Worker) DetectConfigChange(pool *tfv1.GPUPool, status *tfv1.PoolComponentStatus) (bool, string, string) {
	oldHash := status.WorkerVersion
	changed, newHash := utils.CompareAndGetObjectHash(oldHash, pool.Spec.ComponentConfig.Worker)
	return changed, newHash, oldHash
}

func (w *Worker) SetConfigHash(status *tfv1.PoolComponentStatus, hash string) {
	status.WorkerVersion = hash
}

func (w *Worker) GetUpdateInProgressInfo(pool *tfv1.GPUPool) string {
	return pool.Annotations[WorkerUpdateInProgressAnnotation]
}

func (w *Worker) SetUpdateInProgressInfo(pool *tfv1.GPUPool, hash string) {
	pool.Annotations[WorkerUpdateInProgressAnnotation] = hash
}

func (w *Worker) GetBatchUpdateLastTimeInfo(pool *tfv1.GPUPool) string {
	return pool.Annotations[WorkerBatchUpdateLastTimeAnnotation]
}

func (w *Worker) SetBatchUpdateLastTimeInfo(pool *tfv1.GPUPool, time string) {
	pool.Annotations[WorkerBatchUpdateLastTimeAnnotation] = time
}

func (w *Worker) GetUpdateProgress(status *tfv1.PoolComponentStatus) int32 {
	return status.WorkerUpdateProgress
}

func (w *Worker) SetUpdateProgress(status *tfv1.PoolComponentStatus, progress int32) {
	status.WorkerUpdateProgress = progress
	status.WorkerConfigSynced = false
	if progress == 100 {
		status.WorkerConfigSynced = true
	}
}

func (w *Worker) GetResourcesInfo(r client.Client, ctx context.Context, pool *tfv1.GPUPool, configHash string) (int, int, bool, error) {
	log := log.FromContext(ctx)
	workloadList := &tfv1.TensorFusionWorkloadList{}
	if err := r.List(ctx, workloadList, client.MatchingLabels(map[string]string{
		constants.GpuPoolKey: pool.Name,
	})); err != nil {
		return 0, 0, false, fmt.Errorf("failed to list workloads : %w", err)
	}

	total := len(workloadList.Items)

	workerGenerator := &worker.WorkerGenerator{WorkerConfig: pool.Spec.ComponentConfig.Worker}
	for _, workload := range workloadList.Items {
		if !workload.DeletionTimestamp.IsZero() {
			total--
			continue
		}
		if workload.Status.PodTemplateHash == "" {
			log.Info("workload in pending status", "name", workload.Name)
			return 0, 0, true, nil
		}
		podTemplateHash, err := workerGenerator.PodTemplateHash(workload.Spec)
		if err != nil {
			return 0, 0, false, fmt.Errorf("failed to get pod template hash: %w", err)
		}
		if workload.Status.PodTemplateHash != podTemplateHash {
			w.workloadsToUpdate = append(w.workloadsToUpdate, &workload)
		}
	}

	sort.Sort(TensorFusionWorkloadByCreationTimestamp(w.workloadsToUpdate))

	return total, total - len(w.workloadsToUpdate), false, nil
}

func (w *Worker) PerformBatchUpdate(r client.Client, ctx context.Context, pool *tfv1.GPUPool, delta int) (bool, error) {
	log := log.FromContext(ctx)
	log.Info("perform batch update", "component", w.GetName())

	for i := range delta {
		workload := w.workloadsToUpdate[i]
		workload.Status.PodTemplateHash = ""
		if err := r.Status().Update(ctx, workload); err != nil {
			return false, fmt.Errorf("failed to update workload status : %w", err)
		}
		log.Info("workload pod template hash in status has changed", "workload", workload.Name)
	}

	return true, nil
}

type TensorFusionWorkloadByCreationTimestamp []*tfv1.TensorFusionWorkload

func (o TensorFusionWorkloadByCreationTimestamp) Len() int      { return len(o) }
func (o TensorFusionWorkloadByCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }
func (o TensorFusionWorkloadByCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(&o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(&o[j].CreationTimestamp)
}
