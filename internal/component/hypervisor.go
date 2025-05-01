package component

import (
	"context"
	"fmt"
	"sort"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	HypervisorUpdateInProgressAnnotation    = constants.Domain + "/hypervisor-update-in-progress"
	HypervisorBatchUpdateLastTimeAnnotation = constants.Domain + "/hypervisor-batch-update-last-time"
)

type Hypervisor struct {
	nodesToUpdate []*tfv1.GPUNode
}

func (h *Hypervisor) GetName() string {
	return "hypervisor"
}

func (h *Hypervisor) DetectConfigChange(pool *tfv1.GPUPool, status *tfv1.PoolComponentStatus) (bool, string, string) {
	oldHash := status.HypervisorVersion
	changed, newHash := utils.CompareAndGetObjectHash(oldHash, pool.Spec.ComponentConfig.Hypervisor)
	return changed, newHash, oldHash
}

func (h *Hypervisor) SetConfigHash(status *tfv1.PoolComponentStatus, hash string) {
	status.HypervisorVersion = hash
}

func (h *Hypervisor) GetUpdateInProgressInfo(pool *tfv1.GPUPool) string {
	return pool.Annotations[HypervisorUpdateInProgressAnnotation]
}

func (h *Hypervisor) SetUpdateInProgressInfo(pool *tfv1.GPUPool, hash string) {
	pool.Annotations[HypervisorUpdateInProgressAnnotation] = hash
}

func (h *Hypervisor) GetBatchUpdateLastTimeInfo(pool *tfv1.GPUPool) string {
	return pool.Annotations[HypervisorBatchUpdateLastTimeAnnotation]
}

func (h *Hypervisor) SetBatchUpdateLastTimeInfo(pool *tfv1.GPUPool, time string) {
	pool.Annotations[HypervisorBatchUpdateLastTimeAnnotation] = time
}

func (h *Hypervisor) GetUpdateProgress(status *tfv1.PoolComponentStatus) int32 {
	return status.HyperVisorUpdateProgress
}

func (h *Hypervisor) SetUpdateProgress(status *tfv1.PoolComponentStatus, progress int32) {
	status.HyperVisorUpdateProgress = progress
	status.HypervisorConfigSynced = false
	if progress == 100 {
		status.HypervisorConfigSynced = true
	}
}

func (h *Hypervisor) GetResourcesInfo(r client.Client, ctx context.Context, pool *tfv1.GPUPool, configHash string) (int, int, bool, error) {
	log := log.FromContext(ctx)

	nodeList := &tfv1.GPUNodeList{}
	if err := r.List(ctx, nodeList, client.MatchingLabels(map[string]string{
		fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name): "true",
	})); err != nil {
		return 0, 0, false, fmt.Errorf("failed to list nodes: %w", err)
	}

	total := len(nodeList.Items)

	for _, node := range nodeList.Items {
		if !node.DeletionTimestamp.IsZero() {
			total--
			continue
		}
		if node.Status.Phase == tfv1.TensorFusionGPUNodePhasePending {
			log.Info("node in pending status", "name", node.Name)
			return 0, 0, true, nil
		}
		key := client.ObjectKey{
			Namespace: utils.CurrentNamespace(),
			Name:      fmt.Sprintf("hypervisor-%s", node.Name),
		}
		pod := &corev1.Pod{}
		err := r.Get(ctx, key, pod)
		if errors.IsNotFound(err) ||
			pod.Labels[constants.LabelKeyPodTemplateHash] != configHash {
			h.nodesToUpdate = append(h.nodesToUpdate, &node)
		}
	}

	// TODO: sort by creation time desc, need to adjust test
	sort.Sort(GPUNodeByCreationTimestamp(h.nodesToUpdate))

	return total, total - len(h.nodesToUpdate), false, nil
}

func (h *Hypervisor) PerformBatchUpdate(r client.Client, ctx context.Context, pool *tfv1.GPUPool, delta int) (bool, error) {
	log := log.FromContext(ctx)

	log.Info("perform batch update", "component", h.GetName())
	for i := range delta {
		node := h.nodesToUpdate[i]
		if node.Status.Phase != tfv1.TensorFusionGPUNodePhasePending {
			node.Status.Phase = tfv1.TensorFusionGPUNodePhasePending
			if err := r.Status().Update(ctx, node); err != nil {
				return false, fmt.Errorf("failed to update node status : %w", err)
			}
			log.Info("node phase has been updated to pending", "node", node.Name)
		}
	}

	return false, nil
}

type GPUNodeByCreationTimestamp []*tfv1.GPUNode

func (o GPUNodeByCreationTimestamp) Len() int      { return len(o) }
func (o GPUNodeByCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }
func (o GPUNodeByCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(&o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(&o[j].CreationTimestamp)
}
