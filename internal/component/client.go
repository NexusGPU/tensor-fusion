package component

import (
	"context"
	"fmt"
	"sort"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	ClientUpdateInProgressAnnotation    = constants.Domain + "/client-update-in-progress"
	ClientBatchUpdateLastTimeAnnotation = constants.Domain + "/client-batch-update-last-time"
)

type Client struct {
	podsToUpdate []*corev1.Pod
}

func (c *Client) GetName() string {
	return "client"
}

func (c *Client) DetectConfigChange(pool *tfv1.GPUPool, status *tfv1.PoolComponentStatus) (bool, string, string) {
	oldHash := status.ClientVersion
	changed, newHash := utils.CompareAndGetObjectHash(oldHash, pool.Spec.ComponentConfig.Client)
	return changed, newHash, oldHash
}

func (c *Client) SetConfigHash(status *tfv1.PoolComponentStatus, hash string) {
	status.ClientVersion = hash
}

func (c *Client) GetUpdateInProgressInfo(pool *tfv1.GPUPool) string {
	return pool.Annotations[ClientUpdateInProgressAnnotation]
}

func (c *Client) SetUpdateInProgressInfo(pool *tfv1.GPUPool, hash string) {
	pool.Annotations[ClientUpdateInProgressAnnotation] = hash
}

func (c *Client) GetBatchUpdateLastTimeInfo(pool *tfv1.GPUPool) string {
	return pool.Annotations[ClientBatchUpdateLastTimeAnnotation]
}

func (c *Client) SetBatchUpdateLastTimeInfo(pool *tfv1.GPUPool, time string) {
	pool.Annotations[ClientBatchUpdateLastTimeAnnotation] = time
}

func (c *Client) GetUpdateProgress(status *tfv1.PoolComponentStatus) int32 {
	return status.ClientUpdateProgress
}

func (c *Client) SetUpdateProgress(status *tfv1.PoolComponentStatus, progress int32) {
	status.ClientUpdateProgress = progress
	status.ClientConfigSynced = false
	if progress == 100 {
		status.ClientConfigSynced = true
	}
}

func (c *Client) GetResourcesInfo(r client.Client, ctx context.Context, pool *tfv1.GPUPool, configHash string) (int, int, bool, error) {
	podList := &corev1.PodList{}
	if err := r.List(ctx, podList,
		client.MatchingLabels{
			constants.TensorFusionEnabledLabelKey: constants.TrueStringValue,
			constants.GpuPoolKey:                  pool.Name,
		}); err != nil {
		return 0, 0, false, fmt.Errorf("failed to list pods: %w", err)
	}

	total := len(podList.Items)

	for _, pod := range podList.Items {
		if !pod.DeletionTimestamp.IsZero() {
			return 0, 0, true, nil
		}

		if pod.Labels[constants.LabelKeyPodTemplateHash] != configHash {
			c.podsToUpdate = append(c.podsToUpdate, &pod)
		}
	}

	sort.Sort(ClientPodsByCreationTimestamp(c.podsToUpdate))

	return total, total - len(c.podsToUpdate), false, nil
}

func (c *Client) PerformBatchUpdate(r client.Client, ctx context.Context, pool *tfv1.GPUPool, delta int) (bool, error) {
	log := log.FromContext(ctx)

	log.Info("perform batch update", "component", c.GetName())
	for i := range delta {
		pod := c.podsToUpdate[i]
		if err := r.Delete(ctx, pod); err != nil {
			return false, fmt.Errorf("failed to delete pod: %w", err)
		}
	}

	return true, nil
}

type ClientPodsByCreationTimestamp []*corev1.Pod

func (o ClientPodsByCreationTimestamp) Len() int      { return len(o) }
func (o ClientPodsByCreationTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }
func (o ClientPodsByCreationTimestamp) Less(i, j int) bool {
	if o[i].CreationTimestamp.Equal(&o[j].CreationTimestamp) {
		return o[i].Name < o[j].Name
	}
	return o[i].CreationTimestamp.Before(&o[j].CreationTimestamp)
}
