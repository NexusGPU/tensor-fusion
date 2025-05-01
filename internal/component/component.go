package component

import (
	"context"
	"fmt"
	"math"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type Interface interface {
	GetName() string
	DetectConfigChange(pool *tfv1.GPUPool, status *tfv1.PoolComponentStatus) (bool, string, string)
	SetConfigHash(status *tfv1.PoolComponentStatus, hash string)
	GetUpdateInProgressInfo(pool *tfv1.GPUPool) string
	SetUpdateInProgressInfo(pool *tfv1.GPUPool, hash string)
	GetBatchUpdateLastTimeInfo(pool *tfv1.GPUPool) string
	SetBatchUpdateLastTimeInfo(pool *tfv1.GPUPool, time string)
	GetUpdateProgress(status *tfv1.PoolComponentStatus) int32
	SetUpdateProgress(status *tfv1.PoolComponentStatus, progress int32)
	GetResourcesInfo(r client.Client, ctx context.Context, pool *tfv1.GPUPool, hash string) (int, int, bool, error)
	PerformBatchUpdate(r client.Client, ctx context.Context, pool *tfv1.GPUPool, delta int) (bool, error)
}

func ManageUpdate(r client.Client, ctx context.Context, pool *tfv1.GPUPool, component Interface) (*ctrl.Result, error) {
	log := log.FromContext(ctx)

	autoUpdate, batchInterval := getUpdatePolicy(pool)
	newStatus := pool.Status.ComponentStatus.DeepCopy()

	changed, configHash, oldHash := component.DetectConfigChange(pool, newStatus)
	if changed {
		log.Info("component configuration changed", "component", component.GetName(), "old hash", oldHash, "new hash", configHash)
		component.SetConfigHash(newStatus, configHash)
		component.SetUpdateProgress(newStatus, 0)
		if oldHash == "" || !autoUpdate {
			return nil, patchComponentStatus(r, ctx, pool, newStatus)
		}
		if pool.Annotations == nil {
			pool.Annotations = map[string]string{}
		}
		patch := client.MergeFrom(pool.DeepCopy())
		component.SetUpdateInProgressInfo(pool, configHash)
		component.SetBatchUpdateLastTimeInfo(pool, "")
		if err := r.Patch(ctx, pool, patch); err != nil {
			return nil, fmt.Errorf("failed to patch pool: %w", err)
		}
	} else {
		if !autoUpdate || component.GetUpdateInProgressInfo(pool) != configHash {
			return nil, nil
		}
		if timeInfo := component.GetBatchUpdateLastTimeInfo(pool); len(timeInfo) != 0 {
			lastBatchUpdateTime, err := time.Parse(time.RFC3339, timeInfo)
			if err != nil {
				return nil, err
			}
			nextBatchUpdateTime := lastBatchUpdateTime.Add(batchInterval)
			if now := time.Now(); now.Before(nextBatchUpdateTime) {
				log.Info("next batch update time not yet reached", "now", now, "nextBatchUpdateTime", nextBatchUpdateTime)
				return &ctrl.Result{RequeueAfter: nextBatchUpdateTime.Sub(now)}, nil
			}
			log.Info("next batch update time reached", "BatchUpdateTime", nextBatchUpdateTime)
		}
	}

	totalSize, updatedSize, recheck, err := component.GetResourcesInfo(r, ctx, pool, configHash)
	if err != nil {
		return nil, err
	} else if recheck {
		return &ctrl.Result{RequeueAfter: constants.PendingRequeueDuration}, err
	} else if totalSize <= 0 {
		return nil, nil
	}

	batchPercentage := pool.Spec.NodeManagerConfig.NodePoolRollingUpdatePolicy.BatchPercentage
	updateProgress := component.GetUpdateProgress(newStatus)
	delta, newUpdateProgress, currentBatchIndex := calculateDesiredUpdatedDelta(totalSize, updatedSize, batchPercentage, updateProgress)
	component.SetUpdateProgress(newStatus, newUpdateProgress)
	log.Info("update in progress", "component", component.GetName(), "hash", configHash,
		"updateProgress", newUpdateProgress, "totalSize", totalSize, "updatedSize", updatedSize,
		"batchPercentage", batchPercentage, "currentBatchIndex", currentBatchIndex, "delta", delta)

	var ctrlResult *ctrl.Result
	if delta == 0 {
		patch := client.MergeFrom(pool.DeepCopy())
		newUpdateProgress = min((currentBatchIndex+1)*batchPercentage, 100)
		component.SetUpdateProgress(newStatus, newUpdateProgress)
		if newUpdateProgress != 100 {
			component.SetBatchUpdateLastTimeInfo(pool, time.Now().Format(time.RFC3339))
			interval := max(batchInterval, constants.PendingRequeueDuration)
			ctrlResult = &ctrl.Result{RequeueAfter: interval}
			log.Info("current batch update has completed", "progress", newUpdateProgress, "currentBatchIndex", currentBatchIndex, "nextUpdateTime", time.Now().Add(interval))
		} else {
			component.SetUpdateInProgressInfo(pool, "")
			component.SetBatchUpdateLastTimeInfo(pool, "")
			log.Info("all batch update has completed", "component", component.GetName(), "hash", configHash)
		}
		if err := r.Patch(ctx, pool, patch); err != nil {
			return nil, fmt.Errorf("failed to patch pool: %w", err)
		}
	} else if delta > 0 {
		recheck, err := component.PerformBatchUpdate(r, ctx, pool, int(delta))
		if err != nil {
			return nil, err
		} else if recheck {
			ctrlResult = &ctrl.Result{RequeueAfter: constants.PendingRequeueDuration}
		}
	}

	return ctrlResult, patchComponentStatus(r, ctx, pool, newStatus)
}

func patchComponentStatus(r client.Client, ctx context.Context, pool *tfv1.GPUPool, newStatus *tfv1.PoolComponentStatus) error {
	patch := client.MergeFrom(pool.DeepCopy())
	pool.Status.ComponentStatus = *newStatus
	if err := r.Status().Patch(ctx, pool, patch); err != nil {
		return fmt.Errorf("failed to patch pool status: %w", err)
	}
	return nil
}

func getUpdatePolicy(pool *tfv1.GPUPool) (bool, time.Duration) {
	autoUpdate := false
	batchInterval := time.Duration(600) * time.Second

	if pool.Spec.NodeManagerConfig != nil {
		updatePolicy := pool.Spec.NodeManagerConfig.NodePoolRollingUpdatePolicy
		if updatePolicy != nil {
			if updatePolicy.AutoUpdate != nil {
				autoUpdate = *updatePolicy.AutoUpdate
			}

			duration, err := time.ParseDuration(updatePolicy.BatchInterval)
			if err == nil {
				batchInterval = duration
			}
		}
	}

	return autoUpdate, batchInterval
}

func calculateDesiredUpdatedDelta(total int, updatedSize int, batchPercentage int32, updateProgress int32) (int32, int32, int32) {
	batchSize := getValueFromPercent(int(batchPercentage), total, true)
	var delta, desiredSize, currentBatchIndex int32
	newUpdateProgress := updateProgress
	for {
		currentBatchIndex = newUpdateProgress / batchPercentage
		desiredSize = min((currentBatchIndex+1)*int32(batchSize), int32(total))
		delta = desiredSize - int32(updatedSize)
		// if rolling udpate policy changed or new nodes were added during update, we need to update progress
		if delta < 0 {
			newUpdateProgress = min(newUpdateProgress+batchPercentage, 100)
		} else {
			break
		}
	}

	return delta, newUpdateProgress, currentBatchIndex
}

func getValueFromPercent(percent int, total int, roundUp bool) int {
	if roundUp {
		return int(math.Ceil(float64(percent) * (float64(total)) / 100))
	} else {
		return int(math.Floor(float64(percent) * (float64(total)) / 100))
	}
}
