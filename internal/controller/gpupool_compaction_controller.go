package controller

import (
	"context"
	"fmt"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// GPUPoolReconciler reconciles a GPUPool object
type GPUPoolCompactionReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

const defaultCompactionDuration = 1 * time.Minute
const newNodeProtectionDuration = 5 * time.Minute
const manualCompactionReconcileMaxDelay = 3 * time.Second

// to avoid concurrent job for node compaction, make sure the interval
var jobStarted sync.Map

// if it's AutoSelect mode, stop all Pods on it, and let ClusterAutoscaler or Karpenter to delete the node
// if it's Provision mode, stop all Pods on it, and destroy the Node from cloud provider

// Strategy #1: check if any empty node can be deleted (must satisfy 'allocatedCapacity + warmUpCapacity <= currentCapacity - toBeDeletedCapacity') -- Done

// TODO: implement other strategies
// Strategy #2: check if whole Pool can be bin-packing into less nodes, check from low-priority to high-priority nodes one by one, if workloads could be moved to other nodes (using a simulated scheduler), evict it and mark cordoned, let scheduler to re-schedule

// Strategy #3: check if any node can be reduced to 1/2 size. for remaining nodes, check if allocated size < 1/2 * total size, if so, check if can buy smaller instance

// Strategy #4: check if any two same nodes can be merged into one larger node, and make the remained capacity bigger and node number less without violating the capacity constraint and saving the hidden management,license,monitoring costs, potentially schedule more workloads since remaining capacity is single cohesive piece rather than fragments

// Add adds the provide y quantity to the current value. If the current value is zero,
// the format of the quantity will be updated to the format of y.

func (r *GPUPoolCompactionReconciler) checkNodeCompaction(ctx context.Context, pool *tfv1.GPUPool) error {
	log := log.FromContext(ctx)

	// Strategy #1, terminate empty node
	allNodes := &tfv1.GPUNodeList{}
	if err := r.List(ctx, allNodes, client.MatchingLabels(map[string]string{
		fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name): constants.LabelValueTrue,
	})); err != nil {
		return fmt.Errorf("failed to list nodes : %w", err)
	}
	for _, gpuNode := range allNodes.Items {
		// Skip a node that is labeled as NoDisrupt
		if gpuNode.Labels[constants.SchedulingDoNotDisruptLabel] == constants.LabelValueTrue {
			continue
		}

		// Check if node is empty, if not, continue
		var nodeGPUConnection tfv1.TensorFusionConnectionList
		if err := r.List(ctx, &nodeGPUConnection, client.MatchingLabels(map[string]string{
			constants.LabelKeyOwner: gpuNode.Name,
		})); err != nil {
			return fmt.Errorf("failed to list node(%s) GPU connections : %w", gpuNode.Name, err)
		}
		if len(nodeGPUConnection.Items) != 0 {
			// Not is in-using, should not be terminated
			continue
		}
		if gpuNode.Status.Phase != tfv1.TensorFusionGPUNodePhaseRunning {
			// Not running, should not be terminated
			continue
		}
		// Protect new nodes at least 5 minutes to avoid flapping
		if gpuNode.CreationTimestamp.After(time.Now().Add(newNodeProtectionDuration)) {
			continue
		}

		nodeCapTFlops, _ := gpuNode.Status.TotalTFlops.AsInt64()
		nodeCapVRAM, _ := gpuNode.Status.TotalVRAM.AsInt64()

		poolAvailableTFlops, _ := pool.Status.AvailableTFlops.AsInt64()
		poolAvailableVRAM, _ := pool.Status.AvailableVRAM.AsInt64()

		poolWarmUpTFlops, _ := pool.Spec.CapacityConfig.WarmResources.TFlops.AsInt64()
		poolWarmUpVRAM, _ := pool.Spec.CapacityConfig.WarmResources.VRAM.AsInt64()

		couldBeTerminatedByTFlops := poolAvailableTFlops-nodeCapTFlops >= poolWarmUpTFlops
		couldBeTerminatedByVRAM := poolAvailableVRAM-nodeCapVRAM >= poolWarmUpVRAM

		if couldBeTerminatedByTFlops && couldBeTerminatedByVRAM {
			r.Recorder.Eventf(pool, "Compaction", "Node %s is empty and deletion won't impact warm-up capacity, start terminating it", gpuNode.Name)
			if pool.Spec.NodeManagerConfig.ProvisioningMode == tfv1.ProvisioningModeProvisioned {
				// not managed by Kubernetes, managed by TensorFusion, safe to terminate, and finalizer will cause K8S node and related cloud resources to be deleted
				err := r.Delete(ctx, &gpuNode)
				if err != nil {
					return fmt.Errorf("delete node(%s) : %w", gpuNode.Name, err)
				}
			} else {
				// managed by Kubernetes, mark it as destroying, GPUPool capacity should be reduced, and let K8S to delete it
				gpuNode.Status.Phase = constants.PhaseDestroying
				err := r.Update(ctx, &gpuNode)
				if err != nil {
					return fmt.Errorf("update node(%s) : %w", gpuNode.Name, err)
				}

				err = r.Patch(ctx, &corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: gpuNode.Status.KubernetesNodeName,
						Labels: map[string]string{
							constants.NodeDeletionMark: constants.LabelValueTrue,
						},
					},
				}, client.Merge)

				if err != nil {
					return fmt.Errorf("patch node(%s) : %w", gpuNode.Status.KubernetesNodeName, err)
				}
			}
		}
	}

	log.Info("Checking node compaction", "name", pool.Name)
	return nil
}

func (r *GPUPoolCompactionReconciler) getCompactionDuration(ctx context.Context, config *tfv1.NodeManagerConfig) time.Duration {
	log := log.FromContext(ctx)
	if config == nil || config.NodeCompaction == nil || config.NodeCompaction.Period == "" {
		log.Info("Invalid NodeCompaction config, use default value", "duration", defaultCompactionDuration)
		return defaultCompactionDuration
	}
	duration, err := time.ParseDuration(config.NodeCompaction.Period)
	if err != nil {
		log.Error(err, "Can not parse NodeCompaction config, use default value", "input", config.NodeCompaction.Period)
		duration = defaultCompactionDuration
	}
	return duration
}

func (r *GPUPoolCompactionReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	pool := &tfv1.GPUPool{}
	if err := r.Get(ctx, req.NamespacedName, pool); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	needStartCompactionJob := true
	nextDuration := r.getCompactionDuration(ctx, pool.Spec.NodeManagerConfig)

	if lastCompactionTime, loaded := jobStarted.Load(req.String()); loaded {
		// last compaction is less than compaction interval, do nothing, unless its manual triggered
		if time.Now().Before(lastCompactionTime.(time.Time).Add(nextDuration)) {

			if manualCompactionValue, exists := pool.Annotations[constants.TensorFusionPoolManualCompaction]; exists {
				// Parse the annotation value as duration
				if manualTriggerTime, err := time.Parse(time.RFC3339, manualCompactionValue); err == nil {
					// not return empty result, will continue current reconcile logic
					if manualTriggerTime.After(time.Now().Add(manualCompactionReconcileMaxDelay)) {
						log.Info("Manual compaction requested", "name", req.Name)

					} else {
						needStartCompactionJob = false
					}
				} else {
					log.Error(err, "Invalid manual compaction time", "name", req.Name, "time", manualCompactionValue)
					needStartCompactionJob = false
				}
			} else {
				// skip this reconcile, wait for the next ticker
				needStartCompactionJob = false
			}

		}
	}

	if !needStartCompactionJob {
		return ctrl.Result{}, nil
	}

	jobStarted.Store(req.String(), time.Now())
	log.Info("Start compaction check for GPUPool", "name", req.Name)
	defer func() {
		log.Info("Finished compaction check for GPUPool", "name", req.Name)
	}()

	compactionErr := r.checkNodeCompaction(ctx, pool)
	if compactionErr != nil {
		return ctrl.Result{}, compactionErr
	}

	// Next ticker, timer set by user, won't impacted by other reconcile requests
	return ctrl.Result{RequeueAfter: nextDuration}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *GPUPoolCompactionReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		Named("gpupool-compaction").
		WatchesMetadata(&tfv1.GPUPool{}, &handler.EnqueueRequestForObject{}).
		Complete(r)
}
