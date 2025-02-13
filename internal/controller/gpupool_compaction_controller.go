package controller

import (
	"context"
	"fmt"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
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

// TODO: need to write a interval in go coroutine to check if node could be compacted like Karpenter, when it's ok to mark as destroying, change the status and trigger a reconcile
// if it's AutoSelect mode, stop all Pods on it, and let ClusterAutoscaler or Karpenter to delete the node
// if it's Provision mode, stop all Pods on it, and destroy the Node from cloud provider

// Strategy #1: check if any empty node can be deleted (must satisfy 'allocatedCapacity + warmUpCapacity <= currentCapacity - toBeDeletedCapacity') -- Done

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
		fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name): "true",
	})); err != nil {
		return fmt.Errorf("failed to list nodes : %w", err)
	}
	for _, gpuNode := range allNodes.Items {
		// Skip a node that is labeled as NoDisrupt
		if gpuNode.Labels[constants.SchedulingDoNotDisruptLabel] == "true" {
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

		// Protect new nodes at least 5 minutes to avoid flapping
		if gpuNode.CreationTimestamp.Time.After(time.Now().Add(newNodeProtectionDuration)) {
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
							constants.NodeDeletionMark: "true",
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

	log.Info("Start compaction check for GPUPool", "name", req.NamespacedName.Name)
	defer func() {
		log.Info("Finished compaction check for GPUPool", "name", req.NamespacedName.Name)
	}()

	pool := &tfv1.GPUPool{}
	if err := r.Get(ctx, req.NamespacedName, pool); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	compactionErr := r.checkNodeCompaction(ctx, pool)
	if compactionErr != nil {
		return ctrl.Result{}, compactionErr
	}

	// Next ticker
	return ctrl.Result{RequeueAfter: r.getCompactionDuration(ctx, pool.Spec.NodeManagerConfig)}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *GPUPoolCompactionReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		Named("gpupool-compaction").
		WatchesMetadata(&tfv1.GPUNode{}, &handler.EnqueueRequestForObject{}).
		Complete(r)
}
