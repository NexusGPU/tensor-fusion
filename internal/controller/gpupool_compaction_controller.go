package controller

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/retry"
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

	Allocator *gpuallocator.GpuAllocator

	markDeletionNodes map[string]struct{}
}

var defaultCompactionDuration = 1 * time.Minute
var newNodeProtectionDuration = 5 * time.Minute

const manualCompactionReconcileMaxDelay = 3 * time.Second

// to avoid concurrent job for node compaction, make sure the interval
var jobStarted sync.Map

// if it's AutoSelect mode, stop all Pods on it, and let ClusterAutoscaler or Karpenter to delete the node
// if it's Provision mode, stop all Pods on it, and destroy the Node from cloud provider

// Strategy #1: check if any empty node can be deleted (must satisfy 'allocatedCapacity + warmUpCapacity <= currentCapacity - toBeDeletedCapacity') -- Done

// TODO: implement other strategies
// Strategy #2: check if whole Pool can be bin-packing into less nodes, check from low-priority to high-priority nodes one by one, if workloads could be moved to other nodes (using a simulated scheduler), evict it and mark cordoned, let scheduler to re-schedule

// Strategy #3: check if any node can be reduced to 1/2 size. for remaining nodes, check if allocated size < 1/2 * total size, if so, check if can buy smaller instance, note that the compaction MUST be GPU level, not node level

// Strategy #4: check if any two same nodes can be merged into one larger node, and make the remained capacity bigger and node number less without violating the capacity constraint and saving the hidden management,license,monitoring costs, potentially schedule more workloads since remaining capacity is single cohesive piece rather than fragments
func (r *GPUPoolCompactionReconciler) checkNodeCompaction(ctx context.Context, pool *tfv1.GPUPool) error {
	log := log.FromContext(ctx)

	// Strategy #1, terminate empty node
	allNodes := &tfv1.GPUNodeList{}
	if err := r.List(ctx, allNodes, client.MatchingLabels(map[string]string{
		fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name): constants.TrueStringValue,
	})); err != nil {
		return fmt.Errorf("failed to list nodes : %w", err)
	}

	// Use latest in memory data to calculate available resources
	gpuStore, nodeToWorker, _ := r.Allocator.GetAllocationInfo()

	poolAvailableTFlops := int64(0)
	poolAvailableVRAM := int64(0)
	poolTotalTFlops := int64(0)
	poolTotalVRAM := int64(0)

	for _, gpu := range gpuStore {
		if !gpu.DeletionTimestamp.IsZero() || gpu.Labels[constants.GpuPoolKey] != pool.Name ||
			gpu.Status.UsedBy != tfv1.UsedByTensorFusion || len(gpu.Status.NodeSelector) == 0 {
			continue
		}

		k8sNodeName := gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
		if _, ok := r.markDeletionNodes[k8sNodeName]; ok {
			log.V(4).Info("skip node already marked for deletion when calculation capacity", "node", k8sNodeName)
			continue
		}

		availableTFlops, _ := gpu.Status.Available.Tflops.AsInt64()
		poolAvailableTFlops += availableTFlops
		availableVRAM, _ := gpu.Status.Available.Vram.AsInt64()
		poolAvailableVRAM += availableVRAM

		tflops, _ := gpu.Status.Capacity.Tflops.AsInt64()
		poolTotalTFlops += tflops
		vram, _ := gpu.Status.Capacity.Vram.AsInt64()
		poolTotalVRAM += vram
	}

	poolWarmUpTFlops, _ := pool.Spec.CapacityConfig.WarmResources.TFlops.AsInt64()
	poolWarmUpVRAM, _ := pool.Spec.CapacityConfig.WarmResources.VRAM.AsInt64()

	poolMinTFlops, _ := pool.Spec.CapacityConfig.MinResources.TFlops.AsInt64()
	poolMinVRAM, _ := pool.Spec.CapacityConfig.MinResources.VRAM.AsInt64()

	log.Info("Found latest pool capacity constraints before compaction", "pool", pool.Name, "warmUpTFlops", poolWarmUpTFlops, "warmUpVRAM", poolWarmUpVRAM, "minTFlops", poolMinTFlops, "minVRAM", poolMinVRAM, "totalTFlops", poolTotalTFlops, "totalVRAM", poolTotalVRAM)

	toDeleteGPUNodes := []string{}
	for _, gpuNode := range allNodes.Items {
		// Skip a node that is labeled as NoDisrupt
		if gpuNode.Labels[constants.SchedulingDoNotDisruptLabel] == constants.TrueStringValue {
			continue
		}

		// Check if node is empty, if not, continue
		k8sNodeName := gpuNode.Name
		if len(nodeToWorker[k8sNodeName]) != 0 {
			// Node is in-use, should not be terminated
			continue
		}
		if gpuNode.Status.Phase != tfv1.TensorFusionGPUNodePhaseRunning {
			// Node is not running, should not be terminated
			continue
		}
		// Protect new nodes at least 5 minutes to avoid flapping
		if gpuNode.CreationTimestamp.After(time.Now().Add(-newNodeProtectionDuration)) {
			continue
		}

		nodeCapTFlops, _ := gpuNode.Status.TotalTFlops.AsInt64()
		nodeCapVRAM, _ := gpuNode.Status.TotalVRAM.AsInt64()
		if nodeCapTFlops <= 0 || nodeCapVRAM <= 0 {
			continue
		}

		matchWarmUpCapacityConstraints := poolAvailableTFlops-nodeCapTFlops >= poolWarmUpTFlops &&
			poolAvailableVRAM-nodeCapVRAM >= poolWarmUpVRAM
		matchMinCapacityConstraint := poolTotalTFlops-nodeCapTFlops >= poolMinTFlops &&
			poolTotalVRAM-nodeCapVRAM >= poolMinVRAM

		if matchWarmUpCapacityConstraints && matchMinCapacityConstraint {
			if pool.Spec.NodeManagerConfig.ProvisioningMode != tfv1.ProvisioningModeAutoSelect {
				// not managed by Kubernetes, managed by TensorFusion, safe to terminate, and finalizer will cause K8S node and related cloud resources to be deleted
				gpuNodeClaimName := gpuNode.Labels[constants.ProvisionerLabelKey]
				if gpuNodeClaimName == "" {
					log.Info("skip existing nodes managed by other controller when compaction", "node", gpuNode.Name)
					continue
				}
				gpuNodeClaimObj := &tfv1.GPUNodeClaim{}
				if err := r.Get(ctx, client.ObjectKey{Name: gpuNodeClaimName}, gpuNodeClaimObj); err != nil {
					if errors.IsNotFound(err) {
						log.Info("skip existing nodes managed by other controller when compaction", "node", gpuNode.Name)
						continue
					}
					log.Error(err, "get gpuNodeClaim failed", "gpuNodeClaimName", gpuNodeClaimName)
					continue
				}
				// already deleting
				if !gpuNodeClaimObj.DeletionTimestamp.IsZero() {
					log.Info("[Warn] GPUNode deleting during compaction loop, this should not happen", "node", gpuNode.Name)
					continue
				}

				poolAvailableTFlops -= nodeCapTFlops
				poolAvailableVRAM -= nodeCapVRAM
				poolTotalTFlops -= nodeCapTFlops
				poolTotalVRAM -= nodeCapVRAM
				r.markDeletionNodes[k8sNodeName] = struct{}{}

				log.Info("Empty node can be compacted - provision mode", "node", gpuNode.Name,
					"availableTFlopsAfterCompact", poolAvailableTFlops,
					"availableVRAMAfterCompact", poolAvailableVRAM,
					"warmUpTFlops", poolWarmUpTFlops,
					"warmUpVRAM", poolWarmUpVRAM,
					"nodeCapTFlops", nodeCapTFlops,
					"nodeCapVRAM", nodeCapVRAM,
				)

				toDeleteGPUNodes = append(toDeleteGPUNodes, gpuNodeClaimName)
				r.Recorder.Eventf(pool, corev1.EventTypeNormal, "Compaction",
					"Node %s is empty and deletion won't impact warm-up capacity, try terminating it", gpuNode.Name,
				)
			} else {
				// managed by Kubernetes, mark it as destroying, GPUPool capacity should be reduced, and let K8S to delete it
				if err := r.Patch(ctx, &corev1.Node{
					ObjectMeta: metav1.ObjectMeta{
						Name: gpuNode.Name,
						Labels: map[string]string{
							constants.NodeDeletionMark: constants.TrueStringValue,
						},
					},
				}, client.Merge); err != nil {
					log.Error(err, "patch idle node failed", "node", gpuNode.Name)
					continue
				}

				poolAvailableTFlops -= nodeCapTFlops
				poolAvailableVRAM -= nodeCapVRAM
				poolTotalTFlops -= nodeCapTFlops
				poolTotalVRAM -= nodeCapVRAM
				r.markDeletionNodes[k8sNodeName] = struct{}{}

				log.Info("Empty node can be compacted - auto-select mode", "node", gpuNode.Name,
					"availableTFlopsAfterCompact", poolAvailableTFlops,
					"availableVRAMAfterCompact", poolAvailableVRAM,
					"warmUpTFlops", poolWarmUpTFlops,
					"warmUpVRAM", poolWarmUpVRAM,
					"nodeCapTFlops", nodeCapTFlops,
					"nodeCapVRAM", nodeCapVRAM,
				)
			}
		}
	}

	if len(toDeleteGPUNodes) > 0 {
		pendingGPUNodeStateLock.Lock()
		PendingDeletionGPUNodes[pool.Name] = toDeleteGPUNodes
		pendingGPUNodeStateLock.Unlock()
		err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
			if err := r.Get(ctx, client.ObjectKey{Name: pool.Name}, pool); err != nil {
				if errors.IsNotFound(err) {
					return nil
				}
				return err
			}
			pool.Annotations[constants.LastSyncTimeAnnotationKey] = time.Now().Format(time.RFC3339)
			return r.Patch(ctx, pool, client.Merge)
		})
		if err != nil {
			pendingGPUNodeStateLock.Lock()
			PendingDeletionGPUNodes[pool.Name] = []string{}
			pendingGPUNodeStateLock.Unlock()
			return err
		}
		log.Info("GPU node compaction completed, pending deletion nodes: ",
			"name", pool.Name, "nodes", strings.Join(toDeleteGPUNodes, ","))
	}
	return nil
}

func (r *GPUPoolCompactionReconciler) getCompactionDuration(ctx context.Context, config *tfv1.NodeManagerConfig) time.Duration {
	log := log.FromContext(ctx)
	if config == nil || config.NodeCompaction == nil || config.NodeCompaction.Period == "" {
		log.V(4).Info("empty node compaction config, use default value", "duration", defaultCompactionDuration)
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

	if !pool.DeletionTimestamp.IsZero() {
		log.Info("GPUPool is being deleted, skip compaction", "name", req.Name)
		return ctrl.Result{}, nil
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
					if manualTriggerTime.After(time.Now().Add(-manualCompactionReconcileMaxDelay)) {
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

	if !needStartCompactionJob || len(PendingGPUNodeClaim[pool.Name]) > 0 {
		log.Info("Skip compaction because node creating or duration not met", "name", req.Name)
		return ctrl.Result{RequeueAfter: nextDuration}, nil
	}
	if len(PendingDeletionGPUNodes[pool.Name]) > 0 {
		log.Info("Skip compaction because node deleting in progress", "name", req.Name)
		return ctrl.Result{RequeueAfter: nextDuration}, nil
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
	r.markDeletionNodes = make(map[string]struct{})
	return ctrl.NewControllerManagedBy(mgr).
		Named("gpupool-compaction").
		WatchesMetadata(&tfv1.GPUPool{}, &handler.EnqueueRequestForObject{}).
		Complete(r)
}

func SetTestModeCompactionPeriod() {
	defaultCompactionDuration = 700 * time.Millisecond
	newNodeProtectionDuration = 1000 * time.Millisecond
}
