package controller

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
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
	Recorder events.EventRecorder

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

// Future: implement other compaction strategies (e.g. load-based, cost-based)
// Strategy #2: check if whole Pool can be bin-packing into less nodes, check from low-priority to high-priority nodes one by one, if workloads could be moved to other nodes (using a simulated scheduler), evict it and mark cordoned, let scheduler to re-schedule

// Strategy #3: check if any node can be reduced to 1/2 size. for remaining nodes, check if allocated size < 1/2 * total size, if so, check if can buy smaller instance, note that the compaction MUST be GPU level, not node level

// getGPUNodeClaimForCompaction returns the GPUNodeClaim name if the node can be compacted in provision mode.
// Returns empty string if the node should be skipped.
func (r *GPUPoolCompactionReconciler) getGPUNodeClaimForCompaction(ctx context.Context, gpuNode *tfv1.GPUNode) string {
	log := log.FromContext(ctx)
	gpuNodeClaimName := gpuNode.Labels[constants.ProvisionerLabelKey]
	if gpuNodeClaimName == "" {
		log.Info("skip existing nodes managed by other controller when compaction", "node", gpuNode.Name)
		return ""
	}
	gpuNodeClaimObj := &tfv1.GPUNodeClaim{}
	if err := r.Get(ctx, client.ObjectKey{Name: gpuNodeClaimName}, gpuNodeClaimObj); err != nil {
		if errors.IsNotFound(err) {
			log.Info("skip existing nodes managed by other controller when compaction", "node", gpuNode.Name)
			return ""
		}
		log.Error(err, "get gpuNodeClaim failed", "gpuNodeClaimName", gpuNodeClaimName)
		return ""
	}
	if !gpuNodeClaimObj.DeletionTimestamp.IsZero() {
		log.Info("[Warn] GPUNode deleting during compaction loop, this should not happen", "node", gpuNode.Name)
		return ""
	}
	return gpuNodeClaimName
}

type compactionPoolCapacity struct {
	availableTFlops int64
	availableVRAM   int64
	totalTFlops     int64
	totalVRAM       int64
	warmUpTFlops    int64
	warmUpVRAM      int64
	minTFlops       int64
	minVRAM         int64
}

func (r *GPUPoolCompactionReconciler) buildCompactionPoolCapacity(
	ctx context.Context,
	pool *tfv1.GPUPool,
	gpuStore map[types.NamespacedName]*tfv1.GPU,
) compactionPoolCapacity {
	log := log.FromContext(ctx)
	capacity := compactionPoolCapacity{}

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

		capacity.availableTFlops += gpu.Status.Available.Tflops.Value()
		capacity.availableVRAM += gpu.Status.Available.Vram.Value()
		capacity.totalTFlops += gpu.Status.Capacity.Tflops.Value()
		capacity.totalVRAM += gpu.Status.Capacity.Vram.Value()
	}

	if pool.Spec.CapacityConfig != nil && pool.Spec.CapacityConfig.WarmResources != nil {
		capacity.warmUpTFlops = pool.Spec.CapacityConfig.WarmResources.TFlops.Value()
		capacity.warmUpVRAM = pool.Spec.CapacityConfig.WarmResources.VRAM.Value()
	}
	if pool.Spec.CapacityConfig != nil && pool.Spec.CapacityConfig.MinResources != nil {
		capacity.minTFlops = pool.Spec.CapacityConfig.MinResources.TFlops.Value()
		capacity.minVRAM = pool.Spec.CapacityConfig.MinResources.VRAM.Value()
	}

	log.Info("Found latest pool capacity constraints before compaction",
		"pool", pool.Name,
		"warmUpTFlops", capacity.warmUpTFlops,
		"warmUpVRAM", capacity.warmUpVRAM,
		"minTFlops", capacity.minTFlops,
		"minVRAM", capacity.minVRAM,
		"totalTFlops", capacity.totalTFlops,
		"totalVRAM", capacity.totalVRAM)

	return capacity
}

func (c *compactionPoolCapacity) canCompactNode(node *tfv1.GPUNode) bool {
	nodeCapTFlops := node.Status.TotalTFlops.Value()
	nodeCapVRAM := node.Status.TotalVRAM.Value()
	if nodeCapTFlops <= 0 || nodeCapVRAM <= 0 {
		return false
	}

	matchWarmUpCapacityConstraints := c.availableTFlops-nodeCapTFlops >= c.warmUpTFlops &&
		c.availableVRAM-nodeCapVRAM >= c.warmUpVRAM
	matchMinCapacityConstraint := c.totalTFlops-nodeCapTFlops >= c.minTFlops &&
		c.totalVRAM-nodeCapVRAM >= c.minVRAM
	return matchWarmUpCapacityConstraints && matchMinCapacityConstraint
}

func (c *compactionPoolCapacity) consumeNode(node *tfv1.GPUNode) {
	c.availableTFlops -= node.Status.TotalTFlops.Value()
	c.availableVRAM -= node.Status.TotalVRAM.Value()
	c.totalTFlops -= node.Status.TotalTFlops.Value()
	c.totalVRAM -= node.Status.TotalVRAM.Value()
}

func (r *GPUPoolCompactionReconciler) shouldSkipNodeForCompaction(
	gpuNode tfv1.GPUNode,
	nodeToWorker map[string]map[types.NamespacedName]struct{},
) bool {
	k8sNodeName := gpuNode.Name
	switch {
	case gpuNode.Labels[constants.SchedulingDoNotDisruptLabel] == constants.TrueStringValue:
		return true
	case len(nodeToWorker[k8sNodeName]) != 0:
		return true
	case gpuNode.Status.Phase != tfv1.TensorFusionGPUNodePhaseRunning:
		return true
	case gpuNode.CreationTimestamp.After(time.Now().Add(-newNodeProtectionDuration)):
		return true
	default:
		return false
	}
}

func (r *GPUPoolCompactionReconciler) recordProvisionCompaction(
	ctx context.Context,
	pool *tfv1.GPUPool,
	gpuNode *tfv1.GPUNode,
	capacity *compactionPoolCapacity,
	toDeleteGPUNodes *[]string,
) {
	k8sNodeName := gpuNode.Name
	gpuNodeClaimName := r.getGPUNodeClaimForCompaction(ctx, gpuNode)
	if gpuNodeClaimName == "" {
		return
	}

	capacity.consumeNode(gpuNode)
	r.markDeletionNodes[k8sNodeName] = struct{}{}

	log.FromContext(ctx).Info("Empty node can be compacted - provision mode", "node", gpuNode.Name,
		"availableTFlopsAfterCompact", capacity.availableTFlops,
		"availableVRAMAfterCompact", capacity.availableVRAM,
		"warmUpTFlops", capacity.warmUpTFlops,
		"warmUpVRAM", capacity.warmUpVRAM,
		"nodeCapTFlops", gpuNode.Status.TotalTFlops.Value(),
		"nodeCapVRAM", gpuNode.Status.TotalVRAM.Value(),
	)

	*toDeleteGPUNodes = append(*toDeleteGPUNodes, gpuNodeClaimName)
	r.Recorder.Eventf(pool, nil, corev1.EventTypeNormal, "Compaction", "Compacting",
		"Node %s is empty and deletion won't impact warm-up capacity, try terminating it", gpuNode.Name)
}

func (r *GPUPoolCompactionReconciler) recordAutoSelectCompaction(
	ctx context.Context,
	gpuNode *tfv1.GPUNode,
	capacity *compactionPoolCapacity,
) error {
	k8sNodeName := gpuNode.Name
	if err := r.Patch(ctx, &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: gpuNode.Name,
			Labels: map[string]string{
				constants.NodeDeletionMark: constants.TrueStringValue,
			},
		},
	}, client.Merge); err != nil {
		return err
	}

	capacity.consumeNode(gpuNode)
	r.markDeletionNodes[k8sNodeName] = struct{}{}

	log.FromContext(ctx).Info("Empty node can be compacted - auto-select mode", "node", gpuNode.Name,
		"availableTFlopsAfterCompact", capacity.availableTFlops,
		"availableVRAMAfterCompact", capacity.availableVRAM,
		"warmUpTFlops", capacity.warmUpTFlops,
		"warmUpVRAM", capacity.warmUpVRAM,
		"nodeCapTFlops", gpuNode.Status.TotalTFlops.Value(),
		"nodeCapVRAM", gpuNode.Status.TotalVRAM.Value(),
	)
	return nil
}

func (r *GPUPoolCompactionReconciler) persistPendingDeletionNodes(
	ctx context.Context,
	pool *tfv1.GPUPool,
	toDeleteGPUNodes []string,
) error {
	if len(toDeleteGPUNodes) == 0 {
		return nil
	}

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
		if pool.Annotations == nil {
			pool.Annotations = make(map[string]string)
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

	log.FromContext(ctx).Info("GPU node compaction completed, pending deletion nodes: ",
		"name", pool.Name, "nodes", strings.Join(toDeleteGPUNodes, ","))
	return nil
}

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

	gpuStore, nodeToWorker, _ := r.Allocator.GetAllocationInfo()
	capacity := r.buildCompactionPoolCapacity(ctx, pool, gpuStore)

	toDeleteGPUNodes := []string{}
	for _, gpuNode := range allNodes.Items {
		if r.shouldSkipNodeForCompaction(gpuNode, nodeToWorker) || !capacity.canCompactNode(&gpuNode) {
			continue
		}

		if pool.Spec.NodeManagerConfig.ProvisioningMode != tfv1.ProvisioningModeAutoSelect {
			r.recordProvisionCompaction(ctx, pool, &gpuNode, &capacity, &toDeleteGPUNodes)
			continue
		}

		if err := r.recordAutoSelectCompaction(ctx, &gpuNode, &capacity); err != nil {
			log.Error(err, "patch idle node failed", "node", gpuNode.Name)
		}
	}

	return r.persistPendingDeletionNodes(ctx, pool, toDeleteGPUNodes)
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

	pendingGPUNodeStateLock.RLock()
	hasPendingClaim := len(PendingGPUNodeClaim[pool.Name]) > 0
	hasPendingDeletion := len(PendingDeletionGPUNodes[pool.Name]) > 0
	pendingGPUNodeStateLock.RUnlock()
	if !needStartCompactionJob || hasPendingClaim {
		log.Info("Skip compaction because node creating or duration not met", "name", req.Name)
		return ctrl.Result{RequeueAfter: nextDuration}, nil
	}
	if hasPendingDeletion {
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
