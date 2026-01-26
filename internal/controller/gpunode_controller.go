/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"os"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/scheduler/expander"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/go-logr/logr"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	schedulingcorev1 "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

// GPUNodeReconciler reconciles a GPUNode object
type GPUNodeReconciler struct {
	client.Client
	Scheme                               *runtime.Scheme
	Recorder                             record.EventRecorder
	Allocator                            *gpuallocator.GpuAllocator
	Expander                             *expander.NodeExpander
	CompatibleWithNvidiaContainerToolkit bool
}

// For test or troubleshooting purpose, using env var to force GPUNode state
var forceGPUNodeStateMap = make(map[string]tfv1.TensorFusionGPUNodePhase, 4)

func init() {
	if state := os.Getenv("DEBUG_FORCE_GPUNODE_STATE"); state != "" {
		log := log.FromContext(context.Background())
		err := json.Unmarshal([]byte(state), &forceGPUNodeStateMap)
		if err != nil {
			log.Error(err, "failed to unmarshal DEBUG_FORCE_GPUNODE_STATE")
		} else {
			log.Info("DEBUG_FORCE_GPUNODE_STATE set, will force GPUNode state to", "state", state)
		}
	}
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpunodes,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpunodes/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpunodes/finalizers,verbs=update
// +kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=karpenter.sh,resources=*,verbs=*

// Reconcile GPU nodes
func (r *GPUNodeReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	log.Info("Reconciling GPUNode", "name", req.Name)
	defer func() {
		log.Info("Finished reconciling GPUNode", "name", req.Name)
	}()

	node := &tfv1.GPUNode{}
	if err := r.Get(ctx, req.NamespacedName, node); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	shouldReturn, err := utils.HandleFinalizer(ctx, node, r.Client, func(ctx context.Context, node *tfv1.GPUNode) (bool, error) {
		if node.Status.Phase != tfv1.TensorFusionGPUNodePhaseDestroying {
			node.Status.Phase = tfv1.TensorFusionGPUNodePhaseDestroying
			if err := r.Status().Update(ctx, node); err != nil {
				return false, err
			}
		}
		metrics.RemoveNodeMetrics(node.Name)
		return true, nil
	})
	if err != nil {
		return ctrl.Result{}, err
	}
	if shouldReturn || !node.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	poolName := utils.ExtractPoolNameFromNodeLabel(node)
	if poolName == "" {
		log.Error(nil, "failed to get pool name", "node", node.Name)
		return ctrl.Result{}, nil
	}

	poolObj := &tfv1.GPUPool{}
	err = r.Get(ctx, client.ObjectKey{Name: poolName}, poolObj)
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to get tensor-fusion pool, pool: %s", poolName)
	}

	// Check if the Kubernetes node exists; if not, the GPUNode should delete itself.
	coreNode := &corev1.Node{}
	err = r.Get(ctx, client.ObjectKey{Name: node.Name}, coreNode)
	if errors.IsNotFound(err) || !coreNode.DeletionTimestamp.IsZero() {
		// The Kubernetes node does not exist or deleting, delete the GPUNode
		log.Info("Kubernetes node does not exist or deleting, deleting GPUNode",
			"kubernetesNodeName", node.Name)
		if err := r.Delete(ctx, node); err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to delete GPUNode after Kubernetes node was deleted: %w", err)
		}
		// Return early since we've deleted the resource
		return ctrl.Result{}, nil
	}

	// Support a some special case: when OS image shipped with Nvidia Driver, and Nvidia Operator override it
	// Need wait device-plugin to be ready and K8S Node GPU resource to be allocatable,
	// so that to avoid potential version mismatch issues from nvidia container toolkit mounted libs
	if os.Getenv(constants.RunHypervisorUtilGPUAllocatable) == constants.TrueStringValue {
		if len(coreNode.Status.Allocatable) > 0 {
			if _, ok := coreNode.Status.Allocatable[constants.NvidiaGPUKey]; !ok {
				log.Info("GPU resource not allocatable, wait allocatable to be set", "node", node.Name)
				return ctrl.Result{RequeueAfter: constants.StatusCheckInterval}, nil
			}
		} else {
			log.Info("GPU resource not allocatable, node still in init phase", "node", node.Name)
			return ctrl.Result{RequeueAfter: constants.StatusCheckInterval}, nil
		}
	}

	// Check if the node is undergoing NVIDIA GPU driver upgrade
	if isNodeUnderGPUDriverUpgrade(coreNode) {
		log.Info("Node is undergoing GPU driver upgrade, skip reconciling",
			"node", node.Name,
			"upgradeState", coreNode.Labels[constants.NvidiaGPUDriverUpgradeStateLabel])
		return ctrl.Result{RequeueAfter: constants.StatusCheckInterval}, nil
	}

	// Initialize TotalGPUs from device plugin if not yet set
	if node.Status.TotalGPUs == 0 {
		// Try to get GPU count from device plugin allocatable resources
		vendor, err := r.resolveNodeVendor(ctx, node)
		if err != nil {
			log.Error(err, "failed to resolve node vendor", "node", node.Name)
			return ctrl.Result{RequeueAfter: constants.StatusCheckInterval}, nil
		}

		var gpuCount int32
		switch vendor {
		case constants.AcceleratorVendorNvidia:
			if qty, ok := coreNode.Status.Allocatable[constants.NvidiaGPUKey]; ok {
				gpuCount = int32(qty.Value())
			}
		case constants.AcceleratorVendorAMD:
			// AMD device plugin uses amd.com/gpu
			if qty, ok := coreNode.Status.Allocatable[constants.AmdGPUKey]; ok {
				gpuCount = int32(qty.Value())
			}
		default:
			log.Info("Unknown vendor, cannot detect GPU count from device plugin", "vendor", vendor, "node", node.Name)
		}

		if gpuCount > 0 {
			log.Info("Initializing TotalGPUs from device plugin", "node", node.Name, "vendor", vendor, "count", gpuCount)
			node.Status.TotalGPUs = gpuCount
			node.Status.ManagedGPUs = gpuCount
			// Set phase to Pending if not already set (required by CRD validation)
			if node.Status.Phase == "" {
				node.Status.Phase = tfv1.TensorFusionGPUNodePhasePending
			}
			if err := r.Status().Update(ctx, node); err != nil {
				return ctrl.Result{}, fmt.Errorf("failed to update GPUNode TotalGPUs: %w", err)
			}
			// Requeue to proceed with hypervisor creation
			return ctrl.Result{Requeue: true}, nil
		}

		log.Info("GPU on this node has not been discovered, wait next loop", "node", node.Name)
		return ctrl.Result{RequeueAfter: constants.StatusCheckInterval}, nil
	}

	hypervisorName, err := r.reconcileHypervisorPod(ctx, node, poolObj, coreNode)
	if err != nil {
		return ctrl.Result{}, err
	}
	// pod deleted or deleting, wait next reconcile
	if hypervisorName == "" {
		return ctrl.Result{RequeueAfter: constants.PendingRequeueDuration}, nil
	}

	// Check if hypervisor is running well, if so, set as running status
	err = r.checkStatusAndUpdateVirtualCapacity(ctx, hypervisorName, node, poolObj, coreNode)
	if errors.IsNotFound(err) {
		log.Info("Hypervisor pod not found, requeue", "hypervisorName", hypervisorName)
		return ctrl.Result{Requeue: true}, nil
	}
	return ctrl.Result{}, err
}

func (r *GPUNodeReconciler) checkStatusAndUpdateVirtualCapacity(
	ctx context.Context, hypervisorName string, node *tfv1.GPUNode, poolObj *tfv1.GPUPool, coreNode *corev1.Node,
) error {
	pod := &corev1.Pod{}
	fetchErr := r.Get(ctx, client.ObjectKey{Name: hypervisorName, Namespace: utils.CurrentNamespace()}, pod)
	if fetchErr != nil {
		return fetchErr
	}

	// Reconcile GPUNode status with hypervisor pod status, when changed
	hypervisorNotReady := pod.Status.Phase != corev1.PodRunning || !utils.IsPodConditionTrue(pod.Status.Conditions, corev1.PodReady)

	forceGPUNodePhase, exists := forceGPUNodeStateMap[node.Name]
	if exists {
		// Use env var forced value
		switch forceGPUNodePhase {
		case tfv1.TensorFusionGPUNodePhasePending:
			hypervisorNotReady = true
		case tfv1.TensorFusionGPUNodePhaseRunning:
			hypervisorNotReady = false
		default:
			// ignore other values
		}
	}

	if hypervisorNotReady {
		if node.Status.Phase != tfv1.TensorFusionGPUNodePhasePending {
			node.Status.Phase = tfv1.TensorFusionGPUNodePhasePending
			err := r.Status().Update(ctx, node)
			if err != nil {
				return fmt.Errorf("failed to update GPU node status to pending: %w", err)
			}
			metrics.SetNodeMetrics(node, poolObj, nil)
		}

		err := r.syncStatusToGPUDevices(ctx, node, tfv1.TensorFusionGPUPhasePending)
		if err != nil {
			return err
		}

		return nil
	} else {
		gpuModels, err := gpuallocator.RefreshGPUNodeCapacity(ctx, r.Client, node, poolObj, r.Allocator, coreNode)
		if err != nil {
			return err
		}
		if len(gpuModels) == 0 {
			log.FromContext(ctx).Info("GPU models not found, skip update", "node", node.Name)
			return nil
		}

		// update metrics to get historical allocation line chart and trending
		metrics.SetNodeMetrics(node, poolObj, gpuModels)

		// check if need to set GPUNodeClaim to Bound phase after hypervisor pod is running
		if node.Labels != nil && node.Labels[constants.ProvisionerLabelKey] != "" {
			provisionerName := node.Labels[constants.ProvisionerLabelKey]
			gpuNodeClaim := &tfv1.GPUNodeClaim{}
			if err := r.Get(ctx, client.ObjectKey{Name: provisionerName}, gpuNodeClaim); err != nil {
				if errors.IsNotFound(err) {
					log.FromContext(ctx).Info("GPUNodeClaim not found but provisioner is not empty, orphan GPUNode",
						"name", provisionerName)
					return nil
				}
				return fmt.Errorf("failed to get GPUNodeClaim: %w", err)
			}
			if gpuNodeClaim.Status.Phase != tfv1.GPUNodeClaimBound {
				gpuNodeClaim.Status.Phase = tfv1.GPUNodeClaimBound
				if err := r.Status().Update(ctx, gpuNodeClaim); err != nil {
					return fmt.Errorf("failed to update GPUNodeClaim to bound state: %w", err)
				}
			}
		}

		err = r.syncStatusToGPUDevices(ctx, node, tfv1.TensorFusionGPUPhaseRunning)
		if err != nil {
			return err
		}

		if coreNode.Labels != nil && coreNode.Labels[constants.KarpenterExpansionLabel] != "" {
			r.Expander.RemoveInFlightNode(coreNode.Labels[constants.KarpenterExpansionLabel])
		}
		return nil
	}
}

func (r *GPUNodeReconciler) syncStatusToGPUDevices(ctx context.Context, node *tfv1.GPUNode, state tfv1.TensorFusionGPUPhase) error {
	gpuList, err := r.fetchAllOwnedGPUDevices(ctx, node)
	if err != nil {
		return err
	}

	for _, gpu := range gpuList {
		if gpu.Status.Phase != state {
			patch := client.MergeFrom(gpu.DeepCopy())
			gpu.Status.Phase = state
			if err := r.Status().Patch(ctx, &gpu, patch); err != nil {
				return fmt.Errorf("failed to patch GPU device status to %s: %w", state, err)
			}
		}
	}
	return nil
}

func (r *GPUNodeReconciler) fetchAllOwnedGPUDevices(ctx context.Context, node *tfv1.GPUNode) ([]tfv1.GPU, error) {
	gpuList := &tfv1.GPUList{}
	if err := r.List(ctx, gpuList, client.MatchingLabels{constants.LabelKeyOwner: node.Name}); err != nil {
		return nil, fmt.Errorf("failed to list GPUs: %w", err)
	}
	return gpuList.Items, nil
}

func (r *GPUNodeReconciler) reconcileHypervisorPod(
	ctx context.Context,
	node *tfv1.GPUNode,
	pool *tfv1.GPUPool,
	k8sNode *corev1.Node,
) (string, error) {
	log := log.FromContext(ctx)

	if pool.Spec.ComponentConfig == nil || pool.Spec.ComponentConfig.Hypervisor == nil {
		return "", fmt.Errorf("missing hypervisor config")
	}

	key := client.ObjectKey{
		Namespace: utils.CurrentNamespace(),
		Name:      fmt.Sprintf("tf-hypervisor-%s", node.Name),
	}

	// Get current hypervisor pod once to avoid duplicate API calls
	currentPod := &corev1.Pod{}
	podExists := false
	if err := r.Get(ctx, key, currentPod); err != nil {
		if !errors.IsNotFound(err) {
			return "", fmt.Errorf("failed to get current hypervisor pod: %w", err)
		}
	} else {
		podExists = true
	}

	// Check hypervisor prerequisites (e.g., device plugin pod for NVIDIA)
	// Must be checked continuously even if hypervisor exists to maintain consistency
	vendor, err := r.resolveNodeVendor(ctx, node)
	if err != nil {
		return "", fmt.Errorf("failed to resolve node vendor: %w", err)
	}

	handler := r.getVendorHandler(vendor)
	if handler != nil {
		log.V(1).Info("checking hypervisor prerequisites", "node", node.Name, "vendor", vendor)
		prereqs, err := handler.CheckHypervisorPrerequisites(ctx, r, node.Name)
		if err != nil {
			return "", fmt.Errorf("failed to check hypervisor prerequisites: %w", err)
		}

		if !prereqs.Ready {
			if podExists {
				log.Info("hypervisor prerequisites not ready, deleting existing hypervisor pod",
					"node", node.Name,
					"pod", currentPod.Name,
					"podPhase", currentPod.Status.Phase)
				if err := r.Delete(ctx, currentPod); err != nil {
					return "", fmt.Errorf("failed to delete existing hypervisor pod: %w", err)
				}
				log.Info("deleted hypervisor pod due to prerequisites not ready", "node", node.Name, "pod", key.Name)
			} else {
				log.V(1).Info("hypervisor prerequisites not ready, no existing pod to delete", "node", node.Name)
			}
			// Return error to trigger requeue, ensuring eventual creation when prerequisites are met
			return "", fmt.Errorf("waiting for hypervisor prerequisites (device plugin)")
		}
		log.V(1).Info("hypervisor prerequisites are ready", "node", node.Name)
	}

	// If pod exists and prerequisites are satisfied, verify its status
	if podExists {
		// If node is already running, no need to recreate
		if node.Status.Phase == tfv1.TensorFusionGPUNodePhaseRunning {
			return key.Name, nil
		}

		oldHash := currentPod.Labels[constants.LabelKeyPodTemplateHash]
		if !currentPod.DeletionTimestamp.IsZero() {
			log.Info("hypervisor pod is still being deleted", "name", key.Name, "hash", oldHash)
			return "", nil
		}

		newHash := utils.GetObjectHash(pool.Spec.ComponentConfig.Hypervisor)
		if utils.IsPodStopped(currentPod) || oldHash != newHash {
			if err := r.Delete(ctx, currentPod); err != nil {
				return "", fmt.Errorf("failed to delete old hypervisor pod: %w", err)
			}
			log.Info("old hypervisor pod deleted", "name", currentPod.Name, "oldHash", oldHash, "newHash", newHash)
			return "", nil
		}
		return key.Name, nil
	}

	// Create new hypervisor pod
	log.Info("hypervisor pod not found, creating new one", "node", node.Name)
	ready, err := r.ensureDriverProbeReady(ctx, node, pool)
	if err != nil {
		return "", fmt.Errorf("failed to ensure driver probe ready: %w", err)
	}
	if !ready {
		log.Info("driver probe job not ready yet, requeue hypervisor creation", "node", node.Name)
		return "", nil
	}

	if err := r.createHypervisorPod(ctx, key, node, pool, k8sNode); err != nil {
		if errors.IsAlreadyExists(err) {
			log.Info("hypervisor pod already exists, skip creation", "node", node.Name)
			return "", nil
		}
		return "", fmt.Errorf("failed to create hypervisor pod: %w", err)
	}
	return key.Name, nil
}

func (r *GPUNodeReconciler) createHypervisorPod(
	ctx context.Context,
	key client.ObjectKey,
	node *tfv1.GPUNode,
	pool *tfv1.GPUPool,
	k8sNode *corev1.Node,
) error {
	log := log.FromContext(ctx)

	var spec corev1.PodSpec
	var templateLabels map[string]string
	var templateAnnotations map[string]string

	// unmarshal pod template if provided, otherwise use empty spec
	if pool.Spec.ComponentConfig.Hypervisor.PodTemplate != nil && len(pool.Spec.ComponentConfig.Hypervisor.PodTemplate.Raw) > 0 {
		podTmpl := &corev1.PodTemplate{}
		err := json.Unmarshal(pool.Spec.ComponentConfig.Hypervisor.PodTemplate.Raw, podTmpl)
		if err != nil {
			return fmt.Errorf("failed to unmarshal pod template: %w", err)
		}
		spec = podTmpl.Template.Spec
		templateLabels = podTmpl.Template.Labels
		templateAnnotations = podTmpl.Template.Annotations
	} else {
		// Use default empty spec when PodTemplate is not provided
		spec = corev1.PodSpec{}
		templateLabels = make(map[string]string)
		templateAnnotations = make(map[string]string)
	}

	if spec.NodeSelector == nil {
		spec.NodeSelector = make(map[string]string)
	}
	spec.EnableServiceLinks = ptr.To(false)
	spec.NodeName = node.Name
	spec.DNSPolicy = corev1.DNSClusterFirstWithHostNet

	vendor, err := getMatchedVendor(k8sNode, pool.Spec.NodeManagerConfig)
	if err != nil {
		return fmt.Errorf("failed to get matched vendor: %w %s", err, k8sNode.Name)
	}
	log.Info("adding hypervisor manifest for GPU node", "node", node.Name, "vendor", vendor)
	utils.AddTFHypervisorConfAfterTemplate(ctx, &spec, pool, vendor, r.CompatibleWithNvidiaContainerToolkit)

	// add vendor-specific env vars for multi-vendor support
	if node.Labels != nil && node.Labels[constants.AcceleratorLabelVendor] != "" {
		vendor := node.Labels[constants.AcceleratorLabelVendor]
		acceleratorLibPath := constants.GetAcceleratorLibPath(vendor)

		envVars := []corev1.EnvVar{
			{
				Name:  constants.TFHardwareVendorEnv,
				Value: vendor,
			},
			{
				Name:  constants.TFAcceleratorLibPathEnv,
				Value: acceleratorLibPath,
			},
		}

		// Add vendor-specific environment variables
		if vendor == constants.AcceleratorVendorAMD {
			// ROCm-specific environment
			envVars = append(envVars,
				corev1.EnvVar{
					Name:  "ROCM_PATH",
					Value: "/opt/rocm",
				},
				corev1.EnvVar{
					Name:  "PATH",
					Value: "/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
				},
				corev1.EnvVar{
					Name:  "LD_LIBRARY_PATH",
					Value: constants.TFDataPath + "/lib:/opt/rocm/lib:/usr/local/lib",
				},
			)
		} else {
			// Non-AMD vendors
			envVars = append(envVars, corev1.EnvVar{
				Name:  "LD_LIBRARY_PATH",
				Value: constants.TFDataPath + "/lib:/usr/local/lib",
			})
		}

		spec.Containers[0].Env = utils.AppendEnvVarsIfNotExists(spec.Containers[0].Env, envVars...)
		log.Info("added vendor env vars to hypervisor pod", "node", node.Name, "vendor", vendor, "libPath", acceleratorLibPath)
	}

	// add scheduling config for hypervisor
	if pool.Spec.SchedulingConfigTemplate != nil {
		schedulingConfigTemplate := &tfv1.SchedulingConfigTemplate{}
		if err := r.Get(ctx, client.ObjectKey{Name: *pool.Spec.SchedulingConfigTemplate}, schedulingConfigTemplate); err == nil {
			if schedulingConfigTemplate.Spec.Hypervisor != nil {
				if cfg, err := json.Marshal(schedulingConfigTemplate.Spec.Hypervisor); err == nil {
					extraLabelsJson, err := json.Marshal(config.GetGlobalConfig().MetricsExtraPodLabels)
					if err != nil {
						return fmt.Errorf("invalid metricsExtraPodLabels config, not valid map: %w", err)
					}
					spec.Containers[0].Env = append(spec.Containers[0].Env, corev1.EnvVar{
						Name:  constants.HypervisorSchedulingConfigEnv,
						Value: string(cfg),
					}, corev1.EnvVar{
						Name:  constants.HypervisorMetricsFormatEnv,
						Value: config.GetGlobalConfig().MetricsFormat,
					}, corev1.EnvVar{
						Name:  constants.HypervisorMetricsExtraLabelsEnv,
						Value: string(extraLabelsJson),
					})
				}
			}
		}
	}

	// compose the final pod and set tolerations and controller reference
	newHash := utils.GetObjectHash(pool.Spec.ComponentConfig.Hypervisor)
	newPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      key.Name,
			Namespace: key.Namespace,
			Labels: func() map[string]string {
				mergedLabels := make(map[string]string)
				maps.Copy(mergedLabels, templateLabels)
				mergedLabels[fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name)] = "true"
				mergedLabels[constants.LabelKeyPodTemplateHash] = newHash
				mergedLabels[constants.LabelComponent] = constants.ComponentHypervisor
				return mergedLabels
			}(),
			Annotations: templateAnnotations,
		},
		Spec: spec,
	}

	if newPod.Spec.Tolerations == nil {
		newPod.Spec.Tolerations = []corev1.Toleration{}
	}
	// tolerate the nodes that used by TensorFusion system
	newPod.Spec.Tolerations = append(newPod.Spec.Tolerations, corev1.Toleration{
		Key:      constants.NodeUsedByTaintKey,
		Operator: corev1.TolerationOpExists,
	})
	err = controllerutil.SetControllerReference(node, newPod, r.Scheme)
	if err != nil {
		return fmt.Errorf("failed to set controller reference for hypervisor: %w", err)
	}
	// also set node owned by k8s node to allow Karpenter to delete the node while hypervisor exists
	if err := controllerutil.SetOwnerReference(k8sNode, newPod, r.Scheme); err != nil {
		return fmt.Errorf("failed to set owner reference for hypervisor: %w", err)
	}

	// create hypervisor pod
	if err = r.Create(ctx, newPod); err != nil {
		return fmt.Errorf("failed to create hypervisor pod: %w", err)
	}
	log.Info("hypervisor pod created", "name", key.Name, "hash", newHash)
	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *GPUNodeReconciler) SetupWithManager(mgr ctrl.Manager) error {
	// Index Pod by spec.nodeName for efficient queries
	if err := mgr.GetFieldIndexer().IndexField(context.Background(), &corev1.Pod{}, "spec.nodeName", func(obj client.Object) []string {
		pod := obj.(*corev1.Pod)
		return []string{pod.Spec.NodeName}
	}); err != nil {
		return err
	}

	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.GPUNode{}).
		Named("gpunode").
		Watches(&corev1.Node{}, handler.EnqueueRequestsFromMapFunc(
			func(ctx context.Context, obj client.Object) []reconcile.Request {
				return []reconcile.Request{
					{NamespacedName: client.ObjectKey{Name: obj.GetName()}},
				}
			})).
		Watches(
			&corev1.Pod{},
			handler.EnqueueRequestsFromMapFunc(r.mapDevicePluginPodToGPUNode),
			builder.WithPredicates(r.devicePluginPodPredicate()),
		).
		Owns(&batchv1.Job{}).
		Owns(&corev1.Pod{}).
		Owns(&tfv1.GPU{}).
		Complete(r)
}

func (r *GPUNodeReconciler) ensureDriverProbeReady(ctx context.Context, node *tfv1.GPUNode, pool *tfv1.GPUPool) (bool, error) {
	vendor, err := r.resolveNodeVendor(ctx, node)
	if err != nil {
		return false, fmt.Errorf("failed to resolve node vendor: %w", err)
	}

	handler := r.getVendorHandler(vendor)
	if handler == nil {
		return true, nil
	}

	jobTemplate, err := handler.ComposeDriverProbeJob(ctx, node, pool)
	if err != nil {
		return false, fmt.Errorf("failed to compose driver probe job: %w", err)
	}
	if jobTemplate == nil {
		return true, nil
	}

	return r.reconcileDriverProbeJob(ctx, node, jobTemplate)
}

func (r *GPUNodeReconciler) reconcileDriverProbeJob(ctx context.Context, node *tfv1.GPUNode, jobTemplate *batchv1.Job) (bool, error) {
	jobTemplate.SetNamespace(utils.CurrentNamespace())
	jobTemplate.SetName(getDriverProbeJobName(node.Name))
	if jobTemplate.Labels == nil {
		jobTemplate.Labels = map[string]string{}
	}
	jobTemplate.Labels[constants.LabelComponent] = constants.ComponentDriverProbe

	log := log.FromContext(ctx)
	currentJob := &batchv1.Job{}
	if err := r.Get(ctx, client.ObjectKeyFromObject(jobTemplate), currentJob); err != nil {
		if errors.IsNotFound(err) {
			return r.createDriverProbeJob(ctx, node, jobTemplate, log)
		}
		return false, fmt.Errorf("failed to get driver probe job: %w", err)
	}

	return r.checkDriverProbeJobStatus(currentJob, log)
}

func (r *GPUNodeReconciler) createDriverProbeJob(ctx context.Context, node *tfv1.GPUNode, jobTemplate *batchv1.Job, log logr.Logger) (bool, error) {
	if err := ctrl.SetControllerReference(node, jobTemplate, r.Scheme); err != nil {
		return false, fmt.Errorf("failed to set owner reference for driver probe: %w", err)
	}
	if err := r.Create(ctx, jobTemplate); err != nil {
		return false, fmt.Errorf("failed to create driver probe job: %w", err)
	}
	log.Info("driver probe job created", "node", node.Name, "job", jobTemplate.Name)
	return false, nil
}

func (r *GPUNodeReconciler) checkDriverProbeJobStatus(job *batchv1.Job, log logr.Logger) (bool, error) {
	if job.Status.Succeeded > 0 {
		return true, nil
	}

	if job.Status.Failed > 0 {
		log.Error(fmt.Errorf("driver probe job failed"), "job status indicates failure",
			"job", job.Name,
			"failedAttempts", job.Status.Failed)

		if job.Status.Failed >= *job.Spec.BackoffLimit {
			r.Recorder.Eventf(job, corev1.EventTypeWarning, "DriverProbeFailed",
				"Driver probe job failed after %d attempts", job.Status.Failed)
			return false, fmt.Errorf("driver probe job %s exhausted all retry attempts", job.Name)
		}
		return false, nil
	}

	log.V(1).Info("driver probe job still running", "job", job.Name)
	return false, nil
}

func (r *GPUNodeReconciler) resolveNodeVendor(ctx context.Context, node *tfv1.GPUNode) (string, error) {
	log := log.FromContext(ctx)

	// Get the GPUPool that owns this GPUNode
	poolName := ""
	for _, ownerRef := range node.OwnerReferences {
		if ownerRef.Kind == "GPUPool" {
			poolName = ownerRef.Name
			break
		}
	}

	if poolName == "" {
		log.Error(fmt.Errorf("no GPUPool owner found"), "cannot resolve vendor", "node", node.Name)
		return "", fmt.Errorf("GPUNode %s has no GPUPool owner reference", node.Name)
	}

	// Fetch the GPUPool to get vendor configuration
	pool := &tfv1.GPUPool{}
	if err := r.Get(ctx, client.ObjectKey{Name: poolName}, pool); err != nil {
		return "", fmt.Errorf("failed to get GPUPool %s: %w", poolName, err)
	}

	// Check node labels for vendor info (e.g., tensor-fusion.ai/hardware-vendor)
	if vendor, ok := node.Labels[constants.AcceleratorLabelVendor]; ok && vendor != "" {
		log.V(1).Info("resolved vendor from node label", "node", node.Name, "vendor", vendor)
		return vendor, nil
	}

	// Fallback to GPUPool's defaultVendor
	if pool.Spec.NodeManagerConfig != nil && pool.Spec.NodeManagerConfig.DefaultVendor != "" {
		vendor := pool.Spec.NodeManagerConfig.DefaultVendor
		log.V(1).Info("resolved vendor from GPUPool defaultVendor", "node", node.Name, "vendor", vendor)
		return vendor, nil
	}

	// Ultimate fallback to NVIDIA for backward compatibility
	log.V(1).Info("no vendor specified, defaulting to NVIDIA", "node", node.Name)
	return constants.AcceleratorVendorNvidia, nil
}

// HypervisorPrerequisites contains the prerequisites for creating a hypervisor pod
type HypervisorPrerequisites struct {
	Ready bool // Whether the prerequisites are ready
}

type vendorSpecificHandler interface {
	// ComposeDriverProbeJob composes a driver probe job for the vendor
	ComposeDriverProbeJob(ctx context.Context, node *tfv1.GPUNode, pool *tfv1.GPUPool) (*batchv1.Job, error)

	// CheckHypervisorPrerequisites checks and returns prerequisites for creating hypervisor pod
	CheckHypervisorPrerequisites(ctx context.Context, r *GPUNodeReconciler, nodeName string) (*HypervisorPrerequisites, error)
}

func (r *GPUNodeReconciler) getVendorHandler(vendor string) vendorSpecificHandler {
	if vendor == "" {
		return nil
	}
	if vendor == constants.AcceleratorVendorNvidia {
		return &nvidiaHandler{
			compatibleWithNvidiaContainerToolkit: r.CompatibleWithNvidiaContainerToolkit,
		}
	}
	return nil
}

type nvidiaHandler struct {
	compatibleWithNvidiaContainerToolkit bool
}

func (n *nvidiaHandler) ComposeDriverProbeJob(ctx context.Context, node *tfv1.GPUNode, pool *tfv1.GPUPool) (*batchv1.Job, error) {
	// DriverProbeJob is not needed for NVIDIA
	return nil, nil
}

func (n *nvidiaHandler) CheckHypervisorPrerequisites(ctx context.Context, r *GPUNodeReconciler, nodeName string) (*HypervisorPrerequisites, error) {
	log := log.FromContext(ctx)

	if !n.compatibleWithNvidiaContainerToolkit {
		log.V(1).Info("compatible-with-nvidia-container-toolkit is disabled, skipping device plugin check",
			"node", nodeName)
		return &HypervisorPrerequisites{Ready: true}, nil
	}

	pod, err := n.findDevicePluginPod(ctx, r, nodeName)
	if err != nil {
		return nil, err
	}
	if pod == nil {
		log.Info("device plugin pod not found, hypervisor prerequisites not ready",
			"node", nodeName)
		return &HypervisorPrerequisites{Ready: false}, nil
	}

	// Only consider Running pods as ready to avoid premature hypervisor creation
	if pod.Status.Phase != corev1.PodRunning {
		log.Info("device plugin pod not running, hypervisor prerequisites not ready",
			"node", nodeName,
			"pod", pod.Name,
			"phase", pod.Status.Phase)
		return &HypervisorPrerequisites{Ready: false}, nil
	}

	log.V(1).Info("device plugin pod is running, hypervisor prerequisites ready",
		"node", nodeName,
		"pod", pod.Name)
	return &HypervisorPrerequisites{Ready: true}, nil
}

func (n *nvidiaHandler) findDevicePluginPod(ctx context.Context, r *GPUNodeReconciler, nodeName string) (*corev1.Pod, error) {
	log := log.FromContext(ctx)

	podList := &corev1.PodList{}
	if err := r.List(ctx, podList,
		client.InNamespace(config.GetGPUOperatorNamespace()),
		client.MatchingLabels{
			"app": "nvidia-device-plugin-daemonset",
		},
		client.MatchingFields{
			"spec.nodeName": nodeName,
		}); err != nil {
		log.Error(err, "failed to list nvidia-device-plugin pods")
		return nil, err
	}

	candidatePods := podList.Items
	if len(candidatePods) == 0 {
		log.V(1).Info("no nvidia-device-plugin pod found on node, waiting", "node", nodeName)
		return nil, nil
	}

	var bestPod *corev1.Pod
	// Prefer Running pods, fallback to Pending pods, skip deleting/failed pods
	for i := range candidatePods {
		pod := &candidatePods[i]
		if !pod.DeletionTimestamp.IsZero() {
			continue
		}

		switch pod.Status.Phase {
		case corev1.PodRunning:
			return pod, nil
		case corev1.PodPending:
			if bestPod == nil {
				bestPod = pod
			}
		}
	}

	if bestPod != nil {
		log.Info("nvidia-device-plugin pod is pending, waiting for it to be ready", "node", nodeName, "pod", bestPod.Name)
		return bestPod, nil
	}

	// All pods are in Failed/Unknown state or being deleted
	log.Info("nvidia-device-plugin pods exist but none are healthy", "node", nodeName, "count", len(candidatePods))
	return nil, nil
}

func getDriverProbeJobName(gpuNodeName string) string {
	return fmt.Sprintf("driver-probe-%s", gpuNodeName)
}

// isNodeUnderGPUDriverUpgrade checks if the node is undergoing NVIDIA GPU driver upgrade
func isNodeUnderGPUDriverUpgrade(node *corev1.Node) bool {
	if node == nil || node.Labels == nil {
		return false
	}
	state := node.Labels[constants.NvidiaGPUDriverUpgradeStateLabel]
	// Empty state or "upgrade-done" means the node is not upgrading
	return state != "" && state != constants.NvidiaGPUDriverUpgradeStateDone
}

// devicePluginPodPredicate filters device-plugin pods in gpu-operator namespace
func (r *GPUNodeReconciler) devicePluginPodPredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool { return false },
		UpdateFunc: func(e event.UpdateEvent) bool { return false },
		DeleteFunc: func(e event.DeleteEvent) bool {
			pod, ok := e.Object.(*corev1.Pod)
			if !ok {
				return false
			}

			expectedNamespace := config.GetGPUOperatorNamespace()
			matched := pod.Namespace == expectedNamespace &&
				pod.Labels != nil &&
				pod.Labels["app"] == "nvidia-device-plugin-daemonset"

			if matched {
				log.Log.V(1).Info("device plugin pod deletion detected, will trigger GPUNode reconcile",
					"pod", pod.Name,
					"namespace", pod.Namespace,
					"nodeName", pod.Spec.NodeName)
			}

			return matched
		},
		GenericFunc: func(e event.GenericEvent) bool { return false },
	}
}

// mapDevicePluginPodToGPUNode maps device-plugin pod to GPUNode
func (r *GPUNodeReconciler) mapDevicePluginPodToGPUNode(ctx context.Context, obj client.Object) []ctrl.Request {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod.Spec.NodeName == "" {
		log.FromContext(ctx).V(1).Info("mapDevicePluginPodToGPUNode: invalid pod or no nodeName", "ok", ok)
		return nil
	}

	log.FromContext(ctx).V(1).Info("device plugin pod deleted, mapping to GPUNode for reconciliation",
		"pod", pod.Name,
		"namespace", pod.Namespace,
		"nodeName", pod.Spec.NodeName)

	return []ctrl.Request{{NamespacedName: client.ObjectKey{Name: pod.Spec.NodeName}}}
}

func getMatchedVendor(node *corev1.Node, nodeManagerConfig *tfv1.NodeManagerConfig) (string, error) {
	if nodeManagerConfig == nil {
		return constants.AcceleratorVendorNvidia, nil
	}

	// Prioritize MultiVendorNodeSelector if it has entries
	if len(nodeManagerConfig.MultiVendorNodeSelector) > 0 {
		for vendor, nodeSelector := range nodeManagerConfig.MultiVendorNodeSelector {
			if nodeSelector == nil {
				continue
			}
			matches, err := schedulingcorev1.MatchNodeSelectorTerms(node, nodeSelector)
			if err != nil {
				return "", err
			}
			if matches {
				return vendor, nil
			}
		}
		return "", fmt.Errorf("no vendor matched in MultiVendorNodeSelector")
	}

	// Fall back to DefaultVendor
	if nodeManagerConfig.DefaultVendor != "" {
		return nodeManagerConfig.DefaultVendor, nil
	}

	// Default to NVIDIA if nothing is configured
	return constants.AcceleratorVendorNvidia, nil
}
