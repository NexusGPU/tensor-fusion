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

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

// GPUNodeReconciler reconciles a GPUNode object
type GPUNodeReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpunodes,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpunodes/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=gpunodes/finalizers,verbs=update
// +kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch;create;update;patch;delete

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
		return ctrl.Result{}, fmt.Errorf("failed to get tensor-fusion pool, can not create node discovery job, pool: %s", poolName)
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

	if err := r.reconcileNodeDiscoveryJob(ctx, node, poolObj); err != nil {
		return ctrl.Result{}, err
	}

	if node.Status.TotalGPUs == 0 {
		log.Info("GPU on this node has not been discovered, wait next loop", "node", node.Name)
		return ctrl.Result{}, nil
	}

	hypervisorName, err := r.reconcileHypervisorPod(ctx, node, poolObj)
	if err != nil {
		return ctrl.Result{}, err
	}
	// pod deleted or deleting, wait next reconcile
	if hypervisorName == "" {
		return ctrl.Result{RequeueAfter: constants.PendingRequeueDuration}, nil
	}

	// Check if hypervisor is running well, if so, set as running status
	err = r.checkStatusAndUpdateVirtualCapacity(ctx, hypervisorName, node, poolObj)
	if errors.IsNotFound(err) {
		log.Info("Hypervisor pod not found, requeue", "hypervisorName", hypervisorName)
		return ctrl.Result{Requeue: true}, nil
	}
	return ctrl.Result{}, err
}

func (r *GPUNodeReconciler) checkStatusAndUpdateVirtualCapacity(ctx context.Context, hypervisorName string, node *tfv1.GPUNode, poolObj *tfv1.GPUPool) error {
	pod := &corev1.Pod{}
	fetchErr := r.Get(ctx, client.ObjectKey{Name: hypervisorName, Namespace: utils.CurrentNamespace()}, pod)
	if fetchErr != nil {
		return fetchErr
	}

	// Reconcile GPUNode status with hypervisor pod status, when changed
	if pod.Status.Phase != corev1.PodRunning || !utils.IsPodConditionTrue(pod.Status.Conditions, corev1.PodReady) {
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
		gpuModels, err := gpuallocator.RefreshGPUNodeCapacity(ctx, r.Client, node, poolObj)
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
			gpuNodeClaim := &tfv1.GPUNodeClaim{}
			if err := r.Get(ctx, client.ObjectKey{Name: node.Labels[constants.ProvisionerLabelKey]}, gpuNodeClaim); err != nil {
				if errors.IsNotFound(err) {
					log.FromContext(ctx).Info("GPUNodeClaim not found but provisioner is not empty, orphan GPUNode",
						"name", node.Labels[constants.ProvisionerLabelKey])
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

func (r *GPUNodeReconciler) reconcileNodeDiscoveryJob(
	ctx context.Context,
	gpunode *tfv1.GPUNode,
	pool *tfv1.GPUPool,
) error {
	log := log.FromContext(ctx)
	log.Info("starting node discovery job")

	if pool.Spec.ComponentConfig == nil || pool.Spec.ComponentConfig.NodeDiscovery.PodTemplate == nil {
		return fmt.Errorf(`missing node discovery pod template in pool spec`)
	}
	podTmpl := &corev1.PodTemplate{}
	err := json.Unmarshal(pool.Spec.ComponentConfig.NodeDiscovery.PodTemplate.Raw, podTmpl)
	if err != nil {
		return fmt.Errorf("unmarshal pod template: %w", err)
	}
	tmpl := podTmpl.Template
	if tmpl.Labels == nil {
		tmpl.Labels = map[string]string{}
	}
	tmpl.Labels[constants.LabelComponent] = constants.ComponentNodeDiscovery
	tmpl.Spec.NodeName = gpunode.Name
	// allow job to run at any taint Nodes that marked as NoSchedule
	tmpl.Spec.Tolerations = append(tmpl.Spec.Tolerations, corev1.Toleration{
		Key:      string(corev1.TaintEffectNoSchedule),
		Operator: corev1.TolerationOpExists,
	})
	tmpl.Spec.EnableServiceLinks = ptr.To(false)

	utils.AddTFNodeDiscoveryConfAfterTemplate(ctx, &tmpl, pool, gpunode.Name)

	// create node-discovery job
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:        getDiscoveryJobName(gpunode.Name),
			Namespace:   utils.CurrentNamespace(),
			Labels:      tmpl.Labels,
			Annotations: tmpl.Annotations,
		},
		Spec: batchv1.JobSpec{
			TTLSecondsAfterFinished: ptr.To[int32](3600 * 10),
			Template:                tmpl,
		},
	}

	if err := r.Get(ctx, client.ObjectKeyFromObject(job), job); err != nil {
		if errors.IsNotFound(err) {
			if err := ctrl.SetControllerReference(gpunode, job, r.Scheme); err != nil {
				return fmt.Errorf("set owner reference %w", err)
			}
			if err := r.Create(ctx, job); err != nil {
				return fmt.Errorf("create node discovery job %w", err)
			}
		} else {
			return fmt.Errorf("create node discovery job %w", err)
		}
	}

	if job.Status.Failed > 0 {
		log.Info("node discovery job failed, update GPU node status to failed", "node", gpunode.Name)
		// Update phase to failed, require manual address why it failed and restart of node discovery job
		gpunode.Status.Phase = tfv1.TensorFusionGPUNodePhaseFailed
		if err := r.Status().Update(ctx, gpunode); err != nil {
			return fmt.Errorf("failed to update GPU node status to failed: %w", err)
		}
		metrics.SetNodeMetrics(gpunode, pool, nil)
	}

	return nil
}

func (r *GPUNodeReconciler) reconcileHypervisorPod(ctx context.Context, node *tfv1.GPUNode, pool *tfv1.GPUPool) (string, error) {
	log := log.FromContext(ctx)

	if pool.Spec.ComponentConfig == nil || pool.Spec.ComponentConfig.Hypervisor == nil {
		return "", fmt.Errorf("missing hypervisor config")
	}

	key := client.ObjectKey{
		Namespace: utils.CurrentNamespace(),
		Name:      fmt.Sprintf("hypervisor-%s", node.Name),
	}

	currentPod := &corev1.Pod{}
	if err := r.Get(ctx, key, currentPod); err != nil {
		if !errors.IsNotFound(err) {
			return "", fmt.Errorf("failed to get current hypervisor pod: %w", err)
		}
	} else {
		// hypervisor pod found, verify status and podTemplateHash
		if node.Status.Phase == tfv1.TensorFusionGPUNodePhaseRunning {
			return key.Name, nil
		}

		oldHash := currentPod.Labels[constants.LabelKeyPodTemplateHash]
		if !currentPod.DeletionTimestamp.IsZero() {
			log.Info("hypervisor pod is still being deleting", "name", key.Name, "hash", oldHash)
			return "", nil
		}

		newHash := utils.GetObjectHash(pool.Spec.ComponentConfig.Hypervisor)
		if utils.IsPodStopped(currentPod) || oldHash != newHash {
			if err := r.Delete(ctx, currentPod); err != nil {
				return "", fmt.Errorf("failed to delete old hypervisor pod: %w", err)
			}
			log.Info("old hypervisor pod deleted", "name", currentPod.Name, "oldHash", oldHash, "newHash", newHash)
			return "", nil
		} else {
			return key.Name, nil
		}
	}

	log.Info("hypervisor pod not found, creating new one", "node", node.Name)
	if err := r.createHypervisorPod(ctx, key, node, pool); err != nil {
		if errors.IsAlreadyExists(err) {
			log.Info("hypervisor pod already exists, skip creation", "node", node.Name)
			return "", nil
		} else {
			return "", fmt.Errorf("failed to create hypervisor pod: %w", err)
		}
	}
	return key.Name, nil
}

func (r *GPUNodeReconciler) createHypervisorPod(ctx context.Context, key client.ObjectKey, node *tfv1.GPUNode, pool *tfv1.GPUPool) error {
	log := log.FromContext(ctx)

	podTmpl := &corev1.PodTemplate{}

	// unmarshal pod template
	err := json.Unmarshal(pool.Spec.ComponentConfig.Hypervisor.PodTemplate.Raw, podTmpl)
	if err != nil {
		return fmt.Errorf("failed to unmarshal pod template: %w", err)
	}
	spec := podTmpl.Template.Spec
	if spec.NodeSelector == nil {
		spec.NodeSelector = make(map[string]string)
	}
	spec.EnableServiceLinks = ptr.To(false)
	spec.NodeName = node.Name

	// add must-have tensor-fusion hypervisor manifest
	log.Info("adding must-have tensor-fusion hypervisor manifest", "node", node.Name)
	utils.AddTFHypervisorConfAfterTemplate(ctx, &spec, pool)

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
				maps.Copy(mergedLabels, podTmpl.Template.Labels)
				mergedLabels[fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name)] = "true"
				mergedLabels[constants.LabelKeyPodTemplateHash] = newHash
				mergedLabels[constants.LabelComponent] = constants.ComponentHypervisor
				return mergedLabels
			}(),
			Annotations: podTmpl.Template.Annotations,
		},
		Spec: spec,
	}

	if newPod.Spec.Tolerations == nil {
		newPod.Spec.Tolerations = []corev1.Toleration{}
	}
	newPod.Spec.Tolerations = append(newPod.Spec.Tolerations, corev1.Toleration{
		Key:      string(corev1.TaintEffectNoSchedule),
		Operator: corev1.TolerationOpExists,
	})
	err = controllerutil.SetControllerReference(node, newPod, r.Scheme)
	if err != nil {
		return fmt.Errorf("failed to set controller reference: %w", err)
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
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.GPUNode{}).
		Named("gpunode").
		Watches(&corev1.Node{}, handler.EnqueueRequestsFromMapFunc(
			func(ctx context.Context, obj client.Object) []reconcile.Request {
				return []reconcile.Request{
					{NamespacedName: client.ObjectKey{Name: obj.GetName()}},
				}
			})).
		Owns(&batchv1.Job{}).
		Owns(&corev1.Pod{}).
		Owns(&tfv1.GPU{}).
		Complete(r)
}

func getDiscoveryJobName(gpunodeName string) string {
	return fmt.Sprintf("node-discovery-%s", gpunodeName)
}
