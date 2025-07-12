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
	"strings"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	cloudprovider "github.com/NexusGPU/tensor-fusion/internal/cloudprovider"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/types"
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
	"k8s.io/client-go/util/retry"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// GPUNodeReconciler reconciles a GPUNode object
type GPUNodeReconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder

	GlobalConfig *config.GlobalConfig
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

		switch node.Spec.ManageMode {
		case tfv1.GPUNodeManageModeAutoSelect:
			// Do nothing, but if it's managed by Karpenter, should come up with some way to tell Karpenter to terminate the GPU node
		case tfv1.GPUNodeManageModeProvisioned:
			clusterName := node.GetLabels()[constants.LabelKeyClusterOwner]
			cluster := &tfv1.TensorFusionCluster{}
			if err := r.Get(ctx, client.ObjectKey{Name: clusterName}, cluster); err != nil {

				if errors.IsNotFound(err) {
					r.Recorder.Eventf(node, corev1.EventTypeWarning, "OrphanedNode", "provisioned node not found, this could result in orphaned nodes, please check manually: %s", node.Name)
					return true, nil
				}
				return false, err
			}

			vendorCfg := cluster.Spec.ComputingVendor
			if vendorCfg == nil {
				return false, fmt.Errorf("failed to get computing vendor config for cluster %s", clusterName)
			}

			provider, err := cloudprovider.GetProvider(*vendorCfg, r.Client, nil)
			if err != nil {
				return false, err
			}

			if node.Status.NodeInfo.InstanceID == "" {
				r.Recorder.Eventf(node, corev1.EventTypeWarning, "OrphanedNode", "provisioned node without instanceID, this could result in orphaned nodes, please check manually: %s", node.Name)
				return true, nil
			}
			err = (*provider).TerminateNode(ctx, &types.NodeIdentityParam{
				InstanceID: node.Status.NodeInfo.InstanceID,
				Region:     node.Status.NodeInfo.Region,
			})
			if err != nil {
				return false, err
			}

		}
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

	if node.Spec.ManageMode != tfv1.GPUNodeManageModeProvisioned {
		// Check if the Kubernetes node exists; if not, the GPUNode should delete itself.
		if node.Status.KubernetesNodeName != "" {
			// Try to get the Kubernetes node
			coreNode := &corev1.Node{}
			err := r.Get(ctx, client.ObjectKey{Name: node.Status.KubernetesNodeName}, coreNode)
			if err != nil {
				if errors.IsNotFound(err) {
					// The Kubernetes node does not exist, delete the GPUNode
					log.Info("Kubernetes node does not exist, deleting GPUNode",
						"kubernetesNodeName", node.Status.KubernetesNodeName)
					if err := r.Delete(ctx, node); err != nil {
						return ctrl.Result{}, fmt.Errorf("failed to delete GPUNode after Kubernetes node was deleted: %w", err)
					}
					// Return early since we've deleted the resource
					return ctrl.Result{}, nil
				}
				return ctrl.Result{}, fmt.Errorf("failed to get Kubernetes node %s: %w",
					node.Status.KubernetesNodeName, err)
			}
		}
	}
	if err := r.reconcileCloudVendorNode(ctx, node, poolObj); err != nil {
		return ctrl.Result{}, err
	}

	// Only reconcile if the node has a kubernetes node name, otherwise the DaemonSet like workloads can not be scheduled
	if node.Status.KubernetesNodeName == "" {
		return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
	}

	if err := r.reconcileNodeDiscoveryJob(ctx, node, poolObj); err != nil {
		return ctrl.Result{}, err
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
	checkAgain, err := r.checkStatusAndUpdateVirtualCapacity(ctx, hypervisorName, node, poolObj)
	if err != nil {
		return ctrl.Result{}, err
	}
	if checkAgain {
		return ctrl.Result{RequeueAfter: constants.StatusCheckInterval}, nil
	}
	return ctrl.Result{}, nil
}

func (r *GPUNodeReconciler) checkStatusAndUpdateVirtualCapacity(ctx context.Context, hypervisorName string, node *tfv1.GPUNode, poolObj *tfv1.GPUPool) (checkAgain bool, err error) {
	pod := &corev1.Pod{}
	fetchErr := r.Get(ctx, client.ObjectKey{Name: hypervisorName, Namespace: utils.CurrentNamespace()}, pod)
	if fetchErr != nil {
		return false, fmt.Errorf("failed to get hypervisor pod: %w", fetchErr)
	}

	// Reconcile GPUNode status with hypervisor pod status, when changed
	if pod.Status.Phase != corev1.PodRunning || !utils.IsPodConditionTrue(pod.Status.Conditions, corev1.PodReady) {
		if node.Status.Phase != tfv1.TensorFusionGPUNodePhasePending {
			node.Status.Phase = tfv1.TensorFusionGPUNodePhasePending
			err := r.Status().Update(ctx, node)
			if err != nil {
				return true, fmt.Errorf("failed to update GPU node status: %w", err)
			}
		}

		// Update all GPU devices status to Pending
		// TODO, should update in batch, making every GPU pending state led to unschedule for new workers
		err = r.syncStatusToGPUDevices(ctx, node, tfv1.TensorFusionGPUPhasePending)
		if err != nil {
			return true, err
		}

		return true, nil
	} else {
		gpuModels, err := gpuallocator.RefreshGPUNodeCapacity(ctx, r.Client, node, poolObj)
		if err != nil {
			return true, err
		}

		// update metrics to get historical allocation line chart and trending
		metrics.SetNodeMetrics(node, poolObj, gpuModels)

		err = r.syncStatusToGPUDevices(ctx, node, tfv1.TensorFusionGPUPhaseRunning)
		if err != nil {
			return true, err
		}
		return false, nil
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
				return fmt.Errorf("failed to patch GPU device status: %w", err)
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
	tmpl.Spec.NodeName = gpunode.Status.KubernetesNodeName
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
	spec.NodeName = node.Status.KubernetesNodeName

	// add must-have tensor-fusion hypervisor manifest
	log.Info("adding must-have tensor-fusion hypervisor manifest", "node", node.Name)
	utils.AddTFHypervisorConfAfterTemplate(ctx, &spec, pool)

	// add scheduling config for hypervisor
	if pool.Spec.SchedulingConfigTemplate != nil {
		schedulingConfigTemplate := &tfv1.SchedulingConfigTemplate{}
		if err := r.Get(ctx, client.ObjectKey{Name: *pool.Spec.SchedulingConfigTemplate}, schedulingConfigTemplate); err == nil {
			if schedulingConfigTemplate.Spec.Hypervisor != nil {
				if cfg, err := json.Marshal(schedulingConfigTemplate.Spec.Hypervisor); err == nil {
					spec.Containers[0].Env = append(spec.Containers[0].Env, corev1.EnvVar{
						Name:  constants.HypervisorSchedulingConfigEnv,
						Value: string(cfg),
					}, corev1.EnvVar{
						Name:  constants.HypervisorMetricsFormatEnv,
						Value: r.GlobalConfig.MetricsFormat,
					}, corev1.EnvVar{
						Name:  constants.HypervisorMetricsExtraLabelsEnv,
						Value: strings.Join(r.GlobalConfig.MetricsExtraPodLabels, ","),
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

func (r *GPUNodeReconciler) reconcileCloudVendorNode(ctx context.Context, node *tfv1.GPUNode, pool *tfv1.GPUPool) error {
	// Avoid creating duplicated cloud vendor nodes, if not working, keep pending status
	if node.Status.NodeInfo.InstanceID != "" {
		// node already created, check status
		if node.Status.KubernetesNodeName != "" {
			var k8sNode corev1.Node
			if err := r.Get(ctx, client.ObjectKey{Name: node.Status.KubernetesNodeName}, &k8sNode); err != nil {
				if errors.IsNotFound(err) {
					r.Recorder.Eventf(node, corev1.EventTypeNormal, "NodeNotFound", "Kubernetes node has been disappeared, deleting GPUNode", node.Status.KubernetesNodeName)
					err := r.Delete(ctx, node)
					if err != nil {
						return err
					}
				}
			}
			// TODO: sync cordon and drain status to all GPUs it owned
		}
		return nil
	}

	// no cloud vendor param indicates it isn't a managed node, skip cloud vendor reconcile
	if node.Spec.CloudVendorParam == "" {
		if node.Spec.ManageMode != tfv1.GPUNodeManageModeProvisioned {
			return nil
		}
		r.Recorder.Eventf(node, corev1.EventTypeWarning, "CloudVendorParamEmpty", "cloud vendor param is empty, but manage mode is provisioned: %s", node.Name)
		return nil
	}

	// No NodeInfo, should create new one
	provider, _, err := createProvisionerAndQueryCluster(ctx, pool, r.Client)
	if err != nil {
		return err
	}

	// Create node on cloud provider [this can result in cloud vendor bills, be cautious!!!]
	var nodeParam types.NodeCreationParam
	err = json.Unmarshal([]byte(node.Spec.CloudVendorParam), &nodeParam)
	if err != nil {
		return fmt.Errorf("failed to unmarshal cloud vendor param: %w, GPUNode: %s", err, node.Name)
	}

	// TODO: query cloud vendor by node name
	status, err := provider.CreateNode(ctx, &nodeParam)
	if err != nil {
		return err
	}

	// Update GPUNode status about the cloud vendor info
	// To match GPUNode - K8S node, the --node-label in Kubelet is MUST-have, like Karpenter, it force set userdata to add a provisionerId label, k8s node controller then can set its ownerReference to the GPUNode
	gpuNode := &tfv1.GPUNode{}
	err = r.Get(ctx, client.ObjectKey{Name: nodeParam.NodeName}, gpuNode)
	if err != nil {
		return err
	}
	gpuNode.Status.Phase = tfv1.TensorFusionGPUNodePhasePending
	gpuNode.Status.NodeInfo.IP = status.PrivateIP
	gpuNode.Status.NodeInfo.InstanceID = status.InstanceID
	gpuNode.Status.NodeInfo.Region = nodeParam.Region

	// Retry status update until success to handle version conflicts
	err = retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		// Get the latest version before attempting an update
		latest := &tfv1.GPUNode{}
		if err := r.Get(ctx, client.ObjectKey{Name: gpuNode.Name}, latest); err != nil {
			return err
		}

		// Apply our status updates to the latest version
		latest.Status.Phase = tfv1.TensorFusionGPUNodePhasePending
		latest.Status.NodeInfo.IP = status.PrivateIP
		latest.Status.NodeInfo.InstanceID = status.InstanceID
		latest.Status.NodeInfo.Region = nodeParam.Region

		// Attempt to update with the latest version
		return r.Client.Status().Update(ctx, latest)
	})

	if err != nil {
		log.FromContext(ctx).Error(err, "Failed to update GPUNode status after retries, must terminate node to keep operation atomic", "name", nodeParam.NodeName)
		errTerminate := provider.TerminateNode(ctx, &types.NodeIdentityParam{
			InstanceID: status.InstanceID,
			Region:     nodeParam.Region,
		})
		if errTerminate != nil {
			log.FromContext(ctx).Error(errTerminate, "Failed to terminate cloud vendor node when GPUNode status failed to update")
			panic(errTerminate)
		}
		return nil
	}

	r.Recorder.Eventf(pool, corev1.EventTypeNormal, "ManagedNodeCreated", "Created node: %s, IP: %s", status.InstanceID, status.PrivateIP)
	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *GPUNodeReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.GPUNode{}).
		Named("gpunode").
		// TODO: should not own node, let node_claim_controller to own node for cloud vendor VM nodes,
		Owns(&corev1.Node{}).
		Owns(&batchv1.Job{}).
		Owns(&corev1.Pod{}).
		Complete(r)
	// WARN: can not Owns(&tfv1.GPU{}) here, because gpunode controller also reconciles GPU devices,
	// this controller also sync node status to devices status when hypervisor not working
}

func getDiscoveryJobName(gpunodeName string) string {
	return fmt.Sprintf("node-discovery-%s", gpunodeName)
}
