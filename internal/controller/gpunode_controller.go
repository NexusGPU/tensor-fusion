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
	"strings"
	"time"

	cloudprovider "github.com/NexusGPU/tensor-fusion-operator/internal/cloudprovider"
	"github.com/NexusGPU/tensor-fusion-operator/internal/cloudprovider/types"
	"github.com/NexusGPU/tensor-fusion-operator/internal/utils"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
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

// Reconcile GPU nodes
func (r *GPUNodeReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)

	log.Info("Reconciling GPUNode", "name", req.NamespacedName.Name)
	defer func() {
		log.Info("Finished reconciling GPUNode", "name", req.NamespacedName.Name)
	}()

	node := &tfv1.GPUNode{}
	if err := r.Get(ctx, req.NamespacedName, node); err != nil {
		if errors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	deleted, err := utils.HandleFinalizer(ctx, node, r.Client, func(ctx context.Context, node *tfv1.GPUNode) error {
		if node.Spec.ManageMode == tfv1.GPUNodeManageModeAutoSelect {
			// Do nothing, but if it's managed by Karpenter, should come up with some way to tell Karpenter to terminate the GPU node
		} else if node.Spec.ManageMode == tfv1.GPUNodeManageModeProvisioned {
			clusterName := node.GetLabels()[constants.LabelKeyClusterOwner]
			cluster := &tfv1.TensorFusionCluster{}
			if err := r.Get(ctx, client.ObjectKey{Name: clusterName}, cluster); err != nil {
				return err
			}

			vendorCfg := cluster.Spec.ComputingVendor
			if vendorCfg == nil {
				return fmt.Errorf("failed to get computing vendor config for cluster %s", clusterName)
			}

			provider, err := cloudprovider.GetProvider(*vendorCfg)
			if err != nil {
				return err
			}

			if node.Status.NodeInfo.InstanceID == "" {
				r.Recorder.Eventf(node, corev1.EventTypeWarning, "OrphanedNode", "provisioned node without instanceID, this could result in orphaned nodes, please check manually: %s", node.Name)
				return nil
			}
			err = (*provider).TerminateNode(ctx, &types.NodeIdentityParam{
				InstanceID: node.Status.NodeInfo.InstanceID,
				Region:     node.Status.NodeInfo.Region,
			})
			if err != nil {
				return err
			}

		}
		return nil
	})
	if err != nil {
		return ctrl.Result{}, err
	}
	if deleted {
		return ctrl.Result{}, nil
	}

	var poolName string
	for labelKey := range node.Labels {
		after, ok := strings.CutPrefix(labelKey, constants.GPUNodePoolIdentifierLabelPrefix)
		if ok {
			poolName = after
			break
		}
	}

	poolObj := &tfv1.GPUPool{}
	err = r.Client.Get(ctx, client.ObjectKey{Name: poolName}, poolObj)
	if err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to get tensor-fusion pool, can not create node discovery job, pool: %s", poolName)
	}

	if err := r.reconcileCloudVendorNode(ctx, node, poolObj); err != nil {
		return ctrl.Result{}, err
	}

	// Only reconcile if the node has a kubernetes node name, otherwise the DaemonSet like workloads can not be scheduled
	if node.Status.KubernetesNodeName == "" {
		return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
	}
	if poolName == "" {
		log.Error(nil, "failed to get pool name", "node", node.Name)
		return ctrl.Result{}, nil
	}

	if err := r.reconcileNodeDiscoveryJob(ctx, node, poolObj); err != nil {
		return ctrl.Result{}, err
	}

	hypervisorName, err := r.reconcileHypervisorPod(ctx, node, poolObj)
	if err != nil {
		return ctrl.Result{}, err
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

func (r *GPUNodeReconciler) checkStatusAndUpdateVirtualCapacity(ctx context.Context, hypervisorName string, node *tfv1.GPUNode, poolObj *tfv1.GPUPool) (bool, error) {
	pod := &corev1.Pod{}
	fetchErr := r.Client.Get(ctx, client.ObjectKey{Name: hypervisorName, Namespace: utils.CurrentNamespace()}, pod)
	if fetchErr != nil {
		return false, fmt.Errorf("failed to get hypervisor pod: %w", fetchErr)
	}

	// Reconcile GPUNode status with hypervisor pod status, when changed
	if pod.Status.Phase != corev1.PodRunning && !utils.IsPodConditionTrue(pod.Status.Conditions, corev1.PodReady) {
		node.Status.Phase = tfv1.TensorFusionGPUNodePhasePending
		err := r.Status().Update(ctx, node)
		if err != nil {
			return false, fmt.Errorf("failed to update GPU node status: %w", err)
		}
		return true, nil
	} else {

		virtualTFlops, virtualVRAM := r.CalculateVirtualCapacity(node, poolObj)
		node.Status.VirtualTFlops = virtualTFlops
		node.Status.VirtualVRAM = virtualVRAM

		node.Status.Phase = tfv1.TensorFusionGPUNodePhaseRunning
		err := r.Status().Update(ctx, node)
		if err != nil {
			return false, fmt.Errorf("failed to update GPU node status: %w", err)
		}
	}
	return false, nil
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

	templateCopy := podTmpl.Template.DeepCopy()
	if templateCopy.Spec.Affinity == nil {
		templateCopy.Spec.Affinity = &corev1.Affinity{}
	}
	if templateCopy.Spec.Affinity.NodeAffinity == nil {
		templateCopy.Spec.Affinity.NodeAffinity = &corev1.NodeAffinity{}
	}
	if templateCopy.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		templateCopy.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = &corev1.NodeSelector{
			NodeSelectorTerms: make([]corev1.NodeSelectorTerm, 0),
		}
	}
	templateCopy.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms =
		append(templateCopy.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms, corev1.NodeSelectorTerm{
			MatchFields: []corev1.NodeSelectorRequirement{
				{
					Key:      "metadata.name",
					Operator: corev1.NodeSelectorOpIn,
					Values:   []string{gpunode.Status.KubernetesNodeName},
				},
			},
		})
	// allow job to run at any taint Nodes that marked as NoSchedule
	if templateCopy.Spec.Tolerations == nil {
		templateCopy.Spec.Tolerations = []corev1.Toleration{}
	}
	templateCopy.Spec.Tolerations = append(templateCopy.Spec.Tolerations, corev1.Toleration{
		Key:      "NoSchedule",
		Operator: corev1.TolerationOpExists,
	})

	if len(templateCopy.Spec.Containers) > 0 {
		if len(templateCopy.Spec.Containers[0].Env) == 0 {
			templateCopy.Spec.Containers[0].Env = []corev1.EnvVar{}
		}
		templateCopy.Spec.Containers[0].Env = append(templateCopy.Spec.Containers[0].Env, corev1.EnvVar{
			Name:  constants.NodeDiscoveryReportGPUNodeEnvName,
			Value: gpunode.Name,
		})
	}

	// create node-discovery job
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("node-discovery-%s", gpunode.Name),
			Namespace: utils.CurrentNamespace(),
		},
		Spec: batchv1.JobSpec{
			TTLSecondsAfterFinished: ptr.To[int32](3600 * 10),
			Template:                *templateCopy,
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
			return fmt.Errorf("create node job %w", err)
		}
	}

	return nil
}

func (r *GPUNodeReconciler) reconcileHypervisorPod(ctx context.Context, node *tfv1.GPUNode, pool *tfv1.GPUPool) (string, error) {
	if pool == nil {
		return "", fmt.Errorf("failed to get tensor-fusion pool, can not create hypervisor pod")
	}

	namespace := utils.CurrentNamespace()
	log := log.FromContext(ctx)
	hypervisorPodName := fmt.Sprintf("hypervisor-%s", node.Name)

	hypervisorConfig := pool.Spec.ComponentConfig.Hypervisor
	podTmpl := &corev1.PodTemplate{}
	err := json.Unmarshal(hypervisorConfig.PodTemplate.Raw, podTmpl)
	if err != nil {
		return "", fmt.Errorf("failed to unmarshal pod template: %w", err)
	}
	spec := podTmpl.Template.Spec.DeepCopy()
	if spec.NodeSelector == nil {
		spec.NodeSelector = make(map[string]string)
	}
	spec.NodeSelector["kubernetes.io/hostname"] = node.Status.KubernetesNodeName
	newPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      hypervisorPodName,
			Namespace: namespace,
			Labels: map[string]string{
				fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, pool.Name): "true",
			},
		},
		Spec: *spec,
	}

	if newPod.Spec.Tolerations == nil {
		newPod.Spec.Tolerations = []corev1.Toleration{}
	}
	newPod.Spec.Tolerations = append(newPod.Spec.Tolerations, corev1.Toleration{
		Key:      "NoSchedule",
		Operator: corev1.TolerationOpExists,
	})

	e := controllerutil.SetControllerReference(node, newPod, r.Scheme)
	if e != nil {
		return "", fmt.Errorf("failed to set controller reference: %w", e)
	}
	_, err = controllerutil.CreateOrUpdate(ctx, r.Client, newPod, func() error {
		log.Info("Creating or Updating hypervisor pod", "name", hypervisorPodName)
		return nil
	})
	if err != nil {
		return "", fmt.Errorf("failed to create or update hypervisor pod: %w", err)
	}
	return hypervisorPodName, nil
}

func (r *GPUNodeReconciler) reconcileCloudVendorNode(ctx context.Context, node *tfv1.GPUNode, pool *tfv1.GPUPool) error {
	// no cloud vendor param indicates it isn't a managed node, skip cloud vendor reconcile
	if node.Spec.CloudVendorParam == "" {
		if node.Spec.ManageMode != tfv1.GPUNodeManageModeProvisioned {
			return nil
		}
		return fmt.Errorf("cloud vendor param is empty, but manage mode is provisioned: %s", node.Name)
	}

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

	// No NodeInfo, should create new one
	provider, _, err := createProvisionerAndQueryCluster(ctx, pool, r.Client)
	if err != nil {
		return err
	}

	// Create node on cloud provider [this can result in cloud vendor bills, be cautious!!!]
	var nodeParam types.NodeCreationParam
	json.Unmarshal([]byte(node.Spec.CloudVendorParam), &nodeParam)
	status, err := provider.CreateNode(ctx, &nodeParam)
	if err != nil {
		return err
	}

	// Update GPUNode status about the cloud vendor info
	// To match GPUNode - K8S node, the --node-label in Kubelet is MUST-have, like Karpenter, it force set userdata to add a provisionerId label, k8s node controller then can set its ownerReference to the GPUNode
	gpuNode := &tfv1.GPUNode{}
	err = r.Client.Get(ctx, client.ObjectKey{Name: nodeParam.NodeName}, gpuNode)
	if err != nil {
		return err
	}
	gpuNode.Status.NodeInfo.IP = status.PrivateIP
	gpuNode.Status.NodeInfo.InstanceID = status.InstanceID
	gpuNode.Status.NodeInfo.Region = nodeParam.Region
	if err := r.Client.Status().Update(ctx, gpuNode); err != nil {
		log.FromContext(ctx).Info("Failed to update GPUNode status, must terminate node to keep operation atomic", "name", nodeParam.NodeName)
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

func (r *GPUNodeReconciler) CalculateVirtualCapacity(node *tfv1.GPUNode, pool *tfv1.GPUPool) (resource.Quantity, resource.Quantity) {
	diskSize, _ := node.Status.NodeInfo.DataDiskSize.AsInt64()
	ramSize, _ := node.Status.NodeInfo.RAMSize.AsInt64()

	virtualVRAM := node.Status.TotalVRAM.DeepCopy()
	virtualTFlops := node.Status.TotalTFlops.DeepCopy()

	virtualVRAM.Add(*resource.NewQuantity(
		int64(float64(float64(diskSize)*float64(pool.Spec.CapacityConfig.Oversubscription.VRAMExpandToHostDisk)/100.0)),
		resource.DecimalSI),
	)
	virtualVRAM.Add(*resource.NewQuantity(
		int64(float64(float64(ramSize)*float64(pool.Spec.CapacityConfig.Oversubscription.VRAMExpandToHostMem)/100.0)),
		resource.DecimalSI),
	)
	virtualTFlops.Mul(int64(pool.Spec.CapacityConfig.Oversubscription.TFlopsOversellRatio))
	return virtualVRAM, virtualTFlops
}

// SetupWithManager sets up the controller with the Manager.
func (r *GPUNodeReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.GPUNode{}, builder.WithPredicates(predicate.GenerationChangedPredicate{})).
		Named("gpunode").
		Owns(&corev1.Node{}).
		Owns(&batchv1.Job{}).
		Owns(&corev1.Pod{}).
		Complete(r)
}
