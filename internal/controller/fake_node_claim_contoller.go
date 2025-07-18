package controller

import (
	"context"
	"strings"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/awslabs/operatorpkg/status"
	"github.com/samber/lo"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	v1 "sigs.k8s.io/karpenter/pkg/apis/v1"
)

type FakeNodeClaimReconciler struct {
	client client.Client
	Scheme *runtime.Scheme
}

func registerKarpenterGVK(s *runtime.Scheme) {
	// EC2NodeClass
	ec2GVK := schema.GroupVersionKind{
		Group:   "karpenter.k8s.aws",
		Version: "v1",
		Kind:    "EC2NodeClass",
	}
	s.AddKnownTypeWithName(ec2GVK, &unstructured.Unstructured{})
	s.AddKnownTypeWithName(ec2GVK.GroupVersion().WithKind("EC2NodeClassList"),
		&unstructured.UnstructuredList{})
}

func (r *FakeNodeClaimReconciler) SetupWithManager(mgr ctrl.Manager) error {
	registerKarpenterGVK(mgr.GetScheme())

	return ctrl.NewControllerManagedBy(mgr).
		For(&v1.NodeClaim{}).
		Complete(r)
}

type NodeOptions struct {
	metav1.ObjectMeta
	ReadyStatus    corev1.ConditionStatus
	ReadyReason    string
	Conditions     []corev1.NodeCondition
	Unschedulable  bool
	ProviderID     string
	Taints         []corev1.Taint
	Allocatable    corev1.ResourceList
	Capacity       corev1.ResourceList
	OwnerReference []metav1.OwnerReference
}

func (r *FakeNodeClaimReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)
	log.Info("Reconciling FakeNodeClaim")

	nodeClaim := &v1.NodeClaim{}
	if err := r.client.Get(ctx, req.NamespacedName, nodeClaim); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}
	if !nodeClaim.DeletionTimestamp.IsZero() {
		return r.reconcileDeletion(ctx, nodeClaim)
	}
	return r.reconcileCreation(ctx, nodeClaim)
}

// mock delete node
func (r *FakeNodeClaimReconciler) reconcileDeletion(ctx context.Context, nc *v1.NodeClaim) (ctrl.Result, error) {
	var node corev1.Node
	if err := r.client.Get(ctx, client.ObjectKey{Name: nc.Status.ProviderID}, &node); err == nil {
		_ = r.client.Delete(ctx, &node)
	}
	patch := client.MergeFrom(nc.DeepCopy())
	controllerutil.RemoveFinalizer(nc, constants.Finalizer)
	if err := r.client.Patch(ctx, nc, patch); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

// mock create node
func (r *FakeNodeClaimReconciler) reconcileCreation(ctx context.Context, nodeClaim *v1.NodeClaim) (ctrl.Result, error) {
	log := log.FromContext(ctx)
	if !controllerutil.ContainsFinalizer(nodeClaim, constants.Finalizer) {
		patch := client.MergeFrom(nodeClaim.DeepCopy())
		controllerutil.AddFinalizer(nodeClaim, constants.Finalizer)
		if err := r.client.Patch(ctx, nodeClaim, patch); err != nil {
			return ctrl.Result{}, err
		}
	}

	nodeName := nodeClaim.Status.ProviderID
	if nodeName == "" {
		nodeName = nodeClaim.Name
	}

	node := &corev1.Node{}
	err := r.client.Get(ctx, types.NamespacedName{Name: nodeName}, node)
	if err == nil {
		log.Info("Node already exists", "node", nodeName)
		return ctrl.Result{}, nil
	}
	n := Node(
		NodeOptions{
			ObjectMeta: metav1.ObjectMeta{
				Name: nodeName,
				Labels: lo.Assign(map[string]string{
					strings.Split(constants.InitialGPUNodeSelector, "=")[0]:       constants.TrueStringValue,
					v1.NodeClassLabelKey(nodeClaim.Spec.NodeClassRef.GroupKind()): nodeClaim.Spec.NodeClassRef.Name,
				}, nodeClaim.Labels),
				Annotations: nodeClaim.Annotations,
			},
			Taints:      append(nodeClaim.Spec.Taints, nodeClaim.Spec.StartupTaints...),
			Capacity:    nodeClaim.Status.Capacity,
			Allocatable: nodeClaim.Status.Allocatable,
			ProviderID:  nodeClaim.Status.ProviderID,
			OwnerReference: []metav1.OwnerReference{
				{
					APIVersion: nodeClaim.APIVersion,
					UID:        nodeClaim.UID,
					Kind:       nodeClaim.Kind,
					Name:       nodeClaim.Name,
					Controller: lo.ToPtr(true),
				},
			},
		},
	)
	n.Spec.Taints = append(n.Spec.Taints, v1.UnregisteredNoExecuteTaint)
	if err := r.client.Create(ctx, n); err != nil {
		return ctrl.Result{}, err
	}

	patch := client.MergeFrom(nodeClaim.DeepCopy())

	nodeClaim.SetConditions([]status.Condition{
		{
			Type:               v1.ConditionTypeLaunched,
			Status:             metav1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
		},
	})
	nodeClaim.Status.ProviderID = nodeName
	if err := r.client.Status().Patch(ctx, nodeClaim, patch); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{}, nil
}

// mock create node
func Node(options NodeOptions) *corev1.Node {
	if options.ReadyStatus == "" {
		options.ReadyStatus = corev1.ConditionTrue
	}
	if options.Capacity == nil {
		options.Capacity = options.Allocatable
	}

	objectMeta := options.ObjectMeta
	objectMeta.OwnerReferences = options.OwnerReference

	return &corev1.Node{
		ObjectMeta: objectMeta,
		Spec: corev1.NodeSpec{
			Unschedulable: options.Unschedulable,
			Taints:        options.Taints,
			ProviderID:    options.ProviderID,
		},
		Status: corev1.NodeStatus{
			Allocatable: options.Allocatable,
			Capacity:    options.Capacity,
			Conditions:  []corev1.NodeCondition{{Type: corev1.NodeReady, Status: options.ReadyStatus, Reason: options.ReadyReason}},
		},
	}
}
