package expander

import (
	"context"
	"fmt"

	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/karpenter"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type NodeExpander struct {
	client    client.Client
	allocator *gpuallocator.GpuAllocator
	logger    klog.Logger
}

func NewNodeExpander(ctx context.Context, allocator *gpuallocator.GpuAllocator) *NodeExpander {
	return &NodeExpander{
		client:    allocator.Client,
		allocator: allocator,
		logger:    log.FromContext(ctx).WithValues("component", "NodeExpander"),
	}
}

func (e *NodeExpander) ProcessExpansion(ctx context.Context, fwk framework.Framework, pod *corev1.Pod) error {
	if pod == nil {
		return fmt.Errorf("pod cannot be nil")
	}

	// Step 1: Simulate scheduling without GPU plugins
	canSchedule, gpuNodes, err := e.simulateSchedulingWithoutGPU(ctx, fwk, pod)
	if err != nil {
		return fmt.Errorf("simulation failed: %w", err)
	}

	if !canSchedule {
		e.logger.Info("Pod cannot be scheduled even without GPU constraints, skipping expansion", "pod", klog.KObj(pod))
		return nil
	}

	if len(gpuNodes) == 0 {
		e.logger.Info("Pod schedulable but no GPU nodes available, may need manual intervention", "pod", klog.KObj(pod))
		return nil
	}

	// Step 2: Check if it's a GPU resource issue
	isResourceIssue, err := e.checkGPUResourceIssue(ctx, pod)
	if err != nil {
		return fmt.Errorf("GPU resource check failed: %w", err)
	}

	if !isResourceIssue {
		e.logger.Info("Pod scheduling failure not due to GPU resources, skipping expansion", "pod", klog.KObj(pod))
		return nil
	}

	// Step 3: Try node expansion
	return e.expandGPUNodes(ctx, pod, gpuNodes)
}

func (e *NodeExpander) simulateSchedulingWithoutGPU(ctx context.Context, fwk framework.Framework, pod *corev1.Pod) (bool, []*corev1.Node, error) {
	nodeInfos, err := fwk.SnapshotSharedLister().NodeInfos().List()
	if err != nil {
		return false, nil, err
	}

	gpuNodes := []*corev1.Node{}
	schedulableNodes := []*corev1.Node{}

	for _, nodeInfo := range nodeInfos {
		node := nodeInfo.Node()

		// Run basic filters (excluding GPU plugins)
		if e.canScheduleOnNode(ctx, fwk, pod, nodeInfo) {
			schedulableNodes = append(schedulableNodes, node)
		}
	}

	return len(schedulableNodes) > 0, gpuNodes, nil
}

func (e *NodeExpander) checkGPUResourceIssue(ctx context.Context, pod *corev1.Pod) (bool, error) {
	// Get allocation request
	allocRequest, _, err := e.allocator.ComposeAllocationRequest(pod)
	if err != nil {
		return false, err
	}

	// Check if existing + inflight nodes can satisfy the request
	_, _, err = e.allocator.CheckQuotaAndFilter(ctx, allocRequest, false)
	return err != nil, nil
}

func (e *NodeExpander) expandGPUNodes(ctx context.Context, pod *corev1.Pod, candidateNodes []*corev1.Node) error {
	if len(candidateNodes) == 0 {
		return fmt.Errorf("no candidate GPU nodes for expansion")
	}

	// Select best candidate node (smallest capacity first)
	bestNode := e.selectBestCandidateNode(candidateNodes)

	// Create new NodeClaim based on the selected node
	newNodeClaim, err := karpenter.CreateNodeClaimFromNode(ctx, e.client, bestNode)
	if err != nil {
		return fmt.Errorf("failed to create NodeClaim: %w", err)
	}

	if newNodeClaim != nil {
		e.logger.Info("Created new NodeClaim for GPU expansion", "pod", klog.KObj(pod), "nodeClaim", newNodeClaim.Name)
	}

	return nil
}

func (e *NodeExpander) canScheduleOnNode(_ context.Context, _ framework.Framework, pod *corev1.Pod, nodeInfo fwk.NodeInfo) bool {
	// Simple check: node has sufficient CPU/memory and matches affinity
	node := nodeInfo.Node()

	// Check node conditions
	for _, condition := range node.Status.Conditions {
		if condition.Type == corev1.NodeReady && condition.Status != corev1.ConditionTrue {
			return false
		}
	}

	// Check basic resource requirements (CPU, memory) for first container
	if len(pod.Spec.Containers) > 0 {
		for resourceName, quantity := range pod.Spec.Containers[0].Resources.Requests {
			if resourceName == corev1.ResourceCPU || resourceName == corev1.ResourceMemory {
				if available, ok := node.Status.Allocatable[resourceName]; !ok || available.Cmp(quantity) < 0 {
					return false
				}
			}
		}
	}

	return true
}

func (e *NodeExpander) selectBestCandidateNode(nodes []*corev1.Node) *corev1.Node {
	// Simple heuristic: select node with smallest GPU capacity
	bestNode := nodes[0]
	bestGPUCount := e.getGPUCount(bestNode)

	for _, node := range nodes[1:] {
		gpuCount := e.getGPUCount(node)
		if gpuCount < bestGPUCount {
			bestNode = node
			bestGPUCount = gpuCount
		}
	}

	return bestNode
}

func (e *NodeExpander) getGPUCount(node *corev1.Node) int64 {
	if gpu, ok := node.Status.Capacity["nvidia.com/gpu"]; ok {
		return gpu.Value()
	}
	return 0
}
