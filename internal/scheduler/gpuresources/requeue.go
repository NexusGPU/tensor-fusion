package gpuresources

import (
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
)

// How long a wake-up waits for the allocator's store, and how often it looks.
// Variables so tests can shorten them.
var (
	allocatorSyncTimeout      = 30 * time.Second
	allocatorSyncInitialDelay = 20 * time.Millisecond
	allocatorSyncMaxDelay     = time.Second
)

// allocatorSyncState is the bookkeeping of wakeAfterAllocatorSync.
type allocatorSyncState struct {
	mu sync.Mutex
	// waiting maps a GPU name to the pods to activate once the allocator's
	// store shows that GPU Running. One goroutine serves each entry.
	waiting map[string]map[string]*v1.Pod
	// lastSynced is the GPU event (name and resource version) the store was
	// last found up to date with, so one event is compared with the store once
	// and not once per pending pod.
	lastSynced string
	// activate replaces the framework's pod activator in tests.
	activate func(pods map[string]*v1.Pod)
}

// wakeAfterAllocatorSync makes sure a pod that was woken because a GPU turned
// Running gets a scheduling attempt that can see that GPU.
//
// The queueing hint is driven by the scheduler's own informer, while PreFilter
// reads the allocator's store, which a different informer feeds. Nothing
// orders the two. When the hint is delivered first, the pod is retried
// against a store that still holds the GPU in its previous phase, is rejected
// again, and no later event brings it back: it then waits for the periodic
// flush of the unschedulable queue (5 minutes by default), while the node that
// was added for it sits empty.
//
// The store is compared with the event here, before the hint returns, because
// the retry can only start afterwards. If the store is behind, one goroutine
// per GPU waits until it shows the GPU Running and then activates the pods
// woken for it.
func (s *GPUFit) wakeAfterAllocatorSync(pod *v1.Pod, gpu *tfv1.GPU) {
	nodeName := gpu.Status.NodeSelector[constants.KubernetesHostNameLabel]
	if s.allocator == nil || nodeName == "" {
		return
	}
	state := &s.allocatorSync
	state.mu.Lock()
	defer state.mu.Unlock()

	if pods, waiting := state.waiting[gpu.Name]; waiting {
		pods[string(pod.UID)] = pod
		return
	}
	event := gpu.Name + "/" + gpu.ResourceVersion
	if gpu.ResourceVersion != "" && state.lastSynced == event {
		return
	}
	if s.allocatorHasRunningGPU(gpu.Name, nodeName) {
		state.lastSynced = event
		return
	}
	if state.waiting == nil {
		state.waiting = make(map[string]map[string]*v1.Pod)
	}
	state.waiting[gpu.Name] = map[string]*v1.Pod{string(pod.UID): pod}
	go s.activateAfterAllocatorSync(gpu.Name, nodeName)
}

func (s *GPUFit) activateAfterAllocatorSync(gpuName, nodeName string) {
	timeout, delay, maxDelay := allocatorSyncTimeout, allocatorSyncInitialDelay, allocatorSyncMaxDelay
	deadline := time.Now().Add(timeout)
	synced := false
	for !synced && time.Now().Before(deadline) {
		timer := time.NewTimer(delay)
		select {
		case <-s.pluginDone():
			timer.Stop()
			s.takeAllocatorSyncWaiters(gpuName)
			return
		case <-timer.C:
		}
		if delay < maxDelay {
			delay *= 2
		}
		synced = s.allocatorHasRunningGPU(gpuName, nodeName)
	}

	// Pods are taken only after the store was seen up to date: a pod that
	// joins later is handled by its own comparison in wakeAfterAllocatorSync.
	pods := s.takeAllocatorSyncWaiters(gpuName)
	if !synced {
		// The GPU left the Running phase again, or the allocator's informer
		// is not delivering. The pods stay where they are and are retried by
		// the next GPU event or the periodic flush.
		s.logger.Info("allocator store did not show the GPU Running in time, not activating pods",
			"gpu", gpuName, "node", nodeName, "pods", len(pods), "waited", timeout)
		return
	}
	s.logger.Info("allocator store caught up with GPU Running, activating pods woken for it",
		"gpu", gpuName, "node", nodeName, "pods", len(pods))
	if s.allocatorSync.activate != nil {
		s.allocatorSync.activate(pods)
		return
	}
	if s.fh != nil {
		s.fh.Activate(*s.logger, pods)
	}
}

// pluginDone is closed when the plugin's context ends; nil (never ready) for a
// plugin built without one.
func (s *GPUFit) pluginDone() <-chan struct{} {
	if s.ctx == nil {
		return nil
	}
	return s.ctx.Done()
}

func (s *GPUFit) takeAllocatorSyncWaiters(gpuName string) map[string]*v1.Pod {
	s.allocatorSync.mu.Lock()
	defer s.allocatorSync.mu.Unlock()
	pods := s.allocatorSync.waiting[gpuName]
	delete(s.allocatorSync.waiting, gpuName)
	return pods
}

// allocatorHasRunningGPU reports whether the store PreFilter reads holds the
// GPU in the Running phase. The store applies GPU events in order, so a GPU
// that is Running there also carries the capacity published before.
func (s *GPUFit) allocatorHasRunningGPU(gpuName, nodeName string) bool {
	gpu := s.allocator.GetNodeGpuStore()[nodeName][gpuName]
	return gpu != nil && gpu.Status.Phase == tfv1.TensorFusionGPUPhaseRunning
}

// nodeQueueingHint re-queues a pod when the hypervisor's device plugins come
// back on a node. Filter rejects a node whose tensor-fusion.ai/index_0
// allocatable is present but not positive, and the kubelet restores that
// figure on its own schedule, independent of any GPU resource event. Without
// this hint a pod rejected for that reason would wait for the periodic flush
// of the unschedulable queue.
//
// Only the transition Filter depends on counts: the resource was registered
// with nothing allocatable and is positive now. A node that registers it for
// the first time was never rejected for it.
func (s *GPUFit) nodeQueueingHint(logger klog.Logger, pod *v1.Pod, oldObj, newObj any) (fwk.QueueingHint, error) {
	if !utils.IsTensorFusionWorker(pod) {
		return fwk.QueueSkip, nil
	}
	oldNode, ok := oldObj.(*v1.Node)
	if !ok || oldNode == nil {
		return fwk.QueueSkip, nil
	}
	newNode, ok := newObj.(*v1.Node)
	if !ok || newNode == nil {
		return fwk.QueueSkip, nil
	}
	oldIndex, registered := podIndexAllocatable(oldNode)
	newIndex, _ := podIndexAllocatable(newNode)
	if !registered || oldIndex > 0 || newIndex <= 0 {
		return fwk.QueueSkip, nil
	}
	if !nodeMayServePool(newNode, pod.Annotations[constants.GpuPoolKey]) {
		logger.V(5).Info("Node index resources are allocatable again, but the node belongs to another pool, skip",
			"pod", klog.KObj(pod), "node", newNode.Name)
		return fwk.QueueSkip, nil
	}
	logger.V(4).Info("Node index resources are allocatable again, requeue unscheduled pod",
		"pod", klog.KObj(pod), "node", newNode.Name)
	return fwk.Queue, nil
}

// podIndexAllocatable returns the node's tensor-fusion.ai/index_0 allocatable
// and whether the resource is registered on the node at all.
func podIndexAllocatable(node *v1.Node) (int64, bool) {
	idx0Key := v1.ResourceName(constants.PodIndexAnnotation + constants.PodIndexDelimiter + "0")
	quantity, ok := node.Status.Allocatable[idx0Key]
	return quantity.Value(), ok
}

// nodeMayServePool is false only when the node is labelled for GPU pools and
// none of them is the pod's. A node without pool labels is not ruled out.
func nodeMayServePool(node *v1.Node, pool string) bool {
	labelled := false
	for key := range node.Labels {
		if !strings.HasPrefix(key, constants.GPUNodePoolIdentifierLabelPrefix) {
			continue
		}
		if pool != "" && key == constants.GPUNodePoolIdentifierLabelPrefix+pool {
			return true
		}
		labelled = true
	}
	return !labelled
}
