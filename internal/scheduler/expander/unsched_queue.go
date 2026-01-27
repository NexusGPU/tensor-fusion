package expander

import (
	"context"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type queuedPod struct {
	pod       *corev1.Pod
	queueTime time.Time
}

type UnscheduledPodHandler struct {
	mu           sync.RWMutex
	pending      map[string]*corev1.Pod
	queue        chan *queuedPod
	logger       klog.Logger
	ctx          context.Context
	nodeExpander *NodeExpander
}

func NewUnscheduledPodHandler(ctx context.Context, scheduler *scheduler.Scheduler,
	allocator *gpuallocator.GpuAllocator, recorder events.EventRecorder) (*UnscheduledPodHandler, *NodeExpander) {
	nodeExpander := NewNodeExpander(ctx, allocator, scheduler, recorder)
	h := &UnscheduledPodHandler{
		pending:      make(map[string]*corev1.Pod),
		queue:        make(chan *queuedPod, 256),
		logger:       log.FromContext(ctx).WithValues("component", "expander"),
		ctx:          ctx,
		nodeExpander: nodeExpander,
	}

	// Start the queue processor
	go h.processQueue()

	return h, nodeExpander
}

func (h *UnscheduledPodHandler) HandleRejectedPod(ctx context.Context, podInfo fwk.QueuedPodInfo, status *fwk.Status) {
	pod := podInfo.GetPodInfo().GetPod()
	if !utils.IsTensorFusionWorker(pod) {
		return
	}

	if utils.IsDesignatedNodePod(pod) {
		h.logger.Info("Pod has selected the fixed node in nodeSelector/nodeName/nodeAffinity, skipping expansion", "pod", klog.KObj(pod))
		return
	}

	// here if  pod has NominatedNodeName,we should not expand the node,skip
	// Let Kubernetes scheduler handle the preemption process
	if pod.Status.NominatedNodeName != "" {
		h.logger.V(4).Info("Pod has nominated node from preemption, skipping expansion",
			"pod", klog.KObj(pod), "nominatedNode", pod.Status.NominatedNodeName)
		return
	}

	// take snapshot to avoid modify origin Pod info
	pod = pod.DeepCopy()

	h.mu.Lock()
	if _, ok := h.pending[string(pod.UID)]; ok {
		h.mu.Unlock()
		return
	}
	h.pending[string(pod.UID)] = pod
	h.mu.Unlock()

	h.logger.Info("TensorFusion pod rejected, queuing for buffered expansion", "pod", klog.KObj(pod))

	// Enqueue the pod for buffered processing
	select {
	case h.queue <- &queuedPod{
		pod:       pod,
		queueTime: time.Now(),
	}:
		h.logger.V(2).Info("Pod successfully queued for expansion", "pod", klog.KObj(pod))
	case <-ctx.Done():
		h.logger.Info("Context cancelled while queuing pod", "pod", klog.KObj(pod))
		h.mu.Lock()
		delete(h.pending, string(pod.UID))
		h.mu.Unlock()
	default:
		h.logger.Error(nil, "Queue is full, dropping pod", "pod", klog.KObj(pod))
		h.mu.Lock()
		delete(h.pending, string(pod.UID))
		h.mu.Unlock()
	}
}

// processQueue continuously processes queued pods with buffer delay
func (h *UnscheduledPodHandler) processQueue() {
	h.logger.Info("Starting queue processor for unscheduled pods")

	for {
		select {
		case queuedPod := <-h.queue:
			h.processQueuedPod(queuedPod)
		case <-h.ctx.Done():
			h.logger.Info("Pending pod queue processor shutting down")
			return
		}
	}
}

// processQueuedPod handles a single queued pod with buffer delay
func (h *UnscheduledPodHandler) processQueuedPod(qp *queuedPod) {
	// Calculate remaining buffer time
	elapsed := time.Since(qp.queueTime)
	remainingBuffer := constants.UnschedQueueBufferDuration - elapsed

	if remainingBuffer > 0 {
		h.logger.V(2).Info("Buffering pod before expansion",
			"pod", klog.KObj(qp.pod),
			"remainingBuffer", remainingBuffer)

		timer := time.NewTimer(remainingBuffer)
		defer timer.Stop()

		select {
		case <-timer.C:
			// Buffer time elapsed, proceed with expansion
		case <-h.ctx.Done():
			h.logger.Info("Context cancelled while buffering pod", "pod", klog.KObj(qp.pod))
			h.removePendingPod(qp.pod)
			return
		}
	}

	if err := h.nodeExpander.ProcessExpansion(h.ctx, qp.pod); err != nil {
		h.logger.Error(err, "Failed to process node expansion after buffer",
			"pod", klog.KObj(qp.pod))
	} else {
		h.logger.V(5).Info("Successfully processed node expansion after buffer",
			"pod", klog.KObj(qp.pod))
	}
	h.removePendingPod(qp.pod)
}

// removePendingPod removes a pod from the pending map
func (h *UnscheduledPodHandler) removePendingPod(pod *corev1.Pod) {
	h.mu.Lock()
	delete(h.pending, string(pod.UID))
	h.mu.Unlock()
}
