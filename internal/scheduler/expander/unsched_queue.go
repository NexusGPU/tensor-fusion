package expander

import (
	"context"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// BufferDuration is the time to wait before processing expansion
	BufferDuration = 10 * time.Second
)

type queuedPod struct {
	pod       *corev1.Pod
	queueTime time.Time
	framework framework.Framework
}

type UnscheduledPodHandler struct {
	mu           sync.RWMutex
	pending      map[string]*corev1.Pod
	queue        chan *queuedPod
	logger       klog.Logger
	ctx          context.Context
	nodeExpander *NodeExpander
}

func NewUnscheduledPodHandler(ctx context.Context, nodeExpander *NodeExpander) *UnscheduledPodHandler {
	h := &UnscheduledPodHandler{
		pending:      make(map[string]*corev1.Pod),
		queue:        make(chan *queuedPod, 100), // Buffered channel for queue
		logger:       log.FromContext(ctx).WithValues("component", "expander"),
		ctx:          ctx,
		nodeExpander: nodeExpander,
	}

	// Start the queue processor
	go h.processQueue()

	return h
}

func (h *UnscheduledPodHandler) HandleRejectedPod(ctx context.Context, fwk framework.Framework, podInfo *framework.QueuedPodInfo, status *fwk.Status) {
	pod := podInfo.Pod
	if !utils.IsTensorFusionWorker(pod) {
		return
	}

	h.mu.Lock()
	h.pending[string(pod.UID)] = pod
	h.mu.Unlock()

	h.logger.Info("TensorFusion pod rejected, queuing for buffered expansion", "pod", klog.KObj(pod))

	// Enqueue the pod for buffered processing
	select {
	case h.queue <- &queuedPod{
		pod:       pod,
		queueTime: time.Now(),
		framework: fwk,
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
			h.logger.Info("Queue processor shutting down")
			return
		}
	}
}

// processQueuedPod handles a single queued pod with buffer delay
func (h *UnscheduledPodHandler) processQueuedPod(qp *queuedPod) {
	// Calculate remaining buffer time
	elapsed := time.Since(qp.queueTime)
	remainingBuffer := BufferDuration - elapsed

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

	h.processExpansion(qp)
}

// processExpansion handles the actual expansion logic
func (h *UnscheduledPodHandler) processExpansion(qp *queuedPod) {
	defer h.removePendingPod(qp.pod)

	h.logger.Info("Processing expansion for TensorFusion pod after buffer",
		"pod", klog.KObj(qp.pod),
		"bufferTime", BufferDuration)

	// Call the node expander to process expansion
	if h.nodeExpander != nil {
		if err := h.nodeExpander.ProcessExpansion(h.ctx, qp.framework, qp.pod); err != nil {
			h.logger.Error(err, "Failed to process node expansion after buffer",
				"pod", klog.KObj(qp.pod))
		} else {
			h.logger.V(5).Info("Successfully processed node expansion after buffer",
				"pod", klog.KObj(qp.pod))
		}
	} else {
		h.logger.Error(nil, "NodeExpander is nil, cannot process expansion",
			"pod", klog.KObj(qp.pod))
	}
}

// removePendingPod removes a pod from the pending map
func (h *UnscheduledPodHandler) removePendingPod(pod *corev1.Pod) {
	h.mu.Lock()
	delete(h.pending, string(pod.UID))
	h.mu.Unlock()
}
