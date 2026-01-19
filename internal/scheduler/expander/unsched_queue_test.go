package expander

import (
	"context"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
)

// createTestPod creates a test pod with configurable labels
func createTestPod(name, namespace string, uid types.UID, labels map[string]string, nodeName string) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			UID:       uid,
			Labels:    labels,
		},
		Spec: corev1.PodSpec{
			NodeName: nodeName,
		},
	}
	return pod
}

// createTensorFusionWorkerPod creates a TensorFusion worker pod
func createTensorFusionWorkerPod(name, namespace string, uid types.UID) *corev1.Pod {
	return createTestPod(name, namespace, uid, map[string]string{
		constants.LabelComponent: constants.ComponentWorker,
	}, "")
}

// createNonTensorFusionPod creates a regular pod without TensorFusion labels
func createNonTensorFusionPod(name, namespace string, uid types.UID) *corev1.Pod {
	return createTestPod(name, namespace, uid, map[string]string{}, "")
}

// createDesignatedNodePod creates a pod with a specific node name
func createDesignatedNodePod(name, namespace string, uid types.UID, nodeName string) *corev1.Pod {
	return createTestPod(name, namespace, uid, map[string]string{
		constants.LabelComponent: constants.ComponentWorker,
	}, nodeName)
}

// createTestHandler creates a minimal UnscheduledPodHandler for testing
func createTestHandler(ctx context.Context) *UnscheduledPodHandler {
	return &UnscheduledPodHandler{
		pending:      make(map[string]*corev1.Pod),
		queue:        make(chan *queuedPod, 256),
		logger:       log.FromContext(ctx).WithValues("component", "test-expander"),
		ctx:          ctx,
		nodeExpander: nil,
	}
}

var _ = Describe("UnscheduledPodHandler", func() {
	var (
		ctx    context.Context
		cancel context.CancelFunc
	)

	BeforeEach(func() {
		ctx, cancel = context.WithCancel(context.Background())
	})

	AfterEach(func() {
		cancel()
	})

	Describe("HandleRejectedPod", func() {
		It("should ignore non-TensorFusion pods", func() {
			handler := createTestHandler(ctx)
			pod := createNonTensorFusionPod("regular-pod", "default", "uid-1")
			podInfo := &framework.QueuedPodInfo{
				PodInfo: &framework.PodInfo{Pod: pod},
			}

			handler.HandleRejectedPod(ctx, podInfo, nil)

			handler.mu.RLock()
			_, exists := handler.pending[string(pod.UID)]
			handler.mu.RUnlock()
			Expect(exists).To(BeFalse(), "non-TensorFusion pod should not be added to pending map")

			select {
			case <-handler.queue:
				Fail("Queue should be empty for non-TensorFusion pod")
			default:
				// Expected - queue is empty
			}
		})

		It("should ignore pods with designated node", func() {
			handler := createTestHandler(ctx)
			pod := createDesignatedNodePod("designated-pod", "default", "uid-2", "specific-node")
			podInfo := &framework.QueuedPodInfo{
				PodInfo: &framework.PodInfo{Pod: pod},
			}

			handler.HandleRejectedPod(ctx, podInfo, nil)

			handler.mu.RLock()
			_, exists := handler.pending[string(pod.UID)]
			handler.mu.RUnlock()
			Expect(exists).To(BeFalse(), "designated-node pod should not be added to pending map")
		})

		It("should deduplicate pods with same UID", func() {
			handler := createTestHandler(ctx)
			pod := createTensorFusionWorkerPod("worker-pod", "default", "uid-3")
			podInfo := &framework.QueuedPodInfo{
				PodInfo: &framework.PodInfo{Pod: pod},
			}

			// First call - should add to pending
			handler.HandleRejectedPod(ctx, podInfo, nil)

			handler.mu.RLock()
			_, exists := handler.pending[string(pod.UID)]
			handler.mu.RUnlock()
			Expect(exists).To(BeTrue(), "first call should add to pending")

			// Drain the queue
			select {
			case <-handler.queue:
			default:
			}

			// Second call with same UID - should be ignored
			handler.HandleRejectedPod(ctx, podInfo, nil)

			select {
			case <-handler.queue:
				Fail("Duplicate UID should not be queued again")
			default:
				// Expected - no new item queued
			}
		})

		It("should add pod to queue and pending map", func() {
			handler := createTestHandler(ctx)
			pod := createTensorFusionWorkerPod("worker-pod", "default", "uid-4")
			podInfo := &framework.QueuedPodInfo{
				PodInfo: &framework.PodInfo{Pod: pod},
			}

			handler.HandleRejectedPod(ctx, podInfo, nil)

			handler.mu.RLock()
			pendingPod, exists := handler.pending[string(pod.UID)]
			handler.mu.RUnlock()
			Expect(exists).To(BeTrue(), "pod should be added to pending map")
			Expect(pendingPod.Name).To(Equal(pod.Name))

			select {
			case qp := <-handler.queue:
				Expect(qp.pod.Name).To(Equal(pod.Name))
				Expect(qp.queueTime.IsZero()).To(BeFalse(), "queue time should be set")
			default:
				Fail("Pod should be in queue")
			}
		})

		It("should drop pod when queue is full", func() {
			handler := &UnscheduledPodHandler{
				pending: make(map[string]*corev1.Pod),
				queue:   make(chan *queuedPod, 1),
				logger:  log.FromContext(ctx).WithValues("component", "test-expander"),
				ctx:     ctx,
			}

			// Fill the queue
			firstPod := createTensorFusionWorkerPod("worker-1", "default", "uid-fill")
			handler.HandleRejectedPod(ctx, &framework.QueuedPodInfo{
				PodInfo: &framework.PodInfo{Pod: firstPod},
			}, nil)

			// Try to add another pod when queue is full
			secondPod := createTensorFusionWorkerPod("worker-2", "default", "uid-drop")
			handler.HandleRejectedPod(ctx, &framework.QueuedPodInfo{
				PodInfo: &framework.PodInfo{Pod: secondPod},
			}, nil)

			handler.mu.RLock()
			_, exists := handler.pending[string(secondPod.UID)]
			handler.mu.RUnlock()
			Expect(exists).To(BeFalse(), "dropped pod should be removed from pending")
		})

		It("should handle context cancellation", func() {
			cancelCtx, cancelFunc := context.WithCancel(context.Background())

			handler := &UnscheduledPodHandler{
				pending: make(map[string]*corev1.Pod),
				queue:   make(chan *queuedPod), // Unbuffered - will block
				logger:  log.FromContext(cancelCtx).WithValues("component", "test-expander"),
				ctx:     cancelCtx,
			}

			pod := createTensorFusionWorkerPod("worker-pod", "default", "uid-ctx")

			var wg sync.WaitGroup
			wg.Add(1)
			go func() {
				defer wg.Done()
				handler.HandleRejectedPod(cancelCtx, &framework.QueuedPodInfo{
					PodInfo: &framework.PodInfo{Pod: pod},
				}, nil)
			}()

			time.Sleep(10 * time.Millisecond)
			cancelFunc()
			wg.Wait()

			handler.mu.RLock()
			_, exists := handler.pending[string(pod.UID)]
			handler.mu.RUnlock()
			Expect(exists).To(BeFalse(), "pod should be removed from pending after context cancellation")
		})
	})

	Describe("removePendingPod", func() {
		It("should remove pod from pending map", func() {
			handler := createTestHandler(ctx)

			pod := createTensorFusionWorkerPod("worker-pod", "default", "uid-remove")
			handler.mu.Lock()
			handler.pending[string(pod.UID)] = pod
			handler.mu.Unlock()

			handler.mu.RLock()
			_, exists := handler.pending[string(pod.UID)]
			handler.mu.RUnlock()
			Expect(exists).To(BeTrue())

			handler.removePendingPod(pod)

			handler.mu.RLock()
			_, exists = handler.pending[string(pod.UID)]
			handler.mu.RUnlock()
			Expect(exists).To(BeFalse(), "pod should be removed from pending")
		})
	})

	Describe("Buffer Time Calculation", func() {
		It("should calculate negative remaining buffer for old queue time", func() {
			now := time.Now()
			qp := &queuedPod{
				pod:       createTensorFusionWorkerPod("worker-pod", "default", "uid-buffer"),
				queueTime: now.Add(-constants.UnschedQueueBufferDuration - time.Second),
			}

			elapsed := time.Since(qp.queueTime)
			remainingBuffer := constants.UnschedQueueBufferDuration - elapsed

			Expect(remainingBuffer < 0).To(BeTrue(), "remaining buffer should be negative when queue time is old")
		})

		It("should calculate positive remaining buffer for recent queue time", func() {
			now := time.Now()
			qp := &queuedPod{
				pod:       createTensorFusionWorkerPod("worker-pod-2", "default", "uid-buffer-2"),
				queueTime: now,
			}

			elapsed := time.Since(qp.queueTime)
			remainingBuffer := constants.UnschedQueueBufferDuration - elapsed

			Expect(remainingBuffer > 0).To(BeTrue(), "remaining buffer should be positive when queue time is recent")
		})
	})

	Describe("BufferDurationConstant", func() {
		It("should have reasonable buffer duration", func() {
			Expect(constants.UnschedQueueBufferDuration > 0).To(BeTrue(), "buffer duration should be positive")
			Expect(constants.UnschedQueueBufferDuration < time.Minute).To(BeTrue(), "buffer duration should be less than a minute")
		})
	})

	Describe("queuedPod struct", func() {
		It("should store pod and queue time correctly", func() {
			pod := createTensorFusionWorkerPod("test-pod", "default", "uid-struct")
			now := time.Now()

			qp := &queuedPod{
				pod:       pod,
				queueTime: now,
			}

			Expect(qp.pod.Name).To(Equal(pod.Name))
			Expect(qp.queueTime).To(Equal(now))
		})
	})

	Describe("Concurrent Access", func() {
		It("should handle concurrent pending map access safely", func() {
			handler := createTestHandler(ctx)

			var wg sync.WaitGroup
			numGoroutines := 10

			for i := 0; i < numGoroutines; i++ {
				wg.Add(1)
				go func(goroutineID int) {
					defer wg.Done()
					uid := types.UID(time.Now().String() + string(rune('a'+goroutineID)))
					pod := createTensorFusionWorkerPod("worker", "default", uid)
					handler.HandleRejectedPod(ctx, &framework.QueuedPodInfo{
						PodInfo: &framework.PodInfo{Pod: pod},
					}, nil)
				}(i)
			}

			wg.Wait()

			itemCount := 0
			for {
				select {
				case <-handler.queue:
					itemCount++
				default:
					goto done
				}
			}
		done:

			Expect(itemCount > 0).To(BeTrue(), "should have queued some pods")
			Expect(itemCount <= numGoroutines).To(BeTrue(), "should not have more items than goroutines")
		})
	})

	Describe("Handler Initialization", func() {
		It("should initialize handler correctly", func() {
			handler := createTestHandler(ctx)

			Expect(handler.pending).NotTo(BeNil(), "pending map should be initialized")
			Expect(handler.queue).NotTo(BeNil(), "queue should be initialized")
			Expect(handler.ctx).To(Equal(ctx), "context should be set")
		})
	})
})
