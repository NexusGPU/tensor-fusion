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

package kubernetes

import (
	"context"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
)

// workerInfoSubscriber represents a subscriber waiting for worker info for a specific pod index
type workerInfoSubscriber struct {
	ch chan *api.WorkerInfo
}

const subscriberTimeout = 10 * time.Minute

// PodCacheManager manages pod watching and worker information extraction
type PodCacheManager struct {
	ctx        context.Context
	clientset  *kubernetes.Clientset
	restConfig *rest.Config
	nodeName   string

	mu                sync.RWMutex
	cachedPod         map[string]*corev1.Pod  // key: pod UID
	indexToWorkerInfo map[int]*api.WorkerInfo // key: pod index annotation

	stopCh          chan struct{}
	workerChangedCh chan struct{}

	// Pub/Sub mechanism for waiting on worker info by index
	subscribersMu    sync.RWMutex
	indexSubscribers map[int]map[*workerInfoSubscriber]struct{} // key: pod index

	podSubscribersMu sync.RWMutex
	podSubscribers   map[string]chan<- *api.WorkerInfo
}

// NewPodCacheManager creates a new pod cache manager
func NewPodCacheManager(ctx context.Context, restConfig *rest.Config, nodeName string) (*PodCacheManager, error) {
	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes clientset: %w", err)
	}

	kc := &PodCacheManager{
		ctx:               ctx,
		clientset:         clientset,
		restConfig:        restConfig,
		nodeName:          nodeName,
		cachedPod:         make(map[string]*corev1.Pod, 32),
		indexToWorkerInfo: make(map[int]*api.WorkerInfo, 32),
		stopCh:            make(chan struct{}),
		workerChangedCh:   make(chan struct{}, 1),
		indexSubscribers:  make(map[int]map[*workerInfoSubscriber]struct{}),
		podSubscribers:    make(map[string]chan<- *api.WorkerInfo),
	}

	// Start the Pub/Sub event bus goroutine
	go kc.runWorkerChangeEventBus()

	return kc, nil
}

// Start starts watching pods on this node
func (kc *PodCacheManager) Start() error {
	// Create a field selector to watch only pods on this node
	fieldSelector := fields.OneTermEqualSelector("spec.nodeName", kc.nodeName).String()

	// Create a label selector for pods with tensor-fusion.ai/enabled=true
	labelSelector := labels.Set{
		constants.TensorFusionEnabledLabelKey: constants.TrueStringValue,
	}.AsSelector().String()

	// Create list watcher
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fieldSelector
			options.LabelSelector = labelSelector
			return kc.clientset.CoreV1().Pods(metav1.NamespaceAll).List(kc.ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fieldSelector
			options.LabelSelector = labelSelector
			return kc.clientset.CoreV1().Pods(metav1.NamespaceAll).Watch(kc.ctx, options)
		},
	}

	// Create informer
	_, controller := cache.NewInformerWithOptions(cache.InformerOptions{
		ListerWatcher: lw,
		ObjectType:    &corev1.Pod{},
		ResyncPeriod:  0,
		Handler: cache.ResourceEventHandlerFuncs{
			AddFunc:    kc.onPodAdd,
			UpdateFunc: kc.onPodUpdate,
			DeleteFunc: kc.onPodDelete,
		},
	})

	// Start the informer
	go controller.Run(kc.stopCh)

	klog.Infof("Started watching pods on node %s with label %s=%s", kc.nodeName, constants.TensorFusionEnabledLabelKey, constants.TrueStringValue)
	return nil
}

// Stop stops the pod cache manager
func (kc *PodCacheManager) Stop() {
	klog.Info("Stopping pod cache manager")
	close(kc.stopCh)
}

// onPodAdd handles pod addition events
func (kc *PodCacheManager) onPodAdd(obj any) {
	pod := obj.(*corev1.Pod)
	kc.mu.Lock()
	defer kc.mu.Unlock()
	kc.cachedPod[string(pod.UID)] = pod

	_, deviceAllocated := pod.Annotations[constants.PodDeviceAllocatedAnnotation]

	if podIndexAnno, exists := pod.Annotations[constants.PodIndexAnnotation]; exists {
		if podIndex, err := strconv.Atoi(podIndexAnno); err == nil {
			// Parse and store WorkerInfo
			workerInfo := kc.extractWorkerInfo(pod, podIndexAnno)
			kc.notifyWorkerChanged(workerInfo)
			if !deviceAllocated {
				kc.indexToWorkerInfo[podIndex] = workerInfo
				klog.Infof("Pod %s/%s added to pending allocation index %d", pod.Namespace, pod.Name, podIndex)
			}
		} else {
			klog.Errorf("Pod %s/%s has invalid index annotation: %s", pod.Namespace, pod.Name, podIndexAnno)
		}
	} else {
		klog.Infof("Pod %s/%s has no index annotation, waiting for index to be updated", pod.Namespace, pod.Name)
	}
	kc.checkWorkerPendingIndexChanged()
}

// onPodUpdate handles pod update events
func (kc *PodCacheManager) onPodUpdate(oldObj, newObj any) {
	newPod := newObj.(*corev1.Pod)

	kc.mu.Lock()
	defer kc.mu.Unlock()
	kc.cachedPod[string(newPod.UID)] = newPod

	// Handle old index if it changed
	podIndexAnno, indexExists := newPod.Annotations[constants.PodIndexAnnotation]
	_, alreadyAllocated := newPod.Annotations[constants.PodDeviceAllocatedAnnotation]

	// Update WorkerInfo cache if pod has index annotation
	// scheduler PostBind will ensure this index only exists when no index conflict on same node
	if indexExists {
		if podIndex, err := strconv.Atoi(podIndexAnno); err == nil {
			// Parse and store WorkerInfo
			workerInfo := kc.extractWorkerInfo(newPod, podIndexAnno)
			kc.notifyWorkerChanged(workerInfo)
			if !alreadyAllocated {
				kc.indexToWorkerInfo[podIndex] = workerInfo
				klog.Infof("Pod %s/%s (UID: %s) added to pending allocation index %d", newPod.Namespace, newPod.Name, newPod.UID, podIndex)
			}
		}
	}
	klog.Infof("Pod %s/%s (UID: %s) updated, index: %s, allocated: %t", newPod.Namespace, newPod.Name, newPod.UID, podIndexAnno, alreadyAllocated)
	kc.checkWorkerPendingIndexChanged()
}

// onPodDelete handles pod deletion events
func (kc *PodCacheManager) onPodDelete(obj any) {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		// Handle deleted final state unknown
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.Errorf("Unexpected object type, can not parsed to Pod: %T", obj)
			return
		}
		pod, ok = tombstone.Obj.(*corev1.Pod)
		if !ok {
			klog.Errorf("Tombstone contained object that is not a pod: %T", tombstone.Obj)
			return
		}
	}

	kc.mu.Lock()
	defer kc.mu.Unlock()
	podUID := string(pod.UID)
	delete(kc.cachedPod, podUID)
	// Clean up WorkerInfo cache if pod had index annotation
	if podIndexAnno, exists := pod.Annotations[constants.PodIndexAnnotation]; exists {
		if podIndex, err := strconv.Atoi(podIndexAnno); err == nil {
			workerInfo := kc.extractWorkerInfo(pod, podIndexAnno)
			workerInfo.DeletedAt = time.Now()
			kc.notifyWorkerChanged(workerInfo)

			if _, exists := kc.indexToWorkerInfo[podIndex]; exists {
				delete(kc.indexToWorkerInfo, podIndex)
				klog.Infof("Pod %s/%s (UID: %s) removed from pending allocation index %d", pod.Namespace, pod.Name, pod.UID, podIndex)
			}
		}
	}
	klog.V(4).Infof("Pod deleted: %s/%s (UID: %s)", pod.Namespace, pod.Name, pod.UID)
	kc.checkWorkerPendingIndexChanged()
}

// checkWorkerPendingIndexChanged notifies that worker information has changed
func (kc *PodCacheManager) checkWorkerPendingIndexChanged() {
	select {
	case kc.workerChangedCh <- struct{}{}:
	default:
		// Channel is full, skip notification (non-blocking)
	}
}

// runWorkerChangeEventBus runs a standalone goroutine that consumes workerChangedCh
// and notifies all subscribers when worker information changes for their requested index
func (kc *PodCacheManager) runWorkerChangeEventBus() {
	for {
		select {
		case <-kc.stopCh:
			return
		case <-kc.ctx.Done():
			return
		case <-kc.workerChangedCh:
			// Worker information changed, check if any subscribers are waiting
			kc.notifySubscribers()
		}
	}
}

// notifySubscribers checks all subscribers and sends worker info if available
func (kc *PodCacheManager) notifySubscribers() {
	kc.subscribersMu.Lock()
	defer kc.subscribersMu.Unlock()

	kc.mu.RLock()
	defer kc.mu.RUnlock()

	// Iterate through all subscribed indices
	for podIndex, subs := range kc.indexSubscribers {
		// Check if worker info is now available for this index
		if workerInfo, exists := kc.indexToWorkerInfo[podIndex]; exists && workerInfo != nil {
			// Notify all subscribers for this index
			for sub := range subs {
				select {
				case sub.ch <- workerInfo:
					// Successfully sent, remove subscriber
					delete(subs, sub)
					close(sub.ch)
				default:
					// Channel is full or closed, skip
				}
			}
			// Clean up empty subscriber set
			if len(subs) == 0 {
				delete(kc.indexSubscribers, podIndex)
			}
		}
	}
}

func (kc *PodCacheManager) notifyWorkerChanged(workerInfo *api.WorkerInfo) {
	kc.podSubscribersMu.Lock()
	defer kc.podSubscribersMu.Unlock()
	for _, subscriber := range kc.podSubscribers {
		select {
		case subscriber <- workerInfo:
		default:
			// Channel is full or closed, skip
		}
	}
}

func (kc *PodCacheManager) RegisterWorkerInfoSubscriber(name string, subscriber chan<- *api.WorkerInfo) {
	kc.podSubscribersMu.Lock()
	defer kc.podSubscribersMu.Unlock()
	if _, exists := kc.podSubscribers[name]; exists {
		klog.Errorf("Worker info subscriber for %s already registered", name)
		return
	}
	kc.podSubscribers[name] = subscriber
	klog.Infof("Registered worker info subscriber for %s", name)
}

func (kc *PodCacheManager) UnregisterWorkerInfoSubscriber(name string) {
	kc.podSubscribersMu.Lock()
	defer kc.podSubscribersMu.Unlock()
	delete(kc.podSubscribers, name)
	klog.Infof("Unregistered worker info subscriber for %s", name)
}

// GetWorkerInfoForAllocationByIndex finds a pod by its index annotation and extracts worker info
// It implements a Pub/Sub pattern where callers subscribe to worker info changes for a specific pod index.
// If worker info is already available, it returns immediately. Otherwise, it waits for up to 10 minutes
// for the worker info to become available.
func (kc *PodCacheManager) GetWorkerInfoForAllocationByIndex(podIndex int) (*api.WorkerInfo, error) {
	kc.subscribersMu.Lock()
	defer kc.subscribersMu.Unlock()
	// First, check if worker info is already available (fast path)

	kc.mu.RLock()
	if workerInfo, exists := kc.indexToWorkerInfo[podIndex]; exists && workerInfo != nil {
		kc.mu.RUnlock()
		return workerInfo, nil
	}
	kc.mu.RUnlock()

	// Worker info not available yet, subscribe to changes
	subscriber := &workerInfoSubscriber{
		ch: make(chan *api.WorkerInfo, 1),
	}

	// Register subscriber
	if _, exists := kc.indexSubscribers[podIndex]; !exists {
		kc.indexSubscribers[podIndex] = make(map[*workerInfoSubscriber]struct{})
	}
	kc.indexSubscribers[podIndex][subscriber] = struct{}{}

	timeoutTimer := time.NewTimer(subscriberTimeout)
	defer timeoutTimer.Stop()

	select {
	case workerInfo := <-subscriber.ch:
		// Worker info received
		if workerInfo == nil {
			return nil, fmt.Errorf("worker info channel closed for pod index %d", podIndex)
		}
		return workerInfo, nil
	case <-timeoutTimer.C:
		// Timeout reached
		kc.unregisterSubscriber(podIndex, subscriber)
		return nil, fmt.Errorf("timeout waiting for worker info for pod index %d after %v", podIndex, subscriberTimeout)
	case <-kc.ctx.Done():
		// Context cancelled
		kc.unregisterSubscriber(podIndex, subscriber)
		return nil, fmt.Errorf("context cancelled while waiting for worker info for pod index %d", podIndex)
	case <-kc.stopCh:
		// Pod cache manager stopped
		kc.unregisterSubscriber(podIndex, subscriber)
		return nil, fmt.Errorf("pod cache manager stopped while waiting for worker info for pod index %d", podIndex)
	}
}

// unregisterSubscriber removes a subscriber from the subscribers map
func (kc *PodCacheManager) unregisterSubscriber(podIndex int, sub *workerInfoSubscriber) {
	kc.subscribersMu.Lock()
	defer kc.subscribersMu.Unlock()

	if subs, exists := kc.indexSubscribers[podIndex]; exists {
		if _, stillSubscribed := subs[sub]; stillSubscribed {
			delete(subs, sub)
			// Close channel - safe because we just removed it from map, so event bus won't close it
			close(sub.ch)
		}
		// Clean up empty subscriber set
		if len(subs) == 0 {
			delete(kc.indexSubscribers, podIndex)
		}
	}
}

// GetPodByUID retrieves a pod from the cache by its UID
func (kc *PodCacheManager) GetPodByUID(podUID string) *corev1.Pod {
	kc.mu.RLock()
	defer kc.mu.RUnlock()
	return kc.cachedPod[podUID]
}

// extractWorkerInfo extracts worker information from pod annotations using the common utility function
func (kc *PodCacheManager) extractWorkerInfo(pod *corev1.Pod, podIndex string) *api.WorkerInfo {
	// Use common utility function to extract pod worker info
	allocRequest, msg, err := utils.ComposeAllocationRequest(kc.ctx, pod)
	if err != nil {
		klog.Error(err, "Failed to compose allocation request for existing worker Pod, annotation may not be valid", "pod", pod.Name, "msg", msg)
		return nil
	}
	info := &api.WorkerInfo{
		PodUID:            string(pod.UID),
		PodName:           pod.Name,
		Namespace:         pod.Namespace,
		Annotations:       pod.Annotations,
		PodIndex:          podIndex,
		AllocatedDevices:  allocRequest.GPUNames,
		IsolationMode:     allocRequest.Isolation,
		MemoryLimitBytes:  uint64(allocRequest.Limit.Vram.Value()),
		ComputeLimitUnits: uint32(allocRequest.Limit.ComputePercent.Value()),
		TemplateID:        allocRequest.PartitionTemplateID,
	}

	return info
}

// GetAllPods returns all pods currently in the cache
func (kc *PodCacheManager) GetAllPods() map[string]*corev1.Pod {
	kc.mu.RLock()
	defer kc.mu.RUnlock()

	result := make(map[string]*corev1.Pod, len(kc.cachedPod))
	for k, v := range kc.cachedPod {
		result[k] = v
	}
	return result
}
