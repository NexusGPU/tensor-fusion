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
	"slices"
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
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
)

// PodCacheManager manages pod watching and worker information extraction
type PodCacheManager struct {
	ctx        context.Context
	clientset  *kubernetes.Clientset
	restConfig *rest.Config
	nodeName   string

	mu                sync.RWMutex
	podCache          map[string]*corev1.Pod           // key: pod UID
	allocations       map[string]*api.WorkerAllocation // key: pod UID
	indexToWorkerInfo map[int]*api.WorkerInfo          // key: pod index annotation
	indexToPodList    map[int][]string                 // key: pod index annotation, value: list of pod UIDs
	stopCh            chan struct{}
	workerChangedCh   chan struct{}
}

// NewPodCacheManager creates a new pod cache manager
func NewPodCacheManager(ctx context.Context, restConfig *rest.Config, nodeName string) (*PodCacheManager, error) {
	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes clientset: %w", err)
	}

	return &PodCacheManager{
		ctx:               ctx,
		clientset:         clientset,
		restConfig:        restConfig,
		nodeName:          nodeName,
		podCache:          make(map[string]*corev1.Pod),
		allocations:       make(map[string]*api.WorkerAllocation),
		indexToWorkerInfo: make(map[int]*api.WorkerInfo),
		indexToPodList:    make(map[int][]string),
		stopCh:            make(chan struct{}),
		workerChangedCh:   make(chan struct{}, 1),
	}, nil
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
	close(kc.stopCh)
}

// onPodAdd handles pod addition events
func (kc *PodCacheManager) onPodAdd(obj interface{}) {
	pod := obj.(*corev1.Pod)
	kc.mu.Lock()
	kc.podCache[string(pod.UID)] = pod
	if podIndexAnno, exists := pod.Annotations[constants.PodIndexAnnotation]; exists {
		if podIndex, err := strconv.Atoi(podIndexAnno); err == nil {
			// Parse and store WorkerInfo
			workerInfo := kc.extractWorkerInfo(pod, podIndexAnno)
			kc.indexToWorkerInfo[podIndex] = workerInfo
			// Add pod UID to indexToPodList
			kc.indexToPodList[podIndex] = append(kc.indexToPodList[podIndex], string(pod.UID))
		}
	} else {
		klog.Errorf("Pod %s/%s has no index annotation", pod.Namespace, pod.Name)
	}
	kc.mu.Unlock()

	klog.V(4).Infof("Pod added: %s/%s (UID: %s)", pod.Namespace, pod.Name, pod.UID)
	kc.notifyWorkerChanged()
}

// onPodUpdate handles pod update events
func (kc *PodCacheManager) onPodUpdate(oldObj, newObj interface{}) {
	oldPod := oldObj.(*corev1.Pod)
	newPod := newObj.(*corev1.Pod)

	kc.mu.Lock()
	kc.podCache[string(newPod.UID)] = newPod

	// Handle old index if it changed
	oldPodIndexAnno, oldExists := oldPod.Annotations[constants.PodIndexAnnotation]
	newPodIndexAnno, newExists := newPod.Annotations[constants.PodIndexAnnotation]

	if oldExists {
		if oldPodIndex, err := strconv.Atoi(oldPodIndexAnno); err == nil {
			// Remove pod UID from old index
			kc.removePodFromIndex(oldPodIndex, string(newPod.UID))
		}
	}

	// Update WorkerInfo cache if pod has index annotation
	if newExists {
		if podIndex, err := strconv.Atoi(newPodIndexAnno); err == nil {
			// Parse and store WorkerInfo
			workerInfo := kc.extractWorkerInfo(newPod, newPodIndexAnno)
			kc.indexToWorkerInfo[podIndex] = workerInfo
			// Add pod UID to indexToPodList if not already present
			podUID := string(newPod.UID)
			found := slices.Contains(kc.indexToPodList[podIndex], podUID)
			if !found {
				kc.indexToPodList[podIndex] = append(kc.indexToPodList[podIndex], podUID)
			}
		}
	}
	kc.mu.Unlock()

	klog.V(4).Infof("Pod updated: %s/%s (UID: %s)", newPod.Namespace, newPod.Name, newPod.UID)

	// Check if annotations changed (which might affect allocation)
	if !podAnnotationsEqual(oldPod.Annotations, newPod.Annotations) {
		kc.notifyWorkerChanged()
	}
}

// onPodDelete handles pod deletion events
func (kc *PodCacheManager) onPodDelete(obj interface{}) {
	pod, ok := obj.(*corev1.Pod)
	if !ok {
		// Handle deleted final state unknown
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			klog.Errorf("Unexpected object type: %T", obj)
			return
		}
		pod, ok = tombstone.Obj.(*corev1.Pod)
		if !ok {
			klog.Errorf("Tombstone contained object that is not a pod: %T", tombstone.Obj)
			return
		}
	}

	kc.mu.Lock()
	podUID := string(pod.UID)
	delete(kc.podCache, podUID)
	delete(kc.allocations, podUID)
	// Clean up WorkerInfo cache and indexToPodList if pod had index annotation
	if podIndexAnno, exists := pod.Annotations[constants.PodIndexAnnotation]; exists {
		if podIndex, err := strconv.Atoi(podIndexAnno); err == nil {
			delete(kc.indexToWorkerInfo, podIndex)
			kc.removePodFromIndex(podIndex, podUID)
		}
	}
	kc.mu.Unlock()

	klog.V(4).Infof("Pod deleted: %s/%s (UID: %s)", pod.Namespace, pod.Name, pod.UID)
	kc.notifyWorkerChanged()
}

// removePodFromIndex removes a pod UID from the indexToPodList for a given index
func (kc *PodCacheManager) removePodFromIndex(podIndex int, podUID string) {
	podList := kc.indexToPodList[podIndex]
	newList := make([]string, 0, len(podList))
	for _, uid := range podList {
		if uid != podUID {
			newList = append(newList, uid)
		}
	}
	if len(newList) == 0 {
		delete(kc.indexToPodList, podIndex)
	} else {
		kc.indexToPodList[podIndex] = newList
	}
}

// notifyWorkerChanged notifies that worker information has changed
func (kc *PodCacheManager) notifyWorkerChanged() {
	select {
	case kc.workerChangedCh <- struct{}{}:
	default:
	}
}

// GetWorkerInfoForAllocationByIndex finds a pod by its index annotation and extracts worker info
func (kc *PodCacheManager) GetWorkerInfoForAllocationByIndex(ctx context.Context, podIndex int) (*api.WorkerInfo, error) {
	var workerInfo *api.WorkerInfo
	var lastErr error

	// Retry for at most 5 seconds using k8s retry utility with 10ms backoff
	startTime := time.Now()
	err := retry.OnError(wait.Backoff{
		Duration: 10 * time.Millisecond,
		Factor:   1.4,
		Jitter:   0.1,
		Cap:      5 * time.Second,
	}, func(err error) bool {
		// Check if we've exceeded 5 seconds
		if time.Since(startTime) >= 5*time.Second {
			return false
		}
		// Retry if worker info not found
		return true
	}, func() error {
		kc.mu.RLock()
		defer kc.mu.RUnlock()

		// Check for duplicate index - fast fail if multiple pods have same index
		if podList, exists := kc.indexToPodList[podIndex]; exists {
			if len(podList) > 1 {
				// Build error message with pod details
				var matchingPods []string
				for _, podUID := range podList {
					if pod := kc.podCache[podUID]; pod != nil {
						matchingPods = append(matchingPods, fmt.Sprintf("%s/%s (UID: %s)", pod.Namespace, pod.Name, podUID))
					}
				}
				lastErr = fmt.Errorf("duplicate index %d found in pods: %v", podIndex, matchingPods)
				return lastErr
			}
		}

		// Find worker info with matching index annotation
		if info, exists := kc.indexToWorkerInfo[podIndex]; exists {
			workerInfo = info
			return nil // Success, stop retrying
		}

		lastErr = fmt.Errorf("worker info not found for pod index %d", podIndex)
		return lastErr // Return error to trigger retry
	})

	if err != nil {
		return nil, fmt.Errorf("worker info not found for pod index %d after retrying for 5 seconds: %w", podIndex, err)
	}

	return workerInfo, nil
}

// GetPodByUID retrieves a pod from the cache by its UID
func (kc *PodCacheManager) GetPodByUID(podUID string) *corev1.Pod {
	kc.mu.RLock()
	defer kc.mu.RUnlock()
	return kc.podCache[podUID]
}

// RemovePodIndexAnnotation removes the PodIndexAnnotation from a pod after successful allocation
func (kc *PodCacheManager) RemovePodIndexAnnotation(ctx context.Context, podUID string, namespace string, podName string) error {
	kc.mu.RLock()
	pod, exists := kc.podCache[podUID]
	kc.mu.RUnlock()

	// TODO: too complex, just a raw patch should work! and delete pod_cache before calling apiserver API

	if !exists {
		return fmt.Errorf("pod %s/%s not found in cache", namespace, podName)
	}

	// Check if annotation exists
	if pod.Annotations == nil {
		return nil // Nothing to remove
	}

	if _, exists := pod.Annotations[constants.PodIndexAnnotation]; !exists {
		return nil // Annotation already removed
	}

	// Use API client to patch pod and remove annotation
	// Get fresh pod from API server
	currentPod, err := kc.clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get pod %s/%s: %w", namespace, podName, err)
	}

	// Create patch to remove annotation
	if currentPod.Annotations == nil {
		return nil // No annotations to remove
	}

	if _, exists := currentPod.Annotations[constants.PodIndexAnnotation]; !exists {
		return nil // Annotation already removed
	}

	// Remove annotation
	delete(currentPod.Annotations, constants.PodIndexAnnotation)

	// Update pod
	_, err = kc.clientset.CoreV1().Pods(namespace).Update(ctx, currentPod, metav1.UpdateOptions{})
	if err != nil {
		return fmt.Errorf("failed to update pod %s/%s: %w", namespace, podName, err)
	}

	klog.Infof("Successfully removed PodIndexAnnotation from pod %s/%s", namespace, podName)
	return nil
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

// StoreAllocation stores allocation information
func (kc *PodCacheManager) StoreAllocation(podUID string, allocation *api.WorkerAllocation) error {
	kc.mu.Lock()
	defer kc.mu.Unlock()
	kc.allocations[podUID] = allocation
	return nil
}

// GetWorkerChangedChan returns the channel for worker change notifications
func (kc *PodCacheManager) GetWorkerChangedChan() <-chan struct{} {
	return kc.workerChangedCh
}

// GetAllPods returns all pods currently in the cache
func (kc *PodCacheManager) GetAllPods() map[string]*corev1.Pod {
	kc.mu.RLock()
	defer kc.mu.RUnlock()

	result := make(map[string]*corev1.Pod, len(kc.podCache))
	for k, v := range kc.podCache {
		result[k] = v
	}
	return result
}

// podAnnotationsEqual checks if two annotation maps are equal (for relevant keys)
func podAnnotationsEqual(old, new map[string]string) bool {
	if old == nil && new == nil {
		return true
	}
	if old == nil || new == nil {
		return false
	}

	// Check relevant annotation keys
	relevantKeys := []string{
		constants.GPUDeviceIDsAnnotation,
		constants.IsolationModeAnnotation,
		constants.VRAMLimitAnnotation,
		constants.ComputeLimitAnnotation,
		constants.WorkloadProfileAnnotation,
	}

	for _, key := range relevantKeys {
		if old[key] != new[key] {
			return false
		}
	}

	return true
}
