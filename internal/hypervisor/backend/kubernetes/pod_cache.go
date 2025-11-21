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
	"strings"
	"sync"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/api"
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

// WorkerInfo contains information about a worker pod
type WorkerInfo struct {
	PodUID            string
	PodName           string
	Namespace         string
	DeviceUUIDs       []string
	IsolationMode     api.IsolationMode
	MemoryLimitBytes  uint64
	ComputeLimitUnits uint32
	TemplateID        string
	Annotations       map[string]string
	PodIndex          string
}

// PodCacheManager manages pod watching and worker information extraction
type PodCacheManager struct {
	ctx        context.Context
	clientset  *kubernetes.Clientset
	restConfig *rest.Config
	nodeName   string

	mu              sync.RWMutex
	podCache        map[string]*corev1.Pod           // key: pod UID
	allocations     map[string]*api.DeviceAllocation // key: pod UID
	stopCh          chan struct{}
	workerChangedCh chan struct{}
}

// NewPodCacheManager creates a new pod cache manager
func NewPodCacheManager(ctx context.Context, restConfig *rest.Config, nodeName string) (*PodCacheManager, error) {
	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes clientset: %w", err)
	}

	return &PodCacheManager{
		ctx:             ctx,
		clientset:       clientset,
		restConfig:      restConfig,
		nodeName:        nodeName,
		podCache:        make(map[string]*corev1.Pod),
		allocations:     make(map[string]*api.DeviceAllocation),
		stopCh:          make(chan struct{}),
		workerChangedCh: make(chan struct{}, 1),
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
	delete(kc.podCache, string(pod.UID))
	delete(kc.allocations, string(pod.UID))
	kc.mu.Unlock()

	klog.V(4).Infof("Pod deleted: %s/%s (UID: %s)", pod.Namespace, pod.Name, pod.UID)
	kc.notifyWorkerChanged()
}

// notifyWorkerChanged notifies that worker information has changed
func (kc *PodCacheManager) notifyWorkerChanged() {
	select {
	case kc.workerChangedCh <- struct{}{}:
	default:
	}
}

// GetWorkerInfoForAllocationByIndex finds a pod by its index annotation and extracts worker info
func (kc *PodCacheManager) GetWorkerInfoForAllocationByIndex(ctx context.Context, podIndex string) (*WorkerInfo, error) {
	kc.mu.RLock()
	defer kc.mu.RUnlock()

	// Find pod with matching index annotation
	for _, pod := range kc.podCache {
		if pod.Annotations == nil {
			continue
		}

		// Check if pod has matching index annotation
		if podIndexAnno, exists := pod.Annotations[constants.PodIndexAnnotation]; exists && podIndexAnno == podIndex {
			return kc.extractWorkerInfo(pod, podIndex), nil
		}
	}

	return nil, fmt.Errorf("worker info not found for pod index %s", podIndex)
}

// GetPodByUID retrieves a pod from the cache by its UID
func (kc *PodCacheManager) GetPodByUID(podUID string) *corev1.Pod {
	kc.mu.RLock()
	defer kc.mu.RUnlock()
	return kc.podCache[podUID]
}

// CheckDuplicateIndex checks if multiple pods have the same index annotation
// Returns error if duplicate found (excluding the specified podUID)
func (kc *PodCacheManager) CheckDuplicateIndex(ctx context.Context, podIndex string, excludePodUID string) error {
	kc.mu.RLock()
	defer kc.mu.RUnlock()

	var matchingPods []string
	for podUID, pod := range kc.podCache {
		if pod.Annotations == nil {
			continue
		}

		if podIndexAnno, exists := pod.Annotations[constants.PodIndexAnnotation]; exists && podIndexAnno == podIndex {
			if string(pod.UID) != excludePodUID {
				matchingPods = append(matchingPods, fmt.Sprintf("%s/%s (UID: %s)", pod.Namespace, pod.Name, podUID))
			}
		}
	}

	if len(matchingPods) > 0 {
		return fmt.Errorf("duplicate index %s found in pods: %v", podIndex, matchingPods)
	}

	return nil
}

// RemovePodIndexAnnotation removes the PodIndexAnnotation from a pod after successful allocation
func (kc *PodCacheManager) RemovePodIndexAnnotation(ctx context.Context, podUID string, namespace string, podName string) error {
	kc.mu.RLock()
	pod, exists := kc.podCache[podUID]
	kc.mu.RUnlock()

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

// extractWorkerInfo extracts worker information from pod annotations
func (kc *PodCacheManager) extractWorkerInfo(pod *corev1.Pod, podIndex string) *WorkerInfo {
	info := &WorkerInfo{
		PodUID:      string(pod.UID),
		PodName:     pod.Name,
		Namespace:   pod.Namespace,
		Annotations: make(map[string]string),
		PodIndex:    podIndex,
	}

	if pod.Annotations == nil {
		return info
	}

	// Copy annotations
	for k, v := range pod.Annotations {
		info.Annotations[k] = v
	}

	// Extract GPU device IDs
	if gpuIDsStr, exists := pod.Annotations[constants.GPUDeviceIDsAnnotation]; exists {
		info.DeviceUUIDs = parseGPUIDs(gpuIDsStr)
	}

	// Extract isolation mode
	if isolationMode, exists := pod.Annotations[constants.IsolationModeAnnotation]; exists {
		info.IsolationMode = api.IsolationMode(isolationMode)
	} else {
		info.IsolationMode = api.IsolationModeSoft // default
	}

	// Extract pod index
	info.PodIndex = podIndex

	// Extract memory limit
	if vramLimit, exists := pod.Annotations[constants.VRAMLimitAnnotation]; exists {
		if bytes, err := parseMemoryBytes(vramLimit); err == nil {
			info.MemoryLimitBytes = bytes
		}
	}

	// Extract compute limit (compute percent)
	if computeLimit, exists := pod.Annotations[constants.ComputeLimitAnnotation]; exists {
		if percent, err := strconv.ParseUint(strings.TrimSuffix(computeLimit, "%"), 10, 32); err == nil {
			info.ComputeLimitUnits = uint32(percent)
		}
	}

	// Extract template ID (for partitioned mode)
	// First check PartitionTemplateIDAnnotation (set by scheduler)
	if templateID, exists := pod.Annotations[constants.PartitionTemplateIDAnnotation]; exists {
		info.TemplateID = templateID
	} else if templateID, exists := pod.Annotations[constants.WorkloadProfileAnnotation]; exists {
		// Fallback to WorkloadProfileAnnotation
		info.TemplateID = templateID
	}

	return info
}

// parseGPUIDs parses GPU IDs from annotation string
func parseGPUIDs(gpuIDsStr string) []string {
	if gpuIDsStr == "" {
		return nil
	}

	ids := strings.Split(gpuIDsStr, ",")
	result := make([]string, 0, len(ids))
	for _, id := range ids {
		id = strings.TrimSpace(id)
		if id != "" {
			result = append(result, id)
		}
	}
	return result
}

// parseMemoryBytes parses memory bytes from quantity string (e.g., "1Gi", "1024Mi")
func parseMemoryBytes(quantityStr string) (uint64, error) {
	// Simple parsing - in production, use k8s.io/apimachinery/pkg/api/resource
	quantityStr = strings.TrimSpace(quantityStr)

	if strings.HasSuffix(quantityStr, "Gi") {
		val, err := strconv.ParseFloat(strings.TrimSuffix(quantityStr, "Gi"), 64)
		if err != nil {
			return 0, err
		}
		return uint64(val * 1024 * 1024 * 1024), nil
	}

	if strings.HasSuffix(quantityStr, "Mi") {
		val, err := strconv.ParseFloat(strings.TrimSuffix(quantityStr, "Mi"), 64)
		if err != nil {
			return 0, err
		}
		return uint64(val * 1024 * 1024), nil
	}

	if strings.HasSuffix(quantityStr, "Ki") {
		val, err := strconv.ParseFloat(strings.TrimSuffix(quantityStr, "Ki"), 64)
		if err != nil {
			return 0, err
		}
		return uint64(val * 1024), nil
	}

	// Assume bytes
	val, err := strconv.ParseUint(quantityStr, 10, 64)
	return val, err
}

// StoreAllocation stores allocation information
func (kc *PodCacheManager) StoreAllocation(podUID string, allocation *api.DeviceAllocation) error {
	kc.mu.Lock()
	defer kc.mu.Unlock()
	kc.allocations[podUID] = allocation
	return nil
}

// GetAllocation retrieves allocation information
func (kc *PodCacheManager) GetAllocation(podUID string) (*api.DeviceAllocation, bool) {
	kc.mu.RLock()
	defer kc.mu.RUnlock()
	allocation, exists := kc.allocations[podUID]
	return allocation, exists
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
