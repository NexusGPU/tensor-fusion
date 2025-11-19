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
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
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

// KubeletClient manages pod watching and worker information extraction
type KubeletClient struct {
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

// NewKubeletClient creates a new kubelet client
func NewKubeletClient(ctx context.Context, restConfig *rest.Config, nodeName string) (*KubeletClient, error) {
	clientset, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes clientset: %w", err)
	}

	return &KubeletClient{
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
func (kc *KubeletClient) Start() error {
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
	_, controller := cache.NewInformer(
		lw,
		&corev1.Pod{},
		0, // resync period
		cache.ResourceEventHandlerFuncs{
			AddFunc:    kc.onPodAdd,
			UpdateFunc: kc.onPodUpdate,
			DeleteFunc: kc.onPodDelete,
		},
	)

	// Start the informer
	go controller.Run(kc.stopCh)

	klog.Infof("Started watching pods on node %s with label %s=%s", kc.nodeName, constants.TensorFusionEnabledLabelKey, constants.TrueStringValue)
	return nil
}

// Stop stops the kubelet client
func (kc *KubeletClient) Stop() {
	close(kc.stopCh)
}

// onPodAdd handles pod addition events
func (kc *KubeletClient) onPodAdd(obj interface{}) {
	pod := obj.(*corev1.Pod)
	kc.mu.Lock()
	kc.podCache[string(pod.UID)] = pod
	kc.mu.Unlock()

	klog.V(4).Infof("Pod added: %s/%s (UID: %s)", pod.Namespace, pod.Name, pod.UID)
	kc.notifyWorkerChanged()
}

// onPodUpdate handles pod update events
func (kc *KubeletClient) onPodUpdate(oldObj, newObj interface{}) {
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
func (kc *KubeletClient) onPodDelete(obj interface{}) {
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
func (kc *KubeletClient) notifyWorkerChanged() {
	select {
	case kc.workerChangedCh <- struct{}{}:
	default:
	}
}

// GetWorkerInfoForAllocation extracts worker info from pod annotations for allocation
func (kc *KubeletClient) GetWorkerInfoForAllocation(ctx context.Context, containerReq *pluginapi.ContainerAllocateRequest) (*WorkerInfo, error) {
	// Extract pod UID from environment variables or device IDs
	// In practice, kubelet may pass pod info differently
	// For now, we'll search through our pod cache

	kc.mu.RLock()
	defer kc.mu.RUnlock()

	// If not found by device IDs, try to find by pod index annotation
	// The device plugin may use pod index to identify pods
	for _, pod := range kc.podCache {
		if pod.Annotations == nil {
			continue
		}

		// Check if pod has index annotation and matches resource request
		if podIndex, exists := pod.Annotations[constants.PodIndexAnnotation]; exists {
			// Try to match based on resource name and index
			// This is a fallback mechanism

			return kc.extractWorkerInfo(pod, podIndex), nil
		}
	}

	return nil, fmt.Errorf("worker info not found for allocation request")
}

// extractWorkerInfo extracts worker information from pod annotations
func (kc *KubeletClient) extractWorkerInfo(pod *corev1.Pod, podIndex string) *WorkerInfo {
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
	if templateID, exists := pod.Annotations[constants.WorkloadProfileAnnotation]; exists {
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
func (kc *KubeletClient) StoreAllocation(podUID string, allocation *api.DeviceAllocation) error {
	kc.mu.Lock()
	defer kc.mu.Unlock()
	kc.allocations[podUID] = allocation
	return nil
}

// GetAllocation retrieves allocation information
func (kc *KubeletClient) GetAllocation(podUID string) (*api.DeviceAllocation, bool) {
	kc.mu.RLock()
	defer kc.mu.RUnlock()
	allocation, exists := kc.allocations[podUID]
	return allocation, exists
}

// GetWorkerChangedChan returns the channel for worker change notifications
func (kc *KubeletClient) GetWorkerChangedChan() <-chan struct{} {
	return kc.workerChangedCh
}

// GetAllPods returns all pods currently in the cache
func (kc *KubeletClient) GetAllPods() map[string]*corev1.Pod {
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
