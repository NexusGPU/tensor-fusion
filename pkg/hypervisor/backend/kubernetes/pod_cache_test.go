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
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/NexusGPU/tensor-fusion/pkg/hypervisor/api"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// TestGetWorkerInfoForAllocationByIndex_NoDeadlock verifies that the pub/sub mechanism
// does not cause a deadlock when a subscriber is waiting and a new pod arrives.
func TestGetWorkerInfoForAllocationByIndex_NoDeadlock(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Create a minimal PodCacheManager for testing (without kubernetes client)
	kc := &PodCacheManager{
		ctx:               ctx,
		nodeName:          "test-node",
		cachedPod:         make(map[string]*corev1.Pod, 32),
		indexToWorkerInfo: make(map[int]*api.WorkerInfo, 32),
		stopCh:            make(chan struct{}),
		workerChangedCh:   make(chan struct{}, 1),
		indexSubscribers:  make(map[int]map[*workerInfoSubscriber]struct{}),
		podSubscribers:    make(map[string]chan<- *api.WorkerInfo),
	}

	// Start the event bus
	go kc.runWorkerChangeEventBus()
	defer close(kc.stopCh)

	testPodIndex := 42
	var wg sync.WaitGroup
	resultCh := make(chan *api.WorkerInfo, 1)
	errCh := make(chan error, 1)

	// Start subscriber in a goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		workerInfo, err := kc.GetWorkerInfoForAllocationByIndex(testPodIndex)
		if err != nil {
			errCh <- err
			return
		}
		resultCh <- workerInfo
	}()

	// Give the subscriber time to register
	time.Sleep(100 * time.Millisecond)

	// Simulate pod add event with the matching index
	pod := createTestPodWithIndex(testPodIndex)
	kc.onPodAdd(pod)

	// Wait for result with timeout
	select {
	case workerInfo := <-resultCh:
		if workerInfo == nil {
			t.Fatal("Expected non-nil worker info")
		}
		if workerInfo.WorkerUID != string(pod.UID) {
			t.Errorf("Expected WorkerUID %s, got %s", pod.UID, workerInfo.WorkerUID)
		}
		t.Logf("Successfully received worker info: %+v", workerInfo)
	case err := <-errCh:
		t.Fatalf("Unexpected error: %v", err)
	case <-time.After(5 * time.Second):
		t.Fatal("Timeout waiting for worker info - possible deadlock!")
	}

	wg.Wait()
}

// TestGetWorkerInfoForAllocationByIndex_FastPath verifies that if worker info is already
// available, GetWorkerInfoForAllocationByIndex returns immediately.
func TestGetWorkerInfoForAllocationByIndex_FastPath(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	kc := &PodCacheManager{
		ctx:               ctx,
		nodeName:          "test-node",
		cachedPod:         make(map[string]*corev1.Pod, 32),
		indexToWorkerInfo: make(map[int]*api.WorkerInfo, 32),
		stopCh:            make(chan struct{}),
		workerChangedCh:   make(chan struct{}, 1),
		indexSubscribers:  make(map[int]map[*workerInfoSubscriber]struct{}),
		podSubscribers:    make(map[string]chan<- *api.WorkerInfo),
	}
	defer close(kc.stopCh)

	testPodIndex := 123

	// Pre-populate worker info
	expectedWorkerInfo := &api.WorkerInfo{
		WorkerUID:  "test-uid-123",
		WorkerName: "test-pod",
		Status:     api.WorkerStatusDeviceAllocating,
	}
	kc.indexToWorkerInfo[testPodIndex] = expectedWorkerInfo

	// Should return immediately
	start := time.Now()
	workerInfo, err := kc.GetWorkerInfoForAllocationByIndex(testPodIndex)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if workerInfo != expectedWorkerInfo {
		t.Errorf("Expected worker info %+v, got %+v", expectedWorkerInfo, workerInfo)
	}
	if elapsed > 100*time.Millisecond {
		t.Errorf("Fast path took too long: %v", elapsed)
	}
}

// TestGetWorkerInfoForAllocationByIndex_MultipleSubscribers tests that multiple
// subscribers for the same index all receive the worker info.
func TestGetWorkerInfoForAllocationByIndex_MultipleSubscribers(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	kc := &PodCacheManager{
		ctx:               ctx,
		nodeName:          "test-node",
		cachedPod:         make(map[string]*corev1.Pod, 32),
		indexToWorkerInfo: make(map[int]*api.WorkerInfo, 32),
		stopCh:            make(chan struct{}),
		workerChangedCh:   make(chan struct{}, 1),
		indexSubscribers:  make(map[int]map[*workerInfoSubscriber]struct{}),
		podSubscribers:    make(map[string]chan<- *api.WorkerInfo),
	}

	go kc.runWorkerChangeEventBus()
	defer close(kc.stopCh)

	testPodIndex := 99
	numSubscribers := 5
	var wg sync.WaitGroup
	results := make(chan *api.WorkerInfo, numSubscribers)
	errors := make(chan error, numSubscribers)

	// Start multiple subscribers
	for i := 0; i < numSubscribers; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			workerInfo, err := kc.GetWorkerInfoForAllocationByIndex(testPodIndex)
			if err != nil {
				errors <- fmt.Errorf("subscriber %d: %w", id, err)
				return
			}
			results <- workerInfo
		}(i)
	}

	// Give subscribers time to register
	time.Sleep(200 * time.Millisecond)

	// Simulate pod add
	pod := createTestPodWithIndex(testPodIndex)
	kc.onPodAdd(pod)

	// Collect results
	successCount := 0
	timeout := time.After(5 * time.Second)
	for i := 0; i < numSubscribers; i++ {
		select {
		case <-results:
			successCount++
		case err := <-errors:
			t.Errorf("Error: %v", err)
		case <-timeout:
			t.Fatal("Timeout waiting for all subscribers")
		}
	}

	if successCount != numSubscribers {
		t.Errorf("Expected %d successful results, got %d", numSubscribers, successCount)
	}

	wg.Wait()
}

// TestGetWorkerInfoForAllocationByIndex_PodArrivesFirst tests the scenario where
// the pod arrives BEFORE the device plugin allocate request.
// This verifies the fast path works correctly.
func TestGetWorkerInfoForAllocationByIndex_PodArrivesFirst(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	kc := &PodCacheManager{
		ctx:               ctx,
		nodeName:          "test-node",
		cachedPod:         make(map[string]*corev1.Pod, 32),
		indexToWorkerInfo: make(map[int]*api.WorkerInfo, 32),
		stopCh:            make(chan struct{}),
		workerChangedCh:   make(chan struct{}, 1),
		indexSubscribers:  make(map[int]map[*workerInfoSubscriber]struct{}),
		podSubscribers:    make(map[string]chan<- *api.WorkerInfo),
	}

	go kc.runWorkerChangeEventBus()
	defer close(kc.stopCh)

	testPodIndex := 77

	// Step 1: Pod arrives FIRST (before any subscriber)
	pod := createTestPodWithIndex(testPodIndex)
	kc.onPodAdd(pod)

	// Verify workerInfo is stored in indexToWorkerInfo
	kc.mu.RLock()
	storedInfo, exists := kc.indexToWorkerInfo[testPodIndex]
	kc.mu.RUnlock()
	if !exists {
		t.Fatal("Expected workerInfo to be stored in indexToWorkerInfo after onPodAdd")
	}
	t.Logf("Pod arrived first, workerInfo stored: %s", storedInfo.WorkerUID)

	// Step 2: Device plugin allocate request comes LATER
	// This should hit the fast path and return immediately
	start := time.Now()
	workerInfo, err := kc.GetWorkerInfoForAllocationByIndex(testPodIndex)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if workerInfo == nil {
		t.Fatal("Expected non-nil worker info")
	}
	if workerInfo.WorkerUID != string(pod.UID) {
		t.Errorf("Expected WorkerUID %s, got %s", pod.UID, workerInfo.WorkerUID)
	}
	// Fast path should be very quick (no waiting)
	if elapsed > 50*time.Millisecond {
		t.Errorf("Fast path took too long: %v (expected < 50ms)", elapsed)
	}
	t.Logf("Device plugin request succeeded via fast path in %v", elapsed)
}

// createTestPodWithIndex creates a test pod with the specified index annotation
// that will trigger WorkerStatusDeviceAllocating status
func createTestPodWithIndex(index int) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("test-pod-%d", index),
			Namespace: "default",
			UID:       types.UID(fmt.Sprintf("test-uid-%d", index)),
			Labels: map[string]string{
				constants.TensorFusionEnabledLabelKey: constants.TrueStringValue,
				constants.WorkloadKey:                 "test-workload",
			},
			Annotations: map[string]string{
				constants.PodIndexAnnotation:      strconv.Itoa(index),
				constants.TFLOPSRequestAnnotation: "10",
				constants.VRAMRequestAnnotation:   "1Gi",
				constants.TFLOPSLimitAnnotation:   "20",
				constants.VRAMLimitAnnotation:     "2Gi",
				constants.IsolationModeAnnotation: "soft",
				constants.QoSLevelAnnotation:      "low",
				constants.GpuPoolKey:              "default",
				constants.GPUDeviceIDsAnnotation:  "gpu-0",
			},
		},
		Spec: corev1.PodSpec{
			NodeName: "test-node",
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodPending,
		},
	}
}
