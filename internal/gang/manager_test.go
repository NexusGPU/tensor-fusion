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

package gang

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

func createTestPod(name, workloadName string, gangMinMembers int32, gangTimeout string) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
			UID:       types.UID(name + "-uid"),
			Labels: map[string]string{
				constants.WorkloadKey: workloadName,
			},
			Annotations: map[string]string{},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "main",
					Image: "test-image",
				},
			},
		},
	}

	if gangMinMembers > 0 {
		pod.Annotations[constants.GangMinMembersAnnotation] = fmt.Sprintf("%d", gangMinMembers)
	}

	if gangTimeout != "" {
		pod.Annotations[constants.GangTimeoutAnnotation] = gangTimeout
	}

	return pod
}

func createTestAllocReq(pod *corev1.Pod, gpuNames []string) *tfv1.AllocRequest {
	return &tfv1.AllocRequest{
		PoolName: "test-pool",
		WorkloadNameNamespace: tfv1.NameNamespace{
			Name:      pod.Labels[constants.WorkloadKey],
			Namespace: pod.Namespace,
		},
		Request: tfv1.Resource{
			Tflops: resource.MustParse("100"),
			Vram:   resource.MustParse("40Gi"),
		},
		Count:    1,
		GPUNames: gpuNames,
		PodMeta:  pod.ObjectMeta,
	}
}

func TestParseGangConfig(t *testing.T) {
	manager := NewManager(nil, nil, "TestPlugin")

	t.Run("non-gang pod returns disabled config", func(t *testing.T) {
		pod := createTestPod("test-pod", "test-workload", 0, "")
		config := manager.ParseGangConfig(pod)
		assert.False(t, config.Enabled)
	})

	t.Run("gang pod returns enabled config", func(t *testing.T) {
		pod := createTestPod("test-pod", "test-workload", 3, "")
		config := manager.ParseGangConfig(pod)
		assert.True(t, config.Enabled)
		assert.Equal(t, int32(3), config.MinMembers)
		assert.Equal(t, PodGroupKey("default/test-workload"), config.GroupKey)
		assert.Equal(t, time.Duration(0), config.Timeout) // No timeout means wait indefinitely
	})

	t.Run("gang pod with timeout", func(t *testing.T) {
		pod := createTestPod("test-pod", "test-workload", 3, "5m")
		config := manager.ParseGangConfig(pod)
		assert.True(t, config.Enabled)
		assert.Equal(t, int32(3), config.MinMembers)
		assert.Equal(t, 5*time.Minute, config.Timeout)
	})

	t.Run("gang pod with zero timeout means indefinite wait", func(t *testing.T) {
		pod := createTestPod("test-pod", "test-workload", 3, "0")
		config := manager.ParseGangConfig(pod)
		assert.True(t, config.Enabled)
		assert.Equal(t, time.Duration(0), config.Timeout)
	})

	t.Run("pod without workload label is not gang", func(t *testing.T) {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:        "test-pod",
				Namespace:   "default",
				Labels:      map[string]string{},
				Annotations: map[string]string{constants.GangMinMembersAnnotation: "3"},
			},
		}
		config := manager.ParseGangConfig(pod)
		assert.False(t, config.Enabled)
	})
}

func TestPreFilter(t *testing.T) {
	ctx := context.Background()

	t.Run("non-gang pod passes prefilter", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		pod := createTestPod("test-pod", "test-workload", 0, "")
		err := manager.PreFilter(ctx, pod)
		assert.NoError(t, err)
	})

	t.Run("backed off group fails prefilter", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		groupKey := NewPodGroupKey("default", "test-workload")
		manager.SetBackoff(groupKey, time.Minute)

		pod := createTestPod("test-pod", "test-workload", 3, "")
		err := manager.PreFilter(ctx, pod)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "backed off")
	})
}

func TestPermitSinglePod(t *testing.T) {
	ctx := context.Background()
	manager := NewManager(nil, nil, "TestPlugin")

	t.Run("non-gang pod is allowed immediately", func(t *testing.T) {
		pod := createTestPod("test-pod", "test-workload", 0, "")
		allocReq := createTestAllocReq(pod, []string{"gpu-1"})

		status, waitTime, waitingInfo := manager.Permit(ctx, pod, "node-1", allocReq)
		assert.Equal(t, PermitAllow, status)
		assert.Equal(t, time.Duration(0), waitTime)
		assert.Nil(t, waitingInfo)
	})

	t.Run("gang pod waits when not enough members", func(t *testing.T) {
		pod := createTestPod("test-pod", "gang-workload", 3, "")
		allocReq := createTestAllocReq(pod, []string{"gpu-1"})

		status, waitTime, waitingInfo := manager.Permit(ctx, pod, "node-1", allocReq)
		assert.Equal(t, PermitWait, status)
		assert.True(t, waitTime > 0)
		assert.NotNil(t, waitingInfo)
	})
}

func TestPermitGangScheduling(t *testing.T) {
	ctx := context.Background()

	t.Run("gang is allowed when all members ready", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")

		// Create 3 pods for a gang of 3
		pods := make([]*corev1.Pod, 3)
		for i := 0; i < 3; i++ {
			pods[i] = createTestPod(
				"worker-"+string(rune('0'+i)),
				"complete-gang",
				3,
				"",
			)
			pods[i].UID = types.UID("uid-" + string(rune('0'+i)))
		}

		var waitingInfos []*WaitingPodInfo

		// First two pods should wait
		for i := 0; i < 2; i++ {
			allocReq := createTestAllocReq(pods[i], []string{"gpu-" + string(rune('0'+i))})
			status, waitTime, waitingInfo := manager.Permit(ctx, pods[i], "node-1", allocReq)
			waitingInfos = append(waitingInfos, waitingInfo)

			assert.Equal(t, PermitWait, status)
			assert.True(t, waitTime > 0)
			assert.NotNil(t, waitingInfo)
		}

		// Third pod should trigger allow for all
		allocReq := createTestAllocReq(pods[2], []string{"gpu-2"})
		status, waitTime, waitingInfo := manager.Permit(ctx, pods[2], "node-1", allocReq)
		assert.Equal(t, PermitAllow, status)
		assert.Equal(t, time.Duration(0), waitTime)
		assert.NotNil(t, waitingInfo)

		// Verify that waiting pods received allow signal
		for i := 0; i < 2; i++ {
			select {
			case <-waitingInfos[i].AllowCh:
				// Expected - pod was allowed
			case <-time.After(100 * time.Millisecond):
				t.Errorf("Pod %d did not receive allow signal", i)
			}
		}
	})
}

func TestPermitTimeout(t *testing.T) {
	ctx := context.Background()

	t.Run("gang times out when not enough members", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")

		// Create pod with short timeout
		pod := createTestPod("worker-0", "timeout-gang", 3, "100ms")
		allocReq := createTestAllocReq(pod, []string{"gpu-0"})

		status, waitTime, waitingInfo := manager.Permit(ctx, pod, "node-1", allocReq)
		assert.Equal(t, PermitWait, status)
		assert.True(t, waitTime > 0)
		assert.NotNil(t, waitingInfo)

		// Wait for timeout to trigger
		time.Sleep(200 * time.Millisecond)

		// Try to permit another pod - should be rejected due to timeout
		pod2 := createTestPod("worker-1", "timeout-gang", 3, "100ms")
		pod2.UID = types.UID("uid-1")
		allocReq2 := createTestAllocReq(pod2, []string{"gpu-1"})

		status2, _, _ := manager.Permit(ctx, pod2, "node-1", allocReq2)
		assert.Equal(t, PermitReject, status2)

		// Verify backoff is set
		groupKey := NewPodGroupKey("default", "timeout-gang")
		assert.True(t, manager.IsBackedOff(groupKey))
	})
}

func TestWaitForGang(t *testing.T) {
	ctx := context.Background()

	t.Run("WaitForGang returns true when allowed", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		waitingInfo := NewWaitingPodInfo(
			types.UID("test-uid"),
			"test-pod",
			"default",
			"node-1",
			[]string{"gpu-1"},
			nil,
		)

		// Send allow signal in goroutine
		go func() {
			time.Sleep(10 * time.Millisecond)
			waitingInfo.AllowCh <- struct{}{}
		}()

		allowed, reason := manager.WaitForGang(ctx, waitingInfo, time.Second)
		assert.True(t, allowed)
		assert.Empty(t, reason)
	})

	t.Run("WaitForGang returns false when rejected", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		waitingInfo := NewWaitingPodInfo(
			types.UID("test-uid"),
			"test-pod",
			"default",
			"node-1",
			[]string{"gpu-1"},
			nil,
		)

		// Send reject signal in goroutine
		go func() {
			time.Sleep(10 * time.Millisecond)
			waitingInfo.RejectCh <- "test rejection"
		}()

		allowed, reason := manager.WaitForGang(ctx, waitingInfo, time.Second)
		assert.False(t, allowed)
		assert.Equal(t, "test rejection", reason)
	})

	t.Run("WaitForGang returns false on timeout", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		waitingInfo := NewWaitingPodInfo(
			types.UID("test-uid"),
			"test-pod",
			"default",
			"node-1",
			[]string{"gpu-1"},
			nil,
		)

		allowed, reason := manager.WaitForGang(ctx, waitingInfo, 50*time.Millisecond)
		assert.False(t, allowed)
		assert.Equal(t, "gang scheduling timeout", reason)
	})

	t.Run("WaitForGang returns true for nil waitingInfo", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")
		allowed, reason := manager.WaitForGang(ctx, nil, time.Second)
		assert.True(t, allowed)
		assert.Empty(t, reason)
	})
}

func TestUnreserve(t *testing.T) {
	ctx := context.Background()

	t.Run("Unreserve removes pod from waiting", func(t *testing.T) {
		manager := NewManager(nil, nil, "TestPlugin")

		// Add a pod to waiting
		pod := createTestPod("worker-0", "unreserve-gang", 3, "")
		allocReq := createTestAllocReq(pod, []string{"gpu-0"})
		_, _, waitingInfo := manager.Permit(ctx, pod, "node-1", allocReq)
		require.NotNil(t, waitingInfo)

		// Verify pod is in waiting
		groupKey := NewPodGroupKey("default", "unreserve-gang")
		pgInfo := manager.GetPodGroupInfo(groupKey)
		require.NotNil(t, pgInfo)
		assert.Equal(t, 1, pgInfo.GetWaitingCount())

		// Unreserve the pod
		manager.Unreserve(ctx, pod)

		// Verify pod is removed
		assert.Equal(t, 0, pgInfo.GetWaitingCount())
	})
}

func TestConcurrentPermit(t *testing.T) {
	ctx := context.Background()
	manager := NewManager(nil, nil, "TestPlugin")

	numPods := 5
	gangSize := int32(5)

	var wg sync.WaitGroup
	wg.Add(numPods)

	results := make(chan PermitStatus, numPods)

	for i := 0; i < numPods; i++ {
		go func(idx int) {
			defer wg.Done()

			pod := createTestPod(
				"worker-"+string(rune('0'+idx)),
				"concurrent-gang",
				gangSize,
				"",
			)
			pod.UID = types.UID("uid-" + string(rune('0'+idx)))

			allocReq := createTestAllocReq(pod, []string{"gpu-" + string(rune('0'+idx))})
			status, _, waitingInfo := manager.Permit(ctx, pod, "node-1", allocReq)

			if status == PermitWait && waitingInfo != nil {
				// Wait for the gang to be ready
				allowed, _ := manager.WaitForGang(ctx, waitingInfo, 5*time.Second)
				if allowed {
					results <- PermitAllow
				} else {
					results <- PermitReject
				}
			} else {
				results <- status
			}
		}(i)
	}

	wg.Wait()
	close(results)

	// All pods should be allowed
	allowCount := 0
	for status := range results {
		if status == PermitAllow {
			allowCount++
		}
	}
	assert.Equal(t, numPods, allowCount, "All pods should be allowed")
}

func TestPodGroupInfoMethods(t *testing.T) {
	t.Run("IsReady returns correct status", func(t *testing.T) {
		pgInfo := NewPodGroupInfo("test/workload", 3, 0)
		assert.False(t, pgInfo.IsReady())

		// Add 2 waiting pods
		pgInfo.WaitingPods[types.UID("uid-1")] = &WaitingPodInfo{}
		pgInfo.WaitingPods[types.UID("uid-2")] = &WaitingPodInfo{}
		assert.False(t, pgInfo.IsReady())

		// Add 3rd waiting pod
		pgInfo.WaitingPods[types.UID("uid-3")] = &WaitingPodInfo{}
		assert.True(t, pgInfo.IsReady())
	})

	t.Run("IsTimedOut returns correct status", func(t *testing.T) {
		// No timeout set
		pgInfo := NewPodGroupInfo("test/workload", 3, 0)
		assert.False(t, pgInfo.IsTimedOut())

		// With timeout
		pgInfo2 := NewPodGroupInfo("test/workload2", 3, 50*time.Millisecond)
		assert.False(t, pgInfo2.IsTimedOut())

		time.Sleep(100 * time.Millisecond)
		assert.True(t, pgInfo2.IsTimedOut())
	})

	t.Run("RemainingTimeout returns correct duration", func(t *testing.T) {
		// No timeout
		pgInfo := NewPodGroupInfo("test/workload", 3, 0)
		remaining := pgInfo.RemainingTimeout()
		assert.Equal(t, time.Hour, remaining) // Default for no timeout

		// With timeout
		pgInfo2 := NewPodGroupInfo("test/workload2", 3, time.Second)
		remaining2 := pgInfo2.RemainingTimeout()
		assert.True(t, remaining2 > 0 && remaining2 <= time.Second)
	})
}

func TestCleanupPodGroup(t *testing.T) {
	ctx := context.Background()
	manager := NewManager(nil, nil, "TestPlugin")

	// Add some pods
	pod := createTestPod("worker-0", "cleanup-gang", 3, "")
	allocReq := createTestAllocReq(pod, []string{"gpu-0"})
	_, _, waitingInfo := manager.Permit(ctx, pod, "node-1", allocReq)
	require.NotNil(t, waitingInfo)

	groupKey := NewPodGroupKey("default", "cleanup-gang")
	assert.NotNil(t, manager.GetPodGroupInfo(groupKey))

	// Cleanup
	manager.CleanupPodGroup(groupKey)

	// Verify cleanup
	assert.Nil(t, manager.GetPodGroupInfo(groupKey))

	// Verify waiting pod received reject
	select {
	case <-waitingInfo.RejectCh:
		// Expected
	case <-time.After(100 * time.Millisecond):
		t.Error("Waiting pod did not receive reject signal after cleanup")
	}
}

func TestMultipleGangGroups(t *testing.T) {
	ctx := context.Background()
	manager := NewManager(nil, nil, "TestPlugin")

	// Create two different gang groups
	gang1Pods := make([]*corev1.Pod, 2)
	gang2Pods := make([]*corev1.Pod, 3)

	// Setup gang1 (size 2)
	for i := 0; i < 2; i++ {
		gang1Pods[i] = createTestPod(
			"gang1-worker-"+string(rune('0'+i)),
			"gang1-workload",
			2,
			"",
		)
		gang1Pods[i].UID = types.UID("gang1-uid-" + string(rune('0'+i)))
	}

	// Setup gang2 (size 3)
	for i := 0; i < 3; i++ {
		gang2Pods[i] = createTestPod(
			"gang2-worker-"+string(rune('0'+i)),
			"gang2-workload",
			3,
			"",
		)
		gang2Pods[i].UID = types.UID("gang2-uid-" + string(rune('0'+i)))
	}

	// Submit first pod from each gang
	allocReq1 := createTestAllocReq(gang1Pods[0], []string{"gpu-0"})
	status1, _, _ := manager.Permit(ctx, gang1Pods[0], "node-1", allocReq1)
	assert.Equal(t, PermitWait, status1)

	allocReq2 := createTestAllocReq(gang2Pods[0], []string{"gpu-1"})
	status2, _, _ := manager.Permit(ctx, gang2Pods[0], "node-1", allocReq2)
	assert.Equal(t, PermitWait, status2)

	// Complete gang1 (only needs 2)
	allocReq1b := createTestAllocReq(gang1Pods[1], []string{"gpu-2"})
	status1b, _, _ := manager.Permit(ctx, gang1Pods[1], "node-1", allocReq1b)
	assert.Equal(t, PermitAllow, status1b)

	// Gang2 should still be waiting
	groupKey2 := NewPodGroupKey("default", "gang2-workload")
	pgInfo2 := manager.GetPodGroupInfo(groupKey2)
	assert.Equal(t, 1, pgInfo2.GetWaitingCount())
}
