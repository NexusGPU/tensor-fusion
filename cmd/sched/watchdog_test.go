/*
Copyright 2026.

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

package sched

import (
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
)

func TestNominatedPodRequeueWatchdogDefaultInterval(t *testing.T) {
	t.Setenv(nominatedPodRequeueWatchdogIntervalEnv, "")

	if got := nominatedPodRequeueWatchdogInterval(klog.Background()); got != 3*time.Minute {
		t.Fatalf("default watchdog interval = %s, want 3m", got)
	}
}

func TestNominatedPodsToActivate(t *testing.T) {
	now := time.Date(2026, time.August, 19, 12, 0, 0, 0, time.UTC)
	pod := newNominatedTensorFusionPod("pod-1")
	firstObserved := map[types.UID]time.Time{}
	const threshold = 6 * time.Minute

	activated := nominatedPodsToActivate(now, []*corev1.Pod{pod}, firstObserved, threshold)
	if len(activated) != 0 {
		t.Fatalf("first observation activated %d pods, want 0", len(activated))
	}
	if got := firstObserved[pod.UID]; !got.Equal(now) {
		t.Fatalf("first observation time = %s, want %s", got, now)
	}

	activated = nominatedPodsToActivate(now.Add(3*time.Minute), []*corev1.Pod{pod}, firstObserved, threshold)
	if len(activated) != 0 {
		t.Fatalf("pod activated before threshold: %v", activated)
	}

	firstActivationTime := now.Add(threshold)
	activated = nominatedPodsToActivate(firstActivationTime, []*corev1.Pod{pod}, firstObserved, threshold)
	if activated[string(pod.UID)] != pod {
		t.Fatalf("pod was not activated at threshold: %v", activated)
	}
	if got := firstObserved[pod.UID]; !got.Equal(firstActivationTime) {
		t.Fatalf("observation time after activation = %s, want %s", got, firstActivationTime)
	}

	activated = nominatedPodsToActivate(now.Add(9*time.Minute), []*corev1.Pod{pod}, firstObserved, threshold)
	if len(activated) != 0 {
		t.Fatalf("pod activated again before the next threshold: %v", activated)
	}

	activated = nominatedPodsToActivate(now.Add(12*time.Minute), []*corev1.Pod{pod}, firstObserved, threshold)
	if activated[string(pod.UID)] != pod {
		t.Fatalf("pod was not activated again after another threshold: %v", activated)
	}
}

func TestNominatedPodsToActivateCleansUpPodsThatLeaveTheQueue(t *testing.T) {
	now := time.Date(2026, time.August, 19, 12, 0, 0, 0, time.UTC)
	pod := newNominatedTensorFusionPod("pod-1")
	firstObserved := map[types.UID]time.Time{}

	nominatedPodsToActivate(now, []*corev1.Pod{pod}, firstObserved, 6*time.Minute)
	if len(firstObserved) != 1 {
		t.Fatalf("tracked pods = %d, want 1", len(firstObserved))
	}

	nominatedPodsToActivate(now.Add(3*time.Minute), nil, firstObserved, 6*time.Minute)
	if len(firstObserved) != 0 {
		t.Fatalf("tracked pods after leaving queue = %d, want 0", len(firstObserved))
	}
}

func TestNominatedPodsToActivateStopsTrackingIneligiblePods(t *testing.T) {
	now := time.Date(2026, time.August, 19, 12, 0, 0, 0, time.UTC)
	deletionTime := metav1.NewTime(now)

	tests := []struct {
		name   string
		mutate func(*corev1.Pod)
	}{
		{
			name: "nomination cleared",
			mutate: func(pod *corev1.Pod) {
				pod.Status.NominatedNodeName = ""
			},
		},
		{
			name: "pod bound",
			mutate: func(pod *corev1.Pod) {
				pod.Spec.NodeName = "node-a"
			},
		},
		{
			name: "pod deleting",
			mutate: func(pod *corev1.Pod) {
				pod.DeletionTimestamp = &deletionTime
			},
		},
		{
			name: "not TensorFusion worker",
			mutate: func(pod *corev1.Pod) {
				delete(pod.Labels, constants.LabelComponent)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod := newNominatedTensorFusionPod(types.UID(tt.name))
			firstObserved := map[types.UID]time.Time{pod.UID: now.Add(-6 * time.Minute)}
			tt.mutate(pod)

			activated := nominatedPodsToActivate(now, []*corev1.Pod{pod}, firstObserved, 6*time.Minute)
			if len(activated) != 0 {
				t.Fatalf("ineligible pod activated: %v", activated)
			}
			if _, ok := firstObserved[pod.UID]; ok {
				t.Fatal("ineligible pod remains tracked")
			}
		})
	}
}

func newNominatedTensorFusionPod(uid types.UID) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "nominated-pod",
			UID:  uid,
			Labels: map[string]string{
				constants.LabelComponent: constants.ComponentWorker,
			},
		},
		Status: corev1.PodStatus{
			NominatedNodeName: "node-a",
		},
	}
}
