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
	"fmt"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"k8s.io/apimachinery/pkg/types"
)

// PodGroupKey uniquely identifies a gang/pod group
// Format: namespace/workload-name
type PodGroupKey string

// NewPodGroupKey creates a PodGroupKey from namespace and workload name
func NewPodGroupKey(namespace, workloadName string) PodGroupKey {
	return PodGroupKey(fmt.Sprintf("%s/%s", namespace, workloadName))
}

// PermitStatus represents the result of Permit phase
type PermitStatus int

const (
	// PermitAllow means the pod can proceed with binding
	PermitAllow PermitStatus = iota
	// PermitWait means the pod must wait for other gang members
	PermitWait
	// PermitReject means the pod should be rejected (e.g., timeout)
	PermitReject
)

// PodGroupInfo tracks the state of a gang scheduling group
type PodGroupInfo struct {
	Key        PodGroupKey
	MinMembers int32
	// Timeout duration, zero means wait indefinitely
	Timeout time.Duration

	// Creation time of the first pod in this group entering scheduling
	CreationTime time.Time

	// Pods that have reached Permit stage (waiting for gang)
	// Key: pod UID
	WaitingPods map[types.UID]*WaitingPodInfo

	// Pods that have been scheduled successfully
	ScheduledPods map[types.UID]struct{}

	mu sync.RWMutex
}

// WaitingPodInfo stores information about a pod waiting at Permit stage
type WaitingPodInfo struct {
	PodUID       types.UID
	PodName      string
	Namespace    string
	NodeName     string   // Tentatively assigned node
	GPUNames     []string // Tentatively assigned GPUs
	AllocReq     *tfv1.AllocRequest
	WaitingSince time.Time

	// Channel to signal when this pod should be allowed to proceed
	AllowCh chan struct{}
	// Channel to signal when this pod should be rejected
	RejectCh chan string
}

// GangSchedulingConfig parsed from workload/pod annotations
type GangSchedulingConfig struct {
	Enabled    bool
	GroupKey   PodGroupKey
	MinMembers int32
	Timeout    time.Duration // zero means wait indefinitely
}

// NewWaitingPodInfo creates a new WaitingPodInfo
func NewWaitingPodInfo(
	podUID types.UID,
	podName, namespace, nodeName string,
	gpuNames []string,
	allocReq *tfv1.AllocRequest,
) *WaitingPodInfo {
	return &WaitingPodInfo{
		PodUID:       podUID,
		PodName:      podName,
		Namespace:    namespace,
		NodeName:     nodeName,
		GPUNames:     gpuNames,
		AllocReq:     allocReq,
		WaitingSince: time.Now(),
		AllowCh:      make(chan struct{}, 1),
		RejectCh:     make(chan string, 1),
	}
}

// NewPodGroupInfo creates a new PodGroupInfo
func NewPodGroupInfo(key PodGroupKey, minMembers int32, timeout time.Duration) *PodGroupInfo {
	return &PodGroupInfo{
		Key:           key,
		MinMembers:    minMembers,
		Timeout:       timeout,
		CreationTime:  time.Now(),
		WaitingPods:   make(map[types.UID]*WaitingPodInfo),
		ScheduledPods: make(map[types.UID]struct{}),
	}
}

// GetWaitingCount returns the number of waiting pods
func (pg *PodGroupInfo) GetWaitingCount() int {
	pg.mu.RLock()
	defer pg.mu.RUnlock()
	return len(pg.WaitingPods)
}

// GetScheduledCount returns the number of scheduled pods
func (pg *PodGroupInfo) GetScheduledCount() int {
	pg.mu.RLock()
	defer pg.mu.RUnlock()
	return len(pg.ScheduledPods)
}

// IsReady returns true if the gang has enough members ready
func (pg *PodGroupInfo) IsReady() bool {
	pg.mu.RLock()
	defer pg.mu.RUnlock()
	return int32(len(pg.WaitingPods)) >= pg.MinMembers
}

// IsTimedOut returns true if the gang has exceeded its timeout
// Returns false if timeout is zero (wait indefinitely)
func (pg *PodGroupInfo) IsTimedOut() bool {
	if pg.Timeout == 0 {
		return false
	}
	return time.Since(pg.CreationTime) > pg.Timeout
}

// RemainingTimeout returns the remaining time before timeout
// Returns 0 if already timed out, returns a large duration if no timeout set
func (pg *PodGroupInfo) RemainingTimeout() time.Duration {
	if pg.Timeout == 0 {
		// No timeout, return a large duration (1 hour as default wait)
		return time.Hour
	}
	remaining := pg.Timeout - time.Since(pg.CreationTime)
	if remaining < 0 {
		return 0
	}
	return remaining
}
