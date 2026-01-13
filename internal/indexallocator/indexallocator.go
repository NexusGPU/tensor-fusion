package indexallocator

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/util/retry"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

// IndexAllocator manages allocation of 1-128 temporary indices for Pod-to-DevicePlugin communication
// Uses a simple atomic counter that increments from 1 to 128, then wraps around to 1
// No bitmap tracking needed - index reuse is acceptable after 128 cycles
// The availability check will be at PostBind stage, detected by pod index annotation on Node level
type IndexAllocator struct {
	IsLeader bool
	Client   client.Client

	// Atomic counter for index allocation (1-512, wraps around)
	currentIndex  int64
	ctx           context.Context
	storeMutex    sync.RWMutex
	initializedCh chan struct{}

	// in use index from 0x01 -> 0xf8, indicates the pod using this index
	// When pod completed CDI and started or pending image pulling, should be removed from the queue
	nodeIndexQueue map[string]map[int]types.NamespacedName

	podIndexMap map[types.NamespacedName]indexIdentifier

	asyncCheckingMap map[types.NamespacedName]struct{}
}

type indexIdentifier struct {
	nodeName string
	index    int
}

func NewIndexAllocator(ctx context.Context, client client.Client) (*IndexAllocator, error) {
	if client == nil {
		return nil, fmt.Errorf("client cannot be nil")
	}

	allocator := &IndexAllocator{
		Client:        client,
		IsLeader:      false,
		currentIndex:  0, // Will start from 1 on first assignment
		ctx:           ctx,
		initializedCh: make(chan struct{}),

		nodeIndexQueue:   make(map[string]map[int]types.NamespacedName, 128),
		asyncCheckingMap: make(map[types.NamespacedName]struct{}, 128),

		podIndexMap: make(map[types.NamespacedName]indexIdentifier, 128),
	}

	return allocator, nil
}

func (s *IndexAllocator) SetupWithManager(ctx context.Context, mgr manager.Manager) <-chan struct{} {
	readyCh := make(chan struct{}, 1)
	_ = mgr.Add(manager.RunnableFunc(func(ctx context.Context) error {
		<-mgr.Elected()
		s.IsLeader = true
		readyCh <- struct{}{}
		return nil
	}))
	return readyCh
}

// AssignIndex assigns a temporary index (1-128) for Pod-to-DevicePlugin communication
// Uses atomic increment to ensure thread-safe assignment
// Index wraps around from 128 to 1 (simple modulo operation)
func (s *IndexAllocator) AssignIndex(podName string) (int, error) {
	if !s.IsLeader {
		log.FromContext(s.ctx).Error(nil, "only leader can assign index", "podName", podName)
		return 0, fmt.Errorf("only leader can assign index")
	}
	// Atomic increment and wrap around
	next := atomic.AddInt64(&s.currentIndex, 1)
	index := int((next-1)%(constants.IndexModLength*constants.IndexKeyLength)) + 1
	log.FromContext(s.ctx).Info("assigned index successfully", "podName", podName, "index", index)
	return index, nil
}

// ReconcileLockState maintains memory state for node level index assign and release queue
func (s *IndexAllocator) ReconcileLockState(pod *v1.Pod) {
	if pod.Labels[constants.LabelComponent] != constants.ComponentWorker {
		return
	}
	// Check if it's TF indexed Pod by container resource limits
	// If isIndex But PodIndex not set, check phase, if pending, should assign index, next check
	if pod.Spec.NodeName == "" {
		return
	}

	index, err := utils.ParsePodIndexResourceClaim(pod)
	if err != nil {
		log.FromContext(s.ctx).Error(err, "not TF indexed Pod, skip reconcile lock state", "pod", pod.Name)
		return
	}
	_, indexAllocated := pod.Annotations[constants.PodIndexAnnotation]

	// Only pending pods can occupy the node level index
	if utils.IsPodPending(pod) {
		s.storeMutex.Lock()
		indexQueue := s.nodeIndexQueue[pod.Spec.NodeName]
		if indexQueue == nil {
			indexQueue = make(map[int]types.NamespacedName, 8)
			s.nodeIndexQueue[pod.Spec.NodeName] = indexQueue
		}

		// If just started and missing in memory, should complement the index queue and pod index map
		if indexAllocated {
			// occupy the index if missing (when scheduler restarted)
			if _, exists := indexQueue[index]; !exists {
				podMeta := types.NamespacedName{
					Namespace: pod.Namespace,
					Name:      pod.Name,
				}
				indexQueue[index] = podMeta
				s.podIndexMap[podMeta] = indexIdentifier{
					nodeName: pod.Spec.NodeName,
					index:    index,
				}
			}
			s.storeMutex.Unlock()
			return
		}

		if podMeta, exists := indexQueue[index]; exists {
			// If already occupied by other Pod, check if it's the same Pod
			if podMeta.Namespace != pod.Namespace || podMeta.Name != pod.Name {
				log.FromContext(s.ctx).Error(fmt.Errorf("pod index conflict"), "can not reconcile index lock, more than one pending pods occupy the same index", "pod", pod.Name, "index", index)
				s.storeMutex.Unlock()
				return
			}
			// Same pod already exists in the queue, no need to do anything
			s.storeMutex.Unlock()
			return
		} else {
			// new Pod occupy the index, add to index queue
			indexQueue[index] = types.NamespacedName{
				Namespace: pod.Namespace,
				Name:      pod.Name,
			}
			s.podIndexMap[types.NamespacedName{
				Namespace: pod.Namespace,
				Name:      pod.Name,
			}] = indexIdentifier{
				nodeName: pod.Spec.NodeName,
				index:    index,
			}
			s.storeMutex.Unlock()
			// Brand new pending pod, ensure the async checking loop for assigning index annotation
			s.AsyncCheckNodeIndexAvailableAndAssign(pod, index)
		}
	} else if utils.IsPodRunning(pod) {
		s.RemoveNodeIndexQueueForPod(types.NamespacedName{
			Namespace: pod.Namespace,
			Name:      pod.Name,
		})
	}
}

func (s *IndexAllocator) RemoveNodeIndexQueueForPod(namespacedName types.NamespacedName) {
	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	indexIdentifier, exists := s.podIndexMap[namespacedName]
	if !exists {
		return
	}
	if indexQueue, exists := s.nodeIndexQueue[indexIdentifier.nodeName]; exists {
		if val, exists := indexQueue[indexIdentifier.index]; exists {
			if val.Namespace == namespacedName.Namespace && val.Name == namespacedName.Name {
				delete(indexQueue, indexIdentifier.index)
				log.FromContext(s.ctx).Info("Removed pod from node index queue after pod running/stopped/deleted", "pod", namespacedName, "index", indexIdentifier.index)
			}
			delete(s.podIndexMap, namespacedName)
		}
	}
}

func (s *IndexAllocator) CheckNodeIndexAndTryOccupy(pod *v1.Pod, index int) bool {
	<-s.initializedCh
	nodeName := pod.Spec.NodeName
	if nodeName == "" {
		// should not happen, unscheduled pod
		return false
	}

	// Use write lock to ensure atomic check-and-occupy operation
	// This prevents race condition where another goroutine occupies the index
	// between the check and occupy operations
	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	// Initialize index queue if not exists
	indexQueue := s.nodeIndexQueue[nodeName]
	if indexQueue == nil {
		indexQueue = make(map[int]types.NamespacedName, 8)
		s.nodeIndexQueue[nodeName] = indexQueue
	}

	// Atomically check and occupy index
	occupiedPod, exists := indexQueue[index]
	if !exists || (occupiedPod.Namespace == pod.Namespace && occupiedPod.Name == pod.Name) {
		indexQueue[index] = types.NamespacedName{
			Namespace: pod.Namespace,
			Name:      pod.Name,
		}
		return true
	}
	return false
}

func (s *IndexAllocator) SetReady() {
	close(s.initializedCh)
}

func (s *IndexAllocator) AsyncCheckNodeIndexAvailableAndAssign(pod *v1.Pod, index int) {
	podMeta := types.NamespacedName{
		Namespace: pod.Namespace,
		Name:      pod.Name,
	}

	s.storeMutex.Lock()
	if _, exists := s.asyncCheckingMap[podMeta]; exists {
		// already started checking loop, skip
		s.storeMutex.Unlock()
		return
	}
	s.asyncCheckingMap[podMeta] = struct{}{}
	s.storeMutex.Unlock()

	go func() {
		defer func() {
			s.storeMutex.Lock()
			delete(s.asyncCheckingMap, types.NamespacedName{
				Namespace: pod.Namespace,
				Name:      pod.Name,
			})
			s.storeMutex.Unlock()
		}()

		// Infinity backoff retry until index is available, and also reconcile started
		_ = retry.OnError(wait.Backoff{
			Duration: 3 * time.Second,
			Factor:   1.4,
			Jitter:   0.1,
			Steps:    math.MaxInt32,
			Cap:      60 * time.Minute,
		}, func(err error) bool {
			return true
		}, func() error {
			latestPod := &v1.Pod{}
			if err := s.Client.Get(s.ctx, client.ObjectKeyFromObject(pod), latestPod); err != nil {
				if errors.IsNotFound(err) {
					// pod is deleted, stop retrying
					return nil
				}
				return err
			}
			if utils.IsPodStopped(latestPod) {
				return nil
			}
			// Skip if index is already assigned or no annotation
			if latestPod.Annotations == nil || latestPod.Annotations[constants.PodIndexAnnotation] != "" {
				if utils.IsPodRunning(latestPod) {
					log.FromContext(s.ctx).Info("[WARNING] pod is running without index allocation hypervisor may not working",
						"pod", latestPod.Name, "node", latestPod.Spec.NodeName)
					return nil
				}
				// else do nothing, may caused by duplicated reconciling
			}

			if !s.CheckNodeIndexAndTryOccupy(latestPod, index) {
				return fmt.Errorf("index is not available")
			}
			// Index available, patch annotation to transit Pod from Pending to DeviceAllocating in hypervisor
			patchOps := map[string]any{
				"op":    "add",
				"path":  "/metadata/annotations/" + utils.EscapeJSONPointer(constants.PodIndexAnnotation),
				"value": strconv.Itoa(index),
			}
			patchBytes, err := json.Marshal(patchOps)
			if err != nil {
				return err
			}
			err = s.Client.Patch(s.ctx, latestPod, client.RawPatch(types.JSONPatchType, patchBytes))
			if err != nil {
				log.FromContext(s.ctx).Error(err, "failed to patch pod index annotation", "pod", latestPod.Name, "index", index)
				return err
			}
			return nil
		})
	}()
}
