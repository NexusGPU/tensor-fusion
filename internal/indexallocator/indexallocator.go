package indexallocator

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
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
	podMeta := types.NamespacedName{
		Namespace: pod.Namespace,
		Name:      pod.Name,
	}

	// Only pending pods can occupy the node level index
	if utils.IsPodPending(pod) {
		for {
			s.storeMutex.Lock()
			indexQueue := s.nodeIndexQueue[pod.Spec.NodeName]
			if indexQueue == nil {
				indexQueue = make(map[int]types.NamespacedName, 8)
				s.nodeIndexQueue[pod.Spec.NodeName] = indexQueue
			}
			occupiedPod, exists := indexQueue[index]
			if !exists || occupiedPod == podMeta {
				s.trackPodIndexLocked(podMeta, pod.Spec.NodeName, index)
				s.storeMutex.Unlock()
				if !indexAllocated {
					// Pending pods without the index annotation still need the async patch loop.
					s.AsyncCheckNodeIndexAvailableAndAssign(pod, index)
				}
				return
			}
			s.storeMutex.Unlock()

			if !s.tryReleaseStaleOccupant(occupiedPod, pod.Spec.NodeName, index) {
				log.FromContext(s.ctx).Error(
					fmt.Errorf("pod index conflict"),
					"can not reconcile index lock, more than one pending pods occupy the same index",
					"pod", pod.Name,
					"index", index,
				)
				return
			}
		}
	}
	if utils.IsPodRunning(pod) || utils.IsPodStopped(pod) {
		s.RemoveNodeIndexQueueForPod(podMeta)
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
		}
		if len(indexQueue) == 0 {
			delete(s.nodeIndexQueue, indexIdentifier.nodeName)
		}
	}
	delete(s.podIndexMap, namespacedName)
}

func (s *IndexAllocator) CheckNodeIndexAndTryOccupy(pod *v1.Pod, index int) bool {
	<-s.initializedCh
	nodeName := pod.Spec.NodeName
	if nodeName == "" {
		// should not happen, unscheduled pod
		return false
	}
	podMeta := types.NamespacedName{
		Namespace: pod.Namespace,
		Name:      pod.Name,
	}

	for {
		// Use write lock to ensure atomic check-and-occupy operation
		// This prevents race condition where another goroutine occupies the index
		// between the check and occupy operations
		s.storeMutex.Lock()

		// Initialize index queue if not exists
		indexQueue := s.nodeIndexQueue[nodeName]
		if indexQueue == nil {
			indexQueue = make(map[int]types.NamespacedName, 8)
			s.nodeIndexQueue[nodeName] = indexQueue
		}

		// Atomically check and occupy index
		occupiedPod, exists := indexQueue[index]
		if !exists || occupiedPod == podMeta {
			s.trackPodIndexLocked(podMeta, nodeName, index)
			s.storeMutex.Unlock()
			return true
		}
		s.storeMutex.Unlock()

		if !s.tryReleaseStaleOccupant(occupiedPod, nodeName, index) {
			return false
		}
	}
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
			base := latestPod.DeepCopy()
			if latestPod.Annotations == nil {
				latestPod.Annotations = make(map[string]string, 1)
			}
			latestPod.Annotations[constants.PodIndexAnnotation] = strconv.Itoa(index)
			err := s.Client.Patch(s.ctx, latestPod, client.MergeFrom(base))
			if err != nil {
				log.FromContext(s.ctx).Error(err, "failed to patch pod index annotation", "pod", latestPod.Name, "index", index)
				return err
			}
			return nil
		})
	}()
}

func (s *IndexAllocator) trackPodIndexLocked(podMeta types.NamespacedName, nodeName string, index int) {
	indexQueue := s.nodeIndexQueue[nodeName]
	if indexQueue == nil {
		indexQueue = make(map[int]types.NamespacedName, 8)
		s.nodeIndexQueue[nodeName] = indexQueue
	}
	indexQueue[index] = podMeta
	s.podIndexMap[podMeta] = indexIdentifier{
		nodeName: nodeName,
		index:    index,
	}
}

func (s *IndexAllocator) tryReleaseStaleOccupant(occupiedPod types.NamespacedName, nodeName string, index int) bool {
	ctx := log.IntoContext(s.ctx, log.FromContext(s.ctx).WithValues(
		"occupiedPod", occupiedPod,
		"node", nodeName,
		"index", index,
	))

	livePod := &v1.Pod{}
	if err := s.Client.Get(ctx, occupiedPod, livePod); err != nil {
		if !errors.IsNotFound(err) {
			log.FromContext(ctx).Error(err, "failed to verify index occupant")
			return false
		}
		s.releaseSpecificIndexOccupancy(occupiedPod, nodeName, index)
		log.FromContext(ctx).Info("released stale index occupant because pod no longer exists")
		return true
	}

	if livePod.Spec.NodeName != nodeName || utils.IsPodRunning(livePod) || utils.IsPodStopped(livePod) || !livePod.DeletionTimestamp.IsZero() {
		s.releaseSpecificIndexOccupancy(occupiedPod, nodeName, index)
		log.FromContext(ctx).Info(
			"released stale index occupant after verifying pod is no longer active on node",
			"phase", livePod.Status.Phase,
			"deleting", !livePod.DeletionTimestamp.IsZero(),
			"actualNode", livePod.Spec.NodeName,
		)
		return true
	}

	return false
}

func (s *IndexAllocator) releaseSpecificIndexOccupancy(occupiedPod types.NamespacedName, nodeName string, index int) {
	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	if indexQueue, exists := s.nodeIndexQueue[nodeName]; exists {
		if current, ok := indexQueue[index]; ok && current == occupiedPod {
			delete(indexQueue, index)
		}
		if len(indexQueue) == 0 {
			delete(s.nodeIndexQueue, nodeName)
		}
	}

	if mapped, exists := s.podIndexMap[occupiedPod]; exists && mapped.nodeName == nodeName && mapped.index == index {
		delete(s.podIndexMap, occupiedPod)
	}
}
