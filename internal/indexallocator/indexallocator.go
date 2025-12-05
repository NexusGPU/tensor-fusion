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
func (s *IndexAllocator) ReconcileLockState(pod *v1.Pod) bool {
	if pod.Labels[constants.LabelComponent] != constants.ComponentWorker {
		return false
	}
	// Check if it's TF indexed Pod by container resource limits
	// If isIndex But PodIndex not set, check phase, if pending, should assign index, next check
	if pod.Spec.NodeName == "" {
		return false
	}

	index := pod.Annotations[constants.PodIndexAnnotation]
	if index == "" {
		return false
	}
	indexInt, err := strconv.Atoi(index)
	if err != nil {
		return false
	}

	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	// Check Pod status
	// TODO: call in Pod controller and gpu Allocator init stage

	indexQueue := s.nodeIndexQueue[pod.Spec.NodeName]
	if indexQueue == nil {
		indexQueue = make(map[int]types.NamespacedName)
		s.nodeIndexQueue[pod.Spec.NodeName] = indexQueue
	}
	indexQueue[indexInt] = types.NamespacedName{
		Namespace: pod.Namespace,
		Name:      pod.Name,
	}
	return true
}

func (s *IndexAllocator) CheckNodeIndexAvailableForPod(pod *v1.Pod, index int) bool {
	<-s.initializedCh
	nodeName := pod.Spec.NodeName
	if nodeName == "" {
		// should not happen, unscheduled pod
		return false
	}
	s.storeMutex.RLock()
	defer s.storeMutex.RUnlock()
	indexQueue := s.nodeIndexQueue[nodeName]
	if len(indexQueue) == 0 {
		return false
	}
	_, exists := indexQueue[index]
	return !exists
}

func (s *IndexAllocator) SetReady() {
	close(s.initializedCh)
}

func (s *IndexAllocator) CheckNodeIndexAvailableAndAssign(pod *v1.Pod, index int) {
	go func() {
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
			pod := &v1.Pod{}
			if err := s.Client.Get(s.ctx, client.ObjectKeyFromObject(pod), pod); err != nil {
				if errors.IsNotFound(err) {
					// pod is deleted, stop retrying
					return nil
				}
				return err
			}
			if utils.IsPodStopped(pod) {
				return nil
			}
			// Skip if index is already assigned or no annotation
			if pod.Annotations == nil || pod.Annotations[constants.PodIndexAnnotation] != "" {
				if utils.IsPodRunning(pod) {
					log.FromContext(s.ctx).Info("[WARNING] pod is running without index allocation hypervisor may not working",
						"pod", pod.Name, "node", pod.Spec.NodeName)
					return nil
				}
			}

			if !s.CheckNodeIndexAvailableForPod(pod, index) {
				return fmt.Errorf("index is not available")
			}
			// Index available, patch annotation to transit Pod from Pending to DeviceAllocating in hypervisor
			patchOps := map[string]any{
				"op":    "add",
				"path":  "/metadata/annotations/" + utils.EscapeJSONPointer(constants.PodIndexAnnotation),
				"value": index,
			}
			patchBytes, err := json.Marshal(patchOps)
			if err != nil {
				return err
			}
			err = s.Client.Patch(s.ctx, pod, client.RawPatch(types.JSONPatchType, patchBytes))
			if err != nil {
				log.FromContext(s.ctx).Error(err, "failed to patch pod index annotation", "pod", pod.Name, "index", index)
				return err
			}
			return nil
		})
	}()
}
