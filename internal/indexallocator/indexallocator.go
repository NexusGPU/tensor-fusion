package indexallocator

import (
	"context"
	"fmt"
	"math/bits"
	"strconv"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/util/retry"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

const (
	IndexRangeStart = 1
	IndexRangeEnd   = 512
	IndexBitmapSize = (IndexRangeEnd - IndexRangeStart + 63) / 64
)

const RELEASE_INDEX_RETRY_INTERVAL = 10 * time.Second

var RETRY_CONFIG = wait.Backoff{
	Steps:    100,
	Duration: RELEASE_INDEX_RETRY_INTERVAL,
	Factor:   1.1,
	Jitter:   0.1,
}

// IndexAllocator manages allocation of 1-512 temporary indices for Pod-to-DevicePlugin communication
type IndexAllocator struct {
	IsLeader bool

	Bitmap []uint64

	Client client.Client

	storeMutex sync.RWMutex
	ctx        context.Context

	indexReleaseQueue chan struct {
		podName string
		index   int
	}
}

func NewIndexAllocator(ctx context.Context, client client.Client) (*IndexAllocator, error) {
	if client == nil {
		return nil, fmt.Errorf("client cannot be nil")
	}

	allocator := &IndexAllocator{
		Client:   client,
		IsLeader: false,
		Bitmap:   make([]uint64, IndexBitmapSize),

		storeMutex: sync.RWMutex{},
		ctx:        ctx,

		indexReleaseQueue: make(chan struct {
			podName string
			index   int
		}),
	}

	go allocator.releaseIndexUntilPodDeleted()

	return allocator, nil
}

func (s *IndexAllocator) SetupWithManager(ctx context.Context, mgr manager.Manager) <-chan struct{} {
	readyCh := make(chan struct{}, 1)
	_ = mgr.Add(manager.RunnableFunc(func(ctx context.Context) error {
		<-mgr.Elected()
		s.IsLeader = true
		leaderInfo := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constants.LeaderInfoConfigMapName,
				Namespace: utils.CurrentNamespace(),
			},
		}
		err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
			_, err := controllerutil.CreateOrUpdate(ctx, s.Client, leaderInfo, func() error {
				leaderInfo.Data = map[string]string{
					constants.LeaderInfoConfigMapLeaderIPKey: utils.CurrentIP(),
				}
				return nil
			})
			return err
		})
		if err != nil {
			log.FromContext(ctx).Error(err, "Failed to update leader IP info in ConfigMap")
		}

		s.storeMutex.Lock()
		defer s.storeMutex.Unlock()

		// Initialize bitmap from existing pods with index annotation
		s.initBitmap(ctx)

		readyCh <- struct{}{}
		return nil
	}))
	return readyCh
}

func (s *IndexAllocator) GetLeaderIP() string {
	leaderInfo := &v1.ConfigMap{}
	err := s.Client.Get(context.Background(), client.ObjectKey{
		Name:      constants.LeaderInfoConfigMapName,
		Namespace: utils.CurrentNamespace(),
	}, leaderInfo)
	if err != nil {
		log.FromContext(context.Background()).Error(err, "Failed to get leader IP info from ConfigMap")
		return ""
	}
	if leaderInfo.Data == nil {
		return ""
	}
	return leaderInfo.Data[constants.LeaderInfoConfigMapLeaderIPKey]
}

// AssignIndex assigns a temporary index (1-512) for Pod-to-DevicePlugin communication
// Uses distributed lock via leader election to ensure global increment
// Index is assigned in ascending order (1, 2, 3, ...) to maintain consistency
func (s *IndexAllocator) AssignIndex(podName string) (int, error) {
	if !s.IsLeader {
		return 0, fmt.Errorf("only leader can assign index")
	}

	s.storeMutex.Lock()
	defer s.storeMutex.Unlock()

	// Find first available index in ascending order (1, 2, 3, ...)
	// This ensures consistent index assignment across distributed webhook instances
	// TrailingZeros64 finds the first zero bit (lowest available index)
	for i, subMap := range s.Bitmap {
		bitPos := bits.TrailingZeros64(^subMap)
		indexOffset := i*64 + bitPos
		if subMap != 0xFFFFFFFFFFFFFFFF {
			assignedIndex := indexOffset + IndexRangeStart
			if assignedIndex <= IndexRangeEnd {
				// Mark this index as used
				s.Bitmap[i] = subMap | (1 << bitPos)
				return assignedIndex, nil
			} else {
				break
			}
		}
	}

	// If all indices in first pass are used, wrap around and find first available
	// This handles the case when 512 is reached and we need to reuse released indices
	for i := 0; i < IndexBitmapSize; i++ {
		for j := 0; j < 64; j++ {
			indexOffset := i*64 + j
			assignedIndex := indexOffset + IndexRangeStart
			if assignedIndex > IndexRangeEnd {
				break
			}
			// Check if this index is available (bit is 0)
			if s.Bitmap[i]&(1<<j) == 0 {
				s.Bitmap[i] |= 1 << j
				return assignedIndex, nil
			}
		}
	}

	return 0, fmt.Errorf("no available index, all 512 indices are in use")
}

func (s *IndexAllocator) ReleaseIndex(podName string, index int, immediateRelease bool) error {
	if index < IndexRangeStart || index > IndexRangeEnd {
		return fmt.Errorf("index %d out of range [%d, %d]", index, IndexRangeStart, IndexRangeEnd)
	}

	indexOffset := index - IndexRangeStart

	if immediateRelease {
		s.storeMutex.Lock()
		defer s.storeMutex.Unlock()
		s.Bitmap[indexOffset/64] &^= 1 << (indexOffset % 64)
		return nil
	} else {
		// Put into queue, release until Pod not found
		s.indexReleaseQueue <- struct {
			podName string
			index   int
		}{podName, index}
	}
	return nil
}

func (s *IndexAllocator) releaseIndexUntilPodDeleted() {
	for item := range s.indexReleaseQueue {
		podName := item.podName
		indexOffset := item.index - IndexRangeStart

		_ = retry.OnError(RETRY_CONFIG, func(_ error) bool {
			return true
		}, func() error {
			// Try to get pod by name from any namespace
			podList := &v1.PodList{}
			err := s.Client.List(s.ctx, podList)
			if err != nil {
				return err
			}

			found := false
			for i := range podList.Items {
				if podList.Items[i].Name == podName {
					found = true
					break
				}
			}

			if !found {
				s.storeMutex.Lock()
				defer s.storeMutex.Unlock()
				s.Bitmap[indexOffset/64] &^= 1 << (indexOffset % 64)
				return nil
			}
			return fmt.Errorf("pod still there, cannot release index %d for pod %s", item.index, podName)
		})
	}
}

func (s *IndexAllocator) initBitmap(ctx context.Context) {
	log := log.FromContext(ctx)
	podList := &v1.PodList{}
	err := s.Client.List(ctx, podList)
	if err != nil {
		log.Error(err, "failed to list pods for index bitmap initialization")
		return
	}

	for _, pod := range podList.Items {
		if pod.Annotations == nil {
			continue
		}
		indexStr, exists := pod.Annotations[constants.PodIndexAnnotation]
		if !exists || indexStr == "" {
			continue
		}
		index, err := strconv.Atoi(indexStr)
		if err != nil {
			log.V(5).Info("failed to parse index annotation", "pod", pod.Name, "index", indexStr)
			continue
		}
		if index < IndexRangeStart || index > IndexRangeEnd {
			log.V(5).Info("index out of range", "pod", pod.Name, "index", index)
			continue
		}

		// Check if pod is still running/starting, if not, don't mark as used
		if pod.DeletionTimestamp != nil || pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
			continue
		}

		indexOffset := index - IndexRangeStart
		s.Bitmap[indexOffset/64] |= 1 << (indexOffset % 64)
	}
}
