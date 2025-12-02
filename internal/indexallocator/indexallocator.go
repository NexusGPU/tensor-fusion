package indexallocator

import (
	"context"
	"fmt"
	"sync/atomic"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/util/retry"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

// IndexAllocator manages allocation of 1-512 temporary indices for Pod-to-DevicePlugin communication
// Uses a simple atomic counter that increments from 1 to 512, then wraps around to 1
// No bitmap tracking needed - index reuse is acceptable after 512 cycles
type IndexAllocator struct {
	IsLeader bool

	// Atomic counter for index allocation (1-512, wraps around)
	currentIndex int64

	Client client.Client

	ctx context.Context
}

func NewIndexAllocator(ctx context.Context, client client.Client) (*IndexAllocator, error) {
	if client == nil {
		return nil, fmt.Errorf("client cannot be nil")
	}

	allocator := &IndexAllocator{
		Client:       client,
		IsLeader:     false,
		currentIndex: 0, // Will start from 1 on first assignment
		ctx:          ctx,
	}

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
