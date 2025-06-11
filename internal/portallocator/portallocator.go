package portallocator

import (
	"context"
	"fmt"
	"math/bits"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/util/retry"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

const RELEASE_PORT_RETRY_INTERVAL = 10 * time.Second

var RETRY_CONFIG = wait.Backoff{
	Steps:    100,
	Duration: RELEASE_PORT_RETRY_INTERVAL,
	Factor:   1.1,
	Jitter:   0.1,
}

// Offer API for host port allocation, range from user configured port range
// Use label: `tensor-fusion.ai/host-port: auto` to assigned port at cluster level
// vGPU worker's hostPort will be managed by operator
type PortAllocator struct {
	PortRangeStartNode int
	PortRangeEndNode   int

	PortRangeStartCluster int
	PortRangeEndCluster   int

	IsLeader bool

	BitmapPerNode map[string][]uint64
	BitmapCluster []uint64

	Client client.Client

	storeMutexNode    sync.RWMutex
	storeMutexCluster sync.RWMutex
	ctx               context.Context

	clusterLevelPortReleaseQueue chan struct {
		podName string
		port    int
	}

	nodeLevelPortReleaseQueue chan struct {
		nodeName string
		podName  string
		port     int
	}
}

func NewPortAllocator(ctx context.Context, client client.Client, nodeLevelPortRange string, clusterLevelPortRange string) (*PortAllocator, error) {
	if client == nil {
		return nil, fmt.Errorf("client cannot be nil")
	}

	nodeLevelRange := strings.Split(nodeLevelPortRange, "-")
	clusterLevelRange := strings.Split(clusterLevelPortRange, "-")

	portRangeStartNode, _ := strconv.Atoi(nodeLevelRange[0])
	portRangeEndNode, _ := strconv.Atoi(nodeLevelRange[1])

	portRangeStartCluster, _ := strconv.Atoi(clusterLevelRange[0])
	portRangeEndCluster, _ := strconv.Atoi(clusterLevelRange[1])

	allocator := &PortAllocator{
		PortRangeStartNode:    portRangeStartNode,
		PortRangeEndNode:      portRangeEndNode,
		PortRangeStartCluster: portRangeStartCluster,
		PortRangeEndCluster:   portRangeEndCluster,
		Client:                client,
		IsLeader:              false,
		BitmapPerNode:         make(map[string][]uint64),
		BitmapCluster:         make([]uint64, (portRangeEndCluster-portRangeStartCluster)/64+1),

		storeMutexNode:    sync.RWMutex{},
		storeMutexCluster: sync.RWMutex{},
		ctx:               ctx,

		clusterLevelPortReleaseQueue: make(chan struct {
			podName string
			port    int
		}),
		nodeLevelPortReleaseQueue: make(chan struct {
			nodeName string
			podName  string
			port     int
		}),
	}

	go allocator.releaseClusterPortUntilPodDeleted()
	go allocator.releaseNodePortUntilPodDeleted()

	return allocator, nil
}

func (s *PortAllocator) SetupWithManager(ctx context.Context, mgr manager.Manager) <-chan struct{} {
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

		s.storeMutexNode.Lock()
		s.storeMutexCluster.Lock()
		defer s.storeMutexNode.Unlock()
		defer s.storeMutexCluster.Unlock()

		// 1. init bit map from existing pods labeled with tensor-fusion.ai/host-port=auto
		s.initBitMapForClusterLevelPortAssign(ctx)

		// 2. init bit map for existing vGPU workers
		s.initBitMapForNodeLevelPortAssign(ctx)

		readyCh <- struct{}{}
		return nil
	}))
	return readyCh
}

func (s *PortAllocator) GetLeaderIP() string {
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

// AssignHostPort always called by operator itself, thus no Leader-Follower inconsistency issue
func (s *PortAllocator) AssignHostPort(nodeName string) (int, error) {
	if nodeName == "" {
		return 0, fmt.Errorf("node name cannot be empty when assign host port")
	}
	s.storeMutexNode.Lock()
	defer s.storeMutexNode.Unlock()

	bitmap, ok := s.BitmapPerNode[nodeName]
	if !ok {
		// found new nodes not have any ports assigned before
		bitmapSize := (s.PortRangeEndNode - s.PortRangeStartNode + 63) / 64
		s.BitmapPerNode[nodeName] = make([]uint64, bitmapSize)
		bitmap = s.BitmapPerNode[nodeName]
	}
	for i, subMap := range bitmap {
		bitPos := bits.TrailingZeros64(^subMap)
		portOffset := i*64 + bitPos
		if subMap != 0xFFFFFFFFFFFFFFFF {
			assignedPort := portOffset + s.PortRangeStartNode
			if assignedPort < s.PortRangeEndNode {
				bitmap[i] = subMap | (1 << bitPos)
				return assignedPort, nil
			} else {
				break
			}
		}
	}
	return 0, fmt.Errorf("no available port on node %s", nodeName)

}

func (s *PortAllocator) ReleaseHostPort(nodeName string, podName string, port int, immediateRelease bool) error {
	if port == 0 {
		return fmt.Errorf("port cannot be 0 when release host port, may caused by portNumber annotation not detected, nodeName: %s, podName: %s", nodeName, podName)
	}

	if bitmap, ok := s.BitmapPerNode[nodeName]; !ok {
		return fmt.Errorf("node %s not found in bitmap", nodeName)
	} else {

		portOffset := port - s.PortRangeStartNode
		if immediateRelease {
			s.storeMutexNode.Lock()
			defer s.storeMutexNode.Unlock()

			bitmap[portOffset/64] &^= 1 << (portOffset % 64)
		} else {
			// put into queue, release until Pod not found
			s.nodeLevelPortReleaseQueue <- struct {
				nodeName string
				podName  string
				port     int
			}{nodeName, podName, port}
		}
	}
	return nil
}

func (s *PortAllocator) AssignClusterLevelHostPort(podName string) (int, error) {

	s.storeMutexCluster.Lock()
	defer s.storeMutexCluster.Unlock()

	for i, subMap := range s.BitmapCluster {
		bitPos := bits.TrailingZeros64(^subMap)
		portOffset := i*64 + bitPos
		if subMap != 0xFFFFFFFFFFFFFFFF {
			assignedPort := portOffset + s.PortRangeStartCluster
			if assignedPort < s.PortRangeEndCluster {
				s.BitmapCluster[i] |= 1 << bitPos
				return assignedPort, nil
			}
		}
	}
	return 0, fmt.Errorf("no available port on cluster")
}

func (s *PortAllocator) ReleaseClusterLevelHostPort(podName string, port int, immediateRelease bool) error {
	if port == 0 {
		return fmt.Errorf("port cannot be 0 when release host port, may caused by portNumber annotation not detected, podName: %s", podName)
	}

	portOffset := port - s.PortRangeStartCluster

	if immediateRelease {
		s.storeMutexCluster.Lock()
		defer s.storeMutexCluster.Unlock()
		s.BitmapCluster[portOffset/64] &^= 1 << (portOffset % 64)
		return nil
	} else {
		// put into queue, release until Pod not found
		s.clusterLevelPortReleaseQueue <- struct {
			podName string
			port    int
		}{podName, port}
	}
	return nil
}

func (s *PortAllocator) releaseClusterPortUntilPodDeleted() {
	for item := range s.clusterLevelPortReleaseQueue {
		podName := item.podName
		portOffset := item.port - s.PortRangeStartCluster

		_ = retry.OnError(RETRY_CONFIG, func(_ error) bool {
			return true
		}, func() error {
			pod := &v1.Pod{}
			err := s.Client.Get(s.ctx, client.ObjectKey{Name: podName}, pod)
			if errors.IsNotFound(err) {
				s.storeMutexCluster.Lock()
				defer s.storeMutexCluster.Unlock()
				s.BitmapCluster[portOffset/64] &^= 1 << (portOffset % 64)
				return nil
			}
			return fmt.Errorf("pod still there, can not release port %s", podName)
		})
	}
}

func (s *PortAllocator) releaseNodePortUntilPodDeleted() {
	for item := range s.nodeLevelPortReleaseQueue {
		podName := item.podName
		portOffset := item.port - s.PortRangeStartNode

		go func() {
			_ = retry.OnError(RETRY_CONFIG, func(_ error) bool {
				return true
			}, func() error {
				pod := &v1.Pod{}
				err := s.Client.Get(s.ctx, client.ObjectKey{Name: podName}, pod)
				if errors.IsNotFound(err) {
					s.storeMutexNode.Lock()
					defer s.storeMutexNode.Unlock()
					s.BitmapPerNode[item.nodeName][portOffset/64] &^= 1 << (portOffset % 64)
					return nil
				}
				return fmt.Errorf("pod still there, can not release port %s", podName)
			})
		}()
	}
}

func (s *PortAllocator) initBitMapForClusterLevelPortAssign(ctx context.Context) {
	log := log.FromContext(ctx)
	podList := &v1.PodList{}
	err := s.Client.List(ctx, podList, client.MatchingLabels{constants.GenHostPortLabel: constants.GenHostPortLabelValue})
	if err != nil {
		log.Error(err, "failed to list pods with port allocation label")
		return
	}
	usedPorts := []uint16{}
	for _, pod := range podList.Items {
		if pod.Annotations == nil {
			continue
		}
		port, _ := strconv.Atoi(pod.Annotations[constants.GenPortNumberAnnotation])
		if port > s.PortRangeEndCluster || port < s.PortRangeStartCluster {
			log.Error(err, "existing Pod's host port out of range", "port", port, "expected-start", s.PortRangeStartCluster, "expected-end", s.PortRangeEndCluster, "pod", pod.Name)
			continue
		}
		bitOffSet := port - s.PortRangeStartCluster

		usedPorts = append(usedPorts, uint16(bitOffSet))
	}

	for _, port := range usedPorts {
		s.BitmapCluster[port/64] |= 1 << (port % 64)
	}
}

func (s *PortAllocator) initBitMapForNodeLevelPortAssign(ctx context.Context) {
	log := log.FromContext(ctx)
	podList := &v1.PodList{}
	err := s.Client.List(ctx, podList, client.MatchingLabels{constants.LabelComponent: constants.ComponentWorker})
	if err != nil {
		log.Error(err, "failed to list pods with port allocation label")
		return
	}

	size := (s.PortRangeEndNode-s.PortRangeStartNode)/64 + 1
	for _, pod := range podList.Items {
		if pod.Annotations == nil {
			continue
		}
		port, _ := strconv.Atoi(pod.Annotations[constants.GenPortNumberAnnotation])
		if port > s.PortRangeEndNode || port < s.PortRangeStartNode {
			log.Error(err, "existing Pod's node level host port out of range", "port", port, "expected-start", s.PortRangeStartNode, "expected-end", s.PortRangeEndNode, "pod", pod.Name, "node", pod.Spec.NodeName)
			continue
		}
		bitOffSet := port - s.PortRangeStartNode
		if _, ok := s.BitmapPerNode[pod.Spec.NodeName]; !ok {
			s.BitmapPerNode[pod.Spec.NodeName] = make([]uint64, size)
		}
		s.BitmapPerNode[pod.Spec.NodeName][bitOffSet/64] |= 1 << (bitOffSet % 64)
	}

}
