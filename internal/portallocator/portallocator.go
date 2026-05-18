package portallocator

import (
	"context"
	"fmt"
	"math/bits"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/util/retry"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
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
		namespace string
		podName   string
		port      int
	}

	nodeLevelPortReleaseQueue chan struct {
		nodeName string
		podName  string
		port     int
	}
}

// parsePortRange parses an "<start>-<end>" string and validates that both
// sides are well-formed integers and that start < end. A misconfigured flag
// (missing dash, non-numeric, reversed) would otherwise either panic on slice
// indexing, silently produce a zero-size bitmap, or pass a negative length to
// make() — all of which surface only at the first allocation request.
func parsePortRange(raw string, label string) (int, int, error) {
	parts := strings.Split(raw, "-")
	if len(parts) != 2 {
		return 0, 0, fmt.Errorf("invalid %s port range %q: expected format <start>-<end>", label, raw)
	}
	start, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, 0, fmt.Errorf("invalid %s port range %q: start: %w", label, raw, err)
	}
	end, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0, 0, fmt.Errorf("invalid %s port range %q: end: %w", label, raw, err)
	}
	if start <= 0 || end <= 0 || start >= end {
		return 0, 0, fmt.Errorf("invalid %s port range %q: require 0 < start < end", label, raw)
	}
	return start, end, nil
}

func NewPortAllocator(ctx context.Context, client client.Client, nodeLevelPortRange string, clusterLevelPortRange string) (*PortAllocator, error) {
	if client == nil {
		return nil, fmt.Errorf("client cannot be nil")
	}

	portRangeStartNode, portRangeEndNode, err := parsePortRange(nodeLevelPortRange, "node-level")
	if err != nil {
		return nil, err
	}
	portRangeStartCluster, portRangeEndCluster, err := parsePortRange(clusterLevelPortRange, "cluster-level")
	if err != nil {
		return nil, err
	}

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
			namespace string
			podName   string
			port      int
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
	return utils.GetLeaderIP(s.Client)
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

// clusterPortOffset validates port against the cluster range and the bitmap
// size, returning the bit offset to use for masking. Rejecting here keeps the
// bitmap indexing below from panicking on malformed or stale annotations.
func (s *PortAllocator) clusterPortOffset(port int) (int, error) {
	if port < s.PortRangeStartCluster || port >= s.PortRangeEndCluster {
		return 0, fmt.Errorf("port %d out of cluster range [%d, %d)", port, s.PortRangeStartCluster, s.PortRangeEndCluster)
	}
	offset := port - s.PortRangeStartCluster
	if offset/64 >= len(s.BitmapCluster) {
		return 0, fmt.Errorf("port %d offset %d exceeds bitmap size %d", port, offset, len(s.BitmapCluster)*64)
	}
	return offset, nil
}

func (s *PortAllocator) ReleaseClusterLevelHostPort(namespace string, podName string, port int, immediateRelease bool) error {
	portOffset, err := s.clusterPortOffset(port)
	if err != nil {
		return fmt.Errorf("invalid port for pod %s/%s: %w", namespace, podName, err)
	}

	if immediateRelease {
		s.storeMutexCluster.Lock()
		defer s.storeMutexCluster.Unlock()
		s.BitmapCluster[portOffset/64] &^= 1 << (portOffset % 64)
		return nil
	} else {
		// put into queue, release until Pod not found
		s.clusterLevelPortReleaseQueue <- struct {
			namespace string
			podName   string
			port      int
		}{namespace, podName, port}
	}
	return nil
}

func (s *PortAllocator) releaseClusterPortUntilPodDeleted() {
	for item := range s.clusterLevelPortReleaseQueue {
		namespace := item.namespace
		podName := item.podName
		portOffset, offsetErr := s.clusterPortOffset(item.port)
		if offsetErr != nil {
			// Defensive: queue is fed by ReleaseClusterLevelHostPort which
			// already validates, but guard the bitmap indexing anyway so a
			// bad item can never crash the background worker.
			log.Log.Error(offsetErr, "drop release request for invalid port",
				"namespace", namespace, "pod", podName, "port", item.port)
			continue
		}

		_ = retry.OnError(RETRY_CONFIG, func(_ error) bool {
			return true
		}, func() error {
			pod := &v1.Pod{}
			err := s.Client.Get(s.ctx, client.ObjectKey{Namespace: namespace, Name: podName}, pod)
			if errors.IsNotFound(err) {
				s.storeMutexCluster.Lock()
				defer s.storeMutexCluster.Unlock()
				s.BitmapCluster[portOffset/64] &^= 1 << (portOffset % 64)
				return nil
			}
			return fmt.Errorf("pod still there, can not release port %s/%s", namespace, podName)
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
		if pod.Annotations == nil || pod.Annotations[constants.GenPortNumberAnnotation] == "" {
			continue
		}
		port, err := strconv.Atoi(pod.Annotations[constants.GenPortNumberAnnotation])
		if err != nil {
			continue
		}
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
