package portallocator

import (
	"context"
	"fmt"
	"math/bits"
	"strconv"
	"strings"
	"sync"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	v1 "k8s.io/api/core/v1"

	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

// offer API for host port allocation, range from user configured port range
// when started, fetch all allocated TENSOR_FUSION_WORKER_PORT

// - label: tensor-fusion.ai/component=worker
//   portStart: 40000
//   portEnd: 41000
//   byNode: true
// - label: tensor-fusion.ai/workload-type=lab
//   portStart: 41001
//   portEnd: 60000
//   byNode: false

// PodLabel => NodeName => HostPort
// Annotation: tensor-fusion.ai/host-port: assigned port
// Annotation: tensor-fusion.ai/host-port: assigned pod name
type PortAllocator struct {
	PortRangeStartNode int
	PortRangeEndNode   int

	PortRangeStartCluster int
	PortRangeEndCluster   int

	client client.Client

	bitmapPerNode map[string][]uint64
	bitmapCluster []uint64
}

var storeMutexNode sync.RWMutex
var storeMutexCluster sync.RWMutex

// Node level bit map for assigning ports to pods already scheduled to certain node
var BitMapPerNode map[string][]uint64 = make(map[string][]uint64)

// Cluster level bit map for assigning ports to pods which not scheduled anywhere
var BitMapClusterLevel []uint64 = make([]uint64, 0)

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
		PortRangeEndCluster:   int(portRangeEndCluster),
		client:                client,

		bitmapPerNode: make(map[string][]uint64),
		bitmapCluster: make([]uint64, (portRangeEndCluster-portRangeStartCluster)/64+1),
	}

	return allocator, nil
}

func (s *PortAllocator) SetupWithManager(ctx context.Context, mgr manager.Manager) {
	go func() {
		<-mgr.Elected()

		storeMutexNode.Lock()
		storeMutexCluster.Lock()
		defer storeMutexNode.Unlock()
		defer storeMutexCluster.Unlock()

		// 1. init bit map from existing pods labeled with tensor-fusion.ai/host-port=auto
		s.initBitMapForClusterLevelPortAssign(ctx)

		// 2. init bit map for existing vGPU workers
		s.initBitMapForNodeLevelPortAssign(ctx)
	}()
}

// GetHostPort always called by operator itself, thus no Leader-Follower inconsistency issue
func (s *PortAllocator) GetHostPort(nodeName string) (int, error) {
	if nodeName == "" {
		return 0, fmt.Errorf("node name cannot be empty when assign host port")
	}
	storeMutexNode.Lock()
	defer storeMutexNode.Unlock()

	if bitmap, ok := s.bitmapPerNode[nodeName]; !ok {
		return 0, fmt.Errorf("node %s not found in bitmap", nodeName)
	} else {
		// TODO: handle bitmap overflow
		return bits.TrailingZeros64(bitmap[0]), nil
	}

}

// TODO: implement
func (s *PortAllocator) ReleaseHostPort(nodeName string, port int) error {
	storeMutexNode.Lock()
	defer storeMutexNode.Unlock()
	return nil
}

// TODO: side effect, should forward the Pod mutating webhook request to the Leader if current node is not a leader
func (s *PortAllocator) GetClusterLevelHostPort(podName string) (int, error) {
	storeMutexCluster.Lock()
	defer storeMutexCluster.Unlock()
	return bits.TrailingZeros64(uint64(s.PortRangeStartCluster)), nil
}

func (s *PortAllocator) ReleaseClusterLevelHostPort(podName string, port int) error {
	storeMutexCluster.Lock()
	defer storeMutexCluster.Unlock()
	return nil
}

func (s *PortAllocator) initBitMapForClusterLevelPortAssign(ctx context.Context) {
	log := log.FromContext(ctx)
	podList := &v1.PodList{}
	err := s.client.List(ctx, podList, client.MatchingLabels{constants.GenHostPortLabel: constants.GenHostPortLabelValue})
	if err != nil {
		log.Error(err, "failed to list pods with port allocation label")
		return
	}
	usedPorts := []uint16{}
	for _, pod := range podList.Items {
		port, _ := strconv.Atoi(pod.Annotations[constants.GenPortNumberAnnotation])
		bitOffSet := port - s.PortRangeStartCluster
		usedPorts = append(usedPorts, uint16(bitOffSet))
	}

	for _, port := range usedPorts {
		s.bitmapCluster[port/64] |= 1 << (port % 64)
	}
}

func (s *PortAllocator) initBitMapForNodeLevelPortAssign(ctx context.Context) {
	log := log.FromContext(ctx)
	podList := &v1.PodList{}
	err := s.client.List(ctx, podList, client.MatchingLabels{constants.LabelComponent: constants.ComponentWorker})
	if err != nil {
		log.Error(err, "failed to list pods with port allocation label")
		return
	}

	size := (s.PortRangeEndNode-s.PortRangeStartNode)/64 + 1
	for _, pod := range podList.Items {
		port, _ := strconv.Atoi(pod.Annotations[constants.GenPortNumberAnnotation])
		bitOffSet := port - s.PortRangeStartNode
		if _, ok := s.bitmapPerNode[pod.Spec.NodeName]; !ok {
			s.bitmapPerNode[pod.Spec.NodeName] = make([]uint64, size)
		}
		s.bitmapPerNode[pod.Spec.NodeName][bitOffSet/64] |= 1 << (bitOffSet % 64)
	}

}
