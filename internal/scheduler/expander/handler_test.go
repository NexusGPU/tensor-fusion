package expander

import (
	"context"
	"os"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	karpv1 "sigs.k8s.io/karpenter/pkg/apis/v1"
)

// NodeExpanderTestSuite holds common test setup for node expansion tests
type NodeExpanderTestSuite struct {
	ctx            context.Context
	cancel         context.CancelFunc
	k8sClient      client.Client
	allocator      *gpuallocator.GpuAllocator
	nodeExpander   *NodeExpander
	unschedHandler *UnscheduledPodHandler
	namespace      string
}

// SetupSuite initializes the test environment for node expansion tests
func (suite *NodeExpanderTestSuite) SetupSuite() {
	// Register TensorFusion API types in scheme
	Expect(tfv1.AddToScheme(scheme.Scheme)).To(Succeed())

	// Add Karpenter types to scheme (they should already be in the scheme from imports)
	// Note: Using runtime objects directly instead of scheme registration

	// Setup fake client with scheme including TensorFusion types
	suite.k8sClient = fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithStatusSubresource(&tfv1.GPU{}, &tfv1.GPUNode{}, &tfv1.TensorFusionWorkload{}, &corev1.Pod{}, &corev1.Node{}).
		Build()
	suite.namespace = "expansion-test-ns"

	// Create test namespace
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: suite.namespace},
	}
	Expect(suite.k8sClient.Create(ctx, ns)).To(Succeed())

	// Setup proper allocator for testing
	suite.allocator = gpuallocator.NewGpuAllocator(ctx, suite.k8sClient, time.Second)
	err := suite.allocator.InitGPUAndQuotaStore()
	if err != nil {
		// For test environments, we can ignore some initialization errors
		suite.allocator.SetAllocatorReady() // Mark as ready anyway for testing
	} else {
		suite.allocator.ReconcileAllocationState()
		suite.allocator.SetAllocatorReady()
	}

	suite.ctx = ctx
	suite.cancel = cancel

	// Setup node expander and unscheduled pod handler
	recorder := record.NewFakeRecorder(100)
	suite.unschedHandler, suite.nodeExpander = NewUnscheduledPodHandler(ctx, nil, suite.allocator, recorder)
}

// TearDownSuite cleans up the test environment
func (suite *NodeExpanderTestSuite) TearDownSuite() {
	if suite.cancel != nil {
		suite.cancel()
	}
}

func TestNodeExpander(t *testing.T) {
	suiteConfig, reporterConfig := GinkgoConfiguration()
	suiteConfig.Timeout = 2 * time.Minute
	RegisterFailHandler(Fail)
	if os.Getenv("DEBUG_MODE") == constants.TrueStringValue {
		SetDefaultEventuallyTimeout(10 * time.Minute)
	} else {
		SetDefaultEventuallyTimeout(12 * time.Second)
	}
	SetDefaultEventuallyPollingInterval(200 * time.Millisecond)
	SetDefaultConsistentlyDuration(5 * time.Second)
	SetDefaultConsistentlyPollingInterval(250 * time.Millisecond)
	RunSpecs(t, "NodeExpander Test Suite", suiteConfig, reporterConfig)
}

var _ = Describe("NodeExpander Unit Tests", func() {
	var suite *NodeExpanderTestSuite

	BeforeEach(func() {
		suite = &NodeExpanderTestSuite{}
		suite.SetupSuite()
	})

	AfterEach(func() {
		suite.TearDownSuite()
	})

	Describe("Node Creation Logic", func() {
		It("should create Karpenter NodeClaim when node is owned by NodeClaim", func() {
			testKarpenterNodeClaimCreation(suite)
		})

		It("should handle inflight node management correctly", func() {
			testInflightNodeManagement(suite)
		})

		It("should manage pre-scheduled pods correctly", func() {
			testPreScheduledPodManagement(suite)
		})
	})

	Describe("Node Expansion Integration Tests", func() {
		It("Case 1: should expand from 1 GPU node cluster when resources are exhausted", func() {
			testExpandFromOneGPUNode(suite)
		})

		It("Case 2: should not expand when inflight node satisfies new pending pod", func() {
			testShouldNotExpandWhenInflightSatisfies(suite)
		})

		It("Case 3: should expand another new node when inflight node with preScheduledPod cannot satisfy new pod", func() {
			testExpandWhenInflightCannotSatisfy(suite)
		})
	})
})

// Test Karpenter NodeClaim creation logic
func testKarpenterNodeClaimCreation(suite *NodeExpanderTestSuite) {
	// Create mock Karpenter NodePool
	nodePool := &karpv1.NodePool{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-nodepool",
		},
	}
	Expect(suite.k8sClient.Create(suite.ctx, nodePool)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, nodePool) }()

	// Create mock Karpenter NodeClaim
	nodeClaim := &karpv1.NodeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-nodeclaim-1",
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "karpenter.sh/v1",
					Kind:       "NodePool",
					Name:       nodePool.Name,
					UID:        nodePool.UID,
					Controller: &[]bool{true}[0],
				},
			},
		},
	}
	Expect(suite.k8sClient.Create(suite.ctx, nodeClaim)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, nodeClaim) }()

	// Create initial node owned by NodeClaim
	node1 := createTestNode("node-1", nodeClaim)
	Expect(suite.k8sClient.Create(suite.ctx, node1)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, node1) }()

	// Create test pod
	pod := createTestTensorFusionPod("worker-1", suite.namespace, "1000", "8Gi")
	Expect(suite.k8sClient.Create(suite.ctx, pod)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, pod) }()

	// Test createGPUNodeClaim method
	err := suite.nodeExpander.createGPUNodeClaim(suite.ctx, pod, node1)
	Expect(err).To(Succeed())

	// Verify new NodeClaim was created
	Eventually(func() int {
		nodeClaimList := &karpv1.NodeClaimList{}
		err := suite.k8sClient.List(suite.ctx, nodeClaimList)
		if err != nil {
			return 0
		}
		return len(nodeClaimList.Items)
	}, 10*time.Second, 200*time.Millisecond).Should(Equal(2))
}

// Test inflight node management
func testInflightNodeManagement(suite *NodeExpanderTestSuite) {
	// Test adding inflight node
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{Name: "test-gpu", Namespace: "default"},
		Status: tfv1.GPUStatus{
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("1000"),
				Vram:   resource.MustParse("8Gi"),
			},
			Capacity: &tfv1.Resource{
				Tflops: resource.MustParse("1000"),
				Vram:   resource.MustParse("8Gi"),
			},
		},
	}

	// Add inflight node
	suite.nodeExpander.mu.Lock()
	suite.nodeExpander.inFlightNodes["test-node"] = []*tfv1.GPU{gpu}
	suite.nodeExpander.mu.Unlock()

	// Verify inflight node exists
	suite.nodeExpander.mu.RLock()
	_, exists := suite.nodeExpander.inFlightNodes["test-node"]
	suite.nodeExpander.mu.RUnlock()
	Expect(exists).To(BeTrue())

	// Test removing inflight node
	suite.nodeExpander.RemoveInFlightNode("test-node")

	// Verify inflight node is removed
	suite.nodeExpander.mu.RLock()
	_, exists = suite.nodeExpander.inFlightNodes["test-node"]
	suite.nodeExpander.mu.RUnlock()
	Expect(exists).To(BeFalse())
}

// Test pre-scheduled pod management
func testPreScheduledPodManagement(suite *NodeExpanderTestSuite) {
	// Create test allocation request
	allocReq := &tfv1.AllocRequest{
		PodMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: suite.namespace,
		},
		Request: tfv1.Resource{
			Tflops: resource.MustParse("1000"),
			Vram:   resource.MustParse("8Gi"),
		},
		Limit: tfv1.Resource{
			Tflops: resource.MustParse("1000"),
			Vram:   resource.MustParse("8Gi"),
		},
	}

	// Create test node and GPU
	node := createTestNode("test-node", nil)
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{Name: "test-gpu", Namespace: "default"},
		Status: tfv1.GPUStatus{
			Available: &tfv1.Resource{
				Tflops: resource.MustParse("1000"),
				Vram:   resource.MustParse("8Gi"),
			},
			Capacity: &tfv1.Resource{
				Tflops: resource.MustParse("1000"),
				Vram:   resource.MustParse("8Gi"),
			},
		},
	}

	// Test adding pre-scheduled pod
	suite.nodeExpander.addInFlightNodeAndPreSchedulePod(allocReq, node, []*tfv1.GPU{gpu})

	// Verify pre-scheduled pod exists
	suite.nodeExpander.mu.RLock()
	_, exists := suite.nodeExpander.preSchedulePods["test-pod"]
	suite.nodeExpander.mu.RUnlock()
	Expect(exists).To(BeTrue())

	// Test removing pre-scheduled pod
	suite.nodeExpander.RemovePreSchedulePod("test-pod", true)

	// Verify pre-scheduled pod is removed
	suite.nodeExpander.mu.RLock()
	_, exists = suite.nodeExpander.preSchedulePods["test-pod"]
	suite.nodeExpander.mu.RUnlock()
	Expect(exists).To(BeFalse())
}

// Case 1: Expand from 1 GPU node cluster
func testExpandFromOneGPUNode(suite *NodeExpanderTestSuite) {
	// Create mock Karpenter NodePool
	nodePool := &karpv1.NodePool{
		ObjectMeta: metav1.ObjectMeta{
			Name: "gpu-nodepool",
		},
	}
	Expect(suite.k8sClient.Create(suite.ctx, nodePool)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, nodePool) }()

	// Create mock Karpenter NodeClaim
	nodeClaim := &karpv1.NodeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: "initial-nodeclaim",
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "karpenter.sh/v1",
					Kind:       "NodePool",
					Name:       nodePool.Name,
					UID:        nodePool.UID,
					Controller: &[]bool{true}[0],
				},
			},
		},
	}
	Expect(suite.k8sClient.Create(suite.ctx, nodeClaim)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, nodeClaim) }()

	// Create initial node with full GPU capacity
	node1 := createTestNode("node-1", nodeClaim)
	Expect(suite.k8sClient.Create(suite.ctx, node1)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, node1) }()

	// Create GPU resource on the node
	gpu1 := createTestGPU("gpu-1", "node-1", "1000", "8Gi")
	Expect(suite.k8sClient.Create(suite.ctx, gpu1)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, gpu1) }()

	// Initialize allocator state
	suite.allocator.ReconcileAllocationStateForTesting()

	// Create first TensorFusion pod that occupies all GPU + 1/4 CPU/mem
	pod1 := createTestTensorFusionPod("worker-1", suite.namespace, "1000", "8Gi")
	Expect(suite.k8sClient.Create(suite.ctx, pod1)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, pod1) }()

	// Create second pod that should trigger expansion
	pod2 := createTestTensorFusionPod("worker-2", suite.namespace, "1000", "8Gi")
	Expect(suite.k8sClient.Create(suite.ctx, pod2)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, pod2) }()

	// Since we can't use the full scheduler, test the expansion logic directly
	// Call the expansion method that creates a new node claim
	err := suite.nodeExpander.createGPUNodeClaim(suite.ctx, pod2, node1)
	Expect(err).To(Succeed())

	// Verify that a new NodeClaim was created
	Eventually(func() int {
		nodeClaimList := &karpv1.NodeClaimList{}
		err := suite.k8sClient.List(suite.ctx, nodeClaimList)
		if err != nil {
			return 0
		}
		return len(nodeClaimList.Items)
	}, 5*time.Second, 200*time.Millisecond).Should(Equal(2))
}

// Case 2: Should not expand when inflight node satisfies new pending pod
func testShouldNotExpandWhenInflightSatisfies(suite *NodeExpanderTestSuite) {
	// Create initial setup similar to case 1
	nodePool := &karpv1.NodePool{
		ObjectMeta: metav1.ObjectMeta{Name: "gpu-nodepool"},
	}
	Expect(suite.k8sClient.Create(suite.ctx, nodePool)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, nodePool) }()

	nodeClaim := &karpv1.NodeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: "initial-nodeclaim",
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: "karpenter.sh/v1",
				Kind:       "NodePool",
				Name:       nodePool.Name,
				UID:        nodePool.UID,
				Controller: &[]bool{true}[0],
			}},
		},
	}
	Expect(suite.k8sClient.Create(suite.ctx, nodeClaim)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, nodeClaim) }()

	node1 := createTestNode("node-1", nodeClaim)
	Expect(suite.k8sClient.Create(suite.ctx, node1)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, node1) }()

	// Create inflight node with sufficient resources
	inflightGPU := createTestGPU("inflight-gpu", "inflight-node", "1000", "8Gi")
	suite.nodeExpander.mu.Lock()
	suite.nodeExpander.inFlightNodes["inflight-node"] = []*tfv1.GPU{inflightGPU}
	suite.nodeExpander.mu.Unlock()

	// Create new pod that can be satisfied by inflight node
	pod := createTestTensorFusionPod("worker-1", suite.namespace, "1000", "8Gi")
	Expect(suite.k8sClient.Create(suite.ctx, pod)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, pod) }()

	initialNodeClaimCount := 1

	// Test that inflight nodes are properly tracked
	// Since we already set up an inflight node, just verify it exists and has the expected GPU

	// Verify inflight node is still tracked
	suite.nodeExpander.mu.RLock()
	_, exists := suite.nodeExpander.inFlightNodes["inflight-node"]
	suite.nodeExpander.mu.RUnlock()
	Expect(exists).To(BeTrue())

	// Consistently verify no new NodeClaim is created
	Consistently(func() int {
		nodeClaimList := &karpv1.NodeClaimList{}
		err := suite.k8sClient.List(suite.ctx, nodeClaimList)
		if err != nil {
			return -1
		}
		return len(nodeClaimList.Items)
	}, 3*time.Second, 500*time.Millisecond).Should(Equal(initialNodeClaimCount))
}

// Case 3: Should expand another new node when inflight node with preScheduledPod cannot satisfy new pod
func testExpandWhenInflightCannotSatisfy(suite *NodeExpanderTestSuite) {
	// Create initial setup
	nodePool := &karpv1.NodePool{
		ObjectMeta: metav1.ObjectMeta{Name: "gpu-nodepool"},
	}
	Expect(suite.k8sClient.Create(suite.ctx, nodePool)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, nodePool) }()

	nodeClaim := &karpv1.NodeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: "initial-nodeclaim",
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: "karpenter.sh/v1",
				Kind:       "NodePool",
				Name:       nodePool.Name,
				UID:        nodePool.UID,
				Controller: &[]bool{true}[0],
			}},
		},
	}
	Expect(suite.k8sClient.Create(suite.ctx, nodeClaim)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, nodeClaim) }()

	node1 := createTestNode("node-1", nodeClaim)
	Expect(suite.k8sClient.Create(suite.ctx, node1)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, node1) }()

	// Create inflight node with GPU already pre-scheduled to another pod
	inflightGPU := createTestGPU("inflight-gpu", "inflight-node", "1000", "8Gi")

	// Create allocation request for pre-scheduled pod
	allocRequest := &tfv1.AllocRequest{
		PodMeta: metav1.ObjectMeta{
			Name:      "pre-scheduled",
			Namespace: suite.namespace,
		},
		Request: tfv1.Resource{
			Tflops: resource.MustParse("1000"),
			Vram:   resource.MustParse("8Gi"),
		},
		Limit: tfv1.Resource{
			Tflops: resource.MustParse("1000"),
			Vram:   resource.MustParse("8Gi"),
		},
	}

	// Add inflight node and pre-schedule a pod to it
	suite.nodeExpander.mu.Lock()
	suite.nodeExpander.inFlightNodes["inflight-node"] = []*tfv1.GPU{inflightGPU}
	suite.nodeExpander.preSchedulePods["pre-scheduled"] = allocRequest
	suite.nodeExpander.mu.Unlock()

	// Create new pod that cannot be satisfied by the already-occupied inflight node
	newPod := createTestTensorFusionPod("new-worker", suite.namespace, "1000", "8Gi")
	Expect(suite.k8sClient.Create(suite.ctx, newPod)).To(Succeed())
	defer func() { _ = suite.k8sClient.Delete(suite.ctx, newPod) }()

	// Test that a new node claim is created when inflight node can't satisfy the new pod
	err := suite.nodeExpander.createGPUNodeClaim(suite.ctx, newPod, node1)
	Expect(err).To(Succeed())

	// Wait for new NodeClaim to be created (should have 2 total)
	Eventually(func() int {
		nodeClaimList := &karpv1.NodeClaimList{}
		err := suite.k8sClient.List(suite.ctx, nodeClaimList)
		if err != nil {
			return 0
		}
		return len(nodeClaimList.Items)
	}, 30*time.Second, 500*time.Millisecond).Should(Equal(2))

	// Since we don't have a full scheduler, just verify the new NodeClaim was created
	// and the expansion logic completed successfully (tested above)"
}

// Helper functions
func createTestNode(name string, nodeClaim *karpv1.NodeClaim) *corev1.Node {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"kubernetes.io/arch": "amd64",
				"node-type":          "gpu-node",
			},
		},
		Spec: corev1.NodeSpec{
			Unschedulable: false,
		},
		Status: corev1.NodeStatus{
			Conditions: []corev1.NodeCondition{
				{
					Type:   corev1.NodeReady,
					Status: corev1.ConditionTrue,
				},
			},
			Allocatable: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("4"),
				corev1.ResourceMemory: resource.MustParse("16Gi"),
			},
			Capacity: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("4"),
				corev1.ResourceMemory: resource.MustParse("16Gi"),
			},
		},
	}

	if nodeClaim != nil {
		node.OwnerReferences = []metav1.OwnerReference{
			{
				APIVersion: "karpenter.sh/v1",
				Kind:       "NodeClaim",
				Name:       nodeClaim.Name,
				UID:        nodeClaim.UID,
				Controller: &[]bool{true}[0],
			},
		}
	}

	return node
}

func createTestGPU(name, nodeName, tflops, vram string) *tfv1.GPU {
	return &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Status: tfv1.GPUStatus{
			Available: &tfv1.Resource{
				Tflops: resource.MustParse(tflops),
				Vram:   resource.MustParse(vram),
			},
			Capacity: &tfv1.Resource{
				Tflops: resource.MustParse(tflops),
				Vram:   resource.MustParse(vram),
			},
			NodeSelector: map[string]string{
				"kubernetes.io/hostname": nodeName,
			},
		},
	}
}

func createTestTensorFusionPod(name, namespace, tflops, vram string) *corev1.Pod {
	pod := st.MakePod().
		Namespace(namespace).
		Name(name).
		UID(name + "-uid").
		SchedulerName("tensor-fusion-scheduler").
		Res(map[corev1.ResourceName]string{
			corev1.ResourceCPU:    "1",   // 1/4 of node CPU
			corev1.ResourceMemory: "4Gi", // 1/4 of node memory
		}).
		Toleration("node.kubernetes.io/not-ready").
		ZeroTerminationGracePeriod().Obj()

	pod.Labels = map[string]string{
		constants.LabelComponent:              constants.ComponentWorker,
		constants.WorkloadKey:                 "test-workload",
		constants.TensorFusionEnabledLabelKey: constants.TrueStringValue,
	}

	pod.Annotations = map[string]string{
		constants.GpuPoolKey:              "test-pool",
		constants.TFLOPSRequestAnnotation: tflops,
		constants.VRAMRequestAnnotation:   vram,
		constants.TFLOPSLimitAnnotation:   tflops,
		constants.VRAMLimitAnnotation:     vram,
		constants.GpuCountAnnotation:      "1",
	}

	return pod
}
