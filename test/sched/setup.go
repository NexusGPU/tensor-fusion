package sched

import (
	"context"
	"fmt"
	"io"
	"strconv"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	gpuResourceFitPlugin "github.com/NexusGPU/tensor-fusion/internal/scheduler/gpuresources"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/events"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	testutil "sigs.k8s.io/scheduler-plugins/test/util"
)

// BenchmarkConfig holds benchmark configuration
type BenchmarkConfig struct {
	NumNodes  int
	NumGPUs   int
	NumPods   int
	BatchSize int
	PoolName  string
	Namespace string
	Timeout   time.Duration
}

// BenchmarkFixture holds pre-initialized benchmark data
type BenchmarkFixture struct {
	ctx       context.Context
	cancel    context.CancelFunc
	plugin    *gpuResourceFitPlugin.GPUFit
	nodes     []*v1.Node
	pods      []*v1.Pod
	allocator *gpuallocator.GpuAllocator
	client    client.Client
	fwk       framework.Framework
}

// NewBenchmarkFixture creates and initializes a benchmark fixture
func NewBenchmarkFixture(
	b *testing.B, config BenchmarkConfig, client client.Client, realAPIServer bool,
) *BenchmarkFixture {
	// Register scheme
	require.NoError(b, tfv1.AddToScheme(scheme.Scheme))

	if client == nil {
		// Create minimal runtime objects
		client = fake.NewClientBuilder().
			WithScheme(scheme.Scheme).
			WithRuntimeObjects(&tfv1.TensorFusionWorkload{
				ObjectMeta: metav1.ObjectMeta{Name: "benchmark-workload", Namespace: config.Namespace},
			}).
			WithStatusSubresource(&tfv1.GPU{}, &tfv1.GPUNode{}, &tfv1.TensorFusionWorkload{}, &v1.Pod{}, &v1.Node{}).
			Build()
	}
	ctx, cancel := context.WithTimeout(context.Background(), config.Timeout)

	// Suppress verbose logging during benchmarks
	suppressLogging()

	// Generate test data
	nodes := generateNodes(config.NumNodes)
	gpus, totalTflops, totalVRAM := generateGPUs(config.NumGPUs, nodes, config.PoolName)
	b.Logf("%d Nodes created, Total GPU Count: %d. Total TFLOPS: %f, Total VRAM: %f",
		len(nodes), len(gpus), totalTflops, totalVRAM)
	pods, neededTflops, neededVRAM := generatePods(config.NumPods, config.Namespace, config.PoolName)
	b.Logf("%d Pods created, Needed TFLOPS: %f, Needed VRAM: %f", len(pods), neededTflops, neededVRAM)

	// Batch create resources for better performance
	batchCreateResources(b, ctx, client, nodes, gpus, pods, realAPIServer)

	// Setup allocator
	allocator := setupAllocator(b, ctx, client)

	// Setup framework and plugin
	fwk, plugin := setupFrameworkAndPlugin(b, ctx, client, allocator, pods, nodes)

	return &BenchmarkFixture{
		ctx:       ctx,
		cancel:    cancel,
		plugin:    plugin,
		nodes:     nodes,
		pods:      pods,
		allocator: allocator,
		client:    client,
		fwk:       fwk,
	}
}

// Close cleans up the benchmark fixture
func (f *BenchmarkFixture) Close() {
	f.cancel()
}

// suppressLogging reduces log verbosity during benchmarks
func suppressLogging() {
	// Redirect klog output to discard all logs
	klog.SetOutput(io.Discard)
	klog.LogToStderr(false)
}

// generateNodes creates nodes with optimized allocation
func generateNodes(count int) []*v1.Node {
	nodes := make([]*v1.Node, count)
	for i := 0; i < count; i++ {
		nodes[i] = &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("node-%d", i),
				Labels: map[string]string{
					"test": "value",

					constants.KubernetesHostNameLabel: fmt.Sprintf("node-%d", i),
				},
				Annotations: map[string]string{
					"test": "value2",
				},
			},
			Status: v1.NodeStatus{
				Capacity: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("512"),
					v1.ResourceMemory: resource.MustParse("1024Gi"),
					"pods":            resource.MustParse("110"),
				},
				Allocatable: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("512"),
					v1.ResourceMemory: resource.MustParse("1024Gi"),
					"pods":            resource.MustParse("110"),
				},
				Phase: v1.NodeRunning,
				Conditions: []v1.NodeCondition{{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				}},
			},
		}
	}
	return nodes
}

// generateGPUs creates GPUs with better memory allocation
func generateGPUs(totalGPUs int, nodes []*v1.Node, poolName string) ([]*tfv1.GPU, float64, float64) {
	gpus := make([]*tfv1.GPU, totalGPUs)
	gpusPerNode := totalGPUs / len(nodes)

	// Pre-define GPU specs to avoid repeated allocations
	gpuSpecs := []struct{ tflops, vram string }{
		{"2250", "141Gi"}, // High-end
		{"989", "80Gi"},   // Mid-range
		{"450", "48Gi"},   // Entry-level
		{"312", "40Gi"},   // Budget
	}

	gpuIndex := 0
	totalTflops := 0.0
	totalVRAM := 0.0
	for nodeIdx, node := range nodes {
		nodeGPUCount := gpusPerNode
		if nodeIdx < totalGPUs%len(nodes) {
			nodeGPUCount++
		}

		for i := 0; i < nodeGPUCount && gpuIndex < totalGPUs; i++ {
			spec := gpuSpecs[gpuIndex%len(gpuSpecs)]

			gpus[gpuIndex] = &tfv1.GPU{
				ObjectMeta: metav1.ObjectMeta{
					Name: fmt.Sprintf("gpu-%d", gpuIndex),
					Labels: map[string]string{
						constants.GpuPoolKey:    poolName,
						constants.LabelKeyOwner: node.Name,
					},
				},
				Status: tfv1.GPUStatus{
					Phase:        tfv1.TensorFusionGPUPhaseRunning,
					NodeSelector: map[string]string{constants.KubernetesHostNameLabel: node.Name},
					UsedBy:       tfv1.UsedByTensorFusion,
					Capacity: &tfv1.Resource{
						Tflops: resource.MustParse(spec.tflops),
						Vram:   resource.MustParse(spec.vram),
					},
					Available: &tfv1.Resource{
						Tflops: resource.MustParse(spec.tflops),
						Vram:   resource.MustParse(spec.vram),
					},
				},
			}
			totalTflops += gpus[gpuIndex].Status.Capacity.Tflops.AsApproximateFloat64()
			totalVRAM += gpus[gpuIndex].Status.Capacity.Vram.AsApproximateFloat64()
			gpuIndex++
		}
	}
	return gpus[:gpuIndex], totalTflops, totalVRAM
}

// generatePods creates pods with optimized resource allocation
func generatePods(count int, namespace, poolName string) ([]*v1.Pod, float64, float64) {
	pods := make([]*v1.Pod, count)

	// Pre-define pod specs
	podSpecs := []struct {
		tflops   string
		vram     string
		gpuCount int
	}{
		{"20", "4Gi", 1},   // Small
		{"40", "8Gi", 1},   // Medium
		{"100", "16Gi", 1}, // Large
		{"30", "4Gi", 2},   // Multi-GPU
		{"200", "32Gi", 2}, // High-end
	}

	totalTflops := 0.0
	totalVRAM := 0.0
	for i := 0; i < count; i++ {
		spec := podSpecs[i%len(podSpecs)]

		pod := st.MakePod().
			Namespace(namespace).
			Name(fmt.Sprintf("benchmark-pod-%d", i)).
			UID(fmt.Sprintf("benchmark-pod-%d", i)).
			SchedulerName("tensor-fusion-scheduler").
			Res(map[v1.ResourceName]string{
				v1.ResourceCPU:    "10m",
				v1.ResourceMemory: "128Mi",
			}).
			NodeAffinityIn("test", []string{"value", "value2"}, st.NodeSelectorTypeMatchExpressions).
			Toleration("node.kubernetes.io/not-ready").
			ZeroTerminationGracePeriod().Obj()

		pod.Labels = map[string]string{
			constants.LabelComponent: constants.ComponentWorker,
			constants.WorkloadKey:    "benchmark-workload",
		}

		pod.Annotations = map[string]string{
			constants.GpuPoolKey:              poolName,
			constants.TFLOPSRequestAnnotation: spec.tflops,
			constants.VRAMRequestAnnotation:   spec.vram,
			constants.TFLOPSLimitAnnotation:   spec.tflops,
			constants.VRAMLimitAnnotation:     spec.vram,
			constants.GpuCountAnnotation:      strconv.Itoa(spec.gpuCount),
		}

		podTflops := resource.MustParse(spec.tflops)
		totalTflops += podTflops.AsApproximateFloat64() * float64(spec.gpuCount)
		podVRAM := resource.MustParse(spec.vram)
		totalVRAM += podVRAM.AsApproximateFloat64() * float64(spec.gpuCount)
		pods[i] = pod
	}

	return pods, totalTflops, totalVRAM
}

// Helper functions for setup
func batchCreateResources(
	b *testing.B, ctx context.Context, client client.Client,
	nodes []*v1.Node, gpus []*tfv1.GPU, pods []*v1.Pod, realAPIServer bool,
) {
	require.NoError(b, client.Create(ctx, &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "benchmark-ns"},
	}))

	timer := time.Now()
	b.Logf("Creating %d nodes", len(nodes))
	for _, node := range nodes {
		nodeCopy := node.DeepCopy()
		require.NoError(b, client.Create(ctx, nodeCopy))

		if realAPIServer {
			node.ResourceVersion = nodeCopy.ResourceVersion
			require.NoError(b, client.Status().Update(ctx, node))
		}
	}
	b.Logf("%d nodes created, duration: %v", len(nodes), time.Since(timer))

	// Create GPUs
	timer = time.Now()
	b.Logf("Creating %d GPUs", len(gpus))
	for _, gpu := range gpus {
		gpuCopy := gpu.DeepCopy()
		require.NoError(b, client.Create(ctx, gpuCopy))

		if realAPIServer {
			gpu.ResourceVersion = gpuCopy.ResourceVersion
			require.NoError(b, client.Status().Update(ctx, gpu))
		}
	}
	b.Logf("%d GPUs created, duration: %v", len(gpus), time.Since(timer))

	// Create pods
	timer = time.Now()
	b.Logf("Creating %d pods", len(pods))
	for _, pod := range pods {
		require.NoError(b, client.Create(ctx, pod))
	}
	b.Logf("%d pods created, duration: %v", len(pods), time.Since(timer))
}

func setupFrameworkAndPlugin(
	b *testing.B, ctx context.Context, client client.Client,
	allocator *gpuallocator.GpuAllocator, pods []*v1.Pod, nodes []*v1.Node,
) (framework.Framework, *gpuResourceFitPlugin.GPUFit) {
	// Register plugins including our GPU plugin
	registeredPlugins := []tf.RegisterPluginFunc{
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
	}

	// Create framework
	fwk, err := tf.NewFramework(ctx, registeredPlugins, "",
		frameworkruntime.WithPodNominator(testutil.NewPodNominator(nil)),
		frameworkruntime.WithSnapshotSharedLister(testutil.NewFakeSharedLister(pods, nodes)),
		frameworkruntime.WithEventRecorder(&events.FakeRecorder{}),
	)
	require.NoError(b, err)

	// Create plugin directly
	plugin := createPlugin(b, ctx, fwk, allocator, client)

	return fwk, plugin
}

func setupAllocator(
	b *testing.B, ctx context.Context, client client.Client,
) *gpuallocator.GpuAllocator {
	allocator := gpuallocator.NewGpuAllocator(ctx, client, time.Second)
	require.NoError(b, allocator.InitGPUAndQuotaStore())
	allocator.ReconcileAllocationState()
	allocator.SetAllocatorReady()
	return allocator
}

func createPlugin(
	b *testing.B, ctx context.Context, fwk framework.Framework,
	allocator *gpuallocator.GpuAllocator, client client.Client,
) *gpuResourceFitPlugin.GPUFit {
	pluginFactory := gpuResourceFitPlugin.NewWithDeps(allocator, client)
	pluginConfig := &runtime.Unknown{
		Raw: []byte(`{"maxWorkerPerNode": 256, "vramWeight": 0.7, "tflopsWeight": 0.3}`),
	}
	plugin, err := pluginFactory(ctx, pluginConfig, fwk)
	require.NoError(b, err)
	return plugin.(*gpuResourceFitPlugin.GPUFit)
}
