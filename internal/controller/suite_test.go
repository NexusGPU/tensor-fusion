/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/pricing"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	// +kubebuilder:scaffold:imports
)

// These tests use Ginkgo (BDD-style Go testing framework). Refer to
// http://onsi.github.io/ginkgo/ to learn more about Ginkgo.

var cfg *rest.Config
var k8sClient client.Client
var testEnv *envtest.Environment
var ctx context.Context
var cancel context.CancelFunc
var allocator *gpuallocator.GpuAllocator
var metricsRecorder *metrics.MetricsRecorder

func TestControllers(t *testing.T) {
	RegisterFailHandler(Fail)

	if os.Getenv("DEBUG_MODE") == constants.TrueStringValue {
		SetDefaultEventuallyTimeout(10 * time.Minute)
	} else {
		SetDefaultEventuallyTimeout(12 * time.Second)
	}
	SetDefaultEventuallyPollingInterval(200 * time.Millisecond)
	SetDefaultConsistentlyDuration(5 * time.Second)
	SetDefaultConsistentlyPollingInterval(250 * time.Millisecond)
	RunSpecs(t, "Controller Suite")
}

var _ = BeforeSuite(func() {
	// Expect(os.Setenv("USE_EXISTING_CLUSTER", "true")).Should(Succeed())
	logf.SetLogger(zap.New(zap.WriteTo(GinkgoWriter), zap.UseDevMode(true)))

	ctx, cancel = context.WithCancel(context.TODO())

	By("bootstrapping test environment")
	testEnv = &envtest.Environment{
		CRDDirectoryPaths: []string{
			filepath.Join("..", "..", "config", "crd", "bases"),
			filepath.Join("..", "..", "test", "crd"),
		},
		ErrorIfCRDPathMissing: true,

		// The BinaryAssetsDirectory is only required if you want to run the tests directly
		// without call the makefile target test. If not informed it will look for the
		// default path defined in controller-runtime which is /usr/local/kubebuilder/.
		// Note that you must have the required binaries setup under the bin directory to perform
		// the tests directly. When we run make test it will be setup and used automatically.
		BinaryAssetsDirectory: filepath.Join("..", "..", "bin", "k8s",
			fmt.Sprintf("1.31.0-%s-%s", runtime.GOOS, runtime.GOARCH)),
	}

	var err error
	// cfg is defined in this file globally.
	cfg, err = testEnv.Start()

	config.SetGlobalConfig(config.MockGlobalConfig())

	// set tflops map for pricing
	pricing.SetTflopsMapAndInitGPUPricingInfo(ctx, config.MockGpuInfo())

	Expect(err).NotTo(HaveOccurred())
	Expect(cfg).NotTo(BeNil())

	err = tfv1.AddToScheme(scheme.Scheme)
	Expect(err).NotTo(HaveOccurred())

	err = corev1.AddToScheme(scheme.Scheme)
	Expect(err).NotTo(HaveOccurred())

	err = tfv1.AddToScheme(scheme.Scheme)
	Expect(err).NotTo(HaveOccurred())

	// +kubebuilder:scaffold:scheme

	k8sClient, err = client.New(cfg, client.Options{Scheme: scheme.Scheme})
	Expect(err).NotTo(HaveOccurred())
	Expect(k8sClient).NotTo(BeNil())

	Expect(k8sClient.Create(ctx, &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: utils.CurrentNamespace(),
		},
	})).NotTo(HaveOccurred())

	mgr, err := ctrl.NewManager(cfg, ctrl.Options{
		Scheme: scheme.Scheme,
		Metrics: metricsserver.Options{
			BindAddress: "0",
		},
	})

	Expect(err).ToNot(HaveOccurred())

	metricsRecorder = &metrics.MetricsRecorder{
		MetricsOutputPath: "./metrics.log",
		HourlyUnitPriceMap: map[string]float64{
			"A100": 10,
		},
		WorkerUnitPriceMap: make(map[string]map[string]metrics.RawBillingPricing),
	}

	allocator = gpuallocator.NewGpuAllocator(ctx, mgr.GetClient(), 150*time.Millisecond)
	err = allocator.SetupWithManager(ctx, mgr)
	Expect(err).ToNot(HaveOccurred())

	portAllocator, err := portallocator.NewPortAllocator(ctx, mgr.GetClient(), "40000-42000", "42001-60000")
	if err != nil {
		Expect(err).ToNot(HaveOccurred())
	}
	_ = portAllocator.SetupWithManager(ctx, mgr)

	err = (&TensorFusionClusterReconciler{
		Client:          mgr.GetClient(),
		Scheme:          mgr.GetScheme(),
		Recorder:        mgr.GetEventRecorderFor("TensorFusionCluster"),
		MetricsRecorder: metricsRecorder,
	}).SetupWithManager(mgr, false)
	Expect(err).ToNot(HaveOccurred())

	err = (&GPUPoolReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("GPUPool"),
	}).SetupWithManager(mgr, false)
	Expect(err).ToNot(HaveOccurred())

	err = (&GPUNodeReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("GPUNode"),
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	SetTestModeCompactionPeriod()
	err = (&GPUPoolCompactionReconciler{
		Client:    mgr.GetClient(),
		Scheme:    mgr.GetScheme(),
		Recorder:  mgr.GetEventRecorderFor("GPUPoolCompaction"),
		Allocator: allocator,
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	err = (&GPUNodeClassReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	err = (&SchedulingConfigTemplateReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	err = (&PodReconciler{
		Client:        mgr.GetClient(),
		Scheme:        mgr.GetScheme(),
		Allocator:     allocator,
		PortAllocator: portAllocator,
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	err = (&NodeReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("Node"),
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	err = (&WorkloadProfileReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	err = (&TensorFusionConnectionReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("TensorFusionConnection"),
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	err = (&GPUReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(ctx, mgr)
	Expect(err).ToNot(HaveOccurred())

	err = (&TensorFusionWorkloadReconciler{
		Client:        mgr.GetClient(),
		Scheme:        mgr.GetScheme(),
		Recorder:      mgr.GetEventRecorderFor("TensorFusionWorkload"),
		PortAllocator: portAllocator,
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	err = (&FakeNodeClaimReconciler{
		client: mgr.GetClient(),
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	err = (&GPUNodeClaimReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("GPUNodeClaim"),
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	go func() {
		defer GinkgoRecover()
		err = mgr.Start(ctx)
		Expect(err).ToNot(HaveOccurred(), "failed to run manager")
	}()

})

var _ = AfterSuite(func() {
	By("tearing down the test environment")
	allocator.Stop()
	cancel()
	err := testEnv.Stop()
	Expect(err).NotTo(HaveOccurred())
	// Expect(os.Unsetenv("USE_EXISTING_CLUSTER")).To(Succeed())
})

type TensorFusionEnv struct {
	clusterKey        client.ObjectKey
	poolCount         int
	poolNodeMap       map[int]map[int]int
	provisionerConfig *tfv1.ComputingVendorConfig
}

func (c *TensorFusionEnv) GetCluster() *tfv1.TensorFusionCluster {
	GinkgoHelper()
	tfc := &tfv1.TensorFusionCluster{}
	Expect(k8sClient.Get(ctx, c.clusterKey, tfc)).Should(Succeed())
	return tfc
}

func (c *TensorFusionEnv) UpdateCluster(tfc *tfv1.TensorFusionCluster) {
	GinkgoHelper()
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		latest := &tfv1.TensorFusionCluster{}
		if err := k8sClient.Get(ctx, client.ObjectKeyFromObject(tfc), latest); err != nil {
			return err
		}
		latest.Spec = tfc.Spec
		return k8sClient.Update(ctx, latest)
	})
	Expect(err).Should(Succeed())
}

func (c *TensorFusionEnv) Cleanup() {
	GinkgoHelper()
	ProvisioningToggle = false
	for poolIndex, nodeGpuMap := range c.poolNodeMap {
		for nodeIndex := range nodeGpuMap {
			c.DeleteGPUNode(poolIndex, nodeIndex)
		}
	}

	tfc := c.GetCluster()
	tfcCopy := tfc.DeepCopy()
	tfcCopy.Spec.GPUPools = []tfv1.GPUPoolDefinition{}
	c.UpdateCluster(tfcCopy)

	for poolIndex := range c.poolNodeMap {
		Eventually(func(g Gomega) {
			pool := &tfv1.GPUPool{}
			g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: c.getPoolName(poolIndex)}, pool)).Should(HaveOccurred())
		}).Should(Succeed())
		delete(c.poolNodeMap, poolIndex)
		c.poolCount--
	}

	Expect(k8sClient.Delete(ctx, tfc)).Should(Succeed())
	Eventually(func(g Gomega) {
		err := k8sClient.Get(ctx, c.clusterKey, tfc)
		g.Expect(err).Should(HaveOccurred())
	}).Should(Succeed())
}

func (c *TensorFusionEnv) GetGPUPoolList() *tfv1.GPUPoolList {
	GinkgoHelper()
	poolList := &tfv1.GPUPoolList{}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.List(ctx, poolList, client.MatchingLabels(map[string]string{
			constants.LabelKeyOwner: c.clusterKey.Name,
		}))).Should(Succeed())
		g.Expect(poolList.Items).Should(HaveLen(c.poolCount))
	}).Should(Succeed())
	return poolList
}

func (c *TensorFusionEnv) GetGPUPool(poolIndex int) *tfv1.GPUPool {
	GinkgoHelper()
	pool := &tfv1.GPUPool{}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: c.getPoolName(poolIndex)}, pool)).Should(Succeed())
	}).Should(Succeed())
	return pool
}

func (c *TensorFusionEnv) GetGPUNodeList(poolIndex int) *tfv1.GPUNodeList {
	GinkgoHelper()
	nodeList := &tfv1.GPUNodeList{}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.List(ctx, nodeList, client.MatchingLabels(map[string]string{
			fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, c.getPoolName(poolIndex)): "true",
		}))).Should(Succeed())
		g.Expect(nodeList.Items).Should(HaveLen(len(c.poolNodeMap[poolIndex])))
	}).Should(Succeed())
	return nodeList
}

func (c *TensorFusionEnv) GetGPUNode(poolIndex int, nodeIndex int) *tfv1.GPUNode {
	GinkgoHelper()
	node := &tfv1.GPUNode{}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: c.getNodeName(poolIndex, nodeIndex)}, node)).Should(Succeed())
	}).Should(Succeed())
	return node
}

func (c *TensorFusionEnv) DeleteGPUNode(poolIndex int, nodeIndex int) {
	GinkgoHelper()
	c.DeleteNodeGpuList(poolIndex, nodeIndex)
	node := c.GetGPUNode(poolIndex, nodeIndex)
	Expect(k8sClient.Delete(ctx, node)).Should(Succeed())
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: c.getNodeName(poolIndex, nodeIndex)}, node)).Should(HaveOccurred())
	}).Should(Succeed())
	delete(c.poolNodeMap[poolIndex], nodeIndex)
}

func (c *TensorFusionEnv) GetNodeGpuList(poolIndex int, nodeIndex int) *tfv1.GPUList {
	GinkgoHelper()
	gpuList := &tfv1.GPUList{}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.List(ctx, gpuList, client.MatchingLabels(map[string]string{
			constants.LabelKeyOwner: c.getNodeName(poolIndex, nodeIndex),
		}))).Should(Succeed())
		g.Expect(gpuList.Items).Should(HaveLen(c.poolNodeMap[poolIndex][nodeIndex]))
	}).Should(Succeed())
	return gpuList
}

func (c *TensorFusionEnv) DeleteNodeGpuList(poolIndex int, nodeIndex int) {
	GinkgoHelper()
	Expect(k8sClient.DeleteAllOf(ctx, &tfv1.GPU{},
		client.MatchingLabels{constants.LabelKeyOwner: c.getNodeName(poolIndex, nodeIndex)},
	)).Should(Succeed())
}

func (c *TensorFusionEnv) GetPoolGpuList(poolIndex int) *tfv1.GPUList {
	GinkgoHelper()
	gpuList := &tfv1.GPUList{}
	poolGpuCount := 0
	for _, gpuCount := range c.poolNodeMap[poolIndex] {
		poolGpuCount += gpuCount
	}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.List(ctx, gpuList, client.MatchingLabels(map[string]string{
			constants.GpuPoolKey: c.getPoolName(poolIndex),
		}))).Should(Succeed())
		g.Expect(gpuList.Items).Should(HaveLen(poolGpuCount))
	}).Should(Succeed())
	return gpuList
}

// https://book.kubebuilder.io/reference/envtest#testing-considerations
// Unless you’re using an existing cluster, keep in mind that no built-in controllers are running in the test context.
// So the checkStatusAndUpdateVirtualCapacity in gpunode_controller.go checking pod status always pending and the gpunode status can't change to running
func (c *TensorFusionEnv) UpdateHypervisorStatus(checkNodeNum bool) {
	GinkgoHelper()
	if os.Getenv("USE_EXISTING_CLUSTER") != constants.TrueStringValue {
		for poolIndex := range c.poolNodeMap {
			podList := &corev1.PodList{}
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(utils.CurrentNamespace()),
					client.MatchingLabels(map[string]string{
						fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, c.getPoolName(poolIndex)): "true",
					}),
				)).Should(Succeed())
				if checkNodeNum {
					g.Expect(podList.Items).Should(HaveLen(len(c.poolNodeMap[poolIndex])))
				}
			}).Should(Succeed())
			for _, pod := range podList.Items {
				if pod.Status.Phase == corev1.PodRunning {
					continue
				}
				pod.Status.Phase = corev1.PodRunning
				pod.Status.Conditions = append(pod.Status.Conditions, corev1.PodCondition{Type: corev1.PodReady, Status: corev1.ConditionTrue})
				Expect(k8sClient.Status().Update(ctx, &pod)).Should(Succeed())
			}
		}
	}
}

func (c *TensorFusionEnv) AddMockGPU4ProvisionedNodes(gpuNodeClaimList *tfv1.GPUNodeClaimList, gpuNodes *tfv1.GPUNodeList) {
	GinkgoHelper()
	claimToGPUNodeMap := make(map[string]*tfv1.GPUNode)
	for _, gpuNode := range gpuNodes.Items {
		claimToGPUNodeMap[gpuNode.Labels[constants.ProvisionerLabelKey]] = &gpuNode
	}
	for _, gpuNodeClaim := range gpuNodeClaimList.Items {
		if gpuNodeClaim.Status.Phase == tfv1.GPUNodeClaimBound {
			continue
		}
		gpuNode := claimToGPUNodeMap[gpuNodeClaim.Name]
		if gpuNode == nil {
			continue
		}
		gpu := &tfv1.GPU{
			ObjectMeta: metav1.ObjectMeta{
				Name: "gpu-" + gpuNodeClaim.Name,
				Labels: map[string]string{
					constants.LabelKeyOwner: gpuNode.Name,
					constants.GpuPoolKey:    gpuNodeClaim.Labels[constants.LabelKeyOwner],
				},
			},
		}
		_ = controllerutil.SetControllerReference(gpuNode, gpu, scheme.Scheme)
		err := k8sClient.Get(ctx, client.ObjectKey{Name: gpu.Name}, &tfv1.GPU{})
		if errors.IsNotFound(err) {
			Expect(k8sClient.Create(ctx, gpu)).Should(Succeed())

			err = retry.RetryOnConflict(retry.DefaultBackoff, func() error {
				latest := &tfv1.GPU{}
				if err := k8sClient.Get(ctx, client.ObjectKeyFromObject(gpu), latest); err != nil {
					return err
				}
				latest.Status = tfv1.GPUStatus{
					GPUModel: "mocked",
					UsedBy:   tfv1.UsedByTensorFusion,
					Capacity: &tfv1.Resource{
						Tflops: gpuNodeClaim.Spec.TFlopsOffered,
						Vram:   gpuNodeClaim.Spec.VRAMOffered,
					},
					Available: &tfv1.Resource{
						Tflops: gpuNodeClaim.Spec.TFlopsOffered,
						Vram:   gpuNodeClaim.Spec.VRAMOffered,
					},
					Phase: tfv1.TensorFusionGPUPhaseRunning,
					NodeSelector: map[string]string{
						constants.KubernetesHostNameLabel: gpuNode.Name,
					},
				}
				if err := k8sClient.Status().Update(ctx, latest); err != nil {
					return err
				}

				// update GPUNode status to trigger node level reconcile, simulate node discovery job
				if gpuNode.Status.Phase == "" || gpuNode.Status.TotalGPUs == 0 {
					gpuNode.Status = tfv1.GPUNodeStatus{
						Phase:       tfv1.TensorFusionGPUNodePhasePending,
						TotalGPUs:   1,
						ManagedGPUs: 1,
						TotalTFlops: gpuNodeClaim.Spec.TFlopsOffered,
						TotalVRAM:   gpuNodeClaim.Spec.VRAMOffered,
					}
					if err := k8sClient.Status().Update(ctx, gpuNode); err != nil {
						return err
					}
				}
				return nil
			})
			Expect(err).Should(Succeed())
		}
	}
}

func (c *TensorFusionEnv) getPoolName(poolIndex int) string {
	return fmt.Sprintf("%s-pool-%d", c.clusterKey.Name, poolIndex)
}

func (c *TensorFusionEnv) getNodeName(poolIndex int, nodeIndex int) string {
	return fmt.Sprintf("%s-pool-%d-node-%d", c.clusterKey.Name, poolIndex, nodeIndex)
}

func (c *TensorFusionEnv) getGPUName(poolIndex int, nodeIndex int, gpuIndex int) string {
	return fmt.Sprintf("%s-pool-%d-node-%d-gpu-%d", c.clusterKey.Name, poolIndex, nodeIndex, gpuIndex)
}

func (c *TensorFusionEnv) GetConfig() *rest.Config {
	return cfg
}

type TensorFusionEnvBuilder struct {
	*TensorFusionEnv
}

func NewTensorFusionEnvBuilder() *TensorFusionEnvBuilder {
	return &TensorFusionEnvBuilder{
		&TensorFusionEnv{
			poolCount:   0,
			clusterKey:  client.ObjectKey{},
			poolNodeMap: map[int]map[int]int{},
		},
	}
}

func (b *TensorFusionEnvBuilder) AddPoolWithNodeCount(nodeCount int) *TensorFusionEnvBuilder {
	nodeGpuMap := make(map[int]int, nodeCount)
	for i := range nodeCount {
		nodeGpuMap[i] = 0
	}
	b.poolNodeMap[b.poolCount] = nodeGpuMap
	b.poolCount++
	return b
}

func (b *TensorFusionEnvBuilder) SetGpuCountPerNode(gpuCount int) *TensorFusionEnvBuilder {
	poolIndex := b.poolCount - 1
	for nodeIndex := range b.poolNodeMap[poolIndex] {
		b.poolNodeMap[poolIndex][nodeIndex] = gpuCount
	}
	return b
}

func (b *TensorFusionEnvBuilder) SetGpuCountForNode(nodeIndex int, gpuCount int) *TensorFusionEnvBuilder {
	poolIndex := b.poolCount - 1
	b.poolNodeMap[poolIndex][nodeIndex] = gpuCount
	return b
}

func (b *TensorFusionEnvBuilder) SetProvisioningMode(provisionerConfig *tfv1.ComputingVendorConfig) *TensorFusionEnvBuilder {
	b.provisionerConfig = provisionerConfig
	return b
}

var testEnvId int = 0

func (b *TensorFusionEnvBuilder) Build() *TensorFusionEnv {
	GinkgoHelper()
	ProvisioningToggle = true
	GenerateKarpenterEC2NodeClass()

	b.clusterKey = client.ObjectKey{
		Name:      fmt.Sprintf("cluster-%d", testEnvId),
		Namespace: "default",
	}
	testEnvId++

	// generate cluster
	tfc := &tfv1.TensorFusionCluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      b.clusterKey.Name,
			Namespace: b.clusterKey.Namespace,
		},
		Spec: tfv1.TensorFusionClusterSpec{
			GPUPools: []tfv1.GPUPoolDefinition{
				{
					Name:         fmt.Sprintf("pool-%d", b.poolCount),
					SpecTemplate: *config.MockGPUPoolSpec,
				},
			},
		},
	}

	// construct pools
	gpuPools := make([]tfv1.GPUPoolDefinition, b.poolCount)
	for i := range b.poolCount {
		poolSpec := config.MockGPUPoolSpec.DeepCopy()
		poolSpec.NodeManagerConfig.NodeSelector.NodeSelectorTerms[0].MatchExpressions[0].Key =
			fmt.Sprintf("%s-label-%d", tfc.Name, i)
		gpuPools[i] = tfv1.GPUPoolDefinition{
			Name:         fmt.Sprintf("pool-%d", i),
			SpecTemplate: *poolSpec,
		}

		if b.provisionerConfig != nil {
			if b.provisionerConfig.Type == tfv1.ComputingVendorKarpenter {
				gpuPools[i].SpecTemplate.NodeManagerConfig.ProvisioningMode = tfv1.ProvisioningModeKarpenter
				gpuPools[i].SpecTemplate.NodeManagerConfig.NodeProvisioner = &tfv1.NodeProvisioner{
					KarpenterNodeClassRef: &tfv1.GroupKindName{
						Group:   "karpenter.k8s.aws",
						Kind:    "EC2NodeClass",
						Name:    "test-ec2-node-class",
						Version: "v1",
					},
					GPURequirements: []tfv1.Requirement{
						{
							Key:      tfv1.NodeRequirementKeyInstanceType,
							Operator: corev1.NodeSelectorOpIn,
							Values:   []string{"g6.xlarge"},
						},
					},
					GPULabels: map[string]string{
						"mock-label":                            "true",
						fmt.Sprintf("%s-label-%d", tfc.Name, i): "true",
					},
					GPUAnnotation: map[string]string{
						"mock-annotation": "true",
					},
				}
			} else {
				gpuPools[i].SpecTemplate.NodeManagerConfig.ProvisioningMode = tfv1.ProvisioningModeProvisioned
				gpuPools[i].SpecTemplate.NodeManagerConfig.NodeProvisioner = &tfv1.NodeProvisioner{
					NodeClass: "test-node-class",
					GPURequirements: []tfv1.Requirement{
						{
							Key:      tfv1.NodeRequirementKeyInstanceType,
							Operator: corev1.NodeSelectorOpIn,
							Values:   []string{"g6.xlarge"},
						},
					},
					GPULabels: map[string]string{
						"mock-label": "true",
					},
					GPUAnnotation: map[string]string{
						"mock-annotation": "true",
					},
				}
			}
		}
	}

	// set provisioner config
	if b.provisionerConfig != nil {
		tfc.Spec.ComputingVendor = b.provisionerConfig
	}

	tfc.Spec.GPUPools = gpuPools
	Expect(k8sClient.Create(ctx, tfc)).To(Succeed())

	// wait for pools are created
	Eventually(func(g Gomega) {
		gpuPoolList := &tfv1.GPUPoolList{}
		g.Expect(k8sClient.List(ctx, gpuPoolList, client.MatchingLabels(map[string]string{
			constants.LabelKeyOwner: tfc.Name,
		}))).Should(Succeed())
		g.Expect(gpuPoolList.Items).Should(HaveLen(b.poolCount))
	}).Should(Succeed())

	// generate nodes
	selectors := strings.Split(constants.InitialGPUNodeSelector, "=")
	for poolIndex := range b.poolCount {
		nodeCount := len(b.poolNodeMap[poolIndex])
		for nodeIndex := range nodeCount {
			coreNode := &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: b.getNodeName(poolIndex, nodeIndex),
					Labels: map[string]string{
						selectors[0]: selectors[1],
						fmt.Sprintf("%s-label-%d", tfc.Name, poolIndex): "true",
					},
				},
			}
			Expect(k8sClient.Create(ctx, coreNode)).To(Succeed())

			// generate gpus for gpunode
			gpuNode := b.GetGPUNode(poolIndex, nodeIndex)
			gpuCount := b.poolNodeMap[poolIndex][nodeIndex]
			if gpuCount > 0 {
				for gpuIndex := range gpuCount {
					key := client.ObjectKey{
						Name: b.getGPUName(poolIndex, nodeIndex, gpuIndex),
					}
					gpu := &tfv1.GPU{
						ObjectMeta: metav1.ObjectMeta{
							Name: key.Name,
							Labels: map[string]string{
								constants.LabelKeyOwner: gpuNode.Name,
								constants.GpuPoolKey:    b.getPoolName(poolIndex),
							},
						},
					}
					Expect(controllerutil.SetControllerReference(gpuNode, gpu, scheme.Scheme)).To(Succeed())
					Expect(k8sClient.Create(ctx, gpu)).To(Succeed())
					patch := client.MergeFrom(gpu.DeepCopy())
					gpu.Status = tfv1.GPUStatus{
						Phase:    tfv1.TensorFusionGPUPhaseRunning,
						UUID:     key.Name,
						GPUModel: "mock",
						NodeSelector: map[string]string{
							"kubernetes.io/hostname": b.getNodeName(poolIndex, nodeIndex),
						},
						Capacity: &tfv1.Resource{
							Tflops: resource.MustParse("2000"),
							Vram:   resource.MustParse("2000Gi"),
						},
						Available: &tfv1.Resource{
							Tflops: resource.MustParse("2000"),
							Vram:   resource.MustParse("2000Gi"),
						},
						Message: "mock message",
					}
					Expect(k8sClient.Status().Patch(ctx, gpu, patch)).To(Succeed())
				}

				Eventually(func(g Gomega) {
					gpuNode := b.GetGPUNode(poolIndex, nodeIndex)
					gpuNode.Status.Phase = tfv1.TensorFusionGPUNodePhasePending
					gpuNode.Status.TotalGPUs = int32(gpuCount)
					gpuNode.Status.ManagedGPUs = int32(gpuCount)
					gpuNode.Status.TotalTFlops = resource.MustParse(fmt.Sprintf("%d", 2000*gpuCount))
					gpuNode.Status.TotalVRAM = resource.MustParse(fmt.Sprintf("%dGi", 2000*gpuCount))
					gpuNode.Status.AvailableTFlops = gpuNode.Status.TotalTFlops
					gpuNode.Status.AvailableVRAM = gpuNode.Status.TotalVRAM
					Expect(k8sClient.Status().Update(ctx, gpuNode)).To(Succeed())
				}).Should(Succeed())
			}
		}

		b.GetPoolGpuList(poolIndex)
	}
	b.UpdateHypervisorStatus(true)

	return b.TensorFusionEnv
}

func GenerateKarpenterEC2NodeClass() {
	ec2 := &unstructured.Unstructured{}

	// Inject an EC2NodeClass
	ec2.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   "karpenter.k8s.aws",
		Version: "v1",
		Kind:    "EC2NodeClass",
	})
	ec2.SetName("test-ec2-node-class")

	ec2.Object["spec"] = map[string]any{
		// Required
		"role":      "arn:aws:iam::123456789012:role/dummy",
		"amiFamily": "AL2023",

		// subnetSelectorTerms – at least 1 element
		"subnetSelectorTerms": []any{
			map[string]any{
				"tags": map[string]any{
					"kubernetes.io/cluster/test": "owned",
				},
			},
		},

		// securityGroupSelectorTerms – at least 1 element
		"securityGroupSelectorTerms": []any{
			map[string]any{
				"tags": map[string]any{
					"karpenter.sh/discovery": "dummy",
				},
			},
		},

		// amiSelectorTerms – newly added and required in v1; provide a dummy AMI ID
		"amiSelectorTerms": []any{
			map[string]any{
				"id": "ami-0123456789abcdef0",
			},
		},
	}
	// May already exist, try to delete first before creating (for test repeatability)
	_ = k8sClient.Delete(ctx, ec2)
	Expect(k8sClient.Create(ctx, ec2)).To(Succeed())
}
