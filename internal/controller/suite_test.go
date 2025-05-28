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

	ctrl "sigs.k8s.io/controller-runtime"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	corev1 "k8s.io/api/core/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
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

const (
	timeout  = time.Second * 10
	duration = time.Second * 5
	interval = time.Millisecond * 100
)

func TestControllers(t *testing.T) {
	RegisterFailHandler(Fail)

	RunSpecs(t, "Controller Suite")
}

var _ = BeforeSuite(func() {
	// Expect(os.Setenv("USE_EXISTING_CLUSTER", "true")).Should(Succeed())
	logf.SetLogger(zap.New(zap.WriteTo(GinkgoWriter), zap.UseDevMode(true)))

	ctx, cancel = context.WithCancel(context.TODO())

	By("bootstrapping test environment")
	testEnv = &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "config", "crd", "bases")},
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
	Expect(err).NotTo(HaveOccurred())
	Expect(cfg).NotTo(BeNil())

	err = tfv1.AddToScheme(scheme.Scheme)
	Expect(err).NotTo(HaveOccurred())

	err = corev1.AddToScheme(scheme.Scheme)
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
	})
	Expect(err).ToNot(HaveOccurred())
	err = (&TensorFusionClusterReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("TensorFusionCluster"),
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	err = (&GPUPoolReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("GPUPool"),
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	err = (&GPUNodeReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("GPUNode"),
	}).SetupWithManager(mgr)
	Expect(err).ToNot(HaveOccurred())

	// err = (&GPUPoolCompactionReconciler{
	// 	Client:   mgr.GetClient(),
	// 	Scheme:   mgr.GetScheme(),
	// 	Recorder: mgr.GetEventRecorderFor("GPUPoolCompaction"),
	// }).SetupWithManager(mgr)
	// Expect(err).ToNot(HaveOccurred())

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
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
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

	allocator = gpuallocator.NewGpuAllocator(ctx, mgr.GetClient(), 3*time.Second)
	_, err = allocator.SetupWithManager(ctx, mgr)
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
		Client:    mgr.GetClient(),
		Scheme:    mgr.GetScheme(),
		Allocator: allocator,
		Recorder:  mgr.GetEventRecorderFor("TensorFusionWorkload"),
		GpuInfos:  config.MockGpuInfo(),
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
	clusterKey  client.ObjectKey
	poolCount   int
	poolNodeMap map[int]map[int]int
}

func (c *TensorFusionEnv) GetCluster() *tfv1.TensorFusionCluster {
	GinkgoHelper()
	tfc := &tfv1.TensorFusionCluster{}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, c.clusterKey, tfc)).Should(Succeed())
	}).Should(Succeed())
	return tfc
}

func (c *TensorFusionEnv) UpdateCluster(tfc *tfv1.TensorFusionCluster) {
	GinkgoHelper()
	Expect(k8sClient.Update(ctx, tfc)).Should(Succeed())
}

func (c *TensorFusionEnv) Cleanup() {
	GinkgoHelper()
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
		}, timeout, interval).Should(Succeed())
		delete(c.poolNodeMap, poolIndex)
		c.poolCount--
	}

	Expect(k8sClient.Delete(ctx, tfc)).Should(Succeed())
	Eventually(func(g Gomega) {
		err := k8sClient.Get(ctx, c.clusterKey, tfc)
		g.Expect(err).Should(HaveOccurred())
	}, timeout, interval).Should(Succeed())
}

func (c *TensorFusionEnv) GetGPUPoolList() *tfv1.GPUPoolList {
	GinkgoHelper()
	poolList := &tfv1.GPUPoolList{}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.List(ctx, poolList, client.MatchingLabels(map[string]string{
			constants.LabelKeyOwner: c.clusterKey.Name,
		}))).Should(Succeed())
		g.Expect(poolList.Items).Should(HaveLen(c.poolCount))
	}, timeout, interval).Should(Succeed())
	return poolList
}

func (c *TensorFusionEnv) GetGPUPool(poolIndex int) *tfv1.GPUPool {
	GinkgoHelper()
	pool := &tfv1.GPUPool{}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: c.getPoolName(poolIndex)}, pool)).Should(Succeed())
	}, timeout, interval).Should(Succeed())
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
	}, timeout, interval).Should(Succeed())
	return nodeList
}

func (c *TensorFusionEnv) GetGPUNode(poolIndex int, nodeIndex int) *tfv1.GPUNode {
	GinkgoHelper()
	node := &tfv1.GPUNode{}
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: c.getNodeName(poolIndex, nodeIndex)}, node)).Should(Succeed())
	}, timeout, interval).Should(Succeed())
	return node
}

func (c *TensorFusionEnv) DeleteGPUNode(poolIndex int, nodeIndex int) {
	GinkgoHelper()
	c.DeleteNodeGpuList(poolIndex, nodeIndex)
	node := c.GetGPUNode(poolIndex, nodeIndex)
	Expect(k8sClient.Delete(ctx, node)).Should(Succeed())
	Eventually(func(g Gomega) {
		g.Expect(k8sClient.Get(ctx, client.ObjectKey{Name: c.getNodeName(poolIndex, nodeIndex)}, node)).Should(HaveOccurred())
	}, timeout, interval).Should(Succeed())
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
	}, timeout, interval).Should(Succeed())
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
	}, timeout, interval).Should(Succeed())
	return gpuList
}

// https://book.kubebuilder.io/reference/envtest#testing-considerations
// Unless you’re using an existing cluster, keep in mind that no built-in controllers are running in the test context.
// So the checkStatusAndUpdateVirtualCapacity in gpunode_controller.go checking pod status always pending and the gpunode status can't change to running
// When using an existing cluster, the test speed go a lot faster, may change later?
func (c *TensorFusionEnv) UpdateHypervisorStatus() {
	GinkgoHelper()
	if os.Getenv("USE_EXISTING_CLUSTER") != "true" {
		for poolIndex := range c.poolNodeMap {
			podList := &corev1.PodList{}
			Eventually(func(g Gomega) {
				g.Expect(k8sClient.List(ctx, podList,
					client.InNamespace(utils.CurrentNamespace()),
					client.MatchingLabels(map[string]string{
						fmt.Sprintf(constants.GPUNodePoolIdentifierLabelFormat, c.getPoolName(poolIndex)): "true",
					}),
				)).Should(Succeed())
				g.Expect(podList.Items).Should(HaveLen(len(c.poolNodeMap[poolIndex])))
			}, timeout, interval).Should(Succeed())
			for _, pod := range podList.Items {
				pod.Status.Phase = corev1.PodRunning
				pod.Status.Conditions = append(pod.Status.Conditions, corev1.PodCondition{Type: corev1.PodReady, Status: corev1.ConditionTrue})
				Expect(k8sClient.Status().Update(ctx, &pod)).Should(Succeed())
			}
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

var testEnvId int = 0

func (b *TensorFusionEnvBuilder) Build() *TensorFusionEnv {
	GinkgoHelper()
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
	}, timeout, interval).Should(Succeed())

	// generate nodes
	selectors := strings.Split(constants.InitialGPUNodeSelector, "=")
	for poolIndex, nodeGpuMap := range b.poolNodeMap {
		if poolIndex >= b.poolCount {
			continue
		}

		for nodeIndex, gpuCount := range nodeGpuMap {
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
			if gpuCount > 0 {
				gpuNode := b.GetGPUNode(poolIndex, nodeIndex)
				for gpuIndex := range gpuCount {
					key := client.ObjectKey{
						Name: b.getGPUName(poolIndex, nodeIndex, gpuIndex),
					}
					gpu := &tfv1.GPU{
						ObjectMeta: metav1.ObjectMeta{
							Name: key.Name,
							Labels: map[string]string{
								constants.LabelKeyOwner: gpuNode.Name,
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
			}
		}

		b.GetPoolGpuList(poolIndex)
	}

	b.UpdateHypervisorStatus()

	return b.TensorFusionEnv
}
