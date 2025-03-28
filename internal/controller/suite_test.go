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
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	ctrl "sigs.k8s.io/controller-runtime"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	corev1 "k8s.io/api/core/v1"

	tensorfusionaiv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
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

const (
	timeout  = time.Second * 10
	interval = time.Millisecond * 100
)

func TestControllers(t *testing.T) {
	RegisterFailHandler(Fail)

	RunSpecs(t, "Controller Suite")
}

var _ = BeforeSuite(func() {
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

	err = tensorfusionaiv1.AddToScheme(scheme.Scheme)
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

	err = (&GPUPoolCompactionReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorderFor("GPUPoolCompaction"),
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

	//	scheduler := scheduler.NewScheduler(mgr.GetClient())
	//	err = (&TensorFusionConnectionReconciler{
	//		Client:   mgr.GetClient(),
	//		Scheme:   mgr.GetScheme(),
	//		Recorder: mgr.GetEventRecorderFor("TensorFusionConnection"),
	//	}).SetupWithManager(mgr)
	//	Expect(err).ToNot(HaveOccurred())
	//
	err = (&GPUReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(ctx, mgr)
	Expect(err).ToNot(HaveOccurred())

	// gpuInfos, err := config.LoadGpuInfoFromFile("")
	// Expect(err).ToNot(HaveOccurred())

	// err = (&TensorFusionWorkloadReconciler{
	//	Client:    mgr.GetClient(),
	//	Scheme:    mgr.GetScheme(),
	//	Scheduler: scheduler,
	//	Recorder:  mgr.GetEventRecorderFor("tensorfusionworkload"),
	//	GpuInfos:  config.MockGpuInfo(),
	// }).SetupWithManager(mgr)
	// Expect(err).ToNot(HaveOccurred())

	go func() {
		defer GinkgoRecover()
		err = mgr.Start(ctx)
		Expect(err).ToNot(HaveOccurred(), "failed to run manager")
	}()

	// create a cluster with a pool and a node to generate gpunode to simplify testing
	createMockClusterWithPool(ctx)
	createMockGPUNode(ctx, "mock-node")

	// TODO: backward compatible with existing tests, can be removed when unnecessary
	createMockPoolForTestsUsingManualReconcile(ctx)
})

var _ = AfterSuite(func() {
	By("tearing down the test environment")
	cancel()
	err := testEnv.Stop()
	Expect(err).NotTo(HaveOccurred())
})

func createMockClusterWithPool(ctx context.Context) *tfv1.TensorFusionCluster {
	tfc := &tfv1.TensorFusionCluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "mock-cluster",
			Namespace: "default",
		},
		Spec: tfv1.TensorFusionClusterSpec{
			GPUPools: []tfv1.GPUPoolDefinition{
				{
					Name:         "mock-pool",
					SpecTemplate: *config.MockGPUPoolSpec,
				},
			},
		},
	}
	Expect(k8sClient.Create(ctx, tfc)).To(Succeed())
	return tfc
}

func createMockPoolForTestsUsingManualReconcile(ctx context.Context) {
	poolSpecCopy := config.MockGPUPoolSpec.DeepCopy()
	poolSpecCopy.NodeManagerConfig.NodeSelector.NodeSelectorTerms = []corev1.NodeSelectorTerm{
		{
			MatchExpressions: []corev1.NodeSelectorRequirement{
				{
					Key:      "foo-mock-label", // avoid conflict
					Operator: "In",
					Values:   []string{"true"},
				},
			},
		},
	}

	pool := &tfv1.GPUPool{
		ObjectMeta: metav1.ObjectMeta{
			Name: "mock",
		},
		Spec: *poolSpecCopy,
	}
	Expect(k8sClient.Create(ctx, pool)).To(Succeed())
}

func getMockCluster(ctx context.Context) *tfv1.TensorFusionCluster {
	tfc := &tfv1.TensorFusionCluster{}
	Eventually(func() error {
		return k8sClient.Get(ctx, client.ObjectKey{Name: "mock-cluster", Namespace: "default"}, tfc)
	}).Should(Succeed())
	return tfc
}

func getMockGPUPool(ctx context.Context) *tfv1.GPUPool {
	tfc := getMockCluster(ctx)
	poolList := &tfv1.GPUPoolList{}
	Eventually(func() string {
		err := k8sClient.List(ctx, poolList, client.MatchingLabels(map[string]string{
			constants.LabelKeyOwner: tfc.GetName(),
		}))
		Expect(err).NotTo(HaveOccurred())
		if len(poolList.Items) > 0 {
			return poolList.Items[0].Name
		}
		return ""
	}, timeout, interval).Should(Equal(tfc.Name + "-" + tfc.Spec.GPUPools[0].Name))

	return &poolList.Items[0]
}

// mock k8s node with specific labels to generate gpunode
func createMockGPUNode(ctx context.Context, name string) {
	selectors := strings.Split(constants.InitialGPUNodeSelector, "=")
	coreNode := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				selectors[0]: selectors[1],
				// same with the NodeManagerConfig in gpupool_mock.go
				"mock-label": "true",
			},
		},
		Spec: corev1.NodeSpec{},
	}
	Expect(k8sClient.Create(ctx, coreNode)).To(Succeed())
}

func getMockGPUNode(ctx context.Context, name string) *tfv1.GPUNode {
	gpuNode := &tfv1.GPUNode{}
	Eventually(func() string {
		err := k8sClient.Get(ctx, client.ObjectKey{Name: name}, gpuNode)
		if err != nil {
			return ""
		} else {
			return gpuNode.Status.KubernetesNodeName
		}
	}, timeout, interval).Should(Equal(name))

	return gpuNode
}
