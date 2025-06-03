package portallocator

import (
	"context"
	"fmt"
	"path/filepath"
	"runtime"
	"testing"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

// These tests use Ginkgo (BDD-style Go testing framework). Refer to
// http://onsi.github.io/ginkgo/ to learn more about Ginkgo.

var (
	cancel    context.CancelFunc
	cfg       *rest.Config
	ctx       context.Context
	k8sClient client.Client
	testEnv   *envtest.Environment
	mgr       ctrl.Manager
	pa        *PortAllocator
)

func TestPortAllocator(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Port Allocator Suite")
}
func genHostPortPod(name string, nodeName string, port int32, clusterLevel bool) corev1.Pod {
	var labels map[string]string
	if clusterLevel {
		labels = map[string]string{
			constants.GenHostPortLabel:     constants.GenHostPortLabelValue,
			constants.GenHostPortNameLabel: "test",
			constants.LabelKeyOwner:        nodeName,
		}
	} else {
		labels = map[string]string{
			constants.LabelComponent: constants.ComponentWorker,
			constants.LabelKeyOwner:  nodeName,
		}
	}
	return corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
			Labels:    labels,
			Annotations: map[string]string{
				constants.GenPortNumberAnnotation: fmt.Sprintf("%d", port),
			},
		},
		Spec: corev1.PodSpec{
			NodeName: nodeName,
			Containers: []corev1.Container{
				{
					Name:  "test",
					Image: "test-image",
					Ports: []corev1.ContainerPort{
						{
							Name:          "test",
							ContainerPort: 80,
							HostPort:      port,
						},
					},
				},
			},
		},
	}
}

var _ = BeforeSuite(func() {
	logf.SetLogger(zap.New(zap.WriteTo(GinkgoWriter), zap.UseDevMode(true)))

	ctx, cancel = context.WithCancel(context.TODO())

	By("bootstrapping test environment")
	testEnv = &envtest.Environment{
		CRDDirectoryPaths:     []string{filepath.Join("..", "..", "config", "crd", "bases")},
		ErrorIfCRDPathMissing: false,
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

	// Create a Kubernetes client
	k8sClient, err = client.New(cfg, client.Options{Scheme: scheme.Scheme})
	Expect(err).NotTo(HaveOccurred())
	Expect(k8sClient).NotTo(BeNil())

	mgr, err = ctrl.NewManager(cfg, ctrl.Options{
		Scheme: scheme.Scheme,
		Metrics: metricsserver.Options{
			BindAddress: "0",
		},
	})
	Expect(err).NotTo(HaveOccurred())

	// Create test GPUs with metadata only first
	workers := []corev1.Pod{
		genHostPortPod("worker-1", "node-1", 40000, false),
		genHostPortPod("worker-2", "node-1", 40001, false),
		genHostPortPod("worker-3", "node-1", 40127, false),
		genHostPortPod("worker-4", "node-2", 40003, false),
		genHostPortPod("worker-5", "node-2", 40065, false),
		genHostPortPod("lab-1", "node-1", 42001, true),
		genHostPortPod("lab-2", "node-1", 59999, true),
	}

	// First create the GPUs without status
	for i := range workers {
		err = k8sClient.Create(ctx, &workers[i])
		Expect(err).NotTo(HaveOccurred())
	}

	pa, err = NewPortAllocator(ctx, k8sClient, "40000-42000", "42001-60000")
	Expect(err).NotTo(HaveOccurred())
	readyCh := pa.SetupWithManager(ctx, mgr)
	Expect(err).NotTo(HaveOccurred())

	go func() {
		defer GinkgoRecover()
		err = mgr.Start(ctx)
		Expect(err).ToNot(HaveOccurred(), "failed to run manager")
	}()
	<-readyCh
})

var _ = AfterSuite(func() {
	By("tearing down the test environment")
	cancel()
	err := testEnv.Stop()
	Expect(err).NotTo(HaveOccurred())
})
