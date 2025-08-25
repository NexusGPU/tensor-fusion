package sched

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/NexusGPU/tensor-fusion/cmd/sched"
	gpuResourceFitPlugin "github.com/NexusGPU/tensor-fusion/internal/scheduler/gpuresources"
	gpuTopoPlugin "github.com/NexusGPU/tensor-fusion/internal/scheduler/gputopo"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"go.uber.org/zap/zapcore"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	configz "k8s.io/component-base/configz"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

// defaultBenchmarkConfig returns default benchmark configuration
func defaultBenchmarkConfig() BenchmarkConfig {
	return BenchmarkConfig{
		NumNodes:  1000,
		NumGPUs:   4000,
		NumPods:   10000,
		BatchSize: 100,
		PoolName:  "benchmark-pool",
		Namespace: "benchmark-ns",
		Timeout:   10 * time.Minute,
	}
}

var testEnv *envtest.Environment

func setupKubernetes() (*rest.Config, error) {
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
	return testEnv.Start()
}

// Estimated Performance: 400-500 pods/second for 1K nodes, 10K Pods cluster on Mac M4 Pro
// Adjust SchedulerConfig QPS/Burst, percentage of
func BenchmarkScheduler(b *testing.B) {
	klog.SetLogger(zap.New(zap.WriteTo(os.Stderr), zap.UseDevMode(false), zap.Level(zapcore.ErrorLevel)))
	// Setup phase - runs once before all benchmark iterations
	cfg, err := setupKubernetes()
	if err != nil {
		b.Fatal(err)
	}

	// Write kubeconfig to temp file and set KUBECONFIG env var
	tmpKubeconfigPath, err := writeKubeconfigToTempFileAndSetEnv(cfg)
	if err != nil {
		b.Fatal(err)
	}

	k8sClient, err := client.New(cfg, client.Options{Scheme: scheme.Scheme})
	if err != nil {
		b.Fatal(err)
	}

	benchConfig := defaultBenchmarkConfig()
	fixture := NewBenchmarkFixture(b, benchConfig, k8sClient, true)

	utils.SetProgressiveMigration(false)

	gpuResourceFitOpt := app.WithPlugin(
		gpuResourceFitPlugin.Name,
		gpuResourceFitPlugin.NewWithDeps(fixture.allocator, fixture.client),
	)
	gpuTopoOpt := app.WithPlugin(
		gpuTopoPlugin.Name,
		gpuTopoPlugin.NewWithDeps(fixture.allocator, fixture.client),
	)

	ctx, cancel := context.WithCancel(context.Background())
	testCtx := ctx

	cc, scheduler, err := sched.SetupScheduler(testCtx, nil,
		"../../config/samples/scheduler-config.yaml", true, gpuResourceFitOpt, gpuTopoOpt)
	if err != nil {
		b.Fatal(err)
	}

	// Config registration.
	if cz, err := configz.New("componentconfig"); err != nil {
		b.Fatal(err)
	} else {
		cz.Set(cc.ComponentConfig)
	}

	cc.EventBroadcaster.StartRecordingToSink(testCtx.Done())

	startInformersAndWaitForSync := func(ctx context.Context) {
		// Start all informers.
		cc.InformerFactory.Start(ctx.Done())
		// DynInformerFactory can be nil in tests.
		if cc.DynInformerFactory != nil {
			cc.DynInformerFactory.Start(ctx.Done())
		}

		// Wait for all caches to sync before scheduling.
		cc.InformerFactory.WaitForCacheSync(ctx.Done())
		// DynInformerFactory can be nil in tests.
		if cc.DynInformerFactory != nil {
			cc.DynInformerFactory.WaitForCacheSync(ctx.Done())
		}

		// Wait for all handlers to sync (all items in the initial list delivered) before scheduling.
		if err := scheduler.WaitForHandlersSync(testCtx); err != nil {
			b.Fatal(err)
		}
		b.Log("Handlers synced")
	}
	startInformersAndWaitForSync(testCtx)

	_, info := scheduler.SchedulingQueue.PendingPods()
	b.Log("PendingPods info", info)
	b.Log("=====================================")

	b.Run("Schedule", func(b *testing.B) {
		b.Log("Start scheduling benchmark")
		i := 0
		b.ResetTimer() // Reset timer after setup
		for b.Loop() {
			if i >= 10000 {
				b.Error("benchmark too long, pre-created Pods out of bounds, reduce time or increase NumPods: ", i)
				break
			}

			scheduler.ScheduleOne(testCtx)

			if i%500 == 0 {
				_, info := scheduler.SchedulingQueue.PendingPods()
				b.Logf("PendingPods info %s, iter %d", info, i)
			}
			i++
		}

		b.Cleanup(func() {
			b.Log("Clean up k8s environment")
			_, info := scheduler.SchedulingQueue.PendingPods()
			b.Logf("Final PendingPods info %s", info)
			// wait async BindingCycle to write nodeName to pod/bind sub resource
			// wait async events writing to API server
			time.Sleep(30 * time.Second)
			cancel()
			fixture.Close()
			// Cleanup temp kubeconfig file
			if err := cleanupKubeconfigTempFile(tmpKubeconfigPath); err != nil {
				b.Error(err)
			}
			// Stop test environment
			if err := testEnv.Stop(); err != nil {
				b.Error(err)
			}
		})
	})
}

// SchedulingMetrics holds scheduling performance metrics
type SchedulingMetrics struct {
	Scheduled int
	Failed    int
	LatencyNs float64
}

// writeKubeconfigToTempFileAndSetEnv writes the kubeconfig to a temporary file and sets KUBECONFIG env var
func writeKubeconfigToTempFileAndSetEnv(cfg *rest.Config) (string, error) {
	// Create a temporary file
	tmpFile, err := os.CreateTemp("", "kubeconfig-*.yaml")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpFilePath := tmpFile.Name()
	_ = tmpFile.Close()

	// Convert rest.Config to kubeconfig format
	kubeconfig := clientcmdapi.Config{
		Clusters: map[string]*clientcmdapi.Cluster{
			"test-cluster": {
				Server:                   cfg.Host,
				CertificateAuthorityData: cfg.CAData,
			},
		},
		Contexts: map[string]*clientcmdapi.Context{
			"test-context": {
				Cluster:  "test-cluster",
				AuthInfo: "test-user",
			},
		},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"test-user": {
				ClientCertificateData: cfg.CertData,
				ClientKeyData:         cfg.KeyData,
			},
		},
		CurrentContext: "test-context",
	}

	// Write kubeconfig to temp file
	if err := clientcmd.WriteToFile(kubeconfig, tmpFilePath); err != nil {
		return "", fmt.Errorf("failed to write kubeconfig to temp file: %w", err)
	}

	// Set KUBECONFIG environment variable
	if err := os.Setenv("KUBECONFIG", tmpFilePath); err != nil {
		return "", fmt.Errorf("failed to set KUBECONFIG env var: %w", err)
	}

	return tmpFilePath, nil
}

// cleanupKubeconfigTempFile removes the temporary kubeconfig file and unsets KUBECONFIG env var
func cleanupKubeconfigTempFile(tmpFilePath string) error {
	// Remove the temporary file
	if err := os.Remove(tmpFilePath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove temp kubeconfig file: %w", err)
	}

	// Unset KUBECONFIG environment variable
	if err := os.Unsetenv("KUBECONFIG"); err != nil {
		return fmt.Errorf("failed to unset KUBECONFIG env var: %w", err)
	}

	return nil
}
