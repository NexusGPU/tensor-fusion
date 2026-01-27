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

package main

import (
	"context"
	"crypto/tls"
	"flag"
	"os"
	"time"

	// Import all Kubernetes client auth plugins (e.g. Azure, GCP, OIDC, etc.)
	// to ensure that exec-entrypoint and run can make use of them.
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/cmd/sched"
	"github.com/NexusGPU/tensor-fusion/internal/alert"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/pricing"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/controller"
	"github.com/NexusGPU/tensor-fusion/internal/gang"
	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/indexallocator"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/NexusGPU/tensor-fusion/internal/scheduler/expander"
	"github.com/NexusGPU/tensor-fusion/internal/scheduler/gpuresources"
	gpuTopoPlugin "github.com/NexusGPU/tensor-fusion/internal/scheduler/gputopo"
	"github.com/NexusGPU/tensor-fusion/internal/server"
	"github.com/NexusGPU/tensor-fusion/internal/server/router"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/internal/version"
	webhookcorev1 "github.com/NexusGPU/tensor-fusion/internal/webhook/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	k8sVer "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
	"k8s.io/kubernetes/pkg/scheduler"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/filters"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	schemeBuilder "sigs.k8s.io/controller-runtime/pkg/scheme"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
	karpv1 "sigs.k8s.io/karpenter/pkg/apis/v1"
	"sigs.k8s.io/yaml"
	// +kubebuilder:scaffold:imports
)

var (
	scheme            = runtime.NewScheme()
	setupLog          = ctrl.Log.WithName("setup")
	alertCanBeEnabled = false
)

const LeaderElectionID = "85104305.tensor-fusion.ai"

var metricsAddr string
var enableLeaderElection bool
var probeAddr string
var secureMetrics bool
var enableHTTP2 bool
var tlsOpts []func(*tls.Config)
var gpuInfoConfig string
var metricsPath string
var nodeLevelPortRange string
var clusterLevelPortRange string
var enableAutoScale bool
var enableAlert bool
var alertManagerAddr string
var timeSeriesDB *metrics.TimeSeriesDB
var dynamicConfigPath string
var alertEvaluator *alert.AlertEvaluator
var schedulerConfigPath string
var alertEvaluatorReady chan struct{}
var enableAutoExpander bool
var compatibleWithNvidiaContainerToolkit bool

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(tfv1.AddToScheme(scheme))
	// +kubebuilder:scaffold:scheme

	karpenterScheme := &schemeBuilder.Builder{
		GroupVersion: schema.GroupVersion{Group: "karpenter.sh", Version: "v1"},
	}
	karpenterScheme.Register(&karpv1.NodeClaim{}, &karpv1.NodeClaimList{})
	karpenterScheme.Register(&karpv1.NodePool{}, &karpv1.NodePoolList{})
	utilruntime.Must(karpenterScheme.AddToScheme(scheme))
}

//nolint:gocyclo
func main() {
	flag.StringVar(&metricsAddr, "metrics-bind-address", "0", "The address the metrics endpoint binds to. "+
		"Use :8443 for HTTPS or :8080 for HTTP, or leave as 0 to disable the metrics service.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. "+
			"Enabling this will ensure there is only one active controller manager.")
	flag.BoolVar(&secureMetrics, "metrics-secure", false,
		"If set, the metrics endpoint is served securely via HTTPS. Use --metrics-secure=false to use HTTP instead.")
	flag.BoolVar(&enableHTTP2, "enable-http2", false,
		"If set, HTTP/2 will be enabled for the metrics and webhook servers")
	flag.StringVar(&gpuInfoConfig, "gpu-info-config",
		"/etc/tensor-fusion/gpu-info.yaml", "specify the path to gpuInfoConfig file")
	flag.StringVar(&dynamicConfigPath, "dynamic-config",
		"/etc/tensor-fusion/config.yaml", "specify the path to dynamic config file")
	flag.StringVar(&schedulerConfigPath, "scheduler-config", "/etc/tensor-fusion/scheduler-config.yaml",
		"specify the path to TensorFusion scheduler config file")
	flag.StringVar(&metricsPath, "metrics-path", "/logs/metrics.log", "specify the path to metrics file")
	flag.StringVar(&nodeLevelPortRange, "host-port-range", "40000-42000",
		"specify the port range for assigning ports to pre-scheduled Pods such as vGPU workers")
	flag.StringVar(&clusterLevelPortRange, "cluster-host-port-range", "42000-62000",
		"specify the port range for assigning ports to random Pods"+
			" marked with `tensor-fusion.ai/host-port: auto` and `tensor-fusion.ai/port-name: ssh`")
	flag.BoolVar(&enableAlert, "enable-alert", false, "if turn on alert, "+
		"TensorFusion will generate alerts with built-in rules, alert rules are managed in"+
		" configMap `tensor-fusion-alert-rules` of TensorFusion system namespace")
	flag.BoolVar(&enableAutoScale, "enable-auto-scale", false, "if turn on auto scale, "+
		"TensorFusion will auto scale vGPU TFlops and VRAM based on the usage and traffic")
	flag.StringVar(&alertManagerAddr, "alert-manager-addr",
		"alertmanager.tensor-fusion-sys.svc.cluster.local:9093",
		"specify the alert manager address, TensorFusion will generate alerts with "+
			"built-in rules if enabled alert, you can configure routers and receivers "+
			"in your own alertmanager config, "+
			"refer https://prometheus.io/docs/alerting/latest/configuration")
	flag.BoolVar(&enableAutoExpander, "enable-auto-expander", false, "if turn on auto expander, "+
		"TensorFusion will auto expand Nodes then Pending Pods which caused by insufficient GPU resources found")
	flag.BoolVar(&compatibleWithNvidiaContainerToolkit, "compatible-with-nvidia-container-toolkit", false,
		"if enabled, node discovery will wait for NVIDIA Container Toolkit toolkit-ready validation before starting")

	klog.InitFlags(nil)
	flag.Parse()
	ctrl.SetLogger(klog.NewKlogr())
	ctx := context.Background()

	setupLog.Info(version.VersionInfo())

	disableHTTP2 := func(c *tls.Config) {
		setupLog.Info("disabling http/2")
		c.NextProtos = []string{"http/1.1"}
	}

	if !enableHTTP2 {
		tlsOpts = append(tlsOpts, disableHTTP2)
	}

	webhookServer := webhook.NewServer(webhook.Options{
		TLSOpts: tlsOpts,
	})

	gpuInfos := make([]config.GpuInfo, 0)
	gpuPricingMap := make(map[string]float64)
	startWatchGPUInfoChanges(ctx, &gpuInfos, gpuPricingMap)
	utils.InitServiceAccountConfig()

	metricsServerOptions := metricsserver.Options{
		BindAddress:   metricsAddr,
		SecureServing: secureMetrics,
		TLSOpts:       tlsOpts,
	}

	if secureMetrics {
		metricsServerOptions.FilterProvider = filters.WithAuthenticationAndAuthorization
	}

	utils.NormalizeKubeConfigEnv()
	kc := ctrl.GetConfigOrDie()
	mgr, err := ctrl.NewManager(kc, ctrl.Options{
		Scheme:                  scheme,
		Metrics:                 metricsServerOptions,
		WebhookServer:           webhookServer,
		HealthProbeBindAddress:  probeAddr,
		LeaderElection:          enableLeaderElection,
		LeaderElectionID:        LeaderElectionID,
		LeaderElectionNamespace: utils.CurrentNamespace(),
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	k8sClient, err := kubernetes.NewForConfig(kc)
	if err != nil {
		setupLog.Error(err, "unable to create k8s client")
		os.Exit(1)
	}
	version, err := k8sClient.Discovery().ServerVersion()
	if err != nil {
		setupLog.Error(err, "unable to get k8s version")
		os.Exit(1)
	}
	// set env for feature gating, so that to be compatible with different k8s version
	setupLog.Info("detected API server version for feature gating", "version", version.String())
	_ = os.Setenv(constants.KubeApiVersionMajorEnv, version.Major)
	_ = os.Setenv(constants.KubeApiVersionMinorEnv, version.Minor)

	// TODO: there will still be risk after FeatureGate removed when the feature is stable for a long time
	// To be compatible with long-term k8s version, need to patch Kubernetes source code
	k8sVersion := k8sVer.MustParseSemantic(version.String())
	err = feature.DefaultMutableFeatureGate.SetEmulationVersion(k8sVersion)
	if err != nil {
		setupLog.Error(err, "unable to set k8s version for feature gating")
	}

	alertEvaluatorReady = make(chan struct{})
	setupTimeSeriesAndWatchGlobalConfigChanges(ctx, mgr)

	metricsRecorder := startMetricsRecorder(enableLeaderElection, mgr, gpuPricingMap)

	// Initialize Index allocator for Device Plugin communication
	indexAllocator, err := indexallocator.NewIndexAllocator(ctx, mgr.GetClient())
	if err != nil {
		setupLog.Error(err, "unable to set up index allocator")
		os.Exit(1)
	}
	_ = indexAllocator.SetupWithManager(ctx, mgr)

	// Initialize GPU allocator and set up watches
	allocator, portAllocator := startTensorFusionAllocators(ctx, mgr, indexAllocator)

	ensureLeaderInfoConfigMap(mgr)

	startAutoScaler(mgr, allocator)

	// Create pricing provider for webhook
	pricingProvider := pricing.NewStaticPricingProvider()
	startWebhook(mgr, portAllocator, indexAllocator, pricingProvider)

	scheduler, nodeExpander := startScheduler(ctx, allocator, indexAllocator, mgr, k8sVersion)

	startCustomResourceController(ctx, mgr, metricsRecorder, allocator, portAllocator, indexAllocator, nodeExpander)

	startHttpServerForTFClient(ctx, kc, portAllocator, indexAllocator, allocator, scheduler, nodeExpander, mgr.Elected())

	// +kubebuilder:scaffold:builder
	addHealthCheckAPI(mgr)

	addStopHandlers(mgr, allocator)

	setupLog.Info("starting tensor fusion controller manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}

func addHealthCheckAPI(mgr manager.Manager) {
	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}
}

func startTensorFusionAllocators(
	ctx context.Context,
	mgr manager.Manager,
	indexAllocator *indexallocator.IndexAllocator,
) (*gpuallocator.GpuAllocator, *portallocator.PortAllocator) {
	allocator := gpuallocator.NewGpuAllocator(ctx, indexAllocator, mgr.GetClient(), 10*time.Second)
	if err := allocator.SetupWithManager(ctx, mgr); err != nil {
		setupLog.Error(err, "unable to set up GPU allocator watches")
		os.Exit(1)
	}

	// Initialize Port allocator and set up watches
	portAllocator, err := portallocator.NewPortAllocator(ctx, mgr.GetClient(), nodeLevelPortRange, clusterLevelPortRange)
	if err != nil {
		setupLog.Error(err, "unable to set up port allocator")
		os.Exit(1)
	}
	_ = portAllocator.SetupWithManager(ctx, mgr)
	return allocator, portAllocator
}

func startHttpServerForTFClient(
	ctx context.Context,
	kc *rest.Config,
	portAllocator *portallocator.PortAllocator,
	indexAllocator *indexallocator.IndexAllocator,
	allocator *gpuallocator.GpuAllocator,
	scheduler *scheduler.Scheduler,
	nodeExpander *expander.NodeExpander,
	leaderChan <-chan struct{},
) {
	client, err := client.NewWithWatch(kc, client.Options{Scheme: scheme})
	if err != nil {
		setupLog.Error(err, "failed to create client with watch")
		os.Exit(1)
	}
	connectionRouter, err := router.NewConnectionRouter(ctx, client)
	if err != nil {
		setupLog.Error(err, "failed to create connection router")
		os.Exit(1)
	}
	assignHostPortRouter, err := router.NewAssignHostPortRouter(ctx, portAllocator)
	if err != nil {
		setupLog.Error(err, "failed to create assign host port router")
		os.Exit(1)
	}
	assignIndexRouter, err := router.NewAssignIndexRouter(ctx, indexAllocator)
	if err != nil {
		setupLog.Error(err, "failed to create assign index router")
		os.Exit(1)
	}
	allocatorInfoRouter, err := router.NewAllocatorInfoRouter(ctx, allocator, scheduler)
	if err != nil {
		setupLog.Error(err, "failed to create allocator info router")
		os.Exit(1)
	}
	nodeScalerInfoRouter, err := router.NewNodeScalerInfoRouter(ctx, nodeExpander)
	if err != nil {
		setupLog.Error(err, "failed to create node scaler info router")
		os.Exit(1)
	}
	httpServer := server.NewHTTPServer(
		connectionRouter, assignHostPortRouter, assignIndexRouter, allocatorInfoRouter, nodeScalerInfoRouter, leaderChan,
	)
	go func() {
		// Port defaults to 8080, set with env var `PORT`
		err := httpServer.Run()
		if err != nil {
			setupLog.Error(err, "problem running HTTP server")
			os.Exit(1)
		}
	}()
}

func startCustomResourceController(
	ctx context.Context,
	mgr manager.Manager,
	metricsRecorder metrics.MetricsRecorder,
	allocator *gpuallocator.GpuAllocator,
	portAllocator *portallocator.PortAllocator,
	indexAllocator *indexallocator.IndexAllocator,
	nodeExpander *expander.NodeExpander,
) {
	if os.Getenv(constants.EnableCustomResourceControllerEnv) == constants.FalseStringValue {
		return
	}

	var err error
	if err = (&controller.TensorFusionConnectionReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorder("TensorFusionConnection"),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "TensorFusionConnection")
		os.Exit(1)
	}

	if err = (&controller.GPUReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(ctx, mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "GPU")
		os.Exit(1)
	}

	if err = (&controller.TensorFusionClusterReconciler{
		Client:          mgr.GetClient(),
		Scheme:          mgr.GetScheme(),
		Recorder:        mgr.GetEventRecorder("TensorFusionCluster"),
		MetricsRecorder: &metricsRecorder,
	}).SetupWithManager(mgr, true); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "TensorFusionCluster")
		os.Exit(1)
	}

	GPUPoolReconciler := &controller.GPUPoolReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorder("GPUPool"),
	}
	if err = GPUPoolReconciler.SetupWithManager(mgr, true); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "GPUPool")
		os.Exit(1)
	}

	if err = (&controller.GPUNodeReconciler{
		Client:                               mgr.GetClient(),
		Scheme:                               mgr.GetScheme(),
		Recorder:                             mgr.GetEventRecorder("GPUNode"),
		Allocator:                            allocator,
		Expander:                             nodeExpander,
		CompatibleWithNvidiaContainerToolkit: compatibleWithNvidiaContainerToolkit,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "GPUNode")
		os.Exit(1)
	}
	if err = (&controller.GPUPoolCompactionReconciler{
		Client:    mgr.GetClient(),
		Scheme:    mgr.GetScheme(),
		Recorder:  mgr.GetEventRecorder("GPUPoolCompaction"),
		Allocator: allocator,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "GPUPoolCompaction")
		os.Exit(1)
	}
	if err = (&controller.GPUNodeClassReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "GPUNodeClass")
		os.Exit(1)
	}
	if err = (&controller.SchedulingConfigTemplateReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "SchedulingConfigTemplate")
		os.Exit(1)
	}
	if err = (&controller.PodReconciler{
		Client:         mgr.GetClient(),
		Scheme:         mgr.GetScheme(),
		Allocator:      allocator,
		PortAllocator:  portAllocator,
		Expander:       nodeExpander,
		IndexAllocator: indexAllocator,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "Pod")
		os.Exit(1)
	}
	if err = (&controller.NodeReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorder("Node"),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "Node")
		os.Exit(1)
	}

	if err = (&controller.WorkloadProfileReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "WorkloadProfile")
		os.Exit(1)
	}
	if err = (&controller.TensorFusionWorkloadReconciler{
		Client:        mgr.GetClient(),
		Scheme:        mgr.GetScheme(),
		Recorder:      mgr.GetEventRecorder("tensorfusionworkload"),
		PortAllocator: portAllocator,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "TensorFusionWorkload")
		os.Exit(1)
	}
	if err = (&controller.GPUResourceQuotaReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Recorder: mgr.GetEventRecorder("GPUResourceQuota"),

		QuotaStore: allocator.GetQuotaStore(),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "GPUResourceQuota")
		os.Exit(1)
	}

	if err := (&controller.GPUNodeClaimReconciler{
		Client:   mgr.GetClient(),
		Scheme:   mgr.GetScheme(),
		Expander: nodeExpander,
		Recorder: mgr.GetEventRecorder("GPUNodeClaim"),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "GPUNodeClaim")
		os.Exit(1)
	}
}

func startWebhook(
	mgr manager.Manager,
	portAllocator *portallocator.PortAllocator,
	indexAllocator *indexallocator.IndexAllocator,
	pricingProvider pricing.PricingProvider,
) {
	if os.Getenv(constants.EnableWebhookEnv) == constants.FalseStringValue {
		return
	}
	if err := webhookcorev1.SetupPodWebhookWithManager(mgr, portAllocator, indexAllocator, pricingProvider); err != nil {
		setupLog.Error(err, "unable to create webhook", "webhook", "Pod")
		os.Exit(1)
	}
}

func startScheduler(
	ctx context.Context,
	allocator *gpuallocator.GpuAllocator,
	indexAllocator *indexallocator.IndexAllocator,
	mgr manager.Manager,
	k8sVersion *k8sVer.Version,
) (*scheduler.Scheduler, *expander.NodeExpander) {
	if os.Getenv(constants.EnableSchedulerEnv) == constants.FalseStringValue {
		return nil, nil
	}
	if schedulerConfigPath == "" {
		setupLog.Error(nil, "scheduler config path is empty, please and --scheduler-config in command line")
		os.Exit(1)
	}

	// Create gang scheduling manager
	gangManager := gang.NewManager(nil, mgr.GetEventRecorder("GangScheduler"), gpuresources.Name)

	gpuResourceFitOpt := app.WithPlugin(
		gpuresources.Name,
		gpuresources.NewWithDeps(allocator, indexAllocator, gangManager, mgr.GetClient()),
	)
	gpuTopoOpt := app.WithPlugin(
		gpuTopoPlugin.Name,
		gpuTopoPlugin.NewWithDeps(allocator, mgr.GetClient()),
	)

	cc, scheduler, nodeExpander, err := sched.SetupScheduler(
		ctx, mgr, schedulerConfigPath, false, k8sVersion,
		allocator, enableAutoExpander, gpuResourceFitOpt, gpuTopoOpt,
	)
	if err != nil {
		setupLog.Error(err, "unable to create tensor fusion scheduler")
		os.Exit(1)
	}

	if err := sched.RunScheduler(ctx, cc, scheduler, mgr); err != nil {
		setupLog.Error(err, "unable to run tensor fusion scheduler")
		os.Exit(1)
	}
	return scheduler, nodeExpander
}

func setupTimeSeriesAndWatchGlobalConfigChanges(ctx context.Context, mgr manager.Manager) {
	globalConfig := &config.GlobalConfig{}
	if err := utils.LoadConfigFromFile(dynamicConfigPath, globalConfig); err != nil {
		setupLog.Error(err, "unable to load global config file", "configPath", dynamicConfigPath)
		os.Exit(1)
	}
	config.SetGlobalConfig(globalConfig)

	// only init TSDB and evaluator in leader
	needTSDB := enableAlert || enableAutoScale

	err := mgr.Add(manager.RunnableFunc(func(ctx context.Context) error {
		if !needTSDB {
			return nil
		}
		timeSeriesDB = setupTimeSeriesDB()
		alertEvaluator = alert.NewAlertEvaluator(ctx, timeSeriesDB, config.GetGlobalConfig().AlertRules, alertManagerAddr)
		alertCanBeEnabled = true
		close(alertEvaluatorReady)
		setupLog.Info("time series db setup successfully.")
		return nil
	}))
	if err != nil {
		setupLog.Error(err, "unable to add time series setup to manager")
		os.Exit(1)
	}
	go watchAndHandleConfigChanges(ctx, mgr, needTSDB)
}

func watchAndHandleConfigChanges(ctx context.Context, mgr manager.Manager, needTSDB bool) {
	ch, err := utils.WatchConfigFileChanges(ctx, dynamicConfigPath)
	if err != nil {
		ctrl.Log.Error(err, "unable to watch global config file, file may not exist",
			"configPath", dynamicConfigPath)
		os.Exit(1)
	}

	for data := range ch {
		ctrl.Log.Info("global config file loading")
		globalConfig := &config.GlobalConfig{}
		err := yaml.Unmarshal(data, globalConfig)
		if err != nil {
			ctrl.Log.Error(err, "unable to reload global config file, not valid config structure",
				"configPath", dynamicConfigPath)
			continue
		}
		config.SetGlobalConfig(globalConfig)

		// handle alert rules update
		go func() {
			<-alertEvaluatorReady
			if alertCanBeEnabled && enableAlert {
				err = alertEvaluator.UpdateAlertRules(globalConfig.AlertRules)
				if err != nil {
					ctrl.Log.Error(err, "unable to update alert rules", "configPath", dynamicConfigPath)
				}
			}
		}()

		// handle metrics ttl update
		if needTSDB {
			go func() {
				<-mgr.Elected()
				err = timeSeriesDB.SetTableTTL(globalConfig.MetricsTTL)
				if err != nil {
					ctrl.Log.Error(err, "unable to update metrics ttl", "ttl config", globalConfig.MetricsTTL)
				}
			}()
		}
	}
}

func startMetricsRecorder(
	enableLeaderElection bool,
	mgr manager.Manager,
	gpuPricingMap map[string]float64,
) metrics.MetricsRecorder {
	metricsRecorder := metrics.MetricsRecorder{
		MetricsOutputPath:  metricsPath,
		HourlyUnitPriceMap: gpuPricingMap,

		// Worker level map will be updated by cluster reconcile
		// Key is poolName, second level key is QoS level
		WorkerUnitPriceMap: make(map[string]map[string]metrics.RawBillingPricing, 8),
	}
	if enableLeaderElection {
		go func() {
			<-mgr.Elected()
			metricsRecorder.Start()
		}()
	} else {
		go metricsRecorder.Start()
	}
	return metricsRecorder
}

func startAutoScaler(mgr manager.Manager, allocator *gpuallocator.GpuAllocator) {
	if enableAutoScale {
		setupLog.Info("auto scale enabled")
		if err := autoscaler.SetupWithManager(mgr, allocator); err != nil {
			setupLog.Error(err, "unable to start auto scaler")
			os.Exit(1)
		}
	}
}

func startWatchGPUInfoChanges(ctx context.Context, gpuInfos *[]config.GpuInfo, gpuPricingMap map[string]float64) {
	initGpuInfos := make([]config.GpuInfo, 0)
	err := utils.LoadConfigFromFile(gpuInfoConfig, &initGpuInfos)
	if err != nil {
		setupLog.Error(err, "unable to load gpuInfo file, "+
			"file may not exist, this error will cause billing not working", "gpuInfoConfig", gpuInfoConfig)
		return
	}
	*gpuInfos = initGpuInfos
	pricing.SetTflopsMapAndInitGPUPricingInfo(ctx, gpuInfos)

	ch, err := utils.WatchConfigFileChanges(ctx, gpuInfoConfig)
	if err != nil {
		setupLog.Error(err, "unable to watch gpuInfo file, "+
			"file may not exist, this error will cause billing not working", "gpuInfoConfig", gpuInfoConfig)
		return
	}

	go func() {
		for data := range ch {
			updatedGpuInfos := make([]config.GpuInfo, 0)
			err := yaml.Unmarshal(data, &updatedGpuInfos)
			if err != nil {
				setupLog.Error(err, "unable to reload gpuInfo file, file is not valid yaml", "gpuInfoConfig", gpuInfoConfig)
				continue
			}
			*gpuInfos = updatedGpuInfos
			for _, gpuInfo := range updatedGpuInfos {
				gpuPricingMap[gpuInfo.FullModelName] = gpuInfo.CostPerHour
			}
			pricing.SetTflopsMapAndInitGPUPricingInfo(ctx, gpuInfos)
		}
	}()
}

// Setup GreptimeDB connection
func setupTimeSeriesDB() *metrics.TimeSeriesDB {
	timeSeriesDB := &metrics.TimeSeriesDB{}
	connection := metrics.GreptimeDBConnection{
		Host:     utils.GetEnvOrDefault("TSDB_MYSQL_HOST", "127.0.0.1"),
		Port:     utils.GetEnvOrDefault("TSDB_MYSQL_PORT", "4002"),
		User:     utils.GetEnvOrDefault("TSDB_MYSQL_USER", "root"),
		Password: utils.GetEnvOrDefault("TSDB_MYSQL_PASSWORD", ""),
		Database: utils.GetEnvOrDefault("TSDB_MYSQL_DATABASE", "public"),
	}
	if err := timeSeriesDB.Setup(connection); err != nil {
		setupLog.Error(err, "unable to setup time series db, features including alert, "+
			"autoScaling, rebalance won't work", "connection", connection.Host, "port",
			connection.Port, "user", connection.User, "database", connection.Database)
		return nil
	}
	return timeSeriesDB
}

func addStopHandlers(mgr manager.Manager, allocator *gpuallocator.GpuAllocator) {
	err := mgr.Add(manager.RunnableFunc(func(ctx context.Context) error {
		// wait for the context to be done
		<-ctx.Done()
		setupLog.Info("stopping allocator")
		if allocator != nil {
			allocator.Stop()
		}
		return nil
	}))
	if err != nil {
		setupLog.Error(err, "unable to add allocator cleanup to manager")
		os.Exit(1)
	}
}

func ensureLeaderInfoConfigMap(mgr manager.Manager) {
	err := mgr.Add(manager.RunnableFunc(func(ctx context.Context) error {
		<-mgr.Elected()
		leaderInfo := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constants.LeaderInfoConfigMapName,
				Namespace: utils.CurrentNamespace(),
			},
		}
		err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
			_, err := controllerutil.CreateOrUpdate(ctx, mgr.GetClient(), leaderInfo, func() error {
				leaderInfo.Data = map[string]string{
					constants.LeaderInfoConfigMapLeaderIPKey: utils.CurrentIP(),
				}
				return nil
			})
			return err
		})
		if err != nil {
			setupLog.Error(err, "Failed to update leader IP info in ConfigMap")
		}
		return nil
	}))
	if err != nil {
		setupLog.Error(err, "unable to add leader info config map to manager")
		os.Exit(1)
	}
}
