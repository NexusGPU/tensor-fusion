/*
Copyright 2014 The Kubernetes Authors.

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
package sched

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/gpuallocator"
	"github.com/NexusGPU/tensor-fusion/internal/scheduler/expander"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	k8sVer "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/events"
	"k8s.io/component-base/configz"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
	schedulerserverconfig "k8s.io/kubernetes/cmd/kube-scheduler/app/config"
	"k8s.io/kubernetes/cmd/kube-scheduler/app/options"
	"k8s.io/kubernetes/pkg/scheduler"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/latest"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	yaml "sigs.k8s.io/yaml"
)

const (
	schedulerConfigFlagSet = "misc"
	schedulerConfigFlag    = "config"
	configName             = "componentconfig"
	clientConnectionCfgKey = "clientConnection"
	kubeConfigCfgKey       = "kubeconfig"

	nominatedPodRequeueWatchdogIntervalEnv     = "NOMINATED_POD_REQUEUE_WATCHDOG_INTERVAL"
	defaultNominatedPodRequeueWatchdogInterval = 10 * time.Minute
)

func SetupScheduler(
	ctx context.Context,
	mgr manager.Manager,
	schedulerConfigPath string,
	disableHttpEndpoint bool,
	k8sVersion *k8sVer.Version,
	allocator *gpuallocator.GpuAllocator,
	enableNodeExpander bool,
	outOfTreeRegistryOptions ...app.Option,
) (*schedulerserverconfig.CompletedConfig, *scheduler.Scheduler, *expander.NodeExpander, error) {
	opts := options.NewOptions()
	schedulerConfigFlag := opts.Flags.FlagSet(schedulerConfigFlagSet).Lookup(schedulerConfigFlag)
	schedulerConfigFlag.Changed = true

	if disableHttpEndpoint {
		opts.SecureServing.BindPort = 0
	}

	cfgPath, err := preHandleConfig(schedulerConfigPath)
	if err != nil {
		return nil, nil, nil, err
	}
	err = schedulerConfigFlag.Value.Set(cfgPath)
	if err != nil {
		return nil, nil, nil, err
	}
	err = opts.ComponentGlobalsRegistry.Set()
	if err != nil {
		return nil, nil, nil, err
	}

	// Setup enumerationVersion again since it's overridden by the config
	err = feature.DefaultMutableFeatureGate.SetEmulationVersion(k8sVersion)
	if err != nil {
		return nil, nil, nil, err
	}

	if cfg, err := latest.Default(); err != nil {
		return nil, nil, nil, err
	} else {
		opts.ComponentConfig = cfg
	}

	if errs := opts.Validate(); len(errs) > 0 {
		return nil, nil, nil, utilerrors.NewAggregate(errs)
	}

	c, err := opts.Config(ctx)
	if err != nil {
		return nil, nil, nil, err
	}

	// Get the completed config
	cc := c.Complete()

	outOfTreeRegistry := make(runtime.Registry)
	for _, option := range outOfTreeRegistryOptions {
		if err := option(outOfTreeRegistry); err != nil {
			return nil, nil, nil, err
		}
	}

	recorderFactory := getRecorderFactory(&cc)
	completedProfiles := make([]kubeschedulerconfig.KubeSchedulerProfile, 0)

	sched, err := scheduler.New(ctx,
		cc.Client,
		cc.InformerFactory,
		cc.DynInformerFactory,
		recorderFactory,
		scheduler.WithComponentConfigVersion(cc.ComponentConfig.APIVersion),
		scheduler.WithKubeConfig(cc.KubeConfig),
		scheduler.WithProfiles(cc.ComponentConfig.Profiles...),
		scheduler.WithPercentageOfNodesToScore(cc.ComponentConfig.PercentageOfNodesToScore),
		scheduler.WithFrameworkOutOfTreeRegistry(outOfTreeRegistry),
		scheduler.WithPodMaxBackoffSeconds(cc.ComponentConfig.PodMaxBackoffSeconds),
		scheduler.WithPodInitialBackoffSeconds(cc.ComponentConfig.PodInitialBackoffSeconds),
		scheduler.WithPodMaxInUnschedulablePodsDuration(cc.PodMaxInUnschedulablePodsDuration),
		scheduler.WithExtenders(cc.ComponentConfig.Extenders...),
		scheduler.WithParallelism(cc.ComponentConfig.Parallelism),
		scheduler.WithBuildFrameworkCapturer(func(profile kubeschedulerconfig.KubeSchedulerProfile) {
			completedProfiles = append(completedProfiles, profile)
		}),
	)
	if err != nil {
		return nil, nil, nil, err
	}

	if err := options.LogOrWriteConfig(
		klog.FromContext(ctx),
		opts.WriteConfigTo,
		&cc.ComponentConfig,
		completedProfiles,
	); err != nil {
		return nil, nil, nil, err
	}

	// Initialize node expander
	if enableNodeExpander {
		unschedHandler, nodeExpander := expander.NewUnscheduledPodHandler(
			ctx, sched, allocator,
			mgr.GetEventRecorder("TensorFusionScheduler"),
		)

		// Save the original failure handler to avoid infinite recursion
		originalFailureHandler := sched.FailureHandler
		sched.FailureHandler = func(
			ctx context.Context, fwk framework.Framework, podInfo *framework.QueuedPodInfo,
			status *fwk.Status, nominatingInfo *fwk.NominatingInfo, start time.Time,
		) {
			if status.IsRejected() {
				// Handle TensorFusion pods that are rejected due to lack of GPU resources
				// The unschedHandler will queue the pod and process expansion after buffer delay
				unschedHandler.HandleRejectedPod(ctx, podInfo, status)
			}
			// Call the original failure handler to avoid infinite recursion
			originalFailureHandler(ctx, fwk, podInfo, status, nominatingInfo, start)
		}
		return &cc, sched, nodeExpander, nil
	}
	return &cc, sched, nil, nil
}

func RunScheduler(ctx context.Context,
	cc *schedulerserverconfig.CompletedConfig,
	sched *scheduler.Scheduler,
	mgr manager.Manager,
) error {
	logger := klog.FromContext(ctx)

	// Config registration.
	if cz, err := configz.New(configName); err != nil {
		return fmt.Errorf("unable to register config: %s", err)
	} else {
		cz.Set(cc.ComponentConfig)
	}

	cc.EventBroadcaster.StartRecordingToSink(ctx.Done())

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
		if err := sched.WaitForHandlersSync(ctx); err != nil {
			logger.Error(err, "waiting for handlers to sync")
		}
		logger.V(3).Info("Handlers synced")
	}
	startInformersAndWaitForSync(ctx)

	go func() {
		if mgr != nil {
			<-mgr.Elected()
		}
		logger.Info("Starting scheduling cycle")
		startNominatedPodRequeueWatchdog(
			ctx,
			sched,
			cc.PodMaxInUnschedulablePodsDuration,
		)
		sched.Run(ctx)
		cc.EventBroadcaster.Shutdown()
	}()
	return nil
}

func startNominatedPodRequeueWatchdog(
	ctx context.Context,
	sched *scheduler.Scheduler,
	podMaxInUnschedulablePodsDuration time.Duration,
) {
	if sched == nil || sched.SchedulingQueue == nil {
		return
	}

	logger := klog.FromContext(ctx)
	interval := nominatedPodRequeueWatchdogInterval(logger)
	threshold := podMaxInUnschedulablePodsDuration + time.Minute
	if threshold <= time.Minute {
		threshold = 6 * time.Minute
	}
	logger.Info("Starting nominated TensorFusion pod requeue watchdog",
		"interval", interval,
		"threshold", threshold,
	)

	go func() {
		firstObserved := map[types.UID]time.Time{}
		wait.UntilWithContext(ctx, func(ctx context.Context) {
			now := time.Now()
			pods, summary := sched.SchedulingQueue.PendingPods()
			seen := map[types.UID]struct{}{}
			podsToActivate := map[string]*corev1.Pod{}

			for _, pod := range pods {
				if pod == nil {
					continue
				}
				seen[pod.UID] = struct{}{}

				if !shouldWatchNominatedPod(pod) {
					delete(firstObserved, pod.UID)
					continue
				}

				firstSeenAt, ok := firstObserved[pod.UID]
				if !ok {
					firstObserved[pod.UID] = now
					continue
				}
				if now.Sub(firstSeenAt) < threshold {
					continue
				}

				podsToActivate[string(pod.UID)] = pod
				firstObserved[pod.UID] = now
			}

			for uid := range firstObserved {
				if _, ok := seen[uid]; !ok {
					delete(firstObserved, uid)
				}
			}

			if len(podsToActivate) == 0 {
				return
			}
			logger.Info("Force activating nominated TensorFusion pods left in scheduling queue",
				"count", len(podsToActivate),
				"threshold", threshold,
				"summary", summary,
			)
			sched.SchedulingQueue.Activate(logger, podsToActivate)
		}, interval)
	}()
}

func nominatedPodRequeueWatchdogInterval(logger klog.Logger) time.Duration {
	raw := os.Getenv(nominatedPodRequeueWatchdogIntervalEnv)
	if raw == "" {
		return defaultNominatedPodRequeueWatchdogInterval
	}

	interval, err := time.ParseDuration(raw)
	if err != nil || interval <= 0 {
		logger.Info("Invalid nominated pod requeue watchdog interval, using default",
			"env", nominatedPodRequeueWatchdogIntervalEnv,
			"value", raw,
			"default", defaultNominatedPodRequeueWatchdogInterval,
			"error", err,
		)
		return defaultNominatedPodRequeueWatchdogInterval
	}
	return interval
}

func shouldWatchNominatedPod(pod *corev1.Pod) bool {
	return pod.Spec.NodeName == "" &&
		pod.DeletionTimestamp == nil &&
		pod.Status.NominatedNodeName != "" &&
		utils.IsTensorFusionWorker(pod)
}

func getRecorderFactory(cc *schedulerserverconfig.CompletedConfig) profile.RecorderFactory {
	return func(name string) events.EventRecorder {
		return cc.EventBroadcaster.NewRecorder(name)
	}
}

func preHandleConfig(cfgPath string) (string, error) {
	tempDir := os.TempDir()
	tempFile, err := os.CreateTemp(tempDir, "kube-scheduler-config")
	if err != nil {
		return "", err
	}
	defer func() {
		_ = tempFile.Close()
	}()
	cfgBytes, err := os.ReadFile(cfgPath)
	if err != nil {
		return "", err
	}
	var cfgRaw map[string]any
	err = yaml.Unmarshal(cfgBytes, &cfgRaw)
	if err != nil {
		return "", err
	}

	// Replace $HOME with actual home directory
	if cfgRaw[clientConnectionCfgKey].(map[string]any)[kubeConfigCfgKey] != "" {
		cfgRaw[clientConnectionCfgKey].(map[string]any)[kubeConfigCfgKey] = strings.ReplaceAll(
			cfgRaw[clientConnectionCfgKey].(map[string]any)[kubeConfigCfgKey].(string),
			"$HOME",
			os.Getenv("HOME"),
		)
	}

	// Replace to KUBECONFIG path if env var exists
	if os.Getenv("KUBECONFIG") != "" {
		cfgRaw[clientConnectionCfgKey].(map[string]any)[kubeConfigCfgKey] = os.Getenv("KUBECONFIG")
	}

	cfgBytes, err = yaml.Marshal(cfgRaw)
	if err != nil {
		return "", err
	}

	if err := os.WriteFile(tempFile.Name(), cfgBytes, 0644); err != nil {
		return "", err
	}
	return tempFile.Name(), nil
}
