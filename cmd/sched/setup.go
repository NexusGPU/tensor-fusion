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

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/tools/events"
	"k8s.io/component-base/configz"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
	schedulerserverconfig "k8s.io/kubernetes/cmd/kube-scheduler/app/config"
	"k8s.io/kubernetes/cmd/kube-scheduler/app/options"
	"k8s.io/kubernetes/pkg/scheduler"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/latest"
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
)

func SetupScheduler(
	ctx context.Context,
	mgr manager.Manager,
	schedulerConfigPath string,
	outOfTreeRegistryOptions ...app.Option,
) (*schedulerserverconfig.CompletedConfig, *scheduler.Scheduler, error) {
	opts := options.NewOptions()
	schedulerConfigFlag := opts.Flags.FlagSet(schedulerConfigFlagSet).Lookup(schedulerConfigFlag)
	schedulerConfigFlag.Changed = true
	cfgPath, err := preHandleConfig(schedulerConfigPath)
	if err != nil {
		return nil, nil, err
	}
	err = schedulerConfigFlag.Value.Set(cfgPath)
	if err != nil {
		return nil, nil, err
	}
	err = opts.ComponentGlobalsRegistry.Set()
	if err != nil {
		return nil, nil, err
	}

	if cfg, err := latest.Default(); err != nil {
		return nil, nil, err
	} else {
		opts.ComponentConfig = cfg
	}

	if errs := opts.Validate(); len(errs) > 0 {
		return nil, nil, utilerrors.NewAggregate(errs)
	}

	c, err := opts.Config(ctx)
	if err != nil {
		return nil, nil, err
	}

	// Get the completed config
	cc := c.Complete()

	outOfTreeRegistry := make(runtime.Registry)
	for _, option := range outOfTreeRegistryOptions {
		if err := option(outOfTreeRegistry); err != nil {
			return nil, nil, err
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
		return nil, nil, err
	}
	if err := options.LogOrWriteConfig(
		klog.FromContext(ctx),
		opts.WriteConfigTo,
		&cc.ComponentConfig,
		completedProfiles,
	); err != nil {
		return nil, nil, err
	}

	return &cc, sched, nil
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
		<-mgr.Elected()
		logger.Info("Starting scheduling cycle")
		sched.Run(ctx)
		cc.EventBroadcaster.Shutdown()
	}()
	return nil
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
