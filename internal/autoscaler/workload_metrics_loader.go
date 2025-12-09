package autoscaler

import (
	"context"
	"fmt"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/metrics"
	"github.com/NexusGPU/tensor-fusion/internal/autoscaler/workload"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	maxHistoryDataPeriod = 30 * 24 * time.Hour // 30 days
)

type workloadMetricsLoader struct {
	client          client.Client
	metricsProvider metrics.Provider
	workloads       map[WorkloadID]*workloadMetricsState
	mu              sync.RWMutex
}

type workloadMetricsState struct {
	workloadID         WorkloadID
	state              *workload.State
	initialDelay       time.Duration
	evaluationInterval time.Duration
	historyDataPeriod  time.Duration
	initialDelayTimer  *time.Timer
	ticker             *time.Ticker
	ctx                context.Context
	cancel             context.CancelFunc
	firstLoad          bool
	lastQueryTime      time.Time
}

func newWorkloadMetricsLoader(client client.Client, metricsProvider metrics.Provider) *workloadMetricsLoader {
	return &workloadMetricsLoader{
		client:          client,
		metricsProvider: metricsProvider,
		workloads:       make(map[WorkloadID]*workloadMetricsState),
	}
}

func (l *workloadMetricsLoader) addWorkload(ctx context.Context, workloadID WorkloadID, state *workload.State) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if _, exists := l.workloads[workloadID]; exists {
		return
	}

	// Get configuration
	asr := state.Spec.AutoScalingConfig.AutoSetResources
	if asr == nil || !asr.Enable {
		return
	}

	// Parse durations
	initialDelay, _ := parseDurationOrDefault(asr.InitialDelayPeriod, 30*time.Minute)
	evaluationInterval, _ := parseDurationOrDefault(asr.Interval, getDefaultEvaluationInterval())
	historyDataPeriod, _ := parseDurationOrDefault(asr.HistoryDataPeriod, 2*time.Hour)

	// Enforce 30-day max on HistoryDataPeriod
	if historyDataPeriod > maxHistoryDataPeriod {
		log.FromContext(ctx).Info("HistoryDataPeriod exceeds 30 days, limiting to 30 days",
			"workload", workloadID.Name, "requested", historyDataPeriod, "limited", maxHistoryDataPeriod)
		historyDataPeriod = maxHistoryDataPeriod

		// Record warning event
		workloadObj := &tfv1.TensorFusionWorkload{}
		workloadObj.Namespace = workloadID.Namespace
		workloadObj.Name = workloadID.Name
		workloadObj.Kind = "TensorFusionWorkload"
		workloadObj.APIVersion = tfv1.GroupVersion.String()
		// Note: Event recording would need event recorder, but we'll log for now
	}

	loaderCtx, cancel := context.WithCancel(ctx)

	loaderState := &workloadMetricsState{
		workloadID:         workloadID,
		state:              state,
		initialDelay:       initialDelay,
		evaluationInterval: evaluationInterval,
		historyDataPeriod:  historyDataPeriod,
		ctx:                loaderCtx,
		cancel:             cancel,
		firstLoad:          true,
	}

	// Set timer for initial delay
	timeSinceCreation := time.Since(state.CreationTimestamp.Time)
	if timeSinceCreation < initialDelay {
		remainingDelay := initialDelay - timeSinceCreation
		loaderState.initialDelayTimer = time.AfterFunc(remainingDelay, func() {
			l.startWorkloadMetricsLoading(loaderState)
		})
	} else {
		// Already past initial delay, start immediately
		go l.startWorkloadMetricsLoading(loaderState)
	}

	l.workloads[workloadID] = loaderState
}

func (l *workloadMetricsLoader) removeWorkload(workloadID WorkloadID) {
	l.mu.Lock()
	defer l.mu.Unlock()

	if loaderState, exists := l.workloads[workloadID]; exists {
		if loaderState.initialDelayTimer != nil {
			loaderState.initialDelayTimer.Stop()
		}
		if loaderState.ticker != nil {
			loaderState.ticker.Stop()
		}
		loaderState.cancel()
		delete(l.workloads, workloadID)
	}
}

func (l *workloadMetricsLoader) startWorkloadMetricsLoading(loaderState *workloadMetricsState) {
	logger := log.FromContext(loaderState.ctx)
	logger.Info("Starting metrics loading for workload",
		"workload", loaderState.workloadID.Name,
		"firstLoad", loaderState.firstLoad)

	// First load: load history
	if loaderState.firstLoad {
		if err := l.loadHistoryMetricsForWorkload(loaderState); err != nil {
			logger.Error(err, "failed to load history metrics", "workload", loaderState.workloadID.Name)
		}
		loaderState.firstLoad = false
	}

	// Set up ticker for periodic realtime metrics
	loaderState.ticker = time.NewTicker(loaderState.evaluationInterval)
	go func() {
		for {
			select {
			case <-loaderState.ticker.C:
				if err := l.loadRealtimeMetricsForWorkload(loaderState); err != nil {
					logger.Error(err, "failed to load realtime metrics", "workload", loaderState.workloadID.Name)
				}
			case <-loaderState.ctx.Done():
				return
			}
		}
	}()
}

func (l *workloadMetricsLoader) loadHistoryMetricsForWorkload(loaderState *workloadMetricsState) error {
	now := time.Now()
	startTime := now.Add(-loaderState.historyDataPeriod)

	// Use parameterized query with HistoryDataPeriod
	queryCtx, cancel := context.WithTimeout(loaderState.ctx, 60*time.Second)
	defer cancel()

	// Query metrics for this specific workload
	metricsList, err := l.metricsProvider.GetWorkloadHistoryMetrics(queryCtx,
		loaderState.workloadID.Namespace,
		loaderState.workloadID.Name,
		startTime,
		now)
	if err != nil {
		return fmt.Errorf("failed to get workload history metrics: %w", err)
	}

	// Add samples to workload state
	for _, sample := range metricsList {
		loaderState.state.AddSample(sample)
	}

	loaderState.lastQueryTime = now
	return nil
}

func (l *workloadMetricsLoader) loadRealtimeMetricsForWorkload(loaderState *workloadMetricsState) error {
	now := time.Now()
	startTime := loaderState.lastQueryTime
	if startTime.IsZero() {
		startTime = now.Add(-loaderState.evaluationInterval)
	}

	queryCtx, cancel := context.WithTimeout(loaderState.ctx, 15*time.Second)
	defer cancel()

	// Query realtime metrics for this specific workload
	metricsList, err := l.metricsProvider.GetWorkloadRealtimeMetrics(queryCtx,
		loaderState.workloadID.Namespace,
		loaderState.workloadID.Name,
		startTime,
		now)
	if err != nil {
		return fmt.Errorf("failed to get workload realtime metrics: %w", err)
	}

	// Add samples to workload state
	for _, sample := range metricsList {
		loaderState.state.AddSample(sample)
	}

	loaderState.lastQueryTime = now
	return nil
}

func parseDurationOrDefault(durationStr string, defaultDuration time.Duration) (time.Duration, error) {
	if durationStr == "" {
		return defaultDuration, nil
	}
	return time.ParseDuration(durationStr)
}

func getDefaultEvaluationInterval() time.Duration {
	intervalStr := config.GetGlobalConfig().AutoScalingInterval
	if intervalStr == "" {
		return 30 * time.Second
	}
	interval, err := time.ParseDuration(intervalStr)
	if err != nil {
		return 30 * time.Second
	}
	return interval
}
