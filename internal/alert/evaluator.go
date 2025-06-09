package alert

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// 7 seconds jitter factor for evaluation among different rules
const JITTER_FACTOR = 7000

// AlertEvaluator send new or resolved alerts to alertmanager
// use alertmanager to implement deduplication, notification
// it connect TSDB, evaluate all rules to with its own interval
// for each rule, the query result should contains 'val' field,
// which is the value to compare with threshold, and other fields
// which are used to generate alert labelSet and description
type AlertEvaluator struct {
	ctx context.Context

	DB    *metrics.TimeSeriesDB
	Rules []Rule

	alertManagerURL string
	mu              sync.Mutex
	tickers         map[string]*time.Ticker
}

func NewAlertEvaluator(ctx context.Context, db *metrics.TimeSeriesDB, rules []Rule, alertManagerURL string) *AlertEvaluator {
	return &AlertEvaluator{
		DB:    db,
		Rules: rules,

		alertManagerURL: alertManagerURL,
		ctx:             ctx,
		mu:              sync.Mutex{},
		tickers:         make(map[string]*time.Ticker),
	}
}

func (e *AlertEvaluator) UpdateAlertRules(rules []Rule) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.Rules = rules
	err := e.StopEvaluate()
	if err != nil {
		return err
	}
	err = e.StartEvaluate()
	if err != nil {
		return err
	}
	return nil
}

func (e *AlertEvaluator) StartEvaluate() error {

	for _, rule := range e.Rules {
		interval, err := time.ParseDuration(rule.EvaluationInterval)
		if err != nil {
			log.FromContext(e.ctx).Error(err, "failed to parse evaluation interval", "rule", rule)
			return err
		}
		ticker := time.NewTicker(interval)
		e.tickers[rule.Name] = ticker

		go func() {
			// add a jitter to avoid too many evaluations at the same time
			time.Sleep(time.Duration(rand.Intn(JITTER_FACTOR)) * time.Millisecond)
			for {
				select {
				case <-ticker.C:
					if err := e.evaluate(rule); err != nil {
						log.FromContext(e.ctx).Error(err, "failed to evaluate rule", "rule", rule)
					}
				case <-e.ctx.Done():
					return
				}
			}
		}()
	}
	return nil
}

func (e *AlertEvaluator) StopEvaluate() error {
	// stop all tickers
	for _, ticker := range e.tickers {
		ticker.Stop()
	}
	return nil
}

func (e *AlertEvaluator) evaluate(rule Rule) error {
	var result []struct {
		Value float64
	}
	// need get labels from query result
	err := e.DB.Raw(rule.Query).Scan(&result).Error
	if err != nil {
		return fmt.Errorf("failed to evaluate rule %v: %w", rule, err)
	}

	toSendAlerts := []PostableAlert{}

	// TODO: result > 1 indicates multiple alerts
	if len(result) == 0 {
		// no alert, check if pending alerts need to be resolved
		// resolved if still alarming
		toSendAlerts = append(toSendAlerts, rule.ToPostableAlert(LabelSet{}, LabelSet{}, true))
	} else if result[0].Value >= rule.Threshold {
		// alert
		toSendAlerts = append(toSendAlerts, rule.ToPostableAlert(LabelSet{}, LabelSet{}, false))
	} else {
		// resolved if still alarming
		toSendAlerts = append(toSendAlerts, rule.ToPostableAlert(LabelSet{}, LabelSet{}, true))
	}

	// todo use query ID to record unresolved alerts, when query result is less than threshold, delete it and send resolved

	if len(toSendAlerts) > 0 {
		return SendAlert(e.ctx, e.alertManagerURL, toSendAlerts)
	}
	return nil
}
