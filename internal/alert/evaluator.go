package alert

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"math/rand"
	"sync"
	"text/template"
	"time"

	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// 7 seconds jitter factor for evaluation among different rules
const JITTER_FACTOR = 7000

// AlertEvaluator send new or resolved alerts to alertmanager
// use alertmanager to implement deduplication, notification
// it connect TSDB, evaluate all rules to with its own interval
// for each rule, the query result should contains fields that can be compared with threshold,
// and other fields which are used to generate alert labelSet and description
// The query and description can use Go templates with the rule data and query results
type AlertEvaluator struct {
	ctx context.Context

	DB    *metrics.TimeSeriesDB
	Rules []config.AlertRule

	alertManagerURL string
	mu              sync.Mutex
	tickers         map[string]*time.Ticker
}

func NewAlertEvaluator(ctx context.Context, db *metrics.TimeSeriesDB, rules []config.AlertRule, alertManagerURL string) *AlertEvaluator {
	return &AlertEvaluator{
		DB:    db,
		Rules: rules,

		alertManagerURL: alertManagerURL,
		ctx:             ctx,
		mu:              sync.Mutex{},
		tickers:         make(map[string]*time.Ticker),
	}
}

func (e *AlertEvaluator) UpdateAlertRules(rules []config.AlertRule) error {
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
					if _, err := e.evaluate(&rule); err != nil {
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

// renderQueryTemplate renders the SQL query template with rule data
func renderQueryTemplate(rule *config.AlertRule) (string, error) {
	tmpl, err := template.New("query").Parse(rule.Query)
	if err != nil {
		return "", fmt.Errorf("failed to parse query template: %w", err)
	}

	var buf bytes.Buffer
	data := map[string]interface{}{
		"Threshold":  rule.Threshold,
		"Conditions": fmt.Sprintf("ts >= now() - '%s'::INTERVAL", rule.EvaluationInterval),
		"Severity":   rule.Severity,
		"Name":       rule.Name,
	}

	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to execute query template: %w", err)
	}

	return buf.String(), nil
}

// evaluate evaluates a rule against the database and sends alerts if conditions are met
func (e *AlertEvaluator) evaluate(rule *config.AlertRule) ([]config.PostableAlert, error) {
	if e.DB == nil {
		return []config.PostableAlert{}, nil
	}
	renderedQuery, err := renderQueryTemplate(rule)
	if err != nil {
		return nil, fmt.Errorf("failed to render query template for rule %s: %w", rule.Name, err)
	}
	rows, err := e.DB.Raw(renderedQuery).Rows()
	if err != nil {
		return nil, fmt.Errorf("failed to execute query for rule %s: %w", rule.Name, err)
	}
	defer func() {
		if err := rows.Close(); err != nil {
			log.FromContext(e.ctx).Error(err, "failed to close rows")
		}
	}()

	alerts := []config.PostableAlert{}
	processedAlerts, err := e.processQueryResults(rows, rule)
	if err != nil {
		return nil, fmt.Errorf("failed to process query results for rule %s: %w", rule.Name, err)
	}
	if len(processedAlerts) > 0 {
		alerts = append(alerts, processedAlerts...)
	}

	// Send alerts if rule not in test mode
	if !rule.IsTestMode() {
		return alerts, SendAlert(e.ctx, e.alertManagerURL, alerts)
	}
	return alerts, nil
}

// processQueryResults processes query results and returns alerts for matching conditions
func (e *AlertEvaluator) processQueryResults(rows *sql.Rows, rule *config.AlertRule) ([]config.PostableAlert, error) {
	alerts := []config.PostableAlert{}
	firingAlertSet := map[string]struct{}{}

	// Process each row from the query result
	for rows.Next() {
		columns, err := rows.Columns()
		if err != nil {
			return nil, fmt.Errorf("failed to get columns: %w", err)
		}

		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}
		if err := rows.Scan(valuePtrs...); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		rowData := make(map[string]interface{})
		for i, col := range columns {
			rowData[col] = values[i]
		}

		// Add other general fields
		rowData["Threshold"] = rule.Threshold

		alert, consecutiveCountMatched, hash := rule.AddFiringAlertAndCheckResolved(rowData)
		if alert != nil {
			if _, ok := firingAlertSet[hash]; ok {
				// check if duplicated in previous rows
				continue
			}
			firingAlertSet[hash] = struct{}{}
			if consecutiveCountMatched {
				alerts = append(alerts, *alert)
			}
		}

	}

	// check and remove resolved alerts
	resolvedAlerts := rule.CheckAndRemoveFiringAlerts(firingAlertSet)
	alerts = append(alerts, resolvedAlerts...)

	return alerts, nil
}
