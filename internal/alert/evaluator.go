package alert

import (
	"bytes"
	"context"
	"database/sql"
	"fmt"
	"math/rand"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
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
	// stopCh is closed by StopEvaluate to wake up every per-rule goroutine
	// started by StartEvaluate. Without it, those goroutines block forever
	// on `<-ticker.C` (the parent context is process-lifetime, not
	// evaluator-lifetime), and every config hot-reload that calls
	// UpdateAlertRules → StopEvaluate → StartEvaluate leaves a fresh
	// batch of evaluator goroutines from the previous rule set behind.
	stopCh chan struct{}

	// dbState remembers whether the last evaluation found the database
	// connected, so that a change is logged once and not on every evaluation
	// of every rule.
	dbState atomic.Int32
}

const (
	dbStateUnknown int32 = iota
	dbStateConnected
	dbStateNotConnected
)

func NewAlertEvaluator(ctx context.Context, db *metrics.TimeSeriesDB, rules []config.AlertRule, alertManagerURL string) *AlertEvaluator {
	return &AlertEvaluator{
		DB:    db,
		Rules: rules,

		alertManagerURL: alertManagerURL,
		ctx:             ctx,
		mu:              sync.Mutex{},
		tickers:         make(map[string]*time.Ticker),
		stopCh:          make(chan struct{}),
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
	// Reset stopCh so the new round of goroutines binds to a fresh signal.
	// (StopEvaluate closed the previous one to release the prior batch.)
	if e.stopCh == nil {
		e.stopCh = make(chan struct{})
	}
	stopCh := e.stopCh

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
				case <-stopCh:
					return
				case <-e.ctx.Done():
					return
				}
			}
		}()
	}
	return nil
}

func (e *AlertEvaluator) StopEvaluate() error {
	// Stop the underlying tickers AND signal every running per-rule
	// goroutine to exit. ticker.Stop() alone leaves goroutines blocked on
	// ticker.C / e.ctx.Done() — and e.ctx is the long-lived parent
	// context, so the goroutine never returns. Closing stopCh releases
	// them. A nil-check on the map handles double-stop gracefully.
	for _, ticker := range e.tickers {
		ticker.Stop()
	}
	e.tickers = make(map[string]*time.Ticker)
	if e.stopCh != nil {
		close(e.stopCh)
		e.stopCh = nil
	}
	return nil
}

var (
	// conditionsAction is a template action that prints .Conditions and nothing
	// else, with or without white space trimming.
	conditionsAction = regexp.MustCompile(`(\{\{-?\s*)\.Conditions(\s*-?\}\})`)
	// tableReference is a table named after FROM or JOIN, optionally qualified
	// with its database and quoted.
	tableReference = regexp.MustCompile("(?i)\\b(?:FROM|JOIN)\\s+`?((?:[A-Za-z_][A-Za-z0-9_]*`?\\.`?)?[A-Za-z_][A-Za-z0-9_]*)`?")
)

// TimeColumnFunc tells the time index column of a table.
type TimeColumnFunc func(table string) string

func defaultTimeColumn(string) string {
	return metrics.DefaultTimeIndexColumn
}

type queryTable struct {
	position int
	name     string
}

// queryTables lists the tables a query template reads, in the order they appear.
func queryTables(query string) []queryTable {
	matches := tableReference.FindAllStringSubmatchIndex(query, -1)
	tables := make([]queryTable, 0, len(matches))
	for _, match := range matches {
		name := strings.ReplaceAll(query[match[2]:match[3]], "`", "")
		if dot := strings.LastIndex(name, "."); dot >= 0 {
			name = name[dot+1:]
		}
		tables = append(tables, queryTable{position: match[0], name: name})
	}
	return tables
}

// renderQueryTemplate renders the SQL query template with rule data.
//
// The time index of a table is not called the same in every database (see
// metrics.TimeIndexResolver), so a template does not name it:
//   - {{ .Conditions }} is the time filter of the evaluation interval on the
//     table whose FROM or JOIN precedes it, so that a query reading two tables
//     filters each of them on its own time index;
//   - {{ conditions "table" }} is that filter for the named table;
//   - {{ .TimeColumn }} and {{ timeColumn "table" }} are the time index column
//     of the first table of the query and of the named table, for templates
//     that select, group or order by it.
func renderQueryTemplate(rule *config.AlertRule, timeColumn TimeColumnFunc) (string, error) {
	if timeColumn == nil {
		timeColumn = defaultTimeColumn
	}
	columnOf := func(table string) string {
		if table == "" {
			return metrics.DefaultTimeIndexColumn
		}
		return timeColumn(table)
	}
	conditionsOf := func(table string) string {
		return fmt.Sprintf("%s >= now() - '%s'::INTERVAL", columnOf(table), rule.EvaluationInterval)
	}

	tables := queryTables(rule.Query)
	firstTable := ""
	if len(tables) > 0 {
		firstTable = tables[0].name
	}

	// Bind each plain {{ .Conditions }} to the table it filters. A table name
	// matched by tableReference is a plain identifier, safe inside the action.
	var query strings.Builder
	last := 0
	for _, match := range conditionsAction.FindAllStringSubmatchIndex(rule.Query, -1) {
		table := firstTable
		for _, candidate := range tables {
			if candidate.position > match[0] {
				break
			}
			table = candidate.name
		}
		query.WriteString(rule.Query[last:match[0]])
		query.WriteString(rule.Query[match[2]:match[3]])
		fmt.Fprintf(&query, "conditions %q", table)
		query.WriteString(rule.Query[match[4]:match[5]])
		last = match[1]
	}
	query.WriteString(rule.Query[last:])

	tmpl, err := template.New("query").Funcs(template.FuncMap{
		"conditions": conditionsOf,
		"timeColumn": columnOf,
	}).Parse(query.String())
	if err != nil {
		return "", fmt.Errorf("failed to parse query template: %w", err)
	}

	var buf bytes.Buffer
	data := map[string]any{
		"Threshold":  rule.Threshold,
		"Conditions": conditionsOf(firstTable),
		"TimeColumn": columnOf(firstTable),
		"Severity":   rule.Severity,
		"Name":       rule.Name,
	}

	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to execute query template: %w", err)
	}

	return buf.String(), nil
}

// evaluate evaluates a rule against the database and sends alerts if conditions are met
//
// The operator starts before the metrics database is reachable and connects in
// the background. Until then there is nothing to query: the rule yields no
// alerts and no error, and the evaluator says so once.
func (e *AlertEvaluator) evaluate(rule *config.AlertRule) ([]config.PostableAlert, error) {
	// Conn, not the gorm methods promoted from e.DB: the connection is set by
	// another goroutine, and is nil before that.
	db := e.DB.Conn()
	if db == nil {
		if e.dbState.Swap(dbStateNotConnected) != dbStateNotConnected {
			log.FromContext(e.ctx).Info("time series database is not connected, alert rules are not evaluated until it is")
		}
		return []config.PostableAlert{}, nil
	}
	if e.dbState.Swap(dbStateConnected) == dbStateNotConnected {
		log.FromContext(e.ctx).Info("time series database is connected, alert rules are evaluated again")
	}
	renderedQuery, err := renderQueryTemplate(rule, func(table string) string {
		return e.DB.TimeIndexColumn(e.ctx, table)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to render query template for rule %s: %w", rule.Name, err)
	}
	rows, err := db.Raw(renderedQuery).Rows()
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

		values := make([]any, len(columns))
		valuePtrs := make([]any, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}
		if err := rows.Scan(valuePtrs...); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}

		rowData := make(map[string]any)
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
