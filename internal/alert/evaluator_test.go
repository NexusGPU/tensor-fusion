package alert

import (
	"context"
	"database/sql"
	"strconv"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

// Mock data for testing
func setupMockDB(t *testing.T) (*sql.DB, sqlmock.Sqlmock, *metrics.TimeSeriesDB) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)

	gormDB, err := gorm.Open(mysql.New(mysql.Config{
		Conn:                      db,
		SkipInitializeWithVersion: true,
	}), &gorm.Config{})
	require.NoError(t, err)

	tsdb := &metrics.TimeSeriesDB{DB: gormDB}
	return db, mock, tsdb
}

func createTestRule(name string) *Rule {
	return &Rule{
		Name:                name,
		Query:               "SELECT value, instance, job FROM metrics WHERE value > {{.Threshold}} AND {{.Conditions}}",
		Threshold:           80.0,
		EvaluationInterval:  "1m",
		ConsecutiveCount:    1,
		Severity:            "critical",
		Summary:             "High CPU usage on {{.instance}}",
		Description:         "CPU usage is {{.value}}% on instance {{.instance}}",
		RunBookURL:          "https://example.com/runbook",
		AlertTargetInstance: "{{.instance}}",

		testMode: true,
	}
}

func TestAlertEvaluator(t *testing.T) {
	db, mock, tsdb := setupMockDB(t)
	defer func() {
		if err := db.Close(); err != nil {
			t.Error("failed to close db", err)
		}
	}()

	evaluator := newAlertEvaluator(tsdb)

	t.Run("evaluate no results", func(t *testing.T) {
		rule := createTestRule("test-rule")

		// Pre-populate firing alerts
		rule.firingAlerts = make(map[string]*struct {
			alert PostableAlert
			count int
		})

		// Mock empty result set
		mock.ExpectQuery("SELECT value, instance, job FROM metrics").
			WillReturnRows(newAlertQueryRows(0, 0))

		alerts, err := evaluator.evaluate(rule)
		assert.NoError(t, err)
		assert.Empty(t, alerts)

		// Verify that firing alerts were resolved
		assert.Empty(t, rule.firingAlerts)
		assert.NoError(t, mock.ExpectationsWereMet())
	})

	t.Run("evaluate with results", func(t *testing.T) {
		rule := createTestRule("test-rule")

		mock.ExpectQuery("SELECT value, instance, job FROM metrics").
			WillReturnRows(newAlertQueryRows(2, 85.5))

		alerts, err := evaluator.evaluate(rule)
		assert.NoError(t, err)
		assert.Len(t, alerts, 2)

		// Verify that alerts were added to firing alerts
		assert.NotEmpty(t, rule.firingAlerts)
		assert.NoError(t, mock.ExpectationsWereMet())
	})

	t.Run("evaluate consecutive count", func(t *testing.T) {
		rule := createTestRule("test-rule")
		rule.ConsecutiveCount = 3 // Require 3 consecutive alerts

		// First evaluation - should not fire alert yet
		mock.ExpectQuery("SELECT value, instance, job FROM metrics").
			WillReturnRows(newAlertQueryRows(1, 85.5))

		alerts, err := evaluator.evaluate(rule)
		assert.NoError(t, err)
		assert.Len(t, alerts, 0)
		assert.Len(t, rule.firingAlerts, 1)

		mock.ExpectQuery("SELECT value, instance, job FROM metrics").
			WillReturnRows(newAlertQueryRows(1, 85.5))

		alerts, err = evaluator.evaluate(rule)
		assert.NoError(t, err)
		assert.Len(t, alerts, 0)
		assert.Len(t, rule.firingAlerts, 1)

		// Check that count increased
		for _, firingAlert := range rule.firingAlerts {
			assert.Equal(t, 2, firingAlert.count)
		}

		mock.ExpectQuery("SELECT value, instance, job FROM metrics").
			WillReturnRows(newAlertQueryRows(1, 85.5))

		alerts, err = evaluator.evaluate(rule)
		assert.NoError(t, err)
		assert.Len(t, alerts, 1)
		assert.Len(t, rule.firingAlerts, 1)

		// Check that count reached threshold
		for _, firingAlert := range rule.firingAlerts {
			assert.Equal(t, 3, firingAlert.count)
		}

		assert.NoError(t, mock.ExpectationsWereMet())
	})

	t.Run("render query template", func(t *testing.T) {
		rule := createTestRule("test-rule")

		query, err := renderQueryTemplate(rule)
		assert.NoError(t, err)
		assert.Contains(t, query, "80")
		assert.Contains(t, query, "now() - '1m'::INTERVAL")
		assert.Contains(t, query, "value > 80")
	})

	t.Run("render query template invalid template", func(t *testing.T) {
		rule := createTestRule("test-rule")
		rule.Query = "SELECT * FROM metrics WHERE value > {{ .InvalidField | invalid}}"

		_, err := renderQueryTemplate(rule)
		assert.Error(t, err)
	})

	t.Run("process query results empty results", func(t *testing.T) {
		db, mock, tsdb := setupMockDB(t)
		defer func() {
			if err := db.Close(); err != nil {
				t.Error("failed to close db", err)
			}
		}()

		rule := createTestRule("test-rule")
		evaluator := newAlertEvaluator(tsdb)

		// Mock empty result set
		mock.ExpectQuery("SELECT").WillReturnRows(newAlertQueryRows(0, 0))

		sqlRows, err := tsdb.Raw("SELECT value, instance, job FROM metrics").Rows()
		require.NoError(t, err)
		defer func() {
			if err := sqlRows.Close(); err != nil {
				t.Error("failed to close sql rows", err)
			}
		}()

		alerts, err := evaluator.processQueryResults(sqlRows, rule)
		assert.NoError(t, err)
		assert.Empty(t, alerts)
		assert.NoError(t, mock.ExpectationsWereMet())
	})

	t.Run("start evaluate invalid interval", func(t *testing.T) {
		rule := createTestRule("test-rule")
		rule.EvaluationInterval = "invalid-interval"

		evaluator := newAlertEvaluator(nil)
		evaluator.Rules = []Rule{*rule}

		err := evaluator.StartEvaluate()
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "invalid duration")
	})

	t.Run("update alert rules", func(t *testing.T) {
		rule2 := createTestRule("rule2")

		evaluator := newAlertEvaluator(nil)

		// Update with new rules
		err := evaluator.UpdateAlertRules([]Rule{*rule2})
		assert.NoError(t, err)
		assert.Len(t, evaluator.Rules, 1)
		assert.Equal(t, "rule2", evaluator.Rules[0].Name)
	})

	t.Run("stop evaluate", func(t *testing.T) {
		evaluator := newAlertEvaluator(nil)

		// Add a mock ticker
		evaluator.tickers["test"] = time.NewTicker(time.Second)

		err := evaluator.StopEvaluate()
		assert.NoError(t, err)
	})

	// Test edge cases
	t.Run("evaluate database error", func(t *testing.T) {
		db, mock, tsdb := setupMockDB(t)
		defer func() {
			if err := db.Close(); err != nil {
				t.Error("failed to close db", err)
			}
		}()

		rule := createTestRule("test-rule")

		evaluator := newAlertEvaluator(tsdb)

		// Mock database error
		mock.ExpectQuery("SELECT value, instance, job FROM metrics").
			WillReturnError(sql.ErrConnDone)

		alerts, err := evaluator.evaluate(rule)
		assert.Error(t, err)
		assert.Empty(t, alerts)
		assert.Contains(t, err.Error(), "failed to execute query")
		assert.NoError(t, mock.ExpectationsWereMet())
	})
}

func newAlertEvaluator(tsdb *metrics.TimeSeriesDB) *AlertEvaluator {
	return NewAlertEvaluator(context.Background(), tsdb, nil, "http://localhost:9093")
}

func newAlertQueryRows(num int, value float64) *sqlmock.Rows {
	rows := sqlmock.NewRows([]string{"value", "instance", "job"})
	for i := 0; i < num; i++ {
		rows.AddRow(value, "server"+strconv.Itoa(i), "mock")
	}
	return rows
}
