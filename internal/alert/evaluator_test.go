package alert

import (
	"context"
	"database/sql"
	"strconv"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
)

func setupMockDB() (sqlmock.Sqlmock, *metrics.TimeSeriesDB) {
	db, mock, err := sqlmock.New()
	Expect(err).NotTo(HaveOccurred())

	gormDB, err := gorm.Open(mysql.New(mysql.Config{
		Conn:                      db,
		SkipInitializeWithVersion: true,
	}), &gorm.Config{
		Logger: logger.Default.LogMode(logger.Silent),
	})
	Expect(err).NotTo(HaveOccurred())

	tsdb := &metrics.TimeSeriesDB{DB: gormDB}
	return mock, tsdb
}

func createTestRule(name string) *config.AlertRule {
	return &config.AlertRule{
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
		TestMode:            true,
	}
}

func newTestAlertEvaluator(tsdb *metrics.TimeSeriesDB) *AlertEvaluator {
	return NewAlertEvaluator(context.Background(), tsdb, nil, "http://localhost:9093")
}

func newAlertQueryRows(num int, value float64) *sqlmock.Rows {
	rows := sqlmock.NewRows([]string{"value", "instance", "job"})
	for i := 0; i < num; i++ {
		rows.AddRow(value, "server"+strconv.Itoa(i), "mock")
	}
	return rows
}

var _ = Describe("AlertEvaluator", func() {
	var (
		mock      sqlmock.Sqlmock
		tsdb      *metrics.TimeSeriesDB
		evaluator *AlertEvaluator
	)

	BeforeEach(func() {
		mock, tsdb = setupMockDB()
		evaluator = newTestAlertEvaluator(tsdb)
	})

	Describe("evaluate", func() {
		It("should return empty alerts for no results", func() {
			rule := createTestRule("test-rule")
			rule.FiringAlerts = map[string]*config.FiringAlertCache{}

			mock.ExpectQuery("SELECT value, instance, job FROM metrics").
				WillReturnRows(newAlertQueryRows(0, 0))

			alerts, err := evaluator.evaluate(rule)
			Expect(err).NotTo(HaveOccurred())
			Expect(alerts).To(BeEmpty())
			Expect(rule.FiringAlerts).To(BeEmpty())
			Expect(mock.ExpectationsWereMet()).To(Succeed())
		})

		It("should return alerts for query results", func() {
			rule := createTestRule("test-rule")

			mock.ExpectQuery("SELECT value, instance, job FROM metrics").
				WillReturnRows(newAlertQueryRows(2, 85.5))

			alerts, err := evaluator.evaluate(rule)
			Expect(err).NotTo(HaveOccurred())
			Expect(alerts).To(HaveLen(2))
			Expect(rule.FiringAlerts).NotTo(BeEmpty())
			Expect(mock.ExpectationsWereMet()).To(Succeed())
		})

		It("should respect consecutive count requirement", func() {
			rule := createTestRule("test-rule")
			rule.ConsecutiveCount = 3

			// First evaluation - should not fire alert yet
			mock.ExpectQuery("SELECT value, instance, job FROM metrics").
				WillReturnRows(newAlertQueryRows(1, 85.5))

			alerts, err := evaluator.evaluate(rule)
			Expect(err).NotTo(HaveOccurred())
			Expect(alerts).To(BeEmpty())
			Expect(rule.FiringAlerts).To(HaveLen(1))

			// Second evaluation
			mock.ExpectQuery("SELECT value, instance, job FROM metrics").
				WillReturnRows(newAlertQueryRows(1, 85.5))

			alerts, err = evaluator.evaluate(rule)
			Expect(err).NotTo(HaveOccurred())
			Expect(alerts).To(BeEmpty())
			Expect(rule.FiringAlerts).To(HaveLen(1))

			// Check that count increased
			for _, firingAlert := range rule.FiringAlerts {
				Expect(firingAlert.Count).To(Equal(2))
			}

			// Third evaluation - should fire now
			mock.ExpectQuery("SELECT value, instance, job FROM metrics").
				WillReturnRows(newAlertQueryRows(1, 85.5))

			alerts, err = evaluator.evaluate(rule)
			Expect(err).NotTo(HaveOccurred())
			Expect(alerts).To(HaveLen(1))
			Expect(rule.FiringAlerts).To(HaveLen(1))

			// Check that count reached threshold
			for _, firingAlert := range rule.FiringAlerts {
				Expect(firingAlert.Count).To(Equal(3))
			}

			Expect(mock.ExpectationsWereMet()).To(Succeed())
		})

		It("should return error on database failure", func() {
			rule := createTestRule("test-rule")

			mock.ExpectQuery("SELECT value, instance, job FROM metrics").
				WillReturnError(sql.ErrNoRows)

			alerts, err := evaluator.evaluate(rule)
			Expect(err).To(HaveOccurred())
			Expect(alerts).To(BeEmpty())
			Expect(err.Error()).To(ContainSubstring("failed to execute query"))
			Expect(mock.ExpectationsWereMet()).To(Succeed())
		})
	})

	Describe("renderQueryTemplate", func() {
		It("should render query template correctly", func() {
			rule := createTestRule("test-rule")

			query, err := renderQueryTemplate(rule)
			Expect(err).NotTo(HaveOccurred())
			Expect(query).To(ContainSubstring("80"))
			Expect(query).To(ContainSubstring("now() - '1m'::INTERVAL"))
			Expect(query).To(ContainSubstring("value > 80"))
		})

		It("should return error for invalid template", func() {
			rule := createTestRule("test-rule")
			rule.Query = "SELECT * FROM metrics WHERE value > {{ .InvalidField | invalid}}"

			_, err := renderQueryTemplate(rule)
			Expect(err).To(HaveOccurred())
		})
	})

	Describe("processQueryResults", func() {
		It("should handle empty results", func() {
			mock, tsdb := setupMockDB()
			rule := createTestRule("test-rule")
			evaluator := newTestAlertEvaluator(tsdb)

			mock.ExpectQuery("SELECT").WillReturnRows(newAlertQueryRows(0, 0))

			sqlRows, err := tsdb.Raw("SELECT value, instance, job FROM metrics").Rows()
			Expect(err).NotTo(HaveOccurred())
			defer func() { _ = sqlRows.Close() }()

			alerts, err := evaluator.processQueryResults(sqlRows, rule)
			Expect(err).NotTo(HaveOccurred())
			Expect(alerts).To(BeEmpty())
			Expect(mock.ExpectationsWereMet()).To(Succeed())
		})
	})

	Describe("StartEvaluate", func() {
		It("should return error for invalid interval", func() {
			rule := createTestRule("test-rule")
			rule.EvaluationInterval = "invalid-interval"

			evaluator := newTestAlertEvaluator(nil)
			evaluator.Rules = []config.AlertRule{*rule}

			err := evaluator.StartEvaluate()
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("invalid duration"))
		})
	})

	Describe("UpdateAlertRules", func() {
		It("should update rules correctly", func() {
			rule2 := createTestRule("rule2")

			evaluator := newTestAlertEvaluator(nil)

			err := evaluator.UpdateAlertRules([]config.AlertRule{*rule2})
			Expect(err).NotTo(HaveOccurred())
			Expect(evaluator.Rules).To(HaveLen(1))
			Expect(evaluator.Rules[0].Name).To(Equal("rule2"))
		})
	})

	Describe("StopEvaluate", func() {
		It("should stop tickers correctly", func() {
			evaluator := newTestAlertEvaluator(nil)
			evaluator.tickers["test"] = time.NewTicker(time.Second)

			err := evaluator.StopEvaluate()
			Expect(err).NotTo(HaveOccurred())
		})
	})
})
