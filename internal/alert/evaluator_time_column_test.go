package alert

import (
	"regexp"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
)

const timeIndexQueryPattern = "information_schema.columns"

func timeColumns(columns map[string]string) TimeColumnFunc {
	return func(table string) string {
		if column, ok := columns[table]; ok {
			return column
		}
		return metrics.DefaultTimeIndexColumn
	}
}

func ruleWithQuery(query string) *config.AlertRule {
	rule := createTestRule("time-column-rule")
	rule.Query = query
	rule.EvaluationInterval = "5m"
	return rule
}

var _ = Describe("alert query time column", func() {
	const gpuUsageQuery = `SELECT node, pool, uuid, avg(compute_percentage) as compute
FROM tf_gpu_usage
WHERE compute_percentage > {{ .Threshold }} AND {{ .Conditions }}
GROUP BY node, pool, uuid`

	// The shape of the shipped EmptyGPU rule: two tables, each filtered by its
	// own {{ .Conditions }}.
	const emptyGPUQuery = `SELECT DISTINCT node
FROM tf_node_metrics
WHERE {{ .Conditions }} AND node NOT IN (
    SELECT DISTINCT node
    FROM tf_worker_usage
    WHERE {{.Conditions}}
)`

	DescribeTable("filters on the time index of the table the rule reads",
		func(column string) {
			query, err := renderQueryTemplate(ruleWithQuery(gpuUsageQuery), timeColumns(map[string]string{"tf_gpu_usage": column}))
			Expect(err).NotTo(HaveOccurred())
			Expect(query).To(ContainSubstring("compute_percentage > 80 AND " + column + " >= now() - '5m'::INTERVAL"))
			if column != metrics.LegacyTimeIndexColumn {
				Expect(query).NotTo(MatchRegexp(`\bts\b`))
			}
		},
		Entry("table created on ingestion", metrics.DefaultTimeIndexColumn),
		Entry("table created by the legacy DDL", metrics.LegacyTimeIndexColumn),
	)

	It("assumes the default time index when it is not told", func() {
		query, err := renderQueryTemplate(ruleWithQuery(gpuUsageQuery), nil)
		Expect(err).NotTo(HaveOccurred())
		Expect(query).To(ContainSubstring("greptime_timestamp >= now() - '5m'::INTERVAL"))
	})

	It("filters each table of a query on its own time index", func() {
		query, err := renderQueryTemplate(ruleWithQuery(emptyGPUQuery), timeColumns(map[string]string{
			"tf_node_metrics": metrics.LegacyTimeIndexColumn,
			"tf_worker_usage": metrics.DefaultTimeIndexColumn,
		}))
		Expect(err).NotTo(HaveOccurred())
		Expect(query).To(MatchRegexp(`(?s)FROM tf_node_metrics\s+WHERE ts >= now\(\) - '5m'::INTERVAL AND node NOT IN`))
		Expect(query).To(MatchRegexp(`(?s)FROM tf_worker_usage\s+WHERE greptime_timestamp >= now\(\) - '5m'::INTERVAL`))
	})

	It("renders both shipped shapes the same way for both column names", func() {
		for _, column := range []string{metrics.DefaultTimeIndexColumn, metrics.LegacyTimeIndexColumn} {
			query, err := renderQueryTemplate(ruleWithQuery(emptyGPUQuery), timeColumns(map[string]string{
				"tf_node_metrics": column,
				"tf_worker_usage": column,
			}))
			Expect(err).NotTo(HaveOccurred())
			Expect(regexp.MustCompile(regexp.QuoteMeta(column+" >= now() - '5m'::INTERVAL")).FindAllString(query, -1)).To(HaveLen(2))
			Expect(query).NotTo(ContainSubstring("{{"))
		}
	})

	It("lets a template name the time index itself", func() {
		rule := ruleWithQuery("SELECT node, max({{ .TimeColumn }}) AS last_seen, max({{ timeColumn \"tf_gpu_usage\" }}) AS last_used " +
			"FROM `public`.`tf_node_metrics` JOIN tf_gpu_usage USING (node) " +
			"WHERE {{ conditions \"tf_node_metrics\" }} AND {{ conditions \"tf_gpu_usage\" }} GROUP BY node")
		query, err := renderQueryTemplate(rule, timeColumns(map[string]string{
			"tf_node_metrics": metrics.LegacyTimeIndexColumn,
			"tf_gpu_usage":    metrics.DefaultTimeIndexColumn,
		}))
		Expect(err).NotTo(HaveOccurred())
		Expect(query).To(ContainSubstring("max(ts) AS last_seen, max(greptime_timestamp) AS last_used"))
		Expect(query).To(ContainSubstring("WHERE ts >= now() - '5m'::INTERVAL AND greptime_timestamp >= now() - '5m'::INTERVAL"))
	})

	It("keeps .Conditions usable inside other template actions", func() {
		rule := ruleWithQuery("SELECT node FROM tf_node_metrics WHERE {{ if .Threshold }}{{ .Conditions }}{{ end }} AND {{- .Conditions -}} ")
		query, err := renderQueryTemplate(rule, timeColumns(map[string]string{"tf_node_metrics": metrics.LegacyTimeIndexColumn}))
		Expect(err).NotTo(HaveOccurred())
		Expect(query).To(Equal("SELECT node FROM tf_node_metrics WHERE ts >= now() - '5m'::INTERVAL ANDts >= now() - '5m'::INTERVAL"))
	})

	It("lists the tables of a query", func() {
		tables := queryTables("SELECT a FROM tf_node_metrics n join `tf_gpu_usage` g ON n.node = g.node WHERE x IN (SELECT y FROM public.tf_worker_usage) AND z IN (SELECT 1 FROM (SELECT 2))")
		names := make([]string, 0, len(tables))
		for _, table := range tables {
			names = append(names, table.name)
		}
		Expect(names).To(Equal([]string{"tf_node_metrics", "tf_gpu_usage", "tf_worker_usage"}))
	})

	Describe("evaluate", func() {
		DescribeTable("asks the database for the time index and queries with it",
			func(column string) {
				mock, tsdb := setupMockDB()
				evaluator := newTestAlertEvaluator(tsdb)
				rule := ruleWithQuery("SELECT value, instance, job FROM tf_gpu_usage WHERE value > {{.Threshold}} AND {{.Conditions}}")

				mock.ExpectQuery(timeIndexQueryPattern).WithArgs("tf_gpu_usage").
					WillReturnRows(mock.NewRows([]string{"column_name"}).AddRow(column))
				expected := "SELECT value, instance, job FROM tf_gpu_usage WHERE value > 80 AND " + column + " >= now() - '5m'::INTERVAL"
				mock.ExpectQuery(regexp.QuoteMeta(expected)).WillReturnRows(newAlertQueryRows(1, 85.5))
				// The second evaluation does not ask again.
				mock.ExpectQuery(regexp.QuoteMeta(expected)).WillReturnRows(newAlertQueryRows(1, 85.5))

				alerts, err := evaluator.evaluate(rule)
				Expect(err).NotTo(HaveOccurred())
				Expect(alerts).To(HaveLen(1))
				_, err = evaluator.evaluate(rule)
				Expect(err).NotTo(HaveOccurred())
				Expect(mock.ExpectationsWereMet()).To(Succeed())
			},
			Entry("table created on ingestion", metrics.DefaultTimeIndexColumn),
			Entry("table created by the legacy DDL", metrics.LegacyTimeIndexColumn),
		)

		It("still evaluates when the time index can not be detected", func() {
			mock, tsdb := setupMockDB()
			evaluator := newTestAlertEvaluator(tsdb)
			rule := ruleWithQuery("SELECT value, instance, job FROM tf_gpu_usage WHERE {{.Conditions}}")

			mock.ExpectQuery(timeIndexQueryPattern).WithArgs("tf_gpu_usage").WillReturnError(errDetection)
			mock.ExpectQuery(regexp.QuoteMeta("WHERE greptime_timestamp >= now() - '5m'::INTERVAL")).
				WillReturnRows(newAlertQueryRows(0, 0))

			_, err := evaluator.evaluate(rule)
			Expect(err).NotTo(HaveOccurred())
			Expect(mock.ExpectationsWereMet()).To(Succeed())
		})
	})
})

var errDetection = errUnavailable("time series database unavailable")

type errUnavailable string

func (e errUnavailable) Error() string { return string(e) }
