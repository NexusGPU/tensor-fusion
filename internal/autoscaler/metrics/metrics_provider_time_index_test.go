package metrics

import (
	"context"
	"database/sql/driver"
	"fmt"
	"regexp"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// expectTimeIndexDetection expects the lookup of the usage table's time index
// and answers it with the given column.
func expectTimeIndexDetection(mock sqlmock.Sqlmock, column string) {
	mock.ExpectQuery("information_schema.columns").WithArgs("tf_worker_usage").
		WillReturnRows(sqlmock.NewRows([]string{"column_name"}).AddRow(column))
}

var _ = Describe("MetricsProvider time index", func() {
	ctx := context.Background()
	// 18:04:05.123456789 at UTC+8 is 10:04:05.123 UTC.
	end := time.Date(2026, 3, 9, 18, 4, 5, 123_456_789, time.FixedZone("UTC+8", 8*3600))
	start := end.Add(-time.Minute)
	const startArg, endArg = "2026-03-09T10:03:05.123Z", "2026-03-09T10:04:05.123Z"

	const aggregates = "SELECT namespace, workload, worker, max(compute_tflops) as compute_tflops, max(compute_percentage) as compute_percentage, max(memory_bytes) as memory_bytes, "

	usageRows := func(timeColumn string, at time.Time) *sqlmock.Rows {
		return sqlmock.NewRows([]string{"namespace", "workload", "worker", "compute_tflops", "memory_bytes", timeColumn}).
			AddRow("default", "workload-0", "worker-0", 12.5, 1000, at)
	}

	for _, column := range []string{metrics.DefaultTimeIndexColumn, metrics.LegacyTimeIndexColumn} {
		Context("with the time index named "+column, func() {
			realtimeQuery := fmt.Sprintf(aggregates+"max(%[1]s) as ts FROM `tf_worker_usage` WHERE %[1]s > ? and %[1]s <= ?", column)
			historyQuery := fmt.Sprintf(aggregates+"date_bin('1 minute'::INTERVAL, %[1]s) as time_window FROM `tf_worker_usage` WHERE %[1]s > ? and %[1]s <= ?", column)

			It("queries a workload's realtime usage", func() {
				db, mock := NewMockDB()
				expectTimeIndexDetection(mock, column)
				mock.ExpectQuery(regexp.QuoteMeta(realtimeQuery+" and namespace = ? and workload = ? GROUP BY namespace, workload, worker ORDER BY ts asc")).
					WithArgs(startArg, endArg, "default", "workload-0").
					WillReturnRows(usageRows("ts", end))

				got, err := NewProviderWithDB(db).GetWorkloadRealtimeMetrics(ctx, "default", "workload-0", start, end)
				Expect(err).ToNot(HaveOccurred())
				Expect(got).To(HaveLen(1))
				Expect(got[0].Timestamp.Equal(end)).To(BeTrue())
				Expect(got[0].TflopsUsage).To(Equal(12.5))
				Expect(mock.ExpectationsWereMet()).To(Succeed())
			})

			It("queries a workload's usage history", func() {
				db, mock := NewMockDB()
				expectTimeIndexDetection(mock, column)
				mock.ExpectQuery(regexp.QuoteMeta(historyQuery+" and namespace = ? and workload = ? GROUP BY namespace, workload, worker, time_window ORDER BY time_window asc")).
					WithArgs(startArg, endArg, "default", "workload-0").
					WillReturnRows(usageRows("time_window", start))

				got, err := NewProviderWithDB(db).GetWorkloadHistoryMetrics(ctx, "default", "workload-0", start, end)
				Expect(err).ToNot(HaveOccurred())
				Expect(got).To(HaveLen(1))
				Expect(got[0].Timestamp.Equal(start)).To(BeTrue())
				Expect(mock.ExpectationsWereMet()).To(Succeed())
			})

			It("queries every worker's realtime usage, asking for the time index once", func() {
				db, mock := NewMockDB()
				provider := newGreptimeDBProvider(db)
				expectTimeIndexDetection(mock, column)
				query := regexp.QuoteMeta(realtimeQuery + " GROUP BY namespace, workload, worker ORDER BY ts asc")
				timeArg := regexp.MustCompile(`^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$`)
				mock.ExpectQuery(query).WithArgs(matchingArg{timeArg}, matchingArg{timeArg}).WillReturnRows(usageRows("ts", end))
				mock.ExpectQuery(query).WithArgs(matchingArg{timeArg}, matchingArg{timeArg}).WillReturnRows(usageRows("ts", end))

				got, err := provider.GetWorkersMetrics(ctx)
				Expect(err).ToNot(HaveOccurred())
				Expect(got).To(HaveLen(1))
				_, err = provider.GetWorkersMetrics(ctx)
				Expect(err).ToNot(HaveOccurred())
				Expect(mock.ExpectationsWereMet()).To(Succeed())
			})

			It("queries every worker's usage history", func() {
				db, mock := NewMockDB()
				provider := newGreptimeDBProvider(db)
				expectTimeIndexDetection(mock, column)
				timeArg := regexp.MustCompile(`^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$`)
				mock.ExpectQuery(regexp.QuoteMeta(historyQuery+" GROUP BY namespace, workload, worker, time_window ORDER BY time_window asc")).
					WithArgs(matchingArg{timeArg}, matchingArg{timeArg}).
					WillReturnRows(usageRows("time_window", start))

				got, err := provider.GetHistoryMetrics(ctx)
				Expect(err).ToNot(HaveOccurred())
				Expect(got).To(HaveLen(1))
				Expect(mock.ExpectationsWereMet()).To(Succeed())
			})
		})
	}

	It("names the time index in every part of a usage query", func() {
		for _, column := range []string{metrics.DefaultTimeIndexColumn, metrics.LegacyTimeIndexColumn} {
			parts := usageQueryPartsFor(column)
			Expect(parts.latest).To(HaveSuffix("max(" + column + ") as ts"))
			Expect(parts.perMinute).To(HaveSuffix("date_bin('1 minute'::INTERVAL, " + column + ") as time_window"))
			Expect(parts.timeRange).To(Equal(column + " > ? and " + column + " <= ?"))
		}
	})

	It("falls back to the default time index when it can not be detected", func() {
		db, mock := NewMockDB()
		mock.ExpectQuery("information_schema.columns").WillReturnError(fmt.Errorf("unavailable"))
		mock.ExpectQuery(regexp.QuoteMeta("WHERE greptime_timestamp > ? and greptime_timestamp <= ? and namespace = ? and workload = ?")).
			WillReturnRows(usageRows("ts", end))

		got, err := NewProviderWithDB(db).GetWorkloadRealtimeMetrics(ctx, "default", "workload-0", start, end)
		Expect(err).ToNot(HaveOccurred())
		Expect(got).To(HaveLen(1))
		Expect(mock.ExpectationsWereMet()).To(Succeed())
	})
})

// matchingArg matches a string query argument against a pattern.
type matchingArg struct {
	pattern *regexp.Regexp
}

func (m matchingArg) Match(value driver.Value) bool {
	text, ok := value.(string)
	return ok && m.pattern.MatchString(text)
}
