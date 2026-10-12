package metrics

import (
	"context"
	"fmt"
	"regexp"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/NexusGPU/tensor-fusion/internal/metrics"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gorm.io/driver/mysql"
	"gorm.io/gorm"
)

var _ = Describe("MetricsProvider", func() {
	ctx := context.Background()
	Context("when getting real time workers metrics", func() {
		It("should return metrics for every worker", func() {
			db, mock := NewMockDB()
			now := time.Now()
			fakeMetrics := []metrics.HypervisorWorkerUsageMetrics{
				{
					Namespace:     "default",
					WorkloadName:  "workload-0",
					WorkerName:    "worker-0",
					ComputeTflops: 10.3,
					VRAMBytes:     1 * 1000 * 1000 * 1000,
					Timestamp:     now,
				},
				{
					Namespace:     "default",
					WorkloadName:  "workload-1",
					WorkerName:    "worker-1",
					ComputeTflops: 10.3,
					VRAMBytes:     1 * 1000 * 1000 * 1000,
					Timestamp:     now,
				},
			}

			rows := sqlmock.NewRows([]string{"namespace", "workload", "worker", "compute_tflops", "memory_bytes", "ts"})
			for _, row := range fakeMetrics {
				rows.AddRow(row.Namespace, row.WorkloadName, row.WorkerName, row.ComputeTflops, row.VRAMBytes, row.Timestamp)
			}

			mock.ExpectQuery(regexp.QuoteMeta("SELECT namespace, workload, worker, max(compute_tflops) as compute_tflops, max(compute_percentage) as compute_percentage, max(memory_bytes) as memory_bytes, max(greptime_timestamp) as ts FROM `tf_worker_usage` WHERE greptime_timestamp > ? and greptime_timestamp <= ? GROUP BY namespace, workload, worker ORDER BY ts asc")).
				WillReturnRows(rows)
			provider := &greptimeDBProvider{db: db}
			got, _ := provider.GetWorkersMetrics(ctx)
			Expect(got).To(HaveLen(2))
			Expect(got[0].Namespace).To(Equal(fakeMetrics[0].Namespace))
			Expect(got[0].WorkloadName).To(Equal(fakeMetrics[0].WorkloadName))
			Expect(got[0].WorkerName).To(Equal(fakeMetrics[0].WorkerName))
			Expect(got[0].VramUsage).To(Equal(fakeMetrics[0].VRAMBytes))
			Expect(got[0].TflopsUsage).To(Equal(fakeMetrics[0].ComputeTflops))
			Expect(got[0].Timestamp).To(Equal(fakeMetrics[0].Timestamp))
		})
	})

	Context("when usage rows carry no compute figure", func() {
		It("should report the compute usage as unknown instead of zero", func() {
			db, mock := NewMockDB()
			now := time.Now()
			rows := sqlmock.NewRows([]string{"namespace", "workload", "worker", "compute_tflops", "memory_bytes", "ts"}).
				AddRow("default", "workload-0", "uid-measured", 28.48, 6*1000*1000*1000, now).
				AddRow("default", "workload-0", "uid-idle", 0.0, 1*1000*1000*1000, now).
				// max(compute_tflops) over rows that all lack the field is NULL.
				AddRow("default", "workload-0", "uid-unmeasured", nil, 2*1000*1000*1000, now)
			expectTimeIndexDetection(mock, metrics.DefaultTimeIndexColumn)
			mock.ExpectQuery(regexp.QuoteMeta("SELECT namespace, workload, worker, max(compute_tflops) as compute_tflops, max(compute_percentage) as compute_percentage, max(memory_bytes) as memory_bytes, max(greptime_timestamp) as ts FROM `tf_worker_usage` WHERE greptime_timestamp > ? and greptime_timestamp <= ? and namespace = ? and workload = ? GROUP BY namespace, workload, worker ORDER BY ts asc")).
				WillReturnRows(rows)

			got, err := NewProviderWithDB(db).GetWorkloadRealtimeMetrics(ctx, "default", "workload-0", now.Add(-time.Minute), now)
			Expect(err).ToNot(HaveOccurred())
			Expect(got).To(HaveLen(3))
			Expect(got[0].TflopsUnknown).To(BeFalse())
			Expect(got[0].TflopsUsage).To(Equal(28.48))
			Expect(got[1].TflopsUnknown).To(BeFalse(), "a measured zero is a sample")
			Expect(got[1].TflopsUsage).To(BeZero())
			Expect(got[2].TflopsUnknown).To(BeTrue())
			Expect(got[2].VramUsage).To(Equal(uint64(2 * 1000 * 1000 * 1000)))

			// Only measured compute usage is sampled; memory always is. The zero of
			// a worker that has not reported a figure above zero yet is not sampled
			// either (see the sampler's tests).
			aggregator := NewWorkerUsageAggregator(time.Hour)
			for _, sample := range got {
				NewWorkerUsageSampler().AddSample(aggregator, sample)
			}
			Expect(aggregator.TotalSamplesCount).To(Equal(1))
			Expect(aggregator.VramHistogram.IsEmpty()).To(BeFalse())
		})
	})

	Context("when usage rows were written by a release that did not compute the figure", func() {
		const realtimeQuery = "SELECT namespace, workload, worker, max(compute_tflops) as compute_tflops, max(compute_percentage) as compute_percentage, max(memory_bytes) as memory_bytes, max(greptime_timestamp) as ts FROM `tf_worker_usage` WHERE greptime_timestamp > ? and greptime_timestamp <= ? and namespace = ? and workload = ? GROUP BY namespace, workload, worker ORDER BY ts asc"
		const historyQuery = "SELECT namespace, workload, worker, max(compute_tflops) as compute_tflops, max(compute_percentage) as compute_percentage, max(memory_bytes) as memory_bytes, date_bin('1 minute'::INTERVAL, greptime_timestamp) as time_window FROM `tf_worker_usage` WHERE greptime_timestamp > ? and greptime_timestamp <= ? and namespace = ? and workload = ? GROUP BY namespace, workload, worker, time_window ORDER BY time_window asc"

		// Those rows hold compute_tflops = 0 next to the real compute_percentage.
		usageRows := func(timeColumn string, at time.Time) *sqlmock.Rows {
			return sqlmock.NewRows([]string{"namespace", "workload", "worker", "compute_tflops", "compute_percentage", "memory_bytes", timeColumn}).
				AddRow("default", "workload-0", "uid-earlier-release-busy", 0.0, 37.5, 1000, at).
				AddRow("default", "workload-0", "uid-zero-and-idle", 0.0, 0.0, 1000, at).
				AddRow("default", "workload-0", "uid-zero-no-percentage", 0.0, nil, 1000, at).
				AddRow("default", "workload-0", "uid-no-figure", nil, 50.0, 1000, at).
				AddRow("default", "workload-0", "uid-measured", 26.7, 37.5, 1000, at)
		}
		expectRule := func(got []*WorkerUsage) {
			GinkgoHelper()
			Expect(got).To(HaveLen(5))
			Expect(got[0].TflopsUnknown).To(BeTrue(), "a zero figure next to a utilization above zero is not a measurement")
			Expect(got[1].TflopsUnknown).To(BeFalse(), "zero figure, zero utilization: left to the sampler")
			Expect(got[2].TflopsUnknown).To(BeFalse(), "a missing utilization does not make the figure unknown")
			Expect(got[3].TflopsUnknown).To(BeTrue(), "no figure at all")
			Expect(got[4].TflopsUnknown).To(BeFalse())
			Expect(got[4].TflopsUsage).To(Equal(26.7))
			for _, usage := range got {
				Expect(usage.VramUsage).To(Equal(uint64(1000)), "memory usage is read regardless")
			}
		}

		It("applies the rule to the realtime query", func() {
			db, mock := NewMockDB()
			now := time.Now()
			expectTimeIndexDetection(mock, metrics.DefaultTimeIndexColumn)
			mock.ExpectQuery(regexp.QuoteMeta(realtimeQuery)).WillReturnRows(usageRows("ts", now))
			got, err := NewProviderWithDB(db).GetWorkloadRealtimeMetrics(ctx, "default", "workload-0", now.Add(-time.Minute), now)
			Expect(err).ToNot(HaveOccurred())
			expectRule(got)
			Expect(mock.ExpectationsWereMet()).To(Succeed())
		})

		It("applies the rule to the history query", func() {
			db, mock := NewMockDB()
			now := time.Now()
			expectTimeIndexDetection(mock, metrics.DefaultTimeIndexColumn)
			mock.ExpectQuery(regexp.QuoteMeta(historyQuery)).WillReturnRows(usageRows("time_window", now))
			got, err := NewProviderWithDB(db).GetWorkloadHistoryMetrics(ctx, "default", "workload-0", now.Add(-time.Hour), now)
			Expect(err).ToNot(HaveOccurred())
			expectRule(got)
			Expect(mock.ExpectationsWereMet()).To(Succeed())
		})
	})

	Context("when the usage table has no compute columns", func() {
		const tail = "max(memory_bytes) as memory_bytes, max(greptime_timestamp) as ts FROM `tf_worker_usage` WHERE greptime_timestamp > ? and greptime_timestamp <= ? and namespace = ? and workload = ? GROUP BY namespace, workload, worker ORDER BY ts asc"
		const full = "SELECT namespace, workload, worker, max(compute_tflops) as compute_tflops, max(compute_percentage) as compute_percentage, " + tail
		const withoutPercentage = "SELECT namespace, workload, worker, max(compute_tflops) as compute_tflops, " + tail
		const memoryOnly = "SELECT namespace, workload, worker, " + tail

		It("reads the figure without the utilization column", func() {
			db, mock := NewMockDB()
			now := time.Now()
			expectTimeIndexDetection(mock, metrics.DefaultTimeIndexColumn)
			mock.ExpectQuery(regexp.QuoteMeta(full)).WillReturnError(fmt.Errorf("column compute_percentage not found"))
			mock.ExpectQuery(regexp.QuoteMeta(withoutPercentage)).WillReturnRows(
				sqlmock.NewRows([]string{"namespace", "workload", "worker", "compute_tflops", "memory_bytes", "ts"}).
					AddRow("default", "workload-0", "uid-0", 12.5, 1000, now))
			got, err := NewProviderWithDB(db).GetWorkloadRealtimeMetrics(ctx, "default", "workload-0", now.Add(-time.Minute), now)
			Expect(err).ToNot(HaveOccurred())
			Expect(got).To(HaveLen(1))
			Expect(got[0].TflopsUnknown).To(BeFalse())
			Expect(got[0].TflopsUsage).To(Equal(12.5))
			Expect(mock.ExpectationsWereMet()).To(Succeed())
		})

		It("still reads memory usage, with the compute usage unknown", func() {
			db, mock := NewMockDB()
			now := time.Now()
			expectTimeIndexDetection(mock, metrics.DefaultTimeIndexColumn)
			mock.ExpectQuery(regexp.QuoteMeta(full)).WillReturnError(fmt.Errorf("column compute_tflops not found"))
			mock.ExpectQuery(regexp.QuoteMeta(withoutPercentage)).WillReturnError(fmt.Errorf("column compute_tflops not found"))
			mock.ExpectQuery(regexp.QuoteMeta(memoryOnly)).WillReturnRows(
				sqlmock.NewRows([]string{"namespace", "workload", "worker", "memory_bytes", "ts"}).
					AddRow("default", "workload-0", "uid-0", 1000, now))
			got, err := NewProviderWithDB(db).GetWorkloadRealtimeMetrics(ctx, "default", "workload-0", now.Add(-time.Minute), now)
			Expect(err).ToNot(HaveOccurred())
			Expect(got).To(HaveLen(1))
			Expect(got[0].TflopsUnknown).To(BeTrue())
			Expect(got[0].VramUsage).To(Equal(uint64(1000)))
			Expect(mock.ExpectationsWereMet()).To(Succeed())
		})

		It("returns the error of the full query when no variant works", func() {
			db, mock := NewMockDB()
			now := time.Now()
			expectTimeIndexDetection(mock, metrics.DefaultTimeIndexColumn)
			mock.ExpectQuery(regexp.QuoteMeta(full)).WillReturnError(fmt.Errorf("database unavailable"))
			mock.ExpectQuery(regexp.QuoteMeta(withoutPercentage)).WillReturnError(fmt.Errorf("second"))
			mock.ExpectQuery(regexp.QuoteMeta(memoryOnly)).WillReturnError(fmt.Errorf("third"))
			_, err := NewProviderWithDB(db).GetWorkloadRealtimeMetrics(ctx, "default", "workload-0", now.Add(-time.Minute), now)
			Expect(err).To(MatchError(ContainSubstring("database unavailable")))
			Expect(mock.ExpectationsWereMet()).To(Succeed())
		})
	})

	Context("when getting history workers metrics", func() {
		It("should return metrics based on history length", func() {
			db, mock := NewMockDB()
			now := time.Now()
			fakeMetrics := []hypervisorWorkerUsageMetrics{
				{
					HypervisorWorkerUsageMetrics: metrics.HypervisorWorkerUsageMetrics{
						Namespace:     "default",
						WorkloadName:  "workload-0",
						WorkerName:    "worker-0",
						ComputeTflops: 10.3,
						VRAMBytes:     1 * 1000 * 1000 * 1000,
						Timestamp:     now,
					},
					TimeWindow: now,
				},
				{
					HypervisorWorkerUsageMetrics: metrics.HypervisorWorkerUsageMetrics{
						Namespace:     "default",
						WorkloadName:  "workload-1",
						WorkerName:    "worker-1",
						ComputeTflops: 10.3,
						VRAMBytes:     1 * 1000 * 1000 * 1000,
						Timestamp:     now,
					},
					TimeWindow: now,
				},
			}

			rows := sqlmock.NewRows([]string{"namespace", "workload", "worker", "compute_tflops", "memory_bytes", "time_window"})
			for _, row := range fakeMetrics {
				rows.AddRow(row.Namespace, row.WorkloadName, row.WorkerName, row.ComputeTflops, row.VRAMBytes, row.TimeWindow)
			}

			mock.ExpectQuery(regexp.QuoteMeta("SELECT namespace, workload, worker, max(compute_tflops) as compute_tflops, max(compute_percentage) as compute_percentage, max(memory_bytes) as memory_bytes, date_bin('1 minute'::INTERVAL, greptime_timestamp) as time_window FROM `tf_worker_usage` WHERE greptime_timestamp > ? and greptime_timestamp <= ? GROUP BY namespace, workload, worker, time_window ORDER BY time_window asc")).
				WillReturnRows(rows)
			provider := &greptimeDBProvider{db: db}
			got, _ := provider.GetHistoryMetrics(ctx)
			Expect(got).To(HaveLen(2))
			Expect(got[0].Namespace).To(Equal(fakeMetrics[0].Namespace))
			Expect(got[0].WorkloadName).To(Equal(fakeMetrics[0].WorkloadName))
			Expect(got[0].WorkerName).To(Equal(fakeMetrics[0].WorkerName))
			Expect(got[0].VramUsage).To(Equal(fakeMetrics[0].VRAMBytes))
			Expect(got[0].TflopsUsage).To(Equal(fakeMetrics[0].ComputeTflops))
			Expect(got[0].Timestamp).To(Equal(fakeMetrics[0].TimeWindow))
		})
	})
})

func NewMockDB() (*gorm.DB, sqlmock.Sqlmock) {
	GinkgoHelper()
	db, mock, err := sqlmock.New()
	Expect(err).ToNot(HaveOccurred())
	gormDB, err := gorm.Open(mysql.New(mysql.Config{
		Conn:                      db,
		SkipInitializeWithVersion: true,
	}), &gorm.Config{})
	Expect(err).ToNot(HaveOccurred())

	return gormDB, mock
}
