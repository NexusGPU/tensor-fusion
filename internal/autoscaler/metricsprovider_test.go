package autoscaler

import (
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
	Context("when getting real time workers metrics", func() {
		It("should return slices", func() {
			db, mock := NewMockDB()
			now := time.Now()
			fakeMetrics := []metrics.HypervisorWorkerUsageMetrics{
				{
					WorkloadName:  "workload-0",
					WorkerName:    "worker-0",
					ComputeTflops: 10.3,
					VRAMBytes:     1 * 1000 * 1000 * 1000,
					Timestamp:     now,
				},
				{
					WorkloadName:  "workload-1",
					WorkerName:    "worker-1",
					ComputeTflops: 10.3,
					VRAMBytes:     1 * 1000 * 1000 * 1000,
					Timestamp:     now,
				},
			}

			rows := sqlmock.NewRows([]string{"workload", "worker", "compute_tflops", "memory_bytes", "ts"})
			for _, row := range fakeMetrics {
				rows.AddRow(row.WorkloadName, row.WorkerName, row.ComputeTflops, row.VRAMBytes, row.Timestamp)
			}

			mock.ExpectQuery(regexp.QuoteMeta("SELECT workload, worker, max(compute_tflops) as compute_tflops, max(memory_bytes) as memory_bytes, max(ts) as ts FROM `tf_worker_usage` WHERE ts > ? GROUP BY workload, worker")).
				WillReturnRows(rows)
			provider := &greptimeDBProvider{db: db}
			got, _ := provider.GetWorkersMetrics()
			Expect(got).To(HaveLen(2))
			Expect(got[0].WorkloadName).To(Equal(fakeMetrics[0].WorkloadName))
			Expect(got[0].WorkerName).To(Equal(fakeMetrics[0].WorkerName))
			Expect(got[0].VramUsage).To(Equal(ResourceAmount(fakeMetrics[0].VRAMBytes)))
			Expect(got[0].TflopsUsage).To(Equal(resourceAmountFromFloat(fakeMetrics[0].ComputeTflops)))
			Expect(got[0].Timestamp).To(Equal(fakeMetrics[0].Timestamp))
		})
	})

	Context("when getting history workers metrics", func() {
		FIt("should return slices", func() {
			db, mock := NewMockDB()
			now := time.Now()
			fakeMetrics := []hypervisorWorkerUsageMetrics{
				{
					HypervisorWorkerUsageMetrics: metrics.HypervisorWorkerUsageMetrics{
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
						WorkloadName:  "workload-1",
						WorkerName:    "worker-1",
						ComputeTflops: 10.3,
						VRAMBytes:     1 * 1000 * 1000 * 1000,
						Timestamp:     now,
					},
					TimeWindow: now,
				},
			}

			rows := sqlmock.NewRows([]string{"workload", "worker", "compute_tflops", "memory_bytes", "time_window"})
			for _, row := range fakeMetrics {
				rows.AddRow(row.WorkloadName, row.WorkerName, row.ComputeTflops, row.VRAMBytes, row.TimeWindow)
			}

			mock.ExpectQuery(regexp.QuoteMeta("SELECT workload, worker, max(compute_tflops) as compute_tflops, max(memory_bytes) as memory_bytes, date_bin('1 minute'::INTERVAL, ts) as time_window FROM `tf_worker_usage` WHERE ts > ? and ts <= ? GROUP BY workload, worker, time_window ORDER BY time_window asc")).
				WillReturnRows(rows)
			provider := &greptimeDBProvider{db: db}
			got, _ := provider.GetHistoryMetrics()
			Expect(got).To(HaveLen(2))
			Expect(got[0].WorkloadName).To(Equal(fakeMetrics[0].WorkloadName))
			Expect(got[0].WorkerName).To(Equal(fakeMetrics[0].WorkerName))
			Expect(got[0].VramUsage).To(Equal(ResourceAmount(fakeMetrics[0].VRAMBytes)))
			Expect(got[0].TflopsUsage).To(Equal(resourceAmountFromFloat(fakeMetrics[0].ComputeTflops)))
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
