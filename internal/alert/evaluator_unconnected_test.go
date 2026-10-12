package alert

import (
	"context"
	"sync"

	"github.com/go-logr/logr/funcr"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/NexusGPU/tensor-fusion/internal/metrics"
)

// The operator hands the evaluator a database handle that connects in the
// background. Rules are evaluated on their timers from the start.
var _ = Describe("AlertEvaluator while the database is not connected", func() {
	It("yields no alerts, says so once, and evaluates again once connected", func() {
		var mu sync.Mutex
		var lines []string
		logged := func() []string {
			mu.Lock()
			defer mu.Unlock()
			return append([]string(nil), lines...)
		}
		ctx := log.IntoContext(context.Background(), funcr.New(func(_, args string) {
			mu.Lock()
			defer mu.Unlock()
			lines = append(lines, args)
		}, funcr.Options{}))

		tsdb := &metrics.TimeSeriesDB{}
		evaluator := NewAlertEvaluator(ctx, tsdb, nil, "http://localhost:9093")
		rule := createTestRule("unconnected-rule")
		timeColumnRule := ruleWithQuery("SELECT value, instance, job FROM tf_gpu_usage WHERE {{.Conditions}}")

		for range 3 {
			alerts, err := evaluator.evaluate(rule)
			Expect(err).NotTo(HaveOccurred())
			Expect(alerts).To(BeEmpty())
			alerts, err = evaluator.evaluate(timeColumnRule)
			Expect(err).NotTo(HaveOccurred())
			Expect(alerts).To(BeEmpty())
		}
		Expect(logged()).To(HaveLen(1), "one line for the state, not one per evaluation")
		Expect(logged()[0]).To(ContainSubstring("not connected"))

		mock, connected := setupMockDB()
		tsdb.SetConn(connected.DB)
		for range 2 {
			mock.ExpectQuery("SELECT value, instance, job FROM metrics").
				WillReturnRows(newAlertQueryRows(2, 85.5))
			alerts, err := evaluator.evaluate(rule)
			Expect(err).NotTo(HaveOccurred())
			Expect(alerts).To(HaveLen(2), "the same evaluator uses the connection once it is there")
		}
		Expect(mock.ExpectationsWereMet()).To(Succeed())
		Expect(logged()).To(HaveLen(2))
		Expect(logged()[1]).To(ContainSubstring("is connected"))
	})

	It("stays quiet without a database handle at all", func() {
		evaluator := NewAlertEvaluator(context.Background(), nil, nil, "http://localhost:9093")
		alerts, err := evaluator.evaluate(createTestRule("no-database-rule"))
		Expect(err).NotTo(HaveOccurred())
		Expect(alerts).To(BeEmpty())
	})
})
