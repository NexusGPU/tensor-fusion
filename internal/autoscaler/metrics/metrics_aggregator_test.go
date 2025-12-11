package metrics

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("MetricsAggregator", func() {
	It("should return the correct boolean value based on whether the histograms are empty", func() {
		aggregator := NewWorkerUsageAggregator(24 * time.Hour)
		Expect(aggregator.IsEmpty()).To(BeTrue())
		sample := WorkerUsage{
			Namespace:    "test",
			WorkloadName: "test",
			WorkerName:   "test",
			TflopsUsage:  0,
			VramUsage:    0,
			Timestamp:    time.Time{},
		}
		aggregator.AddTflopsSample(&sample)
		Expect(aggregator.IsEmpty()).To(BeFalse())
	})
})
