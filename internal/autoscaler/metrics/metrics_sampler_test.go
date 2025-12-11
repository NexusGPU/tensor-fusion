package metrics

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("MetricsSampler", func() {
	It("should update peak vram based on the vram usage size", func() {
		aggregator := NewWorkerUsageAggregator(24 * time.Hour)
		sampler := NewWorkerUsageSampler()
		now := time.Now()
		workerUsage := WorkerUsage{
			Namespace:    "test",
			WorkloadName: "test",
			WorkerName:   "test",
			TflopsUsage:  0,
			VramUsage:    0,
			Timestamp:    now,
		}
		sampler.AddSample(aggregator, &workerUsage)
		Expect(sampler.VramPeak).To(Equal(workerUsage.VramUsage))

		workerUsage = WorkerUsage{
			Namespace:    "test",
			WorkloadName: "test",
			WorkerName:   "test",
			TflopsUsage:  0,
			VramUsage:    10,
			Timestamp:    now.Add(time.Minute),
		}
		sampler.AddSample(aggregator, &workerUsage)
		Expect(sampler.VramPeak).To(Equal(workerUsage.VramUsage))

		workerUsage = WorkerUsage{
			Namespace:    "test",
			WorkloadName: "test",
			WorkerName:   "test",
			TflopsUsage:  0,
			VramUsage:    5,
			Timestamp:    now.Add(2 * time.Minute),
		}
		sampler.AddSample(aggregator, &workerUsage)
		Expect(sampler.VramPeak).ToNot(Equal(workerUsage.VramUsage))
	})
})
