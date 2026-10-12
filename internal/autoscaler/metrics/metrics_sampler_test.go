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

	// Usage rows of releases before the hypervisor computed compute_tflops hold
	// a zero figure. The rows with a utilization above zero are dropped by the
	// provider; the ones of an idle interval look like a measurement.
	It("should ignore zero compute figures until the worker reported a real one", func() {
		aggregator := NewWorkerUsageAggregator(24 * time.Hour)
		sampler := NewWorkerUsageSampler()
		start := time.Now().Add(-time.Hour)
		sample := func(minute int, tflops float64) *WorkerUsage {
			return &WorkerUsage{WorkerName: "uid-0", TflopsUsage: tflops, VramUsage: 1 << 30,
				Timestamp: start.Add(time.Duration(minute) * time.Minute)}
		}

		for minute := 0; minute < 10; minute++ {
			Expect(sampler.AddTflopsSample(aggregator, sample(minute, 0))).To(BeFalse())
			sampler.AddVramSample(aggregator, sample(minute, 0))
		}
		Expect(aggregator.TflopsHistogram.IsEmpty()).To(BeTrue(), "no compute sample before the first real figure")
		Expect(aggregator.VramHistogram.IsEmpty()).To(BeFalse(), "memory samples are not affected")
		Expect(sampler.TflopsMeasured).To(BeFalse())

		// An unmeasured interval does not count as the first real figure.
		unknown := sample(10, 0)
		unknown.TflopsUnknown = true
		Expect(sampler.AddTflopsSample(aggregator, unknown)).To(BeFalse())
		Expect(sampler.TflopsMeasured).To(BeFalse())

		Expect(sampler.AddTflopsSample(aggregator, sample(11, 28.5))).To(BeTrue())
		Expect(sampler.TflopsMeasured).To(BeTrue())
		Expect(aggregator.TotalSamplesCount).To(Equal(1))

		// From here on a zero is a measured idle interval.
		Expect(sampler.AddTflopsSample(aggregator, sample(12, 0))).To(BeTrue())
		Expect(aggregator.TotalSamplesCount).To(Equal(2))
	})

	It("should keep the first real compute figure per worker", func() {
		aggregator := NewWorkerUsageAggregator(24 * time.Hour)
		measured, notYet := NewWorkerUsageSampler(), NewWorkerUsageSampler()
		now := time.Now()
		Expect(measured.AddTflopsSample(aggregator, &WorkerUsage{TflopsUsage: 12, Timestamp: now})).To(BeTrue())
		// Another worker of the same workload is still without a real figure.
		Expect(notYet.AddTflopsSample(aggregator, &WorkerUsage{TflopsUsage: 0, Timestamp: now})).To(BeFalse())
		Expect(aggregator.TotalSamplesCount).To(Equal(1))
	})
})
