package percentile

import (
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Percentile Recommender", func() {
	It("should return default config when no AutoScalingConfig is set", func() {
		cfg := NewRecommender().getPercentileConfig(nil)
		Expect(cfg).ToNot(BeNil())
		Expect(*cfg).To(Equal(DefaultPercentileConfig))
	})

	It("should parse float fields from AutoSetResources", func() {
		asc := &tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{
				TargetTflopsPercentile:     "0.8",
				LowerBoundTflopsPercentile: "0.1",
				UpperBoundTflopsPercentile: "0.95",
				TargetVramPercentile:       "0.7",
				LowerBoundVramPercentile:   "0.2",
				UpperBoundVramPercentile:   "0.9",
				RequestMarginFraction:      "0.15",
			},
		}
		cfg := NewRecommender().getPercentileConfig(asc)
		Expect(cfg.TargetTflopsPercentile).To(Equal(0.8))
		Expect(cfg.LowerBoundTflopsPercentile).To(Equal(0.1))
		Expect(cfg.UpperBoundTflopsPercentile).To(Equal(0.95))
		Expect(cfg.TargetVramPercentile).To(Equal(0.7))
		Expect(cfg.LowerBoundVramPercentile).To(Equal(0.2))
		Expect(cfg.UpperBoundVramPercentile).To(Equal(0.9))
		Expect(cfg.RequestMarginFraction).To(Equal(0.15))
	})

	It("should ignore invalid float fields and keep defaults", func() {
		asc := &tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{
				TargetTflopsPercentile:     "not-a-float",
				LowerBoundTflopsPercentile: "",
				UpperBoundTflopsPercentile: "0.99",
			},
		}
		cfg := NewRecommender().getPercentileConfig(asc)
		Expect(cfg.TargetTflopsPercentile).To(Equal(DefaultPercentileConfig.TargetTflopsPercentile))
		Expect(cfg.LowerBoundTflopsPercentile).To(Equal(DefaultPercentileConfig.LowerBoundTflopsPercentile))
		Expect(cfg.UpperBoundTflopsPercentile).To(Equal(0.99))
	})

	It("should parse ConfidenceInterval if valid", func() {
		asc := &tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{
				ConfidenceInterval: "30m",
			},
		}
		cfg := NewRecommender().getPercentileConfig(asc)
		Expect(cfg.ConfidenceInterval).To(Equal(30 * time.Minute))
	})

	It("should ignore invalid ConfidenceInterval and keep default", func() {
		asc := &tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{
				ConfidenceInterval: "not-a-duration",
			},
		}
		cfg := NewRecommender().getPercentileConfig(asc)
		Expect(cfg.ConfidenceInterval).To(Equal(DefaultPercentileConfig.ConfidenceInterval))
	})
})
