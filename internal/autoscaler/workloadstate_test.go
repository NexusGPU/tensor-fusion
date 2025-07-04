package autoscaler

import (
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Workload State", func() {
	It("should return default config when no AutoScalingConfig is set", func() {
		ws := NewWorkloadState("test")
		cfg := ws.GetResourceRecommenderConfig()
		Expect(cfg).ToNot(BeNil())
		Expect(*cfg).To(Equal(DefaultResourceRecommenderConfig))
	})

	It("should parse float fields from AutoSetResources", func() {
		ws := NewWorkloadState("test")
		ws.AutoScalingConfig = tfv1.AutoScalingConfig{
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
		cfg := ws.GetResourceRecommenderConfig()
		Expect(cfg.TargetTflopsPercentile).To(Equal(0.8))
		Expect(cfg.LowerBoundTflopsPercentile).To(Equal(0.1))
		Expect(cfg.UpperBoundTflopsPercentile).To(Equal(0.95))
		Expect(cfg.TargetVramPercentile).To(Equal(0.7))
		Expect(cfg.LowerBoundVramPercentile).To(Equal(0.2))
		Expect(cfg.UpperBoundVramPercentile).To(Equal(0.9))
		Expect(cfg.RequestMarginFraction).To(Equal(0.15))
	})

	It("should ignore invalid float fields and keep defaults", func() {
		ws := NewWorkloadState("test")
		ws.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{
				TargetTflopsPercentile:     "not-a-float",
				LowerBoundTflopsPercentile: "",
				UpperBoundTflopsPercentile: "0.99",
			},
		}
		cfg := ws.GetResourceRecommenderConfig()
		Expect(cfg.TargetTflopsPercentile).To(Equal(DefaultResourceRecommenderConfig.TargetTflopsPercentile))
		Expect(cfg.LowerBoundTflopsPercentile).To(Equal(DefaultResourceRecommenderConfig.LowerBoundTflopsPercentile))
		Expect(cfg.UpperBoundTflopsPercentile).To(Equal(0.99))
	})

	It("should parse ConfidenceInterval if valid", func() {
		ws := NewWorkloadState("test")
		ws.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{
				ConfidenceInterval: "30m",
			},
		}
		cfg := ws.GetResourceRecommenderConfig()
		Expect(cfg.ConfidenceInterval).To(Equal(30 * time.Minute))
	})

	It("should ignore invalid ConfidenceInterval and keep default", func() {
		ws := NewWorkloadState("test")
		ws.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{
				ConfidenceInterval: "not-a-duration",
			},
		}
		cfg := ws.GetResourceRecommenderConfig()
		Expect(cfg.ConfidenceInterval).To(Equal(DefaultResourceRecommenderConfig.ConfidenceInterval))
	})

	It("should correctly determine if a resource is the target based on config", func() {
		ws := NewWorkloadState("test")

		Expect(ws.IsTargetResource("tflops")).To(BeTrue())
		Expect(ws.IsTargetResource("vram")).To(BeTrue())

		ws.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{TargetResource: "all"},
		}

		Expect(ws.IsTargetResource("tflops")).To(BeTrue())
		Expect(ws.IsTargetResource("vram")).To(BeTrue())

		ws.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{TargetResource: "tflops"},
		}
		Expect(ws.IsTargetResource("tflops")).To(BeTrue())
		Expect(ws.IsTargetResource("vram")).To(BeFalse())

		ws.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{TargetResource: "vram"},
		}
		Expect(ws.IsTargetResource("tflops")).To(BeFalse())
		Expect(ws.IsTargetResource("vram")).To(BeTrue())
	})

	It("should correctly determine if auto scaling is enabled based on config", func() {
		ws := NewWorkloadState("test")

		ws.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{Enable: true},
		}
		Expect(ws.IsAutoScalingEnabled()).To(BeTrue())
		ws.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{Enable: false},
		}
		Expect(ws.IsAutoScalingEnabled()).To(BeFalse())
	})
})
