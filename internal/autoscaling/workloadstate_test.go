package autoscaling

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Workload State", func() {

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
