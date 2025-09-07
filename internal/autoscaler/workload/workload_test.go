package workload

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Workload", func() {
	It("should correctly determine if a resource is the target based on config", func() {
		ws := NewWorkloadState()

		Expect(ws.ShouldScaleResource(tfv1.ResourceTflops)).To(BeFalse())
		Expect(ws.ShouldScaleResource(tfv1.ResourceVram)).To(BeFalse())

		ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{TargetResource: "all"},
		}

		Expect(ws.ShouldScaleResource(tfv1.ResourceTflops)).To(BeTrue())
		Expect(ws.ShouldScaleResource(tfv1.ResourceVram)).To(BeTrue())

		ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{TargetResource: "tflops"},
		}
		Expect(ws.ShouldScaleResource(tfv1.ResourceTflops)).To(BeTrue())
		Expect(ws.ShouldScaleResource(tfv1.ResourceVram)).To(BeFalse())

		ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{TargetResource: "vram"},
		}
		Expect(ws.ShouldScaleResource(tfv1.ResourceTflops)).To(BeFalse())
		Expect(ws.ShouldScaleResource(tfv1.ResourceVram)).To(BeTrue())
	})

	It("should correctly determine if auto set resources is enabled based on config", func() {
		ws := NewWorkloadState()

		ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{Enable: true},
		}
		Expect(ws.IsAutoSetResourcesEnabled()).To(BeTrue())
		ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{Enable: false},
		}
		Expect(ws.IsAutoSetResourcesEnabled()).To(BeFalse())
	})
})
