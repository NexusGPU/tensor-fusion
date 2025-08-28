package workload

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"
)

var _ = Describe("Workload", func() {
	It("should correctly determine if a resource is the target based on config", func() {
		ws := NewWorkloadState("test")

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
		ws := NewWorkloadState("test")

		ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{Enable: true},
		}
		Expect(ws.IsAutoSetResourcesEnabled()).To(BeTrue())
		ws.Spec.AutoScalingConfig = tfv1.AutoScalingConfig{
			AutoSetResources: tfv1.AutoSetResources{Enable: false},
		}
		Expect(ws.IsAutoSetResourcesEnabled()).To(BeFalse())
	})

	It("should return current resources spec from the annotations", func() {
		ws := NewWorkloadState("test")
		expect := tfv1.Resources{
			Requests: tfv1.Resource{
				Tflops: resource.MustParse("10"),
				Vram:   resource.MustParse("8Gi"),
			},
			Limits: tfv1.Resource{
				Tflops: resource.MustParse("20"),
				Vram:   resource.MustParse("16Gi"),
			},
		}
		ws.Annotations = utils.GPUResourcesToAnnotations(&expect)
		got, _ := ws.GetCurrentResourcesSpec()
		Expect(got.Equal(&expect))
	})
})
