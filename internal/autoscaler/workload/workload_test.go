package workload

import (
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/resource"
)

var _ = Describe("Workload", func() {
	It("should correctly determine if a resource is the target based on config", func() {
		ws := NewWorkloadState("test")

		Expect(ws.ShouldScaleResource(tfv1.ResourceTflops)).To(BeTrue())
		Expect(ws.ShouldScaleResource(tfv1.ResourceVram)).To(BeTrue())

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

	It("should return last resources request from the annotations", func() {
		ws := NewWorkloadState("test")
		tflopsRequest := resource.MustParse("10")
		vramRequest := resource.MustParse("8Gi")
		tflopsLimit := resource.MustParse("20")
		vramLimit := resource.MustParse("16Gi")
		ws.Annotations = map[string]string{
			constants.LastTFLOPSRequestAnnotation: tflopsRequest.String(),
			constants.LastVRAMRequestAnnotation:   vramRequest.String(),
			constants.LastTFLOPSLimitAnnotation:   tflopsLimit.String(),
			constants.LastVRAMLimitAnnotation:     vramLimit.String(),
		}
		resources, _ := ws.GetLastResourcesFromAnnotations()
		Expect(resources.Requests.Tflops.Equal(tflopsRequest)).To(BeTrue())
		Expect(resources.Requests.Vram.Equal(vramRequest)).To(BeTrue())
		Expect(resources.Limits.Tflops.Equal(tflopsLimit)).To(BeTrue())
		Expect(resources.Limits.Vram.Equal(vramLimit)).To(BeTrue())

		ws.Annotations = map[string]string{
			constants.LastVRAMLimitAnnotation: vramLimit.String(),
		}
		_, err := ws.GetLastResourcesFromAnnotations()
		Expect(err).To(HaveOccurred())
	})
})
