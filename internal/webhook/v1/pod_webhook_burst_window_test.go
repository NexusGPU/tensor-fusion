package v1

import (
	"strings"
	"testing"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// A compute burst window that is not a duration, is negative or is above the
// maximum is refused before anything else is looked at, with a message that
// names the annotation and says what is wrong with the value.
func TestParseTensorFusionInfoRejectsAnUnusableBurstWindow(t *testing.T) {
	for raw, reason := range map[string]string{
		"11s":   "above the maximum of 10s",
		"1h":    "above the maximum of 10s",
		"-1s":   "cannot be negative",
		"fast":  "not a duration",
		"0.5":   "not a duration",
		"2 sec": "not a duration",
	} {
		pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
			Namespace: "default", Annotations: map[string]string{constants.ComputeBurstWindowAnnotation: raw},
		}}
		// No client: the value is refused before one is needed.
		_, err := ParseTensorFusionInfo(t.Context(), nil, pod)
		if err == nil || !strings.Contains(err.Error(), constants.ComputeBurstWindowAnnotation) ||
			!strings.Contains(err.Error(), reason) {
			t.Errorf("burst window %q: error = %v, want one about the annotation saying %q", raw, err, reason)
		}
	}
}

// Through the API server and the registered webhook: a Pod with a usable burst
// window is admitted and keeps the annotation as written, which is where the
// hypervisor reads it; one with a value that cannot be applied is not created.
var _ = Describe("Pod webhook compute burst window", func() {
	It("admits a usable burst window unchanged and rejects one that is not", func(ctx SpecContext) {
		ns := newAdmissionTestNamespace(ctx, "burst")
		for _, raw := range []string{"500ms", "2s", "0", "10s", "1.5s"} {
			pod := newAdmissionTestPod(ns, nil, "")
			pod.Annotations[constants.ComputeBurstWindowAnnotation] = raw
			Expect(k8sClient.Create(ctx, pod, client.DryRunAll)).To(Succeed(), "burst window %q", raw)
			Expect(pod.Labels).To(HaveKey(constants.LabelComponent), "the pod went through the webhook")
			Expect(pod.Annotations).To(HaveKeyWithValue(constants.ComputeBurstWindowAnnotation, raw))
		}
		for raw, reason := range map[string]string{
			"10001ms": "above the maximum of 10s",
			"-500ms":  "cannot be negative",
			"half":    "not a duration",
		} {
			pod := newAdmissionTestPod(ns, nil, "")
			pod.Annotations[constants.ComputeBurstWindowAnnotation] = raw
			err := k8sClient.Create(ctx, pod, client.DryRunAll)
			Expect(err).To(HaveOccurred(), "burst window %q", raw)
			Expect(err.Error()).To(ContainSubstring(constants.ComputeBurstWindowAnnotation))
			Expect(err.Error()).To(ContainSubstring(reason))
		}
	})
})
