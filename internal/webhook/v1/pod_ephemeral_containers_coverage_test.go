package v1

import (
	"context"
	"encoding/json"
	"testing"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

// Remote worker pods are created by the operator itself and never carry the
// label the pod webhook selects by. They do carry the node data volume, which
// is what this rule goes by: composed the way the controller composes them,
// they are covered.
func TestRemoteWorkerPodsAreCoveredByTheDataVolumeRule(t *testing.T) {
	template := `{"template":{"spec":{"containers":[{"name":"tensorfusion-worker","image":"registry.example/worker:1"}]}}}`
	workerConfig := &tfv1.WorkerConfig{PodTemplate: &runtime.RawExtension{Raw: []byte(template)}}
	hypervisorConfig := &tfv1.HypervisorConfig{PortNumber: ptr.To[int32](8000)}
	podTemplate := &corev1.PodTemplate{}
	if err := json.Unmarshal(workerConfig.PodTemplate.Raw, podTemplate); err != nil {
		t.Fatal(err)
	}
	workload := &tfv1.TensorFusionWorkload{ObjectMeta: metav1.ObjectMeta{Name: "remote", Namespace: "team-a"}}
	spec := podTemplate.Template.Spec
	utils.AddWorkerConfAfterTemplate(context.Background(), &spec, &workload.Spec, workerConfig, hypervisorConfig, workload)

	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "remote-worker", Namespace: "team-a"}, Spec: spec}
	if utils.IsTensorFusionPod(pod) {
		t.Fatal("the test assumes a worker pod without the label the pod webhook selects by")
	}
	volumes := dataVolumeNames(pod)
	if _, found := volumes[constants.DataVolumeName]; !found {
		t.Fatalf("a remote worker pod carries no data volume the rule recognises: volumes %+v", spec.Volumes)
	}

	// An ephemeral container that mounts the volume is refused, a plain one admitted.
	pod.Spec.EphemeralContainers = []corev1.EphemeralContainer{ephemeralContainer(debugContainerName,
		[]corev1.VolumeMount{{Name: constants.DataVolumeName, MountPath: debugMountPath}}, nil)}
	if err := validateEphemeralContainers(pod, nil); err == nil {
		t.Error("an ephemeral container mounting the data volume of a remote worker pod must be refused")
	}
	pod.Spec.EphemeralContainers = []corev1.EphemeralContainer{ephemeralContainer(debugContainerName, nil, nil)}
	if err := validateEphemeralContainers(pod, nil); err != nil {
		t.Errorf("a plain debug container must be admitted: %v", err)
	}
}

func ephemeralRequest(b *testing.B, pod, oldPod *corev1.Pod) admission.Request {
	b.Helper()
	raw, err := json.Marshal(pod)
	if err != nil {
		b.Fatal(err)
	}
	oldRaw, err := json.Marshal(oldPod)
	if err != nil {
		b.Fatal(err)
	}
	return admission.Request{AdmissionRequest: admissionv1.AdmissionRequest{
		Operation: admissionv1.Update, SubResource: ephemeralContainersSubResource,
		Namespace: pod.Namespace, Name: pod.Name,
		Object: runtime.RawExtension{Raw: raw}, OldObject: runtime.RawExtension{Raw: oldRaw},
	}}
}

// The handler is on the path of every ephemeral-container update in the
// cluster. For a pod without a data volume it decodes the submitted pod and
// answers; for one with a data volume it also decodes the previous object and
// compares the containers.
func BenchmarkEphemeralContainerValidator(b *testing.B) {
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		b.Fatal(err)
	}
	validator := &EphemeralContainerValidator{decoder: admission.NewDecoder(scheme)}
	debug := ephemeralContainer(debugContainerName, nil, nil)

	b.Run("pod without a data volume", func(b *testing.B) {
		req := ephemeralRequest(b, podWithVolumes(false, nil, debug), podWithVolumes(false, nil))
		b.ReportAllocs()
		for b.Loop() {
			if resp := validator.Handle(context.Background(), req); !resp.Allowed {
				b.Fatal("refused")
			}
		}
	})
	b.Run("pod with a data volume", func(b *testing.B) {
		req := ephemeralRequest(b, podWithEphemeralContainers(debug), podWithEphemeralContainers())
		b.ReportAllocs()
		for b.Loop() {
			if resp := validator.Handle(context.Background(), req); !resp.Allowed {
				b.Fatal("refused")
			}
		}
	})
}
