package kubernetes

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// podEventComponent is the source the hypervisor's Events are reported under.
const podEventComponent = "tensor-fusion-hypervisor"

func init() {
	utilruntime.Must(corev1.AddToScheme(scheme))
}

// EmitPodEvent records one Kubernetes Event on a Pod. Every call creates an
// Event of its own; the caller decides how often that is worth doing.
func (b *KubeletBackend) EmitPodEvent(
	ctx context.Context, namespace, podName, podUID, eventType, reason, message string,
) error {
	now := time.Now()
	timestamp := metav1.NewTime(now)
	event := &corev1.Event{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			// Named like the Events of client-go's recorder.
			Name: fmt.Sprintf("%s.%x", podName, now.UnixNano()),
		},
		InvolvedObject: corev1.ObjectReference{
			APIVersion: "v1",
			Kind:       "Pod",
			Namespace:  namespace,
			Name:       podName,
			UID:        types.UID(podUID),
		},
		Type:                eventType,
		Reason:              reason,
		Message:             message,
		Source:              corev1.EventSource{Component: podEventComponent, Host: b.nodeName},
		ReportingController: podEventComponent,
		ReportingInstance:   b.nodeName,
		FirstTimestamp:      timestamp,
		LastTimestamp:       timestamp,
		Count:               1,
	}
	if err := b.apiClient.client.Create(ctx, event); err != nil {
		return fmt.Errorf("record %s event on Pod %s/%s: %w", reason, namespace, podName, err)
	}
	return nil
}
