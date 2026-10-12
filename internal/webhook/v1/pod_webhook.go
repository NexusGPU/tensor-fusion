/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import (
	"context"
	"encoding/json"
	stderrors "errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"

	"gomodules.xyz/jsonpatch/v2"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/cloudprovider/pricing"
	"github.com/NexusGPU/tensor-fusion/internal/indexallocator"
	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var httpClient = &http.Client{Timeout: 10 * time.Second}
var operatorPort = "8080"

func init() {
	if port := os.Getenv("PORT"); port != "" {
		operatorPort = port
	}
}

// SetupPodWebhookWithManager registers the webhook for Pod in the manager.
func SetupPodWebhookWithManager(mgr ctrl.Manager, portAllocator *portallocator.PortAllocator, indexAllocator *indexallocator.IndexAllocator, pricingProvider pricing.PricingProvider) error {
	webhookServer := mgr.GetWebhookServer()

	webhookServer.Register("/mutate-v1-pod",
		&admission.Webhook{
			Handler: &TensorFusionPodMutator{
				decoder:        admission.NewDecoder(mgr.GetScheme()),
				Client:         mgr.GetClient(),
				APIReader:      mgr.GetAPIReader(),
				portAllocator:  portAllocator,
				indexAllocator: indexAllocator,
			},
		})
	// Pod creation is not the only way a container enters a pod: ephemeral
	// containers are added afterwards through their own subresource.
	webhookServer.Register(podEphemeralContainersValidatePath,
		&admission.Webhook{
			Handler: &EphemeralContainerValidator{
				decoder: admission.NewDecoder(mgr.GetScheme()),
			},
		})
	return nil
}

type TensorFusionPodMutator struct {
	Client client.Client
	// APIReader reads from the API server instead of the informer cache behind
	// Client. It is for the one read that must see an object written a moment
	// ago (see createOrUpdateWorkload); without it Client is used.
	APIReader      client.Reader
	decoder        admission.Decoder
	portAllocator  *portallocator.PortAllocator
	indexAllocator *indexallocator.IndexAllocator
}

// counterReleaseTimeout bounds giving back an enabled-replicas slot after the
// admission request itself has failed or timed out.
const counterReleaseTimeout = 5 * time.Second

// Handle implements admission.Handler interface.
//
// A dry-run request (req.DryRun) gets the patch a real admission would get,
// but nothing is written: no TensorFusionWorkload is created or updated, the
// pod is not counted against enabled-replicas, and no device index or host
// port is taken. The patch differs from a real one only where a real one
// carries such a reservation:
//   - the device index resource holds the placeholder slot, not an assigned one;
//   - a pod asking for an automatic host port gets neither the port-number
//     annotation nor a hostPort;
//   - the workload the patch refers to, and its recommendation, are the ones
//     that exist at the time of the dry-run.
//
//nolint:gocyclo
func (m *TensorFusionPodMutator) Handle(ctx context.Context, req admission.Request) admission.Response {
	dryRun := req.DryRun != nil && *req.DryRun
	pod := &corev1.Pod{}
	if err := m.decoder.Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}

	if len(pod.Namespace) == 0 {
		// Using req.Namespace, as pod.Namespace appears to be unset.
		pod.Namespace = req.Namespace
	}

	log := log.FromContext(ctx)
	log.Info("Mutating pod", "generateName", pod.GenerateName, "namespace", pod.Namespace)

	var currentBytes []byte

	// for non tensor fusion pod, check if there are any GPU resource request,
	// when there is, set scheduler to tensor-fusion-scheduler to trigger proxied scheduling
	// this is to ensure that non tensor fusion pod can be scheduled to nodes not conflict with tensor fusion
	if !utils.IsTensorFusionPod(pod) {
		hasGPUResource := utils.HasGPUResourceRequest(pod)
		if !hasGPUResource {
			return admission.Allowed("non tensor fusion pod nor GPU resource request, skipped")
		}

		shouldMigrate, err := ShouldAutoMigrateGPUPod(ctx, m.Client, pod)
		if err != nil {
			log.Error(err, "failed to check auto migration rules", "pod", pod.Name, "namespace", pod.Namespace)
			return admission.Allowed("non tensor fusion pod and invalid migration rule, skipped")
		}
		if shouldMigrate {
			// set original Pod bytes before mutate it, so that patch compare can be done correctly
			currentBytes, err = json.Marshal(pod)
			// Add tensor-fusion.ai/enabled label to mark pod for TensorFusion injection
			if pod.Labels == nil {
				pod.Labels = make(map[string]string)
			}
			if err != nil {
				return admission.Errored(http.StatusBadRequest, fmt.Errorf("failed to marshal current pod: %w", err))
			}
			pod.Labels[constants.TensorFusionEnabledLabelKey] = constants.TrueStringValue
			// Native GPU resources represent whole-card claims. Preserve an explicit
			// TensorFusion isolation annotation, otherwise migrate them to the
			// equivalent shared whole-GPU allocation policy.
			if _, explicitIsolation := pod.Annotations[constants.IsolationModeAnnotation]; !explicitIsolation {
				if pod.Annotations == nil {
					pod.Annotations = make(map[string]string)
				}
				pod.Annotations[constants.IsolationModeAnnotation] = string(tfv1.IsolationModeShared)
			}
		} else {
			if utils.IsProgressiveMigration() {
				return admission.Patched("set scheduler to tensor-fusion-scheduler", jsonpatch.JsonPatchOperation{
					Operation: "replace",
					Path:      "/spec/schedulerName",
					Value:     constants.SchedulerName,
				})
			}
			return admission.Allowed("GPU pod found, skip since NVIDIA_OPERATOR_PROGRESSIVE_MIGRATION not set")
		}
	} else {
		currentBytesOriginal, err := json.Marshal(pod)
		if err != nil {
			return admission.Errored(http.StatusBadRequest, fmt.Errorf("failed to marshal current pod: %w", err))
		}
		currentBytes = currentBytesOriginal
	}

	tfInfo, err := ParseTensorFusionInfo(ctx, m.Client, pod)
	if err != nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("parse tf resources: %w", err))
	}

	counter := &TensorFusionPodCounter{Client: m.Client}
	enabledReplicas := tfInfo.EnabledReplicas

	var podCounterAnnotationKey string
	// injected is set once the pod is certain to be answered with the
	// TensorFusion patch; every other way out gives the slot back.
	injected := false
	if enabledReplicas != nil {
		// Decide whether to patch the current pod by the `tf-pod-count` kept in
		// the owner's annotation. The comparison and the increment have to be
		// one step (see TensorFusionPodCounter.Reserve): pods of one owner are
		// admitted concurrently, and counting only after the patch was built
		// let all of them pass the comparison against the same count.
		var admitted bool
		var podCounterKey string
		var err error
		if dryRun {
			var podCount int32
			podCount, podCounterKey, err = counter.Get(ctx, pod)
			admitted = podCount < *enabledReplicas
		} else {
			admitted, podCounterKey, err = counter.Reserve(ctx, pod, *enabledReplicas)
		}
		if err != nil {
			return admission.Errored(http.StatusInternalServerError, fmt.Errorf("get tf pod count: %w", err))
		}
		if !admitted {
			return admission.Allowed("tensor fusion pod count reached, keep original Pod for tensor fusion grey releasing")
		}
		podCounterAnnotationKey = podCounterKey
		if !dryRun {
			// The pod is mutated below; keep what identifies its owner.
			reservedFor := pod.DeepCopy()
			defer func() {
				if injected {
					return
				}
				// The request context may be the reason admission failed.
				releaseCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), counterReleaseTimeout)
				defer cancel()
				if err := counter.Release(releaseCtx, reservedFor, podCounterKey); err != nil {
					log.Error(err, "failed to release tf pod count of a pod that was not injected",
						"generateName", reservedFor.GenerateName, "namespace", reservedFor.Namespace, "key", podCounterKey)
				}
			}()
		}
	}

	pool := &tfv1.GPUPool{}
	if err := m.Client.Get(ctx, client.ObjectKey{Name: tfInfo.Profile.PoolName}, pool); err != nil {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("gpu pool(%s) does not exist", tfInfo.Profile.PoolName))
	}
	tfInfo.Profile.Qos = calculateQoSLevel(tfInfo.Profile, pool)

	workload, err := m.createOrUpdateWorkload(ctx, pod, &tfInfo, dryRun)
	if err != nil {
		return workloadErrorResponse(err)
	}

	// Pod mutating webhook can not get Pod UID,
	// thus need pod controller to set the controller reference
	if controllerRef := metav1.GetControllerOfNoCopy(workload); controllerRef == nil {
		pod.Annotations[constants.SetPendingOwnedWorkloadAnnotation] = tfInfo.WorkloadName
	}

	// Task 5: If workload already exists and has autoscaling enabled, set recommended annotations
	m.applyRecommendedAnnotations(pod, workload)

	// make sure required Pod info has been changed before generating patches
	if tfInfo.Profile.IsLocalGPU {
		// only patch scheduler when using local-gpu mode
		// for remote vGPU mode, start worker with tensor-fusion scheduler
		pod.Spec.SchedulerName = constants.SchedulerName
	}

	// find container index
	containerIndices := []int{}
	for _, name := range tfInfo.ContainerNames {
		for i := range pod.Spec.Containers {
			if pod.Spec.Containers[i].Name == name {
				containerIndices = append(containerIndices, i)
				break
			}
		}
	}

	if len(containerIndices) == 0 {
		return admission.Allowed("no valid container to inject tensor-fusion, skipped")
	}

	// Check if final profile is valid and contains valid GPU resource requests.
	// In partitioned mode with an explicit partition template, the resources are
	// fixed by the template (the allocator accounts capacity from the template,
	// not from req.Request — see allocPartition/CalculatePartitionResourceUsage),
	// so an empty request is expected and must not be rejected here.
	pinnedPartition := tfInfo.Profile.Isolation == tfv1.IsolationModePartitioned &&
		tfInfo.Profile.PartitionTemplateID != ""
	sharedWholeGPU := tfInfo.Profile.Isolation == tfv1.IsolationModeShared && tfInfo.Profile.GPUCount > 0
	if !pinnedPartition && !sharedWholeGPU &&
		tfInfo.Profile.Resources.Requests.Tflops.IsZero() &&
		tfInfo.Profile.Resources.Requests.ComputePercent.IsZero() &&
		tfInfo.Profile.Resources.Requests.Vram.IsZero() {
		return admission.Errored(http.StatusInternalServerError,
			fmt.Errorf("tflops request is not set, please set tensor-fusion.ai/tflops-request or/and tensor-fusion.ai/compute-percent-request"+
				" or/and tensor-fusion.ai/vram-request annotation on Pod"))
	}

	// Add defaults and tensor-fusion injection logic
	utils.AddOrOverrideTFClientMissingAnnotationsBeforePatch(pod, tfInfo)
	utils.AddTFDefaultClientConfBeforePatch(ctx, pod, pool, tfInfo, containerIndices)

	// Add priorityClass if contains higher QoS level and Pod priority class not specified
	if pod.Spec.PriorityClassName == "" &&
		(tfInfo.Profile.Qos == tfv1.QoSHigh || tfInfo.Profile.Qos == tfv1.QoSCritical) {
		pod.Spec.PriorityClassName = fmt.Sprintf("%s-%s",
			constants.TensorFusionSystemName, string(tfInfo.Profile.Qos))
		// Remove priority field if PriorityClassName is set, as Kubernetes Priority admission controller
		// will compute priority from PriorityClassName and doesn't allow both fields to be set
		pod.Spec.Priority = nil
	}

	// Inject tensor-fusion patches
	patches, err := m.patchTFClient(
		ctx,
		pod,
		pool,
		tfInfo.Profile.IsLocalGPU,
		utils.UseLocalWorkerSidecar(tfInfo.Profile),
		currentBytes,
		containerIndices,
		dryRun,
	)
	if err != nil {
		log.Error(err, "failed to patch tf client", "pod", req.Name, "namespace", req.Namespace)
		return admission.Errored(http.StatusInternalServerError, err)
	}

	if podCounterAnnotationKey != "" {
		// Patch annotation for pod counter
		patch := jsonpatch.JsonPatchOperation{
			Operation: "add",
			Path:      "/metadata/annotations/" + utils.EscapeJSONPointer(constants.TensorFusionPodCounterKeyAnnotation),
			Value:     podCounterAnnotationKey,
		}
		patches = append(patches, patch)
	}

	injected = true
	return admission.Patched("tensor fusion component patched", patches...)
}

// workloadTerminatingRetryAfterSeconds is the delay asked of the client of a
// pod refused with workloadTerminatingError. The workload on its way out has
// no pod left, so its deletion normally ends within this time.
const workloadTerminatingRetryAfterSeconds = 1

// workloadTerminatingError is returned by createOrUpdateWorkload when the
// workload the pod would reuse is on its way out: it is being deleted, it was
// deleted while the pod was being admitted, or it belongs to an owner that has
// been replaced and waits for the garbage collector. Nothing is wrong with the
// pod: the same request succeeds once the workload is gone.
type workloadTerminatingError struct {
	namespace, name string
	// state says which of the cases it is; empty means workloadBeingDeleted.
	state string
}

const (
	workloadBeingDeleted = "is still being deleted"
	workloadJustDeleted  = "was deleted while this pod was being admitted"
	workloadOfOldOwner   = "belongs to a replaced owner of the same name and is about to be deleted"
)

func (e *workloadTerminatingError) Error() string {
	state := e.state
	if state == "" {
		state = workloadBeingDeleted
	}
	return fmt.Sprintf("the previous TensorFusionWorkload %s/%s %s, retry the request in a moment",
		e.namespace, e.name, state)
}

// ownerWasReplaced reports whether the workload is controlled by an earlier
// object of the same kind and name as the pod's controller: for example a
// Deployment that was deleted and created again under its name. The garbage
// collector deletes such a workload as soon as it gets to it, so a pod admitted
// against it would be left without one.
//
// podController is the controller the webhook names workloads after (see
// utils.GetPodControllerRef: the Deployment of a ReplicaSet's pod, the CronJob
// of a Job's pod, otherwise the pod's own controller), which is also the owner
// it gives a workload it creates. Only a difference in UID alone counts. A
// workload without a controller (created by a user, or by this webhook for a
// pod without one), one controlled by another kind or API group of object that
// happens to share the name, and a pod without a controller are all left to
// the rules that applied before; a new ReplicaSet of the same Deployment does
// not change the Deployment's UID.
func ownerWasReplaced(workload *tfv1.TensorFusionWorkload, podController *metav1.OwnerReference) bool {
	owner := metav1.GetControllerOfNoCopy(workload)
	if owner == nil || podController == nil || owner.UID == "" || podController.UID == "" {
		return false
	}
	if owner.Kind != podController.Kind || owner.Name != podController.Name || owner.UID == podController.UID {
		return false
	}
	ownerGV, err := schema.ParseGroupVersion(owner.APIVersion)
	if err != nil {
		return false
	}
	podControllerGV, err := schema.ParseGroupVersion(podController.APIVersion)
	return err == nil && ownerGV.Group == podControllerGV.Group
}

// workloadErrorResponse is the admission response for an error of
// createOrUpdateWorkload.
//
// A workload on its way out (see workloadTerminatingError) is answered with 503
// and a retry delay instead of the 500 of every other error. The API server passes the
// status on and turns the delay into a Retry-After header; client-go, and with
// it kubectl and the workload controllers, repeats a request answered that way
// by itself, so the pod is usually created without its creator seeing a
// failure. 429 is repeated in the same way but tells the client it sent too
// many requests, and a 4xx such as 409 is not repeated at all and is treated
// as final by clients that look at the status class.
func workloadErrorResponse(err error) admission.Response {
	var terminating *workloadTerminatingError
	if !stderrors.As(err, &terminating) {
		return admission.Errored(http.StatusInternalServerError, fmt.Errorf("create tf workload: %w", err))
	}
	return admission.Response{
		AdmissionResponse: admissionv1.AdmissionResponse{
			Allowed: false,
			Result: &metav1.Status{
				Status:  metav1.StatusFailure,
				Code:    http.StatusServiceUnavailable,
				Reason:  metav1.StatusReasonServiceUnavailable,
				Message: terminating.Error(),
				Details: &metav1.StatusDetails{RetryAfterSeconds: workloadTerminatingRetryAfterSeconds},
			},
		},
	}
}

// InjectDecoder injects the decoder.
func (m *TensorFusionPodMutator) InjectDecoder(d admission.Decoder) error {
	m.decoder = d
	return nil
}

// createOrUpdateWorkload returns the workload the pod belongs to, creating it
// or bringing its spec up to date first. With dryRun it returns what that
// workload would be and writes nothing.
func (m *TensorFusionPodMutator) createOrUpdateWorkload(
	ctx context.Context,
	pod *corev1.Pod,
	tfInfo *utils.TensorFusionInfo,
	dryRun bool,
) (*tfv1.TensorFusionWorkload, error) {
	// Create the desired spec for comparison
	desiredSpec := *tfInfo.Profile.DeepCopy()

	workload := &tfv1.TensorFusionWorkload{}
	err := m.Client.Get(ctx, client.ObjectKey{Name: tfInfo.WorkloadName, Namespace: pod.Namespace}, workload)
	if err != nil {
		if !errors.IsNotFound(err) {
			return nil, fmt.Errorf("failed to get workload: %w", err)
		}

		// Create a new workload
		workload = &tfv1.TensorFusionWorkload{
			ObjectMeta: metav1.ObjectMeta{
				Name:      tfInfo.WorkloadName,
				Namespace: pod.Namespace,
				Labels: map[string]string{
					constants.GpuPoolKey: tfInfo.Profile.PoolName,
				},
				Annotations: map[string]string{
					constants.WorkloadModeAnnotation: constants.WorkloadModeDynamic,
				},
			},
			Spec: desiredSpec,
		}

		// Pass through disable features annotation
		if pod.Labels[constants.DisableFeaturesAnnotation] != "" {
			workload.Annotations[constants.DisableFeaturesAnnotation] = pod.Labels[constants.DisableFeaturesAnnotation]
		}

		// Pass through container-gpu-count annotation from client pod to workload
		// This preserves per-container GPU count information for multi-container scenarios
		if pod.Annotations != nil {
			if containerGPUCount, ok := pod.Annotations[constants.ContainerGPUCountAnnotation]; ok && containerGPUCount != "" {
				workload.Annotations[constants.ContainerGPUCountAnnotation] = containerGPUCount
			}
		}

		if tfInfo.PodControllerRef != nil {
			workload.OwnerReferences = []metav1.OwnerReference{*tfInfo.PodControllerRef}
		}

		if dryRun {
			return workload, nil
		}
		createErr := m.Client.Create(ctx, workload)
		if createErr == nil {
			return workload, nil
		}
		if !errors.IsAlreadyExists(createErr) {
			return nil, fmt.Errorf("failed to create workload: %w", createErr)
		}
		// Another pod of the same owner was admitted at the same moment and
		// created the workload first (the pods of a ReplicaSet are created
		// together). That is the workload this pod belongs to as well: carry
		// on with it instead of refusing the pod. It was stored a moment ago
		// and is normally not in the informer cache yet, so it is read from
		// the API server.
		reader := m.APIReader
		if reader == nil {
			reader = m.Client
		}
		existing := &tfv1.TensorFusionWorkload{}
		if err := reader.Get(ctx, client.ObjectKeyFromObject(workload), existing); err != nil {
			if errors.IsNotFound(err) {
				// It existed for the create and is gone for the read.
				return nil, &workloadTerminatingError{namespace: workload.Namespace, name: workload.Name, state: workloadJustDeleted}
			}
			return nil, fmt.Errorf("failed to get workload created by a concurrent admission: %w", err)
		}
		workload = existing
	}

	if ownerWasReplaced(workload, tfInfo.PodControllerRef) {
		return nil, &workloadTerminatingError{namespace: workload.Namespace, name: workload.Name, state: workloadOfOldOwner}
	}

	// A workload this webhook created that has no owner reference is removed by
	// the workload controller once it has had no pod for a grace period (the
	// pod it was created for may have been refused after admission). This pod
	// is not stored yet, so the admission is recorded on the workload: that
	// starts the period again and, by changing the workload, makes a delete
	// that was decided without it fail.
	awaitsPod := workload.Annotations[constants.WorkloadModeAnnotation] == constants.WorkloadModeDynamic &&
		len(workload.OwnerReferences) == 0
	if awaitsPod || !equality.Semantic.DeepEqual(workload.Spec, desiredSpec) {
		patch := client.MergeFrom(workload.DeepCopy())
		workload.Spec = desiredSpec
		if dryRun {
			return workload, nil
		}
		if awaitsPod {
			workload.Annotations[constants.WorkloadLastPodSeenAnnotation] = time.Now().UTC().Format(time.RFC3339)
		}
		if err := m.Client.Patch(ctx, workload, patch); err != nil {
			if errors.IsNotFound(err) {
				// The cache still held a workload the API server no longer has.
				return nil, &workloadTerminatingError{namespace: workload.Namespace, name: workload.Name, state: workloadJustDeleted}
			}
			return nil, fmt.Errorf("failed to patch workload: %w", err)
		}
		if awaitsPod && !workload.DeletionTimestamp.IsZero() {
			// The record came too late. Refuse the pod instead of letting it
			// start without a workload; the next attempt creates a new one.
			return nil, &workloadTerminatingError{namespace: workload.Namespace, name: workload.Name}
		}
	}
	return workload, nil
}

// applyRecommendedAnnotations applies recommended resource annotations to the pod
// if the workload already exists and has autoscaling enabled with a recommendation
func (m *TensorFusionPodMutator) applyRecommendedAnnotations(
	pod *corev1.Pod,
	workload *tfv1.TensorFusionWorkload,
) {
	// Only apply if autoscaling is enabled
	asr := workload.Spec.AutoScalingConfig.AutoSetResources
	if asr == nil || !asr.Enable {
		return
	}

	// Only apply if there's a recommendation
	if workload.Status.Recommendation == nil {
		return
	}

	recommendation := workload.Status.Recommendation

	// Set recommended annotations similar to VPA logic
	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}

	// Apply compute (TFlops) recommendations if target includes compute
	targetResource := asr.TargetResource
	if targetResource == "" || targetResource == tfv1.ScalingTargetResourceAll || targetResource == tfv1.ScalingTargetResourceCompute {
		if !recommendation.Requests.Tflops.IsZero() {
			pod.Annotations[constants.TFLOPSRequestAnnotation] = recommendation.Requests.Tflops.String()
		}
		if !recommendation.Limits.Tflops.IsZero() {
			pod.Annotations[constants.TFLOPSLimitAnnotation] = recommendation.Limits.Tflops.String()
		}
	}

	// Apply VRAM recommendations if target includes vram
	if targetResource == "" || targetResource == tfv1.ScalingTargetResourceAll || targetResource == tfv1.ScalingTargetResourceVRAM {
		if !recommendation.Requests.Vram.IsZero() {
			pod.Annotations[constants.VRAMRequestAnnotation] = recommendation.Requests.Vram.String()
		}
		if !recommendation.Limits.Vram.IsZero() {
			pod.Annotations[constants.VRAMLimitAnnotation] = recommendation.Limits.Vram.String()
		}
	}
}

func (m *TensorFusionPodMutator) patchTFClient(
	ctx context.Context,
	pod *corev1.Pod,
	pool *tfv1.GPUPool,
	isLocalGPU bool,
	sidecarWorker bool,
	currentBytes []byte,
	containerIndices []int,
	dryRun bool,
) ([]jsonpatch.JsonPatchOperation, error) {
	clientConfig := pool.Spec.ComponentConfig.Client

	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	pod.Labels[constants.LabelKeyPodTemplateHash] = utils.ClientTemplateHash(pool)

	assignPodLabelsAndAnnotations(isLocalGPU, sidecarWorker, pod, pool)

	// Index allocation only for worker pods
	// Index is used for Device Plugin communication to match Pod with CDI device
	var index int
	if pod.Labels[constants.LabelComponent] == constants.ComponentWorker && !dryRun {
		// Assign index once per pod (before processing containers)
		// Index must be assigned in webhook stage since scheduler cannot modify Pod
		// This is a special index resource (1-32), not a real device resource
		// Index is assigned in ascending order (1, 2, 3, ...) via distributed lock (leader election)
		// A dry-run takes no index: the patch then carries the placeholder slot
		// that applyPodIndexResource uses for index 0.
		index = m.assignDeviceAllocationIndex(ctx, pod)
	}
	// clean annotation if exists, must be assigned by scheduler to ensure lock of certain index on one node
	delete(pod.Annotations, constants.PodIndexAnnotation)

	for _, containerIndex := range containerIndices {
		container := &pod.Spec.Containers[containerIndex]
		containerJSON, err := json.Marshal(container)
		if err != nil {
			return nil, fmt.Errorf("marshal container: %w", err)
		}

		var patchJSON []byte
		patchJSON, err = serializeContainerInjectionPatchJson(clientConfig, patchJSON)
		if err != nil {
			return nil, err
		}

		patchedJSON, err := strategicpatch.StrategicMergePatch(containerJSON, patchJSON, corev1.Container{})
		if err != nil {
			return nil, fmt.Errorf("apply strategic merge patch to container: %w", err)
		}

		// validate if container decoded successfully after merge patch
		container = &corev1.Container{}
		if err := json.Unmarshal(patchedJSON, container); err != nil {
			return nil, fmt.Errorf("unmarshal patched container, invalid container patch: %w", err)
		}

		removeNativeGPULimits(container)

		// Inject tensor-fusion.ai/index resource for Device Plugin communication (worker pods only)
		// This is a special index resource (not a real device), used for Pod-to-DevicePlugin communication
		if pod.Labels[constants.LabelComponent] == constants.ComponentWorker {
			applyPodIndexResource(container, index)
		}

		if !isLocalGPU {
			addConnectionForRemoteFixedReplicaVirtualGPU(pod, container, clientConfig)
		} else if sidecarWorker {
			addConnectionForLocalSidecarWorker(container)
		}

		pod.Spec.Containers[containerIndex] = *container
	}
	if sidecarWorker {
		for i := range pod.Spec.Containers {
			if pod.Spec.Containers[i].Name == constants.TFContainerNameWorker {
				applyPodIndexResource(&pod.Spec.Containers[i], index)
			}
		}
	}

	// Patch hostPort allocation
	if pod.Labels[constants.GenHostPortLabel] == constants.GenHostPortLabelValue {
		// TODO/FIXME potential bug, when it's deployment created Pod rather than standalone Pod, pod.Name is empty
		if err := m.generateHostPort(pod, pod.Labels[constants.GenHostPortNameLabel], dryRun); err != nil {
			return nil, fmt.Errorf("can not generate host port: %w", err)
		}
	}

	containerPatchedJSON, err := json.Marshal(pod)
	if err != nil {
		return nil, fmt.Errorf("marshal current pod: %w", err)
	}
	patches, err := jsonpatch.CreatePatch(currentBytes, containerPatchedJSON)
	if err != nil {
		return nil, fmt.Errorf("patch to container: %w", err)
	}

	// Additional pod level patch
	strategicpatches, err := calculatePodPatch(currentBytes, pod, clientConfig, isLocalGPU)
	if err != nil {
		return nil, fmt.Errorf("calculate pod patch: %w", err)
	}
	patches = append(patches, strategicpatches...)
	return patches, nil
}

//nolint:unused
func (m *TensorFusionPodMutator) assignDeviceAllocationIndex(ctx context.Context, pod *corev1.Pod) int {
	var index int
	var indexErr error
	podIdentifier := pod.Name
	if podIdentifier == "" {
		// For Deployment/StatefulSet created pods, Name might be empty, use GenerateName + UID(maybe empty)
		podIdentifier = pod.GenerateName + string(pod.UID)
	}

	if m.indexAllocator != nil && m.indexAllocator.IsLeader {
		index, indexErr = m.indexAllocator.AssignIndex(podIdentifier)
		if indexErr != nil {
			log := log.FromContext(ctx)
			log.Error(indexErr, "failed to assign index for pod", "pod", podIdentifier)
			index = 0
		}
	} else if m.indexAllocator != nil && !m.indexAllocator.IsLeader {
		// If not leader, get index from leader via HTTP API (similar to port allocation)
		// This ensures global increment across distributed webhook instances
		index, indexErr = m.assignIndexFromLeader(ctx, pod)
		if indexErr != nil {
			log := log.FromContext(ctx)
			log.Error(indexErr, "failed to assign index from leader", "pod", podIdentifier)
			index = 0
		}
	} else {
		// No allocator available, use 0 as fallback
		index = 0
	}
	return index
}

// Convert the strategic merge patch to JSON
func calculatePodPatch(currentBytes []byte, pod *corev1.Pod, clientConfig *tfv1.ClientConfig, isLocalGPU bool) ([]jsonpatch.JsonPatchOperation, error) {
	var patchBytes []byte
	var err error
	if !isLocalGPU {
		patchBytes, err = json.Marshal(clientConfig.PatchToPod)
	}
	if err != nil {
		return nil, fmt.Errorf("marshal patch: %w", err)
	}

	// Apply the strategic merge patch
	resultBytes, err := strategicpatch.StrategicMergePatch(currentBytes, patchBytes, corev1.Pod{})
	if err != nil {
		return nil, fmt.Errorf("apply strategic merge patch: %w", err)
	}
	// Generate JSON patch operations by comparing original and patched pod
	strategicpatches, err := jsonpatch.CreatePatch(currentBytes, resultBytes)
	if err != nil {
		return nil, fmt.Errorf("create json patch: %w", err)
	}
	// Unmarshal the result back into the pod
	if err := json.Unmarshal(resultBytes, pod); err != nil {
		return nil, fmt.Errorf("unmarshal patched pod: %w", err)
	}
	return strategicpatches, nil
}

func assignPodLabelsAndAnnotations(isLocalGPU bool, sidecarWorker bool, pod *corev1.Pod, pool *tfv1.GPUPool) {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	if isLocalGPU {
		pod.Labels[constants.LabelComponent] = constants.ComponentWorker
		if sidecarWorker {
			delete(pod.Annotations, constants.EmbeddedWorkerAnnotation)
		} else {
			pod.Annotations[constants.EmbeddedWorkerAnnotation] = constants.TrueStringValue
		}
		// no need to add port in local gpu mode, communication is done through shared memory in the same process

		// Add toleration for TensorFusion nodes
		pod.Spec.Tolerations = append(pod.Spec.Tolerations, corev1.Toleration{
			Key:      constants.NodeUsedByTaintKey,
			Operator: corev1.TolerationOpEqual,
			Value:    constants.TensorFusionSystemName,
			Effect:   corev1.TaintEffectPreferNoSchedule,
		})
	} else {
		pod.Labels[constants.LabelComponent] = constants.ComponentClient
	}
	pod.Labels[constants.GpuPoolKey] = pool.Name
}

func applyPodIndexResource(container *corev1.Container, index int) {
	if container.Resources.Limits == nil {
		container.Resources.Limits = make(corev1.ResourceList)
	}
	// IndexAllocator returns a one-based index (1..128), while the resource
	// encoding is zero-based (index_0..index_f with quantities 1..8).
	// Keep the zero value as the first slot for the allocator-unavailable
	// fallback, but never let it produce a negative resource index.
	if index <= 0 || index > constants.IndexKeyLength*constants.IndexModLength {
		index = 1
	}
	zeroBasedIndex := index - 1
	indexQuantity := resource.MustParse(strconv.Itoa((zeroBasedIndex % constants.IndexModLength) + 1))
	indexKey := fmt.Sprintf("%s%s%x", constants.PodIndexAnnotation, constants.PodIndexDelimiter, zeroBasedIndex/constants.IndexModLength)
	container.Resources.Limits[corev1.ResourceName(indexKey)] = indexQuantity
}

func addConnectionForLocalSidecarWorker(container *corev1.Container) {
	upsertContainerEnv(container, corev1.EnvVar{
		Name: constants.ConnectionInfoEnv,
		// protocol+identifier+size+initVersion
		Value: fmt.Sprintf("shmem+%s+%s+1", constants.ConnectionSharedMemName, constants.ConnectionSharedMemSize),
	})
	upsertContainerEnv(container, corev1.EnvVar{
		Name:  constants.DisableVMSharedMemEnv,
		Value: "0",
	})
}

func upsertContainerEnv(container *corev1.Container, env corev1.EnvVar) {
	for i := range container.Env {
		if container.Env[i].Name == env.Name {
			container.Env[i] = env
			return
		}
	}
	container.Env = append(container.Env, env)
}

func addConnectionForRemoteFixedReplicaVirtualGPU(pod *corev1.Pod, container *corev1.Container, clientConfig *tfv1.ClientConfig) {
	var prefix string
	if pod.GenerateName == "" && pod.Name != "" {
		prefix = pod.Name + constants.TFConnectionNamePrefix
	} else {
		prefix = pod.GenerateName + constants.TFConnectionNameNoPrefix
	}
	connectionName := fmt.Sprintf("%s%s", prefix, utils.NewShortID(10))
	connectionNamespace := pod.Namespace

	// metadata TF_POD_NAME and TF_CONNECTION_NAMESPACE
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  constants.ConnectionNameEnv,
		Value: connectionName,
	})
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  constants.ConnectionNamespaceEnv,
		Value: connectionNamespace,
	})
	// operator k8s serviceURL ? namespace
	container.Env = append(container.Env, corev1.EnvVar{
		Name:  constants.GetConnectionURLEnv,
		Value: fmt.Sprintf("%s/api/connection?name=%s&namespace=%s", clientConfig.OperatorEndpoint, connectionName, connectionNamespace),
	})
}

func removeNativeGPULimits(container *corev1.Container) {
	resourceNames := utils.GetGPUResourceNames()
	for _, resourceName := range resourceNames {
		if container.Resources.Requests != nil {
			delete(container.Resources.Requests, resourceName)
		}
		if container.Resources.Limits != nil {
			delete(container.Resources.Limits, resourceName)
		}
	}
}

func serializeContainerInjectionPatchJson(
	clientConfig *tfv1.ClientConfig,
	patchJSON []byte,
) ([]byte, error) {
	var err error
	useContainerPatch := clientConfig.PatchToContainer != nil
	if useContainerPatch {
		patchJSON, err = json.Marshal(clientConfig.PatchToContainer)
		if err != nil {
			return nil, fmt.Errorf("marshal patchToContainer: %w", err)
		}
	}
	return patchJSON, nil
}

// generateHostPort assigns a cluster level host port to the named container
// port. With dryRun it only checks that the port name exists: assigning would
// take the port out of the allocator for a pod that is never created.
func (m *TensorFusionPodMutator) generateHostPort(pod *corev1.Pod, portName string, dryRun bool) error {

	portNameFound := false
	containerIndex := -1
	portIndex := -1
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		for j := range container.Ports {
			port := &container.Ports[j]
			if port.Name == portName {
				portNameFound = true
				containerIndex = i
				portIndex = j
			}
		}
	}
	if !portNameFound {
		return fmt.Errorf("port name %s not found, can not assign host port for pod %s", portName, pod.Name)
	}
	if dryRun {
		return nil
	}

	if !m.portAllocator.IsLeader {
		port, err := m.assignClusterHostPortFromLeader(pod)
		if err != nil {
			return fmt.Errorf("can not assign cluster host port from leader: %w", err)
		}
		pod.Annotations[constants.GenPortNumberAnnotation] = strconv.Itoa(port)
	} else {
		port, err := m.portAllocator.AssignClusterLevelHostPort(pod.Name)
		if err != nil {
			return fmt.Errorf("can not assign cluster level host port: %w", err)
		}
		pod.Annotations[constants.GenPortNumberAnnotation] = strconv.Itoa(port)
	}

	pod.Spec.Containers[containerIndex].Ports[portIndex].HostPort = int32(m.getPortNumber(pod))
	return nil
}

func (m *TensorFusionPodMutator) getPortNumber(pod *corev1.Pod) int {
	portNumber, _ := strconv.Atoi(pod.Annotations[constants.GenPortNumberAnnotation])
	return portNumber
}

func (m *TensorFusionPodMutator) assignClusterHostPortFromLeader(pod *corev1.Pod) (int, error) {
	leaderIP := utils.GetLeaderIP(m.Client)
	if leaderIP == "" {
		return 0, fmt.Errorf("operator leader IP not found")
	}

	urlStr := fmt.Sprintf("http://%s:%s/api/assign-host-port?podName=%s", leaderIP, operatorPort, pod.Name)
	// Server-side registers this endpoint as POST in internal/server/server.go,
	// so the request method must match — using GET here returned 405 Method Not
	// Allowed on non-leader webhook replicas and blocked admission.
	req, err := http.NewRequest("POST", urlStr, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set(constants.AuthorizationHeader, "Bearer "+utils.ReadServiceAccountToken())
	resp, err := httpClient.Do(req)
	if err != nil {
		return 0, fmt.Errorf("failed to assign host port: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("host port allocation failed: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("failed to read allocation response: %w", err)
	}

	return strconv.Atoi(string(body))
}

//nolint:unused
func (m *TensorFusionPodMutator) assignIndexFromLeader(ctx context.Context, pod *corev1.Pod) (int, error) {
	leaderIP := utils.GetLeaderIP(m.Client)
	if leaderIP == "" {
		return 0, fmt.Errorf("operator leader IP not found")
	}

	podIdentifier := pod.Name
	if podIdentifier == "" {
		podIdentifier = pod.GenerateName + string(pod.UID)
	}
	urlStr := fmt.Sprintf("http://%s:%s/api/assign-index?podName=%s", leaderIP, operatorPort, podIdentifier)
	req, err := http.NewRequestWithContext(ctx, "GET", urlStr, nil)
	if err != nil {
		return 0, err
	}
	req.Header.Set(constants.AuthorizationHeader, "Bearer "+utils.ReadServiceAccountToken())
	resp, err := httpClient.Do(req)
	if err != nil {
		return 0, fmt.Errorf("failed to assign index: %w", err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("index allocation failed: %s", resp.Status)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("failed to read allocation response: %w", err)
	}

	return strconv.Atoi(string(body))
}

func calculateQoSLevel(profile *tfv1.WorkloadProfileSpec, pool *tfv1.GPUPool) tfv1.QoSLevel {
	// when not set, assign default QoS
	if profile.Qos == "" {
		sameReqLimits := profile.Resources.Limits.Tflops.Cmp(profile.Resources.Requests.Tflops) == 0 &&
			profile.Resources.Limits.Vram.Cmp(profile.Resources.Requests.Vram) == 0 &&
			!profile.Resources.Requests.Tflops.IsZero() && !profile.Resources.Requests.Vram.IsZero()

		// set to high if req == limits and both are not zero, same logic as Kubernetes QoS
		// critical QoS can preempt other pods, have to be set manually
		if sameReqLimits {
			return constants.QoSLevelHigh
		}

		if pool.Spec.QosConfig == nil || pool.Spec.QosConfig.DefaultQoS == "" {
			return constants.QoSLevelMedium
		}
		return pool.Spec.QosConfig.DefaultQoS
	}
	return profile.Qos
}
