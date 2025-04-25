package v1

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type TFResource struct {
	ContainerName       string
	ConnectionName      string
	ConnectionNamespace string
	TflopsRequest       resource.Quantity
	VramRequest         resource.Quantity
	TflopsLimit         resource.Quantity
	VramLimit           resource.Quantity
}

type TensorFusionInfo struct {
	Profile         *tfv1.WorkloadProfileSpec
	Replicas        int32
	EnabledReplicas *int32
	WorkloadName    string
	ContainerNames  []string
	GenWorkload     bool
}

func ParseTensorFusionInfo(ctx context.Context, k8sClient client.Client, pod *corev1.Pod) (TensorFusionInfo, error) {
	var info TensorFusionInfo
	if pod.Annotations == nil {
		return info, fmt.Errorf("no annotations found")
	}
	enabledReplicas, ok := pod.Annotations[constants.TensorFusionEnabledReplicasAnnotation]
	if !ok {
		info.EnabledReplicas = nil
	} else {
		val, err := strconv.ParseInt(enabledReplicas, 10, 32)
		if err != nil {
			return info, fmt.Errorf("invalid enabledReplicas value: %s, err: %w", enabledReplicas, err)
		}
		val32 := int32(val)
		info.EnabledReplicas = &val32
	}

	workloadName, ok := pod.Annotations[constants.WorkloadKey]
	if !ok {
		return info, fmt.Errorf("workload key not found")
	}
	info.WorkloadName = workloadName
	genWorkload, ok := pod.Annotations[constants.GenWorkloadAnnotation]
	info.GenWorkload = (ok && genWorkload == "true")

	replicas, ok := pod.Annotations[constants.ReplicasAnnotation]

	if !ok {
		info.Replicas = 1
	} else {
		val, err := strconv.ParseInt(replicas, 10, 32)
		if err != nil {
			return info, fmt.Errorf("invalid replicas value: %w", err)
		}
		info.Replicas = int32(val)
	}

	workloadProfileName, ok := pod.Annotations[constants.WorkloadProfileAnnotation]
	workloadProfile := &tfv1.WorkloadProfile{}
	if ok {
		if err := k8sClient.Get(ctx, client.ObjectKey{Name: workloadProfileName, Namespace: pod.Namespace}, workloadProfile); err != nil {
			return info, fmt.Errorf("get workload profile(%s) : %w", workloadProfileName, err)
		}
	}

	poolName, ok := pod.Annotations[constants.GpuPoolKey]
	if !ok {
		// TODO: select default pool
		return info, fmt.Errorf("gpu pool not found")
	}
	workloadProfile.Spec.PoolName = poolName

	tflopsRequest, ok := pod.Annotations[constants.TFLOPSRequestAnnotation]
	if ok {
		workloadProfile.Spec.Resources.Requests.Tflops = resource.MustParse(tflopsRequest)
	}
	vramRequest, ok := pod.Annotations[constants.VRAMRequestAnnotation]
	if ok {
		workloadProfile.Spec.Resources.Requests.Vram = resource.MustParse(vramRequest)
	}
	tflopsLimit, ok := pod.Annotations[constants.TFLOPSLimitAnnotation]
	if ok {
		workloadProfile.Spec.Resources.Limits.Tflops = resource.MustParse(tflopsLimit)
	}
	vramLimit, ok := pod.Annotations[constants.VRAMLimitAnnotation]
	if ok {
		workloadProfile.Spec.Resources.Limits.Vram = resource.MustParse(vramLimit)
	}

	injectContainer, ok := pod.Annotations[constants.InjectContainerAnnotation]
	containerNames := strings.Split(injectContainer, ",")
	if !ok || len(containerNames) == 0 {
		return info, fmt.Errorf("inject container not found")
	}

	info.Profile = &workloadProfile.Spec
	info.ContainerNames = containerNames
	return info, nil
}
