package v1

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	tfv1 "github.com/NexusGPU/tensor-fusion-operator/api/v1"
	"github.com/NexusGPU/tensor-fusion-operator/internal/constants"
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
	Profile        *tfv1.ClientProfileSpec
	Replicas       int32
	WorkloadName   string
	ContainerNames []string
	GenWorkload    bool
}

func ParseTensorFusionInfo(ctx context.Context, k8sclient client.Client, pod *corev1.Pod) (TensorFusionInfo, error) {
	var info TensorFusionInfo
	if pod.Annotations == nil {
		return info, fmt.Errorf("no annotations found")
	}
	workloadName, ok := pod.Annotations[constants.WorkloadKey]
	if !ok {
		return info, fmt.Errorf("workload key not found")
	}
	info.WorkloadName = workloadName
	genWorkload, ok := pod.Annotations[constants.GenWorkload]
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

	clientProfileName, ok := pod.Annotations[constants.ClientProfileAnnotation]
	clientProfile := &tfv1.ClientProfile{}
	if ok {
		if err := k8sclient.Get(ctx, client.ObjectKey{Name: clientProfileName, Namespace: pod.Namespace}, clientProfile); err != nil {
			return info, fmt.Errorf("get client profile(%s) : %w", clientProfileName, err)
		}
	}

	poolName, ok := pod.Annotations[constants.GpuPoolKey]
	if !ok {
		// TODO: select default pool
		return info, fmt.Errorf("gpu pool not found")
	}
	clientProfile.Spec.PoolName = poolName

	tflopsRequest, ok := pod.Annotations[constants.TFLOPSRequestAnnotation]
	if ok {
		clientProfile.Spec.Resources.Requests.Tflops = resource.MustParse(tflopsRequest)
	}
	vramRequest, ok := pod.Annotations[constants.VRAMRequestAnnotation]
	if ok {
		clientProfile.Spec.Resources.Requests.Vram = resource.MustParse(vramRequest)
	}
	tflopsLimit, ok := pod.Annotations[constants.TFLOPSLimitAnnotation]
	if ok {
		clientProfile.Spec.Resources.Limits.Tflops = resource.MustParse(tflopsLimit)
	}
	vramLimit, ok := pod.Annotations[constants.VRAMLimitAnnotation]
	if ok {
		clientProfile.Spec.Resources.Limits.Vram = resource.MustParse(vramLimit)
	}

	injectContainer, ok := pod.Annotations[constants.InjectContainerAnnotation]
	containerNames := strings.Split(injectContainer, ",")
	if !ok || len(containerNames) == 0 {
		return info, fmt.Errorf("inject container not found")
	}

	info.Profile = &clientProfile.Spec
	info.ContainerNames = containerNames
	return info, nil
}
