package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/shirou/gopsutil/mem"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/config"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/samber/lo"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/retry"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/apiutil"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

const TMP_PATH = "/tmp"
const LAPTOP_GPU_SUFFIX = " Laptop GPU"

var Scheme = runtime.NewScheme()

func init() {
	utilruntime.Must(tfv1.AddToScheme(Scheme))
}

func main() {
	var k8sNodeName string

	var gpuInfoConfig string
	flag.StringVar(&k8sNodeName, "hostname", "", "hostname")
	flag.StringVar(&gpuInfoConfig, "gpu-info-config",
		constants.TensorFusionGPUInfoConfigMountPath, "specify the path to gpuInfoConfig file")

	if k8sNodeName == "" {
		k8sNodeName = os.Getenv("HOSTNAME")
	}

	k8sClient, err := kubeClient()
	if err != nil {
		ctrl.Log.Error(err, "unable to create kubeClient")
		os.Exit(1)
	}

	gpuNodeName := os.Getenv(constants.NodeDiscoveryReportGPUNodeEnvName)
	if gpuNodeName == "" {
		gpuNodeName = k8sNodeName
	}

	opts := zap.Options{
		Development: true,
	}

	opts.BindFlags(flag.CommandLine)
	flag.Parse()
	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	gpuInfo := make([]config.GpuInfo, 0)
	err = utils.LoadConfigFromFile(gpuInfoConfig, &gpuInfo)
	if err != nil {
		ctrl.Log.Error(err, "unable to read gpuInfoConfig file")
		os.Exit(1)
	}

	ret := nvml.Init()
	if ret != nvml.SUCCESS {
		ctrl.Log.Error(errors.New(nvml.ErrorString(ret)), "unable to initialize NVML")
		os.Exit(1)
	}
	defer func() {
		ret := nvml.Shutdown()
		if ret != nvml.SUCCESS {
			ctrl.Log.Error(errors.New(nvml.ErrorString(ret)), "unable to shutdown NVML")
			os.Exit(1)
		}
	}()

	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS {
		ctrl.Log.Error(errors.New(nvml.ErrorString(ret)), "unable to get device count")
		os.Exit(1)
	}

	ctx := context.Background()
	gpunode := &tfv1.GPUNode{
		ObjectMeta: metav1.ObjectMeta{
			Name: gpuNodeName,
		},
	}
	if err := k8sClient.Get(ctx, client.ObjectKeyFromObject(gpunode), gpunode); err != nil {
		ctrl.Log.Error(err, "unable to get gpuNode")
		os.Exit(1)
	}

	totalTFlops := resource.Quantity{}
	totalVRAM := resource.Quantity{}
	availableTFlops := resource.Quantity{}
	availableVRAM := resource.Quantity{}

	allDeviceIDs := make([]string, 0)

	for i := range count {
		device, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			ctrl.Log.Error(errors.New(nvml.ErrorString(ret)), "unable to get device", "index", i)
			os.Exit(1)
		}

		uuid, ret := device.GetUUID()
		if ret != nvml.SUCCESS {
			ctrl.Log.Error(errors.New(nvml.ErrorString(ret)), "unable to get uuid of device", "index", i)
			os.Exit(1)
		}
		uuid = strings.ToLower(uuid)
		deviceName, ret := device.GetName()
		if ret != nvml.SUCCESS {
			ctrl.Log.Error(errors.New(nvml.ErrorString(ret)), "unable to get name of device", "index", i)
			os.Exit(1)
		}

		allDeviceIDs = append(allDeviceIDs, uuid)

		memInfo, ret := device.GetMemoryInfo_v2()
		if ret != nvml.SUCCESS {
			ctrl.Log.Error(errors.New(nvml.ErrorString(ret)), "unable to get memory info of device", "index", i)
			os.Exit(1)
		}

		// Nvidia mobile series GPU chips are the same as desktop series GPU, but clock speed is lower
		// so we can use desktop series GPU info to represent mobile series GPU, and set available TFlops with a multiplier
		isLaptopGPU := strings.HasSuffix(deviceName, LAPTOP_GPU_SUFFIX)
		if isLaptopGPU {
			deviceName = strings.ReplaceAll(deviceName, LAPTOP_GPU_SUFFIX, "")
			ctrl.Log.Info("found mobile/laptop GPU, clock speed is lower, will set lower TFlops", "deviceName", deviceName)
		}
		info, ok := lo.Find(gpuInfo, func(info config.GpuInfo) bool {
			return info.FullModelName == deviceName
		})
		if !ok {
			ctrl.Log.Info(
				"[Error] Unknown GPU model, please update `gpu-public-gpu-info` configMap "+
					" to match your GPU model name in `nvidia-smi`, this may cause you workload stuck, "+
					"refer this doc to resolve it in detail: "+
					"https://tensor-fusion.ai/guide/troubleshooting/handbook"+
					"#pod-stuck-in-starting-status-after-enabling-tensorfusion",
				"deviceName", deviceName, "uuid", uuid)
			os.Exit(1)
		}
		tflops := info.Fp16TFlops
		if isLaptopGPU {
			tflops = resource.MustParse(fmt.Sprintf("%.2f",
				tflops.AsApproximateFloat64()*constants.MobileGpuClockSpeedMultiplier))
		}
		ctrl.Log.Info("found GPU info from config", "deviceName", deviceName, "FP16 TFlops", tflops, "uuid", uuid)

		gpu, err := createOrUpdateTensorFusionGPU(k8sClient, ctx, k8sNodeName, gpunode, uuid, deviceName, memInfo, tflops)
		if err != nil {
			ctrl.Log.Error(err, "failed to create or update GPU", "uuid", uuid)
			os.Exit(1)
		}
		totalTFlops.Add(gpu.Status.Capacity.Tflops)
		totalVRAM.Add(gpu.Status.Capacity.Vram)
		availableTFlops.Add(gpu.Status.Available.Tflops)
		availableVRAM.Add(gpu.Status.Available.Vram)
	}

	err = retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		return patchGPUNodeStatus(k8sClient, ctx, gpunode, totalTFlops, totalVRAM, int32(count), allDeviceIDs)
	})
	if err != nil {
		ctrl.Log.Error(err, "failed to patch status of GPUNode after retries")
		os.Exit(1)
	}
}

// Use proper patch-based update with retry on conflict
func patchGPUNodeStatus(k8sClient client.Client, ctx context.Context,
	gpunode *tfv1.GPUNode, totalTFlops resource.Quantity, totalVRAM resource.Quantity,
	count int32, allDeviceIDs []string) error {

	currentGPUNode := &tfv1.GPUNode{}
	if err := k8sClient.Get(ctx, client.ObjectKeyFromObject(gpunode), currentGPUNode); err != nil {
		return err
	}
	patch := client.MergeFrom(currentGPUNode.DeepCopy())
	updateGPUNodeStatus(&currentGPUNode.Status, totalTFlops, totalVRAM, count, allDeviceIDs)
	return k8sClient.Status().Patch(ctx, currentGPUNode, patch)
}

func createOrUpdateTensorFusionGPU(
	k8sClient client.Client, ctx context.Context, k8sNodeName string, gpunode *tfv1.GPUNode,
	uuid string, deviceName string, memInfo nvml.Memory_v2, tflops resource.Quantity) (*tfv1.GPU, error) {
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: uuid,
		},
	}

	if len(gpunode.OwnerReferences) == 0 {
		return nil, fmt.Errorf("GPUNode has no owner references of GPU pool")
	}

	err := retry.OnError(wait.Backoff{
		Steps:    10,
		Duration: time.Second,
		Factor:   1.0,
		Jitter:   0.1,
	}, func(err error) bool {
		return true // Retry on all errors for now
	}, func() error {
		_, err := controllerutil.CreateOrUpdate(ctx, k8sClient, gpu, func() error {
			// Set metadata fields
			gpu.Labels = map[string]string{
				constants.LabelKeyOwner: gpunode.Name,
				constants.GpuPoolKey:    gpunode.OwnerReferences[0].Name,
			}
			gpu.Annotations = map[string]string{
				constants.LastSyncTimeAnnotationKey: time.Now().Format(time.RFC3339),
			}

			if !metav1.IsControlledBy(gpu, gpunode) {
				// Create a new controller ref.
				gvk, err := apiutil.GVKForObject(gpunode, Scheme)
				if err != nil {
					return err
				}
				ref := metav1.OwnerReference{
					APIVersion:         gvk.GroupVersion().String(),
					Kind:               gvk.Kind,
					Name:               gpunode.GetName(),
					UID:                gpunode.GetUID(),
					BlockOwnerDeletion: ptr.To(true),
					Controller:         ptr.To(true),
				}
				gpu.OwnerReferences = []metav1.OwnerReference{ref}
			}
			return nil
		})
		return err
	})
	if err != nil {
		ctrl.Log.Error(err, "failed to create or update GPU after retries", "gpu", gpu)
		return nil, err
	}

	err = retry.OnError(retry.DefaultBackoff, func(err error) bool {
		return true
	}, func() error {
		if err := k8sClient.Get(ctx, client.ObjectKeyFromObject(gpu), gpu); err != nil {
			ctrl.Log.Error(err, "failed to get GPU", "gpu", gpu)
			return err
		}
		gpu.Status.Capacity = &tfv1.Resource{
			Vram:   resource.MustParse(fmt.Sprintf("%dMi", memInfo.Total/1024/1024)),
			Tflops: tflops,
		}
		gpu.Status.UUID = uuid
		gpu.Status.GPUModel = deviceName
		gpu.Status.NodeSelector = map[string]string{
			constants.KubernetesHostNameLabel: k8sNodeName,
		}
		if gpu.Status.Available == nil {
			gpu.Status.Available = gpu.Status.Capacity.DeepCopy()
		}
		if gpu.Status.UsedBy == "" {
			gpu.Status.UsedBy = tfv1.UsedByTensorFusion
		}
		if gpu.Status.Phase == "" {
			gpu.Status.Phase = tfv1.TensorFusionGPUPhasePending
		}
		return k8sClient.Status().Patch(ctx, gpu, client.Merge)
	})
	if err != nil {
		ctrl.Log.Error(err, "failed to update status of GPU after retries", "gpu", gpu)
		return nil, err
	}

	return gpu, nil
}

func kubeClient() (client.Client, error) {
	kubeConfigEnvVar := os.Getenv("KUBECONFIG")
	var config *rest.Config
	var err error
	if kubeConfigEnvVar != "" {
		if strings.HasPrefix(kubeConfigEnvVar, "~") {
			homeDir, err := os.UserHomeDir()
			if err != nil {
				return nil, fmt.Errorf("get home directory %w", err)
			}
			kubeConfigEnvVar = filepath.Join(homeDir, strings.TrimPrefix(kubeConfigEnvVar, "~"))
		}
		config, err = clientcmd.BuildConfigFromFlags("", kubeConfigEnvVar)
	} else {
		config, err = rest.InClusterConfig()
	}
	if err != nil {
		return nil, fmt.Errorf("find cluster kubeConfig %w", err)
	}

	client, err := client.New(config, client.Options{
		Scheme: Scheme,
	})
	if err != nil {
		return nil, fmt.Errorf("create kubeClient %w", err)
	}
	return client, nil
}

func getTotalHostRAM() int64 {
	v, err := mem.VirtualMemory()
	if err != nil {
		fmt.Printf("[warning] getting memory info failed: %v\n", err)
		return 0
	}
	return int64(v.Total)
}

func getDiskInfo(path string) (total int64) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		fmt.Printf("[warning] getting disk path failed: %v\n", err)
		return 0
	}

	var stat syscall.Statfs_t
	err = syscall.Statfs(absPath, &stat)
	if err != nil {
		if errors.Is(err, syscall.ENOENT) {
			err = os.MkdirAll(absPath, 0o755)
			if err != nil {
				fmt.Printf("[warning] creating folder to discover disk space failed: %s, err: %v\n", absPath, err)
				return 0
			}
			err = syscall.Statfs(absPath, &stat)
			if err != nil {
				fmt.Printf("[warning] getting disk stats after creation failed: %v\n", err)
				return 0
			}
		} else {
			fmt.Printf("[warning] getting disk stats failed: %v\n", err)
			return 0
		}
	}

	total = int64(stat.Blocks * uint64(stat.Bsize))
	return total
}

// updateGPUNodeStatus conditionally updates GPUNode status fields
// Only updates phase if it's empty, and available resources if they are empty
func updateGPUNodeStatus(
	status *tfv1.GPUNodeStatus,
	totalTFlops, totalVRAM resource.Quantity,
	totalGPUs int32, deviceIDs []string) {
	// Always update these fields as they represent current state
	status.TotalTFlops = totalTFlops
	status.TotalVRAM = totalVRAM
	status.TotalGPUs = totalGPUs
	status.ManagedGPUs = totalGPUs
	status.ManagedGPUDeviceIDs = deviceIDs
	status.NodeInfo = tfv1.GPUNodeInfo{
		RAMSize:      *resource.NewQuantity(getTotalHostRAM(), resource.DecimalSI),
		DataDiskSize: *resource.NewQuantity(getDiskInfo(TMP_PATH), resource.DecimalSI),
	}
	// Only update phase if it's empty (unset)
	if status.Phase == "" {
		status.Phase = tfv1.TensorFusionGPUNodePhasePending
	}
}
