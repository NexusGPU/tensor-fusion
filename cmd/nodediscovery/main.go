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
	"github.com/samber/lo"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

var Scheme = runtime.NewScheme()

func init() {
	utilruntime.Must(tfv1.AddToScheme(Scheme))
}

func main() {
	var k8sNodeName string

	var gpuInfoConfig string
	flag.StringVar(&k8sNodeName, "hostname", "", "hostname")
	flag.StringVar(&gpuInfoConfig, "gpu-info-config", "", "specify the path to gpuInfoConfig file")

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

	gpuInfo, err := config.LoadGpuInfoFromFile(gpuInfoConfig)
	if err != nil {
		ctrl.Log.Error(err, "unable to read gpuInfoConfig file")
		gpuInfo = make([]config.GpuInfo, 0)
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
		info, ok := lo.Find(gpuInfo, func(info config.GpuInfo) bool {
			return info.FullModelName == deviceName
		})
		tflops := info.Fp16TFlops
		if !ok {
			ctrl.Log.Info(
				"[Error] Unknown GPU model, please update `gpu-public-gpu-info` configMap "+
					" to match your GPU model name in `nvidia-smi`, this may cause you workload stuck, "+
					"refer this doc to resolve it in detail: "+
					"https://tensor-fusion.ai/guide/troubleshooting/handbook"+
					"#pod-stuck-in-starting-status-after-enabling-tensorfusion",
				"deviceName", deviceName, "uuid", uuid)
			os.Exit(1)
		} else {
			ctrl.Log.Info("found GPU info from config", "deviceName", deviceName, "FP16 TFlops", tflops, "uuid", uuid)
		}

		gpu := createOrUpdateTensorFusionGPU(k8sClient, ctx, k8sNodeName, gpunode, uuid, deviceName, memInfo, tflops)

		totalTFlops.Add(gpu.Status.Capacity.Tflops)
		totalVRAM.Add(gpu.Status.Capacity.Vram)
		availableTFlops.Add(gpu.Status.Available.Tflops)
		availableVRAM.Add(gpu.Status.Available.Vram)
	}

	ns := nodeStatus(k8sNodeName)
	ns.TotalTFlops = totalTFlops
	ns.TotalVRAM = totalVRAM
	ns.AvailableTFlops = availableTFlops
	ns.AvailableVRAM = availableVRAM
	ns.TotalGPUs = int32(count)
	ns.ManagedGPUs = int32(count)
	ns.ManagedGPUDeviceIDs = allDeviceIDs
	ns.NodeInfo.RAMSize = *resource.NewQuantity(getTotalHostRAM(), resource.DecimalSI)
	ns.NodeInfo.DataDiskSize = *resource.NewQuantity(getDiskInfo(constants.TFDataPath), resource.DecimalSI)
	gpunode.Status = *ns

	err = retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		currentGPUNode := &tfv1.GPUNode{}
		if err := k8sClient.Get(ctx, client.ObjectKeyFromObject(gpunode), currentGPUNode); err != nil {
			return err
		}

		currentGPUNode.Status = *ns

		return k8sClient.Status().Update(ctx, currentGPUNode)
	})
	if err != nil {
		ctrl.Log.Error(err, "failed to update status of GPUNode after retries")
		os.Exit(1)
	}
}

func createOrUpdateTensorFusionGPU(
	k8sClient client.Client, ctx context.Context, k8sNodeName string, gpunode *tfv1.GPUNode,
	uuid string, deviceName string, memInfo nvml.Memory_v2, tflops resource.Quantity) *tfv1.GPU {
	gpu := &tfv1.GPU{
		ObjectMeta: metav1.ObjectMeta{
			Name: uuid,
		},
	}

	err := retry.OnError(retry.DefaultBackoff, func(err error) bool {
		return true // Retry on all errors for now
	}, func() error {
		_, err := controllerutil.CreateOrUpdate(ctx, k8sClient, gpu, func() error {
			// Set metadata fields
			gpu.Labels = map[string]string{
				constants.LabelKeyOwner: gpunode.Name,
			}
			gpu.Annotations = map[string]string{
				constants.GPULastReportTimeAnnotationKey: time.Now().Format(time.RFC3339),
			}

			if !metav1.IsControlledBy(gpu, gpunode) {
				gpu.OwnerReferences = []metav1.OwnerReference{
					*metav1.NewControllerRef(gpunode, gpunode.GroupVersionKind()),
				}
			}

			return nil
		})
		return err
	})
	if err != nil {
		ctrl.Log.Error(err, "failed to create or update GPU after retries", "gpu", gpu)
		os.Exit(1)
	}

	err = retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		if err := k8sClient.Get(ctx, client.ObjectKeyFromObject(gpu), gpu); err != nil {
			return err
		}

		newStatus := tfv1.GPUStatus{
			Phase: tfv1.TensorFusionGPUPhaseRunning,
			Capacity: &tfv1.Resource{
				Vram:   resource.MustParse(fmt.Sprintf("%dKi", memInfo.Total/1024)),
				Tflops: tflops,
			},
			UUID:     uuid,
			GPUModel: deviceName,
			NodeSelector: map[string]string{
				"kubernetes.io/hostname": k8sNodeName,
			},
		}

		if gpu.Status.Available == nil {
			newStatus.Available = newStatus.Capacity
		} else {
			newStatus.Available = gpu.Status.Available
		}
		gpu.Status = newStatus
		return k8sClient.Status().Update(ctx, gpu)
	})
	if err != nil {
		ctrl.Log.Error(err, "failed to update status of GPU after retries", "gpu", gpu)
		os.Exit(1)
	}

	return gpu
}

func nodeStatus(k8sNodeName string) *tfv1.GPUNodeStatus {
	return &tfv1.GPUNodeStatus{
		KubernetesNodeName: k8sNodeName,
		Phase:              tfv1.TensorFusionGPUNodePhaseRunning,
	}
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
		fmt.Printf("error getting memory info: %v\n", err)
		return 0
	}
	return int64(v.Total)
}

func getDiskInfo(path string) (total int64) {
	absPath, err := filepath.Abs(path)
	if err != nil {
		fmt.Printf("error getting disk path: %v\n", err)
		return 0
	}

	var stat syscall.Statfs_t
	err = syscall.Statfs(absPath, &stat)
	if err != nil {
		if errors.Is(err, syscall.ENOENT) {
			err = os.MkdirAll(absPath, 0o755)
			if err != nil {
				fmt.Printf("error creating folder: %s, err: %v\n", absPath, err)
				return 0
			}
			err = syscall.Statfs(absPath, &stat)
			if err != nil {
				fmt.Printf("error getting disk stats after creation: %v\n", err)
				return 0
			}
		} else {
			fmt.Printf("error getting disk stats: %v\n", err)
			return 0
		}
	}

	total = int64(stat.Blocks * uint64(stat.Bsize))
	return total
}
