该task 的目的是为了支持tensor-fusion.ai/dedicated-gpu annotation，类似
annotations:
  tensor-fusion.ai/gpu-model: A100 
  tensor-fusion.ai/gpu-count: 8
  tensor-fusion.ai/dedicated-gpu: "true"

该annotation的目的是为了让当前请求的pod 忽略Tflops/Vram等条件，独占GPU的全部资源。
以下是实现的步骤

TODO Action Items
完整实现步骤
1. 在 pricingProvider 中新增根据 GPU 模式获取信息的方法
文件: internal/cloudprovider/pricing/pricing.go

在 PricingProvider 接口中添加新方法：

// 根据GPU模式获取GPU容量信息  
GetGPUCapacityByModel(gpuModel string) (*GPUCapacityInfo, bool)
定义新的结构体：

type GPUCapacityInfo struct {  
    TFlops resource.Quantity  
    VRAM   resource.Quantity  
}
在 StaticPricingProvider 中实现该方法

2. 修改 webhook 结构体
文件: internal/webhook/v1/pod_webhook.go

type TensorFusionPodMutator struct {  
    client.Client  
    decoder         admission.Decoder  
    portAllocator   *portallocator.PortAllocator  
    pricingProvider pricing.PricingProvider  // 新增字段  
}
3. 修改 ParseTensorFusionInfo 函数
文件: internal/webhook/v1/tf_parser.go

修改函数签名：

func ParseTensorFusionInfo(  
    ctx context.Context,  
    k8sClient client.Client,  
    pod *corev1.Pod,  
    pricingProvider pricing.PricingProvider,  // 新增参数  
) (utils.TensorFusionInfo, error)
在函数中添加独占 GPU 逻辑：

// 检测独占GPU注解并设置完整容量  
尝试将该逻辑抽象到一个私有方法中
dedicatedGPU, ok := pod.Annotations[constants.DedicatedGPUAnnotation]  
if ok && dedicatedGPU == constants.TrueStringValue {  
    if workloadProfile.Spec.GPUModel != "" {  
        // 使用pricingProvider获取GPU完整容量  
        if capacity, exists := pricingProvider.GetGPUCapacityByModel(workloadProfile.Spec.GPUModel, region); exists {  
            workloadProfile.Spec.Resources.Requests.Tflops = capacity.TFlops  
            workloadProfile.Spec.Resources.Requests.VRAM = capacity.VRAM  
        }  
    }  
}
4. 更新 Handle 方法调用
文件: internal/webhook/v1/pod_webhook.go

tfInfo, err := ParseTensorFusionInfo(ctx, m.Client, pod, m.pricingProvider)
5. 修改 main.go 中的 webhook 初始化
文件: cmd/main.go

创建 pricing provider 并传入 webhook：

pricingProvider := pricing.NewStaticPricingProvider()  
// 初始化 pricing provider...  
  
mutator := &v1.TensorFusionPodMutator{  
    Client:          mgr.GetClient(),  
    decoder:         admission.NewDecoder(mgr.GetScheme()),  
    portAllocator:   portAllocator,  
    pricingProvider: pricingProvider,  
}

6. 增加相关测试