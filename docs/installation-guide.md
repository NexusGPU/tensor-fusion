# TensorFusion Kubernetes 安装说明

本文档说明如何在 Kubernetes 集群中安装 TensorFusion 控制面，并完成最小业务 Pod 验证。

TensorFusion 官方镜像发布在 [Docker Hub](https://hub.docker.com/u/tensorfusion)。需要使用最新
版本时，可在该页面确认 operator、hypervisor、client 和 worker 的可用 tag，再覆盖 Helm
values 或 TensorFusionCluster/ProviderConfig 中的镜像字段。

适用假设：

- 使用本仓库 Helm chart 安装，chart 路径为 `charts/tensor-fusion`。
- Helm release 名称使用 `tensor-fusion-sys`，系统命名空间使用 `tensor-fusion-sys`。
- GPU 功能验证时，集群已有 GPU 节点，节点能被 `nvidia.com/gpu.present=true` 这类 label 选中。无 GPU 的控制面安装验证不要求此条件。
- 执行安装的账号具备创建 CRD、ClusterRole、MutatingWebhookConfiguration、Namespace 的权限。

## 1. 前置条件

安装前确认基础工具和集群状态：

```bash
kubectl version --client=true
helm version
kubectl get nodes -o wide
kubectl get nodes --show-labels | grep -E 'nvidia.com/gpu.present=true|gpu.present=true'
```

GPU 节点需要满足：

- NVIDIA 驱动正常。
- 容器运行时能挂载 GPU 设备。
- GPU 节点带有 TensorFusion 可识别的 label。默认 Helm 参数 `initialGpuNodeLabelSelector` 是 `nvidia.com/gpu.present=true`。
- 如果使用 remote 模式，client Pod 到 worker Pod 所在节点的网络需要可达。

## 2. 安装控制面

### 2.1 默认安装

默认 values 会安装：

- TensorFusion CRD、RBAC、controller。
- Mutating admission webhook。
- TensorFusion scheduler 配置。
- NVIDIA ProviderConfig、SchedulingConfigTemplate，以及默认的 NVIDIA TensorFusionCluster/GPUPool。
- GreptimeDB standalone。
- Alertmanager。
- vector sidecar。cluster-agent 默认关闭，仅在显式配置 `agent.agentId` 后启用。

无 GPU 的集群也可以安装控制面。只要没有节点命中
`initialGpuNodeLabelSelector`，operator 就不会创建 GPUNode 或 Hypervisor Pod；此时只验证
controller、CRD、RBAC、webhook 和默认 CR 是否正确即可。

从仓库根目录执行：

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys \
  --create-namespace
```

如果集群拉取海外镜像不稳定，使用国内镜像 values：

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys \
  --create-namespace \
  -f charts/tensor-fusion/values-cn.yaml
```

### 2.2 生产环境安装

生产环境建议使用外部高可用 GreptimeDB，而不是 chart 内置 standalone。

`charts/tensor-fusion/values-production.yaml` 只会设置 `greptime.installStandalone=false`，不会自动填充外部 GreptimeDB 地址和凭据。需要额外传入 `greptime.host`、`greptime.port`、`greptime.db`，如果使用云 Greptime，还要设置 `greptime.isCloud=true`、`greptime.user`、`greptime.password`。

示例：

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys \
  --create-namespace \
  -f charts/tensor-fusion/values-production.yaml \
  --set greptime.installStandalone=false \
  --set greptime.isCloud=true \
  --set greptime.host='<greptime-host>' \
  --set greptime.port=5001 \
  --set greptime.db='public' \
  --set greptime.user='<greptime-user>' \
  --set greptime.password='<greptime-password>'
```

如果外部 Greptime 不需要账号密码，则保持 `greptime.isCloud=false`，并设置可访问的 host、port、db。

### 2.3 常用 Helm 参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `initialGpuNodeLabelSelector` | controller 初始扫描 GPU 节点的 label selector | `nvidia.com/gpu.present=true` |
| `controller.image.repository` | operator 镜像仓库 | `tensorfusion/tensor-fusion-operator` |
| `controller.image.tag` | operator 镜像 tag；空值时使用 Chart `appVersion` | `""`（当前解析为 `2.15.0`） |
| `controller.replicaCount` | controller 副本数 | `1` |
| `controller.compatibleWithNvidiaContainerToolkit` | 启用 NVIDIA Container Toolkit 校验及驱动库兼容挂载 | `false` |
| `greptime.installStandalone` | 是否安装内置 GreptimeDB standalone | `true` |
| `greptime.host` | GreptimeDB MySQL endpoint host | `greptimedb-standalone.greptimedb.svc.cluster.local` |
| `alert.enabled` | 是否安装 alertmanager | `true` |
| `controller.admissionWebhooks.failurePolicy` | webhook 失败策略 | `Fail` |

安装了 NVIDIA GPU Operator，且 GPU 节点提供
`/run/nvidia/validations/toolkit-ready` 和 `driver-ready` 时，可设置：

```bash
--set controller.compatibleWithNvidiaContainerToolkit=true
```

这会启用 Toolkit 就绪检查和非标准驱动目录发现。没有这些 validation 文件的集群应保持默认
`false`。如需覆盖自动发现，可通过 `ProviderConfig.spec.hypervisor.extraEnv` 显式设置
`TF_CUDA_LIB_PATH` 和 `TF_NVML_LIB_PATH`。

### 2.4 指定组件版本

Chart 1.8.3 起默认固定与 `appVersion=2.15.0` 配套的组件版本，一键安装不再依赖
operator / hypervisor / vgpu-provider / client / worker 的浮动 `latest`。需要使用私有仓库或
其它已验证版本时，可通过 `--set` 覆盖：

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys --create-namespace \
  --set controller.image.tag=2.15.0 \
  --set cluster.hypervisorImage=tensorfusion/tensor-fusion-hypervisor:2.15.0 \
  --set providerConfigs.nvidia.images.middleware=tensorfusion/vgpu-provider-nvidia:1.3.9 \
  --set providerConfigs.nvidia.images.remoteClient=tensorfusion/tensor-fusion-client:v2.25.0 \
  --set providerConfigs.nvidia.images.remoteWorker=tensorfusion/tensor-fusion-worker:v2.25.0
```

各组件对应的 values 键：

| 组件 | values 键 | 说明 |
| --- | --- | --- |
| operator | `controller.image.repository` + `controller.image.tag` | 仓库与 tag 分开两个键 |
| hypervisor | `cluster.hypervisorImage` | **完整镜像**（含仓库+tag），默认 `tensorfusion/tensor-fusion-hypervisor:2.15.0` |
| vgpu-provider | `providerConfigs.nvidia.images.middleware` | 默认 `tensorfusion/vgpu-provider-nvidia:1.3.9` |
| client | `providerConfigs.nvidia.images.remoteClient` | 默认 `tensorfusion/tensor-fusion-client:v2.25.0` |
| worker | `providerConfigs.nvidia.images.remoteWorker` | 默认 `tensorfusion/tensor-fusion-worker:v2.25.0` |

> 注意：`cluster.hypervisorImage` 传的是完整镜像引用（`仓库:tag`），而 operator 用 `repository` + `tag` 两个独立键。

若通过 Helm 仓库（而非本地 chart 路径）安装，可用 `--version` 固定 **chart 版本**（与镜像 tag 相互独立）：

```bash
helm repo add tensor-fusion https://nexusgpu.github.io/tensor-fusion --force-update
helm repo update tensor-fusion
helm upgrade --install tensor-fusion-sys tensor-fusion/tensor-fusion --version 1.8.3 \
  -n tensor-fusion-sys --create-namespace
```

## 3. 验证控制面

安装后检查系统组件：

```bash
kubectl get pods -n tensor-fusion-sys -o wide
kubectl get pods -n greptimedb -o wide
kubectl get crd | grep tensor-fusion.ai
kubectl get mutatingwebhookconfiguration | grep tensor-fusion
```

检查 controller 日志：

```bash
kubectl logs -n tensor-fusion-sys deploy/tensor-fusion-sys-controller -c controller --tail=100
```

正常情况下，controller Pod 至少包含 `controller` 和 `vector` 容器；如果 values 中配置了 `agent.agentId`，还会有 `cluster-agent`。

无 GPU 环境还应确认没有误建 Hypervisor：

```bash
kubectl get pods -n tensor-fusion-sys -o json \
  | jq '[.items[] | select(any(.spec.containers[]?; .name == "tensor-fusion-hypervisor"))] | length'
```

没有节点命中 GPU selector 时，输出必须为 `0`。默认 TFC/GPUPool 因没有 GPUNode 可能保持
`Updating`，不影响 controller-only 安装验收。

## 4. 创建 TensorFusionCluster 和 GPUPool

控制面安装完成后，需要创建 TensorFusionCluster 或 GPUPool，TensorFusion 才能发现和管理 GPU 资源。

> 从 v1.7.8 起，默认 Helm 安装（`cluster.enabled=true`）会自动创建一个 NVIDIA
> `TensorFusionCluster`（单 NVIDIA pool）。**纯 NVIDIA 集群开箱即用，无需手动创建**；
> 其它厂家或多卡场景见下面 4.1 / 4.2。

`config/samples/` 下提供了各厂家真实可用的样例（ProviderConfig + TensorFusionCluster），
`componentConfig` 已是与当前版本匹配的完整配置，可直接 apply。

小贴士：

- 使用 TensorFusionCluster 创建 pool 时，实际 GPUPool 名称通常是 `<cluster-name>-<pool-name>`。
- 业务 Pod 不显式写 `tensor-fusion.ai/gpupool` 时，会落到 `isDefault: true` 的 pool。

**节点切分/隔离能力**通过 GPUPool 的 `nodeManagerConfig` 统一配置。未命中规则的节点
使用 `defaultIsolationMode`；规则按顺序匹配，第一条命中后停止：

```yaml
nodeManagerConfig:
  defaultIsolationMode: soft
  isolationModeRules:
    - mode: partitioned
      selector:
        matchExpressions:
          - key: node.kubernetes.io/instance-type
            operator: In
            values: [p4d.24xlarge, p5.48xlarge]
    - mode: hard
      selector:
        matchLabels:
          tensor-fusion.ai/gpu-model: H100
```

修改默认模式或 selector 规则会进入 GPUPool 现有的 Hypervisor 滚动更新流程。
`autoUpdateHypervisor=false` 时只标记配置未同步；启用后按 `batchPercentage` 和
`batchInterval` 分批更新，并且只重建有效模式发生变化的节点。

soft、hard、partitioned 是互斥的节点能力。`shared` 从 v2.13.0 起表示整卡分配策略，
不是必须单独规划的第四种节点能力：shared workload 可以使用上述任一模式节点上的完整
空闲、未分区 GPU。节点的有效模式完全由 GPUPool 策略决定。

### 4.1 按厂家安装（单一厂家）

| 厂家 | 命令 |
| --- | --- |
| NVIDIA | 默认已创建；或 `kubectl apply -f config/samples/provider-nvidia.yaml -f config/samples/tensorfusioncluster-nvidia.yaml` |
| Ascend | `kubectl apply -f config/samples/provider-ascend.yaml -f config/samples/tensorfusioncluster-ascend.yaml` |
| MooreThreads | `kubectl apply -f config/samples/provider-mthreads.yaml -f config/samples/tensorfusioncluster-mthreads.yaml` |
| PPU | `kubectl apply -f config/samples/provider-ppu.yaml -f config/samples/tensorfusioncluster-ppu.yaml` |

非 NVIDIA 集群安装控制面时，建议关掉默认 NVIDIA cluster，避免多出一个无节点的 pool：

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys --create-namespace \
  --set cluster.enabled=false --set providerConfigs.nvidia.enabled=false
# 再 apply 对应厂家的 provider + tensorfusioncluster
```

**在线安装（Helm 仓库，无需 clone 仓库）**：把 `./charts/tensor-fusion` 换成
`tensor-fusion/tensor-fusion --version <chart 版本>`；样例 CR 不打包进 chart，用仓库
raw URL apply（或先 clone 仓库）。以 Ascend 为例：

```bash
helm repo add tensor-fusion https://nexusgpu.github.io/tensor-fusion --force-update
helm repo update tensor-fusion
helm upgrade --install tensor-fusion-sys tensor-fusion/tensor-fusion --version 1.8.3 \
  -n tensor-fusion-sys --create-namespace \
  --set cluster.enabled=false \
  --set providerConfigs.nvidia.enabled=false \
  --set initialGpuNodeLabelSelector=huawei.com/npu.present=true
RAW=https://raw.githubusercontent.com/NexusGPU/tensor-fusion/main/config/samples
kubectl apply -f $RAW/provider-ascend.yaml -f $RAW/tensorfusioncluster-ascend.yaml
```

换厂家只改 `initialGpuNodeLabelSelector` + 两个文件名：

| 厂家 | selector | 文件 |
| --- | --- | --- |
| Ascend | `huawei.com/npu.present=true` | provider-ascend / tensorfusioncluster-ascend |
| MooreThreads | `mthreads.com/gpu.present=true` | provider-mthreads / tensorfusioncluster-mthreads |
| PPU | `aliyun.com/ppu.present=true` | provider-ppu / tensorfusioncluster-ppu |

### 4.2 多厂家（一个集群多种卡）

一个 Kubernetes 集群只有一个 `TensorFusionCluster`；多厂家时在 `spec.gpuPools` 下放多个
pool（仅一个 `isDefault: true`）。现成例子 `config/samples/tensorfusioncluster-multi-vendor.yaml`：

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys --create-namespace \
  --set cluster.enabled=false \
  --set initialGpuNodeLabelSelector=tensor-fusion.ai/watch-node=true
kubectl apply -f config/samples/provider-nvidia.yaml \
  -f config/samples/provider-ascend.yaml \
  -f config/samples/provider-mthreads.yaml \
  -f config/samples/tensorfusioncluster-multi-vendor.yaml
```

**节点标签（`initialGpuNodeLabelSelector`）**——决定 operator watch 哪些节点：

- **单一厂家**：用该厂家的 present 标签即可（默认 `nvidia.com/gpu.present=true`），
  **无需额外打标签**——该标签由厂家的 device-plugin / feature-discovery 自动打，TensorFusion 不负责打。
- **多厂家**：各厂家 present 标签不同（`nvidia.com/gpu.present` / `huawei.com/npu.present` /
  `mthreads.com/gpu.present` / `aliyun.com/ppu`），单个标签选不全。改用 vendor 无关的
  `tensor-fusion.ai/watch-node=true`，并**手动给每个 GPU 节点打这个标签**（该标签不会自动生成）：

  ```bash
  kubectl label node <gpu-node> tensor-fusion.ai/watch-node=true
  ```

- 或把 `initialGpuNodeLabelSelector` 留空（`--set initialGpuNodeLabelSelector=""`），
  operator 会 watch 所有节点（无需打标签，但大集群有一定性能开销）。

应用配置后检查资源：

```bash
kubectl get tensorfusioncluster
kubectl get gpupool
kubectl get gpunode,gpu -o wide
kubectl describe gpupool <gpupool-name>
```

## 5. 部署业务 Pod 验证

业务 Pod 必须满足以下条件，webhook 才会注入 TensorFusion client：

- Pod label 有 `tensor-fusion.ai/enabled: "true"`。
- 切片请求至少设置一个资源 annotation：`tensor-fusion.ai/tflops-request`、`tensor-fusion.ai/compute-percent-request` 或 `tensor-fusion.ai/vram-request`。标准 shared 整卡请求设置 `isolation: "shared"`、`dedicated-gpu: "true"` 和 `gpu-model`，无需用户填写 TFLOPS/VRAM；`gpu-count` 未填写时默认为 1，建议显式填写。
- 多容器 Pod 必须设置 `tensor-fusion.ai/inject-container`。

先查询目标 pool 中 operator 已识别的准确型号；下面的 `<gpu-model>` 必须替换为
`GPU.status.gpuModel` 中的完整值：

```bash
kubectl get gpu -l tensor-fusion.ai/gpupool=<gpupool-name> \
  -o custom-columns='NAME:.metadata.name,MODEL:.status.gpuModel,CAPACITY:.status.capacity'
```

示例 Deployment：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensor-fusion-smoke
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tensor-fusion-smoke
  template:
    metadata:
      labels:
        app: tensor-fusion-smoke
        tensor-fusion.ai/enabled: "true"
      annotations:
        tensor-fusion.ai/gpupool: "<gpupool-name>"
        tensor-fusion.ai/inject-container: "pytorch"
        tensor-fusion.ai/dedicated-gpu: "true"
        tensor-fusion.ai/isolation: "shared"
        tensor-fusion.ai/is-local-gpu: "false"
        tensor-fusion.ai/gpu-count: "1"
        tensor-fusion.ai/gpu-model: "<gpu-model>"
    spec:
      containers:
        - name: pytorch
          image: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
          command: ["sh", "-c", "sleep infinity"]
```

部署并检查：

```bash
kubectl apply -f smoke.yaml
kubectl get pod -l app=tensor-fusion-smoke -o wide
kubectl describe pod -l app=tensor-fusion-smoke
kubectl get tensorfusionworkload,tensorfusionconnection -A
kubectl get pods -A -l tensor-fusion.ai/component=worker -o wide
```

进入业务容器实际调用 CUDA/NVML；仅 Pod 进入 Running/Ready 不代表 GPU 链路验证通过：

```bash
POD=$(kubectl get pod -l app=tensor-fusion-smoke -o jsonpath='{.items[0].metadata.name}')
WORKER=$(kubectl get pod \
  -l tensor-fusion.ai/component=worker,tensor-fusion.ai/workload=tensor-fusion-smoke \
  -o jsonpath='{.items[0].metadata.name}')
kubectl get pod "$WORKER" -o jsonpath='{.metadata.annotations.tensor-fusion\.ai/gpu-ids}{"\n"}'
kubectl exec -it "$POD" -c pytorch -- nvidia-smi
kubectl exec "$POD" -c pytorch -- python3 -c \
  'import ctypes; cuda=ctypes.CDLL("libcuda.so.1"); count=ctypes.c_int(); print("cuInit=", cuda.cuInit(0)); print("cuDeviceGetCount=", cuda.cuDeviceGetCount(ctypes.byref(count)), "count=", count.value)'
kubectl exec "$POD" -c pytorch -- python3 -c \
  'import torch; print("count=", torch.cuda.device_count()); x=torch.ones(1024, device="cuda"); print("sum=", x.sum().item()); print("uuid=", torch.cuda.get_device_properties(0).uuid)'
kubectl exec "$POD" -c pytorch -- nvidia-smi \
  --query-gpu=uuid,name,memory.total --format=csv,noheader
```

shared 整卡用例应看到 `cuInit=0`、CUDA device count 与 `gpu-count` 一致。上面的 remote
示例由 worker Pod 持有 `tensor-fusion.ai/gpu-ids`；local 示例则由业务 Pod 持有。CUDA/NVML
应返回同一 GPU UUID（忽略 GPU CR 名称的 `gpu-` 前缀及 NVML 的 `GPU-` 格式前缀）。验证后
删除 Deployment，并确认 TensorFusionWorkload/Connection 被清理、GPU available 恢复为 capacity：

```bash
kubectl delete deployment tensor-fusion-smoke
kubectl get tensorfusionworkload,tensorfusionconnection -A
kubectl get gpu -l tensor-fusion.ai/gpupool=<gpupool-name> \
  -o custom-columns='NAME:.metadata.name,CAPACITY:.status.capacity,AVAILABLE:.status.available'
```

如果容器里配置了 `http_proxy` 或 `https_proxy`，本地服务探测建议使用：

```bash
curl --noproxy "*" http://127.0.0.1:8000/v1/models
```

## 6. 常见 workload annotation

| Annotation | 说明 |
| --- | --- |
| `tensor-fusion.ai/gpupool` | 指定 GPUPool。未指定时依赖默认 pool。 |
| `tensor-fusion.ai/is-local-gpu` | `true` 表示业务 Pod 调度到 GPU 节点本地用卡；`false` 表示 remote 模式。 |
| `tensor-fusion.ai/isolation` | `shared`、`soft`、`hard`、`partitioned`。 |
| `tensor-fusion.ai/dedicated-gpu` | `true` 表示按 `gpu-model` 补齐整卡容量；它本身不会把 isolation 改为 shared。标准 shared 请求需要同时设置 `isolation=shared`。 |
| `tensor-fusion.ai/gpu-count` | 请求 GPU 数量。 |
| `tensor-fusion.ai/gpu-model` | 请求的 GPU 型号；使用 `dedicated-gpu=true` 时必须填写，应使用 `GPU.status.gpuModel` 的准确值，并确保 operator 已加载该型号容量。 |
| `tensor-fusion.ai/gpu-indices` | 指定 GPU index，是硬过滤条件。指定后 `gpu-count` 按 index 数量计算。 |
| `tensor-fusion.ai/tflops-request` | 调度用 TFLOPs 请求，推荐优先使用。 |
| `tensor-fusion.ai/compute-percent-request` | 按百分比请求算力。与 TFLOPs request 互斥。 |
| `tensor-fusion.ai/vram-request` | 调度用显存请求。 |
| `tensor-fusion.ai/inject-container` | 多容器 Pod 中指定要注入的容器，多个容器用逗号分隔。 |

`shared` 是完整空闲 GPU 的整卡分配策略，不挂载 soft limiter 或 hard preload。用户无需填写
TFLOPS/VRAM request/limit；webhook 会根据 `gpu-model` 补齐整卡容量，allocator 分配后会将
所选 GPU CR 的 available TFLOPS/VRAM 都扣减为 0。shared 不优先选择带 shared label 的节点，
仍沿用现有 GPU/节点 binpack 评分。

## 7. 排障

### 7.1 Pod Pending

先看 Kubernetes 调度事件：

```bash
kubectl describe pod <pod-name>
kubectl get nodes -o wide
kubectl get nodes --show-labels
```

常见原因：

- 节点 taint 未配置 toleration。
- Pod 的 nodeSelector、nodeAffinity 和 GPU 节点不匹配。
- GPUPool 的 nodeSelector 没选中节点。
- `tensor-fusion.ai/gpu-indices` 指定的卡不存在或已被其他 workload 占用。
- GPU 资源请求超过 pool 中可用资源。

继续检查 TensorFusion 资源：

```bash
kubectl get gpupool,gpunode,gpu -o wide
kubectl describe gpupool <gpupool-name>
kubectl describe gpu <gpu-name>
kubectl logs -n tensor-fusion-sys deploy/tensor-fusion-sys-controller -c controller --tail=200
```

### 7.2 CUDA/NVML driver 库路径异常

如果出现 `nvidia-smi` 报 `NVML Function Not Found`，或者 worker 日志中出现类似
`Assertion failed, cuInit_fn != nullptr`，先确认宿主机和出错容器内实际存在的 driver 库路径。

宿主机上检查 NVIDIA driver 是否正常，以及 driver 库安装在哪个目录：

```bash
nvidia-smi
ldconfig -p | grep -E 'libcuda.so.1|libnvidia-ml.so.1'
ls -l \
  /usr/lib64/libcuda.so.1 \
  /usr/lib64/libnvidia-ml.so.1 \
  /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 2>/dev/null
```

`/usr/lib64` 常见于 RHEL/CentOS/Rocky/openEuler/Kylin 系，`/usr/lib/x86_64-linux-gnu`
常见于 Debian/Ubuntu 系。Pod 中最终能看到哪些路径，还取决于 NVIDIA container runtime 的挂载结果和容器镜像自身的发行版。

然后在出错的 Pod 容器内确认实际链接和可用路径：

```bash
ldd "$(which nvidia-smi)" | grep -E 'tensor-fusion|nvidia-ml|cuda'
cat /etc/ld.so.preload 2>/dev/null || true
ldconfig -p 2>/dev/null | grep -E 'libcuda.so.1|libnvidia-ml.so.1' || true
ls -l \
  /usr/lib64/libcuda.so.1 \
  /usr/lib64/libnvidia-ml.so.1 \
  /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  /usr/local/nvidia/lib64/libcuda.so.1 \
  /usr/local/nvidia/lib64/libnvidia-ml.so.1 2>/dev/null
```

如果需要全量查找：

```bash
find /usr /lib /lib64 /usr/local \( -name 'libcuda.so*' -o -name 'libnvidia-ml.so*' \) 2>/dev/null
```

如果 TensorFusion worker 需要显式指定 driver 库路径，使用出错容器内真实存在的稳定 `.so.1` symlink：

```yaml
env:
  - name: TF_CUDA_LIB_PATH
    value: <container-visible-libcuda.so.1-path>
  - name: TF_NVML_LIB_PATH
    value: <container-visible-libnvidia-ml.so.1-path>
```

不要固定到类似 `/usr/lib64/libnvidia-ml.so.580.126.09` 或
`/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.216.01` 这样的版本号路径。驱动升级后，版本号文件可能变化，`.so.1` symlink 才会跟随当前驱动。

如果日志是 `cuInit_fn != nullptr`，重点检查 `TF_CUDA_LIB_PATH` 指向的 `libcuda.so.1` 是否在出错容器内存在，并且包含 `cuInit` 符号：

```bash
readelf -Ws <container-visible-libcuda.so.1-path> 2>/dev/null | grep ' cuInit'
```

如果宿主机有 driver 库但 Pod 内没有，优先排查 NVIDIA container runtime、NVIDIA device plugin 和 Pod 是否真正分配到了 GPU 设备。如果 Pod 内有正确库但仍报错，确认这些环境变量注入到了真正崩溃的 worker 容器，而不是只注入到了业务容器。

### 7.3 Pod 被 Evicted

如果事件里出现 `The node was low on resource: memory`，说明是节点内存压力驱逐，不是 GPU 调度失败。

检查：

```bash
kubectl describe pod <pod-name>
kubectl top pod <pod-name> --containers
kubectl top node <node-name>
```

处理方式：

- 给业务容器设置合理的 memory request 和 limit。
- remote 模式下，模型加载期间 client 侧可能有额外内存峰值，需要按实际压测结果预留。
- 避免在低内存节点上启动大模型。

### 7.4 vLLM 启动后本地 curl 无返回

先确认端口是否监听：

```bash
ps -ef | grep -E 'vllm|EngineCore|APIServer' | grep -v grep
ss -lntp | grep 8000
tail -n 100 /tmp/vllm.log
```

如果 `curl` 被代理环境变量影响，使用：

```bash
curl --noproxy "*" -v --max-time 10 http://127.0.0.1:8000/v1/models
```

如果日志里有 CUDA OOM，降低 `--gpu-memory-utilization`、减小 `--max-model-len`，或释放同一张 GPU 上其他进程。

## 8. 升级、回滚、卸载

升级前建议备份当前 Helm values 和关键 CR：

```bash
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "backup-${TS}"
helm -n tensor-fusion-sys get values tensor-fusion-sys -o yaml > "backup-${TS}/values.yaml"
kubectl get tensorfusioncluster,gpupool,gpunode,gpu,workloadprofile,tensorfusionworkload,tensorfusionconnection -A -o yaml \
  > "backup-${TS}/tf-state.yaml"
```

同一架构内的常规 Chart 升级：

```bash
helm upgrade tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys \
  -f <your-values.yaml>
```

此命令会更新同一 Helm release 中的数据库、监控等组件，**不能用于 v1→v2 架构升级**。
v1→v2 必须使用精确升级流程：只从固定版本 Helm 包中提取并 apply CRD、RBAC 和
scheduler ConfigMap，在 `operator=0` 窗口内同步切换 operator/Hypervisor 镜像与
GPUPool 隔离策略，不执行整包 `helm upgrade`。

详细步骤及回退方案见：

- `docs/v1-to-v2-upgrade-best-practice.md`

卸载脚本支持 Helm、`helm template | kubectl apply` 和 `make deploy` / Kustomize
三种安装方式。它会删除集群内全部 TensorFusion CR、控制面、webhook、CRD、
TensorFusion Node 标签、NodeOverlay、PVC，以及 Helm、Kustomize 和 GreptimeDB
使用的 namespace。确认不再需要其中的数据和业务后执行：

```bash
NAMESPACE=tensor-fusion-sys HELM_RELEASE=tensor-fusion-sys ./scripts/uninstall.sh
```

运行前可以查看完整参数和删除范围：

```bash
./scripts/uninstall.sh --help
```

默认值：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `NAMESPACE` | `tensor-fusion-sys` | Helm / manifest 控制面 namespace |
| `KUSTOMIZE_NAMESPACE` | `tensor-fusion` | `make deploy` 使用的 namespace |
| `HELM_RELEASE` | `tensor-fusion-sys` | Helm release 名称 |
| `RESOURCE_PREFIX` | 与 `HELM_RELEASE` 相同 | manifest 资源名和 PVC 前缀 |
| `WAIT_TIMEOUT_SECONDS` | `600` | 等待 CR finalizer 和资源删除的超时秒数 |

非默认 release 名称或 `helm template` 资源前缀可以额外设置：

```bash
NAMESPACE=<namespace> \
HELM_RELEASE=<release-name> \
RESOURCE_PREFIX=<rendered-resource-prefix> \
./scripts/uninstall.sh
```

脚本按 finalizer 依赖顺序等待资源删除，任一步失败都会停止，不会继续删除
controller 或 CRD。它会清理 TensorFusion Node 标签、taint、NodeOverlay、
可调度的 `tensor-fusion.ai/index*` 容量以及 TensorFusion PVC/PV。device plugin
断开后，kubelet 会经过约 5 分钟 grace period 清理内部 endpoint 和 checkpoint；
期间 Node status 可能保留非零 capacity，但 allocatable 已变为 `0`。宽限期结束后，
kubelet 会有意保留值为 `0` 的 index key。脚本随后将这些 key 从 Node status 删除，
并经过一个状态同步窗口确认 kubelet 不再写回。
如果超过 `WAIT_TIMEOUT_SECONDS` 仍未删除，脚本会报错。脚本不会删除
`nvidia.com/gpu.present`、`huawei.com/npu.present` 等厂家发现标签。卸载前必须
先备份业务、PVC 数据和 TensorFusion CR 状态。

`scripts/uninstall.sh` 可以作为单文件复制到目标机器执行，不要求同时下载完整仓库。
当前 Helm 和 Kustomize 默认安装产生的集群级 RBAC、webhook、CRD 等资源名已内置
在脚本中。如果脚本旁边存在 `charts/tensor-fusion` 或 `config/default`，还会额外
根据本地 manifest 执行一次兼容性清理；这两个目录不是单文件卸载的必需依赖。
