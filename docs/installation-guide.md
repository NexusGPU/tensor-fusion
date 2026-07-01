# TensorFusion Kubernetes 安装说明

本文档说明如何在 Kubernetes 集群中安装 TensorFusion 控制面，并完成最小业务 Pod 验证。

适用假设：

- 使用本仓库 Helm chart 安装，chart 路径为 `charts/tensor-fusion`。
- Helm release 名称使用 `tensor-fusion-sys`，系统命名空间使用 `tensor-fusion-sys`。
- 集群已有 GPU 节点，节点能被 `nvidia.com/gpu.present=true` 这类 label 选中。
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
- GreptimeDB standalone。
- Alertmanager。
- cluster-agent、vector sidecar。

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
| `controller.image.tag` | operator 镜像 tag | `latest` |
| `controller.replicaCount` | controller 副本数 | `1` |
| `greptime.installStandalone` | 是否安装内置 GreptimeDB standalone | `true` |
| `greptime.host` | GreptimeDB MySQL endpoint host | `greptimedb-standalone.greptimedb.svc.cluster.local` |
| `alert.enabled` | 是否安装 alertmanager | `true` |
| `controller.admissionWebhooks.failurePolicy` | webhook 失败策略 | `Ignore` |

### 2.4 指定组件版本

Chart 默认所有组件镜像都使用浮动 tag `latest`（operator / hypervisor / vgpu-provider / client / worker）。生产环境建议**固定版本**，通过 `--set` 覆盖对应 values 键：

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys --create-namespace \
  --set controller.image.tag=2.11.3 \
  --set cluster.hypervisorImage=tensorfusion/tensor-fusion-hypervisor:2.11.3 \
  --set providerConfigs.nvidia.images.middleware=tensorfusion/vgpu-provider-nvidia:1.3.5 \
  --set providerConfigs.nvidia.images.remoteClient=tensorfusion/tensor-fusion-client:v2.15.1 \
  --set providerConfigs.nvidia.images.remoteWorker=tensorfusion/tensor-fusion-worker:v2.15.1
```

各组件对应的 values 键：

| 组件 | values 键 | 说明 |
| --- | --- | --- |
| operator | `controller.image.repository` + `controller.image.tag` | 仓库与 tag 分开两个键 |
| hypervisor | `cluster.hypervisorImage` | **完整镜像**（含仓库+tag），默认 `...tensor-fusion-hypervisor:latest` |
| vgpu-provider | `providerConfigs.nvidia.images.middleware` | 默认 `...vgpu-provider-nvidia:latest` |
| client | `providerConfigs.nvidia.images.remoteClient` | 默认 `...tensor-fusion-client:latest` |
| worker | `providerConfigs.nvidia.images.remoteWorker` | 默认 `...tensor-fusion-worker:latest` |

> 注意：`cluster.hypervisorImage` 传的是完整镜像引用（`仓库:tag`），而 operator 用 `repository` + `tag` 两个独立键。

若通过 Helm 仓库（而非本地 chart 路径）安装，可用 `--version` 固定 **chart 版本**（与镜像 tag 相互独立）：

```bash
helm repo add tensor-fusion https://helm.tensor-fusion.ai
helm repo update
helm install tensor-fusion-sys tensor-fusion/tensor-fusion --version 1.7.8 \
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
- 至少设置一个资源请求 annotation：`tensor-fusion.ai/tflops-request`、`tensor-fusion.ai/compute-percent-request` 或 `tensor-fusion.ai/vram-request`。
- 多容器 Pod 必须设置 `tensor-fusion.ai/inject-container`。

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
        tensor-fusion.ai/isolation: "shared"
        tensor-fusion.ai/is-local-gpu: "false"
        tensor-fusion.ai/gpu-count: "1"
        tensor-fusion.ai/tflops-request: "10"
        tensor-fusion.ai/vram-request: "1Gi"
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

进入业务容器检查 CUDA/NVML 注入：

```bash
POD=$(kubectl get pod -l app=tensor-fusion-smoke -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it "$POD" -c pytorch -- nvidia-smi
kubectl exec -it "$POD" -c pytorch -- sh -lc 'ldd "$(which nvidia-smi)" | grep -E "tensor-fusion|nvidia-ml|cuda"'
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
| `tensor-fusion.ai/gpu-count` | 请求 GPU 数量。 |
| `tensor-fusion.ai/gpu-indices` | 指定 GPU index，是硬过滤条件。指定后 `gpu-count` 按 index 数量计算。 |
| `tensor-fusion.ai/tflops-request` | 调度用 TFLOPs 请求，推荐优先使用。 |
| `tensor-fusion.ai/compute-percent-request` | 按百分比请求算力。与 TFLOPs request 互斥。 |
| `tensor-fusion.ai/vram-request` | 调度用显存请求。 |
| `tensor-fusion.ai/inject-container` | 多容器 Pod 中指定要注入的容器，多个容器用逗号分隔。 |

`shared` 模式是整卡共享、无运行时 TFLOPs 限制的模式，但调度阶段仍需要资源请求 annotation，用于 TensorFusion 选择 GPU 和记录分配。

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

升级：

```bash
helm upgrade tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys \
  -f <your-values.yaml>
```

更详细的升级和回滚流程见仓库根目录：

- `upgrade.md`
- `rollback.md`

卸载会删除 TensorFusion CR、控制面、webhook、CRD 等资源。确认不再需要后执行：

```bash
NAMESPACE=tensor-fusion-sys HELM_RELEASE=tensor-fusion-sys ./scripts/uninstall.sh
```

卸载前建议先备份业务和 TensorFusion CR 状态。
