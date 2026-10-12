# TensorFusion Kubernetes 安装说明

本文档说明如何在 Kubernetes 集群中安装 TensorFusion 控制面，并完成最小业务 Pod 验证。

TensorFusion 官方镜像发布在 [Docker Hub](https://hub.docker.com/u/tensorfusion)。需要使用最新
版本时，可在该页面确认 operator、hypervisor、client 和 worker 的可用 tag，再覆盖 Helm
values 或 TensorFusionCluster/ProviderConfig 中的镜像字段。

适用假设：

- 使用本仓库 Helm chart 安装，chart 路径为 `charts/tensor-fusion`。
- Helm release 名称使用 `tensor-fusion-sys`，系统命名空间使用 `tensor-fusion-sys`。
- GPU 功能验证时，集群已有 GPU 节点，节点能被 `nvidia.com/gpu.present=true` 这类 label 选中。无 GPU 的控制面安装验证不要求此条件。
- 执行安装的账号具备创建 CRD、ClusterRole、MutatingWebhookConfiguration、ValidatingWebhookConfiguration、Namespace 的权限。

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
- Validating admission webhook（`<release>-validating-webhook`），只处理 `pods/ephemeralcontainers` 的更新，见 2.5。
- TensorFusion scheduler 配置。
- NVIDIA ProviderConfig、SchedulingConfigTemplate，以及默认的 NVIDIA TensorFusionCluster/GPUPool。
- GreptimeDB standalone。
- Alertmanager。
- vector sidecar（镜像固定为 `timberio/vector:0.56.0-alpine`）。controller Pod 始终带有 vector sidecar；每个 hypervisor Pod 默认也带一个（`cluster.enableVector`，见 2.5）。cluster-agent 默认关闭，仅在显式配置 `agent.agentId` 后启用。

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
| `controller.image.tag` | operator 镜像 tag；空值时使用 Chart `appVersion` | `""`（当前解析为 `2.17.3`） |
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

Chart 1.8.3 起默认固定与 Chart `appVersion`（当前为 `2.17.3`）配套的组件版本，一键安装不再依赖
operator / hypervisor / vgpu-provider / client / worker 的浮动 `latest`。需要使用私有仓库或
其它已验证版本时，可通过 `--set` 覆盖：

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys --create-namespace \
  --set controller.image.tag=2.17.3 \
  --set cluster.hypervisorImage=tensorfusion/tensor-fusion-hypervisor:2.17.3 \
  --set providerConfigs.nvidia.images.middleware=tensorfusion/vgpu-provider-nvidia:1.3.9 \
  --set providerConfigs.nvidia.images.remoteClient=tensorfusion/tensor-fusion-client:v2.25.0 \
  --set providerConfigs.nvidia.images.remoteWorker=tensorfusion/tensor-fusion-worker:v2.25.0
```

各组件对应的 values 键：

| 组件 | values 键 | 说明 |
| --- | --- | --- |
| operator | `controller.image.repository` + `controller.image.tag` | 仓库与 tag 分开两个键 |
| hypervisor | `cluster.hypervisorImage` | **完整镜像**（含仓库+tag）；空值时使用 `tensorfusion/tensor-fusion-hypervisor:<appVersion>`，与 operator 同版本 |
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

### 2.5 本版本新增的安装行为

**临时容器的准入校验。** chart 新增一个 `ValidatingWebhookConfiguration`，只匹配 `pods/ephemeralcontainers` 的 `UPDATE`，不带 namespace 或 label 选择器。handler 只对挂了 TensorFusion 节点数据卷（`/run/tensor-fusion` 下的 hostPath）的 Pod 生效：临时容器不能以注入方式之外的形式挂载该卷，也不能自行定义 `TF_SHM_POD_NAMESPACE`、`TF_SHM_POD_NAME`；其他 Pod 和普通的 `kubectl debug` 容器直接放行。失败策略由 `controller.admissionWebhooks.ephemeralContainers.failurePolicy` 控制，默认 `Ignore`（超时 3 秒），operator 不可用时不会阻塞集群内的 `kubectl debug`；对隔离要求高的集群可以设为 `Fail`。集群自身的 Pod 安全策略仍由管理员负责，见 `docs/pod-security.zh.md`。

**cluster-agent 的凭据。** `agent.enrollToken` 与 `agent.agentId` 不再出现在 Pod spec 中：chart 把它们写入 Secret `<release>-agent`，sidecar 通过 `secretKeyRef` 读取。原有的安装命令（`--set agent.agentId=... --set agent.enrollToken=...`，以及 `helm upgrade --reuse-values`）不需要修改。如果希望自己管理 Secret，改用：

```bash
kubectl -n tensor-fusion-sys create secret generic my-agent \
  --from-literal=enroll-token='<token>'

helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys \
  --set agent.agentId='<org>:<env>' \
  --set agent.existingSecret.name=my-agent \
  --set agent.existingSecret.key=enroll-token
```

`agent.existingSecret.agentIdKey` 可选：设置后 agent id 也从该 Secret 读取，`agent.agentId` 可以留空。chart 管理的 Secret 变化时，controller Pod 会随 `checksum/agent-secret` 注解自动重启；自管 Secret 轮换后需要手动重启 controller Pod。Alertmanager 配置同样改为保存在 Secret 中；使用 GreptimeDB Cloud 时，数据库密码只存在于 Secret `<release>-greptimedb-secret`，vector 配置通过环境变量 `GREPTIME_PASSWORD` 引用它。

**vector sidecar 默认开启。** hypervisor Pod 中的 vector 把 `tf_gpu_usage`、`tf_worker_usage` 和日志写入 GreptimeDB，每个 GPU 节点请求 50m CPU / 64Mi 内存，上限 1 CPU / 512Mi；hypervisor 的指标文件最多占用约 1.1 GB 临时存储。GPU 是否可用只取决于 hypervisor 容器，vector 异常不会让 GPU 退出调度。关闭方式：

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys --reuse-values \
  --set cluster.enableVector=false
```

关闭后这两张用量表不再有数据，依赖它们的告警规则和 autoscaler 也随之失效。开启、关闭或升级到本版本都会改变 hypervisor Pod 模板，每个 hypervisor Pod 会被重建一次；业务 Pod 不会重启。

**升级注意事项。** 升级前已经因为缺少 vector ConfigMap 而卡在 ContainerCreating 的 hypervisor Pod 不会被自动替换（该卷改为可选后不计入模板 hash，以免所有 hypervisor 重启），需要手动删除一次。升级顺序是先 hypervisor、后业务 Pod 内的 limiter 库；在两者版本不一致的窗口内，单次 launch 成本超过令牌桶的旧版 limiter Pod 不受限额约束，直到 Pod 重启。已知限制：单次持续数十秒的 launch 在新版 limiter 下会低于其配额运行（不会超出限额）；autoscaler 对从未上报过算力用量的 worker 不会下调算力。

**算力突发窗口。** soft 隔离下，空闲的业务 Pod 可以攒下一段按自身配额速率计的令牌，恢复运行时全速用掉，之后回到配额。攒多少由突发窗口 W（秒）决定，默认 0.1 s，范围 0 到 10 s。池级在 `schedulingConfigTemplate.elasticRateLimitParameters.burstWindow` 设置（字符串，例如 `"0.5"`；只设这一项即可，其余参数留空；hypervisor 在启动时读取该值，修改后已在运行的 hypervisor Pod 不会自动重建，需要删除它们才生效）；单个工作负载用 Pod 注解 `tensor-fusion.ai/compute-burst-window`（Go duration，例如 `500ms`、`2s`），可以在运行中的 Pod 上直接修改，不需要重启，超过 10 s、为负或格式错误的值在创建 Pod 时被拒绝。长期算力配额不受 W 影响：配额为 s 的 Pod 攒满后在空闲设备上最多全速运行约 `s × W / (1 − s)` 秒（W = 2 s 时，25% 配额约 0.67 s，50% 配额约 2 s），同一张卡上其他 Pod 只在这段时间内受到影响。

**指标表的时间列。** `tf_*` 表的时间索引列是 `greptime_timestamp`（由行协议写入时自动建表产生），不是 `ts`。chart 自带的告警规则只使用 `{{ .Conditions }}`，不需要修改；自己编写的告警规则如果直接写了 `ts`，需要改用模板变量 `{{ .TimeColumn }}`（或 `{{ timeColumn "表名" }}`），operator 会按表检测实际的列名，旧集群上名为 `ts` 的表也能继续使用。

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

### 7.5 `tf_gpu_usage` / `tf_worker_usage` 没有数据

Hypervisor 只把指标写到 `/logs/metrics.log`，由同一个 Pod 里的 `vector` sidecar 上传到 GreptimeDB。Hypervisor 看不到上传结果：数据库拒绝写入时，Hypervisor 日志里没有任何报错，要看 `vector` 容器的日志：

```bash
kubectl -n tensor-fusion-sys logs <hypervisor-pod> -c vector --since=10m \
  | grep -E 'Not retriable|Events dropped'
```

被拒绝时每次上传都有这样两行（`component_id` 是 sink 名，Controller 的 sidecar 对应 `sink_greptimedb_operator_metrics`）：

```text
ERROR sink{component_kind="sink" component_id=sink_greptimedb_hypervisor_metrics component_type=http}:request{request_id=12}: vector::sinks::util::retries: Not retriable; dropping the request. reason="Default retry strategy: Bad Request"
ERROR sink{component_kind="sink" component_id=sink_greptimedb_hypervisor_metrics component_type=http}:request{request_id=12}: vector_common::internal_event::component_events_dropped: Events dropped intentional=false count=7 reason="Service call failed. No retries or retries exhausted."
```

Vector 不打印数据库的应答。取一行指标手工提交，就能看到拒绝原因：

```bash
kubectl -n tensor-fusion-sys exec <hypervisor-pod> -c vector -- tail -n 1 /logs/metrics.log > line.txt
curl -sS -X POST --data-binary @line.txt \
  'http://<greptimedb-host>:4000/v1/influxdb/api/v2/write?db=public&precision=ms'
```

应答形如 `column memory_bytes expect type UInt64(UInt64Type), given: INT64(4)`，说明这一行里某个字段的类型和表里已有的列不一致：GreptimeDB 按第一次写入的类型建列，之后类型不同的行全部拒绝。`SHOW CREATE TABLE tf_gpu_usage;` 可以看到列类型。本版本 Hypervisor 写入的类型与旧版 Hypervisor 建的表一致（`memory_bytes` 为无符号整数，`power_usage` 为整数）；如果表是用其他类型建的，并且表里的历史数据可以丢弃，`DROP TABLE` 之后下一次写入会重新建表。Hypervisor 的 sidecar 每行指标单独提交一次，被拒绝的行不影响其他行。

### 7.6 Hypervisor 日志

启用 `vector` sidecar（`enableVector: true`）时，Hypervisor 把自己的日志同时写到 `/logs/pods/<pod>/hypervisor.log`（最多 3 个文件，每个 50 MB，保留 7 天），由 sidecar 写入 `tf_system_log`：

```sql
SELECT greptime_timestamp, pod, message FROM tf_system_log
WHERE component = 'hypervisor' AND container = 'hypervisor'
ORDER BY greptime_timestamp DESC LIMIT 50;
```

默认日志级别是 Info。每个控制周期一行的限速明细（`ERL [...] util=... rate=...`）在日志级别 4 才输出，有两种打开方式。

**临时打开，不重建 Pod。** 给 Hypervisor 进程发 `SIGUSR1`，日志级别升到 4，10 分钟后自动恢复为启动时的级别；这期间再发一次，10 分钟重新计时。启动时的级别已经不低于 4 时，信号不改变任何东西，只输出一行日志。升高和恢复各有一行日志（`Log verbosity raised from 0 to 4 for 10m0s`、`Log verbosity is back to 0`）。

Hypervisor Pod 使用宿主机的 PID 命名空间（`hostPID: true`），进程号不是 1，要先查出来。Hypervisor 是 Pod 的第一个容器，`kubectl exec` 默认进入它；镜像基于 `ubuntu:24.04`，带 shell 和 `pidof`：

```bash
kubectl -n tensor-fusion-sys exec <hypervisor-pod> -- sh -c 'kill -USR1 $(pidof hypervisor)'
```

也可以在节点上以 root 执行 `pkill -USR1 -x hypervisor`。本版本之前的 Hypervisor 不处理这个信号，收到后进程直接退出、容器重启，发送前先确认镜像版本。

**长期打开。** 通过 Pool 的 Hypervisor Pod 模板加启动参数 `-v=4`（`TensorFusionCluster` 的 `spec.gpuPools[].specTemplate.componentConfig.hypervisor.podTemplate`）：

```yaml
podTemplate:
  template:
    spec:
      containers:
        - name: tensor-fusion-hypervisor
          args: ["-v=4"]
```

修改模板会让该 Pool 的 Hypervisor Pod 逐个重建，排查结束后去掉参数时会再重建一次。`-log-output-path=none` 关闭日志文件。

### 7.7 已知问题

- **hard 隔离 + remote-vGPU `v2.26.0` 下 vLLM 0.28.0 启动失败。** client/worker 镜像为 `tensorfusion/tensor-fusion-client:v2.26.0`、`tensorfusion/tensor-fusion-worker:v2.26.0` 时，hard 隔离的 vLLM 0.28.0 报 `call api=cuMemcpy failed … named symbol not found`。缺陷在 remote-vGPU worker（独立组件），不在本 operator。
- **`shared` 隔离是整卡独占。** 同一张 GPU 上的两个 `shared` Pod 先后运行，不会同时运行。
- **算力限额约束的是设备时间占比，不是 solo 吞吐的比例。** 多个 Pod 共用一张卡时，GPU 在两个 context 之间每切换一次约耗时 65 µs（RTX 3090 实测），驱动把这段时间也计入利用率，而且大多记在占用引擎时间最长的 Pod 名下。因此配额较大的 Pod 达到自己的利用率目标时，实际得到的 kernel 时间约少 20%：与一个 30% 配额的邻居同卡，60% 配额实测为 solo 吞吐的 0.485，50% 为 0.385，40% 为 0.309。两个不设限的短 kernel 负载同卡时，合计吞吐约为单独运行一个的 0.83（长 kernel 负载 0.93～0.95；ONNX Runtime 的 CNN / encoder 推理和以拷贝为主的负载没有损失）。
- **混合架构集群上的 Helm hook Job。** 生成和注入 webhook 证书的两个 hook Job 默认只限定 `kubernetes.io/os: linux`，并沿用 controller 的 `affinity`、`tolerations`，不限定 CPU 架构。hook Pod 落到拉不到该镜像的节点时，`helm install` / `helm upgrade` 会一直等到超时。用 `controller.admissionWebhooks.patch.nodeSelector`（例如 `kubernetes.io/arch: amd64`）、`affinity`、`tolerations` 指定 hook Pod 的节点。

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

### 8.1 升级说明 / breaking changes

chart 与 operator、hypervisor 镜像要一起升级，不要用新 chart 搭配旧镜像：旧 operator 不提供临时容器校验的 webhook 路径（默认 `failurePolicy: Ignore` 下校验不生效，设为 `Fail` 时集群内所有 `kubectl debug` 会被拒绝），也不会把 `GREPTIME_PASSWORD` 注入 hypervisor Pod 的 vector 容器（`greptime.isCloud=true` 时该容器无法启动，用量指标中断）。

- **指标列类型。** hypervisor 写入 `tf_gpu_usage`、`tf_worker_usage` 的字段类型已固定：`memory_bytes` 为无符号整数，`power_usage`、`nvlink_rx`、`nvlink_tx` 为有符号整数，其余为浮点。表里已有列的类型与此不同的集群（例如曾用自定义参数运行 hypervisor、由其他写入方建表），升级后该表的每一行都会被数据库以 HTTP 400 拒绝，hypervisor 日志里看不到。按 7.5 的方法确认；处理办法是导出需要保留的历史数据后 `DROP TABLE`，表会在下一次写入时按新类型重建。operator 不会自动删表。
- **NodeOverlay 默认只匹配 AWS 上的 NVIDIA 机型。** `karpenterNodeOverlay.requirements` 的默认值是 `karpenter.k8s.aws/instance-gpu-count > 0` 且 `karpenter.k8s.aws/instance-gpu-manufacturer in [nvidia]`，不再对所有 NodePool 生效。其他云上的 Karpenter 或其他厂商的卡，升级后默认的 overlay 不匹配任何节点，需要自己设置 `karpenterNodeOverlay.requirements`。
- **`reserved` 容量类型的 Pool。** 内置价格表中没有预留实例价格的 13 个 AWS 机型不再被 `reserved` Pool 选中；需要这些机型时改用 `on-demand` 或在 ProviderConfig 中补充价格。
- **没有 owner 的空 TensorFusionWorkload 会被回收。** 由 Pod webhook 自动创建、没有 ownerReference、且已经 5 分钟没有任何 Pod 引用的 workload，升级后在第一次 reconcile 时被删除，其 `status.recommendation` 一并丢失。用户自己创建的 workload，以及带有 owner（Deployment、StatefulSet、Job、Pod 等）的 workload 不受影响。
- **hypervisor Pod 重建一次。** 开启 vector sidecar 的 Pool，升级后每个 hypervisor Pod 会因模板变化重建一次；业务 Pod 不重启。
- **Go 库接口（引用 `pkg/hypervisor`、`pkg/constants` 的项目）。** `(*APIClient).ResolveDeviceFp16TFlops` 已删除；`constants.TFDataPathWorkerExpr` 现在依赖容器内的 `TF_SHM_POD_NAMESPACE`、`TF_SHM_POD_NAME` 两个环境变量，使用它的调用方需要同时设置这两个变量。

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
