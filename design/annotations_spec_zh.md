# TensorFusion Kubernetes 注解与标签规范

本文档提供了 TensorFusion 系统中使用的自定义 Kubernetes 注解（Annotations）和标签（Labels）的详尽指南。这些元数据键均以 `tensor-fusion.ai/` 为前缀，用于控制 GPU 资源调度、工作负载注入、隔离以及自动化的生命周期管理。

---

## 注解

### 资源请求与限制注解
这些注解由用户在 Pod 元数据中指定，用于定义精细化的 GPU 需求，超越了标准的基于整数的 `nvidia.com/gpu` 资源类型。

| 注解名称 | 键 (Key) | 描述 | 示例 |
| :--- | :--- | :--- | :--- |
| **GPU 数量** | `tensor-fusion.ai/gpu-count` | Pod 所需物理 GPU 的总数。 | `"2"` |
| **算力请求 (TFLOPS)** | `tensor-fusion.ai/tflops-request` | 请求的 GPU 计算性能 (TFLOPS)，用于算力切分。 | `"5.5"` |
| **算力限制 (TFLOPS)** | `tensor-fusion.ai/tflops-limit` | 最大计算性能限制 (TFLOPS)。 | `"10.0"` |
| **显存请求** | `tensor-fusion.ai/vram-request` | 请求的 GPU 显存大小。 | `"10Gi"`, `"500Mi"` |
| **显存限制** | `tensor-fusion.ai/vram-limit` | 最大 GPU 显存限制。 | `"20Gi"` |
| **算力百分比** | `tensor-fusion.ai/compute-percent-request` | 申请单个 GPU 的算力百分比（TFLOPS 的替代方案）。 | `"50"` |
| **GPU 厂商** | `tensor-fusion.ai/vendor` | 首选硬件供应商（如 `nvidia`, `amd`）。 | `"nvidia"` |
| **GPU 型号** | `tensor-fusion.ai/gpu-model` | 指定的 GPU 硬件型号。 | `"A100"`, `"H100"` |

---

### 调度与资源绑定
由 **TensorFusion 调度器** 使用，用于管理资源分配并跟踪绑定关系。

| 注解名称 | 键 (Key) | 描述 |
| :--- | :--- | :--- |
| **GPU ID 分配** | `tensor-fusion.ai/gpu-ids` | **(系统设置)** 调度器分配的物理 GPU ID 列表（逗号分隔）。 |
| **GPU 索引** | `tensor-fusion.ai/gpu-indices` | 用于手动绑定的特定物理 GPU 索引列表（逗号分隔）。 |
| **服务质量等级** | `tensor-fusion.ai/qos` | 调度与抢占优先级：`low`, `medium`, `high`, `critical`。 |
| **独占 GPU** | `tensor-fusion.ai/dedicated-gpu` | 若为 `"true"`，确保 Pod 独占所分配的物理卡。 |
| **隔离模式** | `tensor-fusion.ai/isolation` | 资源隔离策略：`soft` (默认), `shared`, `hard`, `partitioned`。 |
| **本地 GPU 模式** | `tensor-fusion.ai/is-local-gpu` | 是否与 Worker 同节点/同进程通信（本地 GPU 模式）。 |
| **资源已释放** | `tensor-fusion.ai/gpu-released` | **(系统设置)** 当 Pod 使用完 GPU 资源后由系统标记。 |

---

### Webhook 与注入配置
由 **Mutating Webhook** 使用，用于自动执行 sidecar 注入和运行时环境设置。

| 注解名称 | 键 (Key) | 描述 |
| :--- | :--- | :--- |
| **注入容器** | `tensor-fusion.ai/inject-container` | 需要注入 TensorFusion 的目标业务容器名称（逗号分隔）。 |
| **配置模板** | `tensor-fusion.ai/workload-profile` | 引用 `WorkloadProfile` CRD 以应用模板化配置。 |
| **多容器 GPU 映射** | `tensor-fusion.ai/container-gpu-count` | JSON 格式定义的每个容器的 GPU 数量分配。 |
| **多容器 GPU 绑定** | `tensor-fusion.ai/container-gpus` | JSON 格式为每个容器指定 GPU ID 列表。 |
| **Pod 索引** | `tensor-fusion.ai/index` | **(内部使用)** 为设备插件/CDI 通信注入的唯一索引 (1-512)。 |
| **定价信息** | `tensor-fusion.ai/hourly-pricing` | 用于基于成本的调度或驱逐保护计算。 |

---

### Worker 与连接管理
定义 TensorFusion Worker（远程 vGPU 服务端组件）的部署和连接方式。

| 注解名称 | 键 (Key) | 描述 |
| :--- | :--- | :--- |
| **嵌入式 Worker** | `tensor-fusion.ai/embedded-worker` | 若为 `"true"`，Worker 运行在业务进程内部（无独立容器）。 |
| **Sidecar Worker** | `tensor-fusion.ai/sidecar-worker` | 若为 `"true"`，Worker 以 Sidecar 容器形式运行（标准模式）。 |
| **独立 Worker** | `tensor-fusion.ai/dedicated-worker` | 若为 `"true"`，Worker 以完全独立的 Pod 形式运行。 |
| **工作负载模式** | `tensor-fusion.ai/workload-mode` | 定义生命周期行为：`dynamic` (动态) 或 `fixed` (固定)。 |
| **Worker 模板** | `tensor-fusion.ai/worker-pod-template` | 自定义 Worker Pod 模板（PodTemplateSpec，YAML 字符串），仅用于远程 vGPU 模式。 |
| **端口号** | `tensor-fusion.ai/port-number` | **(系统设置)** 实际分配到的 hostPort 端口号。 |

### 自动化与稳定性注解
用于自动扩缩容及在维护期间防止中断的注解。

| 注解/标签名称 | 键 (Key) | 描述 |
| :--- | :--- | :--- |
| **开启自动扩缩** | `tensor-fusion.ai/autoscale` | 设置为 `"true"` 以开启 GPU 资源垂直自动扩缩容。 |
| **扩缩目标** | `tensor-fusion.ai/autoscale-target` | 扩缩容目标资源：`compute` (算力), `vram` (显存) 或 `all` (全部)。 |
| **驱逐保护** | `tensor-fusion.ai/eviction-protection` | 防止 Pod 被抢占的保护时长（如 `"30m"`）。 |
| **灰度启用副本数** | `tensor-fusion.ai/enabled-replicas` | 灰度启用 TensorFusion 的 Pod 数量上限。 |

---

## 标签

### 管理与识别标签
用于资源筛选和跨控制器所有权跟踪的标准标签。

| 标签名称 | 键 (Key) | 描述 |
| :--- | :--- | :--- |
| **已启用** | `tensor-fusion.ai/enabled` | 为 Pod 或命名空间启用 TensorFusion 功能。 |
| **组件类型** | `tensor-fusion.ai/component` | 标识组件角色：`client`, `worker`, `hypervisor`, `operator`。 |
| **管理者** | `tensor-fusion.ai/managed-by` | 标识所属控制器（如 `operator`, `gpupool`）。 |
| **GPU 资源池** | `tensor-fusion.ai/gpupool` | 将 Pod 或节点关联到特定的 `GPUPool` 资源。 |
| **集群** | `tensor-fusion.ai/cluster` | 将资源关联到特定的 `TensorFusionCluster`。 |
| **自动分配主机端口** | `tensor-fusion.ai/host-port` | 设置为 `auto` 触发自动分配 hostPort。 |
| **端口名** | `tensor-fusion.ai/port-name` | 指定需要绑定 hostPort 的容器端口名。 |
| **不可中断** | `tensor-fusion.ai/do-not-disrupt` | 防止 Pod 或节点被调度器中断或迁移。 |

---

## 8. 常见使用场景与 YAML 示例

以下示例展示了如何基于标准的 Deployment 模板，通过组合不同的注解来实现 TensorFusion 的核心功能场景。

### 场景 0：自动分配 HostPort
通过标签触发 hostPort 自动分配，并指定要绑定的容器端口名。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-hostport-auto
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hostport
  template:
    metadata:
      labels:
        app: hostport
        tensor-fusion.ai/enabled: 'true'
        tensor-fusion.ai/gpupool: tensor-fusion-shared
        tensor-fusion.ai/host-port: auto
        tensor-fusion.ai/port-name: tf-service
    spec:
      containers:
        - name: client
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
          ports:
            - name: tf-service
              containerPort: 8080
```

### 场景 A：精细化 GPU 共享（切分 GPU）
通过切分算力（TFLOPS）和显存（VRAM），实现多个工作负载共享单块物理 GPU。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-fractional-gpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
        # 启用 TensorFusion 注入
        tensor-fusion.ai/enabled: 'true'
        # 指定 GPU 资源池
        tensor-fusion.ai/gpupool: tensor-fusion-shared
      annotations:
        # 申请 1 块 GPU 的切分资源
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/gpupool: tensor-fusion-shared
        tensor-fusion.ai/is-local-gpu: 'true'
        tensor-fusion.ai/tflops-request: 17800m # 请求 17.8 TFLOPS 算力
        tensor-fusion.ai/tflops-limit: 35600m   # 算力上限 35.6 TFLOPS
        tensor-fusion.ai/vram-request: 128Mi
        tensor-fusion.ai/vram-limit: 1Gi
    spec:
      containers:
        - name: test
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
          command: ["sh", "-c", "sleep 99d"]
          resources:
            limits: { cpu: '4', memory: 16Gi }
            requests: { cpu: 10m, memory: 64Mi }
      terminationGracePeriodSeconds: 30
```

### 场景 B：高性能独占训练
确保工作负载独占多块物理 GPU，并采用硬隔离模式以获得最高性能。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-dedicated-training
spec:
  replicas: 1
  template:
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
      annotations:
        tensor-fusion.ai/gpu-count: '8'        # 使用 8 块 GPU
        tensor-fusion.ai/dedicated-gpu: 'true' # 独占模式
        tensor-fusion.ai/isolation: 'hard'     # 硬件级隔离
        tensor-fusion.ai/qos: 'critical'       # 最高调度优先级
    spec:
      containers:
        - name: test
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
          command: ["sh", "-c", "sleep 99d"]
      terminationGracePeriodSeconds: 30
```

### 场景 C：带自动扩缩容的成本优化型推理
根据实际负载垂直扩缩 GPU 资源，并在高峰期提供驱逐保护。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-autoscale-inference
spec:
  template:
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/vram-request: '2Gi'
        # 启用垂直自动扩缩容
        tensor-fusion.ai/autoscale: 'true'
        tensor-fusion.ai/autoscale-target: 'all'
        # 在高峰期间提供 10 分钟的驱逐保护
        tensor-fusion.ai/eviction-protection: '10m'
    spec:
      containers:
        - name: test
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
          command: ["sh", "-c", "sleep 99d"]
      terminationGracePeriodSeconds: 30
```

### 场景 D：多容器 GPU 流水线
在同一个 Pod 中为不同的业务容器分配特定的 GPU 资源。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-multi-container
spec:
  template:
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
      annotations:
        # 指定需要注入的容器
        tensor-fusion.ai/inject-container: 'pre-process,inference'
        # 通过 JSON 定义每个容器的 GPU 分配
        tensor-fusion.ai/container-gpu-count: '{"pre-process": 1, "inference": 1}'
        tensor-fusion.ai/gpu-model: 'A100'
    spec:
      containers:
        - name: pre-process
          image: my-process-image
        - name: inference
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
      terminationGracePeriodSeconds: 30
```

### 场景 E：远程 vGPU 执行模式（Serverless 化）
将 GPU 计算卸载到远程 Worker 节点，保持客户端 Pod 轻量化。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-remote-vgpu
spec:
  template:
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        # 启用 Sidecar Worker 远程执行
        tensor-fusion.ai/sidecar-worker: 'true'
        tensor-fusion.ai/workload-mode: 'dynamic'
        # 强制使用远程 GPU 资源
        tensor-fusion.ai/is-local-gpu: 'false'
    spec:
      containers:
        - name: test
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
          command: ["sh", "-c", "sleep 99d"]
      terminationGracePeriodSeconds: 30
```
