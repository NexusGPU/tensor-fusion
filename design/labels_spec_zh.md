# TensorFusion Kubernetes 标签 (Labels) 规范与最佳实践

本文档梳理了 TensorFusion 系统中 Kubernetes 标签（Labels）的使用脉络。不同于用于精细化资源请求的注解（Annotations），**标签（Labels）** 主要用于资源识别、分组、生命周期管理和自动发现。

---

## 1. 核心基础设施标签

| 标签键 (Key) | 用途 | 涉及组件 | 取值示例 |
| :--- | :--- | :--- | :--- |
| `tensor-fusion.ai/enabled` | 功能总开关。 | Webhook, 调度器 | `'true'`, `'false'` |
| `tensor-fusion.ai/component` | 标识资源的架构角色。 | 所有组件 | `client`, `worker`, `hypervisor`, `operator` |
| `tensor-fusion.ai/managed-by` | 追踪资源归属和控制器责任。 | 控制器 | `operator`, `gpupool`, `workload-controller` |
| `tensor-fusion.ai/gpupool` | 将资源（Pod, 节点, GPU）逻辑分组。 | 调度器, Operator | 如 `shared-pool-a`, `dedicated-h100` |
| `tensor-fusion.ai/cluster` | 多集群环境下的集群标识。 | Operator | 如 `prod-us-east-1` |

---

## 2. 资源发现与连接标签

用于简化组件间通信，特别是在远程 vGPU 场景下。

| 标签键 (Key) | 使用场景 | 描述 |
| :--- | :--- | :--- |
| `tensor-fusion.ai/worker-name` | 远程 vGPU / 连接 | 当 Worker Pod 状态变化时，匹配相关的 `TensorFusionConnection`。 |
| `tensor-fusion.ai/host-port` | 端口分配 | 设置为 `auto` 时，系统会自动为 Worker 分配宿主机端口。 |
| `tensor-fusion.ai/port-name` | 服务发现 | 为分配的端口命名，方便在连接字符串中引用。 |
| `tensor-fusion.ai/node-provisioner` | 云原生集成 | 将 `GPUNode` 对象与底层 K8s 节点匹配（常用于 cloud-init）。 |

---

## 3. 稳定性与调度控制

| 标签键 (Key) | 使用场景 | 描述 |
| :--- | :--- | :--- |
| `tensor-fusion.ai/do-not-disrupt` | 维护 / 驱逐保护 | 类似 Karpenter，防止 Pod 或节点在资源重排（Rebalance）时被迁移或销毁。 |
| `tensor-fusion.ai/pod-template-hash` | 版本管理 | 用于控制器识别不同版本的 Worker Pod 模板。 |
| `tensor-fusion.ai/node-selector-hash` | 调度优化 | 内部哈希值，用于优化调度器对节点的匹配逻辑。 |

---

## 5. 常见使用场景与 YAML 示例

通过合理配置标签，可以实现集群的高效运维和任务的稳定运行。

### A. 场景：按团队/业务线一键开启功能
无需为每个 Pod 单独配置，在命名空间级别管理功能开关。

```yaml
# 1. 为整个命名空间打标签
apiVersion: v1
kind: Namespace
metadata:
  name: ml-team-alpha
  labels:
    # 启用该命名空间下所有 Pod 的 TensorFusion 功能
    tensor-fusion.ai/enabled: 'true'
---
# 2. 该命名空间下的 Deployment 无需额外标签即可生效
apiVersion: apps/v1
kind: Deployment
metadata:
  name: training-job
  namespace: ml-team-alpha
spec:
  template:
    metadata:
      annotations:
        tensor-fusion.ai/gpu-count: '1'
    spec:
      containers:
        - name: train
          image: my-training-image
```

### B. 场景：长时训练任务的驱逐保护
防止集群在进行资源自动均衡或节点维护时中断正在运行的重要任务。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: long-running-llm-training
spec:
  template:
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
        # 核心标签：防止调度器重排或节点缩容导致任务中断
        tensor-fusion.ai/do-not-disrupt: 'true'
      annotations:
        tensor-fusion.ai/gpu-count: '8'
        tensor-fusion.ai/qos: 'critical'
    spec:
      containers:
        - name: trainer
          image: pytorch/pytorch:latest
```

### C. 场景：基于逻辑资源池的异构调度
将 A100 节点和 T4 节点划分为不同的逻辑池，用户通过标签选择目标硬件。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: high-perf-inference
spec:
  template:
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
        # 核心标签：只在该高性能 GPU 池中进行调度
        tensor-fusion.ai/gpupool: a100-80gb-pool
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/vram-request: '40Gi'
    spec:
      containers:
        - name: server
          image: vllm/vllm-openai:latest
```

### D. 场景：自动化运维中的组件识别
手动部署监控或调试工具时，通过标识组件类型方便系统统一管理和监控。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-debug-tool
  labels:
    # 核心标签：将该 Pod 标记为超算器组件
    tensor-fusion.ai/component: hypervisor
    # 核心标签：标记由管理员手动维护，防止被误删
    tensor-fusion.ai/managed-by: operator
spec:
  containers:
    - name: debugger
      image: nvidia/samples:vectoradd
```
