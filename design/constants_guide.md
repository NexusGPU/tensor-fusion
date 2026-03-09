# TensorFusion Constants Guide: Labels & Annotations

This guide is based on definitions in `internal/constants/constants.go` and provides standard usage and best practices for TensorFusion metadata.

---

## 1. Labels Specification

Labels are primarily used for **resource discovery, logical grouping, lifecycle management, and controller identification**.

| Key | Description | Best Practice |
| :--- | :--- | :--- |
| `tensor-fusion.ai/enabled` | **Core Switch**. Determines if Webhook/Scheduler processes the Pod/Namespace. | Set to `'true'` at Namespace level to simplify Pod config. |
| `tensor-fusion.ai/component` | Identifies the architectural role (`client`, `worker`, `hypervisor`, `operator`). | Required for all system Pods to enable effective monitoring/log aggregation. |
| `tensor-fusion.ai/gpupool` | Associates resources (Nodes, GPUs, Pods) with a logical resource pool. | Group by GPU model or department to abstract physical hardware details. |
| `tensor-fusion.ai/do-not-disrupt` | **Stability Control**. Prevents Pod/Node from being evicted during rebalancing. | Always set to `'true'` for long-running training jobs (>4h) in production. |

---

## 2. Annotations Specification

Annotations are used for **fine-grained resource definition, behavior control, and system state tracking**.

### A. Resource Definition (User Specified)
| Key | Description | Best Practice |
| :--- | :--- | :--- |
| `tensor-fusion.ai/gpu-count` | Total physical GPUs required. | Mandatory. Replaces standard integer resource requests. |
| `tensor-fusion.ai/tflops-request` | Requested compute performance in TFLOPS. | Combine with VRAM for high-density GPU sharing. |
| `tensor-fusion.ai/vram-request` | Requested GPU Video RAM (e.g., `10Gi`). | Always set Limits to prevent memory overruns in shared mode. |

### B. Behavior Control (User Specified)
| Key | Description | Best Practice |
| :--- | :--- | :--- |
| `tensor-fusion.ai/sidecar-worker` | Whether to inject Worker as a Sidecar. | Default and recommended mode for Remote vGPU. |
| `tensor-fusion.ai/autoscale` | Enables Vertical GPU Resource Autoscaling. | Ideal for inference services to boost utilization. |
| `tensor-fusion.ai/eviction-protection` | Duration to protect Pod from preemption. | Use during critical job phases or on volatile Spot instances. |

---

## 3. Usage Examples (YAML)

### Basic Usage: Fractional GPU Sharing
Multiple Pods share a single GPU with fixed resource limits.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-basic-sharing
spec:
  replicas: 1
  template:
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
        tensor-fusion.ai/gpupool: tensor-fusion-shared
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/tflops-request: 17800m 
        tensor-fusion.ai/vram-request: 128Mi
        tensor-fusion.ai/is-local-gpu: 'true'
    spec:
      containers:
        - name: test
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
          command: ["sh", "-c", "sleep 99d"]
```

### Advanced Usage: Remote vGPU with Autoscaling & Protection
High-availability inference setup using remote GPU workers with dynamic scaling and eviction safety.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-advanced-inference
spec:
  replicas: 1
  template:
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
        tensor-fusion.ai/gpupool: high-perf-pool
        # Protection: Prevent disruption during cluster maintenance
        tensor-fusion.ai/do-not-disrupt: 'true'
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        # Remote Mode: Offload computation to specialized GPU nodes
        tensor-fusion.ai/sidecar-worker: 'true'
        tensor-fusion.ai/workload-mode: 'dynamic'
        tensor-fusion.ai/is-local-gpu: 'false'
        # Vertical Autoscaling: Scale TFLOPS/VRAM based on load
        tensor-fusion.ai/autoscale: 'true'
        tensor-fusion.ai/autoscale-target: 'all'
        # Protection: 10 minutes of safety if preemption is triggered
        tensor-fusion.ai/eviction-protection: '10m'
        # Templates: Reference pre-defined configs
        tensor-fusion.ai/workload-profile: 'standard-inference'
    spec:
      containers:
        - name: test
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
          resources:
            limits: { cpu: '4', memory: 16Gi }
```

---

# TensorFusion Constants 指南：Labels 与 Annotations

本文档基于 `internal/constants/constants.go` 的定义，提供 TensorFusion 元数据的标准用法和最佳实践。

---

## 1. Labels (标签) 规范

标签主要用于**资源发现、逻辑分组、生命周期管理和控制器识别**。

| 键 (Key) | 描述 | 最佳实践 |
| :--- | :--- | :--- |
| `tensor-fusion.ai/enabled` | **核心总开关**。决定 Webhook/调度器是否处理该 Pod/命名空间。 | 在 Namespace 级别设置为 `'true'` 以简化 Pod 配置。 |
| `tensor-fusion.ai/component` | 标识架构角色（`client`, `worker`, `hypervisor`, `operator`）。 | 所有系统 Pod 必须携带，以便进行有效的监控和日志聚合。 |
| `tensor-fusion.ai/gpupool` | 将资源（节点、GPU、Pod）关联到逻辑资源池。 | 按 GPU 型号或部门分组，屏蔽物理硬件细节。 |
| `tensor-fusion.ai/do-not-disrupt` | **稳定性控制**。防止 Pod/节点在资源重平衡时被驱逐。 | 生产环境中的长时训练任务（>4h）务必设置为 `'true'`。 |

---

## 2. Annotations (注解) 规范

注解用于**精细化资源定义、行为控制和系统状态跟踪**。

### A. 资源定义 (用户指定)
| 键 (Key) | 描述 | 最佳实践 |
| :--- | :--- | :--- |
| `tensor-fusion.ai/gpu-count` | 所需物理 GPU 总数。 | 必填项。替代标准的整数资源申请。 |
| `tensor-fusion.ai/tflops-request` | 请求的算力 (TFLOPS)。 | 结合显存使用，实现高密度 GPU 共享。 |
| `tensor-fusion.ai/vram-request` | 请求的显存 (如 `10Gi`)。 | 务必设置 Limits 以防止共享模式下的内存溢出。 |

### B. 行为控制 (用户指定)
| 键 (Key) | 描述 | 最佳实践 |
| :--- | :--- | :--- |
| `tensor-fusion.ai/sidecar-worker` | 是否以 Sidecar 模式注入 Worker。 | Remote vGPU 的默认且推荐模式。 |
| `tensor-fusion.ai/autoscale` | 开启垂直 GPU 资源自动扩缩容。 | 推理服务的理想选择，可显著提升利用率。 |
| `tensor-fusion.ai/eviction-protection` | 保护 Pod 不被抢占的时长。 | 在任务关键阶段或易波动的 Spot 实例上使用。 |

---

## 3. 使用场景示例 (YAML)

### 基础用法：精细化 GPU 共享 (Fractional GPU)
多个 Pod 共享单块 GPU，具有固定的资源限制。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-basic-sharing
spec:
  replicas: 1
  template:
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
        tensor-fusion.ai/gpupool: tensor-fusion-shared
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/tflops-request: 17800m 
        tensor-fusion.ai/vram-request: 128Mi
        tensor-fusion.ai/is-local-gpu: 'true'
    spec:
      containers:
        - name: test
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
          command: ["sh", "-c", "sleep 99d"]
```

### 高级用法：带扩缩容与保护的远程 vGPU 模式
使用远程 GPU Worker 的高可用推理设置，支持动态扩缩容和驱逐保护。

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tf-advanced-inference
spec:
  replicas: 1
  template:
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
        tensor-fusion.ai/gpupool: high-perf-pool
        # 稳定性保护：防止集群重平衡或维护期间中断
        tensor-fusion.ai/do-not-disrupt: 'true'
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        # 远程模式：将计算卸载到专业的 GPU 节点
        tensor-fusion.ai/sidecar-worker: 'true'
        tensor-fusion.ai/workload-mode: 'dynamic'
        tensor-fusion.ai/is-local-gpu: 'false'
        # 垂直自动扩缩容：根据负载动态调整算力/显存
        tensor-fusion.ai/autoscale: 'true'
        tensor-fusion.ai/autoscale-target: 'all'
        # 驱逐保护：触发抢占时提供 10 分钟的安全缓冲
        tensor-fusion.ai/eviction-protection: '10m'
        # 配置模板：引用预定义配置
        tensor-fusion.ai/workload-profile: 'standard-inference'
    spec:
      containers:
        - name: test
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
          resources:
            limits: { cpu: '4', memory: 16Gi }
```
