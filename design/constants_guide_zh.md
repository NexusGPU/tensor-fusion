# TensorFusion Constants 深度解析：Labels 与 Annotations

本文档对 TensorFusion 的元数据驱动架构进行了深度的技术解析，涵盖了 `internal/constants/constants.go` 中的所有核心常量，并解释了其背后的工作机制。

---

## 1. 核心连接与生命周期 (Labels)
这些标签控制 Pod 如何被系统识别，以及节点如何被组织。

| 常量名 | 键 (Key) | 技术机制 |
| :--- | :--- | :--- |
| `TensorFusionEnabledLabelKey` | `tensor-fusion.ai/enabled` | **机制**：Webhook 监听此标签。若 Pod 或 Namespace 为 `true`，Webhook 将注入 Sidecar、环境变量和 CDI 设备。 |
| `GpuPoolKey` | `tensor-fusion.ai/gpupool` | **机制**：调度器用于过滤节点。节点打上池标签，Pod 通过此标签锁定目标资源池。 |
| `LabelComponent` | `tensor-fusion.ai/component` | **机制**：用于 RBAC 和日志过滤。取值：`hypervisor`（节点端）、`operator`（管理端）、`client`（用户应用）。 |
| `SchedulingDoNotDisruptLabel` | `tensor-fusion.ai/do-not-disrupt` | **机制**：重平衡器（Rebalancer）会跳过带有此标签的 Pod。对不可重启的任务至关重要。 |
| `ProvisionerLabelKey` | `tensor-fusion.ai/node-provisioner` | **机制**：节点注册时使用，将物理 K8s 节点与 TensorFusion 的 `GPUNode` CRD 关联。 |

---

## 2. 精细化资源规格 (Annotations)
这些注解定义了 GPU 的“切片”规则。

| 常量名 | 键 (Key) | 详细细节与约束 |
| :--- | :--- | :--- |
| `TFLOPSRequestAnnotation` | `tensor-fusion.ai/tflops-request` | **单位**：毫核算力 (m)，如 `5000m` = 5 TFLOPS。控制 GPU 的时间片切分。 |
| `VRAMRequestAnnotation` | `tensor-fusion.ai/vram-request` | **单位**：Ki/Mi/Gi。由 Hypervisor 通过内存管理钩子强制执行，防止 OOM 影响他人。 |
| `ComputeRequestAnnotation` | `tensor-fusion.ai/compute-percent-request` | **警告**：**不推荐使用**。会绕过命名空间配额检查，仅建议用于不计费的简单共享。 |
| `ContainerGPUCountAnnotation` | `tensor-fusion.ai/container-gpu-count` | **格式**：JSON 字典。如 `{"c1": 1}`。为特定容器覆盖 Pod 级别的 GPU 请求数量。 |
| `ContainerGPUsAnnotation` | `tensor-fusion.ai/container-gpus` | **机制**：**系统自动设置**。映射容器名称到具体的物理 GPU UUID。 |

---

## 3. 部署模式与网络控制
控制 vGPU Worker *运行在哪里* 以及 *如何连接*。

| 常量名 | 键 (Key) | 描述 |
| :--- | :--- | :--- |
| `SidecarWorkerAnnotation` | `tensor-fusion.ai/sidecar-worker` | **默认模式**。Worker 与业务运行在同一 Pod。通过 UDS/SHM 实现极低延迟。 |
| `EmbeddedWorkerAnnotation` | `tensor-fusion.ai/embedded-worker` | Worker 逻辑直接链接到业务进程中，无 Sidecar。开销最低。 |
| `DedicatedWorkerAnnotation` | `tensor-fusion.ai/dedicated-worker` | Worker 运行在独立 Pod。适用于生命周期解耦或一个 Worker 供多个 Client 使用。 |
| `IsLocalGPUAnnotation` | `tensor-fusion.ai/is-local-gpu` | 若为 `true`，调度器优先选择同节点 GPU；若为 `false`，开启跨网络“远程 vGPU”模式。 |
| `GenHostPortLabel` | `tensor-fusion.ai/host-port` | 若为 `auto`，系统为 Worker 自动分配一个唯一的 `NodePort` 用于接收远程流量。 |

---

## 4. 高级调度策略与 QoS
生产环境下针对不同优先级任务的策略注解。

| 常量名 | 键 (Key) | 取值与行为 |
| :--- | :--- | :--- |
| `QoSLevelAnnotation` | `tensor-fusion.ai/qos` | `critical`, `high`, `medium`, `low`。资源紧张时，高 QoS 可抢占低 QoS Pod。 |
| `IsolationModeAnnotation` | `tensor-fusion.ai/isolation` | `hard` (MIG/MPS 硬限制), `shared` (时间片), `partitioned` (硬件分区)。 |
| `DedicatedGPUAnnotation` | `tensor-fusion.ai/dedicated-gpu` | 若为 `true`，调度器不会在所分配的物理 GPU 上放置任何其他 Pod。 |
| `EvictionProtectionAnnotation` | `tensor-fusion.ai/eviction-protection` | **格式**：时长（如 `30m`）。即使有高优先级 Pod 进来，在保护期内也不会被驱逐。 |

---

## 5. 使用场景与组合指南 (Scenarios & Combinations)

本章节将 Labels 和 Annotations 分为**用户主动配置 (User-Provided)** 和 **系统自动注入 (System-Injected)** 两类，并提供基础与高级场景的组合示例。

### A. 基础场景 (Fundamentals)

#### 场景 A1：精细化 GPU 共享 (Fractional GPU Sharing)
**目标**：允许多个轻量推理任务共享单块物理 GPU。
*   **用户配置 (User-Provided)**：
    ```yaml
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
        tensor-fusion.ai/gpupool: tensor-fusion-shared
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/vram-request: 512Mi
        tensor-fusion.ai/tflops-request: 5000m  # 5 TFLOPS 算力
    ```
*   **系统注入 (System-Injected)**：`tensor-fusion.ai/index` (CDI 索引), `tensor-fusion.ai/gpu-ids` (物理 ID)。

#### 场景 A2：高性能独占 GPU (Dedicated GPU)
**目标**：确保重载任务获得物理卡的完全控制权。
*   **用户配置 (User-Provided)**：
    ```yaml
    metadata:
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/dedicated-gpu: 'true'
        tensor-fusion.ai/isolation: hard
    ```

---

### B. 高级场景 (Complex Orchestration)

#### 场景 B1：远程 vGPU 执行 (Serverless 算力模式)
**目标**：将 GPU 计算卸载到远程节点，客户端保持轻量。
*   **用户配置 (User-Provided)**：
    ```yaml
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
        tensor-fusion.ai/host-port: auto # 自动分配网络端口
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/is-local-gpu: 'false' # 强制远程执行
        tensor-fusion.ai/sidecar-worker: 'true'
    ```
*   **系统注入 (System-Injected)**：`tensor-fusion.ai/port-number` (分配的宿主机端口)。

#### 场景 B2：高可靠训练任务 (Anti-Preemption)
**目标**：防止长时训练任务被重平衡器（Rebalancer）中断。
*   **用户配置 (User-Provided)**：
    ```yaml
    metadata:
      labels:
        tensor-fusion.ai/do-not-disrupt: 'true' # 禁止重平衡
      annotations:
        tensor-fusion.ai/gpu-count: '8'
        tensor-fusion.ai/qos: critical        # 最高调度优先级
        tensor-fusion.ai/eviction-protection: 4h # 4小时硬性保护
    ```

#### 场景 B3：自动扩缩容推理 (Autoscaling)
**目标**：随着流量增长，自动增加 GPU 算力上限。
*   **用户配置 (User-Provided)**：
    ```yaml
    metadata:
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/autoscale: 'true'
        tensor-fusion.ai/autoscale-target: all
    ```
*   **系统注入 (System-Injected)**：`tensor-fusion.ai/vram-request` 和 `tensor-fusion.ai/tflops-request` 将由 **Autoscaler 根据推荐值动态注入**，用户无需手动维护。

---

## 6. 内部状态注解 (Read-Only)
这些注解由系统设置，用于 `kubectl get pod -o yaml` 调试。

- `tensor-fusion.ai/gpu-ids`：分配的 GPU 物理 UUID 列表。
- `tensor-fusion.ai/index`：为 CDI 挂载分配的唯一索引。
- `tensor-fusion.ai/port-number`：为 Worker Sidecar 分配的网络端口。
- `tensor-fusion.ai/gpu-released`：当调度器确认资源已清理时设为 `true`。
