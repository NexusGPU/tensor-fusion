# Tensor Fusion DRA 使用指南

## 文档概述

本文档面向 **Tensor Fusion 用户**和 **DRA Driver 开发者**，介绍如何在 Tensor Fusion 中启用和使用 Kubernetes Dynamic Resource Allocation (DRA) 功能。

**目标读者**:
- **集群管理员**: 需要启用和配置 DRA
- **应用开发者**: 需要编写支持 DRA 的 Pod 定义
- **DRA Driver 开发者**: 需要对接 Tensor Fusion DRA

**文档版本**: 1.0
**最后更新**: 2025-10-14
**Kubernetes 版本要求**: 1.34+ (DRA v1beta2)

---

## 目录

1. [快速开始](#1-快速开始)
2. [启用 DRA 的前提条件](#2-启用-dra-的前提条件)
3. [DRA 配置步骤](#3-dra-配置步骤)
4. [Pod 注解字段说明](#4-pod-注解字段说明)
5. [CEL 表达式编写指南](#5-cel-表达式编写指南)
6. [DRA Driver 对接指南](#6-dra-driver-对接指南)
7. [常见问题 FAQ](#7-常见问题-faq)

---

## 1. 快速开始

### 1.1 最简单的 DRA Pod 示例

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-gpu-workload
  annotations:
    # 启用 DRA
    tensor-fusion.ai/dra-enabled: "true"

    # GPU 资源需求
    tensor-fusion.ai/tflops-request: "100"
    tensor-fusion.ai/vram-request: "16Gi"
    tensor-fusion.ai/gpu-count: "1"

    # GPU 池选择
    tensor-fusion.ai/gpupool: "default"
spec:
  containers:
    - name: workload
      image: nvidia/cuda:12.0-base
```

**工作流程**:
1. Tensor Fusion Webhook 检测到 `dra-enabled: "true"`
2. 自动生成 CEL 表达式并注入 ResourceClaim 引用
3. Kubernetes 调度器根据 DRA 资源分配 GPU
4. Pod 启动并获得分配的 GPU

---

## 2. 启用 DRA 的前提条件

### 2.1 Kubernetes 集群要求

| 项目 | 要求 | 说明 |
|------|------|------|
| **Kubernetes 版本** | ≥ 1.34 | DRA v1beta2 API |
| **Feature Gate** | `DynamicResourceAllocation=true` | 需在 kube-apiserver、kube-scheduler、kubelet 启用 |
| **调度器** | 原生 kube-scheduler | 支持 DRA 资源调度 |

#### 2.1.1 启用 Feature Gate

在 Kubernetes 集群启动参数中添加：

```bash
# kube-apiserver
--feature-gates=DynamicResourceAllocation=true

# kube-scheduler
--feature-gates=DynamicResourceAllocation=true

# kubelet
--feature-gates=DynamicResourceAllocation=true
```

## 2. DRA 配置步骤

### 2.1 步骤 1: 创建 SchedulingConfigTemplate（全局配置）

创建文件 `scheduling-config.yaml`:

```yaml
apiVersion: tensor-fusion.ai/v1
kind: SchedulingConfigTemplate
metadata:
  name: default-scheduling-config
spec:
  # 启用 DRA
  dra:
    enable: true
    # 可选：指定 ResourceClaimTemplate 名称
    resourceClaimTemplateName: "tensor-fusion-gpu-template"
  # 其他配置
  placement:
    mode: CompactFirst
```

应用配置：
```bash
kubectl apply -f scheduling-config.yaml
```

**说明**:
- `dra.enable: true` 为该配置模板启用 DRA
- 所有使用此模板的 GPUPool 下的 Pod 将默认使用 DRA
- `resourceClaimTemplateName` 默认为 `"tensor-fusion-gpu-template"`，通常无需修改

### 3.2 步骤 2: 创建 ResourceClaimTemplate

创建文件 `resourceclaim-template.yaml`:

```yaml
apiVersion: resource.k8s.io/v1beta2
kind: ResourceClaimTemplate
metadata:
  name: tensor-fusion-gpu-template
  labels:
    # 必须设置此 label
    tensor-fusion.ai/resource-claim-template: "true"
spec:
  spec:
    devices:
      requests:
        - name: gpu-request
          allocationMode: ExactCount
          exactly:
            count: 1
            selector:
              cel:
                # 默认值，由 Controller 自动更新
                expression: "true"
```

应用配置：
```bash
kubectl apply -f resourceclaim-template.yaml
```

**重要**:
- 必须设置 label `tensor-fusion.ai/resource-claim-template: "true"`
- `count` 和 `cel.expression` 会由 ResourceClaim Controller 自动更新，无需手动修改

### 3.3 步骤 3: 将 GPUPool 关联到 SchedulingConfigTemplate

```yaml
apiVersion: tensor-fusion.ai/v1
kind: GPUPool
metadata:
  name: my-gpu-pool
spec:
  # 关联 SchedulingConfigTemplate
  schedulingConfigTemplate: "default-scheduling-config"

  # 其他配置...
```

### 3.4 步骤 4: 提交 DRA Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-workload
  annotations:
    # 启用 DRA (说明：该注解和全局配置任选其一，非必须)
    tensor-fusion.ai/dra-enabled: "true"

    # 必须: 资源需求
    tensor-fusion.ai/tflops-request: "100"
    tensor-fusion.ai/vram-request: "16Gi"

    # 可选: GPU 数量（默认 1）
    tensor-fusion.ai/gpu-count: "2"

    # 必须: GPU 池
    tensor-fusion.ai/gpupool: "my-gpu-pool"

    # 可选: GPU 型号
    tensor-fusion.ai/gpu-model: "A100"

    # 可选: QoS 等级
    tensor-fusion.ai/qos: "high"
spec:
  schedulerName: tensor-fusion-scheduler  # 或使用 default-scheduler
  containers:
    - name: app
      image: my-app:latest
```

### 3.5 验证 DRA 是否生效

```bash
# 1. 检查 Pod 状态
kubectl get pod my-workload

# 2. 检查 ResourceClaim 是否创建
kubectl get resourceclaim

# 3. 检查 ResourceClaim 是否已分配
kubectl get resourceclaim <claim-name> -o yaml | grep allocated

# 4. 检查 ResourceSlice
kubectl get resourceslice

# 5. 查看 Pod events
kubectl describe pod my-workload
```

**成功标志**:
- Pod 状态变为 Running
- ResourceClaim 的 `allocated: true`
- Pod events 中看到 "Successfully assigned resources"

---

## 4. Pod 注解字段说明

### 4.1 必填字段

#### 4.1.1 `tensor-fusion.ai/dra-enabled`

**类型**: String
**可选值**: `"true"`, `"false"`
**默认值**: 根据 SchedulingConfigTemplate 配置

**说明**: 控制是否为此 Pod 启用 DRA 模式。

**示例**:
```yaml
annotations:
  # 显式启用 DRA
  tensor-fusion.ai/dra-enabled: "true"

  # 或显式禁用（使用传统 GPU Allocator）
  tensor-fusion.ai/dra-enabled: "false"
```

**优先级**:
```
Pod annotation > SchedulingConfigTemplate.spec.dra.enable > 默认禁用
```

#### 4.1.2 `tensor-fusion.ai/tflops-request`

**类型**: String (Quantity)
**单位**: TFlops (万亿次浮点运算/秒)
**示例值**: `"100"`, `"312"`, `"500"`

**说明**: 请求的 GPU 算力，用于：
1. 生成 CEL 表达式（过滤容量不足的 GPU）
2. 填充 ResourceClaim 的 Capacity.Requests

**示例**:
```yaml
annotations:
  tensor-fusion.ai/tflops-request: "200"  # 请求 200 TFlops
```

**注意**:
- 必须是有效的 Kubernetes Quantity 格式
- 不要包含单位（默认为 TFlops）

#### 4.1.3 `tensor-fusion.ai/vram-request`

**类型**: String (Quantity)
**单位**: Bytes (可使用 Ki, Mi, Gi 等后缀)
**示例值**: `"16Gi"`, `"32Gi"`, `"80Gi"`

**说明**: 请求的 GPU 显存大小。

**示例**:
```yaml
annotations:
  tensor-fusion.ai/vram-request: "32Gi"  # 请求 32GB 显存
```

#### 4.1.4 `tensor-fusion.ai/gpupool`

**类型**: String
**说明**: 指定从哪个 GPU 池分配资源。

**示例**:
```yaml
annotations:
  tensor-fusion.ai/gpupool: "production-pool"
```

**注意**:
- 必须是已存在的 GPUPool 名称
- Pool 必须关联支持 DRA 的 SchedulingConfigTemplate

### 4.2 可选字段

#### 4.2.1 `tensor-fusion.ai/gpu-count`

**类型**: String (Integer)
**默认值**: `"1"`
**示例值**: `"1"`, `"2"`, `"4"`, `"8"`

**说明**: 请求的 GPU 数量。

**示例**:
```yaml
annotations:
  tensor-fusion.ai/gpu-count: "4"  # 请求 4 块 GPU
```

**注意**:
- DRA 模式支持分配多块 GPU
- 每块 GPU 都会满足 tflops-request 和 vram-request 的要求

#### 4.2.2 `tensor-fusion.ai/gpu-model`

**类型**: String
**示例值**: `"A100"`, `"H100"`, `"V100"`, `"RTX4090"`

**说明**: 指定 GPU 型号要求。

**示例**:
```yaml
annotations:
  tensor-fusion.ai/gpu-model: "A100"  # 只选择 A100 GPU
```

**注意**:
- 必须与 GPU 实际型号完全匹配（区分大小写）
- 如不指定，则接受任何型号的 GPU

#### 4.2.3 `tensor-fusion.ai/qos`

**类型**: String
**可选值**: `"low"`, `"medium"`, `"high"`, `"critical"`
**默认值**: `"medium"`

**说明**: 指定服务质量等级，用于选择相应 QoS 级别的 GPU。

**示例**:
```yaml
annotations:
  tensor-fusion.ai/qos: "high"  # 选择 high QoS 的 GPU
```

**QoS 级别说明**:
- `critical`: 最高优先级，通常用于生产关键业务
- `high`: 高优先级，用于重要业务
- `medium`: 中等优先级（默认）
- `low`: 低优先级，用于测试或开发环境

#### 4.2.4 `tensor-fusion.ai/dra-cel-expression`

**类型**: String (CEL 表达式)
**说明**: 高级用户可提供自定义 CEL 表达式，覆盖自动生成的表达式。

**示例**:
```yaml
annotations:
  tensor-fusion.ai/dra-cel-expression: |
    device.attributes["model"] == "H100" &&
    device.capacity["vram"].AsApproximateFloat64() >= 100 &&
    device.attributes["qos"] == "critical"
```

**注意**:
- 只有熟悉 CEL 语法的高级用户才应使用此字段
- 错误的 CEL 表达式会导致 Pod 无法调度
- 详见[第5章](#5-cel-表达式编写指南)

### 4.3 不支持的字段

以下字段在 DRA 模式下**不生效**或**不需要**：

#### 4.3.1 `tensor-fusion.ai/tflops-limit` 和 `tensor-fusion.ai/vram-limit`

**原因**: DRA 只处理 Requests（调度时的最小保证），Limits 由运行时组件（Hypervisor/Device Plugin）处理。

**如何使用 Limits**:
仍然在 Pod annotations 中指定，但这些值不会影响 DRA 调度，只在 Pod 运行时由 Tensor Fusion 运行时组件读取并执行限制。

```yaml
annotations:
  # DRA 调度使用 request
  tensor-fusion.ai/tflops-request: "100"
  tensor-fusion.ai/vram-request: "16Gi"

  # 运行时限制（不影响 DRA 调度）
  tensor-fusion.ai/tflops-limit: "200"
  tensor-fusion.ai/vram-limit: "32Gi"
```

#### 4.3.2 配额 (Quota) 相关字段

**说明**: DRA 的 ResourceClaim 不包含配额信息。

---

## 5. CEL 表达式编写指南 

### 5.1 什么是 CEL？

**CEL (Common Expression Language)** 是一种声明式表达式语言，用于在 Kubernetes DRA 中描述设备选择条件。

### 5.2 支持的设备属性 (device.attributes)

| 属性名 | 类型 | 说明 | 示例值 |
|--------|------|------|--------|
| `model` | String | GPU 型号 | `"A100"`, `"H100"` |
| `pool_name` | String | GPU 所属池 | `"production-pool"` |
| `pod_namespace` | String | GPU 命名空间 | `"default"` |
| `uuid` | String | GPU 唯一标识 | `"GPU-xxx"` |
| `phase` | String | GPU 状态 | `"Running"`, `"Pending"` |
| `used_by` | String | 当前使用者 | `""` (空=未使用) |
| `node_name` | String | 所在节点 | `"node-gpu-01"` |
| `qos` | String | QoS 等级 | `"low"`, `"medium"`, `"high"`, `"critical"` |
| `node_total_tflops` | String | 节点总算力 | `"2496"` |
| `node_total_vram` | String | 节点总显存 | `"320Gi"` |
| `node_total_gpus` | Int | 节点 GPU 总数 | `8` |
| `node_managed_gpus` | Int | 管理的 GPU 数 | `8` |
| `node_virtual_tflops` | String | 节点虚拟算力 | `"3744"` (超分) |
| `node_virtual_vram` | String | 节点虚拟显存 | `"480Gi"` (超分) |

### 5.3 支持的设备容量 (device.capacity)

| 容量名 | 类型 | 说明 | 示例值 |
|--------|------|------|--------|
| `tflops` | Quantity | 物理 GPU 算力 | `"312"` |
| `vram` | Quantity | 物理 GPU 显存 | `"40Gi"` |
| `virtual_tflops` | Quantity | 每 GPU 虚拟算力 | `"468"` (超分 1.5x) |
| `virtual_vram` | Quantity | 每 GPU 虚拟显存 | `"60Gi"` (超分 1.5x) |

**注意**: `virtual_*` 容量只在 GPUPool 配置了超分时才可用。

### 5.4 CEL 语法基础

#### 5.4.1 访问属性

```cel
// 访问字符串属性
device.attributes["model"]
device.attributes["pool_name"]

// 访问整数属性
device.attributes["node_total_gpus"]
```

#### 5.4.2 访问容量

```cel
// 访问 Quantity（需要转换）
device.capacity["tflops"].AsApproximateFloat64()
device.capacity["vram"].AsApproximateFloat64()
```

**重要**: Quantity 类型必须使用 `.AsApproximateFloat64()` 转换为数值才能比较。

#### 5.4.3 逻辑运算符

```cel
// 逻辑与
condition1 && condition2

// 逻辑或
condition1 || condition2

// 比较运算符
== != < > <= >=
```

### 5.5 CEL 表达式示例

#### 示例 1: 基础选择

```cel
device.attributes["model"] == "A100" &&
device.attributes["pool_name"] == "production" &&
device.attributes["phase"] == "Running"
```

**说明**: 选择 production 池中状态为 Running 的 A100 GPU。

#### 示例 2: 容量要求

```cel
device.capacity["tflops"].AsApproximateFloat64() >= 200 &&
device.capacity["vram"].AsApproximateFloat64() >= 32000000000
```

**说明**:
- TFlops ≥ 200
- VRAM ≥ 32GB (32000000000 bytes)

**注意**: VRAM 单位是 bytes，32Gi = 34359738368 bytes。

#### 示例 3: 多型号选择

```cel
(device.attributes["model"] == "H100" || device.attributes["model"] == "A100") &&
device.capacity["vram"].AsApproximateFloat64() >= 40000000000 &&
device.attributes["qos"] == "high"
```

**说明**: 选择 H100 或 A100，显存 ≥ 40GB，QoS 为 high。

#### 示例 4: 节点级条件

```cel
device.attributes["node_total_gpus"] >= 8 &&
device.attributes["node_total_tflops"].AsApproximateFloat64() > 2000 &&
device.attributes["phase"] == "Running"
```

**说明**: 选择至少有 8 块 GPU 且总算力超过 2000 TFlops 的节点上的 GPU。

#### 示例 5: 排除条件

```cel
device.attributes["model"] != "V100" &&
device.attributes["qos"] != "low" &&
device.attributes["phase"] == "Running"
```

**说明**: 排除 V100 GPU 和 low QoS GPU。

### 5.6 自动生成的 CEL 表达式

当你不提供自定义 CEL 表达式时，系统会根据以下注解自动生成：

```yaml
annotations:
  tensor-fusion.ai/gpu-model: "A100"
  tensor-fusion.ai/gpupool: "prod-pool"
  tensor-fusion.ai/tflops-request: "200"
  tensor-fusion.ai/vram-request: "32Gi"
  tensor-fusion.ai/qos: "high"
```

**生成的 CEL**:
```cel
device.attributes["model"] == "A100" &&
device.attributes["pool_name"] == "prod-pool" &&
device.capacity["tflops"].AsApproximateFloat64() >= 200 &&
device.capacity["vram"].AsApproximateFloat64() >= 34359738368 &&
device.attributes["qos"] == "high" &&
device.attributes["phase"] == "Running" &&
device.attributes["used_by"] == ""
```

**说明**:
- 系统自动添加 `phase == "Running"` 和 `used_by == ""`
- VRAM 自动转换为 bytes

### 5.7 CEL 编写建议

#### ✅ 推荐做法

1. **始终检查 GPU 状态**:
```cel
device.attributes["phase"] == "Running"
```

2. **使用括号提高可读性**:
```cel
(condition1 || condition2) && condition3
```

#### ❌ 避免的做法

1. **不要使用不存在的属性**:
```cel
// ❌ 错误：quota 不是设备属性
device.attributes["quota"]
```

2. **不要忘记 Quantity 转换**:
```cel
// ❌ 错误：Quantity 不能直接比较
device.capacity["tflops"] >= 100

// ✅ 正确
device.capacity["tflops"].AsApproximateFloat64() >= 100
```

3. **不要使用过于复杂的表达式**:
```cel
// ❌ 错误：过于复杂，难以维护
(device.attributes["model"] == "A100" && device.capacity["tflops"].AsApproximateFloat64() >= 200) ||
(device.attributes["model"] == "H100" && device.capacity["tflops"].AsApproximateFloat64() >= 400) ||
(device.attributes["model"] == "V100" && device.capacity["tflops"].AsApproximateFloat64() >= 100 && device.attributes["qos"] == "high")
```

### 5.8 不支持的功能

#### 5.8.1 配额 (Quota)

**不支持原因**: 配额是命名空间级资源管理，不属于单个设备的属性。

#### 5.8.2 Limits

**不支持原因**: DRA 只处理 Requests（调度决策），Limits 是运行时限制。

**正确做法**: 在 Pod annotations 中指定 limits，由hypervisor限制。

## 6. DRA Driver 对接指南

### 6.1 Tensor Fusion DRA Driver 信息

#### 6.1.1 Driver 名称

```
tensor-fusion.ai.dra-driver
```

**用途**:
- 在 ResourceSlice.Spec.Driver 中标识设备提供者
- 在 ResourceClaim.Status.Allocation 中标识分配者
- Kubernetes 调度器通过此名称匹配 ResourceSlice 和 ResourceClaim

### 6.2 ResourceSlice 规范

#### 6.2.1 ResourceSlice 结构示例

```yaml
apiVersion: resource.k8s.io/v1beta2
kind: ResourceSlice
metadata:
  name: tensor-fusion-resource-slice-node-01
  labels:
    tensor-fusion.ai/managed-by: node-01
    kubernetes.io/hostname: node-01
spec:
  # Driver 名称
  driver: tensor-fusion.ai.dra-driver

  # 节点名称
  nodeName: node-01

  # 资源池
  pool:
    name: production-pool
    generation: 123
    resourceSliceCount: 1

  # 设备列表
  devices:
    - name: gpu-a100-01
      # 设备属性
      attributes:
        model:
          stringValue: "A100"
        pool_name:
          stringValue: "production-pool"
        phase:
          stringValue: "Running"
        used_by:
          stringValue: ""
        qos:
          stringValue: "high"
        # ... 更多属性

      # 设备容量
      capacity:
        tflops:
          value: "312"
        vram:
          value: "40Gi"
        virtual_tflops:
          value: "468"
        virtual_vram:
          value: "60Gi"

      # 允许多分配（vGPU 支持）
      allowMultipleAllocations: true
```

#### 6.2.2 关键字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `spec.driver` | ✅ | 必须为 `tensor-fusion.ai.dra-driver` |
| `spec.nodeName` | ✅ | GPU 所在 Kubernetes 节点名称 |
| `spec.pool.name` | ✅ | GPU 所属 Pool 名称 |
| `spec.devices[].name` | ✅ | GPU 唯一名称 |
| `spec.devices[].attributes` | ✅ | 设备属性（见 5.2 节） |
| `spec.devices[].capacity` | ✅ | 设备容量（见 5.3 节） |
| `spec.devices[].allowMultipleAllocations` | ✅ | 必须为 `true`（支持 vGPU） |

### 6.3 ResourceClaimTemplate 规范

#### 6.3.1 必须的 Label

```yaml
metadata:
  labels:
    tensor-fusion.ai/resource-claim-template: "true"
```

**用途**: ResourceClaim Controller 通过此 label 识别需要处理的 ResourceClaim。

#### 6.3.2 模板结构

```yaml
apiVersion: resource.k8s.io/v1beta2
kind: ResourceClaimTemplate
metadata:
  name: tensor-fusion-gpu-template
  labels:
    tensor-fusion.ai/resource-claim-template: "true"
spec:
  spec:
    devices:
      requests:
        - name: gpu-request
          allocationMode: ExactCount
          exactly:
            # 默认值，由 Controller 更新
            count: 1
            selector:
              cel:
                # 默认值，由 Controller 更新
                expression: "true"
            capacity:
              requests: {}  # 由 Controller 填充
```

### 6.4 DeviceClass（说明： 该device class 和 dra driver的名称对应，请重写）

**重要**: Tensor Fusion 当前**不使用** `DeviceClass` 资源。

**原因**:
- ResourceClaimTemplate 已足够灵活
- 减少用户需要管理的资源类型
- 设备配置高度动态，不适合预定义

**未来规划**: 如有需求，可能引入 DeviceClass 用于标准化配置。

---

## 附录: 完整示例

### 示例 1: 基础 DRA 配置

```yaml
# 1. SchedulingConfigTemplate
apiVersion: tensor-fusion.ai/v1
kind: SchedulingConfigTemplate
metadata:
  name: default-config
spec:
  dra:
    enable: true
  placement:
    mode: CompactFirst

---
# 2. ResourceClaimTemplate
apiVersion: resource.k8s.io/v1beta2
kind: ResourceClaimTemplate
metadata:
  name: tensor-fusion-gpu-template
  labels:
    tensor-fusion.ai/resource-claim-template: "true"
spec:
  spec:
    devices:
      requests:
        - name: gpu-request
          allocationMode: ExactCount
          exactly:
            count: 1
            selector:
              cel:
                expression: "true"

---
# 3. GPUPool
apiVersion: tensor-fusion.ai/v1
kind: GPUPool
metadata:
  name: default-pool
spec:
  schedulingConfigTemplate: "default-config"

---
# 4. Pod
apiVersion: v1
kind: Pod
metadata:
  name: my-workload
  annotations:
    tensor-fusion.ai/dra-enabled: "true"
    tensor-fusion.ai/tflops-request: "100"
    tensor-fusion.ai/vram-request: "16Gi"
    tensor-fusion.ai/gpupool: "default-pool"
spec:
  containers:
    - name: app
      image: nvidia/cuda:12.0-base
```

### 示例 2: 高级 DRA 配置（多 GPU + 自定义 CEL）

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: multi-gpu-workload
  annotations:
    # 启用 DRA
    tensor-fusion.ai/dra-enabled: "true"

    # 多 GPU
    tensor-fusion.ai/gpu-count: "4"

    # 资源需求
    tensor-fusion.ai/tflops-request: "200"
    tensor-fusion.ai/vram-request: "40Gi"

    # 自定义 CEL
    tensor-fusion.ai/dra-cel-expression: |
      (device.attributes["model"] == "H100" || device.attributes["model"] == "A100") &&
      device.capacity["vram"].AsApproximateFloat64() >= 40000000000 &&
      device.attributes["qos"] != "low" &&
      device.attributes["node_total_gpus"] >= 8 &&
      device.attributes["phase"] == "Running"

    # Pool
    tensor-fusion.ai/gpupool: "production-pool"
spec:
  schedulerName: tensor-fusion-scheduler
  containers:
    - name: training
      image: pytorch/pytorch:2.0-cuda12.0
      env:
        - name: NCCL_DEBUG
          value: INFO
```

---

**文档结束**

如有更多问题，请参考：
- Kubernetes DRA 官方文档: https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/
- Tensor Fusion 项目地址: https://github.com/NexusGPU/tensor-fusion
