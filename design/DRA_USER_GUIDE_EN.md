# Tensor Fusion DRA User Guide

## Document Overview

This document is intended for **Tensor Fusion users** and **DRA Driver developers**, introducing how to enable and use Kubernetes Dynamic Resource Allocation (DRA) functionality in Tensor Fusion.

**Target Audience**:
- **Cluster Administrators**: Need to enable and configure DRA
- **Application Developers**: Need to write Pod definitions that support DRA
- **DRA Driver Developers**: Need to integrate with Tensor Fusion DRA

**Document Version**: 1.0
**Last Updated**: 2025-10-14
**Kubernetes Version Requirements**: 1.34+ (DRA v1beta2)

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Prerequisites for Enabling DRA](#2-prerequisites-for-enabling-dra)
3. [DRA Configuration Steps](#3-dra-configuration-steps)
4. [Pod Annotation Field Description](#4-pod-annotation-field-description)
5. [CEL Expression Writing Guide](#5-cel-expression-writing-guide)
6. [DRA Driver Integration Guide](#6-dra-driver-integration-guide)

---

## 1. Quick Start

### 1.1 Simplest DRA Pod Example

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-gpu-workload
  annotations:
    # Enable DRA
    tensor-fusion.ai/dra-enabled: "true"

    # GPU resource requirements
    tensor-fusion.ai/tflops-request: "100"
    tensor-fusion.ai/vram-request: "16Gi"
    tensor-fusion.ai/gpu-count: "1"

    # GPU pool selection
    tensor-fusion.ai/gpupool: "default"
spec:
  containers:
    - name: workload
      image: nvidia/cuda:12.0-base
```

**Workflow**:
1. Tensor Fusion Webhook detects `dra-enabled: "true"`
2. Automatically generates CEL expression and injects ResourceClaim reference
3. Kubernetes scheduler allocates GPU based on DRA resources
4. Pod starts and obtains allocated GPU

---

## 2. Prerequisites for Enabling DRA

### 2.1 Kubernetes Cluster Requirements

| Item | Requirement | Description |
|------|-------------|-------------|
| **Kubernetes Version** | ≥ 1.34 | DRA v1beta2 API |
| **Feature Gate** | `DynamicResourceAllocation=true` | Must be enabled on kube-apiserver, kube-scheduler, kubelet |
| **Scheduler** | Native kube-scheduler | Supports DRA resource scheduling |

#### 2.1.1 Enable Feature Gate

Add to Kubernetes cluster startup parameters:

```bash
# kube-apiserver
--feature-gates=DynamicResourceAllocation=true

# kube-scheduler
--feature-gates=DynamicResourceAllocation=true

# kubelet
--feature-gates=DynamicResourceAllocation=true
```

## 3. DRA Configuration Steps

### 3.1 Step 1: Create SchedulingConfigTemplate (Global Configuration)

Create file `scheduling-config.yaml`:

```yaml
apiVersion: tensor-fusion.ai/v1
kind: SchedulingConfigTemplate
metadata:
  name: default-scheduling-config
spec:
  # Enable DRA
  dra:
    enable: true
    # Optional: specify ResourceClaimTemplate name
    resourceClaimTemplateName: "tensor-fusion-gpu-template"
  # Other configurations
  placement:
    mode: CompactFirst
```

Apply configuration:
```bash
kubectl apply -f scheduling-config.yaml
```

**Notes**:
- `dra.enable: true` enables DRA for this configuration template
- All Pods under GPUPools using this template will use DRA by default
- `resourceClaimTemplateName` defaults to `"tensor-fusion-gpu-template"`, usually no need to modify

### 3.2 Step 2: Create ResourceClaimTemplate

Create file `resourceclaim-template.yaml`:

```yaml
apiVersion: resource.k8s.io/v1beta2
kind: ResourceClaimTemplate
metadata:
  name: tensor-fusion-gpu-template
  labels:
    # Must set this label
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
                # Default value, automatically updated by Controller
                expression: "true"
```

Apply configuration:
```bash
kubectl apply -f resourceclaim-template.yaml
```

**Important**:
- Must set label `tensor-fusion.ai/resource-claim-template: "true"`
- `count` and `cel.expression` will be automatically updated by ResourceClaim Controller, no manual modification needed

### 3.3 Step 3: Associate GPUPool with SchedulingConfigTemplate

```yaml
apiVersion: tensor-fusion.ai/v1
kind: GPUPool
metadata:
  name: my-gpu-pool
spec:
  # Associate SchedulingConfigTemplate
  schedulingConfigTemplate: "default-scheduling-config"

  # Other configurations...
```

### 3.4 Step 4: Submit DRA Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-workload
  annotations:
    # Enable DRA (Note: this annotation and global config are alternatives, not required)
    tensor-fusion.ai/dra-enabled: "true"

    # Required: resource requirements
    tensor-fusion.ai/tflops-request: "100"
    tensor-fusion.ai/vram-request: "16Gi"

    # Optional: GPU count (default 1)
    tensor-fusion.ai/gpu-count: "2"

    # Required: GPU pool
    tensor-fusion.ai/gpupool: "my-gpu-pool"

    # Optional: GPU model
    tensor-fusion.ai/gpu-model: "A100"

    # Optional: QoS level
    tensor-fusion.ai/qos: "high"
spec:
  schedulerName: tensor-fusion-scheduler  # or use default-scheduler
  containers:
    - name: app
      image: my-app:latest
```

### 3.5 Verify DRA is Working

```bash
# 1. Check Pod status
kubectl get pod my-workload

# 2. Check if ResourceClaim is created
kubectl get resourceclaim

# 3. Check if ResourceClaim is allocated
kubectl get resourceclaim <claim-name> -o yaml | grep allocated

# 4. Check ResourceSlice
kubectl get resourceslice

# 5. View Pod events
kubectl describe pod my-workload
```

**Success Indicators**:
- Pod status becomes Running
- ResourceClaim shows `allocated: true`
- Pod events show "Successfully assigned resources"

---

## 4. Pod Annotation Field Description

### 4.1 Required Fields

#### 4.1.1 `tensor-fusion.ai/dra-enabled`

**Type**: String
**Valid Values**: `"true"`, `"false"`
**Default Value**: Based on SchedulingConfigTemplate configuration

**Description**: Controls whether to enable DRA mode for this Pod.

**Example**:
```yaml
annotations:
  # Explicitly enable DRA
  tensor-fusion.ai/dra-enabled: "true"

  # Or explicitly disable (use traditional GPU Allocator)
  tensor-fusion.ai/dra-enabled: "false"
```

**Priority**:
```
Pod annotation > SchedulingConfigTemplate.spec.dra.enable > Default disabled
```

#### 4.1.2 `tensor-fusion.ai/tflops-request`

**Type**: String (Quantity)
**Unit**: TFlops (trillion floating-point operations per second)
**Example Values**: `"100"`, `"312"`, `"500"`

**Description**: Requested GPU compute power, used for:
1. Generating CEL expression (filtering GPUs with insufficient capacity)
2. Filling ResourceClaim's Capacity.Requests

**Example**:
```yaml
annotations:
  tensor-fusion.ai/tflops-request: "200"  # Request 200 TFlops
```

**Notes**:
- Must be valid Kubernetes Quantity format
- Don't include unit (default is TFlops)

#### 4.1.3 `tensor-fusion.ai/vram-request`

**Type**: String (Quantity)
**Unit**: Bytes (can use Ki, Mi, Gi suffixes)
**Example Values**: `"16Gi"`, `"32Gi"`, `"80Gi"`

**Description**: Requested GPU memory size.

**Example**:
```yaml
annotations:
  tensor-fusion.ai/vram-request: "32Gi"  # Request 32GB memory
```

#### 4.1.4 `tensor-fusion.ai/gpupool`

**Type**: String
**Description**: Specifies which GPU pool to allocate resources from.

**Example**:
```yaml
annotations:
  tensor-fusion.ai/gpupool: "production-pool"
```

**Notes**:
- Must be an existing GPUPool name
- Pool must be associated with a SchedulingConfigTemplate that supports DRA

### 4.2 Optional Fields

#### 4.2.1 `tensor-fusion.ai/gpu-count`

**Type**: String (Integer)
**Default Value**: `"1"`
**Example Values**: `"1"`, `"2"`, `"4"`, `"8"`

**Description**: Number of requested GPUs.

**Example**:
```yaml
annotations:
  tensor-fusion.ai/gpu-count: "4"  # Request 4 GPUs
```

**Notes**:
- DRA mode supports allocating multiple GPUs
- Each GPU will meet the tflops-request and vram-request requirements

#### 4.2.2 `tensor-fusion.ai/gpu-model`

**Type**: String
**Example Values**: `"A100"`, `"H100"`, `"V100"`, `"RTX4090"`

**Description**: Specifies GPU model requirement.

**Example**:
```yaml
annotations:
  tensor-fusion.ai/gpu-model: "A100"  # Only select A100 GPUs
```

**Notes**:
- Must exactly match actual GPU model (case-sensitive)
- If not specified, accepts any GPU model

#### 4.2.3 `tensor-fusion.ai/qos`

**Type**: String
**Valid Values**: `"low"`, `"medium"`, `"high"`, `"critical"`
**Default Value**: `"medium"`

**Description**: Specifies quality of service level, used to select GPUs with corresponding QoS level.

**Example**:
```yaml
annotations:
  tensor-fusion.ai/qos: "high"  # Select GPUs with high QoS
```

**QoS Level Description**:
- `critical`: Highest priority, typically for production-critical workloads
- `high`: High priority, for important workloads
- `medium`: Medium priority (default)
- `low`: Low priority, for testing or development environments

#### 4.2.4 `tensor-fusion.ai/dra-cel-expression`

**Type**: String (CEL expression)
**Description**: Advanced users can provide custom CEL expression, overriding auto-generated expression.

**Example**:
```yaml
annotations:
  tensor-fusion.ai/dra-cel-expression: |
    device.attributes["model"] == "H100" &&
    device.capacity["vram"].AsApproximateFloat64() >= 100 &&
    device.attributes["qos"] == "critical"
```

**Notes**:
- Only advanced users familiar with CEL syntax should use this field
- Incorrect CEL expression will cause Pod to be unschedulable
- See [Chapter 5](#5-cel-expression-writing-guide) for details

### 4.3 Unsupported Fields

The following fields are **ineffective** or **not needed** in DRA mode:

#### 4.3.1 `tensor-fusion.ai/tflops-limit` and `tensor-fusion.ai/vram-limit`

**Reason**: DRA only handles Requests (minimum guarantee during scheduling), Limits are handled by runtime components (Hypervisor/Device Plugin).

**How to Use Limits**:
Still specify in Pod annotations, but these values won't affect DRA scheduling, only read and enforced by Tensor Fusion runtime components during Pod runtime.

```yaml
annotations:
  # DRA scheduling uses requests
  tensor-fusion.ai/tflops-request: "100"
  tensor-fusion.ai/vram-request: "16Gi"

  # Runtime limits (don't affect DRA scheduling)
  tensor-fusion.ai/tflops-limit: "200"
  tensor-fusion.ai/vram-limit: "32Gi"
```

#### 4.3.2 Quota Related Fields

**Description**: DRA's ResourceClaim doesn't include quota information.

---

## 5. CEL Expression Writing Guide

### 5.1 What is CEL?

**CEL (Common Expression Language)** is a declarative expression language used to describe device selection conditions in Kubernetes DRA.

### 5.2 Supported Device Attributes (device.attributes)

| Attribute Name | Type | Description | Example Values |
|----------------|------|-------------|----------------|
| `model` | String | GPU model | `"A100"`, `"H100"` |
| `pool_name` | String | GPU pool | `"production-pool"` |
| `pod_namespace` | String | GPU namespace | `"default"` |
| `uuid` | String | GPU unique ID | `"GPU-xxx"` |
| `phase` | String | GPU status | `"Running"`, `"Pending"` |
| `used_by` | String | Current user | `""` (empty=unused) |
| `node_name` | String | Node location | `"node-gpu-01"` |
| `qos` | String | QoS level | `"low"`, `"medium"`, `"high"`, `"critical"` |
| `node_total_tflops` | String | Node total compute | `"2496"` |
| `node_total_vram` | String | Node total memory | `"320Gi"` |
| `node_total_gpus` | Int | Node total GPUs | `8` |
| `node_managed_gpus` | Int | Managed GPU count | `8` |
| `node_virtual_tflops` | String | Node virtual compute | `"3744"` (oversubscribed) |
| `node_virtual_vram` | String | Node virtual memory | `"480Gi"` (oversubscribed) |

### 5.3 Supported Device Capacity (device.capacity)

| Capacity Name | Type | Description | Example Values |
|---------------|------|-------------|----------------|
| `tflops` | Quantity | Physical GPU compute | `"312"` |
| `vram` | Quantity | Physical GPU memory | `"40Gi"` |
| `virtual_tflops` | Quantity | Per GPU virtual compute | `"468"` (1.5x oversubscription) |
| `virtual_vram` | Quantity | Per GPU virtual memory | `"60Gi"` (1.5x oversubscription) |

**Note**: `virtual_*` capacity is only available when GPUPool has oversubscription configured.

### 5.4 CEL Syntax Basics

#### 5.4.1 Accessing Attributes

```cel
// Access string attributes
device.attributes["model"]
device.attributes["pool_name"]

// Access integer attributes
device.attributes["node_total_gpus"]
```

#### 5.4.2 Accessing Capacity

```cel
// Access Quantity (requires conversion)
device.capacity["tflops"].AsApproximateFloat64()
device.capacity["vram"].AsApproximateFloat64()
```

**Important**: Quantity type must be converted using `.AsApproximateFloat64()` before comparison.

#### 5.4.3 Logical Operators

```cel
// Logical AND
condition1 && condition2

// Logical OR
condition1 || condition2

// Comparison operators
== != < > <= >=
```

### 5.5 CEL Expression Examples

#### Example 1: Basic Selection

```cel
device.attributes["model"] == "A100" &&
device.attributes["pool_name"] == "production" &&
device.attributes["phase"] == "Running"
```

**Description**: Select A100 GPUs in production pool with Running status.

#### Example 2: Capacity Requirements

```cel
device.capacity["tflops"].AsApproximateFloat64() >= 200 &&
device.capacity["vram"].AsApproximateFloat64() >= 32000000000
```

**Description**:
- TFlops ≥ 200
- VRAM ≥ 32GB (32000000000 bytes)

**Note**: VRAM unit is bytes, 32Gi = 34359738368 bytes.

#### Example 3: Multi-Model Selection

```cel
(device.attributes["model"] == "H100" || device.attributes["model"] == "A100") &&
device.capacity["vram"].AsApproximateFloat64() >= 40000000000 &&
device.attributes["qos"] == "high"
```

**Description**: Select H100 or A100, memory ≥ 40GB, QoS is high.

#### Example 4: Node-Level Conditions

```cel
device.attributes["node_total_gpus"] >= 8 &&
device.attributes["node_total_tflops"].AsApproximateFloat64() > 2000 &&
device.attributes["phase"] == "Running"
```

**Description**: Select GPUs on nodes with at least 8 GPUs and total compute exceeding 2000 TFlops.

#### Example 5: Exclusion Conditions

```cel
device.attributes["model"] != "V100" &&
device.attributes["qos"] != "low" &&
device.attributes["phase"] == "Running"
```

**Description**: Exclude V100 GPUs and low QoS GPUs.

### 5.6 Auto-Generated CEL Expression

When you don't provide a custom CEL expression, the system automatically generates one based on the following annotations:

```yaml
annotations:
  tensor-fusion.ai/gpu-model: "A100"
  tensor-fusion.ai/gpupool: "prod-pool"
  tensor-fusion.ai/tflops-request: "200"
  tensor-fusion.ai/vram-request: "32Gi"
  tensor-fusion.ai/qos: "high"
```

**Generated CEL**:
```cel
device.attributes["model"] == "A100" &&
device.attributes["pool_name"] == "prod-pool" &&
device.capacity["tflops"].AsApproximateFloat64() >= 200 &&
device.capacity["vram"].AsApproximateFloat64() >= 34359738368 &&
device.attributes["qos"] == "high" &&
device.attributes["phase"] == "Running" &&
device.attributes["used_by"] == ""
```

**Description**:
- System automatically adds `phase == "Running"` and `used_by == ""`
- VRAM automatically converted to bytes

### 5.7 CEL Writing Recommendations

#### ✅ Recommended Practices

1. **Always check GPU status**:
```cel
device.attributes["phase"] == "Running"
```

2. **Use parentheses for better readability**:
```cel
(condition1 || condition2) && condition3
```

#### ❌ Practices to Avoid

1. **Don't use non-existent attributes**:
```cel
// ❌ Wrong: quota is not a device attribute
device.attributes["quota"]
```

2. **Don't forget Quantity conversion**:
```cel
// ❌ Wrong: Quantity cannot be compared directly
device.capacity["tflops"] >= 100

// ✅ Correct
device.capacity["tflops"].AsApproximateFloat64() >= 100
```

3. **Don't use overly complex expressions**:
```cel
// ❌ Wrong: too complex, hard to maintain
(device.attributes["model"] == "A100" && device.capacity["tflops"].AsApproximateFloat64() >= 200) ||
(device.attributes["model"] == "H100" && device.capacity["tflops"].AsApproximateFloat64() >= 400) ||
(device.attributes["model"] == "V100" && device.capacity["tflops"].AsApproximateFloat64() >= 100 && device.attributes["qos"] == "high")
```

### 5.8 Unsupported Features

#### 5.8.1 Quota

**Reason for Not Supporting**: Quota is namespace-level resource management, not an attribute of individual devices.

#### 5.8.2 Limits

**Reason for Not Supporting**: DRA only handles Requests (scheduling decisions), Limits are runtime constraints.

**Correct Approach**: Specify limits in Pod annotations, but they are read and enforced by hypervisor at runtime.

## 6. DRA Driver Integration Guide

### 6.1 Tensor Fusion DRA Driver Information

#### 6.1.1 Driver Name

```
tensor-fusion.ai.dra-driver
```

**Purpose**:
- Identifies device provider in ResourceSlice.Spec.Driver
- Identifies allocator in ResourceClaim.Status.Allocation
- Kubernetes scheduler matches ResourceSlice and ResourceClaim through this name

### 6.2 ResourceSlice Specification

#### 6.2.1 ResourceSlice Structure Example

```yaml
apiVersion: resource.k8s.io/v1beta2
kind: ResourceSlice
metadata:
  name: tensor-fusion-resource-slice-node-01
  labels:
    tensor-fusion.ai/managed-by: node-01
    kubernetes.io/hostname: node-01
spec:
  # Driver name
  driver: tensor-fusion.ai.dra-driver

  # Node name
  nodeName: node-01

  # Resource pool
  pool:
    name: production-pool
    generation: 123
    resourceSliceCount: 1

  # Device list
  devices:
    - name: gpu-a100-01
      # Device attributes
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
        # ... more attributes

      # Device capacity
      capacity:
        tflops:
          value: "312"
        vram:
          value: "40Gi"
        virtual_tflops:
          value: "468"
        virtual_vram:
          value: "60Gi"

      # Allow multiple allocations (vGPU support)
      allowMultipleAllocations: true
```

#### 6.2.2 Key Field Description

| Field | Required | Description |
|-------|----------|-------------|
| `spec.driver` | ✅ | Must be `tensor-fusion.ai.dra-driver` |
| `spec.nodeName` | ✅ | Kubernetes node name where GPU is located |
| `spec.pool.name` | ✅ | GPU pool name |
| `spec.devices[].name` | ✅ | GPU unique name |
| `spec.devices[].attributes` | ✅ | Device attributes (see section 5.2) |
| `spec.devices[].capacity` | ✅ | Device capacity (see section 5.3) |
| `spec.devices[].allowMultipleAllocations` | ✅ | Must be `true` (supports vGPU) |

### 6.3 ResourceClaimTemplate Specification

#### 6.3.1 Required Label

```yaml
metadata:
  labels:
    tensor-fusion.ai/resource-claim-template: "true"
```

**Purpose**: Tensor Fusion's ResourceClaim Controller identifies ResourceClaims to process through this label.

#### 6.3.2 Template Structure

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
            # Default value, updated by Controller
            count: 1
            selector:
              cel:
                # Default value, updated by Controller
                expression: "true"
            capacity:
              requests: {}  # Filled by Controller
```

---

## Appendix: Complete Examples

### Example 1: Basic DRA Configuration

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

### Example 2: Advanced DRA Configuration (Multi-GPU + Custom CEL)

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: multi-gpu-workload
  annotations:
    # Enable DRA
    tensor-fusion.ai/dra-enabled: "true"

    # Multi-GPU
    tensor-fusion.ai/gpu-count: "4"

    # Resource requirements
    tensor-fusion.ai/tflops-request: "200"
    tensor-fusion.ai/vram-request: "40Gi"

    # Custom CEL
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

**End of Document**

For more information, please refer to:
- Kubernetes DRA Official Documentation: https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/
- Tensor Fusion Project: https://github.com/NexusGPU/tensor-fusion

