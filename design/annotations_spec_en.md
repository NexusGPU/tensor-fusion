# TensorFusion Kubernetes Annotations & Labels Specification

This document provides a comprehensive guide to the custom Kubernetes Annotations and Labels used by the TensorFusion system. These metadata keys, prefixed with `tensor-fusion.ai/`, control GPU resource scheduling, workload injection, isolation, and automated lifecycle management.

---

## 1. Resource Request & Limit Annotations
These annotations are specified by users in the Pod metadata to define fine-grained GPU requirements, moving beyond the standard integer-based `nvidia.com/gpu` resource type.

| Annotation | Key | Description | Example |
| :--- | :--- | :--- | :--- |
| **GPU Count** | `tensor-fusion.ai/gpu-count` | Total number of physical GPUs required by the Pod. | `"2"` |
| **TFLOPS Request** | `tensor-fusion.ai/tflops-request` | Requested GPU compute performance in TFLOPS. Used for compute slicing. | `"5.5"` |
| **TFLOPS Limit** | `tensor-fusion.ai/tflops-limit` | Maximum compute performance limit in TFLOPS. | `"10.0"` |
| **VRAM Request** | `tensor-fusion.ai/vram-request` | Requested GPU Video RAM. | `"10Gi"`, `"500Mi"` |
| **VRAM Limit** | `tensor-fusion.ai/vram-limit` | Maximum GPU Video RAM limit. | `"20Gi"` |
| **Compute Percent** | `tensor-fusion.ai/compute-percent-request` | Requested percentage of a single GPU's compute (alternative to TFLOPS). | `"50"` |
| **GPU Vendor** | `tensor-fusion.ai/vendor` | Preferred hardware vendor (e.g., `nvidia`, `amd`). | `"nvidia"` |
| **GPU Model** | `tensor-fusion.ai/gpu-model` | Required GPU hardware model. | `"A100"`, `"H100"` |

---

## 2. Scheduling & Resource Binding
Used by the **TensorFusion Scheduler** to manage resource allocation and track assignments.

| Annotation | Key | Description |
| :--- | :--- | :--- |
| **GPU ID Assignment** | `tensor-fusion.ai/gpu-ids` | **(System Set)** Comma-separated list of physical GPU IDs assigned by the scheduler. |
| **GPU Indices** | `tensor-fusion.ai/gpu-indices` | Comma-separated list of specific GPU physical indices to use (for pinning). |
| **QoS Level** | `tensor-fusion.ai/qos` | Priority for scheduling and preemption: `low`, `medium`, `high`, `critical`. |
| **Dedicated GPU** | `tensor-fusion.ai/dedicated-gpu` | If `"true"`, ensures the Pod has exclusive access to the assigned physical GPUs. |
| **Isolation Mode** | `tensor-fusion.ai/isolation` | Resource isolation strategy: `soft` (default), `shared`, `hard`, or `partitioned`. |
| **GPU Released** | `tensor-fusion.ai/gpu-released` | **(System Set)** Marked when the Pod has finished using its assigned GPUs. |

---

## 3. Webhook & Injection Configuration
Used by the **Mutating Webhook** to automate sidecar injection and runtime environment setup.

| Annotation | Key | Description |
| :--- | :--- | :--- |
| **Inject Container** | `tensor-fusion.ai/inject-container` | Comma-separated list of business containers to inject with TensorFusion. |
| **Workload Profile** | `tensor-fusion.ai/workload-profile` | Reference to a `WorkloadProfile` CRD to apply templated configurations. |
| **Container GPU Map** | `tensor-fusion.ai/container-gpu-count` | JSON mapping defining GPU counts per container for multi-container Pods. |
| **Pod Index** | `tensor-fusion.ai/index` | **(Internal)** A unique index (1-512) injected for Device Plugin/CDI communication. |
| **Pricing Info** | `tensor-fusion.ai/hourly-pricing` | Used for cost-based scheduling or eviction protection calculations. |

---

## 4. Worker & Connectivity Management
Defines how the TensorFusion Worker (the remote vGPU server component) is deployed and connected.

| Annotation | Key | Description |
| :--- | :--- | :--- |
| **Embedded Worker** | `tensor-fusion.ai/embedded-worker` | If `"true"`, the worker runs inside the client process (no separate container). |
| **Sidecar Worker** | `tensor-fusion.ai/sidecar-worker` | If `"true"`, the worker runs as a sidecar container (standard mode). |
| **Dedicated Worker** | `tensor-fusion.ai/dedicated-worker` | If `"true"`, the worker runs as a completely separate Pod. |
| **Workload Mode** | `tensor-fusion.ai/workload-mode` | Defines lifecycle behavior: `dynamic` or `fixed`. |
| **Port Number** | `tensor-fusion.ai/port-number` | **(System Set)** The host port assigned for client-worker communication. |

---

## 5. Automation & Stability Labels
Labels and annotations used for autoscaling and preventing disruption during maintenance.

| Annotation/Label | Key | Description |
| :--- | :--- | :--- |
| **Autoscale Enable** | `tensor-fusion.ai/autoscale` | Set to `"true"` to enable Vertical GPU Resource Autoscaling. |
| **Autoscale Target** | `tensor-fusion.ai/autoscale-target` | Target resource to scale: `compute`, `vram`, or `all`. |
| **Eviction Protection** | `tensor-fusion.ai/eviction-protection` | Duration (e.g., `"30m"`) to prevent the Pod from being preempted. |
| **Do Not Disrupt** | `tensor-fusion.ai/do-not-disrupt` | **(Label)** Prevents the Pod or Node from being interrupted by the scheduler. |

---

## 6. Management & Identification Labels
Standard labels used for resource filtering and ownership tracking across controllers.

| Label | Key | Description |
| :--- | :--- | :--- |
| **Enabled** | `tensor-fusion.ai/enabled` | Enables TensorFusion features for a Pod or Namespace. |
| **Component** | `tensor-fusion.ai/component` | Identifies the type: `client`, `worker`, `hypervisor`, `operator`. |
| **Managed By** | `tensor-fusion.ai/managed-by` | Indicates which controller owns the resource (e.g., `operator`, `gpupool`). |
| **GPU Pool** | `tensor-fusion.ai/gpupool` | Links a Pod or Node to a specific `GPUPool` resource. |
| **Cluster** | `tensor-fusion.ai/cluster` | Links a resource to a specific `TensorFusionCluster`. |

---

## 8. Common Usage Scenarios & YAML Examples

The following examples demonstrate how to enable various TensorFusion features by applying different annotation combinations to a standard Deployment.

### Scenario A: Fine-grained GPU Sharing (Fractional GPU)
Allows multiple workloads to share a single physical GPU by slicing compute (TFLOPS) and memory (VRAM).

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
        tensor-fusion.ai/enabled: 'true'
        tensor-fusion.ai/gpupool: tensor-fusion-shared
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/gpupool: tensor-fusion-shared
        tensor-fusion.ai/is-local-gpu: 'true'
        tensor-fusion.ai/tflops-request: 17800m
        tensor-fusion.ai/tflops-limit: 35600m
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

### Scenario B: High-Performance Dedicated Training
Ensures exclusive access to multiple physical GPUs with hardware isolation and high priority.

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
        tensor-fusion.ai/gpu-count: '8'
        tensor-fusion.ai/dedicated-gpu: 'true'
        tensor-fusion.ai/isolation: 'hard'
        tensor-fusion.ai/qos: 'critical'
    spec:
      containers:
        - name: test
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
          command: ["sh", "-c", "sleep 99d"]
      terminationGracePeriodSeconds: 30
```

### Scenario C: Cost-Optimized Inference with Autoscaling
Automatically adjusts GPU resources vertically based on load and provides eviction protection.

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
        tensor-fusion.ai/autoscale: 'true'
        tensor-fusion.ai/autoscale-target: 'all'
        tensor-fusion.ai/eviction-protection: '10m'
    spec:
      containers:
        - name: test
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
          command: ["sh", "-c", "sleep 99d"]
      terminationGracePeriodSeconds: 30
```

### Scenario D: Multi-Container GPU Pipeline
Assigns specific GPU resources to different containers within the same Pod.

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
        tensor-fusion.ai/inject-container: 'pre-process,inference'
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

### Scenario E: Remote vGPU Execution (Serverless-like)
Offloads GPU computation to a remote worker node, keeping the client Pod lightweight.

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
        tensor-fusion.ai/sidecar-worker: 'true'
        tensor-fusion.ai/workload-mode: 'dynamic'
        tensor-fusion.ai/is-local-gpu: 'false'
    spec:
      containers:
        - name: test
          image: registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime
          command: ["sh", "-c", "sleep 99d"]
      terminationGracePeriodSeconds: 30
```
