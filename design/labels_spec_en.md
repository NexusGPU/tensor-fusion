# TensorFusion Kubernetes Labels Specification & Best Practices

This document outlines the usage of Kubernetes Labels in the TensorFusion system. While Annotations are used for fine-grained resource requests, **Labels** are primarily used for resource identification, grouping, lifecycle management, and discovery.

---

## 1. Core Label Infrastructure

| Label Key | Purpose | Component | Values / Example |
| :--- | :--- | :--- | :--- |
| `tensor-fusion.ai/enabled` | Activation switch for TensorFusion features. | Webhook, Scheduler | `'true'`, `'false'` |
| `tensor-fusion.ai/component` | Identifies the architectural role of the resource. | All | `client`, `worker`, `hypervisor`, `operator`, `node-discovery` |
| `tensor-fusion.ai/managed-by` | Tracks ownership and controller responsibility. | Controllers | `operator`, `gpupool`, `workload-controller` |
| `tensor-fusion.ai/gpupool` | Groups resources (Pods, Nodes, GPUs) into a logical pool. | Scheduler, Operator | e.g., `shared-pool-a`, `dedicated-h100` |
| `tensor-fusion.ai/cluster` | Multi-cluster identification (if applicable). | Operator | e.g., `prod-us-east-1` |

---

## 2. Resource Discovery & Connectivity

Labels used to facilitate communication between components, especially in remote vGPU scenarios.

| Label Key | Usage Scenario | Description |
| :--- | :--- | :--- |
| `tensor-fusion.ai/worker-name` | Remote vGPU / Connection | Matches `TensorFusionConnection` objects to specific Worker Pods when states change. |
| `tensor-fusion.ai/host-port` | Port Allocation | Set to `auto` to trigger automatic host port allocation for the worker. |
| `tensor-fusion.ai/port-name` | Service Discovery | Names the allocated port for easier reference in connection strings. |
| `tensor-fusion.ai/node-provisioner` | Cloud Integration | Matches `GPUNode` objects with underlying K8s nodes (used in cloud-init/userdata). |

---

## 3. Stability & Scheduling Control

| Label Key | Usage Scenario | Description |
| :--- | :--- | :--- |
| `tensor-fusion.ai/do-not-disrupt` | Maintenance / Eviction | Similar to Karpenter, prevents the Pod or Node from being moved or destroyed during rebalancing. |
| `tensor-fusion.ai/pod-template-hash` | Rollout Management | Used by controllers to identify different versions of Worker Pod templates. |
| `tensor-fusion.ai/node-selector-hash` | Scheduling | Internal hash used to optimize node matching logic in the scheduler. |

---

## 5. Common Usage Scenarios & YAML Examples

Strategic use of labels ensures efficient cluster operations and workload stability.

### A. Scenario: Team-wide Feature Activation
Manage feature toggles at the Namespace level instead of per-Pod configuration.

```yaml
# 1. Label the entire namespace
apiVersion: v1
kind: Namespace
metadata:
  name: ml-team-alpha
  labels:
    # Enable TensorFusion for all Pods in this namespace
    tensor-fusion.ai/enabled: 'true'
---
# 2. Deployments in this namespace work automatically
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

### B. Scenario: Eviction Protection for Long-running Jobs
Prevent the system from interrupting critical jobs during rebalancing or node maintenance.

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
        # Core Label: Prevent disruption during cluster optimization
        tensor-fusion.ai/do-not-disrupt: 'true'
      annotations:
        tensor-fusion.ai/gpu-count: '8'
        tensor-fusion.ai/qos: 'critical'
    spec:
      containers:
        - name: trainer
          image: pytorch/pytorch:latest
```

### C. Scenario: Heterogeneous Scheduling via Pools
Direct workloads to specific hardware pools (e.g., A100 vs T4) using logical identifiers.

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
        # Core Label: Schedule only within the high-performance pool
        tensor-fusion.ai/gpupool: a100-80gb-pool
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/vram-request: '40Gi'
    spec:
      containers:
        - name: server
          image: vllm/vllm-openai:latest
```

### D. Scenario: Component Identification for Operations
Tag manually deployed tools (e.g., debuggers) to ensure they are tracked by monitoring.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-debug-tool
  labels:
    # Core Label: Mark as a hypervisor component for observability
    tensor-fusion.ai/component: hypervisor
    # Core Label: Mark as managed-by operator for lifecycle consistency
    tensor-fusion.ai/managed-by: operator
spec:
  containers:
    - name: debugger
      image: nvidia/samples:vectoradd
```
