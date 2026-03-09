# TensorFusion Constants Deep Dive: Labels & Annotations

This document provides a technical deep dive into the metadata-driven architecture of TensorFusion. It categorizes every constant from `internal/constants/constants.go` by its functional role and explains the underlying mechanisms.

---

## 1. Core Connectivity & Lifecycle (Labels)
These labels control how Pods are recognized by the system and how nodes are organized.

| Constant | Key | Technical Mechanism |
| :--- | :--- | :--- |
| `TensorFusionEnabledLabelKey` | `tensor-fusion.ai/enabled` | **Mechanism**: Monitored by the Mutating Webhook. If `true` at Pod or Namespace level, the Webhook injects sidecars, env vars, and CDI devices. |
| `GpuPoolKey` | `tensor-fusion.ai/gpupool` | **Mechanism**: Used by the Scheduler for filtering. Nodes are labeled with pools; Pods with this label are restricted to matching nodes. |
| `LabelComponent` | `tensor-fusion.ai/component` | **Mechanism**: Used for RBAC and log filtering. Values: `hypervisor` (on nodes), `operator` (management), `client` (user apps). |
| `SchedulingDoNotDisruptLabel` | `tensor-fusion.ai/do-not-disrupt` | **Mechanism**: The Rebalancer/Descheduler skips any Pod with this label. Essential for non-restartable jobs. |
| `ProvisionerLabelKey` | `tensor-fusion.ai/node-provisioner` | **Mechanism**: Used during node registration to link a physical K8s node to the TensorFusion `GPUNode` CRD. |

---

## 2. Fine-Grained Resource Specification (Annotations)
These define the "Slices" of the GPU.

| Constant | Key | Details & Constraints |
| :--- | :--- | :--- |
| `TFLOPSRequestAnnotation` | `tensor-fusion.ai/tflops-request` | **Unit**: Milli-TFLOPS (e.g., `5000m` = 5 TFLOPS). Controls compute time-slicing on the GPU. |
| `VRAMRequestAnnotation` | `tensor-fusion.ai/vram-request` | **Unit**: Ki/Mi/Gi. Enforced by the Hypervisor using memory management hooks to prevent OOM for other tenants. |
| `ComputeRequestAnnotation` | `tensor-fusion.ai/compute-percent-request` | **Warning**: **Not Recommended**. Bypasses Namespace Quotas. Used only for simple sharing without accounting. |
| `ContainerGPUCountAnnotation` | `tensor-fusion.ai/container-gpu-count` | **Format**: JSON map. e.g., `{"c1": 1}`. Overrides Pod-level GPU requests for specific containers. |
| `ContainerGPUsAnnotation` | `tensor-fusion.ai/container-gpus` | **Mechanism**: **System Set**. Maps container names to specific physical GPU UUIDs. |

---

## 3. Deployment Modes & Networking
Controls *where* and *how* the vGPU worker runs.

| Constant | Key | Description |
| :--- | :--- | :--- |
| `SidecarWorkerAnnotation` | `tensor-fusion.ai/sidecar-worker` | **Default**. Worker runs in the same Pod. Best for low latency via local Unix Domain Sockets/SHM. |
| `EmbeddedWorkerAnnotation` | `tensor-fusion.ai/embedded-worker` | Worker logic is linked into the user process. No sidecar. Lowest overhead. |
| `DedicatedWorkerAnnotation` | `tensor-fusion.ai/dedicated-worker` | Worker runs in a separate Pod. Best for decoupling lifecycle or sharing one Worker across many Clients. |
| `IsLocalGPUAnnotation` | `tensor-fusion.ai/is-local-gpu` | If `true`, scheduler prioritizes local GPUs on the same node. If `false`, enables "Remote vGPU" over the network. |
| `GenHostPortLabel` | `tensor-fusion.ai/host-port` | If `auto`, the system allocates a unique `NodePort` for the worker to receive remote traffic. |

---

## 4. Advanced Advanced Scheduling & QoS
Policy-driven annotations for production workloads.

| Constant | Key | Values & Behavior |
| :--- | :--- | :--- |
| `QoSLevelAnnotation` | `tensor-fusion.ai/qos` | `critical`, `high`, `medium`, `low`. Higher QoS can preempt lower QoS when resources are tight. |
| `IsolationModeAnnotation` | `tensor-fusion.ai/isolation` | `hard` (MIG/MPS hard limit), `shared` (time-sliced), `partitioned` (hardware partitioned). |
| `DedicatedGPUAnnotation` | `tensor-fusion.ai/dedicated-gpu` | If `true`, the scheduler will not place any other Pods on the assigned physical GPUs. |
| `EvictionProtectionAnnotation` | `tensor-fusion.ai/eviction-protection` | **Format**: Duration (e.g., `30m`). Prevents preemption even if a higher QoS Pod arrives. |

---

## 5. Usage Scenarios & Combinations Guide

This section categorizes Labels and Annotations into **User-Provided** and **System-Injected**, providing combinations for both basic and advanced scenarios.

### A. Basic Scenarios (Fundamentals)

#### Scenario A1: Fractional GPU Sharing
**Goal**: Allow multiple small inference tasks to run on the same GPU.
*   **User-Provided**:
    ```yaml
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
        tensor-fusion.ai/gpupool: tensor-fusion-shared
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/vram-request: 512Mi
        tensor-fusion.ai/tflops-request: 5000m  # 5 TFLOPS
    ```
*   **System-Injected**: `tensor-fusion.ai/index` (CDI Index), `tensor-fusion.ai/gpu-ids` (Physical IDs).

#### Scenario A2: Dedicated High-Performance GPU (Dedicated GPU)
**Goal**: Ensure a heavy job has exclusive access to a physical card.
*   **User-Provided**:
    ```yaml
    metadata:
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/dedicated-gpu: 'true'
        tensor-fusion.ai/isolation: hard
    ```

---

### B. Advanced Scenarios (Complex Orchestration)

#### Scenario B1: Remote vGPU (Serverless Computation)
**Goal**: Run GPU logic on a remote high-end node while keeping the client lightweight.
*   **User-Provided**:
    ```yaml
    metadata:
      labels:
        tensor-fusion.ai/enabled: 'true'
        tensor-fusion.ai/host-port: auto # Auto-allocate network port
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/is-local-gpu: 'false' # Force remote execution
        tensor-fusion.ai/sidecar-worker: 'true'
    ```
*   **System-Injected**: `tensor-fusion.ai/port-number` (Allocated host port).

#### Scenario B2: High-Availability Training (Anti-Preemption)
**Goal**: Prevent long-running training from being interrupted by cluster rebalancing.
*   **User-Provided**:
    ```yaml
    metadata:
      labels:
        tensor-fusion.ai/do-not-disrupt: 'true' # Disable rebalancing
      annotations:
        tensor-fusion.ai/gpu-count: '8'
        tensor-fusion.ai/qos: critical        # Highest scheduling priority
        tensor-fusion.ai/eviction-protection: 4h # 4-hour hard protection
    ```

#### Scenario B3: Vertical GPU Autoscaling for Inference (Autoscaling)
**Goal**: Automatically increase GPU resources as traffic grows.
*   **User-Provided**:
    ```yaml
    metadata:
      annotations:
        tensor-fusion.ai/gpu-count: '1'
        tensor-fusion.ai/autoscale: 'true'
        tensor-fusion.ai/autoscale-target: all
    ```
*   **System-Injected**: `tensor-fusion.ai/vram-request` and `tensor-fusion.ai/tflops-request` will be **dynamically injected by the Autoscaler** based on recommendations; users do not need to maintain them manually.

---

## 6. Internal Status Annotations (Read-Only)
These are set by the system. Use them for debugging with `kubectl get pod -o yaml`.

- `tensor-fusion.ai/gpu-ids`: The actual UUIDs of the GPUs being used.
- `tensor-fusion.ai/index`: The unique index assigned for CDI mounting.
- `tensor-fusion.ai/port-number`: The network port assigned to the Worker sidecar.
- `tensor-fusion.ai/gpu-released`: Set to `true` when the scheduler has confirmed cleanup.
