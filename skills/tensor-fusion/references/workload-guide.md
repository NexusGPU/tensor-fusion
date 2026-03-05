# Workload Creation Guide

## Choose a mode
- **Local GPU**: `tensor-fusion.ai/is-local-gpu: "true"`
  - Use when the client should run on the same GPU node.
  - Pod uses `tensor-fusion-scheduler`.
- **Remote vGPU**: `tensor-fusion.ai/is-local-gpu: "false"`
  - System creates `TensorFusionConnection` and workers.

## Recommended path: Pod/Deployment annotations (auto-create TensorFusionWorkload)
1) Minimum required
- Label: `tensor-fusion.ai/enabled: "true"`
- Annotations (resources):
  - `tensor-fusion.ai/tflops-request`
  - `tensor-fusion.ai/vram-request`
  - (optional) `tensor-fusion.ai/tflops-limit` / `tensor-fusion.ai/vram-limit`
- Annotations (pool and container):
  - `tensor-fusion.ai/gpupool` (required if no default pool)
  - `tensor-fusion.ai/inject-container` (required for multi-container, comma-separated)

2) Common optional annotations
- `tensor-fusion.ai/qos`: `low|medium|high|critical`
- `tensor-fusion.ai/gpu-count`: GPU count
- `tensor-fusion.ai/gpu-indices`: explicit GPU indices (e.g., `0,1`)
- `tensor-fusion.ai/gpu-model`: GPU model requirement
- `tensor-fusion.ai/vendor`: GPU vendor requirement
- `tensor-fusion.ai/isolation`: `shared|soft|hard|partitioned`
- `tensor-fusion.ai/partition-id` / `tensor-fusion.ai/partition`: partition mode
- `tensor-fusion.ai/autoscale`: `true` to enable autoscaling
- `tensor-fusion.ai/autoscale-target`: `compute|vram|all`
- `tensor-fusion.ai/gang-min-members` / `tensor-fusion.ai/gang-timeout`
- `tensor-fusion.ai/dedicated-gpu`: `true` (requires `gpu-model`)
- `tensor-fusion.ai/compute-percent-request` / `tensor-fusion.ai/compute-percent-limit`: percent-based resources (mutually exclusive with TFLOPs)

3) Minimal example (single container)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pytorch-demo
  namespace: tensor-fusion
spec:
  replicas: 1
  selector:
    matchLabels:
      app: pytorch-demo
  template:
    metadata:
      labels:
        app: pytorch-demo
        tensor-fusion.ai/enabled: "true"
      annotations:
        tensor-fusion.ai/gpupool: "shared-tensor-fusion"
        tensor-fusion.ai/tflops-request: "10"
        tensor-fusion.ai/vram-request: "1Gi"
        tensor-fusion.ai/is-local-gpu: "true"
        tensor-fusion.ai/inject-container: "pytorch"
    spec:
      containers:
      - name: pytorch
        image: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
        command: ["sh", "-c", "sleep infinity"]
```

## Use WorkloadProfile templates (team reuse)
1) Create a `WorkloadProfile`
```yaml
apiVersion: tensor-fusion.ai/v1
kind: WorkloadProfile
metadata:
  name: default-profile
  namespace: tensor-fusion
spec:
  poolName: shared-tensor-fusion
  resources:
    requests:
      tflops: "10"
      vram: "1Gi"
    limits:
      tflops: "10"
      vram: "1Gi"
  qos: medium
  isLocalGPU: true
```

2) Reference it in Pod/Deployment annotations
```yaml
metadata:
  labels:
    tensor-fusion.ai/enabled: "true"
  annotations:
    tensor-fusion.ai/workload-profile: "default-profile"
    tensor-fusion.ai/inject-container: "pytorch"
```

## Advanced: manually create TensorFusionWorkload (fixed worker replicas)
- Use only when you need fixed worker replicas or advanced control.
- Still annotate the client Pod so `TensorFusionConnection` is created.
```yaml
apiVersion: tensor-fusion.ai/v1
kind: TensorFusionWorkload
metadata:
  name: fixed-workload
  namespace: tensor-fusion
spec:
  poolName: shared-tensor-fusion
  replicas: 2
  resources:
    requests:
      tflops: "10"
      vram: "1Gi"
    limits:
      tflops: "10"
      vram: "1Gi"
  qos: medium
  isLocalGPU: false
```

## Rules and cautions
- TFLOPs and compute-percent are mutually exclusive; prefer TFLOPs.
- compute-percent bypasses quotas; use with care.
- Multi-container workloads must set `tensor-fusion.ai/inject-container`.
- Remote vGPU auto-creates `TensorFusionConnection`.

## Troubleshooting
- "gpu pool not found": ensure `GPUPool` exists and a default pool is set.
- No container injection: check `tensor-fusion.ai/inject-container` matches container names.
- Resources not applied: ensure TFLOPs and compute-percent are not both set.
- "tflops request is not set": add `tensor-fusion.ai/tflops-request` or `tensor-fusion.ai/compute-percent-request`.
