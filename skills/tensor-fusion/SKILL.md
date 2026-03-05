---
name: tensor-fusion
description: Operational guide for the Tensor Fusion (tensor-fusion.ai) Kubernetes GPU virtualization/pooling repo. Use when installing/deploying the operator or Helm chart, configuring CRDs (GPUPool/ProviderConfig/SchedulingConfigTemplate/WorkloadProfile), understanding CR relationships, creating or troubleshooting TensorFusionWorkload/Connection and annotated GPU workloads, and tuning scheduling, autoscaling, and quotas.
---

# Tensor Fusion

## Overview
Use this skill to operate the Kubernetes GPU virtualization/pooling system in this repo. Use it to install, configure pools, create workloads, and troubleshoot.

## Quick navigation (read first)
- Install/deploy/upgrade: read `references/install-and-setup.md`.
- CR relationships and data flow: read `references/crd-map.md`.
- Create tasks/workloads/annotations: read `references/workload-guide.md`.

## Workflow (priority order)
1) Identify the scenario
- Decide between "deployment/configuration" vs "run workloads/troubleshoot".
- Confirm the target environment (cluster, cloud vendor, GPU/NPU model, existing GPU nodes).

2) Establish baseline config
- Ensure the Operator/CRDs are installed and running.
- Ensure there is an available `GPUPool` (set a default pool when possible).
- Create `ProviderConfig` when vendor images/metadata/partition templates are required.

3) Create tasks (workloads)
- Prefer Pod/Deployment annotations to auto-create `TensorFusionWorkload`.
- Manually create `TensorFusionWorkload` only when fixed worker replicas or advanced control is needed.

4) Observe and troubleshoot
- Check `GPUPool/GPUNode/GPU/TensorFusionWorkload/TensorFusionConnection` status.
- Verify `tensor-fusion-scheduler` and annotations are effective.

## Key paths (read when needed)
- CRDs and fields: `config/crd/bases/` and `api/v1/*_types.go` (CRD is source of truth).
- Helm/deploy config: `charts/tensor-fusion/values*.yaml`.
- Pod annotation parsing/injection: `internal/webhook/v1/tf_parser.go`, `internal/utils/compose.go`.
- Controllers: `internal/controller/*`.
- Scheduler config: `config/samples/scheduler-config.yaml` and Helm `values.yaml`.

## Required questions before creating tasks
- What cluster and namespace? Is Tensor Fusion already installed?
- Local GPU or remote vGPU? (maps to `isLocalGPU`)
- Which GPU pool? Is a default pool set?
- Required resources: TFLOPs, VRAM, GPU count/model, QoS, isolation mode?
- Need autoscaling/cron scaling/external scaler?
- Need cloud provider node provisioning (Provisioned/Karpenter)?

## Important notes
- Sample YAML can be stale; CRD definitions are source of truth.
- TFLOPs/VRAM and compute-percent are mutually exclusive; compute-percent bypasses quota.
- Remote vGPU auto-creates `TensorFusionConnection`; usually no manual creation.
- Set default pool via annotation `tensor-fusion.ai/is-default-pool: "true"`.

## Resources
- `references/install-and-setup.md`: installation/deployment and runtime options.
- `references/crd-map.md`: CR relationships and data flow.
- `references/workload-guide.md`: task creation, annotations, and examples.
