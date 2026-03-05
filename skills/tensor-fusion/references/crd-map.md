# CR Relationships and Data Flow

## CRD roles and scope (quick view)
- **Cluster-scoped**
  - `TensorFusionCluster`: top-level cluster config and pool orchestration.
  - `GPUPool`: GPU resource pool (capacity, node management, component config).
  - `GPUNodeClass`: cloud GPU node template (image/network/disk/permissions).
  - `GPUNodeClaim`: on-demand node claim (triggers cloud instance creation).
  - `GPUNode`: GPU node status and capacity view.
  - `GPU`: single GPU device status and partition info.
  - `ProviderConfig`: vendor config (images, metadata, partition templates).
  - `SchedulingConfigTemplate`: scheduling/autoscaling strategy template.
- **Namespaced**
  - `WorkloadProfile`: workload config template.
  - `TensorFusionWorkload`: workload instance (usually auto-created from Pod annotations).
  - `TensorFusionConnection`: client <-> worker binding.
  - `GPUResourceQuota`: namespace-level GPU quota.

> Field source of truth: `config/crd/bases/*.yaml` and `api/v1/*_types.go`.

## Key relationships (conceptual)
```
TensorFusionCluster
  -> GPUPool (many)
     -> NodeManagerConfig -> GPUNodeClaim -> cloud node -> GPUNode
     -> ProviderConfig (per vendor images/metadata)
     -> SchedulingConfigTemplate (placement/autoscaling)
     -> GPU/GPUNode status aggregation

Pod/Deployment
  -> (webhook parses annotations) -> TensorFusionWorkload
     -> WorkloadProfile (optional template)
     -> GPUResourceQuota (defaults/quota checks)
     -> Worker Pods (created by controller in fixed-replica mode)
     -> TensorFusionConnection (client <-> worker binding)
```

## Workload data flow (run tasks)
1) User creates Pod/Deployment with `tensor-fusion.ai/enabled: "true"`.
2) Webhook parses annotations -> create/update `TensorFusionWorkload`.
3) Pod controller creates `TensorFusionConnection` for client Pods (remote vGPU).
4) Workload controller creates/scales worker Pods (fixed replica mode).
5) Scheduler assigns GPU resources and updates `GPU/GPUNode/GPUPool` status.

## Pool and node data flow (provisioning/management)
1) `TensorFusionCluster` creates/maintains multiple `GPUPool`.
2) `GPUPool` uses `NodeManagerConfig`:
   - **AutoSelect**: select existing GPU nodes.
   - **Provisioned/Karpenter**: create `GPUNodeClaim` -> cloud instance -> `GPUNode`.
3) Hypervisor/Worker take over nodes and expose `GPU` resources.
4) `GPUPool` aggregates capacity, availability, cost, and component status.

## CR highlights by type
- **TensorFusionCluster**
  - `spec.gpuPools[]`: embedded pool definitions; controller creates/updates `GPUPool`.
  - `spec.computingVendor`: cloud provider connection for auto-provisioning.

- **GPUPool**
  - `spec.nodeManagerConfig`: node selection/provisioning (`NodeSelector` or `NodeProvisioner`).
  - `spec.capacityConfig`: min/max/warm capacity and oversubscription.
  - `spec.componentConfig`: worker/hypervisor/client images and templates.
  - `spec.schedulingConfigTemplate`: placement/autoscaling template reference.
  - Annotation `tensor-fusion.ai/is-default-pool: "true"`: mark as default pool.

- **ProviderConfig**
  - `spec.vendor`: vendor (NVIDIA/AMD/Ascend, etc.).
  - `spec.images`: component images (override `GPUPool.componentConfig`).
  - `spec.hardwareMetadata`/`virtualizationTemplates`: device capabilities and partition templates.

- **GPUNodeClass / GPUNodeClaim / GPUNode**
  - `GPUNodeClass`: cloud node template.
  - `GPUNodeClaim`: created by pool; triggers cloud instance creation.
  - `GPUNode`: node capacity/state, aggregated by pools.

- **GPU**
  - Device-level status (capacity, available, partitions, usage).

- **SchedulingConfigTemplate**
  - `spec.placement`: placement strategy.
  - `spec.verticalScalingRules`: label-based autoscaling rules.
  - `spec.hypervisor`: queueing and rate-limit tuning.

- **WorkloadProfile / TensorFusionWorkload**
  - `WorkloadProfile`: reusable template (optional).
  - `TensorFusionWorkload`: instance (usually created by webhook).
  - Key fields: `poolName`, `resources`, `qos`, `isLocalGPU`, `isolation`, `gpuCount`.

- **TensorFusionConnection**
  - Binds client Pod to worker; `status.connectionURL` is used by clients.

- **GPUResourceQuota**
  - Namespace-level quota; webhook applies defaults and checks quotas.
