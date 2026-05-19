# Backup: tensor-fusion main state @ 2026-05-16

Snapshot of the cluster state immediately before validating
v1 → main 2.6.4 upgrade per `tensor-fusion/promts.md`.

Cluster: tf-dev (k3s), 3 nodes (rhzs control, ubuntu+kylin-pc workers)

## Pre-backup state

- Helm release: `tensor-fusion-sys/tensor-fusion-sys` chart `tensor-fusion-1.5.6`
  (appVersion 1.43.5), installed 2025-09-04, revision 1 — managed via
  in-place image overrides since.
- Operator image (running): `tensor-fusion-operator:fix-cap-drift-441e72e-local`
- Hypervisor image (running): `tensor-fusion-hypervisor:dev`
- GPU nodes:
  - `ubuntu`: 2x NVIDIA RTX 3090 (isolationMode=hard)
  - `kylin-pc`: 1x Ascend 310P3 NPU (isolationMode=partitioned)
- ProviderConfigs: nvidia-provider, ascend-provider (main-only feature)
- TensorFusionWorkloads: 32 stale Pending entries (no worker pods running)

## Layout

| Dir | Contents |
|---|---|
| helm/ | helm get all / values / manifest / release |
| crds/ | 12 TF CRD schemas |
| crs/ | All TF custom-resource instances (per-kind YAML lists) |
| cluster/ | ClusterRole/Binding, MutatingWebhookConfig (TF only) |
| nodes/ | All node YAML + labels/annotations as JSON |
| ns/ | tensor-fusion-sys ns: deploy/sts/ds/svc/cm/secret/sa/role |
| pods/ | All pods (yaml + describe), individual hypervisor + operator deploys |

## Restore notes

This snapshot is a **point-in-time view** for diagnostics — it does NOT
auto-restore. Treat the prior-state binary images as the source of truth
for image overrides.
