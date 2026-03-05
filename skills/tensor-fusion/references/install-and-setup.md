# Installation and Setup (this repo)

## Choose a path
- Production/quick install: use Helm chart (`charts/tensor-fusion`).
- Local development/debug: use Make + Kustomize (`config/`, `Makefile`).

## Helm install (recommended)
1) Pick a values file as baseline
- Default: `charts/tensor-fusion/values.yaml`
- Production example: `charts/tensor-fusion/values-production.yaml`
- Multi-vendor example: `charts/tensor-fusion/values-multi-vendor.yaml`
- Chinese example: `charts/tensor-fusion/values-cn.yaml`

2) Install/upgrade
```bash
# First install
helm install tensor-fusion ./charts/tensor-fusion \
  -n tensor-fusion --create-namespace \
  -f charts/tensor-fusion/values.yaml

# Upgrade
helm upgrade tensor-fusion ./charts/tensor-fusion \
  -n tensor-fusion \
  -f <your-values>.yaml
```

3) Key Helm overrides (as needed)
- `controller.image.repository/tag`: Operator image.
- `controller.command`: Operator args (enable autoscale/alert/expander).
- `schedulerConfig`: scheduler plugin config (align with `config/samples/scheduler-config.yaml`).
- `dynamicConfig`: global dynamic config and alert rules.
- `initialGpuNodeLabelSelector`: initial GPU node selector label (default `nvidia.com/gpu.present=true`).
- `agent`/`greptime`/`alert`: observability/control-plane components.

## Kustomize/Make install (dev/test)
1) Prepare dependencies and patches
```bash
make vendor  # vendor deps and apply patches
```

2) Install CRDs and deploy the Operator
```bash
make install   # install CRDs
make deploy    # deploy controller
```

3) Run controller locally (dev)
```bash
make run
```

## GPU node preparation
- Ensure GPU nodes have the right labels or match `GPUPool.spec.nodeManagerConfig.nodeSelector`.
- For auto-provisioning (Provisioned/Karpenter), ensure `TensorFusionCluster.spec.computingVendor` and `GPUNodeClass` are configured.

## Verification
- `kubectl get gpupools,gpunodes,gpus -A`: pool/node/GPU status.
- `kubectl get tensorfusionworkloads,tensorfusionconnections -A`: workloads and connections.
- `kubectl get pods -n tensor-fusion`: Operator/scheduler/components are healthy.
