# TensorFusion Kubernetes Installation Guide

This document explains how to install the TensorFusion control plane in a Kubernetes cluster and complete a minimal workload Pod verification.

The official TensorFusion images are published on [Docker Hub](https://hub.docker.com/u/tensorfusion). To use the latest
version, check the available tags of the operator, hypervisor, client, and worker on that page, then override the image
fields in the Helm values or in TensorFusionCluster/ProviderConfig.

Assumptions:

- The installation uses the Helm chart in this repository; the chart path is `charts/tensor-fusion`.
- The Helm release name is `tensor-fusion-sys` and the system namespace is `tensor-fusion-sys`.
- For GPU functional verification, the cluster already has GPU nodes, and the nodes can be selected by a label such as `nvidia.com/gpu.present=true`. Verifying a control plane installation without GPUs does not require this.
- The account that runs the installation has permission to create CRDs, ClusterRoles, MutatingWebhookConfigurations, ValidatingWebhookConfigurations, and Namespaces.

## 1. Prerequisites

Before installing, confirm the basic tools and the cluster state:

```bash
kubectl version --client=true
helm version
kubectl get nodes -o wide
kubectl get nodes --show-labels | grep -E 'nvidia.com/gpu.present=true|gpu.present=true'
```

GPU nodes must meet the following:

- The NVIDIA driver works correctly.
- The container runtime can mount GPU devices.
- GPU nodes carry a label that TensorFusion can recognise. The default Helm parameter `initialGpuNodeLabelSelector` is `nvidia.com/gpu.present=true`.
- If remote mode is used, the network from the client Pod to the node where the worker Pod runs must be reachable.

## 2. Install the control plane

### 2.1 Default installation

The default values install:

- TensorFusion CRDs, RBAC, and the controller.
- Mutating admission webhook.
- Validating admission webhook (`<release>-validating-webhook`), which only handles updates of `pods/ephemeralcontainers`, see 2.5.
- TensorFusion scheduler configuration.
- The NVIDIA ProviderConfig, the SchedulingConfigTemplate, and the default NVIDIA TensorFusionCluster/GPUPool.
- GreptimeDB standalone.
- Alertmanager.
- vector sidecar (the image is pinned to `timberio/vector:0.56.0-alpine`). The controller Pod always has a vector sidecar; each hypervisor Pod also has one by default (`cluster.enableVector`, see 2.5). cluster-agent is disabled by default and is enabled only after `agent.agentId` is explicitly configured.

The control plane can also be installed in a cluster without GPUs. As long as no node matches
`initialGpuNodeLabelSelector`, the operator does not create GPUNodes or Hypervisor Pods; in that case only verify that the
controller, CRDs, RBAC, webhook, and default CRs are correct.

Run from the repository root:

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys \
  --create-namespace
```

If the cluster cannot pull overseas images reliably, use the values for registries in mainland China:

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys \
  --create-namespace \
  -f charts/tensor-fusion/values-cn.yaml
```

### 2.2 Production installation

For production, an external highly available GreptimeDB is recommended instead of the standalone instance built into the chart.

`charts/tensor-fusion/values-production.yaml` only sets `greptime.installStandalone=false`; it does not fill in the external GreptimeDB address and credentials automatically. You need to pass `greptime.host`, `greptime.port`, and `greptime.db` in addition, and if you use cloud Greptime, also set `greptime.isCloud=true`, `greptime.user`, and `greptime.password`.

Example:

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys \
  --create-namespace \
  -f charts/tensor-fusion/values-production.yaml \
  --set greptime.installStandalone=false \
  --set greptime.isCloud=true \
  --set greptime.host='<greptime-host>' \
  --set greptime.port=5001 \
  --set greptime.db='public' \
  --set greptime.user='<greptime-user>' \
  --set greptime.password='<greptime-password>'
```

If the external Greptime needs no username and password, keep `greptime.isCloud=false` and set a reachable host, port, and db.

### 2.3 Common Helm parameters

| Parameter | Description | Default |
| --- | --- | --- |
| `initialGpuNodeLabelSelector` | Label selector the controller uses for its initial scan of GPU nodes | `nvidia.com/gpu.present=true` |
| `controller.image.repository` | operator image repository | `tensorfusion/tensor-fusion-operator` |
| `controller.image.tag` | operator image tag; when empty, the Chart `appVersion` is used | `""` (see `appVersion` in `charts/tensor-fusion/Chart.yaml`) |
| `controller.replicaCount` | Number of controller replicas | `1` |
| `controller.compatibleWithNvidiaContainerToolkit` | Enables the NVIDIA Container Toolkit check and the compatibility mounts for driver libraries | `false` |
| `greptime.installStandalone` | Whether to install the built-in GreptimeDB standalone | `true` |
| `greptime.host` | GreptimeDB MySQL endpoint host | `greptimedb-standalone.greptimedb.svc.cluster.local` |
| `alert.enabled` | Whether to install alertmanager | `true` |
| `controller.admissionWebhooks.failurePolicy` | webhook failure policy | `Fail` |

When the NVIDIA GPU Operator is installed and the GPU nodes provide
`/run/nvidia/validations/toolkit-ready` and `driver-ready`, you can set:

```bash
--set controller.compatibleWithNvidiaContainerToolkit=true
```

This enables the Toolkit readiness check and the discovery of non-standard driver directories. Clusters without these validation files should keep the default
`false`. To override the automatic discovery, set `TF_CUDA_LIB_PATH` and `TF_NVML_LIB_PATH` explicitly through
`ProviderConfig.spec.hypervisor.extraEnv`.

### 2.4 Specifying component versions

Starting with Chart 1.8.3, the defaults pin the component versions that match the Chart `appVersion` (currently `2.17.3`), so a one-command installation no longer depends on
the floating `latest` of the operator / hypervisor / vgpu-provider / client / worker. To use a private registry or
another verified version, override them with `--set`:

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys --create-namespace \
  --set controller.image.tag=2.17.3 \
  --set cluster.hypervisorImage=tensorfusion/tensor-fusion-hypervisor:2.17.3 \
  --set providerConfigs.nvidia.images.middleware=tensorfusion/vgpu-provider-nvidia:1.3.9 \
  --set providerConfigs.nvidia.images.remoteClient=tensorfusion/tensor-fusion-client:v2.25.0 \
  --set providerConfigs.nvidia.images.remoteWorker=tensorfusion/tensor-fusion-worker:v2.25.0
```

The values keys for each component:

| Component | values key | Description |
| --- | --- | --- |
| operator | `controller.image.repository` + `controller.image.tag` | Repository and tag are two separate keys |
| hypervisor | `cluster.hypervisorImage` | **Full image** (repository + tag); when empty, `tensorfusion/tensor-fusion-hypervisor:<appVersion>` is used, the same version as the operator |
| vgpu-provider | `providerConfigs.nvidia.images.middleware` | Default `tensorfusion/vgpu-provider-nvidia:1.3.9` |
| client | `providerConfigs.nvidia.images.remoteClient` | Default `tensorfusion/tensor-fusion-client:v2.25.0` |
| worker | `providerConfigs.nvidia.images.remoteWorker` | Default `tensorfusion/tensor-fusion-worker:v2.25.0` |

> Note: `cluster.hypervisorImage` takes a full image reference (`repository:tag`), whereas the operator uses the two separate keys `repository` + `tag`.

When installing from the Helm repository (rather than a local chart path), `--version` can pin the **chart version** (which is independent of the image tags):

```bash
helm repo add tensor-fusion https://nexusgpu.github.io/tensor-fusion --force-update
helm repo update tensor-fusion
helm upgrade --install tensor-fusion-sys tensor-fusion/tensor-fusion --version 1.8.3 \
  -n tensor-fusion-sys --create-namespace
```

### 2.5 Installation behaviour added in this version

**Admission validation of ephemeral containers.** The chart adds a `ValidatingWebhookConfiguration` that matches only `UPDATE` of `pods/ephemeralcontainers`, with no namespace or label selector. The handler takes effect only for Pods that mount the TensorFusion node data volume (a hostPath under `/run/tensor-fusion`): an ephemeral container may not mount this volume in any form other than the injected one, and may not define `TF_SHM_POD_NAMESPACE` or `TF_SHM_POD_NAME` itself; other Pods and ordinary `kubectl debug` containers are admitted directly. The failure policy is controlled by `controller.admissionWebhooks.ephemeralContainers.failurePolicy` and defaults to `Ignore` (timeout 3 seconds), so `kubectl debug` in the cluster is not blocked when the operator is unavailable; clusters with strict isolation requirements can set it to `Fail`. The cluster's own Pod security policy remains the administrator's responsibility, see `docs/pod-security.md`.

**Credentials of cluster-agent.** `agent.enrollToken` and `agent.agentId` no longer appear in the Pod spec: the chart writes them into the Secret `<release>-agent`, and the sidecar reads them through `secretKeyRef`. Existing install commands (`--set agent.agentId=... --set agent.enrollToken=...`, and `helm upgrade --reuse-values`) need no change. If you want to manage the Secret yourself, use this instead:

```bash
kubectl -n tensor-fusion-sys create secret generic my-agent \
  --from-literal=enroll-token='<token>'

helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys \
  --set agent.agentId='<org>:<env>' \
  --set agent.existingSecret.name=my-agent \
  --set agent.existingSecret.key=enroll-token
```

`agent.existingSecret.agentIdKey` is optional: when it is set, the agent id is also read from that Secret and `agent.agentId` can be left empty. When the chart-managed Secret changes, the controller Pod restarts automatically along with the `checksum/agent-secret` annotation; after rotating a self-managed Secret the controller Pod must be restarted manually. The Alertmanager configuration is likewise now stored in a Secret; with GreptimeDB Cloud, the database password exists only in the Secret `<release>-greptimedb-secret`, and the vector configuration references it through the environment variable `GREPTIME_PASSWORD`.

**The vector sidecar is enabled by default.** vector in the hypervisor Pod writes `tf_gpu_usage`, `tf_worker_usage`, and logs to GreptimeDB; on each GPU node it requests 50m CPU / 64Mi memory with limits of 1 CPU / 512Mi, and the hypervisor's metrics file occupies at most about 1.1 GB of ephemeral storage. Whether a GPU is available depends only on the hypervisor container; a vector failure does not take the GPU out of scheduling. To disable it:

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys --reuse-values \
  --set cluster.enableVector=false
```

After it is disabled, these two usage tables no longer receive data, and the alert rules and the autoscaler that depend on them stop working as well. Enabling it, disabling it, or upgrading to this version all change the hypervisor Pod template, and each hypervisor Pod is recreated once; workload Pods are not restarted.

**Upgrade notes.** hypervisor Pods that were already stuck in ContainerCreating before the upgrade because the vector ConfigMap was missing are not replaced automatically (after the volume was made optional it is not included in the template hash, to avoid restarting every hypervisor), and must be deleted manually once. The upgrade order is the hypervisor first, then the limiter library inside the workload Pods; during the window in which the two versions differ, an old-version limiter Pod whose single launch cost exceeds the token bucket is not constrained by its limit until the Pod restarts. Known limitations: a single launch that lasts tens of seconds runs below its quota under the new limiter (it does not exceed the limit); the autoscaler does not scale compute down for a worker that has never reported compute usage.

**Compute burst window.** Under soft isolation, an idle workload Pod can accumulate a stretch of tokens measured at its own quota rate, spend them at full speed when it resumes, and then return to its quota. How much it accumulates is determined by the burst window W (seconds), which defaults to 0.1 s and ranges from 0 to 10 s. At the pool level it is set in `schedulingConfigTemplate.elasticRateLimitParameters.burstWindow` (a string, for example `"0.5"`; setting this one item is enough and the other parameters can be left empty; the hypervisor reads the value at startup, so hypervisor Pods that are already running are not recreated automatically after a change and must be deleted for it to take effect); for a single workload use the Pod annotation `tensor-fusion.ai/compute-burst-window` (Go duration, for example `500ms` or `2s`), which can be changed directly on a running Pod without a restart; values that exceed 10 s, are negative, or are malformed are rejected when the Pod is created. The long-term compute quota is not affected by W: once its bucket is full, a Pod with quota s runs at full speed on an idle device for at most about `s × W / (1 − s)` seconds (with W = 2 s: about 0.67 s at a 25% quota, about 2 s at a 50% quota), and other Pods on the same GPU are affected only during that time.

**Time column of the metric tables.** The time index column of the `tf_*` tables is `greptime_timestamp` (produced when the table is created automatically on a line protocol write), not `ts`. The alert rules shipped with the chart use only `{{ .Conditions }}` and need no change; alert rules you wrote yourself that use `ts` directly must switch to the template variable `{{ .TimeColumn }}` (or `{{ timeColumn "table name" }}`); the operator detects the actual column name per table, so tables whose column is named `ts` on old clusters keep working.

## 3. Verify the control plane

After installation, check the system components:

```bash
kubectl get pods -n tensor-fusion-sys -o wide
kubectl get pods -n greptimedb -o wide
kubectl get crd | grep tensor-fusion.ai
kubectl get mutatingwebhookconfiguration | grep tensor-fusion
```

Check the controller logs:

```bash
kubectl logs -n tensor-fusion-sys deploy/tensor-fusion-sys-controller -c controller --tail=100
```

Normally the controller Pod contains at least the `controller` and `vector` containers; if `agent.agentId` is configured in the values, there is also `cluster-agent`.

In an environment without GPUs, also confirm that no Hypervisor was created by mistake:

```bash
kubectl get pods -n tensor-fusion-sys -o json \
  | jq '[.items[] | select(any(.spec.containers[]?; .name == "tensor-fusion-hypervisor"))] | length'
```

When no node matches the GPU selector, the output must be `0`. The default TFC/GPUPool may stay in
`Updating` because there is no GPUNode; this does not affect acceptance of a controller-only installation.

## 4. Create the TensorFusionCluster and GPUPool

After the control plane is installed, a TensorFusionCluster or GPUPool must be created before TensorFusion can discover and manage GPU resources.

> Since v1.7.8, the default Helm installation (`cluster.enabled=true`) automatically creates an NVIDIA
> `TensorFusionCluster` (a single NVIDIA pool). **NVIDIA-only clusters work out of the box with no manual creation**;
> for other vendors or multi-GPU-type scenarios see 4.1 / 4.2 below.

`config/samples/` provides working samples for each vendor (ProviderConfig + TensorFusionCluster);
their `componentConfig` is already a complete configuration matching the current version and can be applied directly.

Tips:

- When a pool is created through a TensorFusionCluster, the actual GPUPool name is usually `<cluster-name>-<pool-name>`.
- A workload Pod that does not explicitly set `tensor-fusion.ai/gpupool` lands in the pool with `isDefault: true`.

**Node partitioning/isolation capability** is configured in one place through the GPUPool's `nodeManagerConfig`. Nodes that match no rule
use `defaultIsolationMode`; rules are matched in order and matching stops at the first hit:

```yaml
nodeManagerConfig:
  defaultIsolationMode: soft
  isolationModeRules:
    - mode: partitioned
      selector:
        matchExpressions:
          - key: node.kubernetes.io/instance-type
            operator: In
            values: [p4d.24xlarge, p5.48xlarge]
    - mode: hard
      selector:
        matchLabels:
          tensor-fusion.ai/gpu-model: H100
```

Changing the default mode or the selector rules goes through the GPUPool's existing Hypervisor rolling update process.
With `autoUpdateHypervisor=false` the configuration is only marked as out of sync; when it is enabled, updates run in batches according to `batchPercentage` and
`batchInterval`, and only nodes whose effective mode changed are recreated.

soft, hard, and partitioned are mutually exclusive node capabilities. Since v2.13.0 `shared` denotes a whole-GPU allocation policy,
not a fourth node capability that has to be planned separately: a shared workload can use a complete,
idle, unpartitioned GPU on a node of any of the modes above. The effective mode of a node is determined entirely by the GPUPool policy.

### 4.1 Install by vendor (single vendor)

| Vendor | Command |
| --- | --- |
| NVIDIA | Created by default; or `kubectl apply -f config/samples/provider-nvidia.yaml -f config/samples/tensorfusioncluster-nvidia.yaml` |
| Ascend | `kubectl apply -f config/samples/provider-ascend.yaml -f config/samples/tensorfusioncluster-ascend.yaml` |
| MooreThreads | `kubectl apply -f config/samples/provider-mthreads.yaml -f config/samples/tensorfusioncluster-mthreads.yaml` |
| PPU | `kubectl apply -f config/samples/provider-ppu.yaml -f config/samples/tensorfusioncluster-ppu.yaml` |

When installing the control plane on a non-NVIDIA cluster, it is recommended to turn off the default NVIDIA cluster to avoid an extra pool with no nodes:

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys --create-namespace \
  --set cluster.enabled=false --set providerConfigs.nvidia.enabled=false
# then apply the provider + tensorfusioncluster of the corresponding vendor
```

**Online installation (Helm repository, no need to clone the repository)**: replace `./charts/tensor-fusion` with
`tensor-fusion/tensor-fusion --version <chart version>`; the sample CRs are not packaged into the chart, so apply them from the repository's
raw URLs (or clone the repository first). Using Ascend as an example:

```bash
helm repo add tensor-fusion https://nexusgpu.github.io/tensor-fusion --force-update
helm repo update tensor-fusion
helm upgrade --install tensor-fusion-sys tensor-fusion/tensor-fusion --version 1.8.3 \
  -n tensor-fusion-sys --create-namespace \
  --set cluster.enabled=false \
  --set providerConfigs.nvidia.enabled=false \
  --set initialGpuNodeLabelSelector=huawei.com/npu.present=true
RAW=https://raw.githubusercontent.com/NexusGPU/tensor-fusion/main/config/samples
kubectl apply -f $RAW/provider-ascend.yaml -f $RAW/tensorfusioncluster-ascend.yaml
```

To switch vendors, change only `initialGpuNodeLabelSelector` + the two file names:

| Vendor | selector | Files |
| --- | --- | --- |
| Ascend | `huawei.com/npu.present=true` | provider-ascend / tensorfusioncluster-ascend |
| MooreThreads | `mthreads.com/gpu.present=true` | provider-mthreads / tensorfusioncluster-mthreads |
| PPU | `aliyun.com/ppu.present=true` | provider-ppu / tensorfusioncluster-ppu |

### 4.2 Multiple vendors (several GPU types in one cluster)

One Kubernetes cluster has only one `TensorFusionCluster`; with multiple vendors, put multiple pools under `spec.gpuPools`
(only one with `isDefault: true`). A ready-made example is `config/samples/tensorfusioncluster-multi-vendor.yaml`:

```bash
helm upgrade --install tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys --create-namespace \
  --set cluster.enabled=false \
  --set initialGpuNodeLabelSelector=tensor-fusion.ai/watch-node=true
kubectl apply -f config/samples/provider-nvidia.yaml \
  -f config/samples/provider-ascend.yaml \
  -f config/samples/provider-mthreads.yaml \
  -f config/samples/tensorfusioncluster-multi-vendor.yaml
```

**Node label (`initialGpuNodeLabelSelector`)** determines which nodes the operator watches:

- **Single vendor**: the vendor's present label is enough (default `nvidia.com/gpu.present=true`), and
  **no extra labelling is needed**: the label is applied automatically by the vendor's device-plugin / feature-discovery, and TensorFusion is not responsible for applying it.
- **Multiple vendors**: each vendor has a different present label (`nvidia.com/gpu.present` / `huawei.com/npu.present` /
  `mthreads.com/gpu.present` / `aliyun.com/ppu`), so a single label cannot select them all. Use the vendor-neutral
  `tensor-fusion.ai/watch-node=true` instead and **apply this label to every GPU node manually** (it is not generated automatically):

  ```bash
  kubectl label node <gpu-node> tensor-fusion.ai/watch-node=true
  ```

- Or leave `initialGpuNodeLabelSelector` empty (`--set initialGpuNodeLabelSelector=""`), and
  the operator watches all nodes (no labelling needed, but with some performance overhead in large clusters).

After applying the configuration, check the resources:

```bash
kubectl get tensorfusioncluster
kubectl get gpupool
kubectl get gpunode,gpu -o wide
kubectl describe gpupool <gpupool-name>
```

## 5. Deploy a workload Pod for verification

A workload Pod must meet the following conditions before the webhook injects the TensorFusion client:

- The Pod has the label `tensor-fusion.ai/enabled: "true"`.
- A fractional request sets at least one resource annotation: `tensor-fusion.ai/tflops-request`, `tensor-fusion.ai/compute-percent-request`, or `tensor-fusion.ai/vram-request`. A standard shared whole-GPU request sets `isolation: "shared"`, `dedicated-gpu: "true"`, and `gpu-model`, and the user does not need to fill in TFLOPS/VRAM; `gpu-count` defaults to 1 when not set, and setting it explicitly is recommended.
- A multi-container Pod must set `tensor-fusion.ai/inject-container`.

First query the exact models the operator has recognised in the target pool; `<gpu-model>` below must be replaced with
the full value from `GPU.status.gpuModel`:

```bash
kubectl get gpu -l tensor-fusion.ai/gpupool=<gpupool-name> \
  -o custom-columns='NAME:.metadata.name,MODEL:.status.gpuModel,CAPACITY:.status.capacity'
```

Example Deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensor-fusion-smoke
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tensor-fusion-smoke
  template:
    metadata:
      labels:
        app: tensor-fusion-smoke
        tensor-fusion.ai/enabled: "true"
      annotations:
        tensor-fusion.ai/gpupool: "<gpupool-name>"
        tensor-fusion.ai/inject-container: "pytorch"
        tensor-fusion.ai/dedicated-gpu: "true"
        tensor-fusion.ai/isolation: "shared"
        tensor-fusion.ai/is-local-gpu: "false"
        tensor-fusion.ai/gpu-count: "1"
        tensor-fusion.ai/gpu-model: "<gpu-model>"
    spec:
      containers:
        - name: pytorch
          image: pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
          command: ["sh", "-c", "sleep infinity"]
```

Deploy and check:

```bash
kubectl apply -f smoke.yaml
kubectl get pod -l app=tensor-fusion-smoke -o wide
kubectl describe pod -l app=tensor-fusion-smoke
kubectl get tensorfusionworkload,tensorfusionconnection -A
kubectl get pods -A -l tensor-fusion.ai/component=worker -o wide
```

Enter the workload container and actually call CUDA/NVML; a Pod that merely reaches Running/Ready does not mean the GPU path has been verified:

```bash
POD=$(kubectl get pod -l app=tensor-fusion-smoke -o jsonpath='{.items[0].metadata.name}')
WORKER=$(kubectl get pod \
  -l tensor-fusion.ai/component=worker,tensor-fusion.ai/workload=tensor-fusion-smoke \
  -o jsonpath='{.items[0].metadata.name}')
kubectl get pod "$WORKER" -o jsonpath='{.metadata.annotations.tensor-fusion\.ai/gpu-ids}{"\n"}'
kubectl exec -it "$POD" -c pytorch -- nvidia-smi
kubectl exec "$POD" -c pytorch -- python3 -c \
  'import ctypes; cuda=ctypes.CDLL("libcuda.so.1"); count=ctypes.c_int(); print("cuInit=", cuda.cuInit(0)); print("cuDeviceGetCount=", cuda.cuDeviceGetCount(ctypes.byref(count)), "count=", count.value)'
kubectl exec "$POD" -c pytorch -- python3 -c \
  'import torch; print("count=", torch.cuda.device_count()); x=torch.ones(1024, device="cuda"); print("sum=", x.sum().item()); print("uuid=", torch.cuda.get_device_properties(0).uuid)'
kubectl exec "$POD" -c pytorch -- nvidia-smi \
  --query-gpu=uuid,name,memory.total --format=csv,noheader
```

The shared whole-GPU case should show `cuInit=0` and a CUDA device count equal to `gpu-count`. In the remote
example above the worker Pod holds `tensor-fusion.ai/gpu-ids`; in a local example the workload Pod holds it. CUDA/NVML
should return the same GPU UUID (ignoring the `gpu-` prefix of the GPU CR name and the `GPU-` format prefix of NVML). After verification,
delete the Deployment and confirm that the TensorFusionWorkload/Connection is cleaned up and the GPU available value returns to capacity:

```bash
kubectl delete deployment tensor-fusion-smoke
kubectl get tensorfusionworkload,tensorfusionconnection -A
kubectl get gpu -l tensor-fusion.ai/gpupool=<gpupool-name> \
  -o custom-columns='NAME:.metadata.name,CAPACITY:.status.capacity,AVAILABLE:.status.available'
```

If `http_proxy` or `https_proxy` is configured in the container, use the following to probe a local service:

```bash
curl --noproxy "*" http://127.0.0.1:8000/v1/models
```

## 6. Common workload annotations

| Annotation | Description |
| --- | --- |
| `tensor-fusion.ai/gpupool` | Selects the GPUPool. When not set, the default pool is relied on. |
| `tensor-fusion.ai/is-local-gpu` | `true` means the workload Pod is scheduled to a GPU node and uses the GPU locally; `false` means remote mode. |
| `tensor-fusion.ai/isolation` | `shared`, `soft`, `hard`, `partitioned`. |
| `tensor-fusion.ai/dedicated-gpu` | `true` means the whole-GPU capacity is filled in according to `gpu-model`; by itself it does not change isolation to shared. A standard shared request must also set `isolation=shared`. |
| `tensor-fusion.ai/gpu-count` | Number of GPUs requested. |
| `tensor-fusion.ai/gpu-model` | Requested GPU model; required when `dedicated-gpu=true` is used. Use the exact value of `GPU.status.gpuModel` and make sure the operator has loaded the capacity of that model. |
| `tensor-fusion.ai/gpu-indices` | Selects GPU indices; this is a hard filter. When set, `gpu-count` is computed from the number of indices. |
| `tensor-fusion.ai/tflops-request` | TFLOPs request used for scheduling; recommended as the first choice. |
| `tensor-fusion.ai/compute-percent-request` | Requests compute as a percentage. Mutually exclusive with the TFLOPs request. |
| `tensor-fusion.ai/vram-request` | GPU memory request used for scheduling. |
| `tensor-fusion.ai/inject-container` | Selects the containers to inject in a multi-container Pod; separate multiple containers with commas. |

`shared` is a whole-GPU allocation policy for complete idle GPUs; it mounts neither the soft limiter nor the hard preload. The user does not need to fill in
TFLOPS/VRAM request/limit; the webhook fills in the whole-GPU capacity according to `gpu-model`, and after allocation the allocator reduces
both the available TFLOPS and VRAM of the selected GPU CR to 0. shared does not prefer nodes that carry a shared label;
it still uses the existing GPU/node binpack scoring.

## 7. Troubleshooting

### 7.1 Pod Pending

First look at the Kubernetes scheduling events:

```bash
kubectl describe pod <pod-name>
kubectl get nodes -o wide
kubectl get nodes --show-labels
```

Common causes:

- No toleration is configured for a node taint.
- The Pod's nodeSelector or nodeAffinity does not match the GPU nodes.
- The GPUPool's nodeSelector selects no nodes.
- The GPU specified by `tensor-fusion.ai/gpu-indices` does not exist or is already occupied by another workload.
- The GPU resource request exceeds the resources available in the pool.

Then check the TensorFusion resources:

```bash
kubectl get gpupool,gpunode,gpu -o wide
kubectl describe gpupool <gpupool-name>
kubectl describe gpu <gpu-name>
kubectl logs -n tensor-fusion-sys deploy/tensor-fusion-sys-controller -c controller --tail=200
```

### 7.2 Wrong CUDA/NVML driver library paths

If `nvidia-smi` reports `NVML Function Not Found`, or the worker log shows something like
`Assertion failed, cuInit_fn != nullptr`, first confirm the driver library paths that actually exist on the host and in the failing container.

On the host, check that the NVIDIA driver works and which directory the driver libraries are installed in:

```bash
nvidia-smi
ldconfig -p | grep -E 'libcuda.so.1|libnvidia-ml.so.1'
ls -l \
  /usr/lib64/libcuda.so.1 \
  /usr/lib64/libnvidia-ml.so.1 \
  /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 2>/dev/null
```

`/usr/lib64` is common on the RHEL/CentOS/Rocky/openEuler/Kylin family, and `/usr/lib/x86_64-linux-gnu`
is common on the Debian/Ubuntu family. Which paths are finally visible in the Pod also depends on what the NVIDIA container runtime mounts and on the distribution of the container image itself.

Then, inside the failing Pod container, confirm the actual linkage and the available paths:

```bash
ldd "$(which nvidia-smi)" | grep -E 'tensor-fusion|nvidia-ml|cuda'
cat /etc/ld.so.preload 2>/dev/null || true
ldconfig -p 2>/dev/null | grep -E 'libcuda.so.1|libnvidia-ml.so.1' || true
ls -l \
  /usr/lib64/libcuda.so.1 \
  /usr/lib64/libnvidia-ml.so.1 \
  /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
  /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 \
  /usr/local/nvidia/lib64/libcuda.so.1 \
  /usr/local/nvidia/lib64/libnvidia-ml.so.1 2>/dev/null
```

If a full search is needed:

```bash
find /usr /lib /lib64 /usr/local \( -name 'libcuda.so*' -o -name 'libnvidia-ml.so*' \) 2>/dev/null
```

If the TensorFusion worker needs the driver library paths specified explicitly, use the stable `.so.1` symlinks that really exist in the failing container:

```yaml
env:
  - name: TF_CUDA_LIB_PATH
    value: <container-visible-libcuda.so.1-path>
  - name: TF_NVML_LIB_PATH
    value: <container-visible-libnvidia-ml.so.1-path>
```

Do not pin a versioned path such as `/usr/lib64/libnvidia-ml.so.580.126.09` or
`/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.216.01`. After a driver upgrade the versioned file may change; only the `.so.1` symlink follows the current driver.

If the log says `cuInit_fn != nullptr`, focus on whether the `libcuda.so.1` that `TF_CUDA_LIB_PATH` points to exists in the failing container and contains the `cuInit` symbol:

```bash
readelf -Ws <container-visible-libcuda.so.1-path> 2>/dev/null | grep ' cuInit'
```

If the host has the driver libraries but the Pod does not, first investigate the NVIDIA container runtime, the NVIDIA device plugin, and whether the Pod was really allocated a GPU device. If the Pod has the correct libraries but still fails, confirm that these environment variables were injected into the worker container that actually crashes, not only into the workload container.

### 7.3 Pod is Evicted

If the events show `The node was low on resource: memory`, this is an eviction caused by node memory pressure, not a GPU scheduling failure.

Check:

```bash
kubectl describe pod <pod-name>
kubectl top pod <pod-name> --containers
kubectl top node <node-name>
```

What to do:

- Set a reasonable memory request and limit for the workload container.
- In remote mode the client side may have an extra memory peak while a model loads; reserve memory according to actual load test results.
- Avoid starting large models on nodes with little memory.

### 7.4 Local curl gets no response after vLLM starts

First confirm whether the port is listening:

```bash
ps -ef | grep -E 'vllm|EngineCore|APIServer' | grep -v grep
ss -lntp | grep 8000
tail -n 100 /tmp/vllm.log
```

If `curl` is affected by proxy environment variables, use:

```bash
curl --noproxy "*" -v --max-time 10 http://127.0.0.1:8000/v1/models
```

If the log shows CUDA OOM, lower `--gpu-memory-utilization`, reduce `--max-model-len`, or free other processes on the same GPU.

### 7.5 `tf_gpu_usage` / `tf_worker_usage` have no data

The Hypervisor only writes metrics to `/logs/metrics.log`; the `vector` sidecar in the same Pod uploads them to GreptimeDB. The Hypervisor cannot see the upload result: when the database rejects a write, there is no error in the Hypervisor log, and you have to look at the log of the `vector` container:

```bash
kubectl -n tensor-fusion-sys logs <hypervisor-pod> -c vector --since=10m \
  | grep -E 'Not retriable|Events dropped'
```

When writes are rejected, every upload produces these two lines (`component_id` is the sink name; the Controller's sidecar corresponds to `sink_greptimedb_operator_metrics`):

```text
ERROR sink{component_kind="sink" component_id=sink_greptimedb_hypervisor_metrics component_type=http}:request{request_id=12}: vector::sinks::util::retries: Not retriable; dropping the request. reason="Default retry strategy: Bad Request"
ERROR sink{component_kind="sink" component_id=sink_greptimedb_hypervisor_metrics component_type=http}:request{request_id=12}: vector_common::internal_event::component_events_dropped: Events dropped intentional=false count=7 reason="Service call failed. No retries or retries exhausted."
```

Vector does not print the database's response. Take one metric line and submit it by hand to see the reason for the rejection:

```bash
kubectl -n tensor-fusion-sys exec <hypervisor-pod> -c vector -- tail -n 1 /logs/metrics.log > line.txt
curl -sS -X POST --data-binary @line.txt \
  'http://<greptimedb-host>:4000/v1/influxdb/api/v2/write?db=public&precision=ms'
```

A response such as `column memory_bytes expect type UInt64(UInt64Type), given: INT64(4)` means that the type of some field in this line differs from the existing column in the table: GreptimeDB creates a column with the type of the first write and afterwards rejects every line with a different type. `SHOW CREATE TABLE tf_gpu_usage;` shows the column types. The types written by the Hypervisor in this version match the tables created by older Hypervisors (`memory_bytes` is an unsigned integer, `power_usage` is an integer); if the table was created with other types and the historical data in it can be discarded, after `DROP TABLE` the next write recreates the table. The Hypervisor's sidecar submits each metric line separately, so a rejected line does not affect the other lines.

### 7.6 Hypervisor logs

When the `vector` sidecar is enabled (`enableVector: true`), the Hypervisor also writes its own logs to `/logs/pods/<pod>/hypervisor.log` (at most 3 files of 50 MB each, kept for 7 days), and the sidecar writes them to `tf_system_log`:

```sql
SELECT greptime_timestamp, pod, message FROM tf_system_log
WHERE component = 'hypervisor' AND container = 'hypervisor'
ORDER BY greptime_timestamp DESC LIMIT 50;
```

The default log level is Info. The rate limiting detail printed as one line per control cycle (`ERL [...] util=... rate=...`) is output only at log level 4, which can be turned on in two ways.

**Turn it on temporarily, without recreating the Pod.** Send `SIGUSR1` to the Hypervisor process: the log level rises to 4 and returns automatically to the startup level after 10 minutes; sending it again during that time restarts the 10 minutes. When the startup level is already 4 or higher, the signal changes nothing and only outputs one log line. Raising and restoring each produce one log line (`Log verbosity raised from 0 to 4 for 10m0s`, `Log verbosity is back to 0`).

The Hypervisor Pod uses the host's PID namespace (`hostPID: true`), so the process ID is not 1 and has to be looked up first. The Hypervisor is the first container of the Pod, so `kubectl exec` enters it by default; the image is based on `ubuntu:24.04` and includes a shell and `pidof`:

```bash
kubectl -n tensor-fusion-sys exec <hypervisor-pod> -- sh -c 'kill -USR1 $(pidof hypervisor)'
```

You can also run `pkill -USR1 -x hypervisor` as root on the node. Hypervisors before this version do not handle this signal: on receiving it the process exits and the container restarts, so confirm the image version before sending it.

**Turn it on permanently.** Add the startup argument `-v=4` through the Pool's Hypervisor Pod template (`spec.gpuPools[].specTemplate.componentConfig.hypervisor.podTemplate` of the `TensorFusionCluster`):

```yaml
podTemplate:
  template:
    spec:
      containers:
        - name: tensor-fusion-hypervisor
          args: ["-v=4"]
```

Changing the template recreates the Hypervisor Pods of that Pool one by one, and they are recreated once more when the argument is removed after troubleshooting. `-log-output-path=none` turns off the log file.

### 7.7 Known issues

- **vLLM 0.28.0 fails to start with hard isolation + remote-vGPU `v2.26.0`.** With the client/worker images `tensorfusion/tensor-fusion-client:v2.26.0` and `tensorfusion/tensor-fusion-worker:v2.26.0`, vLLM 0.28.0 under hard isolation reports `call api=cuMemcpy failed … named symbol not found`. The defect is in the remote-vGPU worker (a separate component), not in this operator.
- **`shared` isolation is exclusive use of a whole GPU.** Two `shared` Pods on the same GPU run one after the other, not at the same time.
- **The compute limit constrains the share of device time, not the fraction of solo throughput.** When several Pods share a GPU, each switch between two contexts costs the GPU about 65 µs (measured on an RTX 3090); the driver counts this time as utilization too, and mostly charges it to the Pod that holds the engine time longest. As a result, when a Pod with a larger quota reaches its own utilization target, the kernel time it actually gets is about 20% less: sharing a GPU with a neighbour at a 30% quota, a 60% quota measured 0.485 of solo throughput, 50% measured 0.385, and 40% measured 0.309. When two unlimited short-kernel workloads share a GPU, their combined throughput is about 0.83 of one running alone (0.93 to 0.95 for long-kernel workloads; ONNX Runtime CNN / encoder inference and copy-dominated workloads lose nothing).
- **When several limited Pods share a GPU, a small-quota Pod with a short-kernel workload may stay below its quota for a long time.** The driver samples utilization about every 200 ms and charges the whole sample to the process occupying the device at that moment; combined with the token refill every 10 ms, the process utilization of the small-quota Pod reads 0 for more than ten seconds in a row. With such readings the default control logic may hold its rate down for about two minutes (measured on an RTX 3090: a 0.28 ms matrix multiplication workload, two Pods at 30% / 60%, the 30% side gets only 0.16 of solo throughput). Long-kernel workloads and inference with 3B to 7B LLMs are not affected. You can set the environment variable `TF_ERL_SHARED_CALIBRATION=confirmed` in the pool's hypervisor Pod template to enable a different handling (process utilization is used only for slow calibration, and changes are confirmed by the device's own utilization): the scenario above becomes 0.27, but when workloads with very different token costs are co-located, one side may exceed its quota by 10% to 30%, and may exceed it by more in the first 10 seconds when a workload suddenly becomes heavier. This switch is off by default and is an experimental option.
- **Helm hook Jobs on mixed-architecture clusters.** The two hook Jobs that generate and inject the webhook certificate are by default restricted only to `kubernetes.io/os: linux` and reuse the controller's `affinity` and `tolerations`; they are not restricted to a CPU architecture. When a hook Pod lands on a node that cannot pull the image, `helm install` / `helm upgrade` waits until it times out. Use `controller.admissionWebhooks.patch.nodeSelector` (for example `kubernetes.io/arch: amd64`), `affinity`, and `tolerations` to choose the nodes for the hook Pods.

## 8. Upgrade, rollback, uninstall

Before upgrading, back up the current Helm values and the key CRs:

```bash
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "backup-${TS}"
helm -n tensor-fusion-sys get values tensor-fusion-sys -o yaml > "backup-${TS}/values.yaml"
kubectl get tensorfusioncluster,gpupool,gpunode,gpu,workloadprofile,tensorfusionworkload,tensorfusionconnection -A -o yaml \
  > "backup-${TS}/tf-state.yaml"
```

A regular Chart upgrade within the same architecture:

```bash
helm upgrade tensor-fusion-sys ./charts/tensor-fusion \
  -n tensor-fusion-sys \
  -f <your-values.yaml>
```

This command updates the database, monitoring, and other components in the same Helm release and **must not be used for the v1→v2 architecture upgrade**.
v1→v2 must follow the precise upgrade procedure: extract and apply only the CRDs, RBAC, and
scheduler ConfigMap from the pinned-version Helm package, switch the operator/Hypervisor images and
the GPUPool isolation policy together inside an `operator=0` window, and do not run a whole-package `helm upgrade`.

For the detailed steps and the rollback plan see:

- `docs/v1-to-v2-upgrade-best-practice.md`

### 8.1 Upgrade notes / breaking changes

Upgrade the chart together with the operator and hypervisor images; do not pair a new chart with old images: an old operator does not serve the webhook path for ephemeral container validation (with the default `failurePolicy: Ignore` the validation does not take effect, and when it is set to `Fail` every `kubectl debug` in the cluster is rejected), and it does not inject `GREPTIME_PASSWORD` into the vector container of the hypervisor Pod (with `greptime.isCloud=true` that container cannot start and usage metrics are interrupted).

- **Metric column types.** The field types the hypervisor writes to `tf_gpu_usage` and `tf_worker_usage` are now fixed: `memory_bytes` is an unsigned integer, `power_usage`, `nvlink_rx`, and `nvlink_tx` are signed integers, and the rest are floats. On clusters where existing columns in a table have different types (for example, the hypervisor was once run with custom arguments, or the table was created by another writer), after the upgrade every row for that table is rejected by the database with HTTP 400, and nothing shows in the hypervisor log. Confirm with the method in 7.5; the fix is to export the historical data you need to keep and then `DROP TABLE`, after which the table is recreated with the new types on the next write. The operator does not drop tables automatically.
- **NodeOverlay matches only NVIDIA instance types on AWS by default.** The default value of `karpenterNodeOverlay.requirements` is `karpenter.k8s.aws/instance-gpu-count > 0` and `karpenter.k8s.aws/instance-gpu-manufacturer in [nvidia]`, and it no longer applies to all NodePools. With Karpenter on other clouds or GPUs from other vendors, the default overlay matches no node after the upgrade, and you need to set `karpenterNodeOverlay.requirements` yourself.
- **Pools with the `reserved` capacity type.** The 13 AWS instance types that have no reserved instance price in the built-in price table are no longer selected by `reserved` Pools; if you need these instance types, switch to `on-demand` or add the prices in the ProviderConfig.
- **Empty TensorFusionWorkloads without an owner are reclaimed.** A workload that was created automatically by the Pod webhook, has no ownerReference, and has not been referenced by any Pod for 5 minutes is deleted at the first reconcile after the upgrade, and its `status.recommendation` is lost with it. Workloads created by users themselves, and workloads with an owner (Deployment, StatefulSet, Job, Pod, and so on), are not affected.
- **hypervisor Pods are recreated once.** In Pools with the vector sidecar enabled, each hypervisor Pod is recreated once after the upgrade because of the template change; workload Pods are not restarted.
- **Go library interfaces (projects that import `pkg/hypervisor` or `pkg/constants`).** `(*APIClient).ResolveDeviceFp16TFlops` has been removed; `constants.TFDataPathWorkerExpr` now depends on the two environment variables `TF_SHM_POD_NAMESPACE` and `TF_SHM_POD_NAME` inside the container, and callers that use it must set both variables as well.

The uninstall script supports three installation methods: Helm, `helm template | kubectl apply`, and `make deploy` / Kustomize.
It deletes all TensorFusion CRs in the cluster, the control plane, webhooks, CRDs,
TensorFusion Node labels, NodeOverlay, PVCs, and the namespaces used by Helm, Kustomize, and GreptimeDB.
Run it after confirming that the data and workloads in them are no longer needed:

```bash
NAMESPACE=tensor-fusion-sys HELM_RELEASE=tensor-fusion-sys ./scripts/uninstall.sh
```

Before running it you can view the full parameters and the scope of deletion:

```bash
./scripts/uninstall.sh --help
```

Defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NAMESPACE` | `tensor-fusion-sys` | Control plane namespace for Helm / manifest installs |
| `KUSTOMIZE_NAMESPACE` | `tensor-fusion` | Namespace used by `make deploy` |
| `HELM_RELEASE` | `tensor-fusion-sys` | Helm release name |
| `RESOURCE_PREFIX` | Same as `HELM_RELEASE` | Prefix of manifest resource names and PVCs |
| `WAIT_TIMEOUT_SECONDS` | `600` | Timeout in seconds for waiting on CR finalizers and resource deletion |

For a non-default release name or `helm template` resource prefix, set these in addition:

```bash
NAMESPACE=<namespace> \
HELM_RELEASE=<release-name> \
RESOURCE_PREFIX=<rendered-resource-prefix> \
./scripts/uninstall.sh
```

The script waits for resources to be deleted in finalizer dependency order and stops if any step fails, without going on to delete
the controller or CRDs. It cleans up TensorFusion Node labels, taints, NodeOverlay,
the schedulable `tensor-fusion.ai/index*` capacity, and TensorFusion PVCs/PVs. After the device plugin
disconnects, the kubelet takes a grace period of about 5 minutes to clean up its internal endpoints and checkpoint;
during this time the Node status may keep a non-zero capacity, but allocatable has already become `0`. After the grace period ends,
the kubelet intentionally keeps the index keys with a value of `0`. The script then removes these keys from the Node status
and waits one status sync window to confirm that the kubelet no longer writes them back.
If they are still not deleted after `WAIT_TIMEOUT_SECONDS`, the script reports an error. The script does not delete
vendor discovery labels such as `nvidia.com/gpu.present` and `huawei.com/npu.present`. Before uninstalling, you must
first back up the workloads, PVC data, and TensorFusion CR state.

`scripts/uninstall.sh` can be copied to the target machine and run as a single file; it does not require downloading the full repository as well.
The names of the cluster-scoped RBAC, webhook, CRD, and other resources produced by the current default Helm and Kustomize installations are built into
the script. If `charts/tensor-fusion` or `config/default` exists next to the script, it additionally
runs one compatibility cleanup based on the local manifests; these two directories are not required for a single-file uninstall.
