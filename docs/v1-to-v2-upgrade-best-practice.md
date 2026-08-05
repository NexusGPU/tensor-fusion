# TensorFusion v1 → v2(main) 升级最佳实践

> 适用范围：从 `v1` 分支线升级到 `main` 分支线（v2 架构）。
> **前提**：升级所用的 CRD 必须来自 CRD v1 兼容改动（`feat/crd-v1-compat`）合并之后的 main——该改动把 v1 独有字段（`nvLink`、`model`、`nodeDiscovery`、`remoteModeImage`/`embeddedModeImage` 等）补回了 v2 CRD。**不要使用合并前 main 上的 CRD**，否则 v1 的 spec 数据会被 apiserver 裁剪（pruning），回退时无法恢复。

---

## 核心结论

- CRD 的 group/version 不变，仍为 `tensor-fusion.ai/v1`，单一 served/storage 版本，无 Conversion Webhook，无 K8s API 版本迁移。
- **v2 CRD 是 v1 schema 的严格超集**（逐字段比对验证）：无字段删除，v1 写入的数据在 v2 CRD 下完整保留。
- 回退时 **CRD 保持 v2 不动**，但不能只换 operator 镜像：v1/v2 的 scheduler ConfigMap 和 Hypervisor 镜像必须与各自 operator 配套切换。
- 升级顺序：**下载并固定 Helm Chart → 确认 `autoUpdateHypervisor=true` → apply CRD → 创建 NVIDIA ProviderConfig → 同步 RBAC → 切换 ConfigMap → 切换 operator 并等待 Ready → 切换 Hypervisor 镜像**。不需要将 operator 缩容到 0，也不需要手工删除 Hypervisor Pod。本文使用正式 Helm 包作为资源来源，但不执行 `helm upgrade`，避免更新同一 release 中的数据库、监控等其他组件。

---

## 获取正式 Helm 升级包

升级使用包含节点隔离策略、自动更新恢复修复和一键安装修复的 Chart `1.8.3`。必须等该版本
发布到公开 Helm 仓库后再执行；如果仓库中查不到 `1.8.3`，停止升级，不要退回使用缺少
最新 Helm 修复的旧包，也不要使用未固定版本的最新包：

```bash
chart_version=1.8.3
upgrade_dir=${TMPDIR:-/tmp}/tensor-fusion-upgrade-${chart_version}

helm repo add tensor-fusion https://nexusgpu.github.io/tensor-fusion --force-update
helm repo update tensor-fusion
helm search repo tensor-fusion/tensor-fusion --versions | head

rm -rf "${upgrade_dir}"
mkdir -p "${upgrade_dir}"
helm pull tensor-fusion/tensor-fusion \
  --version "${chart_version}" \
  --untar \
  --untardir "${upgrade_dir}"

chart=${upgrade_dir}/tensor-fusion
helm show chart "${chart}" | grep -E '^(version|appVersion):'
```

版本检查必须输出：

```text
appVersion: 2.15.0
version: 1.8.3
```

后续 CRD、ProviderConfig、RBAC 和 ConfigMap 都从 `${chart}` 读取。不要执行 `helm upgrade` 或 `helm upgrade --install`；本文只使用 `helm template` 和 `kubectl apply` 精确更新 operator 所需资源，不会更新数据库或同一 Helm release 中的其他组件。

由于没有执行 `helm upgrade`，升级后 `helm list` 仍会显示原 release 的旧 Chart/App
版本，这是预期行为。实际版本应通过 controller Deployment 与 Hypervisor Pod 的镜像核验，
不能只看 Helm release 元数据。

---

## 升级前预检

### 1. 确认 CRD 包含 v1 兼容字段（关键）

```bash
grep -l 'nvLink' "${chart}/crds/tensor-fusion.ai_gpus.yaml" \
  && grep -l 'remoteModeImage' "${chart}/crds/tensor-fusion.ai_gpupools.yaml" \
  && grep -l 'defaultIsolationMode' "${chart}/crds/tensor-fusion.ai_gpupools.yaml" \
  && grep -l 'isolationModeRules' "${chart}/crds/tensor-fusion.ai_gpupools.yaml" \
  && echo "OK: compat fields present"
```

输出非 OK 则说明 CRD 版本不对，停止升级。

### 2. 校验收紧项检查

v2 CRD 和控制器行为引入了以下兼容性检查点，CRD 校验跟随 CRD 生效、**与运行哪个镜像无关**（即镜像回滚也救不了），需提前确认存量数据不命中：

```bash
# nodeManagerConfig 变为必填（正常部署必有此字段，确认即可）
kubectl get gpupools -A -o json \
  | jq -r '.items[] | select(.spec.nodeManagerConfig == null)
    | "MISSING: \(.metadata.name)"'

# gpuCount 新增 minimum:1 / maximum:128（v1 webhook 默认填 1，正常不会命中）
kubectl get tensorfusionworkloads,workloadprofiles -A -o json \
  | jq -r '.items[] | select(.spec.gpuCount != null and (.spec.gpuCount < 1 or .spec.gpuCount > 128))
    | "BAD: \(.kind)/\(.metadata.namespace)/\(.metadata.name): gpuCount=\(.spec.gpuCount)"'

# Hypervisor 自动滚动必须显式启用。缺失 policy 或字段为 false 都不会 panic，
# 但镜像/ProviderConfig 变化只会记录新 hash，不会主动替换已有 Running Pod。
kubectl get gpupools -A -o json \
  | jq -r '.items[]
    | select(.spec.nodeManagerConfig.nodePoolRollingUpdatePolicy.autoUpdateHypervisor != true)
    | "AUTO-UPDATE-DISABLED: \(.metadata.name)"'
```

三条命令输出均为空才继续；有输出则先修正对应对象。对于 TFC 管理的 pool，必须修改 TFC 的 `spec.gpuPools[].specTemplate`，不要只改派生出的 GPUPool：

```yaml
nodeManagerConfig:
  nodePoolRollingUpdatePolicy:
    autoUpdateHypervisor: true
    batchInterval: 1s
    batchPercentage: 20
```

升级前显式启用可以避免新镜像和隔离策略停留在待同步状态。若运维期间主动关闭自动更新，配置
会保持待同步；重新启用后，状态机会自动创建 rollout campaign 并补做积压更新。

### 3. 提前规划节点隔离/切分模式（关键，暂不写入）

升级前必须先在 TensorFusionCluster（TFC）中规划每个 GPUPool 的节点策略：哪些节点使用
`soft`、哪些使用 `hard`，需要硬件切分的节点使用 `partitioned`。TFC 管理的 pool 以
`spec.gpuPools[].specTemplate.nodeManagerConfig` 为配置源，不要直接修改派生出的 GPUPool，
也不要依赖 Node 或 GPUNode 上的 `tensor-fusion.ai/isolationMode` label。

`defaultIsolationMode` 和 `isolationModeRules` 是 v2 CRD 新增字段。v1 CRD 不包含这两个
字段，因此升级前这里只做节点盘点和配置审核，**不要将下面的示例提前 apply 到现有 TFC**。
实际写入必须在步骤 1 apply v2 CRD 之后进行。

规划时遵循以下规则：

- 用 `defaultIsolationMode` 定义未命中任何 selector 的节点模式；没有特殊需求时建议为 `soft`。
- `isolationModeRules` 按声明顺序匹配，第一条命中即生效。selector 有重叠时必须确认优先级符合预期。
- selector 应使用升级后仍会稳定存在的 Kubernetes Node label。先检查每条 selector 实际命中的节点，避免拼写错误或 label 缺失使节点意外落到默认模式。
- 同时核对现有 workload 的 isolation 请求与目标节点能力一致；`soft`、`hard`、`partitioned` 是互斥的节点能力。

例如，先盘点用于分组的 label，再形成待写入的 TFC 配置：

```bash
mkdir -p tf-backup
kubectl get nodes -L node.kubernetes.io/instance-type,tensor-fusion.ai/gpu-model
kubectl get nodes -l 'node.kubernetes.io/instance-type in (p4d.24xlarge,p5.48xlarge)'
kubectl get nodes -l 'tensor-fusion.ai/gpu-model=H100'
kubectl get tensorfusioncluster -A -o yaml > tf-backup/tfc-before-v2.yaml
```

```yaml
spec:
  gpuPools:
    - name: nvidia
      specTemplate:
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

本阶段只确定每个 pool 的最终 `defaultIsolationMode` 和有序 `isolationModeRules`。
步骤 1 apply v2 CRD 后，步骤 5 再将这份已审核的配置与 Hypervisor 镜像一起写入 TFC；
不能直接照抄文中的示例值。

### 4. 全量备份（保险，不再是回退的依赖项）

```bash
release=tensor-fusion-sys  # 改成实际 Helm release 名称
ns=tensor-fusion-sys       # 改成实际 operator 所在 namespace
controller_deploy=$(kubectl -n ${ns} get deploy \
  -l tensor-fusion.ai/component=operator \
  -o jsonpath='{.items[0].metadata.name}')
mkdir -p tf-backup
helm -n ${ns} get values ${release} -o yaml > tf-backup/helm-values.yaml
for resource in $(kubectl api-resources --api-group=tensor-fusion.ai -o name); do
  kubectl get ${resource} -A -o yaml > tf-backup/${resource}.yaml
done
kubectl get crd -o name | grep 'tensor-fusion.ai' \
  | xargs kubectl get -o yaml > tf-backup/crds.yaml
kubectl -n ${ns} get deploy ${controller_deploy} -o yaml > tf-backup/controller-deploy.yaml
kubectl -n ${ns} get configmap ${release}-config -o json \
  | jq 'del(.metadata.resourceVersion, .metadata.uid, .metadata.creationTimestamp,
      .metadata.managedFields)' \
  > tf-backup/config-v1.json
kubectl -n ${ns} get pods -o json | jq -r '
  .items[]
  | select(.metadata.name | startswith("hypervisor-"))
  | [.metadata.name, .metadata.uid, .spec.nodeName,
     (.spec.containers[] | select(.name == "tensor-fusion-hypervisor") | .image)]
  | @tsv' > tf-backup/hypervisor-v1.tsv
```

v1 没有 ProviderConfig CRD；升级到 v2 时必须创建 NVIDIA ProviderConfig。先固定 Chart
`1.8.3` 配套的名称和镜像：

```bash
provider_name=nvidia-provider
middleware_image=tensorfusion/vgpu-provider-nvidia:1.3.9
remote_client_image=tensorfusion/tensor-fusion-client:v2.15.0
remote_worker_image=tensorfusion/tensor-fusion-worker:v2.15.0

printf 'middleware=%s\nremoteClient=%s\nremoteWorker=%s\n' \
  "${middleware_image}" "${remote_client_image}" "${remote_worker_image}"
test -n "${middleware_image}"
test -n "${remote_client_image}"
test -n "${remote_worker_image}"
```

三个镜像都必须使用明确版本，不要使用 `latest`。升级前应在实际 GPU 节点预拉取或离线导入
`${middleware_image}`；仅在 controller 节点拉取不能证明 Hypervisor 所在节点可用。

使用固定镜像渲染 v2 ProviderConfig，检查生成结果后留到步骤 2 apply：

```bash
helm template ${release} "${chart}" -n ${ns} \
  -f tf-backup/helm-values.yaml \
  --set providerConfigs.nvidia.enabled=true \
  --set-string providerConfigs.nvidia.name="${provider_name}" \
  --set-string providerConfigs.nvidia.images.middleware="${middleware_image}" \
  --set-string providerConfigs.nvidia.images.remoteClient="${remote_client_image}" \
  --set-string providerConfigs.nvidia.images.remoteWorker="${remote_worker_image}" \
  -s templates/provider-config-nvidia.yaml \
  > tf-backup/providerconfig-v2.yaml

grep -E '^(kind: ProviderConfig|  name:|    (middleware|remoteClient|remoteWorker):)' \
  tf-backup/providerconfig-v2.yaml
```

---

## 升级步骤

### 步骤 1：先升级 CRD

```bash
kubectl apply --server-side --force-conflicts -f "${chart}/crds/"
kubectl get crd providerconfigs.tensor-fusion.ai   # 新增 CRD 已就绪
```

### 步骤 2：创建 ProviderConfig 并同步 RBAC

步骤 1 已经安装 ProviderConfig CRD。先 apply 预检阶段渲染并核对过的 NVIDIA
ProviderConfig：

```bash
kubectl apply -f tf-backup/providerconfig-v2.yaml
test "$(kubectl get providerconfig "${provider_name}" \
  -o jsonpath='{.spec.vendor}')" = "NVIDIA"
test "$(kubectl get providerconfig "${provider_name}" \
  -o jsonpath='{.spec.images.middleware}')" = "${middleware_image}"
test "$(kubectl get providerconfig "${provider_name}" \
  -o jsonpath='{.spec.images.remoteClient}')" = "${remote_client_image}"
test "$(kubectl get providerconfig "${provider_name}" \
  -o jsonpath='{.spec.images.remoteWorker}')" = "${remote_worker_image}"
```

四条 `test` 均通过才继续。

不走 `helm upgrade` 时，chart 模板的变更需要手动同步。v1 → v2 的 RBAC 差异有两处（缺了 operator/hypervisor 会报 Forbidden）：

- **operator ClusterRole**（`rbac.yaml`）：`tensor-fusion.ai` 资源列表新增 `providerconfigs`
- **hypervisor ClusterRole/ClusterRoleBinding**（`rbac-hypervisor.yaml`）：`tensor-fusion.ai` 资源列表需要包含 `providerconfigs`，并保留模板里的 `get/list/watch/create/update/patch` verbs

分别渲染并 apply 两份 RBAC，不能只更新其中一个：

```bash
helm template ${release} "${chart}" -n ${ns} \
  -f tf-backup/helm-values.yaml \
  -s templates/rbac.yaml > tf-backup/rbac-operator-v2.yaml

helm template ${release} "${chart}" -n ${ns} \
  -f tf-backup/helm-values.yaml \
  -s templates/rbac-hypervisor.yaml > tf-backup/rbac-hypervisor-v2.yaml

kubectl apply --server-side \
  --dry-run=server \
  -f tf-backup/rbac-operator-v2.yaml \
  -f tf-backup/rbac-hypervisor-v2.yaml

kubectl apply --server-side \
  -f tf-backup/rbac-operator-v2.yaml \
  -f tf-backup/rbac-hypervisor-v2.yaml
```

### 步骤 3：更新 Karpenter 资源声明（使用 Karpenter 时必须）

worker pod 的 kubelet 扩展资源由 v1 的单一 `tensor-fusion.ai/index: 1` 变为 v2 的 16 个 bucket 资源 `tensor-fusion.ai/index_0..index_f`（每个容量 36）。Karpenter 的扩容判定依赖声明的资源容量——不更新的话，**v2 worker pod 触发的扩容永远不会发生**。

```yaml
apiVersion: karpenter.sh/v1alpha1
kind: NodeOverlay
metadata:
  name: tensor-fusion-overlay
spec:
  requirements: []
  capacity:
    # 保留旧 key：回滚到 v1 后新建的 worker 仍请求它，删了扩容就断
    tensor-fusion.ai/index: "512"
    tensor-fusion.ai/index_0: "36"
    tensor-fusion.ai/index_1: "36"
    tensor-fusion.ai/index_2: "36"
    tensor-fusion.ai/index_3: "36"
    tensor-fusion.ai/index_4: "36"
    tensor-fusion.ai/index_5: "36"
    tensor-fusion.ai/index_6: "36"
    tensor-fusion.ai/index_7: "36"
    tensor-fusion.ai/index_8: "36"
    tensor-fusion.ai/index_9: "36"
    tensor-fusion.ai/index_a: "36"
    tensor-fusion.ai/index_b: "36"
    tensor-fusion.ai/index_c: "36"
    tensor-fusion.ai/index_d: "36"
    tensor-fusion.ai/index_e: "36"
    tensor-fusion.ai/index_f: "36"
```

- chart 里的 `templates/node-overlay.yaml` 是同样内容（不含旧 key），但模板用 `lookup` 探测 CRD，离线 `helm template` 渲染为空——直接 `kubectl apply` 上面的 YAML 即可。
- 新旧 key 并存无副作用；v2 稳定运行后再删除旧的 `tensor-fusion.ai/index`。
- 旧 key 的容量按你现网原有声明值保留（v1 device plugin 实际广播 512 个 slot，每 worker 请求 1）。

### 步骤 4：确认待写入的节点隔离/切分模式

步骤 1 已经 apply v2 CRD，此时 TFC schema 才支持 `defaultIsolationMode` 和
`isolationModeRules`。再次确认升级前预检第 3 节审核的规划；实际写入由步骤 5 中切换
Hypervisor 镜像的同一个 TFC patch 完成。v2 通过每个 GPUPool 的 `nodeManagerConfig` 统一
管理节点能力，不需要在 Kubernetes Node 或 GPUNode 上写
`tensor-fusion.ai/isolationMode` label。未命中规则的节点使用
`defaultIsolationMode`；规则按顺序匹配，第一条命中后停止：

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

`nodeManagerConfig` 是节点能力的期望配置；业务 Pod 上的
`tensor-fusion.ai/isolation` annotation 是 workload 请求。soft、hard、partitioned
是互斥的节点能力，需要保证 pool 策略和 workload 请求一致。未配置规则时默认值为
`soft`，一键安装和升级不需要额外的逐节点操作。

后续修改 `defaultIsolationMode` 或 `isolationModeRules` 会进入 GPUPool 现有的 Hypervisor
滚动更新流程。启用 `autoUpdateHypervisor` 后按批次更新，并且只重建最终有效模式发生变化
的节点。

### 步骤 5：切换 operator 和 Hypervisor

依次切换 scheduler ConfigMap 和 operator 镜像；确认 v2 operator Ready 后，再修改 TFC
中的 Hypervisor 镜像。

`<release>-config` 的 `scheduler-config.yaml` 在 v1/v2 间互不兼容，因此在切换 operator
镜像前渲染并 apply v2 ConfigMap。

先设置目标镜像，并再次确认所有待升级 pool 已显式开启自动滚动：

```bash
v2_operator_image=<registry>/tensor-fusion-operator:<v2-tag>
v2_hypervisor_image=<registry>/tensor-fusion-hypervisor:<v2-tag>

kubectl get gpupools -A -o json \
  | jq -e 'all(.items[];
      .spec.nodeManagerConfig.nodePoolRollingUpdatePolicy.autoUpdateHypervisor == true)'
```

命令输出 `true` 才继续。以下以 TFC 中第一个 pool 为例；多 pool 环境必须逐个修改对应路径：

```bash
# 1. 渲染并切换到 v2 scheduler 配置
helm template ${release} "${chart}" -n ${ns} \
  -f tf-backup/helm-values.yaml \
  -s templates/config.yaml > tf-backup/config-v2.yaml

kubectl apply -f tf-backup/config-v2.yaml

# 2. 立即切换 operator，并等待 v2 operator 使用配套配置启动
kubectl -n ${ns} set image deploy/${controller_deploy} \
  controller=${v2_operator_image}
kubectl -n ${ns} rollout status deploy/${controller_deploy} --timeout=180s

# 3. v2 operator Ready 后再更新 Hypervisor 期望状态
# 以下仅以 gpuPools[0]、默认 soft、无规则为语法示例；不要直接用于生产环境。
# 多 pool 环境按实际数组索引逐个修改，并写入升级前已审核的 defaultIsolationMode
# 和 isolationModeRules。
kubectl patch tensorfusioncluster <cluster-name> --type=json -p="[
  {\"op\":\"replace\",\"path\":\"/spec/gpuPools/0/specTemplate/componentConfig/hypervisor/image\",\"value\":\"${v2_hypervisor_image}\"},
  {\"op\":\"add\",\"path\":\"/spec/gpuPools/0/specTemplate/nodeManagerConfig/defaultIsolationMode\",\"value\":\"soft\"},
  {\"op\":\"add\",\"path\":\"/spec/gpuPools/0/specTemplate/nodeManagerConfig/isolationModeRules\",\"value\":[]}
]"
```

### 步骤 6：验证

workload 的 annotation 样例和验证方法见[跨厂商测试矩阵](cross-vendor-test-matrix.md)。

```bash
kubectl get tensorfusioncluster -A    # Phase=Running
kubectl get gpupool -A                # 全部 Running
kubectl get gpu -A -o wide            # status 正常更新
# 提交测试 Workload，并在业务容器中实际调用 CUDA/NVML；仅 Pod Ready 不算通过
```

逐节点验证 v1 Hypervisor 已由同名 v2 Pod 自动替换。以下命令应全部成功：

```bash
node=<gpu-node-name>
pod=hypervisor-${node}
v1_uid=$(awk -v pod=${pod} '$1 == pod {print $2}' tf-backup/hypervisor-v1.tsv)

kubectl -n ${ns} wait --for=condition=Ready pod/${pod} --timeout=180s
v2_uid=$(kubectl -n ${ns} get pod ${pod} -o jsonpath='{.metadata.uid}')
v2_image=$(kubectl -n ${ns} get pod ${pod} \
  -o jsonpath='{.spec.containers[?(@.name=="tensor-fusion-hypervisor")].image}')
hypervisor_count=$(kubectl -n ${ns} get pods -o json | jq \
  --arg node "${node}" '[.items[]
    | select(.spec.nodeName == $node)
    | select(any(.spec.containers[]?; .name == "tensor-fusion-hypervisor"))]
    | length')

test -n "${v1_uid}"
test "${v2_uid}" != "${v1_uid}"
test "${v2_image}" = "${v2_hypervisor_image}"
test "${hypervisor_count}" = "1"

kubectl get gpupools -A \
  -o custom-columns='POOL:.metadata.name,PROGRESS:.status.componentStatus.hypervisorUpdateProgress,SYNCED:.status.componentStatus.hypervisorConfigSynced'

test "$(kubectl get providerconfig ${provider_name} \
  -o jsonpath='{.spec.images.middleware}')" = "${middleware_image}"
test "$(kubectl get providerconfig ${provider_name} \
  -o jsonpath='{.spec.images.remoteClient}')" = "${remote_client_image}"
test "$(kubectl get providerconfig ${provider_name} \
  -o jsonpath='{.spec.images.remoteWorker}')" = "${remote_worker_image}"
```

有 GPUNode 的待升级 pool 最终必须满足 `PROGRESS=100`、`SYNCED=true`，并且每个 GPU 节点只有一个 Hypervisor Pod。没有 GPUNode 的空 pool 不执行实际滚动，进度字段可能为空，应检查其 vendor、镜像和 rolling policy 配置，不要要求 `PROGRESS=100`。正常升级过程中不应出现第二种 Pod 名称，也不应残留 v1 UID或 v1镜像。

存量业务验证应在升级前记录 Pod UID、restartCount 和 GPU UUID，升级后确认三者不变，并分别验证：

- 升级前已初始化并持续运行的 CUDA 进程没有中断；
- 原 Pod 内升级后新启动的 `nvidia-smi`/CUDA 进程可以完成 Hypervisor 初始化；
- 实际 CUDA kernel 和显存分配成功，而不是只检查 Pod Running。

建议至少验证以下组合：

- 在 soft 节点先创建 soft 切片 workload，再创建 shared 整卡 workload，确认分配到不同 GPU。
- 在 hard 节点创建两组 `20% + 1Gi` 的 remote hard workload，确认按 binpack 分配到同一 GPU，再创建 shared workload 使用另一张完整空闲卡。
- hard 绝对值用例确认 worker 实际收到正确的 `TF_CUDA_SM_PERCENT_LIMIT`；显存用例分别验证低于 limit 的分配成功、高于 limit 的分配失败。
- local/remote 业务容器均确认 `cuInit=0`、CUDA/NVML device count 正确，并且只看到分配的 GPU UUID。

---

## 回退方案（Rollback）

使用升级前备份的 v1 ConfigMap；也可以用原 v1 Chart 和升级前 values 重新渲染：

```bash
helm template ${release} <v1-chart-path> -n ${ns} \
  -f tf-backup/helm-values.yaml \
  -s templates/config.yaml > tf-backup/config-v1-rendered.yaml
```

按以下步骤回退：

```bash
v1_operator_image=<registry>/tensor-fusion-operator:<v1-tag>
v1_hypervisor_image=<registry>/tensor-fusion-hypervisor:<v1-tag>

# 1. 恢复 v1 scheduler ConfigMap（二选一）
kubectl apply -f tf-backup/config-v1.json
# kubectl apply -f tf-backup/config-v1-rendered.yaml

# 2. 切换 v1 operator 并等待 Ready
kubectl -n ${ns} set image deploy/${controller_deploy} \
  controller=${v1_operator_image}
kubectl -n ${ns} rollout status deploy/${controller_deploy} --timeout=180s

# 3. 恢复 TFC 中的 v1 Hypervisor 镜像
kubectl patch tensorfusioncluster <cluster-name> --type=json -p="[{
  \"op\":\"replace\",
  \"path\":\"/spec/gpuPools/0/specTemplate/componentConfig/hypervisor/image\",
  \"value\":\"${v1_hypervisor_image}\"
}]"

# 4. 等待每个节点的 Hypervisor 使用 v1 镜像恢复 Ready
kubectl -n ${ns} wait --for=condition=Ready \
  pod/hypervisor-<node-name> --timeout=180s
```

v2 CRD、ProviderConfig、RBAC 和 Karpenter NodeOverlay 保持不变。

---

## 检查清单

- [ ] 已从公开 Helm 仓库下载并固定 Chart `1.8.3`，`appVersion=2.15.0`；兼容字段和隔离策略字段预检通过，且未执行 `helm upgrade`
- [ ] CRD 来自 compat 改动合并后的 main（预检 1 通过）
- [ ] `nodeManagerConfig` / `gpuCount` 预检通过，所有待升级 pool 均为 `autoUpdateHypervisor=true`
- [ ] 已全量备份 CRD、CR、Helm values、v1 ConfigMap 与 controller Deployment
- [ ] v2 NVIDIA ProviderConfig 已使用固定版本渲染并在步骤 2 创建；middleware 已在实际 GPU 节点验证可拉取或完成离线导入
- [ ] 先 apply CRD，再创建 NVIDIA ProviderConfig、同步 RBAC、更新 Karpenter（如使用）并确认隔离策略；最后按“v2 ConfigMap → v2 operator Ready → v2 Hypervisor 镜像”的顺序执行
- [ ] （Karpenter 环境）NodeOverlay 已声明 `index_0..index_f`，且保留旧 `tensor-fusion.ai/index`
- [ ] 每个待升级 GPUPool 已规划 `defaultIsolationMode` 和有序的 `isolationModeRules`；每条 selector 的实际命中节点及重叠优先级均已核对
- [ ] v1/v2 Hypervisor 名称均为 `hypervisor-<node>`；升级后 UID 已变化、镜像为 v2、每节点仅一个 Pod
- [ ] 有 GPUNode 的待升级 pool 为 `hypervisorUpdateProgress=100`、`hypervisorConfigSynced=true`；空 pool 已单独核对配置
- [ ] 升级后 Cluster/Pool=Running，local/remote 测试 Workload 可调度并能实际调用 CUDA/NVML
- [ ] shared workload 只使用完整空闲、未分区 GPU；hard 百分比/绝对值与显存限额验证通过
- [ ] 存量 soft/hard Pod 的 UID/restartCount/GPU UUID 不变，升级后新 CUDA/NVML 调用正常
- [ ] 回退预案明确：CRD 不动；按“v1 ConfigMap → v1 operator Ready → v1 Hypervisor 镜像”的顺序执行，由 operator 自动替换同名 Pod
