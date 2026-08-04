# TensorFusion v1 → v2(main) 升级最佳实践

> 适用范围：从 `v1` 分支线升级到 `main` 分支线（v2 架构）。
> **前提**：升级所用的 CRD 必须来自 CRD v1 兼容改动（`feat/crd-v1-compat`）合并之后的 main——该改动把 v1 独有字段（`nvLink`、`model`、`nodeDiscovery`、`remoteModeImage`/`embeddedModeImage` 等）补回了 v2 CRD。**不要使用合并前 main 上的 CRD**，否则 v1 的 spec 数据会被 apiserver 裁剪（pruning），回退时无法恢复。

---

## 核心结论

- CRD 的 group/version 不变，仍为 `tensor-fusion.ai/v1`，单一 served/storage 版本，无 Conversion Webhook，无 K8s API 版本迁移。
- **v2 CRD 是 v1 schema 的严格超集**（逐字段比对验证）：无字段删除，v1 写入的数据在 v2 CRD 下完整保留。
- 回退时 **CRD 保持 v2 不动**，但不能只换 operator 镜像：v1/v2 的 scheduler ConfigMap 和 Hypervisor 镜像必须与各自 operator 配套切换。
- v1/v2 使用相同的 Hypervisor Pod 名称 `hypervisor-<node>`。升级时由 v2 operator 删除旧 UID，并以相同名称创建 v2 Pod；正常流程不需要手动清理旧 Pod。
- 升级顺序：**先 apply CRD → 同步 RBAC → 确认 `autoUpdateHypervisor=true` → 将 operator 缩容到 0 → 同步切换 ConfigMap、operator 和 Hypervisor 镜像 → 恢复 operator 副本**。如使用 Helm 管理完整升级，使用 Chart `1.8.0`，并在生产 values 中固定配套的 operator/hypervisor 镜像版本。

---

## 升级前预检

### 1. 确认 CRD 包含 v1 兼容字段（关键）

```bash
grep -l 'nvLink' charts/tensor-fusion/crds/tensor-fusion.ai_gpus.yaml \
  && grep -l 'remoteModeImage' charts/tensor-fusion/crds/tensor-fusion.ai_gpupools.yaml \
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

必须在修改 Hypervisor 镜像之前启用。配置变化发生后再单独打开开关，当前状态机不会补做之前错过的滚动更新。

### 3. 全量备份（保险，不再是回退的依赖项）

```bash
release=tensor-fusion-sys  # 改成实际 Helm release 名称
ns=tensor-fusion-sys       # 改成实际 operator 所在 namespace
controller_deploy=$(kubectl -n ${ns} get deploy \
  -l tensor-fusion.ai/component=operator \
  -o jsonpath='{.items[0].metadata.name}')
controller_replicas=$(kubectl -n ${ns} get deploy ${controller_deploy} \
  -o jsonpath='{.spec.replicas}')
mkdir -p tf-backup
helm -n ${ns} get values ${release} -o yaml > tf-backup/helm-values.yaml
for resource in $(kubectl api-resources --api-group=tensor-fusion.ai -o name); do
  kubectl get ${resource} -A -o yaml > tf-backup/${resource}.yaml
done
kubectl get crd -o name | grep 'tensor-fusion.ai' \
  | xargs kubectl get -o yaml > tf-backup/crds.yaml
kubectl -n ${ns} get deploy ${controller_deploy} -o yaml > tf-backup/controller-deploy.yaml
kubectl -n ${ns} get configmap ${release}-config -o yaml > tf-backup/config-v1.yaml
kubectl -n ${ns} get pods -o json | jq -r '
  .items[]
  | select(.metadata.name | startswith("hypervisor-"))
  | [.metadata.name, .metadata.uid, .spec.nodeName,
     (.spec.containers[] | select(.name == "tensor-fusion-hypervisor") | .image)]
  | @tsv' > tf-backup/hypervisor-v1.tsv
```

---

## 升级步骤

### 步骤 1：先升级 CRD

```bash
kubectl apply --server-side --force-conflicts -f charts/tensor-fusion/crds/
kubectl get crd providerconfigs.tensor-fusion.ai   # 新增 CRD 已就绪
```

### 步骤 2：同步 RBAC（v2 新增权限，必须先做）

不走 `helm upgrade` 时，chart 模板的变更需要手动同步。v1 → v2 的 RBAC 差异有两处（缺了 operator/hypervisor 会报 Forbidden）：

- **operator ClusterRole**（`rbac.yaml`）：`tensor-fusion.ai` 资源列表新增 `providerconfigs`
- **hypervisor ClusterRole/ClusterRoleBinding**（`rbac-hypervisor.yaml`）：`tensor-fusion.ai` 资源列表需要包含 `providerconfigs`，并保留模板里的 `get/list/watch/create/update/patch` verbs

可以直接用新版 chart 渲染后单独 apply 这两个文件：

```bash
helm template ${release} ./charts/tensor-fusion -n ${ns} \
  -f tf-backup/helm-values.yaml \
  -s templates/rbac.yaml -s templates/rbac-hypervisor.yaml | kubectl apply --server-side -f -
```

**ConfigMap 也必须同步**：`<release>-config` 的 `scheduler-config.yaml` 在 v1/v2 间互不兼容。v2 为 `GPUResourcesFit` 增加了 `permit`、`postFilter`、`preBind`、`preEnqueue` 等扩展点，v1 二进制没有实现这些接口；`GPUNetworkTopologyAware` 的参数结构也已改变。

此处只提前渲染并检查 v2 ConfigMap，不要在 v1 operator 仍运行时 apply。v1 operator 使用 v2 scheduler 配置会因 `GPUNetworkTopologyAware` 扩展点不兼容而 CrashLoop；实际 apply 放在步骤 3 的 `operator=0` 窗口内：

```bash
helm template ${release} ./charts/tensor-fusion -n ${ns} \
  -f tf-backup/helm-values.yaml \
  -s templates/config.yaml > tf-backup/config-v2.yaml
```

### 步骤 2.5：更新 Karpenter 资源声明（使用 Karpenter 时必须）

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

### 步骤 2.6：规划并标记节点隔离/切分模式

v2 需要提前规划每个 GPU 节点承担的切分/隔离能力，并在 **Kubernetes Node** 上打 label：

```bash
kubectl label node <node-name> tensor-fusion.ai/isolationMode=soft --overwrite
# 推荐值：soft / hard / partitioned
```

这不是业务 Pod 上的 `tensor-fusion.ai/isolation` annotation。两者职责不同：

| 位置 | Key | 作用 |
|---|---|---|
| Node label | `tensor-fusion.ai/isolationMode` | operator 同步到 GPUNode，并作为该节点 hypervisor 的 `--isolation-mode=<mode>` 启动参数；hypervisor 上报 GPU.status.isolationMode |
| Pod annotation | `tensor-fusion.ai/isolation` | workload 请求的分配/隔离策略；soft、hard、partitioned 按节点能力过滤，shared 按完整空闲 GPU 过滤 |

soft、hard、partitioned 是互斥的节点切分/隔离能力，需要保证节点规划和 workload 请求一致：soft workload 落到 soft 节点，hard workload 落到 hard 节点，partitioned workload 落到 partitioned 节点。未规划时不要依赖默认值，尤其是默认 workload isolation 为 `soft`，但 hypervisor 侧默认参数不是升级策略的一部分，可能导致升级后调度过滤无可用 GPU。

`shared` 从 v2.13.0 起表示**整卡分配策略**，不是必须单独规划的第四种节点能力：

- shared workload 可以使用 soft、hard、partitioned 或兼容 shared 节点上的完整空闲、未分区 GPU。
- GPU 只要已有 soft/hard 切片、已有整卡分配或已被 partitioned/MIG 切分，就不再满足 shared 请求。
- 不要求节点存在 `tensor-fusion.ai/isolationMode=shared` label；已有 shared label 继续兼容。
- scheduler 不优先 shared label 节点，仍沿用现有 GPU/节点紧凑（binpack）评分，保留更多完整空闲卡和空闲节点。

建议升级前按 pool/节点用途一次性标好：

```bash
kubectl label node <soft-node> tensor-fusion.ai/isolationMode=soft --overwrite
kubectl label node <hard-node> tensor-fusion.ai/isolationMode=hard --overwrite
kubectl label node <partition-node> tensor-fusion.ai/isolationMode=partitioned --overwrite
kubectl get nodes -L tensor-fusion.ai/isolationMode
```

后续修改该 label 时，v2 operator 会删除并重建对应节点的 hypervisor pod 以应用新的 `--isolation-mode`，应按维护变更处理，避免和运行中 worker/业务混在同一个升级动作里。

### 步骤 3：在 operator=0 窗口内原子切换

不能在 v1 operator 仍运行时先修改 v2 Hypervisor 镜像，否则 v1 controller 可能抢先处理新配置并启动不兼容的 v2 Hypervisor。反过来也不能让 v2 operator 启动 v1 Hypervisor：两代 CLI 参数不兼容，会在 init container 阶段 CrashLoop。

先设置目标镜像，并再次确认所有待升级 pool 已显式开启自动滚动：

```bash
v2_operator_image=<registry>/tensor-fusion-operator:<v2-tag>
v2_hypervisor_image=<registry>/tensor-fusion-hypervisor:<v2-tag>

kubectl get gpupools -A -o json \
  | jq -e 'all(.items[];
      .spec.nodeManagerConfig.nodePoolRollingUpdatePolicy.autoUpdateHypervisor == true)'
```

命令输出 `true` 才继续。然后停止 operator，在没有 controller 竞争的窗口中同时切换 ConfigMap 和镜像配置。以下以 TFC 中第一个 pool 为例；多 pool 环境必须逐个修改对应路径：

```bash
# 1. 停止 v1 operator，并确认旧 Pod 已退出
kubectl -n ${ns} scale deploy/${controller_deploy} --replicas=0
kubectl -n ${ns} wait --for=delete pod \
  -l tensor-fusion.ai/component=operator --timeout=120s

# 2. 切换到 v2 scheduler 配置
kubectl apply -f tf-backup/config-v2.yaml

# 3. operator 停止期间只更新期望状态，不会被 v1 抢先 reconcile
kubectl patch tensorfusioncluster <cluster-name> --type=json -p="[{
  \"op\":\"replace\",
  \"path\":\"/spec/gpuPools/0/specTemplate/componentConfig/hypervisor/image\",
  \"value\":\"${v2_hypervisor_image}\"
}]"
kubectl -n ${ns} set image deploy/${controller_deploy} \
  controller=${v2_operator_image}

# 4. 启动 v2 operator
kubectl -n ${ns} scale deploy/${controller_deploy} \
  --replicas=${controller_replicas}
kubectl -n ${ns} rollout status deploy/${controller_deploy} --timeout=180s
```

- v1/v2 Hypervisor 都使用 `hypervisor-<node>`。v2 operator 会删除旧 UID，再用相同名称创建 v2 Pod；不要把手动删除旧 Pod作为正常升级步骤。
- 如果同名 Pod 长时间卡在 Terminating/CrashLoop，可在确认 v2 operator、TFC 镜像和 scheduler ConfigMap 均正确后，按节点删除该 Pod作为故障恢复；不要用宽泛 selector 一次删除所有节点。
- hypervisor / worker / client 镜像由 GPUPool / TensorFusionCluster 的 `componentConfig` 控制，不需要修改 operator Deployment 中的其他容器。
- `INITIAL_GPU_NODE_LABEL_SELECTOR`：v2 chart 去掉了默认值 `nvidia.com/gpu.present=true`，但现网 Deployment 里该 env 已是渲染后的实际值，只换镜像不受影响。
- v2 Hypervisor 必须包含“启动时根据运行中 Pod 的 `gpu-ids`、isolation、算力和显存注解恢复 soft/shared/hard allocation”的修复。未包含该修复的版本虽然不会重建业务 Pod，但存量 Pod 内新启动的 CUDA/NVML 进程会报 `Pod ... not found`。

### 步骤 4：升级后的功能迁移（不阻塞升级，按需进行）

v2 operator 不再读取以下 v1 字段（字段保留仅为兼容与回退，数据不会丢，但功能上由新机制接管）：

| v1 字段 | v2 接管方式 |
|---|---|
| `client.remoteModeImage` / `embeddedModeImage` | 合并为 `client.image`（ProviderConfig 存在时以其为准） |
| `componentConfig.nodeDiscovery` | 节点发现已内置 |
| `gpu.status.nvLink` / `model` | `status.topology`（控制器自动重建）/ ProviderConfig.hardwareMetadata |
| `gpu-info` ConfigMap | ProviderConfig CRD（按厂商创建，参考 `config/samples/v1_providerconfig.yaml`，NVIDIA 默认配置可用 chart 模板 `templates/provider-config-nvidia.yaml` 渲染） |

#### shared 整卡 workload

沿用已有 dedicated GPU 请求方式，用户侧不需要填写 TFLOPS/VRAM request/limit：

```yaml
metadata:
  labels:
    tensor-fusion.ai/enabled: "true"
  annotations:
    tensor-fusion.ai/dedicated-gpu: "true"
    tensor-fusion.ai/gpu-count: "1"
    tensor-fusion.ai/gpu-model: "<gpu-model>"
    tensor-fusion.ai/gpupool: "<pool-name>"
    tensor-fusion.ai/is-local-gpu: "true" # 远程模式改为 "false"
    tensor-fusion.ai/isolation: "shared"
    tensor-fusion.ai/vendor: "NVIDIA"
```

webhook 根据 `dedicated-gpu` 和 `gpu-model` 在内部补齐所选型号的整卡 TFLOPS/VRAM 容量；allocator 最终将所选 GPU 的 available TFLOPS/VRAM 都记为 0。不要在 shared workload 上填写切片值（例如 `10 TFLOPS / 1Gi`），避免用户声明与整卡实际分配不一致。

#### hard workload 的算力与显存

hard 模式支持两种算力写法，二选一：

```yaml
# 百分比：直接注入 hard limiter
tensor-fusion.ai/compute-percent-request: "20"
tensor-fusion.ai/compute-percent-limit: "20"

# 绝对值：scheduler 选卡后按 GPU CR 容量换算百分比
tensor-fusion.ai/tflops-request: "10"
tensor-fusion.ai/tflops-limit: "10"
```

绝对 TFLOPS 使用所选 `GPU.status.capacity.tflops` 换算并向上取整。例如 GPU 容量为 80 TFLOPS、请求 10 TFLOPS 时，`10 / 80 * 100` 注入为 `TF_CUDA_SM_PERCENT_LIMIT=13`。不要使用 hypervisor 运行时探测出的理论 TFLOPS 作为换算基数。

VRAM 仍只支持绝对值（例如 `1Gi`），不支持百分比：

```yaml
tensor-fusion.ai/vram-request: "1Gi"
tensor-fusion.ai/vram-limit: "1Gi"
```

### 步骤 5：验证

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

升级和回退都要避免旧 operator 抢先处理新一代 Hypervisor 镜像。回退时同样先将 operator 缩容到 0，再在停止窗口内恢复 v1 scheduler ConfigMap、v1 Hypervisor 镜像和 v1 operator 镜像。

可直接使用升级前备份，或用原 v1 chart 和升级前 values 重新渲染 v1 ConfigMap：

```bash
helm template ${release} <v1-chart-path> -n ${ns} \
  -f tf-backup/helm-values.yaml \
  -s templates/config.yaml > tf-backup/config-v1-rendered.yaml
```

执行原子回退：

```bash
v1_operator_image=<registry>/tensor-fusion-operator:<v1-tag>
v1_hypervisor_image=<registry>/tensor-fusion-hypervisor:<v1-tag>

kubectl -n ${ns} scale deploy/${controller_deploy} --replicas=0
kubectl -n ${ns} wait --for=delete pod \
  -l tensor-fusion.ai/component=operator --timeout=120s

# 二选一：使用升级前备份，或使用刚渲染的 v1 ConfigMap
kubectl apply -f tf-backup/config-v1.yaml
# kubectl apply -f tf-backup/config-v1-rendered.yaml

kubectl patch tensorfusioncluster <cluster-name> --type=json -p="[{
  \"op\":\"replace\",
  \"path\":\"/spec/gpuPools/0/specTemplate/componentConfig/hypervisor/image\",
  \"value\":\"${v1_hypervisor_image}\"
}]"
kubectl -n ${ns} set image deploy/${controller_deploy} \
  controller=${v1_operator_image}

kubectl -n ${ns} scale deploy/${controller_deploy} \
  --replicas=${controller_replicas}
kubectl -n ${ns} rollout status deploy/${controller_deploy} --timeout=180s
```

确认 v1 operator Ready 后，等待同名 `hypervisor-<node>` 自动替换为 v1 镜像并恢复 2/2 Ready。正常回退不需要删除另一种名称的 Pod；如果同名 Pod 卡住，仅将按节点删除作为故障恢复。不能只执行 `rollout undo`：Deployment 会回滚镜像，但不会恢复 v1 ConfigMap和 TensorFusionCluster/GPUPool 中的 Hypervisor 镜像。

- **不要把 v1 旧 CRD apply 回去**。v2 CRD 是超集，v1 operator 在其下完全正常工作；反向 apply 旧 CRD 会触发 pruning，把 v2 写入的字段（`topology`、`gangScheduling`、`isolationMode` 等）剪掉，影响二次升级。带默认值的字段还可能在重新 apply v2 CRD 后被静默填入默认值，key 级 diff 无法发现值已变化，因此恢复时必须使用备份做值级比对。
- 回退后 v1 operator 全量更新对象时会抹掉它不认识的 v2-only status 字段——无害，二次升级时 v2 控制器会自动重新发现填充。
- ProviderConfig 等 v2 专有资源及新增的 RBAC 权限回退后闲置即可，无需删除。
- Karpenter NodeOverlay 也不用动：旧 `tensor-fusion.ai/index` key 已按步骤 2.5 保留，v1 worker 的扩容不受影响。

---

## 检查清单

- [ ] CRD 来自 compat 改动合并后的 main（预检 1 通过）
- [ ] `nodeManagerConfig` / `gpuCount` 预检通过，所有待升级 pool 均为 `autoUpdateHypervisor=true`
- [ ] 已全量备份 CRD、CR、Helm values、v1 ConfigMap 与 controller Deployment
- [ ] 先 apply CRD/RBAC，再将 operator 缩容到 0，并在停止窗口内同步切换 v2 ConfigMap、operator/Hypervisor 镜像
- [ ] （Karpenter 环境）NodeOverlay 已声明 `index_0..index_f`，且保留旧 `tensor-fusion.ai/index`
- [ ] GPU 节点已按用途标记 `tensor-fusion.ai/isolationMode=soft|hard|partitioned`（已有 shared label 可保留兼容，但不再要求）
- [ ] v1/v2 Hypervisor 名称均为 `hypervisor-<node>`；升级后 UID 已变化、镜像为 v2、每节点仅一个 Pod
- [ ] 有 GPUNode 的待升级 pool 为 `hypervisorUpdateProgress=100`、`hypervisorConfigSynced=true`；空 pool 已单独核对配置
- [ ] 升级后 Cluster/Pool=Running，local/remote 测试 Workload 可调度并能实际调用 CUDA/NVML
- [ ] shared workload 只使用完整空闲、未分区 GPU；hard 百分比/绝对值与显存限额验证通过
- [ ] 存量 soft/hard Pod 的 UID/restartCount/GPU UUID 不变，升级后新 CUDA/NVML 调用正常
- [ ] 回退预案明确：CRD 不动；operator 缩容到 0 后同步恢复 v1 ConfigMap、operator/Hypervisor 镜像，由 operator 自动替换同名 Pod
