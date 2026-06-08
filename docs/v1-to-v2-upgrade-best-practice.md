# TensorFusion v1 → v2(main) 升级最佳实践

> 适用范围：从 `v1` 分支线（chart 1.7.6 / appVersion 1.58.4）升级到 `main` 分支线（v2 架构）。
> 本文基于 `git diff v1 main` 对 `api/v1` 与 `config/crd/bases` 的实际差异编写。

---

## 0. 一句话结论

- **CRD 的 group/version 没有变**，仍然是 `tensor-fusion.ai/v1`（所有 CRD 均 `served: true / storage: true`，单版本）。因此**不需要** Conversion Webhook，也没有 K8s API 版本迁移。
- 这是一次**同版本内的 Schema 演进**：新增 1 个 CRD（`ProviderConfig`），并对 `GPUPool` 做了**破坏性字段调整**。
- 最大风险来自两点：
  1. **结构化 Schema 的字段裁剪（pruning）**——升级后被删除的字段会在 CR 下次写入时被自动丢弃。
  2. **Helm 不会自动升级 CRD**（`crds/` 目录在 `helm upgrade` 时被忽略），必须手动 `kubectl apply`。
- 升级与回退的安全网都是同一件事：**升级前完整备份所有 TensorFusion CR**。

---

## 1. CRD 变更总览

| CRD | 变更类型 | 说明 |
|---|---|---|
| **ProviderConfig**（新增，Cluster 级） | 🟢 新增 | 替代集中式 `gpu-info` ConfigMap，按硬件厂商管理镜像 / 硬件元数据 / 虚拟化(分区)模板 / 设备插件探测 |
| **GPUPool** | 🔴 破坏性 | `componentConfig.client` 镜像字段合并、删除 `nodeDiscovery`、`nodeManagerConfig` 变为必填、新增多厂商支持 |
| **GPU**（status） | 🟡 状态字段 | 删除 `model`/`nvLink`，新增 `isolationMode`/`allocatedPartitions`/`topology`（由控制器重建，非用户编辑） |
| **WorkloadProfile / TensorFusionWorkload** | 🟢 新增字段 | 新增 `partitionTemplateId`、`gangScheduling` |
| **SchedulingConfigTemplate** | ⚪ 默认值 | AutoSetResources 百分位默认值调整（仅注释/默认值，无结构变化） |
| **TensorFusionCluster** | ⚪ 内部 | 仅常量包重构（`api/v1/constants.go` → `pkg/constants`），CRD schema 实质不变 |
| 其余 CRD（gpunodes / gpunodeclaims / gpunodeclasses / gpuresourcequotas / connections / nodeclasses） | ⚪ 无实质变更 | 仅 controller-gen 由 `v0.16.4` 升到 `v0.20.0` 重新生成 |

---

## 2. 破坏性变更详解（必须人工迁移）

### 2.1 GPUPool —— `componentConfig.client` 镜像字段合并

```yaml
# v1（旧）
spec:
  componentConfig:
    client:
      remoteModeImage: <img>          # ❌ 已删除
      embeddedModeImage: <img>        # ❌ 已删除
      patchToEmbeddedWorkerContainer: ...   # ❌ 已删除
      patchEmbeddedWorkerToPod: ...         # ❌ 已删除

# v2（新）
spec:
  componentConfig:
    client:
      image: <img>                    # ✅ 合并为单一 image（无 ProviderConfig 时的默认值）
```

> v2 中 `client.image` / `hypervisor.image` / `worker.image` 都只是**回退默认值**；当存在对应厂商的 `ProviderConfig` 时，以 ProviderConfig 中的 `remoteClient` / `middleware` / `remoteWorker` 镜像**优先**。

### 2.2 GPUPool —— 删除 `componentConfig.nodeDiscovery`

```yaml
# v1
spec:
  componentConfig:
    nodeDiscovery:                    # ❌ 整块删除
      image: <img>
      podTemplate: ...
```
节点发现逻辑已内置/迁移，旧配置块需移除。

### 2.3 GPUPool —— `nodeManagerConfig` 变为必填

`spec.nodeManagerConfig` 由 optional 变为 **required**。若历史 GPUPool 未显式提供该块，需补齐，否则新 schema 校验会失败。

### 2.4 GPUPool —— 新增多厂商字段（可选，但是 v2 核心能力）

```yaml
spec:
  nodeManagerConfig:
    defaultVendor: NVIDIA             # 🟢 新增，单厂商模式默认厂商（默认 NVIDIA）
    multiVendorNodeSelector:          # 🟢 新增，启用多厂商模式
      AMD:    { nodeSelectorTerms: [...] }
      Ascend: { nodeSelectorTerms: [...] }
```

### 2.5 GPU.status 字段变化（控制器自动重建，无需手工迁移）

| 旧（v1 status） | 新（v2 status） |
|---|---|
| `model` | 移除（改由 ProviderConfig.hardwareMetadata 提供） |
| `nvLink`（peerCount/peers/bandwidthMBps...） | `topology.peers[]`（`peerGPUUUID` / `tier` / `linkType` / `bandwidth`） |
| — | `isolationMode`（默认 `soft`，枚举 `shared\|soft\|hard\|partitioned`） |
| — | `allocatedPartitions`（MIG/分区分配跟踪，key=podUID） |

> 这些是**观测态(status)**，由控制器在下一次发现/调度周期重新填充，无需也无法手工迁移。注意：`tensor-fusion.ai/v1` 是单存储版本，因此**旧的 `model`/`nvLink` 字段会在 GPU 对象下次更新时被裁剪丢弃**——这是预期行为。

---

## 3. 新增能力（向后兼容，按需启用）

- **ProviderConfig（新 CRD）**：v2 的核心架构变更，把原来集中在 `gpu-info` ConfigMap 的厂商信息下沉为按厂商的 CR。`main` 仓库提供示例：`config/samples/v1_providerconfig.yaml`。
- **WorkloadProfile / Workload `partitionTemplateId`**：partitioned 隔离模式下指定分区模板（来源于 Pod 注解 `tensor-fusion.ai/partition`）。
- **WorkloadProfile / Workload `gangScheduling`**：`minMembers`（>0 启用）+ `timeout`，用于多 Pod 联合调度。
- **SchedulingConfigTemplate 默认值微调**：AutoSetResources 的 target 百分位 `0.95→0.9`、upper `0.99→0.95`（不影响已显式配置的对象）。

## 4. 非 CRD 但需同步的配置（operator / Helm）

> 这些不属于 CRD，但属于同一次升级，遗漏会导致功能异常。

- **RBAC**：新增对 `providerconfigs` 资源的权限（chart 已包含；自建 RBAC 需补齐）。
- **Scheduler 配置**：`GPUResourceFitPlus` 下的 NVLink 评分旋钮（`enableNvlinkAware`/`resourceScoreWeight`/`topologyScoreWeight`/`singleGpuProtectWeight`）已删除，拓扑感知改由 `GPUNetworkTopologyAware` 插件承担。
- **dynamicConfig**：新增 `preemptClusterWideFromEnv: true`。
- **controller-deployment**：`INITIAL_GPU_NODE_LABEL_SELECTOR` 不再内置默认 `nvidia.com/gpu.present=true`，需在 values 中显式提供。
- **controller-gen**：`v0.16.4 → v0.20.0`，所有 CRD 重新生成（即便“无实质变更”的 CRD 文件也会变）。

### ⚠️ 4.1 Helm 版本号倒挂（重要）

`main` 的 chart 版本（`1.7.5` / appVersion `1.48.6`）**低于** `v1` 分支（`1.7.6` / appVersion `1.58.4`），因为 `v1` 是独立维护的补丁线。这意味着：

- `helm upgrade` 在语义版本上会被视为**降级**，部分流程/工具可能拒绝或告警。
- 升级前请确认 `main` chart 的 `version`/`appVersion` 已按你的发布策略**手动抬升**到高于现网值，或在升级命令中明确指定本地 chart 路径，不要依赖 repo 的 semver 排序。

---

## 5. 升级步骤（Best Practice）

> 目标判据：升级后 `kubectl get tensorfusioncluster` 处于 `Running`，所有 GPUPool `Running`，GPU `status.topology` 被重新填充，Workload 正常调度。

### 步骤 1：完整备份（升级与回退的共同安全网）

```bash
ns=tensor-fusion   # 按实际 namespace 调整
mkdir -p tf-backup && cd tf-backup

# 备份所有 TensorFusion CR（含 status）
for kind in tensorfusionclusters gpupools gpus gpunodes gpunodeclaims \
            gpunodeclasses gpuresourcequotas schedulingconfigtemplates \
            tensorfusionworkloads workloadprofiles tensorfusionconnections; do
  kubectl get ${kind}.tensor-fusion.ai -A -o yaml > ${kind}.yaml
done

# 备份旧 CRD 定义本身（回退时需要）
kubectl get crd -o name | grep tensor-fusion.ai \
  | xargs -I{} sh -c 'kubectl get {} -o yaml > crd-$(basename {}).yaml'

# 备份 gpu-info ConfigMap（v2 用 ProviderConfig 取代，留底以便对照迁移）
kubectl -n ${ns} get configmap -o yaml > configmaps.yaml

# 备份当前 Helm release values
helm -n ${ns} get values tensor-fusion > helm-values.yaml
```

### 步骤 2：迁移 GPUPool 配置（处理破坏性变更）

对每个 GPUPool（及在 TensorFusionCluster 内联的 pool 定义）按 [第 2 节](#2-破坏性变更详解必须人工迁移) 修改：

- `client.remoteModeImage` / `embeddedModeImage` → 合并为 `client.image`
- 删除 `client.patchToEmbeddedWorkerContainer` / `patchEmbeddedWorkerToPod`
- 删除 `componentConfig.nodeDiscovery`
- 补齐必填的 `nodeManagerConfig`（如缺）
- （可选）设置 `nodeManagerConfig.defaultVendor` / `multiVendorNodeSelector`

### 步骤 3：先升级 CRD（Helm 不会自动做）

```bash
# 用 main 分支/新 chart 中的 CRD 定义
kubectl apply --server-side --force-conflicts \
  -f charts/tensor-fusion/crds/

# 校验新 CRD 已就绪
kubectl get crd providerconfigs.tensor-fusion.ai
```

> 使用 `--server-side` 可降低大 CRD 触发 `metadata.annotations too long` 的风险。

### 步骤 4：创建 ProviderConfig（替代 gpu-info ConfigMap）

参考 `config/samples/v1_providerconfig.yaml`，依据步骤 1 备份的 `gpu-info` 内容，为每个在用厂商创建 ProviderConfig（镜像、hardwareMetadata、虚拟化模板、`inUseResourceNames`、`devicePluginDetection`）。

```bash
kubectl apply -f providerconfig-nvidia.yaml
```

### 步骤 5：升级控制平面（Operator / Helm）

```bash
helm -n ${ns} upgrade tensor-fusion ./charts/tensor-fusion \
  -f helm-values.yaml \
  --set ...   # 视需要补上 initialGpuNodeLabelSelector 等不再有默认值的项
```

> **替代方式（surgical / 不动 OOO 组件）**：若不想让 greptime / alertmanager / vector / agent 跟着 `helm upgrade` 一起滚动重启，可只用 `kubectl set image` 换 operator 镜像：
> ```bash
> kubectl -n ${ns} set image deploy/tensor-fusion-controller \
>   controller=<registry>/tensor-fusion-operator:<new-tag>
> # hypervisor / worker / client 镜像由 GPUPool/TFC 的 componentConfig 控制，改 TFC 即可
> ```
> 注意：`kubectl set image` 只换镜像，**不会同步 chart 的其它变化**（新增 RBAC、ConfigMap、模板等）。若新版本 chart 给 operator/hypervisor 加了新权限或新 cm，必须手动补齐，否则会缺权限。**因此默认推荐 `helm upgrade`**（OOO 组件只是滚动一次，影响可接受）；仅在严格不允许动数据组件时才用 surgical 方式。

### 步骤 5.5：标注 GPU 节点的隔离模式（soft / hard 隔离必需）

v2 的 hypervisor 隔离模式由**节点标签** `tensor-fusion.ai/isolationMode` 决定（控制器据此给 hypervisor 容器加 `--isolation-mode` 参数）。**不打这个标签时，hypervisor 默认按 `shared` 运行**——即使 Workload 注解写了 `tensor-fusion.ai/isolation: soft`，节点侧的软隔离限流器也不会按 soft 启动。

```bash
# 对每个参与软隔离的 GPU 节点打标签（hard 隔离同理，值改 hard）
kubectl label node <gpu-node> tensor-fusion.ai/isolationMode=soft --overwrite

# 改标签后需重启该节点的 hypervisor 让新参数生效
kubectl -n ${ns} delete pod -l tensor-fusion.ai/component=hypervisor --field-selector spec.nodeName=<gpu-node>
```

> 取值枚举（节点不打标签时 hypervisor 按 `shared` 运行）：
>
> | 值 | 含义 | 说明 |
> |---|---|---|
> | `shared` | **整卡共享** | 多 Workload 时间片共享整张卡，无计算隔离、无开销；但 TFLOPs 限额不生效、无法区分 QoS，可能资源争抢 |
> | `soft` | **软隔离**（Workload 默认） | PID 控制器分时间片限流，可设 QoS、TFLOPs 限额相对准确；约 1% 开销，突发额度耗尽时可能争抢 |
> | `hard` | **硬隔离** | 用专属 SM 划分算力，隔离性更好、无性能开销、可超卖；**仅 Remote 或 Local+SidecarWorker 模式可用，纯 LocalGPU 模式不支持**（没有 TF Worker） |
> | `partitioned` | **MIG / 驱动级分区** | GPU 驱动层硬件分区，完全隔离、无开销、无争抢；需硬件支持，不能超卖 |

### 步骤 6：验证

```bash
kubectl get tensorfusioncluster -A          # Phase=Running
kubectl get gpupool -A                       # 全部 Running
kubectl get gpu -A -o wide                    # status.topology 被重新填充
kubectl get providerconfig                    # 存在且 Vendor 正确
kubectl -n ${ns} logs deploy/tensor-fusion-controller | grep -iE "error|provider|topology"
# 提交一个测试 Workload，确认能正常调度到 GPU
```

---

## 6. 回退方案（Rollback）

> 因为 CRD 是**单存储版本**，回退到旧 schema 时，v2 新增字段（`providerconfigs` 整个 CRD、`gangScheduling`、`partitionTemplateId`、`allocatedPartitions`、`topology` 等）会被**裁剪**。所以回退必须依赖步骤 1 的备份。

### 回退步骤

1. **Helm 回滚控制平面**到 v1：
   ```bash
   helm -n ${ns} history tensor-fusion
   helm -n ${ns} rollback tensor-fusion <上一个 v1 REVISION>
   ```
   或用 v1 chart 显式 `helm upgrade`（注意第 4.1 节的版本号倒挂，可能需 `--force` 或指定本地 v1 chart 路径）。

2. **回退 CRD 定义**到 v1（让 schema 与 v1 控制器匹配）：
   ```bash
   kubectl apply --server-side --force-conflicts -f tf-backup/crd-*.yaml
   ```
   > ⚠️ 此操作会使 v2 专有字段在相关 CR 下次写入时被裁剪。

3. **删除孤立的 v2 资源**（可选）：
   ```bash
   kubectl delete crd providerconfigs.tensor-fusion.ai   # 连带删除其下所有 ProviderConfig CR
   ```

4. **从备份恢复关键 CR 的 spec**（尤其是被迁移过的 GPUPool）：
   ```bash
   kubectl apply -f tf-backup/gpupools.yaml
   kubectl apply -f tf-backup/tensorfusionclusters.yaml
   ```
   > 注意：备份里含 `status`/`resourceVersion`，恢复 spec 时通常只需 apply；如遇冲突，去掉 `status` 与 `resourceVersion`/`uid` 等字段后再 apply。

5. **验证**：`kubectl get tensorfusioncluster -A` 回到 `Running`，Workload 调度正常。

### 回退注意事项

- GPU 的 `status`（`topology`/`allocatedPartitions` 等）回退后由 v1 控制器重新发现，**不需要也不应手工恢复**。
- 若回退前已经有用户在使用 `gangScheduling` / `partitioned` 等 v2 专有能力，回退会导致这些 Workload 配置丢失并行为退化——回退前应先确认没有线上 Workload 依赖 v2 专有字段。
- `gpu-info` ConfigMap 在 v1 仍然是事实来源，回退后确保它存在且内容正确（步骤 1 已备份）。

---

## 7. 风险与检查清单

- [ ] 已完整备份所有 TF CR + CRD 定义 + gpu-info ConfigMap + Helm values
- [ ] 已迁移所有 GPUPool 的 `client` 镜像字段、删除 `nodeDiscovery`、补齐 `nodeManagerConfig`
- [ ] 已为每个在用厂商创建 ProviderConfig，并与旧 gpu-info 内容核对
- [ ] 已确认 main chart 的 version/appVersion 不低于现网（避免 Helm 倒挂）
- [ ] 已显式设置不再有默认值的 `initialGpuNodeLabelSelector`
- [ ] 已先 `kubectl apply` CRD，再 `helm upgrade`
- [ ] 需要 soft/hard 隔离的 GPU 节点已打 `tensor-fusion.ai/isolationMode` 标签，并重启了 hypervisor
- [ ] 升级后 Cluster/Pool=Running，GPU topology 重建，测试 Workload 可调度
- [ ] 已演练过回退路径（建议先在预发环境完整跑一遍升级 + 回退）
