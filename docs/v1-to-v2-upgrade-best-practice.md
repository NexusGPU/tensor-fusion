# TensorFusion v1 → v2(main) 升级最佳实践

> 适用范围：从 `v1` 分支线升级到 `main` 分支线（v2 架构）。
> **前提**：升级所用的 CRD 必须来自 CRD v1 兼容改动（`feat/crd-v1-compat`）合并之后的 main——该改动把 v1 独有字段（`nvLink`、`model`、`nodeDiscovery`、`remoteModeImage`/`embeddedModeImage` 等）补回了 v2 CRD。**不要使用合并前 main 上的 CRD**，否则 v1 的 spec 数据会被 apiserver 裁剪（pruning），回退时无法恢复。

---

## 核心结论

- CRD 的 group/version 不变，仍为 `tensor-fusion.ai/v1`，单一 served/storage 版本，无 Conversion Webhook，无 K8s API 版本迁移。
- **v2 CRD 是 v1 schema 的严格超集**（逐字段比对验证）：无字段删除，v1 写入的数据在 v2 CRD 下完整保留。
- 回退时 **CRD 保持 v2 不动**，但不能只换 operator 镜像：v1/v2 的 scheduler ConfigMap 和 Hypervisor 镜像必须与各自 operator 配套切换。
- 升级顺序：**先 apply CRD → 同步 RBAC → apply v2 ConfigMap → 切换配套的 operator/Hypervisor 镜像**；控制平面直接改 Deployment 镜像即可，无需 `helm upgrade`。如使用 Helm 管理完整升级，使用 Chart `1.8.0`，并在生产 values 中固定配套的 operator/hypervisor 镜像版本。

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

# nodePoolRollingUpdatePolicy 缺失检查（rhzs 演练实测踩坑）：
# v2 的 component.go isAutoUpdateEnable 对该字段解引用没判 nil，
# pool 缺这块配置时升级后组件配置一旦变化，gpupool reconcile 持续 panic（recovered 但卡死该 pool 的组件更新）
kubectl get gpupools -A -o json \
  | jq -r '.items[] | select(.spec.nodeManagerConfig.nodePoolRollingUpdatePolicy == null)
    | "MISSING-POLICY: \(.metadata.name)"'
```

三条命令输出均为空才继续；有输出则先修正对应对象（policy 缺失的 pool 在 TFC 的 specTemplate 里补上 `nodePoolRollingUpdatePolicy` 块）。

### 3. 全量备份（保险，不再是回退的依赖项）

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
kubectl -n ${ns} get configmap ${release}-config -o yaml > tf-backup/config-v1.yaml
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

**ConfigMap 也必须同步**（rhzs 演练实测）：`<release>-config` 的 `scheduler-config.yaml` 在 v1/v2 间互不兼容。v2 为 `GPUResourcesFit` 增加了 `permit`、`postFilter`、`preBind`、`preEnqueue` 等扩展点，v1 二进制没有实现这些接口；`GPUNetworkTopologyAware` 的参数结构也已改变。换镜像前先 apply 新版 ConfigMap：

```bash
helm template ${release} ./charts/tensor-fusion -n ${ns} \
  -f tf-backup/helm-values.yaml \
  -s templates/config.yaml | kubectl apply -f -
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

### 步骤 3：换 operator 镜像（直接改 Deployment）

先把 NVIDIA pool 示例中的 Hypervisor 镜像改为与 v2 operator 配套、且包含存量 worker 恢复修复的版本。实际路径按集群中的 pool 名调整：

```bash
kubectl patch tensorfusioncluster <cluster-name> --type=json -p='[
  {"op":"replace","path":"/spec/gpuPools/0/specTemplate/componentConfig/hypervisor/image","value":"<registry>/tensor-fusion-hypervisor:<v2-tag>"}
]'
```

```bash
kubectl -n ${ns} set image deploy/${controller_deploy} \
  controller=<registry>/tensor-fusion-operator:<v2-tag>
kubectl -n ${ns} rollout status deploy/${controller_deploy}
```

- hypervisor / worker / client 镜像由 GPUPool / TensorFusionCluster 的 `componentConfig` 控制，改 CR 即可，不需要动 Deployment。不能让 v2 operator 启动 v1 Hypervisor：两代 CLI 参数不兼容，会在 init container 阶段 CrashLoop。
- `INITIAL_GPU_NODE_LABEL_SELECTOR`：v2 chart 去掉了默认值 `nvidia.com/gpu.present=true`，但现网 Deployment 里该 env 已是渲染后的实际值，只换镜像不受影响。
- **清理旧代 hypervisor pod**（rhzs 演练实测）：v1 与 v2 的 hypervisor pod 命名不同（`hypervisor-<node>` vs `tf-hypervisor-<node>`），新 operator 不会接管旧代 pod。同节点新旧两代并存会争抢 device-plugin socket / 共享内存，导致新 hypervisor 反复退出。确认 v2 operator Ready 后，按节点删除旧名字的 `hypervisor-<node>`，等待对应 `tf-hypervisor-<node>` 2/2 Ready；不要用会同时匹配新旧 Pod 的宽泛 selector 一次性删除。
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
    tensor-fusion.ai/gpu-model: "NVIDIA GeForce RTX 3090"
    tensor-fusion.ai/gpupool: "tensor-fusion-shared"
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

绝对 TFLOPS 使用所选 `GPU.status.capacity.tflops` 换算并向上取整。例如 RTX 3090 在 ProviderConfig/GPU CR 中容量为 71 TFLOPS，`10 / 71 * 100` 注入为 `TF_CUDA_SM_PERCENT_LIMIT=15`。不要使用 hypervisor 运行时探测出的理论 TFLOPS 作为换算基数。

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
# 多厂商环境：确认节点 hardware-vendor 标签与所属 pool 的 defaultVendor 一致
# （升级窗口内 GPUPool 被写入时 CRD 默认值可能把 vendor 临时填成 NVIDIA，
#   旧版 operator 会按错值给节点打标，导致 hypervisor 的 soft 限流器选错厂商库）
kubectl get nodes -o custom-columns='NODE:.metadata.name,VENDOR:.metadata.labels.tensor-fusion\.ai/hardware-vendor,ISOLATION:.metadata.labels.tensor-fusion\.ai/isolationMode'
# 提交测试 Workload，并在业务容器中实际调用 CUDA/NVML；仅 Pod Ready 不算通过
```

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

升级失败时，**先恢复 v1 scheduler ConfigMap，再切 v1 operator 镜像**。可直接 apply 升级前备份，或用原 v1 chart 和升级前 values 重新渲染：

```bash
kubectl apply -f tf-backup/config-v1.yaml
# 或：
helm template ${release} <v1-chart-path> -n ${ns} \
  -f tf-backup/helm-values.yaml \
  -s templates/config.yaml | kubectl apply -f -
```

同时把 TensorFusionCluster/GPUPool 中的 Hypervisor 镜像恢复为配套 v1 版本，然后切 operator：

```bash
kubectl patch tensorfusioncluster <cluster-name> --type=json -p='[
  {"op":"replace","path":"/spec/gpuPools/0/specTemplate/componentConfig/hypervisor/image","value":"<registry>/tensor-fusion-hypervisor:<v1-tag>"}
]'
kubectl -n ${ns} set image deploy/${controller_deploy} \
  controller=<registry>/tensor-fusion-operator:<v1-tag>
kubectl -n ${ns} rollout status deploy/${controller_deploy}
```

确认 v1 operator Ready 后，删除残留的 `tf-hypervisor-<node>`，等待 v1 的 `hypervisor-<node>` 2/2 Ready。不能只执行 `rollout undo`：Deployment 会回滚镜像，但不会恢复 v1 ConfigMap和 TensorFusionCluster/GPUPool 中的 Hypervisor 镜像。

- **不要把 v1 旧 CRD apply 回去**。v2 CRD 是超集，v1 operator 在其下完全正常工作；反向 apply 旧 CRD 才会触发 pruning，把 v2 写入的字段（`topology`、`gangScheduling`、`isolationMode` 等）剪掉，影响二次升级。rhzs 演练实测还发现一个更隐蔽的后果：被 prune 的带默认值字段（如 `defaultVendor`）在重新 apply 新 CRD 后会被**默认值静默填错**（Ascend/MooreThreads pool 全变 NVIDIA），且 key 级 diff 看不出来——真发生了只能靠备份做值级比对恢复。
- 回退后 v1 operator 全量更新对象时会抹掉它不认识的 v2-only status 字段——无害，二次升级时 v2 控制器会自动重新发现填充。
- ProviderConfig 等 v2 专有资源及新增的 RBAC 权限回退后闲置即可，无需删除。
- Karpenter NodeOverlay 也不用动：旧 `tensor-fusion.ai/index` key 已按步骤 2.5 保留，v1 worker 的扩容不受影响。

---

## 检查清单

- [ ] CRD 来自 compat 改动合并后的 main（预检 1 通过）
- [ ] `nodeManagerConfig` / `gpuCount` 预检通过（预检 2 输出为空）
- [ ] 已全量备份 CRD、CR、Helm values、v1 ConfigMap 与 controller Deployment
- [ ] 先 `kubectl apply` CRD，再同步 RBAC、apply v2 ConfigMap，最后切换配套 operator/Hypervisor 镜像
- [ ] （Karpenter 环境）NodeOverlay 已声明 `index_0..index_f`，且保留旧 `tensor-fusion.ai/index`
- [ ] GPU 节点已按用途标记 `tensor-fusion.ai/isolationMode=soft|hard|partitioned`（已有 shared label 可保留兼容，但不再要求）
- [ ] 升级后 Cluster/Pool=Running，local/remote 测试 Workload 可调度并能实际调用 CUDA/NVML
- [ ] shared workload 只使用完整空闲、未分区 GPU；hard 百分比/绝对值与显存限额验证通过
- [ ] 存量 soft/hard Pod 的 UID/restartCount/GPU UUID 不变，升级后新 CUDA/NVML 调用正常
- [ ] 回退预案明确：CRD 不动；先恢复 v1 ConfigMap，再切 v1 operator/Hypervisor 镜像并清理 v2 名称的 Hypervisor Pod
