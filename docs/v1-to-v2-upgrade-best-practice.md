# TensorFusion v1 → v2(main) 升级最佳实践

> 适用范围：从 `v1` 分支线升级到 `main` 分支线（v2 架构）。
> **前提**：升级所用的 CRD 必须来自 CRD v1 兼容改动（`feat/crd-v1-compat`）合并之后的 main——该改动把 v1 独有字段（`nvLink`、`model`、`nodeDiscovery`、`remoteModeImage`/`embeddedModeImage` 等）补回了 v2 CRD。**不要使用合并前 main 上的 CRD**，否则 v1 的 spec 数据会被 apiserver 裁剪（pruning），回退时无法恢复。

---

## 核心结论

- CRD 的 group/version 不变，仍为 `tensor-fusion.ai/v1`，单一 served/storage 版本，无 Conversion Webhook，无 K8s API 版本迁移。
- **v2 CRD 是 v1 schema 的严格超集**（逐字段比对验证）：无字段删除，v1 写入的数据在 v2 CRD 下完整保留。
- 因此**回退 = 只换回 v1 镜像，CRD 保持 v2 不动**。无数据丢失，无需恢复备份。
- 升级顺序：**先 apply CRD → 同步 RBAC → 再换 operator 镜像**；控制平面直接改 Deployment 镜像即可，无需 `helm upgrade`。

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

### 3. 轻量备份（保险，不再是回退的依赖项）

```bash
ns=tensor-fusion  # 改成实际 Helm release/operator 所在 namespace
mkdir -p tf-backup
for kind in tensorfusionclusters gpupools workloadprofiles schedulingconfigtemplates; do
  kubectl get ${kind}.tensor-fusion.ai -A -o yaml > tf-backup/${kind}.yaml
done
kubectl -n ${ns} get deploy -l tensor-fusion.ai/component=operator -o yaml > tf-backup/controller-deploy.yaml
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
- **hypervisor Role**（`rbac-hypervisor.yaml`）：`tensor-fusion.ai` 资源列表需要包含 `providerconfigs`，并保留模板里的 `get/list/watch/create/update/patch` verbs

可以直接用新版 chart 渲染后单独 apply 这两个文件：

```bash
helm template tensor-fusion ./charts/tensor-fusion -n ${ns} \
  -s templates/rbac.yaml -s templates/rbac-hypervisor.yaml | kubectl apply --server-side -f -
```

**ConfigMap 也必须同步**（rhzs 演练实测）：`<release>-config` 的 scheduler-config 在 v1/v2 间互不兼容——v2 启用了 `permit`/`postFilter` 扩展点（v1 的 GPUResourcesFit 没实现这两个接口，调度器直接起不来）；v1 配置里有 `enableNvlinkAware` 等 v2 不认的插件参数。换镜像前先 apply 新版 CM：

```bash
helm template tensor-fusion ./charts/tensor-fusion -n ${ns} \
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

v2 需要提前规划每个 GPU 节点承担的隔离模式，并在 **Kubernetes Node** 上打 label：

```bash
kubectl label node <node-name> tensor-fusion.ai/isolationMode=soft --overwrite
# 可选值：shared / soft / hard / partitioned
```

这不是业务 Pod 上的 `tensor-fusion.ai/isolation` annotation。两者职责不同：

| 位置 | Key | 作用 |
|---|---|---|
| Node label | `tensor-fusion.ai/isolationMode` | operator 同步到 GPUNode，并作为该节点 hypervisor 的 `--isolation-mode=<mode>` 启动参数；hypervisor 上报 GPU.status.isolationMode |
| Pod annotation | `tensor-fusion.ai/isolation` | workload 请求的隔离模式；scheduler/allocator 会按 GPU.status.isolationMode 过滤 |

因此要保证节点规划和 workload 请求一致：soft workload 只能稳定落到 soft 节点，hard workload 落到 hard 节点，partitioned workload 落到 partitioned 节点。未规划时不要依赖默认值，尤其是默认 workload isolation 为 `soft`，但 hypervisor 侧默认参数不是升级策略的一部分，可能导致升级后调度过滤无可用 GPU。

建议升级前按 pool/节点用途一次性标好：

```bash
kubectl label node <soft-node> tensor-fusion.ai/isolationMode=soft --overwrite
kubectl label node <hard-node> tensor-fusion.ai/isolationMode=hard --overwrite
kubectl label node <partition-node> tensor-fusion.ai/isolationMode=partitioned --overwrite
kubectl get nodes -L tensor-fusion.ai/isolationMode
```

后续修改该 label 时，v2 operator 会删除并重建对应节点的 hypervisor pod 以应用新的 `--isolation-mode`，应按维护变更处理，避免和运行中 worker/业务混在同一个升级动作里。

### 步骤 3：换 operator 镜像（直接改 Deployment）

```bash
kubectl -n ${ns} set image deploy/tensor-fusion-controller \
  controller=<registry>/tensor-fusion-operator:<v2-tag>
kubectl -n ${ns} rollout status deploy/tensor-fusion-controller
```

- hypervisor / worker / client 镜像由 GPUPool / TensorFusionCluster 的 `componentConfig` 控制，改 CR 即可，不需要动 Deployment。
- `INITIAL_GPU_NODE_LABEL_SELECTOR`：v2 chart 去掉了默认值 `nvidia.com/gpu.present=true`，但现网 Deployment 里该 env 已是渲染后的实际值，只换镜像不受影响。
- **清理旧代 hypervisor pod**（rhzs 演练实测）：v1 与 v2 的 hypervisor pod 命名不同（`hypervisor-<node>` vs `tf-hypervisor-<node>`），新 operator 不会接管旧代 pod。同节点新旧两代并存会争抢 device-plugin socket / 共享内存，导致新 hypervisor 反复退出——换镜像后手动删掉旧代 pod：`kubectl -n ${ns} delete pod -l 'tensor-fusion.ai/component in (hypervisor)'` 后由新 operator 重建，或按节点逐个删旧名字的 pod。

### 步骤 4：升级后的功能迁移（不阻塞升级，按需进行）

v2 operator 不再读取以下 v1 字段（字段保留仅为兼容与回退，数据不会丢，但功能上由新机制接管）：

| v1 字段 | v2 接管方式 |
|---|---|
| `client.remoteModeImage` / `embeddedModeImage` | 合并为 `client.image`（ProviderConfig 存在时以其为准） |
| `componentConfig.nodeDiscovery` | 节点发现已内置 |
| `gpu.status.nvLink` / `model` | `status.topology`（控制器自动重建）/ ProviderConfig.hardwareMetadata |
| `gpu-info` ConfigMap | ProviderConfig CRD（按厂商创建，参考 `config/samples/v1_providerconfig.yaml`，NVIDIA 默认配置可用 chart 模板 `templates/provider-config-nvidia.yaml` 渲染） |

### 步骤 5：验证

```bash
kubectl get tensorfusioncluster -A    # Phase=Running
kubectl get gpupool -A                # 全部 Running
kubectl get gpu -A -o wide            # status 正常更新
# 多厂商环境：确认节点 hardware-vendor 标签与所属 pool 的 defaultVendor 一致
# （升级窗口内 GPUPool 被写入时 CRD 默认值可能把 vendor 临时填成 NVIDIA，
#   旧版 operator 会按错值给节点打标，导致 hypervisor 的 soft 限流器选错厂商库）
kubectl get nodes -o custom-columns='NODE:.metadata.name,VENDOR:.metadata.labels.tensor-fusion\.ai/hardware-vendor,ISOLATION:.metadata.labels.tensor-fusion\.ai/isolationMode'
# 提交一个测试 Workload，确认正常调度
```

---

## 回退方案（Rollback）

升级失败时：

```bash
# 只把镜像换回 v1，CRD 保持 v2 不动
kubectl -n ${ns} set image deploy/tensor-fusion-controller \
  controller=<registry>/tensor-fusion-operator:<v1-tag>
# 或直接 rollout undo
kubectl -n ${ns} rollout undo deploy/tensor-fusion-controller
```

- **不要把 v1 旧 CRD apply 回去**。v2 CRD 是超集，v1 operator 在其下完全正常工作；反向 apply 旧 CRD 才会触发 pruning，把 v2 写入的字段（`topology`、`gangScheduling`、`isolationMode` 等）剪掉，影响二次升级。rhzs 演练实测还发现一个更隐蔽的后果：被 prune 的带默认值字段（如 `defaultVendor`）在重新 apply 新 CRD 后会被**默认值静默填错**（Ascend/MooreThreads pool 全变 NVIDIA），且 key 级 diff 看不出来——真发生了只能靠备份做值级比对恢复。
- 回退后 v1 operator 全量更新对象时会抹掉它不认识的 v2-only status 字段——无害，二次升级时 v2 控制器会自动重新发现填充。
- ProviderConfig 等 v2 专有资源及新增的 RBAC 权限回退后闲置即可，无需删除。
- Karpenter NodeOverlay 也不用动：旧 `tensor-fusion.ai/index` key 已按步骤 2.5 保留，v1 worker 的扩容不受影响。

---

## 检查清单

- [ ] CRD 来自 compat 改动合并后的 main（预检 1 通过）
- [ ] `nodeManagerConfig` / `gpuCount` 预检通过（预检 2 输出为空）
- [ ] 已轻量备份 CR 与 controller Deployment
- [ ] 先 `kubectl apply` CRD，再同步 RBAC，最后换镜像
- [ ] （Karpenter 环境）NodeOverlay 已声明 `index_0..index_f`，且保留旧 `tensor-fusion.ai/index`
- [ ] GPU 节点已按用途标记 `tensor-fusion.ai/isolationMode=shared|soft|hard|partitioned`
- [ ] 升级后 Cluster/Pool=Running，测试 Workload 可调度
- [ ] 回退预案明确：只回滚镜像，CRD 不动
