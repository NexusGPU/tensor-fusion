# 跨厂商隔离模式测试矩阵

记录各厂商加速卡在本地 / 远程模式下、各隔离模式的功能与异常恢复测试覆盖情况。

每个已实现的用例覆盖：
1. **正常**：创建业务 Pod（必要时含 worker pod），验证 GPU/NPU 资源就位与正常释放、`gpu` CR 资源恢复。
2. **异常**：创建完成后重启 hypervisor，验证 pod 仍可正常释放、`gpu` CR 资源恢复。

图例：✅ 已实现并通过测试 ｜ TODO 未实现 / 未测试

---

## 注解与标签约定

下面所有用例的注解都写在 user pod 的 `metadata.annotations` 上。webhook 进来后做注解 → 内部 spec 解析，并把若干字段镜像到 label 给 scheduler / controller selector 使用。

### user 必须写的 label

| 标签 | 必需性 | 说明 |
|---|---|---|
| `tensor-fusion.ai/enabled: 'true'` | 通常必需 | 唯一显式触发 webhook 注入的开关（[reconcile.go:214](internal/utils/reconcile.go#L214) 的 `IsTensorFusionPod`）。**例外**：若 pod 已经在 container `resources.limits` 里写了 GPU 资源（`nvidia.com/gpu` 等），且集群开了 auto-migration (`globalConfig.AutoMigration.Enable=true`)，webhook 会自动加这个 label，不必手写。 |
| `tensor-fusion.ai/enabled: 'false'` | 仅显式禁用时 | 强制跳过 TensorFusion 注入，即使匹配 auto-migration 规则也不动。 |

### user 不要写的 label（webhook 自动镜像 / 注入）

| 标签 | 来源 | 用途 |
|---|---|---|
| `tensor-fusion.ai/gpupool: <name>` | webhook 从 annotation 复制 ([pod_webhook.go:568](internal/webhook/v1/pod_webhook.go#L568)) | scheduler / 控制器 label selector |
| `tensor-fusion.ai/component: client \| worker` | webhook 注入 ([pod_webhook.go:566](internal/webhook/v1/pod_webhook.go#L566)) | 区分业务 Pod 与 worker Pod |

worker pod 还会被控制器额外加 `tensor-fusion.ai/workload`、`tensor-fusion.ai/template-hash`，由 operator 管理，user 不必关心。

### user 必须写的 annotation（不写会失败 / 走默认）

| 注解 | 行为 |
|---|---|
| `tensor-fusion.ai/gpupool: <name>` | 只在 annotation 上读 ([tf_parser.go:589](internal/webhook/v1/tf_parser.go#L589))，不写就回退到 default pool；找不到 default 报 `gpu pool not found`。 |
| `tensor-fusion.ai/vendor: <X>` | 显式指定厂商 ([tf_parser.go:549-552](internal/webhook/v1/tf_parser.go#L549-L552))。**可省**：当 container 在 `resources.limits` 里写了厂商专属资源名（如 `tensor-fusion.ai/ascend-npu`），webhook 会从资源名反推；都没写则空字符串走 Nvidia 默认行为，**非 Nvidia 卡上必填**。 |
| `tensor-fusion.ai/isolation: shared\|soft\|hard\|partitioned` | 隔离模式，决定是否注入 limiter / sidecar / 走分区 |
| `tensor-fusion.ai/is-local-gpu: 'true' \| 'false'` | 本地 / 远程模式 |

其余如 `compute-percent-*`、`vram-*`、`gpu-count`、`gpu-model`、`partition-id`、`sidecar-worker`、`dedicated-gpu` 都是按用例选填，下面每个 ✅ 用例样本里写明。

---

## 总览

### 本地模式

| 厂商 | shared (整卡) | soft | hard | partitioned |
|---|---|---|---|---|
| 华为 Ascend | ✅ | TODO | TODO | ✅ |
| 沐曦 MetaX | TODO | TODO | TODO | TODO |
| 海光 Hygon | TODO | TODO | TODO | TODO |
| NVIDIA | ✅ | ✅ | ✅ | ✅ (MIG) |

### 远程模式

| 厂商 | shared (整卡) | soft | hard | partitioned |
|---|---|---|---|---|
| 华为 Ascend | ✅ | TODO | TODO | TODO |
| 沐曦 MetaX | TODO | TODO | TODO | TODO |
| 海光 Hygon | TODO | TODO | TODO | TODO |
| NVIDIA | ✅ | ✅ | ✅ | TODO |

---

## 已通过用例的注解与镜像样本

### 华为 Ascend

#### 本地 · partitioned

```yaml
tensor-fusion.ai/compute-percent-limit: '50'
tensor-fusion.ai/compute-percent-request: '10'
tensor-fusion.ai/gpu-count: '1'
tensor-fusion.ai/gpu-model: Ascend 310P3
tensor-fusion.ai/gpupool: tensor-fusion-shared
tensor-fusion.ai/is-local-gpu: 'true'
tensor-fusion.ai/isolation: partitioned
tensor-fusion.ai/vendor: Ascend
```

镜像：`docker.m.daocloud.io/openeuler/vllm-ascend:0.11.0rc0-torch_npu2.5.1-cann8.1.rc1-python3.10-oe2203lts`

支持两种分区指定方式：
- 通过 `compute-percent-request` 自动匹配模板
- 通过 `tensor-fusion.ai/partition-id: vir04` 显式指定模板（指定模板时优先于 request，直接按模板创建）

验证手段：`npu-smi info -t info-vnpu -i 2 -c 0` 可看到对应 vNPU；删除时 vNPU 自动释放。

#### 本地 · shared

```yaml
tensor-fusion.ai/dedicated-gpu: 'true'
tensor-fusion.ai/gpu-count: '1'
tensor-fusion.ai/gpu-model: Ascend 310P3
tensor-fusion.ai/gpupool: tensor-fusion-shared
tensor-fusion.ai/is-local-gpu: 'true'
tensor-fusion.ai/isolation: shared
tensor-fusion.ai/vendor: Ascend
```

镜像：`docker.m.daocloud.io/openeuler/vllm-ascend:0.11.0rc0-torch_npu2.5.1-cann8.1.rc1-python3.10-oe2203lts`

#### 远程 · shared

```yaml
tensor-fusion.ai/dedicated-gpu: 'true'
tensor-fusion.ai/gpu-count: '1'
tensor-fusion.ai/gpu-model: Ascend 310P3
tensor-fusion.ai/gpupool: tensor-fusion-shared
tensor-fusion.ai/is-local-gpu: 'false'
tensor-fusion.ai/isolation: shared
tensor-fusion.ai/vendor: Ascend
```

镜像：`docker.m.daocloud.io/openeuler/vllm-ascend:0.11.0rc0-torch_npu2.5.1-cann8.1.rc1-python3.10-oe2203lts`

### NVIDIA

#### 本地 · shared

```yaml
tensor-fusion.ai/dedicated-gpu: 'true'
tensor-fusion.ai/gpu-count: '1'
tensor-fusion.ai/gpu-model: NVIDIA GeForce RTX 3090
tensor-fusion.ai/gpupool: tensor-fusion-nvidia-shared
tensor-fusion.ai/is-local-gpu: 'true'
tensor-fusion.ai/isolation: shared
tensor-fusion.ai/vendor: NVIDIA
```

镜像：`registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime`

#### 本地 · soft

```yaml
tensor-fusion.ai/compute-percent-limit: '30'
tensor-fusion.ai/compute-percent-request: '20'
tensor-fusion.ai/gpu-count: '2'
tensor-fusion.ai/gpupool: tensor-fusion-nvidia-shared
tensor-fusion.ai/is-local-gpu: 'true'
tensor-fusion.ai/isolation: soft
tensor-fusion.ai/vendor: NVIDIA
tensor-fusion.ai/vram-limit: 12Gi
tensor-fusion.ai/vram-request: 512Mi
```

镜像：`registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime`

#### 本地 · hard

```yaml
tensor-fusion.ai/compute-percent-limit: '70'
tensor-fusion.ai/compute-percent-request: '60'
tensor-fusion.ai/gpu-count: '1'
tensor-fusion.ai/gpupool: tensor-fusion-nvidia-shared
tensor-fusion.ai/is-local-gpu: 'true'
tensor-fusion.ai/isolation: hard
tensor-fusion.ai/sidecar-worker: 'true'
tensor-fusion.ai/vendor: NVIDIA
tensor-fusion.ai/vram-limit: 2Gi
tensor-fusion.ai/vram-request: 1Gi
```

镜像：`registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime`

#### 本地 · partitioned (MIG，仅本地模式)

```yaml
tensor-fusion.ai/compute-percent-limit: "43"
tensor-fusion.ai/compute-percent-request: "10"
tensor-fusion.ai/gpu-count: "1"
tensor-fusion.ai/gpu-model: NVIDIA A100-SXM4-80GB
tensor-fusion.ai/gpupool: tensor-fusion-nvidia-shared
tensor-fusion.ai/is-local-gpu: "true"
tensor-fusion.ai/isolation: partitioned
tensor-fusion.ai/partition-id: "9"
tensor-fusion.ai/vendor: NVIDIA
```

镜像：`registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime`

#### 远程 · shared

```yaml
tensor-fusion.ai/dedicated-gpu: 'true'
tensor-fusion.ai/gpu-count: '1'
tensor-fusion.ai/gpu-model: NVIDIA GeForce RTX 3090
tensor-fusion.ai/gpupool: tensor-fusion-nvidia-shared
tensor-fusion.ai/is-local-gpu: 'false'
tensor-fusion.ai/isolation: shared
tensor-fusion.ai/vendor: NVIDIA
```

镜像：`registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime`

#### 远程 · hard

```yaml
tensor-fusion.ai/compute-percent-limit: '70'
tensor-fusion.ai/compute-percent-request: '60'
tensor-fusion.ai/gpu-count: '2'
tensor-fusion.ai/gpupool: tensor-fusion-nvidia-shared
tensor-fusion.ai/is-local-gpu: 'false'
tensor-fusion.ai/isolation: hard
tensor-fusion.ai/vendor: NVIDIA
tensor-fusion.ai/vram-limit: 10Gi
tensor-fusion.ai/vram-request: 5Gi
```

镜像：`registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime`

#### 远程 · soft

```yaml
tensor-fusion.ai/compute-percent-limit: '30'
tensor-fusion.ai/compute-percent-request: '20'
tensor-fusion.ai/gpu-count: '2'
tensor-fusion.ai/gpupool: tensor-fusion-nvidia-shared
tensor-fusion.ai/is-local-gpu: 'false'
tensor-fusion.ai/isolation: soft
tensor-fusion.ai/vendor: NVIDIA
tensor-fusion.ai/vram-limit: 12Gi
tensor-fusion.ai/vram-request: 512Mi
```

镜像：`registry.cn-hangzhou.aliyuncs.com/tensorfusion/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
