# 跨厂商隔离模式测试矩阵

记录各厂商加速卡在本地 / 远程模式下、各隔离模式的功能与异常恢复测试覆盖情况。

每个已实现的用例覆盖：
1. **正常**：创建业务 Pod（必要时含 worker pod），验证 GPU/NPU 资源就位与正常释放、`gpu` CR 资源恢复。
2. **异常**：创建完成后重启 hypervisor，验证 pod 仍可正常释放、`gpu` CR 资源恢复。

图例：✅ 已实现并通过测试 ｜ TODO 未实现 / 未测试

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
