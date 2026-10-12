# 真实 GPU 上线执行单

GPU 配额到位后，按本文从头到尾执行一遍即可。命令都在 `e2e/` 目录下执行。
写于 2026-10-12：此时账号的 G/VT 配额（按需、spot）都是 0，提额申请在 AWS 人工审核中，**下面的 GPU 步骤没有任何一步被执行过**。

## 0. 前置检查（2 分钟，$0）

```bash
export AWS_PROFILE=tf-e2e
cd e2e
make preflight
make quotas          # L-DB2E81BA（按需 G/VT）与 L-3819A6DF（spot G/VT）至少 8，建议 16；L-1216C47A（标准按需）≥ 8
```

| 配额 | 最低 | 说明 |
|---|---|---|
| `L-DB2E81BA` 按需 G/VT | 8 vCPU | 2 台 g4dn.xlarge；`karpenter-finops` 的整卡基线要 3 台（12）；路径 B 再加 1 台 g4dn.2xlarge（8） |
| `L-3819A6DF` spot G/VT | 0 也能跑 | 为 0 时 Karpenter 自动回退按需；有 8–16 时大部分节点会是 spot，成本减半 |
| `L-1216C47A` 标准按需 | 8 vCPU | 当前就是 8，够用 |

只有 8 个按需 G vCPU 时：`make up TF_ARGS='-var gpu_node_limit=2'`，并跳过 `FIN-01` 的整卡基线（`FOCUS` 里去掉它）和 `KRP-B02`。

## 1. 起环境（14–15 分钟，约 $0.05）

```bash
# 用修复后的镜像时，把 tag 作为参数传入（任何镜像引用都是参数，仓库里不写死）
export TF_RELEASE_TAG=<operator 与 hypervisor 的 tag>        # 不设 = 2.17.23（已发布）
export TF_VGPU_PROVIDER_TAG=<vgpu-provider-nvidia 的 tag>    # 不设 = 1.3.9
# 可选：注册到控制台（值只放在当前 shell 里）
# export TF_AGENT_ID=… TF_ENROLL_TOKEN=… TF_CLOUD_ENDPOINT=…

make up TTL=6h
export KUBECONFIG=$PWD/terraform/.kube/config
make mock-uninstall    # 之前做过演练的话：mock pool 留着会让真实 pool 的 Pod 拉起 mock 节点
make test-smoke        # 约 7 分钟；先确认控制面与 Karpenter 正常
```

镜像已包含 pricing 初始化与 hypervisor 指标修复时，再加 `export E2E_FIXED_IMAGES=1`（放开 `KRP-B02`）。`OBS-02`、`WHK-08`、`ISO-14`、`KRP-A15` 在修复前会报 `known-fail`，修复后应转为通过。

## 2. 各档位（每档一条命令）

| 顺序 | 命令 | 需要 | 预计时长 | 预计成本（含集群空转 $0.19/h） |
|---|---|---|---|---|
| 1 | `make test-core` | 1–2 台 g4dn.xlarge | 40 分钟 | 按需 $1.0 ／ spot $0.5 |
| 2 | `make test-isolation` | 1 台 g4dn.xlarge | 50 分钟 | 按需 $0.7 |
| 3 | `make test-karpenter-finops` | 最多 3 台 g4dn.xlarge（+ 路径 B 1 台 g4dn.2xlarge） | 90 分钟 | 按需 $1.7 ／ spot $0.9 |
| 4 | `make test-perf-quick` | 1 台 g4dn.xlarge | 60 分钟（首轮含 vLLM 镜像拉取，压缩后 11.9GB） | 按需 $0.75 |
| 5（可选） | `E2E_PERF_PROFILE=full make test FILTER=perf FOCUS='PERF-LLM-0[12]'` | 1 台 | 50 分钟 | 按需 $0.6 |
| 6（可选，换 GPU 型号做跨硬件确认） | `make test-perf-quick` 前给 NodePool 只留 `g5.xlarge` 或 `g6.xlarge`：`TF_VAR_gpu_instance_types='["g5.xlarge"]' make up` | 1 台 g5.xlarge（A10G）/ g6.xlarge（L4） | 60 分钟 | g5 $1.2 ／ g6 $1.0 |
| 最后 | `make test FILTER=disruptive`（`UPG-03` 卸载）→ `make cost` → `make down` | — | 5 + 12 分钟 | $0.06 |

全量（1–4 + 收尾）：约 **4.7 小时，按需约 $4.4**，有 spot 配额约 $2.7。加上第 5、6 步约 $6.2。预算 $100。
时长和成本是按用例里的固定等待与测量窗口算出来的，没有真实 GPU 运行数据；mock 演练档实测约 75 分钟。

过程中：

```bash
make deadman           # 自毁开关还剩多久
make extend TTL=3h     # 不够就续
make cost              # 花了多少
make status            # 节点 / NodeClaim / GPU
```

每档结束看 `results/<run-id>/summary.md`。失败的用例单独重跑：`make test FILTER=core FOCUS='ISO-01'`。全部跑完后：

```bash
make report RUNS="results/<core> results/<isolation> results/<karpenter> results/<perf>"
```

裸金属（2×RTX 3090）用同一套命令加 `KUBECONFIG=…`；把裸金属与 EKS 的结果目录一起传给 `make report` 就是并排对比。

## 3. 演练已经证明了什么（真实 AWS，2026-10-11 ～ 12）

以下都是在真实账号上执行得到的，模式标注在括号里。

### 3.1 环境与成本安全

- `make up`：第 2、3 轮各一次成功，13m43s 与 15m21s（EKS 控制面 9–10.5 分钟，节点组约 2 分钟，Karpenter 与 NodePool 约 1 分钟，TensorFusion 105s）。`make down`：节点还在时 12m07s，已被自毁开关清掉时 5–6 分钟。三轮的逐轮记录（含第 1 轮 apply 重跑、第 2 轮残留 2 个 launch template 的原因与修复）见 `terraform/README.md`。
- 自毁开关（真实触发两次，TTL 2 分钟）：过期后 5m58s、8m12s 全部计算资源消失，包括 Karpenter 在第一次清理后重新拉起的那台。
- 最后一次销毁后 `sweep` 为空；另用直接的 describe 调用核对过：区域内 0 实例、0 卷、0 ENI、0 EKS 集群、0 非默认 VPC、0 Lambda、0 日志组、0 SSM 参数、0 launch template、0 个 `tf-e2e*` IAM 角色。
- 花费：三轮合计估算约 $1.1。

### 3.2 GPU 启动路径：只差配额

- `make gpu-dryrun`：EKS 1.36 的 NVIDIA AL2023 AMI 可解析（`amazon-eks-node-al2023-x86_64-nvidia-1.36-v20261003`）；g4dn.xlarge / g4dn.2xlarge / g5.xlarge / g6.xlarge × 两个 AZ × 按需 / spot 共 16 种组合，`ec2 run-instances --dry-run` 全部返回「would launch」（参数、子网、安全组、Karpenter 的 instance profile、标签、卷都被接受）。dry-run 不校验配额。
- 真实的 NodeClaim 尝试（mode `dry-run`）：一个不带 nodeSelector 的 TF Pod（只有 label + annotation）Pending 后，Karpenter 为 **`tf-gpu` NodePool** 提名并创建了 NodeClaim，EC2 的回答是：
  - spot：`InsufficientCapacityError … UnfulfillableCapacity … MaxSpotInstanceCountExceeded: Max spot instance count exceeded`
  - 随后自动换按需：`InsufficientCapacityError … VcpuLimitExceeded: You have requested more vCPU capacity than your current vCPU limit of 0 allows for the instance bucket that the specified instance type belongs to.`
  
  这同时说明 spot → on-demand 的回退顺序是通的。配额到位后，这一步之后的部分（节点加入、hypervisor、GPU 注册）就是 mock 上已经跑通的那条链路，区别只在设备是真的。

### 3.3 控制链路（mock GPU，`rehearsal` 档）

`rehearsal` 档在 mock GPU 上执行（EKS 1.36 + Karpenter 1.14.1 + operator 2.17.23，节点是 t3.medium spot），逐条结果见 `README.md` 9.3。要点：

- **冷启动（路径 A）**：Pending 的分片 Pod → Karpenter NodeClaim 6s → 节点加入 → hypervisor 注册 mock GPU 51s → GPU / GPUNode CR → Pod Running 56s。
- **扩容（路径 A 核心闭环）**：GPU 占满后再建 Pod → 扩容器 12.5s 建出 NodeClaim（owner 为 NodePool、带 `expansion-source`、不带 `do-not-disrupt`）→ 节点 → Pod Running。**但有竞态（G-16）**：Pod 要等调度器 5 分钟的兜底重试才绑定（实测 330s）；`consolidateAfter` ≤ 5m 时节点会先被 Karpenter 回收。
- **装箱**：3 个 Pending 的分片 Pod 只新增 1 个 NodeClaim，落在同一节点同一张卡；两个节点时 4 个小 Pod 全在一个节点。
- **缩容**：空节点先被 TF 打 `should-delete`，随后由 Karpenter consolidation 删除，GPUNode / GPU CR 一并清理。
- **spot → on-demand**：spot 配额用尽时扩容器的 NodeClaim 失败后自动换到 on-demand。
- **节点删除**：GPUNode / GPU CR 清理，Deployment 的 Pod 59s 后在新节点重建。
- **恢复**：hypervisor 重启 23s、operator 重启 21s，分配状态不丢、无重复分配。
- **FinOps**：3 个分片负载用 1 台节点，整卡基线用 3 台（成本按 g4dn.xlarge 单价代入）。
- **路径 B 没有演练**：发布镜像上不可用（G-08），而且 mock 无法覆盖（机型取自内置 GPU 价格表、NodeClaim 请求 `nvidia.com/gpu`）。

演练中确认的产品问题在真实 GPU 上同样会出现。

### 3.4 需要 GPU 的用例做到了哪一步（`dryrun` 档）

- `DRY-ISO-01`：`isolation`、`perf` 两档要创建的 11 份清单（soft / hard / shared 探针、vLLM soft / hard / eager、ONNX Runtime、TorchBench、MLPerf、原生基线）全部通过 server-side dry-run。
- `DRY-IMG-01`：所有镜像在 registry 上可解析（PyTorch 3161 MiB、vLLM 11877 MiB，linux/amd64 压缩大小）；Qwen2.5-0.5B / 1.5B 不需要 token。
- `DRY-PERF-01`：LLM 压测端、指标采集、判定逻辑、报告在 CPU stand-in 上端到端跑通，并且对「平滑按比例 / 500ms 阶梯放行 / 1s 停顿 / 不限流」四种行为给出了正确的通过与失败结论。
- `DRY-PERF-02`：ONNX Runtime、TorchBench-mini、MLPerf LoadGen 三个脚本在 CPU 上完成导出、建会话、限时测量、结果解析。
- 没有做到的：这些脚本的 CUDA 路径（`CUDAExecutionProvider`、`torch.cuda`、vLLM 本体）一行都没有执行过。

## 4. 还没有被证明的（第一次真实运行时重点看）

1. **所有与 CUDA 有关的结论**：容器内 GPU 是否可用、显存与算力限额是否生效、邻居隔离、运行期改限额、性能随份额的比例性与尾延迟。`isolation`、`perf` 两档的断言阈值是设计值，首轮结果很可能需要校准阈值或修测试，不要把首轮失败直接当成产品缺陷。
2. **真实 GPU 节点的时间尺度**：NVIDIA AMI 启动、hypervisor 与 `vgpu-provider` 镜像拉取、业务镜像（PyTorch、vLLM，压缩后分别约 3161 MiB、11877 MiB）拉取。mock 节点从 Pending 到 Pod Running 约 1 分钟，真实 GPU 节点预计 3–6 分钟。
   **路径 A 的扩容有一个竞态**：新节点的 GPU 注册好之后，之前调度失败的 Pod 不会被立刻重新入队，要等调度器 5 分钟的兜底重试；这段时间新节点在 Karpenter 看来是空的。演练里 `consolidateAfter` 为 1m、5m 时各出现过一次「节点先被回收、Pod 一直 Pending」。本栈因此把 GPU / mock NodePool 的 `consolidateAfter` 默认设为 10m。`KRP-A02` 记录 `node_initialized_to_pod_bound_seconds`，`KRP-A16` 对它断言（当前 known-fail）。真实 GPU 上若 `KRP-A02` 仍然不稳定，先看这个数字。
3. **路径 B**（`provisioningMode: Karpenter`）：发布镜像上不可用，mock 上无法演练（机型取自内置 GPU 价格表）。等修复镜像后第一次执行。
4. **hypervisor 指标入库**（`OBS-02`）：mock 上确认 `tf_node_metrics`、`tf_pool_metrics`、`tf_worker_resources` 有数据，`tf_gpu_usage`、`tf_worker_usage` 没有（已知缺陷）。修复镜像后应全部到齐。
5. **remote 模式**（`ISO-06`）、**hard 隔离**（`ISO-03`、`ISO-04`、`PERF-LLM-02`）：需要 NVIDIA 的 worker，mock 上跑不了。
6. **原生基线**（mode `native`）依赖节点的默认 container runtime 是 nvidia（EKS AL2023 NVIDIA AMI 应该是）；不是的话设 `E2E_NATIVE_RUNTIME_CLASS=nvidia`。
7. **vLLM 镜像 tag**（`vllm/vllm-openai:v0.11.0`）与 PyTorch 镜像只验证了 registry 上存在，没有在 T4 上实际启动过；T4 不支持 bf16，脚本已用 `--dtype half`。
8. **TorchBench 的 `--impl torchbench`**（真正的 pytorch/benchmark harness）没有运行过，只在裸金属的 full 档使用。
9. `make cost` 里 Cost Explorer 按标签的数字：标签要被 AWS 账单发现约 24 小时后才能激活，本次演练结束时还是 $0，实时估算部分可用。

## 5. 出问题时

| 现象 | 处理 |
|---|---|
| AWS 调用报 session 过期 | `aws sso login --profile tf-e2e`，然后重跑刚才的命令（`make up` / `make down` 都可重入） |
| `terraform destroy` 中途失败 | 直接再跑一次 `make down`；还不行用 `make sweep` 看残留、`make sweep-delete` 清掉 |
| 人走了、agent 死了 | 不用管。自毁开关到点会把节点组缩到 0、终止所有实例；再过 6 小时删掉控制面。之后 `make down` 收尾 |
| 自毁开关已经触发，想继续 | `make up TTL=3h`（先续期，再把节点组恢复） |
| GPU NodeClaim 一直 `InsufficientCapacityError` | `make quotas`；或 `TF_VAR_gpu_capacity_types='["on-demand"]' make up` |
| 某个用例把 namespace 留下了 | `kubectl delete ns -l tf-e2e/run` |
