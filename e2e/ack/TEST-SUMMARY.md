# ACK（阿里云）真实 GPU 端到端测试总结

2026-10-12 ～ 13，阿里云 us-east-1，ACK 托管版 Kubernetes 1.36.2，Karpenter 阿里云 provider 0.2.5（带本仓库的补丁），GPU 节点 `ecs.gn6i-c4g1.xlarge` / `c8g1.2xlarge`（各 1 张 Tesla T4），TensorFusion chart 为仓库 checkout。这是 `e2e/README.md` 里标为「未执行」的 GPU 档位第一次在真实 GPU 上运行。

环境说明见 [`README.md`](README.md)，手工复现步骤见 [`MANUAL-TEST-CASES.md`](MANUAL-TEST-CASES.md)，性能数字的图表版在 `../perf-history/index.html`。

## 1. 结论

- **兼容模式（路径 A：`AutoSelect` + Karpenter NodePool）在阿里云上完整可用。** 冷启动、扩容器闭环、一次扩容接多个 Pod、N 个整卡 Pod 恰好扩 N 台、缩容、节点删除后重建、FinOps 对比，全部通过。前提是给 provider 打一个补丁（见 3.1），否则一个 Pending Pod 会拉起 4 台 GPU 节点。
- **路径 B（`provisioningMode: Karpenter`）修复后可用。** 修复前 operator 的机型目录里没有阿里云机型、也读不了 `ECSNodeClass`，池子永远建不出 GPUNodeClaim（#820，已随 v2.17.30 发布）。
- **soft 隔离的核心行为成立**：一卡多 Pod、显存上限、算力限额、邻居隔离、运行期改限额（修复 #822 后）。算力限额控制的是设备忙时比例，精度很高（目标 50% / 25% 时实测 53% / 26%）；吞吐与忙时不成线性，见 4.2。
- **hard 隔离与 remote 模式在开源栈上不可用**：community 版 worker 启动时报 `License is not set` 后退出，chart 与 operator 没有下发 License 的配置。这不是环境问题，也不是这次能修的。
- 另有两个已知问题在真实 GPU 上得到确认：soft 下 `disable-features: gpu-limiter` 不生效；`tf_worker_usage` 没有数据。

## 2. 用例结果

共 61 条：**53 通过、6 已知问题（按正确行为断言、当前产品不满足）、2 失败**。用例执行时间合计 225 分钟（不含重跑）。耗时包含等待 GPU 节点的时间：第一次需要 GPU 的用例要等约 5 分钟的冷启动。

「版本」是该条用例最后一次执行时的 operator 版本。`dev-ack-20261012-1` = v2.17.28 + #820；`dev-ack-20261012-2` 再加 #822；两者的代码都已包含在 v2.17.30 里，hypervisor 在这三个版本之间没有变化。

| ID | 用例 | 状态 | 耗时 (s) | 版本 |
|---|---|---|---|---|
| CLD-01 | cluster-agent sidecar 运行（集群注册到控制台） | 通过 | 1 | 2.17.28 |
| FIN-01 | FinOps：N 个分片负载 vs 整卡基线的节点小时与成本 | 通过 | 1523 | dev-ack-20261012-2 |
| INS-01 | Helm 安装后控制面就绪 | 通过 | 6 | 2.17.28 |
| INS-02 | 默认集群与 Pool 被调谐（零节点） | 通过 | 2 | 2.17.28 |
| INS-03 | GPU 节点与 GPU 自动发现（含 KRP-A01 纳管 Karpenter 节点） | 通过 | 14 | dev-ack-20261012-2 |
| INS-04 | 非 GPU 节点不被纳管 | 通过 | 2 | 2.17.28 |
| ISO-01 | soft 分片：一张卡上运行多个 Pod | 通过 | 40 | dev-ack-20261012-1 |
| ISO-02 | soft 算力限额实测：吞吐随 share 成比例、无停顿 | 通过 | 1032 | dev-ack-20261012-2 |
| ISO-03 | 显存上限实测：容器内只见配额、超限分配失败、邻居不受影响 | 通过 | 50 | dev-ack-20261012-2 |
| ISO-04 | hard 算力限额实测（sidecar worker） | 已知问题 | 99 | dev-ack-20261012-2 |
| ISO-05 | shared 整卡：独占后其它 Pod 不会落到这张卡 | 通过 | 70 | dev-ack-20261012-1 |
| ISO-06 | remote 模式：业务 Pod 在 CPU 节点，经 TensorFusionConnection 使用远端 GPU | 已知问题 | 21 | 2.17.30 |
| ISO-13 | disable-features: gpu-limiter 关闭限流后不再受限 | 已知问题 | 126 | dev-ack-20261012-2 |
| ISO-14 | soft 模式对业务镜像的要求：无 libstdc++ 的镜像进程无法启动 | 已知问题 | 20 | dev-ack-20261012-1 |
| ISO-15 | 邻居隔离：吵闹的 Pod 不会把安静的 Pod 压到份额以下 | 通过 | 648 | 2.17.30 |
| ISO-16 | 运行期调整限额生效（cron 伸缩规则，即 ASC-02） | 通过 | 571 | dev-ack-20261012-2 |
| KRP-00 | Karpenter 就绪：NodePool / EC2NodeClass Ready，nodeOverlay 特性门打开 | 通过 | 1 | 2.17.28 |
| KRP-01 | NodeOverlay 与 operator 的 NodeClaim RBAC 就位 | 通过 | 2 | 2.17.28 |
| KRP-A02 | GPU 不足 → 扩容器创建 NodeClaim → 新节点 → Pod 运行（核心闭环） | 通过 | 338 | dev-ack-20261012-2 |
| KRP-A03 | 一次扩容承接多个 Pending Pod（不是一个 Pod 一台机器） | 通过 | 343 | dev-ack-20261012-2 |
| KRP-A08 | 从 0 个 GPU 节点冷启动：Pending 的 TF Pod → Karpenter NodeClaim → 节点 → GPU 注册 → Pod 运行 | 通过 | 570 | dev-ack-20261012-2 |
| KRP-A09 | 负载清空后节点被回收（should-delete 标记 + Karpenter consolidation） | 通过 | 141 | dev-ack-20261012-2 |
| KRP-A15 | chart 默认 NodeOverlay 会让 Karpenter 为 TF Pod 拉起非 TF 节点 | 通过 | 113 | dev-ack-20261012-2 |
| KRP-A16 | 扩容节点的 GPU 就绪后，Pending Pod 应及时绑定（不等调度器 5 分钟兜底重试） | 通过 | 1 | dev-ack-20261012-2 |
| KRP-A17 | N 个整卡 Pod 同时 Pending：恰好扩 N 台（不多扩、不少扩） | 通过 | 644 | dev-ack-20261012-2 |
| KRP-B02 | 路径 B：容量水位 → GPUNodeClaim → 同名 NodeClaim → 节点 → Bound（含 B01 连接检查、B09 删除 Pool 清理） | 通过 | 298 | 2.17.30 |
| KRP-B10 | 路径 B：删除 Pool 后 GPUNodeClaim / NodeClaim / 节点 / GPUNode / GPU 全部回收（接在 KRP-B02 之后） | 通过 | 2 | 2.17.30 |
| KRP-C01 | Karpenter 拉起并回收一台非 GPU 节点（通用 NodePool） | 通过 | 186 | 2.17.28 |
| NEG-02 | 指定不存在的 pool 在准入被拒 | 通过 | 12 | 2.17.28 |
| NEG-03 | 本地 hard 不写 sidecar-worker：实际被接受并自动加 sidecar | 通过 | 13 | 2.17.28 |
| NEG-04 | partitioned 不带 partition-id 被拒 | 通过 | 18 | 2.17.28 |
| NEG-05 | 整卡请求缺 gpu-model 被拒 | 通过 | 12 | 2.17.28 |
| NEG-06 | gang-min-members=1 被拒 | 通过 | 12 | 2.17.28 |
| NEG-11 | tflops 与 compute-percent 同时指定：实际被接受，tflops 生效 | 通过 | 18 | 2.17.28 |
| NEG-12 | soft 模式未写任何资源请求被拒 | 通过 | 17 | 2.17.28 |
| NEG-13 | 非法 isolation / qos 值的报错位置 | 通过 | 12 | 2.17.28 |
| NLC-01 | Hypervisor 重启：业务不中断、GPU 先 Pending 后恢复、不丢账 | 通过 | 50 | dev-ack-20261012-1 |
| NLC-02 | Operator 重启：分配状态重建、无重复分配 | 通过 | 111 | dev-ack-20261012-1 |
| NLC-09 | 节点被删除后状态收敛，Deployment 的 Pod 在新节点重建 | 通过 | 40 | dev-ack-20261012-2 |
| OBS-01 | Operator metrics 端点可抓取 | 通过 | 1 | 2.17.28 |
| OBS-02 | GPU / worker / 节点 / pool 指标在限定时间内写入 GreptimeDB | 已知问题 | 388 | dev-ack-20261012-1 |
| OBS-06 | GreptimeDB 可通过 HTTP SQL 查询 | 通过 | 1 | 2.17.28 |
| PERF-CO-01 | 切分 + 共调度：两个 vLLM 实例各占 50% 同卡并发，各自的吞吐与尾延迟、合计吞吐 | 通过 | 558 | 2.17.30 |
| PERF-LLM-01 | vLLM 流式推理 vs 算力份额（soft）：吞吐成比例、TTFT / TPOT 尾延迟无断崖 | 通过 | 1265 | 2.17.30 |
| PERF-LLM-02 | vLLM 流式推理 vs 算力份额（hard） | 已知问题 | 1772 | 2.17.30 |
| PERF-MLP-01 | MLPerf Inference 风格（LoadGen Offline / Server，缩减时长，非合规提交）vs 份额 | **失败** | 543 | 2.17.30 |
| PERF-ORT-01 | ONNX Runtime (CUDA EP)：CNN / 编码器 / 语音编码器的 QPS 与延迟分位 vs 份额 | 通过 | 992 | 2.17.30 |
| PERF-TB-01 | TorchBench 代表性子集（conv / transformer / 小 kernel 密集）vs 份额 | **失败** | 465 | 2.17.30 |
| SCH-01 | 默认装箱：小负载集中到一个节点 | 通过 | 45 | dev-ack-20261012-2 |
| SCH-04 | 按 GPU 型号调度：匹配的型号可调度，不存在的型号保持 Pending | 通过 | 85 | dev-ack-20261012-1 |
| SCH-10 | 命名空间配额拦截超限请求 | 通过 | 40 | dev-ack-20261012-1 |
| SCH-11 | Gang 调度（正向）：成员齐备后一起运行，注解与状态可见 | 通过 | 27 | dev-ack-20261012-1 |
| SCH-12 | 分配查询与扩容器状态 API 可用 | 通过 | 1 | 2.17.28 |
| SCH-14 | 自动分配 HostPort | 通过 | 18 | 2.17.28 |
| WHK-01 | 只加 label + annotation 即被改写 | 通过 | 18 | 2.17.28 |
| WHK-02 | 引用 WorkloadProfile，annotation 可覆盖 | 通过 | 13 | 2.17.28 |
| WHK-03 | 多容器 Pod 只注入指定容器 | 通过 | 13 | 2.17.28 |
| WHK-05 | enabled-replicas 只接入部分副本 | 通过 | 18 | 2.17.28 |
| WHK-06 | 未启用的 Pod 完全不被改动 | 通过 | 13 | 2.17.28 |
| WHK-07 | 删除负载后 TensorFusionWorkload 与 GPU 资源归还 | 通过 | 24 | dev-ack-20261012-1 |
| WHK-08 | server-side dry-run 不产生副作用 | 通过 | 34 | 2.17.28 |

## 3. 发现的问题与处理

### 3.1 已修复

| # | 问题 | 影响 | 处理 |
|---|---|---|---|
| 1 | operator 的机型目录只有 AWS 与 Azure，路径 B 在阿里云上选不出任何机型 | `provisioningMode: Karpenter` 的 pool 永远不建 GPUNodeClaim | #820：新增阿里云 ECS GPU 机型表（gn6i / T4、gn7i / A10，us-east-1 按量目录价），按 pool 引用的 NodeClass 的 API 组选表。v2.17.30 |
| 2 | ClusterRole 放行的是 `karpenter.k8s.alibabacloud`，官方 provider 的组是 `karpenter.alibabacloud.com` | operator 读不到 `ECSNodeClass` | #820：两个组都放行（marker、`config/rbac`、chart） |
| 3 | autoscaler 对所有 workload 等 30 分钟初始延迟才开始评估，cron 伸缩规则也被挡住 | 窗口落在 workload 前半小时内的 cron 规则不生效；短命 workload 永远不被伸缩 | #822：带可用 cron 规则的 workload 立即进入评估。v2.17.30。实测：窗口内吞吐降到 26%（目标 30%），窗口后恢复 |
| 4 | 阿里云 provider 返回的 NodeClaim 容量不含 NodeOverlay 声明的资源 | 一个 Pending 的 TF Pod 每 10 秒多拉一台，直到 NodePool 上限（实测 1 个 Pod → 4 台 T4） | `patches/karpenter-alibaba-0.2.5-overlay-capacity.patch`，`build-karpenter.sh` 构建并导入。打补丁后 1 个 Pod → 1 台。**应向上游提 issue / PR** |
| 5 | provider 的 `--feature-gates` 参数不接受 core 的 `NodeOverlay`，启动即 panic | Karpenter 起不来 | 只通过环境变量 `FEATURE_GATES` 设置。上游问题 |
| 6 | 用例自身的问题：`NLC-01` 在 GPU CR 记账前取基线；`ISO-06` 假定 worker 落在第一张卡的节点；`ISO-15` 在邻居还在时等「资源全部归还」；soft 算力判据用了对称的吞吐带；LLM 预热 15 秒不够；原生 vLLM 占整卡显存时在 T4 上自己崩溃 | 误报失败 | 逐条修正，见 4.2 与提交记录 |

### 3.2 未修复，需要跟进

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| 1 | **hard 隔离、remote 模式不可用** | `tensor-fusion-worker:v2.25.0`（community）日志：`VerifyLicense: License is not set` → `quit server`；hard 下业务进程随后 `Bus error`，remote 下 worker `CrashLoopBackOff`（退出码 0） | 产品决策：要么 chart / ProviderConfig 提供下发 License 的方式并写进文档，要么文档明确这两种模式不属于开源版。`ISO-04`、`ISO-06`、`PERF-LLM-02` 标为 known-fail |
| 2 | **soft 在低份额下按吞吐算略偏紧** | T4、matmul：限额 25% 时设备利用率 26%（准），吞吐为满载的 21%（比份额低约 15%）；50% 时利用率 53%、吞吐 79% | 控制环只看忙时比例，看不到「每毫秒完成的计算」随占空比变化。候选改法（未实现）：在已校准分支给令牌速率加一个下限 `limit × 未限流时的令牌速率`，只会放松不会收紧。先把每个 worker 的速率、增益、利用率作为指标导出，再决定 |
| 3 | soft 下 `disable-features: gpu-limiter` 不生效 | 30% 限额、带该注解，吞吐仍是不限时的 27.7% | webhook 只在 remote / hard 的客户端路径上把它转成环境变量（`compose_client.go`），soft 分支在此之前返回。`ISO-13` 标为 known-fail |
| 4 | `tf_worker_usage` 没有数据 | 负载运行 6 分钟后该表仍为空，其余四张表有数据 | 依赖它的用量告警、按用量伸缩、控制台的 worker 级图表没有数据源。`OBS-02` 仍是 known-fail |
| 5 | **兼容模式下扩容器建的 NodeClaim 不受 NodePool `limits` 约束** | 实测：把 `tf-gpu` 的上限调到 `nvidia.com/gpu: 1`，已有 1 台 GPU 节点时再起两个整卡 Pod，扩容器建的 `tensor-fusion-*` NodeClaim 照常 `Launched`，NodePool 状态变成 `limits=1, used=2`。Karpenter 只在自己的调度里检查上限 | 成本上限不能指望 NodePool：文档里写明要靠 GPUPool 的容量配置或配额；或让扩容器在建 NodeClaim 前检查 NodePool 的 `limits`。本环境另有 `guard.sh` 兜底 |
| 6 | 设备遥测每 60 秒一行 | `tf_gpu_usage` 每张卡每分钟一个瞬时样本 | 短于两分钟的测量窗口只有 0–1 个样本；控制台的利用率曲线分辨率也是一分钟 |
| 7 | vLLM 0.11.0 在 T4 上的两个上游问题 | FlashInfer 采样器的架构检查在计算能力 7.x 的卡上抛异常；原生占整卡 85% 显存、8 路并发时 `CUDA error: an illegal memory access` | 与 TensorFusion 无关。用例里关掉 FlashInfer 采样器，原生基线使用与租户相同的显存预算 |
| 9 | 单次调用几十毫秒的卷积推理在 50% 限额下尾延迟放大 | TorchBench-mini resnet50：p99 从 43.8 ms 到 208.7 ms，是按比例预期（87.6 ms）的 2.4 倍；吞吐 43.8%，正常 | 一次调用的令牌花费跨多个 10 ms 补充周期时，等待会叠加。与第 2 条一起看：需要先能观测控制环的内部量。`PERF-TB-01` 因此失败 |
| 10 | 批处理形态可能被限得比份额紧 | MLPerf 风格 resnet50 Offline：soft 50% 为 soft 100% 的 38.7%（下限 42.5%）。同一轮里原生只有 soft 100% 的 46%，短测波动很大 | 用 full 档的长窗口复测后再下结论。`PERF-MLP-01` 因此失败 |
| 8 | 无 libstdc++ 的业务镜像在 soft 下无法启动 | busybox：`error while loading shared libraries` | 已知问题（G-03），`ISO-14` |

## 4. 性能

### 4.1 Karpenter 与节点时间线

| 测量 | 数值 | 来源 |
|---|---|---|
| CPU 节点：Pending → NodeClaim | 7.0 s | KRP-C01 |
| CPU 节点：Pending → 节点 Ready | 63.7 s | KRP-C01 |
| CPU 节点：Pending → Pod Running | 74.5 s | KRP-C01 |
| CPU 节点：清空 → 节点消失（`consolidateAfter=30s`） | 51.0 s | KRP-C01 |
| **GPU 节点冷启动：Pending → NodeClaim** | 6.5 s | KRP-A08 |
| **GPU 节点冷启动：Pending → GPU 注册为 Running** | 309.7 s | KRP-A08 |
| **GPU 节点冷启动：Pending → Pod Running** | 315.5 s | KRP-A08 |
| 兼容模式扩容：Pending → 扩容器建出 NodeClaim | 13.8 s | KRP-A02 |
| 兼容模式扩容：Pending → Pod Running | 313.5 s | KRP-A02 |
| 新节点 GPU Running → Pending Pod 被绑定 | 8.0 s | KRP-A02 / A16（2.17.23 上是 330 s） |
| 3 个 20% 的 Pod 一次扩容，全部 Running | 318.1 s，1 个 NodeClaim、1 台节点、同一张卡 | KRP-A03 |
| 3 个整卡 Pod 同时 Pending，全部 Running | 362.0 s，恰好 3 个 NodeClaim、3 台节点 | KRP-A17 |
| GPU 节点：清空 → 节点消失（`consolidateAfter=2m`） | 128.3 s | KRP-A09 |
| 节点被删 → GPUNode / GPU CR 清理 | 0.8 s | NLC-09 |
| 节点被删 → Deployment 的 Pod 在另一台节点 Running | 7.5 s（另一台节点已存在） | NLC-09 |
| 路径 B：建 pool → GPUNodeClaim | 23.1 s | KRP-B02 |
| 路径 B：建 pool → Bound、GPU Running | 239.4 s / 240.2 s | KRP-B02 |
| 路径 B：指定该 pool 的 Pod 创建 → Running | 6.1 s | KRP-B02 |
| 路径 B：删 pool → 全部对象消失 | 8.3 s | KRP-B10 |
| hypervisor 重启恢复 | 18.1 s | NLC-01 |
| operator 重启恢复 | 23.0 s | NLC-02 |

GPU 节点从 Pending 到可用约 5 分 10 秒，其中 NodeClaim 到实例创建只有几秒，其余是实例启动、ACK 接入脚本安装 NVIDIA 驱动、hypervisor 拉取并发现设备。路径 B 的 `c8g1.2xlarge` 快约 70 秒。

FinOps（FIN-01，每个场景稳态保持 3 分钟，单价取 `../pricing/alibaba-us-east-1.json`）：

| 方案 | 节点数 | 节点小时 | 按量成本 | 冷启动到全部 Running |
|---|---|---|---|---|
| 3 个分片负载（各 20% 算力、2Gi） | 1 | 0.133 | $0.146 | 305 s |
| 3 个整卡负载（基线） | 3 | 0.483 | $0.529 | 602 s |

节省 72.4%。稳态下是 1 台对 3 台（67%）；这次还多省了一些，因为整卡方案的后两台是在第一台就绪后由扩容器再拉的。

### 4.2 soft 算力限额（ISO-02、ISO-15、ISO-16，matmul FP16 2048×2048）

| 限额 | 吞吐 | 相对 100% | 设备利用率 | 实际算力读数 | 功耗 | p99 单次时延 |
|---|---|---|---|---|---|---|
| 100% | 1320 it/s | 1.00 | 96% | 62.4 TFLOPS | 70 W | 0.82 ms |
| 50% | 1047 it/s | 0.79 | 53% | 34.5 TFLOPS | 77 W | 6.36 ms |
| 25% | 281 it/s | 0.21 | 26% | 16.9 TFLOPS | 35 W | 10.35 ms |

- 设备利用率就是限流器控制的量，三档都在目标的 3 个点以内。
- 50% 档吞吐高于 50% 不是收敛慢：单独跑 5 分钟，逐 30 秒均值是 975、1019、1005、1005…… 全程平稳。原因是 T4 持续满载时受功耗 / 温度限制，间歇负载下每个忙毫秒完成的计算更多（单次迭代中位数 0.52 ms 对 0.77 ms）。75% 档为满载的 93–95%。
- 25% 档按吞吐算比份额低约 15%，见 3.2 第 2 条。
- 判据因此改为：吞吐不得低于份额的 85%；吞吐不高于「份额 + 0.35」（上限 0.98）；窗口内有两个以上设备样本时，利用率不高于「份额 + 10 个点」；尾延迟在按比例放大的基础上加一个 12 ms 的补充周期余量。

邻居隔离（ISO-15）：40% 份额的 Pod 独占时 496.7 it/s，旁边有一个打满 60% 的邻居时 503.6 it/s（1.014 倍），没有 ≥0.4 秒的停顿。

运行期改限额（ISO-16，cron 规则把限额从 100% 降到 30% 持续 4 分钟）：窗口前 1271 it/s，窗口内 336 it/s（26.4%），窗口后 1276 it/s。

显存上限（ISO-03）：4Gi 限额下申请 6Gi，拿到 3840 MiB 后失败；容器内看到的总显存是 4096 MiB。

### 4.3 LLM 流式推理（vLLM 0.11.0，Qwen2.5-0.5B-Instruct，FP16）

每个租户显存上限 6Gi；8 路并发流式请求闭环打满，每个请求 64 个输出 token；90 秒预热后测 60 秒。原生基线使用与租户相同的显存预算。operator / hypervisor 2.17.30。

| 配置 | tokens/s | 相对原生 | TPM | TTFT p50 / p95 | TPOT p50 / p99 | ITL p99 / max | 停顿 ≥0.4s | 设备利用率 | 功耗 |
|---|---|---|---|---|---|---|---|---|---|
| 原生（不经 TensorFusion） | 361.8 | — | 21706 | 79 / 232 ms | 19.9 / 24.6 ms | 45 / 191 ms | 0 | 82% | 56 W |
| soft 100% | 360.5 | −0.4% | 21632 | 92 / 300 ms | 19.8 / 24.1 ms | 49 / 190 ms | 0 | 82% | 56 W |
| soft 50% | 265.8 | −26.5% | 15950 | 180 / 231 ms | 27.5 / 30.2 ms | 93 / 192 ms | 0 | 60% | 47 W |
| soft 50%，固定到达率（原生容量的 30%） | 105.9 | 全部按时完成 | 6354 | 94 / 106 ms | 21.2 / 22.7 ms | 65 / 71 ms | 0 | 46% | 47 W |
| soft 25% | 100.9 | −72.1% | 6054 | 528 / 736 ms | 64.0 / 85.2 ms | 179 / 450 ms | 8 | 39% | 30 W |

- **不限额时隔离没有可测的代价**：soft 100% 与原生的吞吐差 0.4%，TPOT 相同。
- 50% 份额拿到 73.7% 的吞吐、25% 拿到 28.0%：都不低于份额；延迟随份额变慢但没有断崖，全部判据通过。
- 负载低于份额容量时（固定到达率那一行）限流器不介入：TPOT、TTFT 与不限额时相当。
- 25% 档出现 8 次 0.4–0.45 秒的 token 间隔（每千 token 1.3 次，判据上限 5 次），没有接近 1 秒的。
- 设备利用率每个测量点只有 1 个样本（遥测每分钟一行），仅供参考。

### 4.4 切分 + 共调度（PERF-CO-01）

两个 vLLM 实例各拿 50% 算力、6Gi 显存，落在同一张 T4 上，同时被打满。

| 配置 | tokens/s | TTFT p50 / p95 | TPOT p50 / p99 | ITL p99 / max | 停顿 ≥0.4s | 设备利用率 | 显存占用 |
|---|---|---|---|---|---|---|---|
| soft 100%，独占整卡（参照） | 354.5 | 78 / 144 ms | 20.7 / 25.0 ms | 42 / 193 ms | 0 | 81% | 5.2 GiB |
| 租户 A（50%） | 192.2 | 368 / 429 ms | 35.3 / 40.7 ms | 87 / 255 ms | 0 | 96%（整卡） | 9.9 GiB（整卡） |
| 租户 B（50%） | 215.3 | 214 / 365 ms | 33.6 / 38.4 ms | 85 / 238 ms | 0 | 同上 | 同上 |
| **两租户合计** | **407.5** | | | | | | |

- 合计吞吐是单租户独占整卡的 **115%**：一个租户用不满这张卡（独占时利用率 81%），两个租户把它填到 96%。
- 两个租户的吞吐之比 0.89，较慢的一方拿到合计的 47%。
- 每个租户的 TPOT 约为独占时的 1.6–1.7 倍，TTFT p95 在 0.45 秒以内，没有停顿。

### 4.5 其它负载（quick 档，T4）

| 负载 · 模型 | 原生 | soft 100% | soft 50% | soft 50% / soft 100% | 备注 |
|---|---|---|---|---|---|
| ONNX Runtime · resnet50 | 365.2 samples/s | 364.4（−0.2%） | 206.6 | 56.7% | p99 时延 45.0 → 45.1 → 81.0 ms |
| ONNX Runtime · minilm | 968.9 samples/s | 965.7（−0.3%） | 601.2 | 62.3% | p99 16.9 → 17.0 → 30.6 ms |
| TorchBench-mini · resnet50（eval） | 23.9 it/s | 23.7（−0.8%） | 10.4 | 43.8% | **p99 43.3 → 43.8 → 208.7 ms**，是按比例预期的 2.4 倍，判据未过 |
| TorchBench-mini · bert_base（eval） | 21.0 it/s | 20.8（−0.6%） | 14.6 | 70.2% | p99 49.0 → 48.9 → 72.8 ms |
| TorchBench-mini · tiny_mlp（eval） | 1068.8 it/s | 1071.6（+0.3%） | 1038.6 | 96.9% | 这个负载自身只让 GPU 忙约 20%，50% 的限额不构成约束；判据因缺少基线的设备样本仍按「限额无效」判了未过，属判定局限 |
| MLPerf 风格 · resnet50 Offline | 170.2 samples/s | 371.2 | 143.7 | 38.7% | 低于下限 42.5%，判据未过。同一轮里原生比 soft 100% 低一半，说明这个秒级的 Offline 短测波动很大，这一行只能当线索 |

- 不限额时的开销在所有负载上都在 1% 以内。
- 50% 份额下的吞吐从 39% 到 97% 不等，取决于负载每个忙毫秒做多少事、以及它本来用不用得满这张卡。这正是不能用「吞吐 = 份额」来判 soft 限额的原因。
- 两条需要跟进的线索：单次调用几十毫秒的卷积推理在 50% 限额下尾延迟放大明显（TorchBench resnet50）；批处理形态可能被限得比份额紧（MLPerf Offline，需用 full 档的长窗口复测）。


## 5. 环境起停耗时与成本

数据来自 `../results/ack-ledger.tsv` 与各脚本的日志。

| 阶段 | 耗时 | 说明 |
|---|---|---|
| VPC + 2 个 vSwitch | 约 20 s | |
| ACK 集群创建到 `running` | 176 s | 托管版 Basic，Kubernetes 1.36.2 |
| system 节点池创建到节点 Ready | 约 58 s | 其中节点 Ready 的等待 49 s |
| Karpenter（RAM 角色、chart、ECSNodeClass、NodePool Ready） | 约 25–35 s | 不含补丁镜像的构建；镜像导入 system 节点约 30 s |
| TensorFusion chart（`helm --wait`） | 82–95 s | 多次安装 / 升级的范围 |
| **整套拉起合计** | **约 7 分钟** | 各阶段相加的估算。首次创建时各阶段是分开跑的；`run.sh up` 的单次合计会由 2026-10-13 08:00 的定时拉起记入 ledger（`up-total`） |
| Karpenter 拉起一台 CPU 节点到 Pod Running | 74.5 s | 4.1 节 |
| **Karpenter 拉起一台 GPU 节点到 Pod Running** | **315.5 s** | 4.1 节；NodeClaim 6.5 s，其余是实例启动、驱动安装、GPU 发现 |
| **整套销毁（`run.sh down`）** | **463 s** | 其中删除集群 431 s，RAM 角色、vSwitch、VPC 共约 30 s；销毁后清点全部为 clean，另用独立的 describe 调用核对：区域内 0 实例、0 磁盘、0 EIP、0 NAT、0 SLB |

环境存活约 7 小时 40 分（2026-10-12 12:34Z ～ 20:14Z）。成本估算，不是账单：

| 项 | 估算 |
|---|---|
| GPU 节点 | 约 9 节点小时 × $1.09 ≈ $10（按守卫日志每 2 分钟的实例数累计，含首轮因 provider 缺陷多拉的 3 台约 15 分钟） |
| system 节点（`ecs.g7.xlarge`） | 7.5 h × $0.15 ≈ $1.1 |
| NAT 网关、SLB、EIP、流量 | 约 $0.5–1 |
| 合计 | **约 $12** |

用例执行时间按档位（最后一次完整执行）：smoke 约 9 分钟、core 约 23 分钟、isolation 约 45 分钟、karpenter-finops 约 68 分钟（路径 B 另 5 分钟）、perf-quick 约 80 分钟（其中 30 分钟是 hard 用例等一个已失败的 Pod，等待逻辑已修）。每条用例的耗时见第 2 节的表。

## 6. 合入的改动

| PR | 内容 |
|---|---|
| #820 | fix：阿里云上的 Karpenter 供给模式（机型目录、NodeClass RBAC） |
| #821 | test(e2e)：ACK 档（起停脚本、provider 补丁、成本守卫、套件识别 `ECSNodeClass`）与首轮用例修正 |
| #822 | fix：cron 伸缩规则从 workload 创建起就评估 |
| 本次 PR | test(e2e)：真实 GPU 上的判据与用例修正（soft 算力判据、LLM 预热、原生基线显存预算、ONNX Runtime 加载路径、已知问题标注）、共调度用例 `PERF-CO-01`、机器指标、性能留档与查看器、ACK 文档 |

v2.17.29（含 #820）与 v2.17.30（再含 #822）由 main 上的发布流程自动产出。
