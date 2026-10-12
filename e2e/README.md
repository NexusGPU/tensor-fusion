# TensorFusion 云上端到端验证（AWS EKS + Karpenter）

这个目录把「在云上验证 TensorFusion」固化成可以反复执行的东西：

| 目录 | 内容 |
|---|---|
| `terraform/` | 一次性的 EKS + Karpenter 环境，带 AWS 侧超时自毁、成本查询、残留清扫。说明见 `terraform/README.md` |
| `suite/` | Go + Ginkgo 的端到端用例（build tag `e2e`），靠 kubeconfig 驱动，对任何集群都能跑（EKS、裸金属） |
| `suite/workloads/` | 在 Pod 里运行的探针与压测脚本（Python） |
| `mock/` | 演练模式用的 mock accelerator 构建脚本 |
| `kwok/` | 不需要云账号的一档：kind + KWOK 假节点 + Karpenter KWOK provider + chart + fake hypervisor（`fakehv/`）。见 6.9 |
| `tools/report.py` | 把一次或多次运行的结果生成对比报告（Markdown + CSV + 图表用 JSON） |
| `pricing/` | 实例单价表（来自 AWS Price List API），`make cost` 与 FinOps 用例共用 |
| `GO-LIVE.md` | GPU 配额到位后的真实 GPU 执行步骤 |

本文与 `suite/` 的实际实现一一对应：**表里的每个 ID 都是一个真实存在的用例**，没有实现的能力列在第 8 节并写明原因。

---

## 1. 现状（2026-10-12）

- GPU 配额为 0（按需和 spot 的 G/VT vCPU 都是 0，提额申请在 AWS 人工审核中），**没有任何用例在真实 GPU 上执行过**。
- 已在真实 AWS 上执行的：环境起停（3 轮）、自毁开关（2 次）、`smoke-nogpu` 档、`rehearsal` 档（mock GPU）、`dryrun` 档。结果见第 9 节。
- `isolation`、`perf` 两档只做到了 dry-run：清单过了 server-side dry-run、镜像与模型可解析、压测脚本与判定逻辑在 CPU 上跑通。它们的断言阈值是设计值，**第一次真实执行前不要把这两档的任何结论当成已验证**。

## 2. 执行模式：谁也别把演练当成硬件验证

每个用例的结果都带一个 `mode` 字段，JUnit 之外的汇总（`summary.json` / `summary.md` / `report.md`）里逐条标出：

| mode | 含义 | 能证明 | 不能证明 |
|---|---|---|---|
| `real-gpu` | Pod 跑在真实 NVIDIA GPU 上 | 全部 | — |
| `mock-gpu` | 演练：CPU 节点 + 仓库自带的 example accelerator（mock 设备，16 GiB / 312 TFLOPS） | webhook、调度与装箱、配额、分配记账、hypervisor / operator 重启恢复、Karpenter 扩缩容全链路、清理 | 任何与 CUDA 有关的事：显存 / 算力限额是否生效、容器内 GPU 是否可用、性能 |
| `fake-gpu` | KWOK 假节点（没有 kubelet，Pod 不会真的启动）+ fake hypervisor 代写 GPU CR 与 `index_*` 容量 | 只有控制面：扩容器 / Karpenter 建多少 NodeClaim、从哪个 NodePool 建、GPU 就绪后调度器多久绑定、回收与清理、路径 B 的对象链 | 节点上的一切：hypervisor 本身、设备发现、limiter、device plugin、kubelet、镜像拉取、任何耗时的绝对值（节点启动与 GPU 注册的延迟是参数，不是测量） |
| `no-gpu` | 不需要 GPU 的用例 | 控制面、准入、API | — |
| `dry-run` | 需要 GPU 的用例做到「只差一块卡」 | 清单被准入接受、镜像与模型存在、脚本与判定逻辑可执行、Karpenter 为 GPU NodePool 创建了 NodeClaim 且 EC2 只因配额拒绝 | 功能本身 |

已知在当前发布镜像上必然失败的用例标了 `KnownIssue`，汇总里显示为 `known-fail`（JUnit 里仍是失败，故意的）。

## 3. 为什么是 Go + Ginkgo + kubeconfig

- 仓库已有的 `test/e2e` 就是 Ginkgo，`vendor/` 里有 Ginkgo、Gomega、client-go、controller-runtime，套件不需要改 `go.mod`，也不需要 `make vendor`。
- 全部用 `unstructured` 对象，**不 import operator 的 Go 类型**：同一个测试二进制可以评判任何已发布版本和裸金属集群，注解名、label 名以字面量写在测试里，代码里改名会让测试失败（它们是文档承诺的公共契约）。
- build tag `e2e`：`go build ./...`、`go test ./...`、`make lint` 都看不到这个包。
- 只在两处 shell out 到 `kubectl`：`exec`（仓库没 vendor remotecommand）。HTTP 类访问（controller API、GreptimeDB、metrics）走 API server 的 service / pod proxy，不起 port-forward 进程。
- Ginkgo 的 label 过滤做分档，`--junit-report` 出 JUnit；汇总由 `ReportAfterSuite` 生成。
- 负载与测量逻辑放在 Python 脚本里随 ConfigMap 下发到 Pod 内执行，笔记本上不跑任何负载。

每个用例：稳定 ID、独立 namespace（结束即删，并等 `TensorFusionWorkload` 消失、GPU `available == capacity`）、有上限的超时、测得的数字进汇总。

## 4. 怎么跑

```bash
export AWS_PROFILE=tf-e2e
cd e2e

make up TTL=3h            # 环境（14–15 分钟），自毁开关 3 小时后触发
make test-smoke           # smoke-nogpu
make rehearsal            # mock-install + rehearsal 档
make test-dryrun          # dryrun 档
make cost                 # 花了多少、现在每小时多少
make down                 # 销毁（约 12 分钟；节点已被自毁开关清掉时 5–6 分钟）+ 残留清扫

# 真实 GPU（配额到位后，见 GO-LIVE.md）
make test-core
make test-isolation
make test-karpenter-finops
make test-perf-quick
E2E_PERF_PROFILE=full make test-perf-full

# 任意 kubeconfig（裸金属）
make test-core KUBECONFIG=~/.kube/bare-metal
E2E_PERF_PROFILE=full make test-perf-full KUBECONFIG=~/.kube/bare-metal

make test FILTER='core && !disruptive' FOCUS='ISO-0'   # 任意 label 过滤 + 按 ID 聚焦
make list                                              # 列出全部用例与档位
make report RUNS="results/<a> results/<b>"             # 多次运行并排对比
```

结果在 `results/<run-id>/`（已 git-ignore）：`junit.xml`、`ginkgo.json`、`summary.json`、`summary.md`、`run.log`，以及各用例的附件（`<ID>.perf.json`、`<ID>.finops.json`、`<ID>.nodes.json`）。

常用环境变量：

| 变量 | 作用 | 默认 |
|---|---|---|
| `E2E_GPU_MODE` | `real` / `mock` / `none` / `auto`。Makefile 的各档目标已设好 | 按档位 |
| `E2E_FIXED_IMAGES=1` | 被测镜像已包含 pricing 初始化修复时设置，放开路径 B 用例 | 未设 |
| `E2E_PERF_PROFILE` | `quick` / `full` | `quick` |
| `E2E_IMAGE_TORCH` / `E2E_IMAGE_VLLM` / `E2E_IMAGE_PYTHON` / `E2E_IMAGE_BASE` | 负载镜像（走镜像站时覆盖） | 见 `suite/framework_test.go` |
| `E2E_LLM_MODEL` | 指定 LLM；不设则按 VRAM 限额自动选 | 自动 |
| `E2E_NATIVE_RUNTIME_CLASS` | 原生基线 Pod 的 runtimeClass（节点默认 runtime 不是 nvidia 时） | 空 |
| `E2E_POOL` / `E2E_MOCK_POOL` | 目标 GPUPool 名 | `tensor-fusion-shared` / `tensor-fusion-mock` |

## 5. 分档：成本与时长

单价（us-west-2，`pricing/us-west-2.json`，2026-10-11）：EKS 控制面 $0.10/h，t3.large $0.0832/h，g4dn.xlarge 按需 $0.526/h（spot 约 $0.2–0.25/h），g5.xlarge $1.006/h，g6.xlarge $0.8048/h，t3.medium $0.0416/h。集群空转约 **$0.19/h**。

| 档位（label） | 需要 | 用例数 | 时长 | 成本（含集群空转） | 状态 |
|---|---|---|---|---|---|
| `smoke-nogpu` | 无 GPU | 27 | **7 分钟（实测）** | **约 $0.03（实测估算）** | 已执行 |
| `rehearsal` | mock GPU（1–3 台 t3.medium spot） | 21 | **约 75 分钟（实测）** | **约 $0.35（实测估算）** | 已执行 |
| `dryrun` | 无 GPU（1 台 CPU spot 节点） | 7 | **10 分钟（实测）** | **约 $0.05（实测估算）** | 已执行 |
| `kwok` | 不需要云账号：一台有 docker 的 Linux 主机（kind 单节点，内存上限 6g） | 16 | 起栈 72–76 秒；全量一次 1156 秒（约 19 分钟，实测） | $0 | 16 条全部执行过（9.6）：本分支的 operator 上全部通过，其中 `KRP-C01`、`NLC-10` 是改用例后重跑通过；发布镜像 ≤ 2.17.27 上 `KRP-B02` / `KRP-B10` 失败 |
| `core` | 1–2 台 g4dn.xlarge | 13 | 约 40 分钟（预估） | 按需约 $1.0，spot 约 $0.5（预估） | 未执行 |
| `isolation` | 1 台 g4dn.xlarge | 6 | 约 50 分钟（预估） | 按需约 $0.7（预估） | 未执行 |
| `karpenter-finops` | 最多 3 台 g4dn.xlarge（+ 路径 B 的 1 台 g4dn.2xlarge） | 11 | 约 90 分钟（预估；`consolidateAfter=10m` 下每次等节点回收要 10 分钟以上） | 按需约 $1.7，spot 约 $0.9（预估） | 未执行（其中 9 条已在 mock 上演练） |
| `perf`（quick，label `perf-quick`） | 1 台 g4dn.xlarge | 5 | 约 60 分钟（预估，首轮含 vLLM 镜像拉取） | 按需约 $0.75（预估） | 未执行 |
| `perf`（full） | 主要给免费的裸金属 2×RTX 3090；EKS 上只建议跑 `PERF-LLM-01/02` | 7 | 裸金属约 6 小时；EKS 精简子集约 50 分钟 | EKS 子集按需约 $0.6（预估） | 未执行 |
| `disruptive` | 任意 | 1（`UPG-03`） | 约 5 分钟 | — | 已过（no-gpu，卸载 190s） |

EKS 上全量跑一遍（smoke + core + isolation + karpenter-finops + perf-quick）：约 4.5 小时，按需约 **$4.2**，尽量用 spot 约 $2.6。即使每档重跑两三次也在 $15 以内，离 $100 预算很远。预估值是按用例里的固定等待和窗口加起来的，没有真实 GPU 运行数据支撑。

## 6. 用例清单

列：ID ｜ 档位 ｜ 验证什么 ｜ 通过标准（摘要）｜ 状态。状态一栏：**已过** = 在真实 AWS 上执行并通过（括号里是模式）；**known-fail** = 按正确行为断言、当前发布镜像不满足；**未执行** = 代码已实现，等 GPU。

### 6.1 安装与发现

| ID | 档位 | 验证什么 | 通过标准 | 状态 |
|---|---|---|---|---|
| INS-01 | smoke | Helm 安装后控制面就绪 | 12 个 CRD Established；controller Available；webhook 有 caBundle；leader lease 有持有者；GreptimeDB、alertmanager Ready；记录 controller 重启次数 | 已过（no-gpu） |
| INS-02 | smoke | 默认集群与 Pool 被调谐 | TFC、默认 pool、`tf-scheduling-config` 存在；pool 写出了 status；是默认 pool；有 GPU 时必须 `Running`，零节点时记录实际 phase（实测 `Updating`，G-04） | 已过（no-gpu） |
| INS-03 | core / rehearsal / karpenter | GPU 节点与 GPU 自动发现，Karpenter 节点被纳管（原 KRP-A01） | GPU CR 容量 > 0；GPUNode `Running`；hypervisor Ready 且 owner 同时含 GPUNode 与 Node；节点有 16 个 `index_*` 资源、归属 Karpenter NodePool；pool 汇总正确 | 已过（mock-gpu）；真实 GPU 未执行 |
| INS-04 | smoke / rehearsal | 非 GPU 节点不被纳管 | system / cpu 节点没有 GPUNode、没有 pool label | 已过（no-gpu、mock） |

### 6.2 分片、隔离

| ID | 档位 | 验证什么 | 通过标准 | 状态 |
|---|---|---|---|---|
| ISO-01 | core / rehearsal | soft 分片：一张卡跑多个 Pod | 3 个 20% 的 Pod `gpu-ids` 相同；GPU CR 的已用量 = 3 份请求（记录 CR 记账延迟）；`activeIsolationMode=soft`；真实 GPU 上另查容器内 `nvidia-smi`；删除后资源归还 | 已过（mock-gpu）；真实 GPU 未执行 |
| ISO-05 | core / rehearsal | shared 整卡 | 该卡 `available` 归零；之后的分片 Pod 不会落到这张卡 | 已过（mock-gpu）；真实 GPU 未执行 |
| ISO-06 | core | remote 模式 | client 在非 GPU 节点、label `component=client`；有 `TensorFusionConnection`；worker Pod 在 GPU 节点；client 容器内 `torch.cuda` 可用且只见 1 张卡 | 未执行 |
| ISO-14 | core / rehearsal | soft 模式对业务镜像的要求 | 无 libstdc++ 的镜像（busybox）在 soft 下应能运行 | **known-fail**（G-03，mock 上已复现） |
| ISO-02 | isolation | soft 算力限额实测 | matmul 吞吐在 100% / 50% / 25% 份额下满足 7.3 节的比例带；无 ≥0.9s 停顿 | 未执行 |
| ISO-03 | isolation | 显存上限实测（soft、hard 各一遍） | 4Gi 限额下申请 6Gi 失败；实际拿到 3.5–4.1Gi；容器内看到的总显存 ≈ 4Gi；同卡邻居不重启、不受影响 | 未执行 |
| ISO-04 | isolation | hard 算力限额实测 | 同 ISO-02，用 hard 的比例带 | 未执行 |
| ISO-15 | isolation | 邻居隔离（soft、hard） | 安静 Pod（40%）在吵闹邻居（60%，满负荷）存在时吞吐 ≥ 单独运行时的 80%；无 ≥0.9s 停顿 | 未执行 |
| ISO-16 | isolation | 运行期调整限额（cron 伸缩规则，即原 ASC-02） | 窗口内吞吐降到 30% 份额的比例带内，窗口结束后恢复到 ≥80% | 未执行 |
| ISO-13 | isolation | `disable-features: gpu-limiter` | 带 30% 限额但关闭限流时吞吐 ≥ 不限时的 85% | 未执行 |

`partitioned`（MIG）在 g4dn / g5 / g6 上不可测（T4、A10G、L4 不支持 MIG），放在裸金属 / 其它机型，见第 8 节。

### 6.3 调度

| ID | 档位 | 验证什么 | 通过标准 | 状态 |
|---|---|---|---|---|
| SCH-01 | core / rehearsal / karpenter | 默认装箱（`NodeCompactGPULowLoad`） | 两个 GPU 节点时，4 个 20% 的 Pod 全在一个节点上 | 已过（mock-gpu）；真实 GPU 未执行 |
| SCH-04 | core / rehearsal | 按 GPU 型号调度 | 型号匹配的 Pod 调度到该型号；不存在的型号 60 秒内不被调度，且不会为它新建 NodeClaim | 已过（mock-gpu）；真实 GPU 未执行 |
| SCH-10 | smoke / core / rehearsal | 命名空间配额 | 超出 `total.requests.tflops` 的 Pod 保持 Pending 且事件含 quota；有 GPU 时额度内的 Pod 运行，`status.used` 正确 | 已过（no-gpu、mock） |
| SCH-11 | core / rehearsal | Gang 调度（正向） | 2 副本带 `gang-enabled` 一起 Running；`gang-required-members=2`；同一个 group key | 已过（mock-gpu）；真实 GPU 未执行 |
| SCH-12 | smoke | 分配查询 / 扩容器状态 API | `GET /api/allocation`、`/api/node-scaler` 可用 | 已过（no-gpu） |
| SCH-14 | smoke | 自动分配 HostPort | `port-number` 注解在 40000–42000 | 已过（no-gpu） |

### 6.4 Webhook 与负向

| ID | 档位 | 验证什么 | 通过标准 | 状态 |
|---|---|---|---|---|
| WHK-01 | smoke / rehearsal | 只加 label + annotation 即接入（含原 OBS-04 的 Pending 原因可诊断） | `schedulerName`、`component=worker`、pool label、1 个 `index_*` limit、soft 的 init 容器与环境变量、生成 `TensorFusionWorkload`；零 GPU 时事件写明 `no gpu devices in pool` | 已过（no-gpu、mock） |
| WHK-02 | smoke | 引用 WorkloadProfile，annotation 覆盖 | 资源、QoS 取自 profile；annotation 能覆盖 | 已过（no-gpu） |
| WHK-03 | smoke | 多容器 Pod 指定注入容器 | 不写 `inject-container` 被拒；写了只有该容器带 index 资源 | 已过（no-gpu） |
| WHK-05 | smoke | `enabled-replicas` 灰度 | 4 副本里恰好 2 个被接入 | 三轮里过 2 轮；1 轮接入了 3 个，**间歇性 known-fail**（G-17） |
| WHK-06 | smoke | 未启用的 Pod 不被改动 | 只有 annotation 没有 label 时：默认调度器、无 init 容器、无 workload | 已过（no-gpu） |
| WHK-07 | smoke / core / rehearsal | 删除负载后资源归还 | workload 随 owner 消失；有 GPU 时 `available == capacity` | 已过（no-gpu、mock） |
| WHK-08 | smoke | server-side dry-run 无副作用 | dry-run 创建 Pod 后不出现 `TensorFusionWorkload` | **known-fail**（G-02） |
| NEG-02 | smoke | 不存在的 pool | 准入被拒，`gpu pool not found` | 已过 |
| NEG-03 | smoke | 本地 hard 不写 sidecar | 实际：被接受并自动加 `tensorfusion-worker`（G-09） | 已过（按现状断言） |
| NEG-04 | smoke | partitioned 不带 partition-id | 准入被拒（报错文案与 partition 无关，G-12） | 已过 |
| NEG-05 | smoke | 整卡缺 `gpu-model` | 准入被拒 | 已过 |
| NEG-06 | smoke | `gang-min-members=1` | 准入被拒 | 已过 |
| NEG-11 | smoke | `tflops-*` 与 `compute-percent-*` 同时写 | 实际：被接受，tflops 生效（G-10） | 已过（按现状断言） |
| NEG-12 | smoke | soft 不写任何资源 | 准入被拒 | 已过 |
| NEG-13 | smoke | 非法 `isolation` / `qos` | 不被接受（由 CRD 枚举拒绝，G-11） | 已过 |

### 6.5 恢复

| ID | 档位 | 验证什么 | 通过标准 | 状态 |
|---|---|---|---|---|
| NLC-01 | core / rehearsal | hypervisor 重启 | 新 hypervisor Ready、GPU 回到 `Running`；业务 Pod 不重启；GPU `available` 不变；记录恢复耗时与是否观察到 `Pending` | 已过（mock-gpu）；真实 GPU 未执行 |
| NLC-02 | core / rehearsal | operator 重启 | `/api/allocation` 的 key 集合重启前后一致；已占 60% 的卡不会再被分给另一个 60% 的 Pod；`available` 正确 | 已过（mock-gpu）；真实 GPU 未执行 |
| NLC-09 | karpenter / rehearsal | 节点被删除 | GPUNode 与 GPU CR 被清理；Deployment 的 Pod 在另一节点重新 Running；记录两段耗时 | 已过（mock-gpu）；真实 GPU 未执行 |

### 6.6 Karpenter 兼容与 FinOps

两条互斥路径（`docs/stack-context.md` §2.4）：**路径 A** = GPUPool `AutoSelect` + Karpenter NodePool（chart 默认）；**路径 B** = `provisioningMode: Karpenter`。

| ID | 档位 | 验证什么 | 通过标准 | 状态 |
|---|---|---|---|---|
| KRP-00 | smoke | Karpenter 就绪 | 所有 NodePool / EC2NodeClass Ready；`FEATURE_GATES` 含 `NodeOverlay=true`；记录解析出的 AMI | 已过（no-gpu） |
| KRP-01 | smoke | NodeOverlay 与 RBAC | overlay 含 `index: 512` 与 `index_0..f: 36` 且被 Karpenter 接受；operator 的 ServiceAccount 可 create / delete / list / watch NodeClaim | 已过（no-gpu） |
| KRP-C01 | smoke | Karpenter 拉起并回收非 GPU 节点 | 通用 NodePool 出现新 NodeClaim → Initialized → Pod Running；没有 GPU 节点被拉起；删除后 NodeClaim 与节点消失；记录各段耗时与节点小时 | 已过（no-gpu） |
| KRP-A08 | karpenter / rehearsal / dryrun | 路径 A 冷启动：0 节点 → Pending Pod → NodeClaim → 节点 → GPU 注册 → Pod Running | Karpenter 提名的 NodeClaim 属于 TF 的 NodePool；GPU 注册、Pod Running，记录三段耗时；**无配额时**：捕获 EC2 的配额报错作为证据后跳过 | 已过（mock-gpu，56s）；dry-run：GPU NodeClaim 已创建，EC2 以配额拒绝 |
| KRP-A15 | karpenter / dryrun | chart 默认 NodeOverlay 的副作用 | 去掉本栈的防护 overlay 后，Pending 的 TF Pod 不应让非 TF NodePool 出现 NodeClaim | **known-fail**（G-01），dry-run 档已复现 |
| KRP-A02 | karpenter / rehearsal | 路径 A 核心闭环：GPU 不足 → 扩容器建 NodeClaim → 新节点 → Pod Running | 一个 Pending Pod 只新增 1 个 NodeClaim，属于目标 NodePool、不带 `do-not-disrupt`；Pod Running；记录 `expansion-source`、「节点就绪 → Pod 绑定」耗时、节点小时 | mock-gpu：`consolidateAfter=5m` 时失败 1 次（G-16），`=10m` 时通过（381s，其中 330s 在等调度器重试）；真实 GPU 未执行 |
| KRP-A16 | karpenter / rehearsal | 扩容节点的 GPU 就绪后 Pending Pod 应及时绑定 | `KRP-A02` 测得的「节点 Initialized → Pod 绑定」< 120s | **known-fail**（G-16，mock 实测 330s） |
| KRP-A17 | karpenter / kwok | N 个整卡 Pod 同时 Pending 时不多扩 | 默认 N=4（`E2E_BURST`）：全程出现过的新 NodeClaim 恰好 N 个、全部来自 TF 的 NodePool；Pod 落在 N 个节点上；全部 Running 后 60 秒内不再出现新的 NodeClaim | 已过（fake-gpu，4 → 4，45s）；真实 GPU 未执行 |
| KRP-A03 | karpenter / rehearsal | 一次扩容承接多个 Pending Pod | 3 个 20% 的 Pod 只新增 1 个 NodeClaim，落在同一节点同一张卡 | 已过（mock-gpu）；真实 GPU 未执行 |
| KRP-A09 | karpenter / rehearsal | 负载清空后节点被回收 | 所有空 GPU 节点的 Node / NodeClaim / GPUNode / GPU CR 消失；记录回收耗时、是否先被打 `should-delete` | 已过（mock-gpu）；真实 GPU 未执行 |
| FIN-01 | karpenter / rehearsal | FinOps：N=3 个分片负载 vs 整卡基线 | 分片方案只用 1 个节点，整卡基线用更多节点；输出两种方案的节点数、节点小时、按需成本、节省比例（演练时用 g4dn.xlarge 单价代入并标 `mock-gpu`） | 已过（mock-gpu）；真实 GPU 未执行 |
| KRP-B02 | karpenter / dryrun / kwok | 路径 B：容量水位 → GPUNodeClaim → 同名 NodeClaim → 节点 → GPU 注册 → Bound → 负载调度到该节点（含原 B01 连接检查） | `CloudVendorConnectionReady=True`；GPUNodeClaim 带 `hourly-pricing`；NodeClaim owner 是 GPUNodeClaim、带 `do-not-disrupt`、请求 `nvidia.com/gpu: 1`；Bound；GPU CR Running 且属于该 NodeClaim 的节点，GPUNode 带 `node-provisioner` 与本 pool；指定该 pool 的分片 Pod 落在这台节点这张卡上；30 秒内不出现第二个 GPUNodeClaim；删 pool 后有残留即判失败 | fake-gpu 已过（本分支的 operator）；发布镜像 ≤ 2.17.27 上 **known-fail**（G-08 及其后两个缺陷，见 9.6），需 `E2E_FIXED_IMAGES=1`；真实 EC2 未执行 |
| KRP-B10 | karpenter / kwok | 路径 B：删除 Pool 后的回收（接在 `KRP-B02` 之后，报告其清理耗时） | GPUPool、GPUNodeClaim、NodeClaim、节点、GPUNode、GPU 全部消失 | fake-gpu 已过（10s）；真实 EC2 未执行 |

spot 与 on-demand 回退：NodePool 同时允许两种容量类型，节点账本（`<ID>.nodes.json`）记录每台节点实际的 `capacity_type`。回退路径本身在无配额账号上已被动验证：Karpenter 先试 spot（`MaxSpotInstanceCountExceeded`），随即换 on-demand（`VcpuLimitExceeded`），见 `GO-LIVE.md`。有配额后无法稳定制造 spot 容量不足，所以没有单独的断言用例（第 8 节）。

### 6.7 可观测、控制台、卸载

| ID | 档位 | 验证什么 | 通过标准 | 状态 |
|---|---|---|---|---|
| OBS-01 | smoke | operator metrics 端点 | 返回 `controller_runtime_reconcile_total` | 已过（no-gpu） |
| OBS-06 | smoke | GreptimeDB 可查询 | HTTP SQL API 可用；记录存在的 `tf_*` 表（零 GPU 时实测为 0 张，G-06） | 已过（no-gpu） |
| OBS-02 | core / rehearsal | 指标真正入库 | 负载运行后 6 分钟内 `tf_gpu_usage`、`tf_worker_usage`、`tf_node_metrics`、`tf_pool_metrics`、`tf_worker_resources` 都有近 15 分钟的行；记录每张表首行到达耗时 | **known-fail**（G-07，依赖修复后的镜像），mock 上已复现：3 张表有数据，`tf_gpu_usage`、`tf_worker_usage` 没有 |
| CLD-01 | smoke | 控制台 agent | 安装时给了 `TF_AGENT_ID` 才执行：`cluster-agent` 容器 Ready；日志里不出现 enroll token。控制台界面是否出现集群需人工确认 | 跳过（未配置 agent） |
| UPG-03 | disruptive | 卸载脚本 | 12 个 CRD、webhook、namespace、NodeOverlay 消失；节点上没有 `tensor-fusion.ai/*` label 与容量；普通 Pod 可创建 | 已过（no-gpu，卸载 190s） |

### 6.8 dry-run 档（给还没有 GPU 的阶段）

| ID | 验证什么 | 通过标准 | 状态 |
|---|---|---|---|
| DRY-PERF-01 | LLM 测量与判定流水线 | CPU stand-in 服务端模拟四种限流行为：平滑按比例 → 判通过；500ms 阶梯放行 → 判出尾延迟断崖与双峰；注入 1s 停顿 → 判出停顿；不限流 → 判出「限额无效」 | 已过（dry-run） |
| DRY-PERF-02 | ORT / TorchBench-mini / MLPerf-LoadGen 脚本 | 在 CPU 节点上完成导出、建会话、限时测量、结果解析，8 个结果点全部有吞吐 | 已过（dry-run） |
| DRY-ISO-01 | GPU 档位清单 | 11 份清单（soft / hard / shared 探针、vLLM soft / hard / eager、ORT、TorchBench、MLPerf、原生基线）通过 server-side dry-run | 已过（dry-run） |
| DRY-IMG-01 | 镜像与模型 | 所有用到的镜像在 registry 上能解析出 digest 与 linux/amd64 压缩大小；LLM 模型的 `config.json` 不带 token 可访问 | 已过（dry-run） |
| KRP-A08 / KRP-A15 / KRP-B02 | 见 6.6 | | |

### 6.9 KWOK 档（不需要云账号，label `kwok`）

目的：在没有 AWS 账号、没有 GPU 的时候，仍然能回归 Karpenter 集成的**控制面**——尤其是 2.17.24 修的「GPU 就绪后 Pending Pod 不被重新入队」（G-16，修复前 330s）。

组成（`kwok/up.sh`，参数在 `kwok/lib.sh`）：

| 组件 | 版本 / 来源 | 说明 |
|---|---|---|
| kind 单节点 | `kindest/node:v1.36.4` | 唯一有 kubelet 的节点，label `e2e.tensor-fusion.ai/role=system`，跑全部控制面 |
| KWOK | v0.8.0（`kwok.yaml` + `stage-fast.yaml`） | 假 kubelet：给假节点续心跳，把调度到假节点上的 Pod 标成 Running / Ready |
| Karpenter | kubernetes-sigs/karpenter **v1.14.1** 的 `kwok/` provider，从源码构建（上游不发布镜像） | 与 EKS 栈同一个 Karpenter 核心版本；`FEATURE_GATES` 含 `NodeOverlay=true`；机型来自 `kwok/instance-types.json`（g4dn.xlarge / g4dn.2xlarge / g5.xlarge / g6.xlarge / c5.large / m5.large / t3.large，按需价取自 `pricing/us-west-2.json`，spot 按 0.4 折） |
| NodePool | `tf-gpu`（weight 10，WhenEmpty，`consolidateAfter=1m`）、`tf-cpu`（weight 100，30s），`KWOKNodeClass/tf-kwok`（`nodeRegistrationDelay=10s`） | 与 EKS 栈同形；GPU NodePool 的节点带 `nvidia.com/gpu.present=true` 与 `e2e.tensor-fusion.ai/shared-pool=true` |
| TensorFusion | 本仓库 chart + operator 镜像（默认发布镜像 `tensor-fusion-operator:2.17.24`；`TF_OPERATOR_IMAGE` / `TF_RELEASE_TAG` 可指向本机 docker 里已有的开发镜像，本机有就不再拉取） | 与 rehearsal 相同的 values；差异：GreptimeDB、alertmanager、hypervisor 的 vector sidecar 关闭，controller 去掉 `-enable-alert`、`-enable-auto-scale`（依赖 TSDB）；`karpenterNodeOverlay.requirements` 改为 `karpenter.kwok.sh/instance-family In [g4dn, g5, g6]`（chart 默认值用的是 AWS provider 的 label，KWOK provider 不产生这些 label） |
| fake hypervisor | `kwok/fakehv`（Go，build tag `e2e`，在主机上以 kind 的 kubeconfig 运行） | 见下 |

fake hypervisor 代写的内容，逐项对应真实 hypervisor 的写入点：

| 对象 | 字段 | 对应代码 |
|---|---|---|
| `Node.status.capacity / allocatable` | 16 个 `tensor-fusion.ai/index_<hex>`，各 36 | `pkg/hypervisor/backend/kubernetes/deviceplugin.go` |
| `GPU`（每个假设备一个，label `e2e.tensor-fusion.ai/fake=true`） | label `managed-by`（`utils.LabelSafeValue(节点名)`，与真实 hypervisor 同一个函数）/ `gpupool`、指向 GPUNode 的 controller ownerRef、`status.capacity`、首次的 `available`、`uuid`、`gpuModel`、`vendor=NVIDIA`、`index`、`numaNode`、`isolationMode` / `isolationPolicy`（取自 operator 写进 hypervisor Pod 的环境变量）、`nodeSelector`、`usedBy`、`phase=Pending`、`message` | `KubeletBackend.mutateGPUResourceState` |
| `GPUNode.status` | `totalTFlops`、`totalVRAM`、`totalGPUs`、`managedGPUs`、`managedGPUDeviceIDs`、`nodeInfo` | `APIClient.UpdateGPUNodeStatus` |

它只处理同时满足三个条件的节点：带 `kwok.x-k8s.io/node=fake`、已有 GPUNode（即已被 TensorFusion 纳管）、hypervisor Pod 为 Running；并且在该 Pod 首次 Running 之后等 `KWOK_FAKE_HV_DELAY`（默认 20s，代表驱动初始化 + hypervisor 启动 + 设备发现）才写。`phase=Running`、之后的 `available`、`runningApps` 仍然由 operator 写，与生产一致。每节点 1 张卡：g4dn → T4（65 TFLOPS / 15360Mi）、g5 → A10G（125 / 23028Mi）、g6 → L4（121 / 23034Mi）。

这一档**不能证明**的：真实设备与驱动、hypervisor 进程本身（启动、重启恢复、device plugin 注册、worker 记账、指标）、limiter 与隔离、kubelet 行为（镜像拉取、探针、驱逐）、EC2 侧的一切（配额、容量不足、spot 中断、AMI）。假节点上「Pod Running」只表示调度器绑定了它。所有耗时里只有控制面的反应时间是测量值；节点启动与 GPU 注册的延迟是参数。

用例（16 条）：`KRP-00`、`KRP-01`、`KRP-C01`、`INS-03`、`KRP-A08`、`KRP-A15`、`KRP-A02`、`KRP-A16`、`KRP-A03`、`KRP-A17`、`KRP-A09`、`FIN-01`、`NLC-09`、`NLC-10`、`KRP-B02`、`KRP-B10`。`NLC-10` 是这一档独有的：KWOK provider 给节点起的名字是 `kwok-<NodeClaim>-<两个词>-<n>-<随机串>-<uint32>`，扩容器建的 NodeClaim（`tensor-fusion-<10 位>`）和路径 B 的 NodeClaim 对应的节点名都超过 63 个字符（实测 68、71），用例先占满现有 GPU 再多起一个 Pod 逼出这样一台节点，然后断言 GPU 的 `managed-by` label 合法且等于 `LabelSafeValue(节点名)`、完整节点名在 ownerReference 里、GPUNode 的已分配量随 Pod 变化（说明 operator 的 selector 找得到这张卡）、Pod 调度到该节点。`up.sh` 不再修改上游 provider 的节点命名。与其它档共用同一份用例代码；KWOK 特有的只有 `E2E_GPU_MODE=fake`（mode 记为 `fake-gpu`）、路径 B 的 NodeClass 引用（`KWOKNodeClass`）、`FIN-01` 的稳态窗口（`E2E_FINOPS_HOLD=45s`）。`KRP-A02` 在所有档都新增了 `gpu_running_to_pod_bound_seconds`（GPU CR 变为 Running → Pod 绑定，500ms 采样），`KRP-A16` 对它的判据是 < 30s。`KRP-A15` 在没有防护 overlay 的集群上不再跳过，直接对 chart 自带的 overlay 断言。

```bash
make kwok-up kwok-test kwok-down                      # 本机有 docker 时
KWOK_SSH_HOST=<主机> make kwok-up kwok-test kwok-down  # checkout 拷到该主机上执行，结果拷回 results/
FOCUS='KRP-A02|KRP-A16' make kwok-test
TF_RELEASE_TAG=2.17.23 make kwok-up                   # 换被测版本（对比修复前）
```

镜像站、代理、主机路径通过环境变量给（`KWOK_IMAGE_MIRROR`、`KWOK_K8S_IMAGE_MIRROR`、`KWOK_PULL_TOOL=crane`、`GOPROXY`、`KWOK_HOME`；远程执行时放在 `KWOK_REMOTE_ENV` 指向的远端文件里），不写进仓库。

主机前提：`fs.inotify.max_user_instances` 要有余量（kind 的已知要求，建议 ≥ 512）。kind 节点容器里的 kubelet、containerd 以主机 root 身份占用 inotify 实例，主机上已有别的 Kubernetes 发行版在跑时默认的 128 可能不够，表现为 `kubeadm init` 超时、节点日志里 `error creating fsnotify watcher: too many open files`（未调高时的现象；调到 1024 后一次成功）。

路径 B：chart ≥ 1.8.9 的 ClusterRole 允许 operator 读取各 Karpenter provider 的 NodeClass（含 `karpenter.kwok.sh`），实测 `kubectl auth can-i get/list/watch kwoknodeclasses` 为 yes、create 为 no；`up.sh` 里原先的 `KWOK_NODECLASS_RBAC=1` 旁路已删除。`KRP-B02` 用的 pool 没有 `nodeSelector`（`NodeProvisioner` 与 `NodeSelector` 互斥是 API 的约定），节点靠 NodeClaim 透传下来的 `tensor-fusion.ai/node-provisioner` label 归属到 pool。

## 7. 性能与隔离效果的设计（`isolation`、`perf`）

### 7.1 测什么

不只测吞吐。限额「有效」和「不把服务质量搞坏」是两件事，每个测量点同时记录：

| 负载 | 吞吐 | 延迟 | 停顿 / 异常 |
|---|---|---|---|
| LLM 流式推理（vLLM） | 输出 tokens/s、TPM、请求/s | TTFT、TPOT、逐 token 间隔（ITL）、请求总时延，各 p50 / p95 / p99 / max | ITL ≥0.4s、≥0.9s 的次数；落在 0.5s、1s 附近的次数；ITL 的双峰系数；错误率、超时数 |
| ONNX Runtime / TorchBench / MLPerf / matmul | samples/s 或 iters/s | 单次调用时延 p50 / p95 / p99 / max | 单次调用 ≥0.4s、≥0.9s 的次数 |

为什么盯 0.5s / 1s：soft limiter 等 token 的超时是 1s，令牌每 500ms 补一次。限流器「饿死」负载时的特征就是周期性的约 0.5s / 1s 停顿。

LLM 压测端（`workloads/llm_client.py`，纯标准库）两种负载：闭环固定并发（饱和）和开环固定到达率（低于容量时不应有任何退化）。

### 7.2 负载与运行时

限流器拦截 kernel launch、CUDA graph、显存分配的方式因框架而异，所以覆盖不同的执行形态：

| 运行时 | 模型 / 形态 | 说明 |
|---|---|---|
| vLLM（流式，CUDA graph / 编译路径；另有 `--enforce-eager` 变体） | `Qwen/Qwen2.5-0.5B-Instruct`（VRAM 限额 < 10Gi，fp16 约 1GiB，能放进 T4 的 4Gi 分片）；`Qwen/Qwen2.5-1.5B-Instruct`（≥ 10Gi） | Apache-2.0，**不需要 Hugging Face token**（`DRY-IMG-01` 每次核对）。按检测到的 GPU 显存自动选，`E2E_LLM_MODEL` 可覆盖 |
| ONNX Runtime GPU（CUDA EP；`--provider tensorrt` 可选，未默认启用） | `resnet50`（CNN，少量大 kernel）、`mobilenet_v2`（大量小 kernel）、`bert_base` / `minilm`（编码器 / embedding 形态）、`whisper_enc`（语音编码器：conv1d + transformer） | 随机权重、在 Pod 内 `torch.onnx.export`，不下载数据集和权重 |
| TorchBench 代表性子集 | `resnet50`、`mobilenet_v2`、`bert_base`（hf_Bert）、`gpt2`（hf_GPT2）、`tiny_mlp`（launch-bound，对应历史结果里开销最大的 `drq`）；eval，full 档加 train | 默认 `--impl mini`：用 TorchBench 封装的同一批模型类重新实现，**不是** pytorch/benchmark 的 harness；`--impl torchbench` 走真正的 `run.py`（给裸金属，未验证） |
| MLPerf Inference 风格 | 官方 LoadGen（`mlcommons-loadgen`）+ ONNX Runtime SUT；`resnet50`、`bert`；Offline、Server | 明确**不是合规提交**：随机权重、合成输入、`min_duration` 为秒级、Server 的目标 QPS 取 Offline 实测的 60% |
| matmul（`gpu_probe.py burn`） | FP16 2048×2048 | 纯 GPU 受限的参照物，隔离用例用它 |

沿用的既有思路：`gemm.py` 的 matmul 思路、vLLM 压测端的指标定义（TTFT / TPOT / 分位数）、TorchBench 子集的选择依据（2025 年结果里 `drq` +44.8%、其余 ±1%）。没有直接复用，本目录的清单是重新编写的；MLPerf 部分没有留下任何日志样例。历史数字可作基线参考：vLLM hard 模式 −1.3%～−2.4%（Qwen2.5-7B，2025-11）；TorchBench 84 个模型绝大多数 ±1%。

### 7.3 通过标准（双向）

记 `s` 为算力份额，基线为同一模式、同一 VRAM 限额、同一负载在 100% 份额下的结果，native 为不经过 TensorFusion 的结果。实现在 `suite/perf_eval_test.go`，阈值集中在 `criteriaFor()`。

| 判据 | soft | hard | 说明 |
|---|---|---|---|
| **(a) 有效：吞吐成比例** `thr(s)/thr(100%)` | `[0.75·s, 1.25·s + 0.05]` | `[0.85·s, 1.15·s + 0.05]` | soft 是进程内令牌桶 + 对利用率的 PID 反馈、500ms 补充一次，围绕目标调节，±25% 是诚实的带宽；hard 在执行调用的 worker 里直接限制，机制更直接，取 ±15% |
| (a) 限额不是摆设 | 最低份额（≤60%）的吞吐 ≤ 100% 时的 0.85 | 同 | 曲线是平的就是没限住 |
| (a) 100% 份额相对 native 的开销 | 吞吐损失 ≤ 5% | ≤ 15%（shared ≤ 3%） | 「不限时隔离没有代价」是要验证的说法；hard 有 remoting 开销 |
| **(b) 无断崖：尾延迟** TPOT（LLM，按请求）或单次调用时延的 p99（样本不足 300 时用 p95） | ≤ 1.5 × 基线 / s | ≤ 1.3 × | 按比例变慢是预期，超过比例的部分是断崖 |
| (b) ITL p99（LLM，按 token） | ≤ 1.5 × 基线 ITL p99 / s | ≤ 1.3 × | TPOT 是每个请求的均值，会把停顿抹平；逐 token 间隔不会。演练里「500ms 阶梯放行」的 stand-in：TPOT p50 完全成比例，ITL p99 是比例值的 4.5 倍 |
| (b) p50 | ≤ 1.3 × 基线 p50 / s | 同 | |
| (b) TTFT p95 | ≤ 1.5 × 基线 p95 / s + 0.25s | 同 | 加一个固定项，用来区分「按比例变慢」和「首 token 前卡了一下」 |
| (b) 无停顿特征 | 没有 ≥0.9s 的间隔；≥0.4s 的间隔 ≤ 5 次 / 千 token（或千次调用） | 同 | 对所有受限点独立判定，不需要基线 |
| (b) 不双峰 | ITL 的 p99/p50 不超过基线该比值的 3 倍 | 同 | 平均值达标但呈双峰就是断崖 |
| 错误率 | ≤ 1% | 同 | |
| 固定到达率（低于份额容量） | 完成的吞吐 ≥ 提供负载的 95% | 同 | |

样本量：p99 只在该分布样本数 ≥ 300 时参与判定，60–299 个样本时改判 p95，更少则只报告。quick 档 60s 窗口约 200 个请求、约 1 万 token：ITL p99 可判，TPOT 判 p95，TTFT 判 p95；full 档 180s 窗口约 600 个请求、3 万以上 token：TPOT p99 也可判。所有分位数无论是否参与判定都写进结果。

这些阈值是设计值。`DRY-PERF-01` 用可控的 stand-in 证明了判定逻辑对四种限流行为给出正确结论，但阈值本身是否合适要等第一次真实 GPU 运行后校准。

### 7.4 因子与试验设计

因子：隔离模式（native / shared / soft / hard）× 算力份额 × VRAM 限额（紧 / 松）× 负载与运行时 × 共租（独占 / 吵闹邻居）× 负载强度（饱和 / 固定到达率）。全笛卡尔积上千组，不跑。

- **只在「比例性」是问题本身的地方做完整扫描**：vLLM 在 soft（`PERF-LLM-01`）和 hard（`PERF-LLM-02`）下的份额扫描，外加一次固定到达率。
- **其余用成对（pairwise）设计**（`PERF-PW-01`，`perfDesign()`）：任意两个因子的任意两个水平至少同时出现一次，每个因子的主效应仍可估计，放弃的是三阶以上交互。5 个因子 180 种组合约 17 次运行。不可能存在的组合被排除（整卡 shared 没有份额、没有邻居、没有 VRAM 限额；100% 份额没有邻居）。
- 每次测量限时：短预热 + 固定窗口。模型与 pip 包缓存在节点的 hostPath（`/var/lib/tf-e2e/cache`），每个节点只下载一次。
- 服务端按份额重建而不是原地改限额：TensorFusion 除了自动伸缩之外没有原地改限额的接口（`ISO-16` 单独用 cron 规则验证原地调整）；模型来自节点缓存，重建约 1–2 分钟。
- 并行：用例在一个进程内串行执行，避免相互干扰测量。要缩短墙钟时间，在两个节点 / 两张卡上各起一次 `make test FILTER=perf FOCUS=…`（不同 `RUN_ID`），未做成自动扇出。

| ID | 档位 | 内容 |
|---|---|---|
| PERF-LLM-01 | perf / perf-quick | vLLM：native 基线 + soft 份额扫描（quick：100 / 50 / 25；full：100 / 75 / 50 / 25）+ 50% 份额下的固定到达率 |
| PERF-LLM-02 | perf / perf-quick | vLLM：native + hard 份额扫描（quick：100 / 50） |
| PERF-LLM-03 | perf | vLLM 变体：整卡 shared；VRAM 紧 / 松 × soft 100 / 50；`--enforce-eager` |
| PERF-ORT-01 | perf / perf-quick | ONNX Runtime：native、soft 100、soft 50（full 追加 soft 25、shared、hard 100 / 50、soft 50 + 吵闹邻居） |
| PERF-TB-01 | perf / perf-quick | TorchBench 子集，同上配置 |
| PERF-MLP-01 | perf / perf-quick | MLPerf 风格，同上配置（quick：resnet50 Offline；full：+ bert、+ Server） |
| PERF-PW-01 | perf | 成对设计 |

两个 profile：

| profile | 用途 | 每个测量点 | 预计 |
|---|---|---|---|
| `perf-quick`（默认） | 每次都跑，适合 EKS spot 节点 | 15s 预热 + 60s 窗口 | 1 张 T4 上约 60 分钟，约 $0.75（按需） |
| `perf-full`（`E2E_PERF_PROFILE=full`） | 完整设计，主要给免费的裸金属 2×RTX 3090；EKS 上只建议 `FOCUS='PERF-LLM-0[12]'` 做跨硬件确认 | 30s 预热 + 180s 窗口 | 裸金属约 6 小时；EKS 精简子集约 50 分钟、约 $0.6 |

### 7.5 输出

每次运行一个机器可读结果（`summary.json`）加每个用例的 `<ID>.perf.json`（每个测量点的全部指标与逐条判据）。`make report` 生成：

- `report.md`：用例状态并排表；按负载分组的「指标 vs 份额」表（每种模式一行，含相对 100% 与相对 native 的比值、通过 / 失败、失败的判据）；FinOps 表；
- `perf.csv`：每个测量点一行；
- `perf.json`：画图用的序列（每个 运行 × 负载 × 模型 × 模式 × 指标 一条，点为 `[份额, 值, 是否通过]`）。

把裸金属和 EKS 的结果目录一起传给 `make report RUNS="…"` 就是并排对比；把两个版本的结果一起传就能看回归。

## 8. 没有自动化的用例及原因

早先清单（101 条，按代码阅读写成）里没有进入套件的部分。**清单里的 ID 不复用**，以后实现时沿用原 ID。

| 原 ID | 内容 | 为什么没自动化 |
|---|---|---|
| ISO-08、SCH-03、SCH-05、SCH-07、SCH-13、INS-07、NLC-07 | 同节点多卡：动态隔离、多卡 Pod、指定卡号、拓扑、整卡预留回归、与原生 device plugin 共存、defrag | 本栈只有单卡机型；多卡在裸金属 2×RTX 3090 上覆盖。套件对任何 kubeconfig 可用，但这几条还没写 |
| `partitioned`（MIG）正向 | MIG 分区 | T4 / A10G / L4 不支持 MIG；且 chart 默认的 dynamic 策略拒绝 partitioned，需要另一套 static 策略部署 |
| ISO-07、NLC-10 | remote + hard、remote 连接自愈 | 先要 `ISO-06` 在真实 GPU 上跑通 |
| ISO-10、ISO-11 | 真实训练 / 推理、性能基线 | 由 `perf` 档取代 |
| ISO-12 | 算力超卖比 | 预期行为不明确，需要先在真实 GPU 上观察 |
| SCH-02 | 切换 placement 策略 | 要改全局 `SchedulingConfigTemplate`，会影响并行的其它用例；适合单独环境 |
| SCH-06 | nodeSelector / 污点 | 低优先级，未写 |
| SCH-08、SCH-09 | 优先级抢占、驱逐保护 | 抢占与节点扩容会竞争（Pending 的高优 Pod 同时触发扩容器），要先把 NodePool 顶到上限才稳定；且仓库里对应测试是 `PIt`（未启用） |
| SCH-11 负向 | Gang 资源不足时全部不绑定、超时 | 有 Karpenter 时「资源不足」不成立（会扩容），要配合 quota 或 NodePool 上限 |
| WHK-04、NEG-07 | 原生 `nvidia.com/gpu` 自动迁移；operator 不可用时的影响面 | 要改 chart value 并重启 controller，且会去掉 webhook 的 `objectSelector`（影响全集群 Pod 创建）。破坏性，适合人工在专门窗口做 |
| ASC-01、ASC-03、ASC-04、ASC-05 | 按用量自动伸缩及其变体 | 依赖用量指标入库（G-07），发布镜像上没有数据；`ISO-16` 已覆盖 cron 规则 |
| OBS-03 | 内置告警触发 | 依赖指标（G-07）；`GPUVRAMFull` 查询的列恒为 0（G-29） |
| OBS-05 | 指标附加业务 label | 依赖指标（G-07） |
| NLC-03、NLC-04 | Terminating Pod 跨重启保留、高频创建删除无泄漏 | 未写；NLC-04 耗时长，适合夜间 |
| NLC-05、NLC-06 | TF compaction 标记、手动触发、免打扰 | `KRP-A09` 记录了是否先出现 `should-delete`，没有单独断言：新节点有 5 分钟保护期，Karpenter 的 `consolidateAfter` 先到 |
| NLC-08 | hypervisor 分批滚动更新 | 需要两个可用的 hypervisor 镜像 tag，未写 |
| KRP-A04、A05、A07、A10、A12、A13、A14 | 路径 A 的各类变体（同名 Pod、机型 / 可用区约束、不该扩容的情况、consolidation 与运行中负载、与非 TF 负载共存、并发上限、扩容中重启 operator） | 未写。A12 的一部分由 `KRP-C01`（CPU 负载不触发 GPU 节点）和 `KRP-A15` 覆盖；A07 的「不为不可满足的请求扩容」由 `SCH-04` 覆盖 |
| KRP-A06 | 容量不足时按偏好回退 | 无法稳定制造 `InsufficientCapacityError`；无配额账号上被动观察到 spot → on-demand 的回退序列 |
| KRP-A11、KRP-B07 | Spot 中断（FIS）、路径 B 的 spot 与透传参数 | 需要 FIS 角色和中断队列（`enable_spot_interruption_queue`），未写 |
| KRP-B03～B06、B08 | 路径 B 的扩容、上限、回收、重试、孤儿标记 | 未写。路径 B 在发布镜像 ≤ 2.17.27 上不可用（G-08 及 9.6 列的两处）；本分支上 `KRP-B02` / `KRP-B10` 已在 KWOK 档跑通，这几条可以接着写 |
| NEG-01、NEG-08、NEG-09、NEG-10 | 超大请求、NodeClass 配错、无 Karpenter CRD、托管节点组 GPU | 未写；NEG-09 需要没有 Karpenter 的集群，NEG-10 需要 `gpu_static_count > 0` |
| CLD-01 界面部分、CLD-02 | 控制台里出现集群、用控制台模板建配置 | 需要账号和人看界面；`CLD-01` 的集群侧已自动化 |
| UPG-01、UPG-02 | 同架构 Helm 升级、v1 → v2 升级 | 需要两个版本的镜像 / 独立的 v1 环境 |
| UPG-04 | 销毁前无残留 | 由 `make down` 的 `sweep` 承担 |
| 非 NVIDIA 厂商、NVLink、GPU 掉卡、`Provisioned` 模式、副本数伸缩、reBalancer / auto-freeze | | EKS 上没有对应硬件或 API，或功能未实现（见早先清单第 6 节的原因，仍然成立） |

## 9. 已执行的结果

数据在 `results/`（不入库）。运行环境：EKS 1.36.4、Karpenter 1.14.1、chart 1.8.8（仓库 checkout）、operator / hypervisor `2.17.23`、`vgpu-provider-nvidia:1.3.9`。

### 9.1 环境与自毁开关

见 `terraform/README.md`「实测耗时与成本」。摘要：`make up` 13m43s、15m21s（第 2、3 轮，一次成功）；`make down` 节点还在时 12m07s，节点已被清掉时 5–6 分钟；三轮里有两轮销毁后 `sweep` 直接为空，第 2 轮剩 2 个不计费的 launch template（已修）；自毁开关两次都在过期后 6–8 分钟内清掉全部计算资源；三轮合计花费估算约 $1.1。

### 9.2 `smoke-nogpu`（三轮，每轮约 7 分钟）

| 运行 | 结果 |
|---|---|
| smoke-1（首轮，含测试自身的问题） | 22 通过、1 known-fail、3 失败。3 个失败：`KRP-C01`（被 G-01 拉起的残留 CPU 节点干扰，后加了本栈的防护 overlay）、`NEG-11`（预期写错，实际行为见 G-10）、`OBS-06`（零 GPU 时没有任何 `tf_*` 表，见 G-06，改为记录现状） |
| smoke-2 | **25 通过、1 known-fail（`WHK-08`）** |
| smoke-3（第 2 轮环境） | 24 通过、2 失败：`WHK-08`（known）和 `WHK-05`（这一轮 4 个副本里 3 个被接入，G-17，此后标为间歇性 known-fail） |

`CLD-01`（smoke-3 起）：未配置 agent，按设计跳过。

### 9.3 `rehearsal`（mock GPU，CPU 实例）

三次运行：rehearsal-1 因测试没有等 GPU CR 记账而中断；rehearsal-2 是完整的一轮（18 条，68 分钟）；rehearsal-3 把 `consolidateAfter` 调到 10m 后重跑了与扩容有关的 3 条，并补跑了后加的 2 条。

| ID | 结果 | 实测 |
|---|---|---|
| INS-03 | 通过 | mock GPU 上报 312 TFLOPS / 16GiB；hypervisor 的 owner 含 GPUNode 与 Node；节点有 16 个 index 资源 |
| ISO-01 | 通过 | 3 个 20% 的 soft Pod 在同一张卡；GPU CR 记账延迟 5.4s；删除后 10.7s 归还 |
| ISO-05 | 通过 | 整卡被占满后，分片 Pod 不会落到这张卡 |
| ISO-14 | known-fail | busybox 在 soft 下以 127 退出：`libstdc++.so.6: cannot open shared object file` |
| SCH-01 | 通过 | 两个节点时 4 个 20% 的 Pod 全在一个节点 |
| SCH-04 | 通过 | 型号匹配的 Pod 调度成功；不存在的型号不调度、不新建 NodeClaim（事件 `NodeExpansionFailed: no potential GPU nodes can fit`） |
| SCH-10 | 通过 | 超配额的 Pod 被拦；额度内的 Pod 运行且 `status.used` 正确 |
| SCH-11 | 通过 | 2 个 gang 成员 7.5s 后一起 Running；`status.gang.phase=Running` |
| WHK-01、WHK-07、INS-04 | 通过 | workload 0.2s 回收，GPU 资源归还 |
| NLC-01 | 通过 | hypervisor 23s 恢复；期间 GPU 经过 `Pending`；业务 Pod 未重启，分配量不变 |
| NLC-02 | 通过 | operator 21s 恢复；`/api/allocation` 重启前后一致；无重复分配 |
| NLC-09 | 通过 | 删除节点后 GPUNode / GPU CR 0.6s 内清理；Pod 59s 后在新节点 Running |
| OBS-02 | known-fail | `tf_node_metrics`、`tf_pool_metrics`、`tf_worker_resources` 有数据；`tf_gpu_usage`、`tf_worker_usage` 6 分钟内无数据 |
| KRP-A08 | 通过 | 冷启动：NodeClaim 6s、GPU 注册 51s、Pod Running 56s |
| KRP-A02 | 5m 时失败，10m 时通过 | 扩容器 12.5s 建出 NodeClaim（owner 为 NodePool、带 `expansion-source`、不带 `do-not-disrupt`），恰好 1 个；`consolidateAfter=5m` 时节点先被回收、Pod Pending 12 分钟超时；`=10m` 时 Pod 381s 后 Running，其中「节点 Initialized → Pod 绑定」330s |
| KRP-A16 | known-fail | 330s（要求 < 120s） |
| KRP-A03 | 通过 | 3 个 Pod 只新增 1 个 NodeClaim，同节点同卡，62.6s 全部 Running（这一次调度重试赶上了） |
| KRP-A09 | 通过 | 空节点 46s 内消失；先被 TF 打了 `should-delete` |
| FIN-01 | 通过 | 见下 |

路径 A 的另一处实测：mock NodePool 的 spot 配额用尽时，扩容器的第一个 NodeClaim 因容量不足失败（事件 `NodeExpansionCandidatesExhausted`），调度器重排后扩容器建出的下一个 NodeClaim 落在 **on-demand**（t3a.medium）——spot → on-demand 的回退在 TF 扩容路径上也走通了。

FinOps（`FIN-01`，mock，按 g4dn.xlarge 按需 $0.526/h 代入，每个场景稳态保持 3 分钟）：

| 方案 | 节点数 | 节点小时 | 成本 | 冷启动到全部 Running |
|---|---|---|---|---|
| 3 个分片负载（各 20% 算力、2Gi） | **1** | 0.149 | $0.079 | 364s（其中约 5 分钟是 G-16 的调度重试等待） |
| 3 个整卡负载（基线） | **3** | 0.219 | $0.115 | 126s |

稳态下是 1 台对 3 台（节省 67%）。这次短窗口测得的节省只有 31.7%，因为分片方案的节点小时里包含了 G-16 造成的 5 分钟空等。真实 GPU 上应重测。

### 9.4 `dryrun`

| ID | 结果 | 实测 |
|---|---|---|
| DRY-PERF-01 | 通过 | stand-in 平滑限流：372 → 189 → 95 tokens/s（100% / 50% / 25%），TPOT、ITL 按 1/s 放大，全部判据通过。阶梯放行 @25%：吞吐 79（比例达标），TPOT p50 80.7ms（成比例），但 ITL p99 399ms（p50 20ms）、TTFT p95 436ms → 判出双峰与 ITL 断崖。注入 1s 停顿：29 次 ≥0.9s → 判出。不限流 @25%：吞吐 367 → 判出「不成比例」「限额无效」 |
| DRY-PERF-02 | 通过 | CPU 上 ORT（tiny_mlp、minilm）、torch mini（tiny_mlp、mobilenet_v2 × eval / train）、MLPerf LoadGen（Offline、Server）共 8 个结果点；pip 安装 + 运行约 3.5 分钟 |
| DRY-ISO-01 | 通过 | 11 份 GPU 清单全部通过准入；dry-run 留下 10 个 `TensorFusionWorkload`（即 G-02） |
| DRY-IMG-01 | 通过 | PyTorch 3161 MiB、vLLM **11877 MiB**、operator 102 MiB、hypervisor 71 MiB、worker 33 MiB、client 31 MiB、vgpu-provider 4 MiB（均为 linux/amd64 压缩大小）；两个 Qwen 模型的 `config.json` 无 token 可访问 |
| KRP-A08（dry-run） | 以 dry-run 结论跳过 | 不带 nodeSelector 的 TF Pod → Karpenter 为 `tf-gpu` NodePool 建 NodeClaim → spot：`MaxSpotInstanceCountExceeded`；on-demand：`VcpuLimitExceeded … current vCPU limit of 0` |
| KRP-A15 | known-fail | 去掉防护 overlay 后 38s 内出现 `tf-cpu` 的 NodeClaim（c5.large） |
| KRP-B02 | 跳过 | 需要修复后的镜像 |

另有 `make gpu-dryrun`：4 种机型 × 2 个 AZ × 按需 / spot 共 16 种组合的 `RunInstances --dry-run` 全部可启动。

### 9.5 `disruptive`

`UPG-03`：卸载脚本 190s 完成，12 个 CRD、webhook、namespace、NodeOverlay 均消失，节点上没有残留的 `tensor-fusion.ai/*` label 与容量。

### 9.6 `kwok`（fake-gpu，2026-10-12）

环境：kind v0.33.0 / Kubernetes 1.36.4、KWOK v0.8.0、Karpenter v1.14.1 KWOK provider（未修改的上游源码，节点名不再缩短）、chart 为仓库 checkout（1.8.9）、hypervisor 镜像名 `2.17.27`（假节点上不运行）。两轮：第一轮（上午）operator 为发布镜像 `2.17.24` / `2.17.25`、节点名缩短；第二轮 operator 为**本分支构建的开发镜像**（发布镜像 `2.17.27` 的运行层 + 本分支的 `/manager`），全档 16 条一次跑完 1156 秒，另有两条单独重跑。`up.sh` 从零到可用 72–76 秒。fake hypervisor 延迟 20s，`nodeRegistrationDelay=10s`，GPU NodePool `consolidateAfter=1m`。全部是控制面结论（mode `fake-gpu`，见第 2 节）；耗时由 5 秒轮询得到，精度 5 秒。

| ID | 结果 | 实测 |
|---|---|---|
| KRP-A02 | 通过（2.17.24 两次、2.17.25 一次、本分支一次） | 一个 Pending Pod 恰好 1 个 NodeClaim（扩容器创建，NodePool `tf-gpu`）；NodeClaim 10–15s，Pod Running 45s；**GPU CR Running → Pod 绑定 < 1s**（500ms 采样、条件时间戳 1s 精度，读数为 0）；节点 Initialized → Pod 绑定 19–20s，等于 fake hypervisor 的延迟 |
| KRP-A16 | 通过 | 同上两个数，判据 < 120s 与 < 30s。修复前（2.17.23，mock，EKS）为 330s；这一档没有在 2.17.23 上对照 |
| KRP-A17 | 通过（2.17.24、2.17.25、本分支各一次） | 4 个整卡 Pod → 恰好 4 个 NodeClaim、4 个节点，45s 全部 Running，之后 60s 内没有多余的 NodeClaim |
| KRP-A09 | 通过 | 本分支：1 个空节点 65s 消失；2.17.25：4 个 90s；2.17.24：1 个 75s。都是 NodePool 的 `consolidateAfter` 先到，没有出现 `should-delete` |
| KRP-A15 | 通过（2.17.24、2.17.25、本分支各一次） | overlay 的 requirements 限定为 GPU 机型族之后，Pending 的 TF Pod 90s 内没有让 `tf-cpu` 出现 NodeClaim。chart 默认的 requirements 用的是 AWS provider 的 label，AWS 上仍未复测 |
| KRP-A08 | 通过（本分支，一次） | 从 0 个 GPU 节点冷启动：Pending → Karpenter 自己的 NodeClaim 5s → GPU 注册 35s → Pod Running 35s（用例总时长 221s，其余是等上一条用例的节点被回收） |
| KRP-A03 | 通过（本分支，一次） | 3 个 20% 的 Pod → 1 个 NodeClaim、1 个节点、同一张卡，45s 全部 Running；之后 node-scaler 的 in-flight 状态为空 |
| FIN-01 | 通过（本分支，一次） | 3 个分片负载用 1 个节点（35s 全部 Running），整卡基线用 3 个节点（75s）；按节点小时 × 按需价折算节省 71.5%。稳态窗口只有 45s，节点小时是 0.022h 对 0.077h：比例有意义，绝对金额没有 |
| NLC-09 | 通过（本分支，一次） | 删除节点后 GPUNode / GPU CR 5s 内清理，Deployment 的 Pod 5s 内在另一台节点 Running（另一台节点当时已存在；假节点上没有真实的重建耗时） |
| KRP-C01 | 通过（本分支；第一次失败，改用例后重跑通过） | `tf-cpu` 5s 出 NodeClaim，15s 节点 Initialized 且 Pod Running（t3.large，spot），删除后 50s 节点消失。第一次失败是用例的前提不成立：它断言 GPU NodePool 里一个 NodeClaim 都没有，而全档跑时前一条用例留着一台 GPU 节点；已改成「不新增 GPU NodeClaim」 |
| NLC-10 | 通过（本分支；全档那次跳过，改用例后重跑通过） | 节点 `kwok-tensor-fusion-<10 位>-<词>-17-<随机串>-<uint32>` 共 68 个字符：GPU 注册成功，`managed-by` 为截断前缀加 10 位哈希，Pod 45s Running。全档那次跳过是因为用例只在现有节点里找长名字，而冷启动节点（`kwok-tf-gpu-…`）不到 63 个字符；已改成由扩容器逼出一台 |
| KRP-B02 | **发布镜像 2.17.27 + chart 1.8.9：失败**；本分支：通过（两次） | 2.17.27：GPUNodeClaim、同名 NodeClaim、节点在 20s 内依次出现，节点带 `node-provisioner` label，之后没有 GPUNode、没有 hypervisor Pod，GPUNodeClaim 的 phase 一直为空（见下）。本分支：水位 → GPUNodeClaim 5s / 20s → Bound 且 GPU Running 35s / 55s（两次的差别是 pool 首次调谐落在哪个周期）→ 指定该 pool 的分片 Pod 5s Running，落在这台节点（名字 68 / 71 个字符）这张卡上 |
| KRP-B10 | 通过（本分支，两次） | 删除 pool 后 10s 内 GPUPool、GPUNodeClaim、NodeClaim、节点、GPUNode、GPU 全部消失。2.17.27 上同一操作 152s 后 GPUNodeClaim 仍在（带 finalizer 与 deletionTimestamp） |
| INS-03 | 通过 | 冷启动到 GPU Running 35s；节点带 16 个 `index_*`、归属 NodePool |
| KRP-00、KRP-01 | 通过 | NodePool / KWOKNodeClass Ready，`NodeOverlay=true`；overlay 被 Karpenter 接受，NodeClaim RBAC 就位 |

路径 B 在这一档查出的三处，都不是 KWOK 的假象：

1. chart 的 ClusterRole 没有 Karpenter NodeClass 的读权限（chart 1.8.9 已修）。
2. `NodeReconciler` 只按 pool 的 `nodeSelector` 把节点归到 pool。供给型 pool（`Karpenter` / `Provisioned`）没有 `nodeSelector`，它拉起的节点匹配不到任何 pool，于是没有 GPUNode、没有 hypervisor、GPUNodeClaim 永远不 Bound。修复：节点带 `node-provisioner` label 时，归属到该 GPUNodeClaim 的 pool，`nodeSelector` 匹配作为回退。真实 provider 在这一点上与 KWOK 没有区别——label 都是从 NodeClaim 透传到节点的。控制器的 envtest 之所以一直是绿的，是因为它给供给型 pool 同时配了 `nodeSelector` 和与之匹配的 `gpuNodeLabels`。
3. pool 的 finalizer 删除了它的 GPUNodeClaim，却只等 GPUNode 消失；没有 GPUNode 时 pool 立即被删，而 GPUNodeClaim controller 在 pool 不存在时直接返回，claim 的 finalizer 永远没人摘。修复：pool 等到它的 GPUNodeClaim 也都消失。

节点名超过 63 个字符：hypervisor 给每个 GPU CR 打 `tensor-fusion.ai/managed-by=<节点名>`，label 值上限 63 字节，上午那一轮 fake hypervisor（写法与真实 hypervisor 相同）被 API server 拒绝：`metadata.labels: Invalid value: "kwok-tensor-fusion-…": must be no more than 63 bytes`。修复后 label 值是 `utils.LabelSafeValue(节点名)`：放得下就是节点名本身（已有对象不变），放不下就是前缀加 10 位 SHA-256；写入方（hypervisor）与三处 selector（GPUNode controller、allocator 的容量刷新、hypervisor 清理过期 GPU）用同一个函数，需要节点名的读取方改从 GPU 的 GPUNode ownerReference 取。**真实 hypervisor 二进制没有在长节点名上跑过**：这一档的写入方是 fake hypervisor，证明的是 operator 一侧和 API server 的校验；hypervisor 一侧只有单元测试（100 与 253 个字符）。同类但没有改的：pool 名出现在 label 键 `tensor-fusion.ai/pool-<pool 名>` 里（键名上限 63，即 pool 名上限 58）和多个 label 值里；`tensor-fusion.ai/workload`、`worker-name` 的值是 workload / Pod 名，被多处当作名字读回。它们超长时仍会被 API server 拒绝。

### 9.7 镜像拉取的结论

没有上 ECR pull-through cache：TensorFusion 的镜像在 Docker Hub，pull-through 需要把 Docker Hub 凭证放进 Secrets Manager；而这些镜像一共不到 250 MiB，mock 节点上从节点就绪到 hypervisor Running 约 15 秒。值得关心的只有负载镜像：PyTorch 3.2 GiB、vLLM 11.9 GiB。对策是每个节点只拉一次（perf 用例在同一节点上串行）、GPU 节点根卷吞吐调到 300 MiB/s、模型与 pip 包缓存在节点 hostPath。真实拉取耗时会由 Pod 事件记录，首次 GPU 运行后再决定是否值得预拉。
