# TensorFusion 全栈上下文索引

本文说明本仓库在整个 TensorFusion 技术栈中的位置：它部署哪些组件、这些组件由哪个仓库构建、与控制台（Cloud Console）如何通信。目的是让只打开本仓库的工程师或 AI agent 能找到其余上下文。

- 整理日期：2026-10-11。除本仓库外的内容来自对相邻仓库的只读梳理，标注"推断"的条目没有直接证据。
- 本仓库是公开仓库。**不要**在这里写入任何密钥、token、连接串、内网地址、云账号 ID，也不要写未修复安全问题的利用细节。此类内容放在本地不入库的 `tmp.*` 文件里（`.gitignore` 已忽略 `tmp*`）。

## 1. 组件总览

```
                         ┌──────────────────────────────────────────────┐
                         │ Cloud Console  app.tensor-fusion.ai          │
                         │ (tensor-fusion-cloud，非公开)                │
                         └───────────────▲──────────────────────────────┘
                                         │ WebSocket（集群主动外连，无需入站）
┌────────────────────────────────────────┼─────────────────────────────────────┐
│ 用户 Kubernetes 集群                   │                                     │
│  ┌─────────────────────────────────────┴───────────────┐                     │
│  │ controller Deployment（本仓库 cmd/main.go）         │                     │
│  │  ├ controller：CRD 控制器 + Pod webhook + 内嵌调度器│                     │
│  │  ├ vector：指标日志 → GreptimeDB                    │                     │
│  │  └ cluster-agent（可选 sidecar，来自 cloud 仓库）   │                     │
│  └─────────────────────────────────────────────────────┘                     │
│  ┌──────────────────────────┐   ┌───────────────────────────────────────┐    │
│  │ GreptimeDB（tf_* 指标表）│   │ Karpenter / 云厂商 API（节点伸缩）    │    │
│  └──────────────────────────┘   └───────────────────────────────────────┘    │
│                                                                              │
│  每个 GPU 节点：                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │ hypervisor DaemonSet Pod（本仓库 cmd/hypervisor，Go，CGO 关闭）       │   │
│  │  ├ purego 加载厂商 accelerator 库（vgpu-provider）                    │   │
│  │  ├ device plugin / kubelet checkpoint 对接                            │   │
│  │  └ 写每个 Pod 的共享内存限额文件 /run/tensor-fusion/shm/<ns>/<pod>/   │   │
│  ├───────────────────────────────────────────────────────────────────────┤   │
│  │ 业务 Pod（webhook 注入）                                              │   │
│  │  ├ soft：/etc/ld.so.preload → libcuda_limiter.so（vgpu-provider）     │   │
│  │  ├ hard / sidecar：client 库 + worker sidecar（tensor-fusion-core）   │   │
│  │  └ remote：client 库 ──TensorFusionConnection──▶ 独立 worker Pod      │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 2. 本仓库

Go module `github.com/NexusGPU/tensor-fusion`（GitHub 上仓库名同为 `NexusGPU/tensor-fusion`，旧名 `tensor-fusion-operator` 仍可重定向）。Kubebuilder 结构，依赖 `k8s.io/kubernetes v1.36.1`（vendor 后打补丁）、`controller-runtime v0.24.0`、`sigs.k8s.io/karpenter v1.8.2`。

### 2.1 目录索引

| 路径 | 内容 |
| --- | --- |
| `cmd/main.go` | controller 入口：控制器、webhook、调度器、HTTP server 全在一个进程 |
| `cmd/sched/` | 内嵌 kube-scheduler 的装配 |
| `cmd/hypervisor/` | 节点 hypervisor 入口；`shm_init/` 是挂载 shm 的子命令 |
| `cmd/hypervisor-tui/` | hypervisor 的终端调试界面 |
| `api/v1/` | CRD 类型，独立 Go module（`github.com/NexusGPU/tensor-fusion/api`） |
| `internal/controller/` | 各 CRD 控制器；节点伸缩见 `gpupool_node_provision.go`、`gpunodeclaim_controller.go`、`gpupool_compaction_controller.go`、`gpupool_defrag.go` |
| `internal/webhook/v1/` | Pod mutating webhook，注解解析在 `tf_parser.go` |
| `internal/utils/compose.go` | 生成 hypervisor / worker / client / limiter 注入的 Pod 片段，所有镜像、环境变量、挂载的拼装点 |
| `internal/scheduler/` | 调度插件：`gpuresources`（GPU 资源过滤与预留）、`gputopo`（拓扑）、`expander`（Pending Pod 触发扩容） |
| `internal/gpuallocator/` | GPU 分配状态机，调度器与控制器共用 |
| `internal/cloudprovider/` | 节点供给：`karpenter/`、`aws/`、`alibaba/`、`pricing/` |
| `internal/autoscaler/` | 纵向自动伸缩（限额推荐与原地调整） |
| `internal/quota/`、`internal/gang/` | 命名空间配额、gang 调度 |
| `internal/metrics/`、`internal/alert/` | 指标编码写入（经 vector 到 GreptimeDB）与告警评估 |
| `internal/server/` | controller 的 HTTP API，含 client 取连接信息的 `/api/connection` |
| `internal/provider/` | ProviderConfig 管理：按厂商解析镜像与硬件信息 |
| `pkg/hypervisor/` | hypervisor 实现，见 2.3 |
| `pkg/constants/` | 注解名、环境变量名、厂商映射（`vendors.go`） |
| `provider/` | accelerator / limiter 的 C ABI 头文件与 example 实现（`make build-provider`） |
| `charts/tensor-fusion/` | Helm chart；`values-cn.yaml`、`values-production.yaml`、`values-multi-vendor.yaml` 是叠加文件 |
| `config/` | kustomize 清单、CRD、samples |
| `patches/`、`scripts/patch-scheduler.sh` | 对 vendored kube-scheduler 的补丁，`make vendor` 时应用 |
| `test/e2e/`（Kind）、`test/sched/`（调度集成与基准） | 仓内测试 |
| `e2e/` | AWS EKS 演示与端到端用例清单、Terraform |
| `docs/`、`design/` | 设计与运维文档，见第 6 节 |

### 2.2 CRD（`tensor-fusion.ai/v1`）

`TensorFusionCluster` → `GPUPool` → `GPUNode` → `GPU` 是资源层级；`GPUNodeClaim`、`GPUNodeClass` 用于节点供给；`TensorFusionWorkload`、`WorkloadProfile`、`TensorFusionConnection` 描述工作负载；`SchedulingConfigTemplate`、`GPUResourceQuota`、`ProviderConfig` 是策略与厂商配置。

关键枚举：
- 隔离模式（`api/v1/workloadprofile_types.go`）：`shared`、`soft`、`hard`、`partitioned`。
- 池的节点供给模式（`api/v1/gpupool_types.go`）：`Provisioned`、`AutoSelect`、`Karpenter`。
- QoS：`low`、`medium`、`high`、`critical`。

### 2.3 hypervisor（`pkg/hypervisor/`）

| 子目录 | 职责 |
| --- | --- |
| `device/` | 通过 purego 调用厂商 accelerator 库，设备发现、指标、分区（MIG） |
| `backend/kubernetes/` | device plugin、Pod 缓存、kubelet checkpoint、GPU / GPUNode CR 上报 |
| `backend/single_node/` | 非 Kubernetes 的单机模式，文件状态；被 `gpu-go` 作为库引用 |
| `worker/` | worker 生命周期与设备分配；`state/` 是 soft limiter 的共享内存布局；`computing/quota_controller.go` 是算力限流控制环 |
| `server/` | 节点 HTTP API，`handlers/legacy.go` 保留 vgpu.rs 时代的接口 |
| `metrics/`、`tui/`、`framework/`、`api/` | 指标、调试界面、接口定义 |

`pkg/hypervisor` 和 `api/v1` 被 `gpu-go` 作为 Go 库依赖，改这两处的导出接口等同于改公共 API。

### 2.4 Karpenter 兼容的两条路径

二者互斥，由 GPUPool 配置决定：
- **`AutoSelect` + Karpenter NodePool**（chart 默认）：GPU 节点归 Karpenter 管；`internal/scheduler/expander` 发现 Pending Pod 后直接创建 Karpenter `NodeClaim`；缩容时只给节点打 `tensor-fusion.ai/should-delete=true`，实际回收交给 Karpenter consolidation。
- **`provisioningMode: Karpenter`**：TensorFusion 按池的容量配置创建 `GPUNodeClaim`，再生成同名 `NodeClaim`（带 `karpenter.sh/do-not-disrupt`），回收由 TensorFusion 自己的 compaction 负责。

代码入口：`internal/cloudprovider/karpenter/`、`internal/controller/gpunodeclaim_controller.go`、`charts/tensor-fusion/templates/node-overlay.yaml`。

### 2.5 构建、测试、发布

- `make build` / `make build-hypervisor` / `make build-provider`；`make vendor` 会重新打调度器补丁。
- `make test`（envtest，覆盖率门槛见 Makefile）、`make ut F=<文件>`、`make ut-sched`、`make test-e2e`（Kind）、`make lint`。
- `.github/workflows/release.yml` 同时发布 operator 与 hypervisor 两个镜像（同一版本号）；`release-helm.yml` 发布 chart（tag 形如 `tensor-fusion-helm-<ver>`）；`release-api.yml` 发布 `api/` 模块。

## 3. 组件来源：谁构建了什么

| 镜像 / 产物 | 引用位置 | 来源仓库 |
| --- | --- | --- |
| `tensorfusion/tensor-fusion-operator` | `values.yaml` `controller.image` | 本仓库，`dockerfile/operator.Dockerfile` |
| `tensorfusion/tensor-fusion-hypervisor`（2.x） | `values.yaml` `cluster.hypervisorImage` | 本仓库，`dockerfile/hypervisor.Dockerfile` |
| `tensorfusion/vgpu-provider-nvidia` | `values.yaml` `providerConfigs.nvidia.images.middleware` | `vgpu-provider` |
| `tensorfusion/tensor-fusion-client`、`tensor-fusion-worker`（v2.x） | `values.yaml` `images.remoteClient` / `remoteWorker` | `tensor-fusion-core`（非公开） |
| `tensorfusion/tensor-fusion-agent` | `values.yaml` `agent.image` | `tensor-fusion-cloud` 的 `agent/` |
| vector、GreptimeDB、alertmanager、kube-webhook-certgen | `values.yaml` | 第三方 |
| `libaccelerator_example.so` | `cmd/hypervisor/main.go` 默认值 | 本仓库 `provider/` |

厂商到库文件的映射在 `pkg/constants/vendors.go`：

| 厂商 | accelerator 库 | soft limiter 库 | 来源 |
| --- | --- | --- | --- |
| NVIDIA | `libaccelerator_nvidia.so` | `libcuda_limiter.so` | `vgpu-provider` |
| AMD | `libaccelerator_amd.so` | 无 | `vgpu-provider`（仅骨架） |
| Ascend | `libaccelerator_ascend.so` | `libascend_limiter.so` | `vgpu-provider-internal`（非公开） |
| MooreThreads | `libaccelerator_musa.so` | `libmusa_limiter.so` | 同上 |
| PPU | `libaccelerator_ppu.so` | `libppu_limiter.so` | 同上 |

middleware 镜像里的 `/build/*` 有三个消费点：hypervisor 镜像构建时拷入；hypervisor Pod 的 `init-runtime` 容器拷到宿主机 `/run/tensor-fusion`；业务 Pod 的 `init-soft-limiter` 容器拷到 emptyDir 供 preload。

## 4. 相关仓库

本地开发时这些仓库通常与本仓库并列放在同一父目录。标注"非公开"的仓库外部读者无法访问，这里只说明它们与本仓库的接口关系，不描述其内部实现；它们发布的镜像是公开的。

### 4.1 数据面

| 仓库 | 定位 | 语言 | 状态 | 与本仓库的关系 |
| --- | --- | --- | --- | --- |
| `vgpu-provider`（非公开） | 各厂商 accelerator 库、NVIDIA soft limiter、`nvidia-smi` 包装脚本 | C/C++ | 活跃 | `provider/accelerator.h`、`provider/limiter.h` 是手工拷贝，不是 submodule；共享内存布局三方手工同步（见 5.1） |
| `vgpu.rs` | 最初的 Rust hypervisor + CUDA limiter | Rust | hypervisor 已被本仓库的 Go 实现取代（2026-01）；仍发布 `bootstrap`、`libadd_path.so` 和 worker 镜像内的 Rust limiter | 共享内存布局、`/api/v1/pod` 与 `/api/v1/process` 接口、ERL 参数沿用它的定义 |
| `tensor-fusion-core`（非公开） | API remoting 的 client 库与 worker，现行 v2.x | C/C++ | 活跃 | `client` / `worker` 镜像；读取 `TENSOR_FUSION_OPERATOR_GET_CONNECTION_URL`、`TF_CUDA_SM_PERCENT_LIMIT`、`TF_GPU_MEMORY_LIMIT` 等环境变量 |
| `tensor-fusion-worker`、`tensor-fusion-client`、`tensor-fusion-common`、`TensorFusion` | v1 时代的 client / worker 及更早的单体仓库 | C/C++ | 已被 `tensor-fusion-core` 取代（推断） | 无直接依赖；`compose.go` 仍识别旧容器名 |
| `ngpu` | client 与 worker 同进程的实验分支 | C/C++ | 停滞（推断） | `TENSOR_FUSION_NGPU_PATH` 等常量可能只有它读取 |

三种工作负载形态对应的注入（`internal/utils/compose.go`）：
- 本地 soft：只注入 vgpu-provider 的 limiter，无 worker。
- 本地 hard 或显式 sidecar：client init 容器 + worker sidecar，经 `/dev/shm` 通信。
- 远程：client init 容器 + 独立 worker Pod，通过 `TensorFusionConnection` 配对。
- `partitioned`、`shared`：都不注入。

### 4.2 控制台与运维

| 仓库 | 定位 | 与本仓库的关系 |
| --- | --- | --- |
| `tensor-fusion-cloud`（非公开） | 控制台 `app.tensor-fusion.ai` 及集群侧 `cluster-agent` | 见 5.2。其 CRD 类型由本仓库 `config/crd/bases` 生成；它提供的 TensorFusionCluster 模板和 Helm 安装命令需与本仓库的 CRD、chart values 保持一致 |
| `auto-ops`（非公开） | 部署与测试用的基础设施代码、基准用例 | 会安装本仓库的 chart；是基准与演示清单的来源；负责把镜像同步到国内镜像仓库，供 `values-cn.yaml` 使用 |
| `tensor-fusion-website`（非公开） | 官网、产品文档（`tensor-fusion-engine` 部分）、账号与鉴权 | Helm values、指标、注解、告警规则变更后需同步文档 |
| `tensor-fusion-site` | 早期的公开文档站 | CRD 参考由本仓库 `config/crd/bases` 生成 |
| `gpu-go` | "GPU Go" 产品的 CLI 与 agent | 以库方式引用本仓库 `pkg/hypervisor` 与 `api/v1`（单机模式） |
| `benchmark` | vGPU 基准的 Helm chart 与结果 | 不被本仓库消费 |

### 4.3 常见的并列第三方仓库

`kubernetes`（阅读调度器源码、编写 `patches/`）、`kube-scheduler-simulator`、`alibaba-cloud-csi-driver`、`libnvidia-container`、`iceoryx2` 等是参考用的克隆，不属于本栈。

## 5. 跨仓库契约

### 5.1 soft 隔离的共享内存

- 路径：宿主机 `/run/tensor-fusion/shm/<namespace>/<pod>/shm`，以 `subPathExpr` 挂进业务容器，容器内由 `TF_SHM_PATH` 指向。
- 布局定义有三份，必须同时修改：本仓库 `pkg/hypervisor/worker/state/shm_layout.go`（令牌操作在 `shm_erl.go`）、`vgpu-provider` 的共享内存布局头文件、`vgpu.rs` 的 `crates/utils/src/shared_memory/mod.rs`。布局中没有 magic 或版本字段。
- hypervisor 写入限额、心跳、显存用量和令牌桶参数；limiter 在进程内读取并扣减令牌。控制环的驱动在 `pkg/hypervisor/worker/computing/quota_controller.go`，速率控制在 `quota_rate.go`，计量与校准在 `quota_meter.go`，分摊在 `quota_attribution.go`，补充在 `quota_refill.go`，配置在 `quota_config.go`。
- 限额字段 `up_limit`、`mem_limit` 可以在映射存活期间被原地改写（autoscaler 原地调整、ProviderConfig 硬件表变更）。写入时机有三类：映射被打开或重新打开时；worker 更新、设备容量更新这两种事件；以及 500 ms 同步环的每一轮（值没有变化时不写，也不分配内存）。每次写入都在同一个临界区里按当时的 worker 与设备容量重新推导，因此事件到达的先后不影响结果，任何原因造成的不一致会在一个同步周期内被纠正。limiter 必须每次从映射读取，不能只在启动时读一次。
- 令牌桶的两个节拍：控制环每 500 ms 根据利用率调整 `erl_token_refill_rate` 与 `erl_token_capacity`；补充环每 10 ms（`erlRefillInterval`）按当前速率发放令牌，使被限流的进程是均匀变慢而不是每个控制周期跑一段、停一段。桶容量是 `速率 × 0.1 s`（`burstWindow`）：空闲的 worker 最多攒下相当于其配额 0.1 秒的工作量，恢复运行时一次性用掉。
- 突发窗口 W（完整规则写在 `quota_config.go` 中 `burstWindow` 上方的注释里）：W 是 worker 空闲时最多可以攒下的、按它自己的补充速率计的秒数，桶容量 = `速率 × W`。取值顺序：Pod 注解 `tensor-fusion.ai/compute-burst-window`（Go duration，如 `500ms`、`2s`，可在运行中的 Pod 上原地修改，一个控制周期内生效，不需要重启）；否则是池级 `SchedulingConfigTemplate` 的 `elasticRateLimitParameters.burstWindow`（秒，字符串；经 `TF_HYPERVISOR_SCHEDULING_CONFIG` 在 hypervisor 启动时读取，不计入模板 hash，修改后要重建 hypervisor Pod 才生效）；否则是编译进 hypervisor 的默认值 0.1 s。范围 0 到 10 s（`constants.ComputeBurstWindowMax`）；桶至少保留两个补充周期（20 ms）的量，所以 W = 0 表示“几乎不攒”。webhook 在创建 Pod 时拒绝格式错误、为负或超过上限的值；hypervisor 对之后被改动的注解再做一次校验（越界值取最近的边界，无法解析的值按未设置处理，各记一次日志）。长期配额不随 W 变化：突发只花掉自己按配额速率攒下的令牌，速率不会因为 W 被调高，空闲 worker 的速率也不会上涨；任意 T 秒内最多得到 `配额 × (T + W)` 的设备时间。配额为 s 的 worker 在攒满之后，独占设备时可以全速运行约 `s × W / (1 − s)` 秒（W = 2 s 时，25% 约 0.67 s，50% 约 2 s），之后回到均匀限速；攒满需要空闲 W 秒。透支规则不变（见下一条的 `debtBound`）。调大 W 时桶按速率逐渐攒满，调小时超出新容量的部分在下一次补充时被去掉，其余不动。limiter 侧不需要改动：容量只是映射里的一个数。
- 令牌补充协议（`SharedDeviceInfoV2.RefillERLTokens`，布局不变，只用已有字段上的原子操作）：读 `erl_last_token_update`，按同一时钟算出经过的时间；用 CAS 把时间戳推进到当前时刻；只有 CAS 成功的一方把 `速率 × 经过时间` 加到 `erl_current_tokens`，且不超过容量。时间戳不会被往回拨；远在未来（超过 5 s）的时间戳被重置为当前时刻且不发放令牌；单次最多记 5 s。所有会补充令牌的写入方都必须遵守：hypervisor 的两个环，以及发现无人补充（时间戳陈旧约 100 ms 以上）时自行补充的 limiter 进程。这样一段时间只会被记账一次，交接时不丢也不重复。时间戳是 Unix 秒（f64）。算术的共同定义是测试向量 `pkg/hypervisor/worker/state/testdata/erl_refill_vectors.json`（Go 侧由 `TestRefillVectors` 执行，limiter 侧的测试应使用同一份）；字段的字节布局由同目录 `soft_limiter_refill_test.go` 中的 golden 值固定。
- 余额约定（共享债务）：`erl_current_tokens` 是 Pod 内所有进程共用的账户。余额大于 0 时 limiter 放行并扣除全部成本，余额因此可以为负（有界透支）；余额不大于 0 时不放行任何工作，补充的令牌先偿还债务。hypervisor 把任何有限值都当作合法余额，只有 NaN / Inf 视为无效并按空桶处理；透支上限是 `max(capacity, rate × 60 s × 100 / up_limit)`（整卡全速运行 60 秒的成本，且不小于一个桶），完整规则写在 `pkg/hypervisor/worker/computing/quota_rate.go` 中 `debtBound` 上方的注释里，limiter 与 hypervisor 共用这一段文字：limiter 扣费不会把余额扣到 `-bound` 以下；hypervisor 只把低于 `-bound` 的余额抬到 `-bound`，并且在债务未清期间上限不会缩小（取自上次余额非负以来发布过的最大值）。余额不是数字、涨幅超过 hypervisor 自己补充的量（加上 limiter 按已发布速率自补的量）、或低于透支上限时，该桶在本周期及随后 20 个周期按“未计量”处理；同一设备上只要有一个受限 worker 未计量，这个周期整台设备的令牌分摊就视为未知（速率保持、继续补充），不会把设备利用率分给其余 worker。时钟回拨：补充环连续三次发现时间戳在未来 50 ms 以上时，从该时间戳重新开始计时且不补发令牌（`RestartERLRefillClock`）。hypervisor 用“上次余额 + 期间自己补充的量 − 当前余额”计量 worker 的消耗；发现别人（limiter 自补充）推进过时间戳的那个周期不计量。
- 心跳：`last_heartbeat`（Unix 秒）由独立的 500 ms 环写入（`pkg/hypervisor/worker/heartbeat.go`），只做一次原子写，不经过任何 provider 调用或 API 调用；映射被打开时也立即写一次。hypervisor 正常退出时不写任何字段：心跳自然过期后 limiter 按已发布的速率自行补充，重启后的控制环接管已发布的速率并保持 10 秒不调整。
- 显存采样时间 `pod_memory_sampled_at[i]`（仅 V2 段）：占用 V2 尾部 512 字节 padding 的前 128 字节，每个设备槽位一个 8 字节字，位于段起始 `34992 + 8 × i`（与 `devices[i]` 同下标；该槽位的 `pod_memory_used` 在 `96 + 136 × i`）。段大小仍是 35504 字节，其余偏移不变，剩余 padding 从 35120 开始，仍然没有版本字段。取值是 Unix 秒的 f64 位模式，与 `erl_last_token_update` 同一时钟、同一编码；0 表示没有发布时间。它是产生这个 `pod_memory_used` 的那次进程查询开始之前读到的墙上时钟，不是查询之后，也不是写入的时刻，limiter 据此把“采样之后本进程分配的显存”加到采样值上。没有序号字段，协议就是唯一写入方 hypervisor 对每个槽位的三次原子写的顺序：时间写 0 → 写 `pod_memory_used` → 写时间。limiter 依次读时间、用量、时间，两次时间相同且不为 0 才把（用量，时间）当作同一次采样，否则重读，若干次后只用用量。同一槽位上新采样的时间一定与此前发布的不同且不回退：时钟没有前进或被回拨时发布 `nextafter(上一次, +∞)`，“上一次”取自映射本身，所以 hypervisor 重启后依然成立；映射里已经是同一组（用量，时间）时不重复写。采样失败、以及设备进程列表读不到且仍在保留期内时，两个字段都不写。不是由某次查询得到的用量不带时间：新建或重建的段、`SetPodMemoryUsed`、设备连续 60 s 读不到之后记为 0 的那次写入，时间都是 0，limiter 只拿用量比较，与没有这个字段时一样。兼容性：旧 limiter 把这 128 字节当作 padding，不读；旧 hypervisor（包括 `vgpu.rs`，它目前没有这个字段）一直把它们写成 0，新 limiter 读到 0 就按原来的方式工作，因此两种新旧搭配的行为都与扩展之前相同，两侧可以分别升级。Go 侧实现在 `soft_limiter_shm.go` 的 `SetPodMemorySample` / `GetPodMemorySample`，调用点是 `controller_shm.go` 的 `publishPodMemoryUsed`；偏移由 `soft_limiter_memory_sample_test.go` 中的 `TestPodMemorySampleFieldsKeepTheLimiterLayout` 固定，limiter 侧有对应的布局测试。
- 业务容器侧的环境变量：`TF_ISOLATION_MODE`、`TF_SHM_PATH`、`HYPERVISOR_IP`、`HYPERVISOR_PORT`、`POD_NAME`、`POD_NAMESPACE`、`CONTAINER_NAME`，定义在 `pkg/constants/env.go`。
- 设计背景：`docs/dynamic-gpu-isolation-design.md`、`docs/soft-isolation-preload.md`。

### 5.1.1 TFLOPS 的单位与设备容量

- TensorFusion 的 TFLOPS 只有一个口径：厂商公布的 FP16 TFLOPS，来自 ProviderConfig 的 `hardwareMetadata[].fp16TFlops`。请求、限额、GPU CR 的 `status.capacity.tflops`、GPUNode / GPUPool 的总量都是这个单位。
- accelerator 库上报的 `maxTflops` 是它自己的理论估算（NVIDIA 为 FP32：CUDA 核数 × 最高 SM 时钟 × 2，RTX 3090 为 44.08，A100 约 19.5），与 FP16（3090 为 71.2，A100 为 312）不可比。它在 hypervisor 内只以 `DeviceInfo.ProviderEstimatedTflops` 保留，仅在硬件表没有该型号时充当容量。
- hypervisor 内每个设备只有一个容量 `DeviceInfo.TflopsCapacity`，由设备控制器在发现 / 刷新设备时解析一次（`pkg/hypervisor/device/tflops_capacity.go`），硬件表变化时重新解析并通知所有订阅方。GPU CR 上报、节点总量、HTTP / TUI、限额换算都读这个字段，任何地方都不再直接除以厂商库的估算值。
- 硬件表通过 hypervisor Pod 的环境变量 `TF_PROVIDER_HARDWARE_METADATA` 下发；ProviderConfig 变更后 operator 按池的滚动策略重建 hypervisor Pod。单机后端（`backend/single_node`，`gpu-go` 使用）没有硬件表，容量就是厂商库的估算值，行为与之前一致。
- 容量来源写在 GPU CR 的注解 `tensor-fusion.ai/tflops-capacity-source` 上：`provider-config` 表示 FP16；`provider-estimate` 表示型号不在硬件表中、退回到了厂商库估算值。后者同时会在 hypervisor 日志中对每个设备告警一次。一个池里同时出现两种取值，说明该池混用了两种单位，需要把缺失型号补进 ProviderConfig。
- TFLOPS → 百分比只有一个函数 `utils.TflopsToComputePercent`（`internal/utils/resource.go`）：向上取整到整数百分比，范围 1–100。调度器（hard 模式写入 `tensor-fusion.ai/effective-hard-sm-percent` 注解）与 hypervisor（`computeUpLimit`，soft 模式写入共享内存 `up_limit`）都调用它，且除数是同一个容量。hard 模式优先使用调度器写入的注解；soft 模式在 hypervisor 内换算，原地调整限额时不需要第二个写入方。

### 5.1.2 accelerator 库的“未提供”标记

`provider/accelerator.h` 与 `vgpu-provider/accelerator/accelerator.h` 保持逐字一致。

可选特性通过 `AccelSetFeatures(requested, &enabled)` 协商。这个函数是后加的，hypervisor 把它当作可选符号解析（`pkg/hypervisor/device/accelerator_unix.go` / `accelerator_windows.go`），并在每次 `AccelInit` 成功之后、开始轮询之前调用一次（`AcceleratorInterface.initialize`）；`AccelShutdown` 会清除选择。hypervisor 声明自己理解哪些特性的唯一位置是 `pkg/hypervisor/device/accelerator.go` 的 `requestedFeatures`，协商结果保存在 `AcceleratorInterface.TelemetryMarkers()` 上，不是全局变量。

`ACCEL_FEATURE_TELEMETRY_MARKERS` 未启用时（旧库没有这个符号、调用失败或库不认识该位）：`computeUtilizationPercent` 恒为 0、不携带任何信息，Go 侧把每个进程都标成 `ComputeUtilizationUnavailable`，限流控制环改用设备利用率与令牌消耗；库也不会返回任何标记。启用后，读不到的内容以带内标记返回，这些标记只在 `accelerator.go` 中被识别，不会作为数值流出：

- `ProcessInformation.computeUtilizationPercent == -1.0`：该设备这一次采样没有进程级利用率（0.0 表示测得空闲）。Go 侧转成 `ComputeUtilizationUnavailable`；该设备上的 worker 这一拍改用设备利用率，不当作 0，也不改变任何记忆中的状态。
- `processId` 为空的行：该设备的进程列表这一次读不到（不是“没有进程”）。Go 侧转成 `ProcessInformation.ProcessListUnavailable`；该设备上各 worker 的 `pod_memory_used` 保持上一次的值，不写 0，状态变化时记一条日志。
- `DeviceMetrics.utilizationPercent == UINT32_MAX`、`DeviceMetrics.memoryUsedBytes == UINT64_MAX`：对应的单个字段没读到，其余字段有效。Go 侧转成 `GPUUsageMetrics.ComputeUnavailable` / `MemoryUnavailable`，`tf_gpu_usage` 行里省略对应字段；两者同时被标记表示设备打不开，丢弃这次采样。
- 进程列表放不下时库返回 `ACCEL_ERROR_RESOURCE_EXHAUSTED` 且不返回任何行；Go 侧把缓冲区翻倍重试，上限 16384 行，缓冲区在调用之间复用。

`AccelGetProcessInformation` 启用标记后每块有进程的 GPU 约耗时 4 ms，只由 500 ms 同步环每轮调用一次（`WorkerController.processInformation`），其他读者复用这一份采样；10 ms 补充环与心跳环不做任何 provider 调用。

进程级利用率的单次读数噪声很大（驱动把每个短采样整体归给一个进程，占用约两成的进程约有六分之一的读数恰为 0），而且描述的是过去一秒。控制环不直接使用单次读数：每种读数各有一份校准（`utilizationCalibration`），学习“每个令牌每秒对应多少利用率”，worker 的利用率是该比值乘以它实际消耗的令牌；速率直接取 `目标占比 ÷ 比值`。新 worker 先由设备利用率的校准接管（立即精确），进程级读数累计到足够多之后逐步改用进程级校准。

### 5.2 控制台与集群

- `cluster-agent` 是 controller Deployment 里的可选 sidecar，只有设置 `agent.agentId` 时才启用（`charts/tensor-fusion/templates/controller-deployment.yaml`）。`agentId` 形如 `<组织>:<环境名>`，与 `agent.enrollToken` 一起由控制台的接入向导生成。
- agent 的凭据不出现在 Pod spec 里：`agent.enrollToken` 与 `agent.agentId` 由 chart 写入 Secret `<release>-agent`，sidecar 通过 `secretKeyRef` 读取；也可以用 `agent.existingSecret`（`name`、`key`、可选的 `agentIdKey`）指向自己管理的 Secret。chart 管理的 Secret 变化时，Pod 模板上的 `checksum/agent-secret` 注解随之变化，controller Pod 会重启；自管 Secret 轮换后需要手动重启。`hack/test-chart-secrets.sh`（`make test-chart`）用 `helm template` 检查这些渲染结果。
- vector sidecar 的镜像固定为 `timberio/vector:0.56.0-alpine`（controller 与 hypervisor 两处相同，内存上限均为 512Mi）。使用 GreptimeDB Cloud 时，vector 配置（ConfigMap）只引用环境变量 `GREPTIME_PASSWORD`：controller 的 sidecar 由 chart 从 Secret `<release>-greptimedb-secret` 注入；hypervisor 的 sidecar 由 operator 在组装 Pod 时注入（`composeVectorContainer`），Secret 名通过 controller 的环境变量 `TSDB_PASSWORD_SECRET_NAME` 传入。
- `tf_system_metrics` 的 `total_scale_up_cnt` / `total_scale_down_cnt` 是“自 operator 进程启动以来”的累计值，operator 重启或换主后归零，消费方按可重置计数器处理。`tf_*` 表的时间索引列是 `greptime_timestamp`；operator 自己的 SQL 按表探测列名，用户自写的告警规则用 `{{ .TimeColumn }}`。
- agent 主动连到 `agent.cloudEndpoint`（默认 `wss://app.tensor-fusion.ai`）。控制台对集群的所有读写都经这条 WebSocket 代理到集群内 API server，使用 controller 的 ServiceAccount 权限，并且 agent 侧有路径白名单。因此 `charts/tensor-fusion/templates/rbac.yaml` 的改动会改变控制台能做的事。
- 指标不出集群：operator 与 hypervisor 把 `tf_gpu_usage`、`tf_node_metrics`、`tf_pool_metrics`、`tf_worker_resources`、`tf_worker_usage`、`tf_system_metrics` 等表写入集群内 GreptimeDB；控制台画图时通过 agent 在集群内执行查询。表结构变更需要同步控制台的指标查询。
- 控制台直接操作的资源：全部 `tensor-fusion.ai/v1` CRD，以及 Pod、Deployment、Namespace、ConfigMap、Event。
- 控制台的 Helm 安装命令使用 `https://download.tensor-fusion.ai` 作为 chart 仓库，它是 `helm.tensor-fusion.ai` 的镜像。

### 5.3 client 与 worker

- client 通过 `TENSOR_FUSION_OPERATOR_GET_CONNECTION_URL` 向 controller 的 `/api/connection` 查询 worker 地址（`internal/server/`）。
- `TensorFusionConnection` 控制器用 `internal/worker/worker.go` 选择 worker。
- hard 限额通过 worker 容器的环境变量下发，由 `tensor-fusion-core` 的 worker 执行。

## 6. 仓内文档索引

| 文档 | 内容 |
| --- | --- |
| `docs/installation-guide.md` | 安装与 Helm values 说明 |
| `docs/cross-vendor-test-matrix.md` | 多厂商 × 隔离模式的测试矩阵 |
| `docs/dynamic-gpu-isolation-design.md` | 动态隔离模式设计 |
| `docs/soft-isolation-preload.md` | soft limiter 的 preload 机制 |
| `docs/v1-to-v2-upgrade-best-practice.md` | v1 到 v2 升级 |
| `design/gputopo_scheduler_design_cn.md` | GPU 拓扑调度设计 |
| `design/namespaced_gpu_resource_quota.md` | 命名空间配额设计 |
| `e2e/README.md` | AWS EKS 演示与端到端用例清单 |
| `e2e/terraform/README.md` | EKS 测试栈的 Terraform 说明 |

## 7. 已知的不一致

修改相关部分前先确认现状：

1. `charts/tensor-fusion/Chart.yaml` 的 `appVersion` 是手工维护的，落后于最新的 operator tag。
2. chart 默认的 `hypervisorImage` tag 落后于 operator，虽然两者由同一个 workflow 以同一版本发布；`config/samples` 之间也不一致。
3. `tensorfusion/tensor-fusion-hypervisor` 这个镜像名同时被 `vgpu.rs`（1.x，Rust）和本仓库（2.x，Go）发布，入口与环境变量不兼容。
4. chart 固定了 `vgpu-provider-nvidia` 的版本，而 `dockerfile/hypervisor.Dockerfile` 构建时取 `:latest`，hypervisor 镜像内的库与 init 容器拷贝的库可能不是同一版本。
5. `tensor-fusion-node-discovery` 镜像仍被 chart 模板和 samples 引用，但对应代码已删除，没有 workflow 构建它。
6. `provider/limiter.h` 中面向 hypervisor 的 `Limiter*` 函数在 Go 侧没有调用方，`vgpu-provider` 的同名头文件里也没有这些声明。（`provider/accelerator.h` 已与 `vgpu-provider` 同步，见 5.1.2。）
7. `README.md` 仍把 `vgpu.rs` 描述为 hypervisor 的实现。
