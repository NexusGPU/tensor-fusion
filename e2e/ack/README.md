# TensorFusion e2e 用的一次性 ACK + Karpenter 环境

阿里云 ACK 托管集群 + Karpenter 的阿里云 provider，GPU 节点是真实的 T4。用途与 `../terraform`（EKS）相同：给 `../suite` 提供可重复的云上环境。这是第一套让 GPU 档位在真实 GPU 上执行的环境。

- 测试结论、耗时与成本：[`TEST-SUMMARY.md`](TEST-SUMMARY.md)
- 手工测试步骤：[`MANUAL-TEST-CASES.md`](MANUAL-TEST-CASES.md)

## 快速开始

```bash
# 依赖：aliyun CLI（已 aliyun configure）、kubectl、helm 3、jq、curl；构建补丁镜像时另需 docker、git、go
cd e2e
make ack-up                              # VPC + 集群 + Karpenter + TensorFusion + 成本守卫，约 10 分钟
export KUBECONFIG=$PWD/ack/.work/kubeconfig
make ack-test TIER=smoke                 # 任意档位：smoke | core | isolation | karpenter-finops | perf-quick
make ack-test TIER=core FOCUS='ISO-01'   # 单条用例
make ack-sweep                           # 列出这套环境现存的云资源
make ack-down                            # 销毁并清点残留
```

参数全部在 `lib.sh`，用环境变量覆盖（地域、机型、NodePool 上限、版本）。凭据只来自 aliyun CLI 的 profile，不写进仓库；集群内的 Karpenter 通过 RRSA 扮演 RAM 角色，集群里没有 AccessKey。

## 组成

```
VPC 10.43.0.0/16（2 个 vSwitch / 2 个可用区，NAT 网关 + EIP 出网）
└─ ACK 托管版 Basic，Kubernetes 1.36（Flannel，RRSA，公网 API 端点）
   ├─ 节点池 system   1 × ecs.g7.xlarge，Alibaba Cloud Linux 3 容器优化版（cgroup v2）
   ├─ Karpenter：AliyunContainerService/karpenter-provider-alibabacloud 0.2.5（core 1.8，NodeOverlay 特性门打开）
   │   ├─ NodePool tf-cpu   weight 100  ecs.c7.large / ecs.g7.large，limits cpu=8，30 秒回收
   │   ├─ NodePool tf-gpu   weight 10   ecs.gn6i-c4g1.xlarge / c8g1.2xlarge（各 1 张 T4），按量，limits nvidia.com/gpu=3，空闲 2 分钟回收
   │   └─ ECSNodeClass tf-gpu / tf-cpu：集群自己的节点镜像、安全组、worker RAM 角色
   ├─ ACK 自带的 nvidia device plugin（GPU 节点上报 nvidia.com/gpu，路径 B 的 NodeClaim 靠它 Initialized）
   └─ TensorFusion chart（本仓库 checkout），默认 pool 选 e2e.tensor-fusion.ai/shared-pool=true 的节点
```

| 脚本 | 做什么 |
|---|---|
| `run.sh up` / `down` | 整套起停；`make ack-up` / `ack-down` 调的就是它 |
| `up.sh` | VPC、vSwitch、集群、system 节点池、kubeconfig |
| `install.sh` | Karpenter 的 RAM 角色、chart、ECSNodeClass、NodePool |
| `install-tf.sh` | TensorFusion chart，镜像 tag 显式传入；控制台 agent 的凭据从环境变量进 Secret |
| `build-karpenter.sh` | 用 `patches/` 构建 provider 并导入 system 节点 |
| `load-image.sh` | 把本机镜像（或 `docker save` 的归档）导入 system 节点的 containerd |
| `guard.sh` | 成本守卫 |
| `down.sh` | 按名字和标签删除全部资源，最后清点残留 |

`run.sh up` 会读取 `ack/.work/` 下的三样本地输入（都不入库）：`images/*.tar`（没有镜像仓库承载的开发构建）、`stack.env`（如 `KARPENTER_ALIBABA_IMAGE=`、`TF_OPERATOR_TAG=`）、`agent.env`（控制台环境的 `TF_AGENT_ID` / `TF_ENROLL_TOKEN`，权限 600）。

## 成本安全

- 所有资源带标签 `Project=tf-e2e-ack`。
- NodePool `limits`：GPU 最多 3 张，CPU 池最多 8 核。
- `guard.sh`（`run.sh up` 会在后台启动，日志 `ack/.work/guard.log`）每 2 分钟：终止没有 NodeClaim 也不属于节点池的 ECS 实例（存在超过 5 分钟才动手）；GPU 实例超过 4 台时把所有 NodePool 的上限置 0 并删除路径 B 的 GPUNodeClaim。解除冻结需要人来决定（重跑 `install.sh`）。
- 守卫是本机进程，机器休眠它就不工作。**这套环境没有 EKS 栈那种云侧的超时自毁**，用完执行 `make ack-down`。
- 空转成本：system 节点 + NAT 网关 + SLB + EIP，约 $0.25/h；每台 T4 节点约 $1.09/h（us-east-1 按量目录价，见 `../pricing/alibaba-us-east-1.json`）。

## 踩到的坑

| # | 现象 | 处理 |
|---|---|---|
| 1 | `CreateCluster` 报 `ErrKubernetesAuditRoleNotAttach` / `AddonRoleNotAuthorized`：账号没有授权 ACK 的服务角色 | 一次性的账号级授权（RAM 控制台，或按报错里的链接）。不属于本栈，`down.sh` 不删 |
| 2 | 节点池创建失败 `CgroupV1NotSupport`：Kubernetes ≥ 1.35 不接受 cgroup v1 节点，而 `AliyunLinux3` 镜像是 cgroup v1 | 节点镜像用 `AliyunLinux3ContainerOptimized`；节点池单独创建，失败时可以只换节点池 |
| 3 | Karpenter 启动即 panic：`unknown feature gate "NodeOverlay"` | chart 的 `settings.featureGates` 同时变成 `--feature-gates` 参数，provider 用自己的清单解析它。只通过环境变量 `FEATURE_GATES` 设置 |
| 4 | `RunInstances` 报 `InvalidUser.PassRoleForbidden` | `ram:PassRole` 按角色 ARN 限定不被接受；用 `Resource: *` 加条件 `acs:Service=ecs.aliyuncs.com` |
| 5 | 一个 NodeClaim 对应两台 ECS | provider 在 `RunInstances` 成功、随后的 `DescribeInstances` 失败时直接重试，不回收第一台。触发条件是运行中替换 RAM 策略；脚本改为原地更新策略版本，守卫兜底 |
| 6 | **一个 Pending 的 TensorFusion Pod 拉起 4 台 GPU 节点** | provider 返回的 NodeClaim 容量不含 NodeOverlay 声明的资源，Karpenter 每轮都认为在途节点装不下这个 Pod。`patches/` 里的补丁把 overlay 容量补进去；`build-karpenter.sh` 构建并导入 |
| 7 | 控制台提示 Karpenter 的节点是「游离节点」 | 预期行为：provider 用集群接入脚本注册节点，不经过 ACK 节点池。节点池的托管升级、自愈不作用于这些节点 |
| 8 | `KRP-C01` 等不到 CPU NodePool 为空 | ACK 的 CoreDNS 是 2 副本反亲和，第二个副本占着 Karpenter 的 CPU 节点。`install.sh` 把它收成 1 副本 |
| 9 | 开发镜像推不到任何镜像仓库 | controller 和 Karpenter 都固定在 system 节点，`load-image.sh` 直接导入该节点的 containerd。hypervisor 跑在 Karpenter 拉起的节点上，这个办法不适用，它需要镜像仓库 |

## 与 EKS 栈的差异

| | EKS（`../terraform`） | ACK（本目录） |
|---|---|---|
| 起停 | Terraform | shell + aliyun CLI |
| 超时自毁 | Lambda，云侧 | 没有；本机守卫只管实例泄漏和上限 |
| 容量类型 | spot 优先，回退按量 | 仅按量 |
| GPU 驱动 | NVIDIA AMI | ACK 接入脚本在首次启动时安装，节点就绪约 4–5 分钟 |
| 存储 | 无 CSI，hostPath PV | 相同（ACK 默认装了 CSI，但没有默认 StorageClass） |
| NodeOverlay 范围 | `tf-e2e-no-index` 把非 TF NodePool 清零 | chart 的 `karpenterNodeOverlay.requirements` 直接限定到 `karpenter.alibabacloud.com/instance-gpu-manufacturer=nvidia` |
