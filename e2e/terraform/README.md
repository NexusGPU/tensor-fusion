# TensorFusion e2e 用的一次性 EKS + Karpenter 环境

一条命令拉起、一条命令销毁，带 AWS 侧的超时自毁开关。用途只有一个：给 `e2e/suite` 提供可重复的云上环境。
设计优先级：**① 不留任何计费资源 ② 起停可靠 ③ 每小时成本**。

> 状态（2026-10-12）：在真实账号上完成了 3 轮 `make up` → 测试 → `make down`，逐轮结果见「实测耗时与成本」。
> GPU 配额为 0，所以 **GPU NodePool 只验证到 EC2 返回配额错误为止**（证据见 `../GO-LIVE.md`）。实测数字见本文「实测耗时与成本」。

## 快速开始

```bash
# 依赖：aws cli v2、kubectl、helm 3、jq、bc、curl、unzip。terraform 由 make tools 下载到 .tools/（固定版本，校验 SHA256，不用 brew / sudo）
export AWS_PROFILE=tf-e2e            # IAM Identity Center 的命名 profile；脚本和 Terraform 都只认这个环境变量
cd e2e/terraform

make quotas              # EC2 vCPU 配额与在途的提额申请
make up TTL=3h           # 基础设施 + Karpenter + TensorFusion；自毁开关在 3 小时后触发
export KUBECONFIG=$PWD/.kube/config   # 独立 kubeconfig，不动 ~/.kube/config
make status

make extend TTL=2h       # 把自毁时间改成「现在起 2 小时后」
make cost                # 已发生费用（Cost Explorer）+ 当前在跑资源的实时估算
make down                # 销毁（节点还在时约 12 分钟）+ 残留检查（sweep 的输出就是「没有残留」的证据）
```

`e2e/Makefile` 把这些目标原样转发，在 `e2e/` 目录下执行也可以。

## 架构

```
VPC 10.42.0.0/16（2 个公有子网 / 2 AZ，IGW，无 NAT、无私有子网、无 VPC endpoint）
├─ 自毁开关：EventBridge 规则（每 5 分钟）→ Lambda tf-e2e-deadman → SSM 参数 /tf-e2e/deadman/deadline
└─ EKS 1.36（API 认证模式，Pod Identity，无 OIDC / KMS / 控制面日志）
   ├─ 托管节点组 system   1 × t3.large On-Demand   CoreDNS、Karpenter、TF controller、GreptimeDB、alertmanager
   ├─ Karpenter 1.14.1（featureGates.nodeOverlay=true）
   │   ├─ NodePool tf-cpu   weight 100  通用 CPU 节点（非 GPU 测试 Pod、remote 模式 client、压测端）  limits cpu=4
   │   ├─ NodePool tf-mock  weight 50   演练模式的「假 GPU」节点（CPU 机型 + mock accelerator）        limits cpu=6
   │   ├─ NodePool tf-gpu   weight 10   g4dn.xlarge / g4dn.2xlarge / g5.xlarge / g6.xlarge，spot 优先回退 on-demand，limits nvidia.com/gpu=4
   │   ├─ EC2NodeClass tf-gpu（80Gi gp3 @300MiB/s）/ tf-cpu（30Gi gp3），AMI alias al2023@latest（GPU 机型自动解析到 NVIDIA 版）
   │   └─ NodeOverlay tf-e2e-no-index（weight 100）：把非 TF NodePool 的 tensor-fusion.ai/index* 容量清零，见「踩到的坑」
   ├─ NVIDIA k8s-device-plugin（只跑在真实 GPU 节点上；路径 B 的 NodeClaim 需要 nvidia.com/gpu 才会 Initialized）
   └─ TensorFusion Helm chart（阶段 3，单独安装，镜像 tag 每次显式传入）
       ├─ GPUPool tensor-fusion-shared（AutoSelect，选 e2e.tensor-fusion.ai/shared-pool=true 的节点 = tf-gpu）
       └─ GPUPool tensor-fusion-mock（`make mock-install` 后才有，选 e2e.tensor-fusion.ai/mock-gpu=true 的节点 = tf-mock）
```

三个阶段：

| 阶段 | 内容 | 执行者 |
| --- | --- | --- |
| 1 | 自毁开关、VPC、EKS、system 节点组、IAM、Pod Identity | Terraform 资源 |
| 2 | Karpenter、EC2NodeClass / NodePool、防误扩容的 NodeOverlay、NVIDIA device plugin | `scripts/install.sh`（`terraform_data.install` 在 apply 末尾调用；`make install` 可单独重跑） |
| 3 | GreptimeDB 用的 hostPath PV、TensorFusion chart | `scripts/install-tf.sh`（`make up` 末尾调用；`make tf-install` 可带参数重跑；`SKIP_TF=1 make up` 跳过） |

没有用 helm / kubernetes provider：它们在集群不可达或半删除时会让 `terraform destroy` 直接失败。脚本方式下 Terraform state 里只有 AWS 资源。

## 版本选择与依据

| 组件 | 版本 | 依据 |
| --- | --- | --- |
| EKS | **1.36** | operator 内嵌的是 `k8s.io/kubernetes v1.36.1` 的 kube-scheduler（`go.mod`）。Kubernetes 版本偏差策略不允许 scheduler 比 apiserver 新，所以 apiserver 至少要 1.36。1.36 也是 EKS 当前默认版本，standard support 到 2027-08（`aws eks describe-cluster-versions`，2026-10-11） |
| Karpenter | **1.14.1** | Karpenter 兼容矩阵（`website/content/en/docs/upgrading/compatibility.md`，2026-10-11）：K8s 1.34 需 ≥ 1.6，1.35 需 ≥ 1.9，**1.36 需 ≥ 1.13**，1.37 只有 1.15.x。1.14.1 是 1.14 线的最新补丁（2026-08-21）；1.15.0 是前一天才发的 |
| 与 operator 依赖的兼容性 | `sigs.k8s.io/karpenter v1.8.2` | operator 只用它的 API 类型。对比了 v1.8.2 与 v1.14.1 的 CRD：`karpenter.sh/v1` 的 NodeClaim / NodePool 只有新增（`Gte` / `Lte` 操作符、list-type 标注、放宽的 CEL），没有删改字段；`NodeOverlay` 仍是 `v1alpha1`、仍在 `nodeOverlay` feature gate 之后，schema 与 chart 模板一致。实测：operator 2.17.23 创建的 NodeClaim 被 1.14.1 接受并拉起节点 |
| 早先草稿的 1.35 + 1.9.2 | 弃用 | 矩阵上是合法组合，但 apiserver 1.35 比内嵌 scheduler 旧一个小版本 |
| terraform | 1.16.5（`make tools`） | `.tools/terraform`，下载时校验 SHA256 |
| terraform-aws-modules/eks//modules/karpenter | ~> 21.29 | 只用它的 IAM / Pod Identity / instance profile；21.29.0（2026-10-07）的 controller 策略已含 1.14 需要的 `ec2:DescribePlacementGroups` |
| 镜像 | operator / hypervisor `2.17.23`，`vgpu-provider-nvidia:1.3.9` | Docker Hub 上的最新发布（2026-10-11）。chart 的默认 tag 是空的或落后的，所以 `install-tf.sh` **每次都显式传 tag** |

## 成本安全（硬要求）

1. **标签**：所有资源带 `Project=tf-e2e`（provider `default_tags`、托管节点组的 launch template `tag_specifications`、EC2NodeClass `tags`）。Karpenter 实例另有 `eks:eks-cluster-name`、`karpenter.sh/nodeclaim`。
2. **自毁开关（不依赖任何本地进程）**：
   - EventBridge 规则每 5 分钟调用 Lambda `tf-e2e-deadman`，Lambda 读 SSM 参数里的截止时间。
   - 过期后：所有托管节点组缩到 0 → 终止所有带本栈标签的实例（Karpenter 的、TF 路径 B 直接建的都算）→ 删除游离的 EBS 卷。
   - 过期再超过 `deadman_cluster_grace`（默认 6h）：删除节点组和 EKS 控制面（$0.10/h）。之后剩下的 VPC、IAM、Lambda 不计费。
   - **电平触发**而不是一次性定时：只要截止时间已过，每 5 分钟都会再清一遍，Karpenter 或 ASG 重新拉起的实例活不过下一轮。
   - EKS 集群 `depends_on` 自毁开关：它先于第一个计费资源创建，后于最后一个销毁。
   - `make up TTL=…` 在 apply 之前先重置截止时间；`make extend TTL=…` 随时改；`make deadman` 看状态和最近几次执行。
   - 实测（2026-10-12）：两次，截止时间都设为 2 分钟后，之后本地只做只读轮询。
     - 第一次（只有 system 节点）：过期后 58s 的那次执行把节点组缩到 0；节点组优雅排空超过 5 分钟还没结束，下一次执行直接终止了实例。**过期后 5m58s 全部计算资源消失。**
     - 第二次（system 节点 + 1 台 Karpenter spot 节点）：过期后 3m12s 的执行把节点组缩到 0 并终止了 Karpenter 实例；Karpenter 还活着，立刻又拉起一台；下一次执行（过期后 8m12s）把它也终止了。**过期后 8m12s 全部消失**，这正是做成电平触发的原因。
     - 最坏情况是「过期 + 两个周期」约 10 分钟。
3. **`make cost`**：Cost Explorer 按 `Project` 标签的月初至今 + 全账号按服务拆分（CE 有最长 24h 延迟，标签要先被 AWS 发现才能激活为成本分配标签，脚本会尝试激活）；再加上按在跑资源算的实时估算（spot 用当前 spot 价，on-demand 用 `../pricing/us-west-2.json`，EKS 控制面、公网 IPv4、gp3）。每次 CE API 调用 $0.01。
4. **`make down`**：销毁前记一次成本快照，`terraform destroy`（失败自动重试一次），然后跑 `sweep`。`sweep` 按标签和名称前缀检查 EKS、EC2 实例、launch template、VPC / ENI / LB / SG / NAT、EBS 卷与快照、EIP、IAM、Lambda、SSM、日志组、SQS、EventBridge、Budgets，有残留时退出码为 1。不依赖 Terraform state：`make sweep NAME=tf-e2e REGION=us-west-2`；`make sweep-delete` 是最后手段。
5. NodePool `limits`（`cpu_limit`、`mock_cpu_limit`、`gpu_node_limit`）限制 Karpenter 能起多少；路径 B 的 NodeClaim 不受 NodePool 限制，靠 GPUPool `maxResources` 和自毁开关。

唯一故意保留的东西：`AWSServiceRoleForEC2Spot`（账号级 service-linked role，免费，新账号没有它 spot 会失败；`install.sh` 缺了才建，`make down` 不删）。

## TensorFusion 安装参数（阶段 3）

全部从环境变量读取，镜像 tag 永远显式传给 helm：

| 变量 | 含义 | 默认 |
| --- | --- | --- |
| `TF_CHART` | chart 来源。空 = 本仓库 `charts/tensor-fusion`（被测分支）；`tensor-fusion/tensor-fusion` = 已发布的 chart | 仓库 checkout |
| `TF_CHART_VERSION` | 已发布 chart 的版本 | 最新 |
| `TF_RELEASE_TAG` | operator 与 hypervisor 的版本（同一次发布同一个 tag） | `2.17.23` |
| `TF_OPERATOR_TAG` / `TF_OPERATOR_IMAGE` | 单独覆盖 operator 的 tag / 仓库 | 跟随 `TF_RELEASE_TAG` |
| `TF_HYPERVISOR_TAG` / `TF_HYPERVISOR_IMAGE` | 单独覆盖 hypervisor | 跟随 `TF_RELEASE_TAG` |
| `TF_VGPU_PROVIDER_TAG` | `vgpu-provider-nvidia` 的 tag | `1.3.9` |
| `TF_CLIENT_TAG` / `TF_WORKER_TAG` | remote client / worker | chart 默认 |
| `TF_AGENT_ID` / `TF_ENROLL_TOKEN` / `TF_CLOUD_ENDPOINT` | 控制台 agent。只从环境读，经 0600 的临时 values 文件传给 helm，不进命令行、不回显、不落仓库 | 不启用 |
| `TF_HELM_ARGS` | 追加给 `helm upgrade --install` 的参数 | 空 |

```bash
make tf-install                                   # 仓库 chart + 已发布镜像
TF_RELEASE_TAG=2.17.24 make tf-install            # 修复后的镜像出来以后
TF_CHART=tensor-fusion/tensor-fusion TF_CHART_VERSION=1.8.8 make tf-install
TF_AGENT_ID=… TF_ENROLL_TOKEN=… make tf-install   # 注册到控制台（值来自你的 shell，不要写进文件）
make tf-uninstall                                 # 仓库的 scripts/uninstall.sh
```

## 演练模式：没有 GPU 也把控制链路跑通

`make mock-install` 让 `tf-mock` NodePool 的 CPU 节点（t3.medium，spot 约 $0.014/h）变成带 mock 设备的 TensorFusion GPU 节点：

1. 把仓库里的 example provider 源码（`provider/example`）和 `e2e/mock/{mock_patch.h,build.sh}` 放进 ConfigMap；
2. 在 system 节点上跑一个一次性 Job，用公共的 `gcc:13` 镜像编译，产物以 base64 打到日志；
3. 脚本把它存成 ConfigMap `tf-mock-accelerator`（binaryData）；mock pool 的 hypervisor 多一个 init 容器把它拷到 `/run/tensor-fusion`；
4. 建 `ProviderConfig`（vendor `STUB`）和 GPUPool `mock`（`defaultVendor: STUB`，选 mock 节点）。

**不需要本机 Docker，不需要构建或推送任何测试镜像。** 对 example provider 的改动只有三行（构建时 sed，找不到就失败）：设备 UUID 带上节点短主机名（原版写死 `example-device-0..3`，GPU CR 是集群级对象，两台 mock 节点会互相覆盖）；设备数从 `TF_MOCK_GPU_COUNT` 读（默认 1，对应真实 GPU NodePool 的单卡机型）。每个 mock 设备上报 16 GiB / 312 TFLOPS。

演练能证明什么、不能证明什么，见 `../README.md` 的「执行模式」。

## 实测耗时与成本（us-west-2）

| 轮次 | `make up` | 其中 | `make down` | 销毁后 `sweep` |
| --- | --- | --- | --- | --- |
| 1 | 约 16.5 分钟，分两次 | EKS 控制面 10m28s；system 节点组 2m12s；阶段 2（Karpenter、NodePool）53s；TensorFusion 109s。第一次 apply 在最后的安装钩子处失败：apply 进行中我修改了 `scripts/install.sh`，`filesha256` 前后不一致。重跑 `make up` 53s 收敛 | 6m05s（destroy 332s） | 空 |
| 2 | **13m43s**，一次成功 | 阶段 1 + 2 合计 715s（EKS 9m03s，节点组 1m50s）；TensorFusion 105s（helm 96s） | 4m54s（destroy 263s） | 剩 2 个 Karpenter launch template（不计费）。原因：自毁开关测试时 Karpenter 在创建 LT 与启动实例之间被终止，而销毁钩子按旧标签 `karpenter.k8s.aws/cluster` 找 LT，1.14 用的是 `eks:eks-cluster-name`。已修；修好的钩子对这两个真实残留执行后 `sweep` 为空 |
| 3 | **15m21s**，一次成功 | 阶段 1 + 2 合计 812s（EKS 10m36s，节点组 1m51s）；TensorFusion 105s（helm 98s） | **12m07s**（destroy 695s） | 空 |

`make down` 的 5–6 分钟（第 1、2 轮）是在自毁开关已经把节点清掉之后测的；节点还在时（第 3 轮）要等节点组排空和删除，约 12 分钟。

Karpenter（通用 NodePool，c5.large spot）：Pending → NodeClaim 6s → 节点就绪 39s → Pod Running 39s；清空后 87s 节点消失（`consolidateAfter=30s`）。
mock GPU 节点（t3.medium spot）：Pending → NodeClaim 6s → mock GPU 注册 51s → Pod Running 56s。

成本：三轮合计估算约 **$1.1**（EKS 控制面约 4.2h × $0.10；t3.large 约 3.5h；Karpenter 的 spot 小节点合计约 5.5 节点小时；Cost Explorer API 约 10 次）。Cost Explorer 有延迟，写本文时月初至今显示 $0.04；`Project` 标签要约 24 小时后才能激活为成本分配标签。集群空转实测 **$0.216/h**（含 1 台 mock 节点），无 Karpenter 节点时 $0.19/h。

## 踩到的坑（早先草稿的缺陷）

| # | 缺陷 | 处理 |
| --- | --- | --- |
| 1 | 版本选了 EKS 1.35 + Karpenter 1.9.2，没考虑内嵌 scheduler 是 1.36 | 改为 1.36 + 1.14.1，依据见上 |
| 2 | 默认 `gpu_static_count=1`：用托管节点组起 GPU 节点。TF 的扩容器只克隆「owner 是 Karpenter NodeClaim」的节点，这种节点没法作扩容模板；配额为 0 时还会让 apply 挂 25 分钟 | 默认 0，GPU 全部来自 Karpenter；变量保留给 NEG-10 |
| 3 | 没有任何超时自毁，README 里写的是「建议让 agent 定时 make down」 | 加了 Lambda 自毁开关、`make extend`、`make deadman` |
| 4 | 没有 `make cost` | 加了 |
| 5 | 托管节点组的实例、卷、ENI 不带 `Project` 标签（`default_tags` 不会传到 ASG 拉起的实例） | 节点组改用带 `tag_specifications` 的 launch template |
| 6 | 只有 GPU NodePool：system 节点放不下的普通 Pod 会让 Karpenter 起 GPU 节点 | 加了 `tf-cpu` NodePool，权重高于 GPU |
| 7 | **加了 CPU NodePool 之后暴露出产品问题**：chart 的 NodeOverlay 对所有 NodePool 生效，Pending 的 TF Pod 会让 Karpenter 拉起 CPU 节点，且该节点永远不 Initialized、不会被回收（`../DOCS-GAPS.md` G-01） | 加了更高权重的 `NodeOverlay/tf-e2e-no-index`；`KRP-A15` 专门复现这个问题 |
| 8 | 为 GreptimeDB 的一个 PVC 装了 EBS CSI addon（多一个 addon、一个 IAM 角色、约 1.5 分钟起停时间、以及「卷比集群活得久」这一类残留） | 去掉。PVC 没有 storageClassName，预建一个 hostPath PV 绑定即可 |
| 9 | 新账号没有 `AWSServiceRoleForEC2Spot`，spot 请求会失败 | `install.sh` 缺了才建 |
| 10 | TensorFusion 安装和基础设施绑在同一个脚本里，镜像 tag 靠 chart 默认值（落后 / 为空） | 拆成阶段 3，tag 显式传入 |
| 11 | AZ 只按第一个 GPU 机型选 | 按「提供的 GPU 机型数量」给 AZ 排序后取前两个 |
| 12 | NodePool `consolidateAfter` 设 1m 和 5m 时，TF 扩容器加的节点都出现过在 Pod 落上去之前被 Karpenter 当空节点回收（Pod 要等调度器 5 分钟的兜底重试才会再被调度，见 `../DOCS-GAPS.md` G-16） | 默认 10m；`KRP-A02` 记录「节点就绪 → Pod 绑定」耗时，`KRP-A16` 对它断言（known-fail） |
| 13 | `sweep` 里负载均衡一项在都为空时会误报（字符串是一个空格） | 修了 |
| 15 | 销毁钩子和 `sweep` 按 `karpenter.k8s.aws/cluster` 标签找 Karpenter 的 launch template；Karpenter 1.14 打的是 `eks:eks-cluster-name` / `kubernetes.io/cluster/<name>`，Karpenter 被中途杀掉时留下的 LT 删不掉（第 2 轮销毁后剩 2 个） | 按新旧标签都查，只删名字以 `karpenter.k8s.aws/` 开头的 |
| 14 | `terraform apply` 进行中修改 `scripts/install.sh` 会让 `filesha256` 前后不一致而失败 | 运行期间不要改脚本；失败后重跑 `make up` 即可收敛 |

## 配额

单位 vCPU，`make quotas` 查看：

| 配额 | Code | 本栈需要 |
| --- | --- | --- |
| Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) | `L-1216C47A` | 8 够用：system 2 + tf-cpu ≤ 4（on-demand 时）；spot 另算 |
| All Standard Spot Instance Requests | `L-34B43A08` | 8 够演练：最多 3 台 t3.medium mock 节点 + 1 台 CPU 节点 |
| Running On-Demand G and VT | `L-DB2E81BA` | 真实 GPU 档位需要 ≥ 8（2 台 g4dn.xlarge）；全量建议 16 |
| All G and VT Spot Instance Requests | `L-3819A6DF` | 用 spot 时同上；为 0 时 Karpenter 回退 on-demand |

## 远程 state

默认本地 state（`terraform.tfstate`，已 git-ignore，里面有账号 ID，不要提交）。多人共用时在 `versions.tf` 加 S3 backend（`use_lockfile = true`）。本地 state 丢了用 `make sweep-delete` 清理。
