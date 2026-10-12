# TensorFusion Operator

GPU 虚拟化与池化的 Kubernetes operator。一个仓库产出两个二进制：controller（CRD 控制器、Pod webhook、内嵌 kube-scheduler）和节点 hypervisor（fractional GPU 的控制面）。

## 先读哪里

- 整个技术栈、相关仓库、跨仓库契约：[`docs/stack-context.md`](docs/stack-context.md)。涉及镜像、limiter、client/worker、控制台的问题，先看这份，再决定是否需要去别的仓库。
- 安装与 Helm values：`docs/installation-guide.md`
- 隔离模式设计：`docs/dynamic-gpu-isolation-design.md`、`docs/soft-isolation-preload.md`
- 调度设计：`design/gputopo_scheduler_design_cn.md`
- EKS 演示与端到端用例：`e2e/README.md`、`e2e/terraform/README.md`

## 代码地图

| 想改什么 | 去哪里 |
| --- | --- |
| CRD 字段 | `api/v1/`，改完跑 `make manifests generate`，chart 里的 CRD 在 `charts/tensor-fusion/crds/` |
| Pod 注入（limiter、client、worker、环境变量、挂载） | `internal/webhook/v1/`、`internal/utils/compose.go` |
| 调度与 GPU 分配 | `internal/scheduler/`、`internal/gpuallocator/` |
| 节点伸缩、Karpenter | `internal/controller/gpupool_node_provision.go`、`gpunodeclaim_controller.go`、`internal/cloudprovider/`、`internal/scheduler/expander/` |
| 算力 / 显存限额的执行 | `pkg/hypervisor/worker/`（共享内存布局在 `state/`，控制环在 `computing/`） |
| 设备发现、厂商适配 | `pkg/hypervisor/device/`、`provider/`、`pkg/constants/vendors.go` |
| 注解名、环境变量名 | `pkg/constants/` |
| Helm chart | `charts/tensor-fusion/` |

## 常用命令

```bash
make build              # controller
make build-hypervisor   # hypervisor（CGO 关闭，purego）
make build-provider     # example accelerator 库
make test               # envtest 全量
make ut F=<文件>        # 单个测试文件
make ut-sched           # 调度器测试
make lint
make vendor             # 重新 vendor 并应用 patches/ 下的调度器补丁
```

## 需要知道的约束

- `vendor/` 里的 kube-scheduler 被 `patches/` 修改过，不要直接手改 vendor，改补丁后跑 `make vendor`。
- soft 隔离的共享内存布局同时定义在本仓库、`vgpu-provider`、`vgpu.rs` 三处，没有版本字段，改动必须三方同步。
- `pkg/hypervisor` 和 `api/v1` 被 `gpu-go` 作为库引用，导出接口的变更是破坏性变更。
- `provider/*.h` 是 `vgpu-provider` 头文件的手工拷贝。
- CRD、Helm values、指标表、注解发生变更时，`tensor-fusion-cloud`（控制台）和 `tensor-fusion-site`（文档）需要跟着改。

## 敏感信息

这是公开仓库。密钥、token、kubeconfig、连接串、内网地址、云账号 ID、未修复安全问题的利用细节一律不入库。本地笔记写在 `tmp.*` 文件里，`.gitignore` 已忽略。
