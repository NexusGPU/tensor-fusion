# GPU 扩容逻辑速查（TensorFusion）

> 目标：把“GPU 扩容”相关的入口、触发条件、资源创建顺序、失败条件统一整理，方便排查 Pending/扩容不生效问题。

## 1. 扩容分两条主链路

TensorFusion 里“扩容”不是单一路径，而是两类：

1. **调度失败触发扩容（Scheduler 侧）**
   - 入口：`cmd/sched/setup.go` 的 `sched.FailureHandler`
   - 处理器：`internal/scheduler/expander/unsched_queue.go` + `internal/scheduler/expander/handler.go`
   - 典型场景：TF Worker Pod 因 GPU 资源不足被拒绝调度后，尝试创建新节点

2. **池容量触发扩容（Controller 侧）**
   - 入口：`internal/controller/gpupool_controller.go`
   - 核心：`internal/controller/gpupool_node_provision.go`
   - 典型场景：`minResources/warmResources` 未满足时，主动补节点

---

## 2. 调度失败触发扩容（NodeExpander）

### 2.1 触发入口

`sched.FailureHandler` 中只要 `status.IsRejected()` 就会进入 unsched handler（`cmd/sched/setup.go`）。

注意：`IsRejected()` 包含三类状态（`vendor/k8s.io/kube-scheduler/framework/interface.go`）：
- `Unschedulable`
- `UnschedulableAndUnresolvable`
- `Pending`

所以 **Pending 也会进入“扩容检查”队列**。

### 2.2 被处理前的过滤条件

`HandleRejectedPod()`（`internal/scheduler/expander/unsched_queue.go`）会先过滤：

- 不是 TF Worker（`!utils.IsTensorFusionWorker(pod)`）=> 不处理
- 指定了固定节点（`nodeName/nodeSelector/nodeAffinity`）=> 不扩容
- `pod.Status.NominatedNodeName != ""`（正在走抢占）=> 不扩容
- 已在 pending map 里 => 去重

之后进入 10s 缓冲队列（`constants.UnschedQueueBufferDuration`）。

### 2.3 NodeExpander 核心步骤

`ProcessExpansion()`（`internal/scheduler/expander/handler.go`）主要流程：

1. **限流**：`inFlightNodes >= 15`（`MaxInFlightNodes`）时跳过本次扩容
2. **去掉 GPU 插件做一次模拟调度**（`simulateSchedulingWithoutGPU`）
   - 用于判断“是否先被 CPU/内存/亲和性等非 GPU 条件拦住”
3. **汇总候选 GPU**
   - 现有可调度节点上的 GPU + in-flight 节点快照 GPU
4. **判断是否真的是 GPU 资源问题**（`checkGPUFitWithInflightNodes`）
   - 含 quota 校验
   - 若已能被现有/in-flight 资源满足，则不再创建新节点
5. **尝试从可扩模板节点克隆扩容**
   - 要求模板节点有已知 owner（`GPUNodeClaim` 或 `Karpenter NodeClaim`）
   - 构造“新节点+新GPU快照”并做 GPU fit 校验
   - 成功后创建扩容资源并记录 `inFlightNodes + preSchedulePods`

### 2.4 调度扩容创建什么资源

由 `createGPUNodeClaim()` 决定，分支如下（`internal/scheduler/expander/handler.go`）：

- 模板节点 owner 是 `GPUNodeClaim`：**克隆一个新的 `GPUNodeClaim`**（`cloneGPUNodeClaim`）
- 模板节点 owner 是 `Karpenter NodeClaim`：
  - 若其 controller parent 是 `GPUNodeClaim`：仍走克隆 `GPUNodeClaim`
  - 若其 parent 是 `NodePool`：**直接创建 Karpenter `NodeClaim`**（`createKarpenterNodeClaimDirect`）

### 2.5 in-flight 是什么

- `inFlightNodes`：已发起扩容、预计将提供 GPU 的“临时节点快照”
- `preSchedulePods`：认为应该能被这些 in-flight 资源满足的 Pod
- 有 10 分钟超时观察；超时仍未调度会打 warning event 并移除预调度状态

---

## 3. 池容量触发扩容（GPUPoolReconciler）

### 3.1 何时触发

`GPUPool` 在 `Provisioned` / `Karpenter` 模式下会执行容量检查（`internal/controller/gpupool_controller.go`）：

- 初始化窗口过后才开始
- 有 pending claim 时会先等前一轮收敛，避免并发扩容抖动

### 3.2 扩容判定规则

`reconcilePoolCapacityWithProvisioner()`（`internal/controller/gpupool_node_provision.go`）逻辑：

1. 把 `PendingGPUNodeClaim` 的资源当作 **assumed** 先加到总量里
2. 先看 `minResources` 缺口（`total` 维度）
3. 若池是 Running，再看 `warmResources` 缺口（`available` 维度）
4. 命中扩容条件后，再检查 `maxResources`
   - 若 `total >= max`（TFLOPS 或 VRAM 任一）则阻止扩容
   - 记录事件 `MaxResourceConstraintReached`

### 3.3 池容量扩容创建什么资源

- 先调用云厂商规划器算“最小成本节点组合”
- 然后并发创建 **`GPUNodeClaim`**（不是直接建 `GPUNode`）
- 新建 claim 会进入 `PendingGPUNodeClaim[pool]`，作为下一轮 assumed 资源

---

## 4. 从 Claim 到可调度 GPU 的资源创建顺序

无论哪条扩容路径，后半段资源生命周期基本一致：

1. `GPUNodeClaim`（或 Karpenter `NodeClaim`）创建
2. 云厂商/Karpenter 创建 K8S `Node`
3. `NodeReconciler` 基于 Node 创建/维护 `GPUNode`（`internal/controller/node_controller.go`）
4. `GPUNodeReconciler` 启动 `node-discovery` Job
5. `node-discovery` 创建/更新 `GPU` CR（`cmd/nodediscovery/main.go`）
6. `GPUNodeReconciler` 再推进 hypervisor，最终把 `GPU/GPUNode` phase 推到 Running

> 重点：`GPU CR` 是 node-discovery 建立与更新的，不是 hypervisor 首次创建。

---

## 5. 哪些情况不会触发扩容 / 会扩容失败

### 5.1 调度失败链路不会触发

- Pod 不是 TF Worker
- Pod 指定了固定节点
- Pod 处于 preemption nominated 阶段
- 非 GPU 问题（模拟去掉 GPU 插件后依然不可调度）
- quota 问题（`checkGPUFitWithInflightNodes` 返回非资源型）

### 5.2 调度扩容失败

- `inFlightNodes` 过多（>=15）直接跳过
- 找不到可克隆模板节点（owner 不合法）
- 模板节点克隆后仍无法通过 GPU fit
- 创建 `GPUNodeClaim/NodeClaim` 失败

### 5.3 池容量扩容失败/不触发

- `ProvisioningToggle=false`
- 不在 Provisioned/Karpenter 模式
- 不满足 `min/warm` 缩容条件
- 命中 `maxResources` 上限
- 规划器无法给出可行节点组合
- 创建 `GPUNodeClaim` 出错

---

## 6. maxResources 约束的边界（非常关键）

- **池容量扩容链路**：有显式 `maxResources` 检查（`gpupool_node_provision.go`）
- **调度失败扩容链路（NodeExpander）**：代码中**没有看到同等显式 maxResources 闸门**

因此，若你要严格保证“任意扩容都不超过 maxResources”，需要额外在 NodeExpander 链路加同类约束。

---

## 7. 快速定位索引（排查时直接看）

- 调度失败扩容入口：`cmd/sched/setup.go`
- rejected pod 入队：`internal/scheduler/expander/unsched_queue.go`
- 扩容主流程：`internal/scheduler/expander/handler.go`
- 池容量扩容：`internal/controller/gpupool_controller.go`
- 容量扩容判定与创建 claim：`internal/controller/gpupool_node_provision.go`
- Node -> GPUNode：`internal/controller/node_controller.go`
- GPUNode -> node-discovery/hypervisor：`internal/controller/gpunode_controller.go`
- GPU CR 创建：`cmd/nodediscovery/main.go`

