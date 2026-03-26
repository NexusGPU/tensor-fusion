# GPU Topology Aware Scheduling 设计方案

## 状态

草案

## 1. 背景

TensorFusion 当前已经具备较完整的 GPU 资源调度能力：

- `GPUResourcesFit` 负责 GPU 资源过滤、打分、预占、保留和最终绑定
- `GPUNetworkTopologyAware` 已作为独立 scheduler plugin 注册，但当前仍是空实现

现有代码边界如下：

- `GPUResourcesFit` 已承担完整资源分配生命周期，见 `internal/scheduler/gpuresources/gpuresources.go`
- `GPUNetworkTopologyAware` 已独立注册到调度器，见 `internal/scheduler/gputopo/gpu_network_topo.go`
- 调度配置中也已单独启用 `GPUNetworkTopologyAware`，见 `config/samples/scheduler-config.yaml`

当前需要补齐的是一个可以落地的 `gputopo` 设计，使其在不破坏现有 `gpufit` 逻辑的前提下，为多 GPU 请求提供节点内 GPU 拓扑感知能力。

## 2. 设计目标

### 2.1 目标

1. `gputopo` 作为独立 scheduler plugin 落地，不并入 `gpufit`
2. 复用 `GPUResourcesFit` 已产出的候选 GPU 集合，不重复实现资源分配逻辑
3. 在单节点内优先选择拓扑更优的 GPU 组合
4. MVP 先支持 NUMA-first 策略，后续扩展到 NVLink/HCCS/其他厂商互联
5. 保持实现边界清晰，方便后续维护和演进

### 2.2 非目标

1. MVP 不实现跨节点 GPU 网络拓扑联合求解
2. MVP 不直接依赖 Koordinator 的 `ClusterNetworkTopology` CRD
3. MVP 不在 scheduler 中解析厂商私有原始拓扑格式
4. MVP 不引入复杂全图优化器或全局最优搜索

## 3. 设计原则

### 3.1 职责分离

- `GPUResourcesFit` 负责回答“这个 Pod 在哪些 node 上有可分配 GPU”
- `GPUNetworkTopologyAware` 负责回答“这些可分配 GPU 中哪一组拓扑更优”

### 3.2 复用已有 CycleState

`gputopo` 直接消费 `GPUResourcesFit` 已写入的调度状态，不重新计算候选 GPU。

### 3.3 统一拓扑抽象

插件不直接理解厂商私有拓扑名词，而是只消费统一的 tier/domain 抽象。

### 3.4 数据面与控制面分离

- hypervisor/provider 负责发现 GPU NUMA 与 interconnect 拓扑
- scheduler plugin 只消费归一化后的 topology metadata
- scheduler 不直接调用运行时硬件接口，不解析厂商原始拓扑格式

### 3.5 渐进演进

MVP 先落 NUMA，后续通过 evaluator/provider 扩展到厂商互联拓扑。

### 3.6 可维护性与低耦合优先

从架构设计角度，本方案必须满足以下约束：

1. 保持高内聚、低耦合
2. 保证 topology 能力以独立插件方式演进，不侵入已有资源调度主流程
3. 新增厂商拓扑支持时，应优先扩展数据模型和 evaluator，而不是修改调度主链路
4. 插件关闭时，绝不能影响现有调度流程、性能路径和行为结果

这意味着：

- `gputopo` 只能通过明确的 plugin 边界和 `CycleState` 与 `GPUResourcesFit` 协作
- 不允许把 topology 逻辑散落到 `allocator`、gang manager、controller 主流程中
- 不允许让未启用插件的场景承担额外行为变化或隐式语义变化

可以将这条约束概括为：

- 插件启用时增强能力
- 插件关闭时零侵入、零副作用、零行为漂移

## 4. 参考设计

本方案参考了 HAMI 与 Volcano 的已有实践，但不直接照搬其实现。

### 4.1 HAMI 的可借鉴点

HAMI 的设备调度策略值得借鉴的点：

1. 将“资源是否可分配”和“按拓扑选哪组设备”分离
2. 先为候选节点挑出最佳设备组合，再基于该结果打分
3. 通过统一的拓扑评分矩阵做组合选择，而不是把拓扑逻辑塞进资源过滤器

本方案借鉴这些思想，但保持 TensorFusion 现有 `GPUResourcesFit` 生命周期不变。

### 4.2 Volcano 的可借鉴点

Volcano 的 network topology 调度值得借鉴的点：

1. 将 topology 状态建模为独立调度状态，而不是塞进某个资源插件
2. 使用 tier/hypernode 的统一层级抽象表达拓扑优先级
3. 在 controller 侧先做 topology 配置归一化，再由 scheduler 消费

本方案借鉴 Volcano 的独立状态和 tier 抽象，但本期只关注单节点内 GPU topology。

## 5. 为什么不直接复用 Koordinator 的 NetworkTopology 能力

Koordinator 现有 `NetworkTopologyAware` 主要解决的是：

- 跨节点网络拓扑
- GangGroup / PodGroup 场景
- block / spine / datacenter 等层级域调度

而 `gputopo` 当前要解决的是：

- 单节点内 GPU 之间的拓扑关系
- NUMA / NVLink / HCCS / PCIe Switch 等本机互联关系
- 单 Pod 的 GPU 组合选择问题

因此两者在数据来源、作用范围和调度粒度上都不同。

结论：

- 不直接复用 Koordinator 的现成业务逻辑
- 可以借鉴其 topology tree / tier 的建模思想

## 6. 总体架构

调度职责划分如下：

1. `GPUResourcesFit.PreFilter`
   生成候选节点及其可分配 GPU 集合，并写入 `CycleStateGPUSchedulingResult`
2. `GPUNetworkTopologyAware`
   读取候选 GPU，计算每个节点的最佳拓扑组合
3. `GPUNetworkTopologyAware.Filter`
   在 `hard` 模式下拒绝不满足拓扑要求的节点
4. `GPUNetworkTopologyAware.Score`
   在 `soft` 模式下对拓扑更优的节点给更高分
5. `GPUResourcesFit.Reserve/PostBind`
   优先消费 `gputopo` 产出的最佳 GPU 组合

关键点：

- `gputopo` 只做组合选择，不做资源实际分配
- 最终分配、缓存更新、注解 patch 仍由 `GPUResourcesFit` 负责
- `gputopo` 是否启用由 scheduler plugin 配置显式决定，未启用时不改变原有任何调度链路

## 7. Topology 数据面

当前 TensorFusion 已具备 GPU topology 与 NUMA 的基础数据采集能力：

1. hypervisor/provider 已提供 `GetAllDevicesTopology` 接口，可返回设备间拓扑矩阵
2. provider 的 topology 结构中已包含每张 GPU 的 `numaNode` 和 peer topology 信息
3. TensorFusion 的 `GPU` 对象状态中已存在 `NUMANode` 字段

这意味着：

- `gputopo` 的数据来源不是未来规划项，而是现有能力接入问题
- MVP 可直接基于已有 NUMA 数据落地
- 后续 vendor topology 模式也具备清晰的数据来源

### 7.1 现有能力

当前已有接口与字段：

- `provider/accelerator.h`
  - `GetAllDevicesTopology(ExtendedDeviceTopology* topology)`
  - `DeviceTopologyInfo.numaNode`
  - `DeviceTopologyInfo.peers`
- `api/v1/gpu_types.go`
  - `GPUStatus.NUMANode`

### 7.2 责任边界

数据面职责如下：

- hypervisor/provider
  - 采集原始硬件 topology 与 NUMA 信息
  - 将厂商原始信息转换为统一的调度视图
- controller/provider sync 链路
  - 将可调度所需的 topology metadata 持久化到稳定对象
- scheduler plugin
  - 读取归一化结果进行 Filter/Score

### 7.3 scheduler 消费原则

`gputopo` 不应：

1. 直接调用 hypervisor runtime 接口
2. 在调度时临时采集硬件拓扑
3. 解析 NVLink/HCCS/厂商私有原始输出

`gputopo` 应：

1. 只依赖稳定对象上的归一化 metadata
2. 优先使用已同步到 `GPU` 对象的数据
3. 在 metadata 不完整时按 `Unknown` 或 NUMA fallback 处理

### 7.4 推荐归一化模型

建议后续在数据面统一为如下抽象：

```go
type DeviceTopology struct {
    NUMANode *int32        `json:"numaNode,omitempty"`
    Links    []DeviceLink  `json:"links,omitempty"`
}

type DeviceLink struct {
    PeerGPU   string `json:"peerGPU"`
    LinkType  string `json:"linkType,omitempty"`
    Tier      int32  `json:"tier"`
    Bandwidth int64  `json:"bandwidth,omitempty"`
}
```

字段含义：

- `NUMANode`
  供 MVP 的 NUMA-first 策略直接使用
- `Links`
  供后续 NVLink/HCCS/PCIe Switch 等细粒度 topology 评估使用
- `Tier`
  scheduler 统一消费的可比较拓扑等级

### 7.5 持久化建议

推荐优先将 topology metadata 挂在 `GPU` 对象上，而不是只保存在 provider 进程内缓存。

原因：

1. scheduler 读取路径稳定
2. controller 与 scheduler 状态一致
3. 便于 debug、观测和回放问题
4. 后续扩展 vendor topology 时不需要让 scheduler 直接依赖 provider 进程

### 7.5.1 建议的 API 落点

建议将统一 topology metadata 放在 `GPU CR status` 中，而不是 annotation。

推荐字段：

```go
type GPUStatus struct {
    ...
    NUMANode *int32             `json:"numaNode,omitempty"`
    Topology *GPUTopologyStatus `json:"topology,omitempty"`
}

type GPUTopologyStatus struct {
    Peers []GPUPeerLinkStatus `json:"peers,omitempty"`
}

type GPUPeerLinkStatus struct {
    PeerGPUUUID string `json:"peerGPUUUID"`
    Tier        int32  `json:"tier"`
    LinkType    string `json:"linkType,omitempty"`
    Bandwidth   int64  `json:"bandwidth,omitempty"`
}
```

设计说明：

1. `NUMANode` 继续保留，兼容现有逻辑与 MVP
2. `Topology` 作为新增结构化字段，承载多厂商统一拓扑信息
3. `Peers` 描述当前 GPU 到其他 GPU 的关系
4. `PeerGPUUUID` 使用稳定 UUID，而不是 index，避免设备顺序变化导致引用不稳定

### 7.5.2 为什么放在 status

放在 `status` 的原因：

1. 属于运行时发现的观察数据，而不是用户声明的期望
2. 与 `Capacity`、`Available`、`NUMANode` 同属设备实际状态
3. 便于由 controller/provider 周期性刷新
4. scheduler 与调试工具读取路径一致

不建议放在 annotation，原因：

1. annotation 缺乏结构约束
2. 演进成本高
3. 对 peer graph 这类结构化数据可读性和可维护性都较差

### 7.5.3 向后兼容策略

API 演进建议：

1. 保持 `NUMANode` 不变
2. 新增 `Topology` 字段
3. MVP 中 `Topology` 可为空，调度逻辑退化为 NUMA-only
4. provider 补齐 `Topology` 后，`AutoEvaluator` 自动切换到统一 peer-tier 模型

这样可以保证：

- 旧数据不需要一次性迁移
- 调度器可以渐进支持新拓扑模型
- 不影响当前仅依赖 `NUMANode` 的场景

### 7.6 与 v1 历史 `nvlink` 字段的关系

历史版本中曾存在过以 `nvlink` 为中心表达 GPU 互联能力的设计思路。这个方向说明：

- 早期版本已经意识到 GPU interconnect 对调度结果有价值
- scheduler 需要从 GPU 元数据中读取本机互联关系

但新版本不建议直接恢复为单一 `nvlink` 字段，原因如下：

1. `nvlink` 是 NVIDIA 特有语义，不适合作为多厂商统一抽象
2. 后续需要兼容 Ascend HCCS、AMD XGMI、MetaX 等其他厂商互联
3. 单一字段只能表达“是否有 NVLink”，无法表达 peer 之间的层级与强弱关系

因此，建议将历史 `nvlink` 能力演进为统一 topology 模型：

- 保留 `NUMANode` 作为通用 fallback
- 将历史 `nvlink` 语义吸收到 `links[]` 或 `peer-tier` 模型中
- 让 scheduler 依赖统一的 `tier` / `peer` / `linkType`，而不是依赖某个厂商字段名

演进关系如下：

- v1 历史思路：`nvlink` 字段描述 NVIDIA 互联
- 当前 MVP：优先使用 `NUMANode`
- 后续统一模型：`NUMANode + Links + Tier`

这种演进方式既保留了 v1 的经验，也避免把数据面设计锁死在单一厂商上。

## 8. 调度流程

### 7.1 PreFilter

`GPUResourcesFit.PreFilter` 保持现有逻辑不变，输出：

- `CycleStateAllocateRequest`
- `CycleStateGPUSchedulingResult`

其中 `CycleStateGPUSchedulingResult.NodeGPUs[nodeName]` 表示该节点上的候选 GPU 集合。

### 7.2 gputopo 评估

`GPUNetworkTopologyAware` 基于每个节点的候选 GPU 集合，生成该节点的最佳拓扑 plan。

步骤如下：

1. 读取 Pod 的 GPU 请求数量和分配模式
2. 读取某个节点上的候选 GPU 列表
3. 枚举满足请求数量的 GPU 组合
4. 计算每个组合的 topology tier 和组合分
5. 选出当前节点的最佳 plan
6. 将结果写入新的 CycleState

### 7.3 Filter

- `hard` 模式：
  只有当该节点存在满足拓扑要求的 plan 时才返回 `Success`
- `soft` 模式：
  节点放行，但是否更优由 `Score` 决定

### 7.4 Score

基于节点最佳 plan 的 topology score 返回标准化分值。

### 7.5 Reserve

`GPUResourcesFit` 在最终选卡时优先消费 `gputopo` 给出的 `BestGPUIds`，避免资源插件和拓扑插件得出不同结果。

## 9. CycleState 设计

建议新增一个独立的 topology state：

```go
const CycleStateGPUTopologyResult = "gpuTopologyResult"

type NodeTopologyPlan struct {
    NodeName        string
    CandidateGPUIds []string
    BestGPUIds      []string
    Tier            int
    Score           int64
    ModeSatisfied   bool
    Reason          string
}

type GPUTopologyStateData struct {
    Plans map[string]*NodeTopologyPlan
}
```

字段说明：

- `CandidateGPUIds`: 来自 `GPUResourcesFit` 的候选 GPU
- `BestGPUIds`: `gputopo` 为该节点挑选的最佳 GPU 组合
- `Tier`: 归一化后的拓扑层级，越小表示越近
- `Score`: 组合分数，用于 `Score` 插件阶段
- `ModeSatisfied`: 是否满足 `hard` 模式要求
- `Reason`: 调试和事件输出使用

## 10. 统一拓扑抽象

为兼容大部分厂商，建议将拓扑抽象从“NUMA-only”升级为“peer graph + tier”模型。

建议统一使用如下抽象：

```go
type GPUTopology struct {
    GPUUUID  string        `json:"gpuUUID"`
    NUMANode *int32        `json:"numaNode,omitempty"`
    Peers    []GPUPeerLink `json:"peers,omitempty"`
}

type GPUPeerLink struct {
    PeerGPUUUID string `json:"peerGPUUUID"`
    Tier        int32  `json:"tier"`
    LinkType    string `json:"linkType,omitempty"`
    Bandwidth   int64  `json:"bandwidth,omitempty"`
}
```

其中：

- `Tier` 是 scheduler 的核心输入
- `LinkType` 只用于可观测性和调试，不作为主调度语义
- `NUMANode` 是最小保真度 fallback
- `Peers` 用于表达多卡组合之间的真实拓扑关系

建议为调度层定义统一 tier：

```go
type GPUAffinityTier int

const (
    TierSameInterconnect GPUAffinityTier = 0
    TierSameNUMA         GPUAffinityTier = 1
    TierCrossNUMA        GPUAffinityTier = 2
    TierUnknown          GPUAffinityTier = 3
)
```

MVP 实际只落地：

- `TierSameNUMA`
- `TierCrossNUMA`
- `TierUnknown`

后续可扩展：

- 同 NVLink 域
- 同 HCCS 域
- 同 PCIe Switch
- 跨 socket

当 provider 已上报更细粒度的 topology tier 时，`auto` 模式可采用如下优先级：

1. 优先使用 vendor interconnect tier
2. 若缺失则退化到 NUMA
3. 若 NUMA 也缺失，则视为 `TierUnknown`

## 11. 配置模型

当前 `GPUNetworkTopologyAwareConfig` 只有带宽字段，无法真正驱动调度行为。建议调整为：

```go
type GPUNetworkTopologyAwareConfig struct {
    Mode                  string `json:"mode"`
    TopologySource        string `json:"topologySource"`
    MaxAllowedTier        int    `json:"maxAllowedTier"`
    UnknownTopologyPolicy string `json:"unknownTopologyPolicy"`
    ScoreWeight           int64  `json:"scoreWeight"`
    PreferLeastDamage     bool   `json:"preferLeastDamage"`
}
```

建议默认值：

- `mode: soft`
- `topologySource: auto`
- `maxAllowedTier: 1`
- `unknownTopologyPolicy: treat-as-worst`
- `scoreWeight: 3`
- `preferLeastDamage: true`

字段说明：

- `mode`
  - `hard`: 不满足要求直接过滤
  - `soft`: 放行并降分
- `topologySource`
  - `auto`: 自动选择可用数据源
  - `numa`: 强制按 NUMA
  - `vendor`: 强制按厂商拓扑
- `maxAllowedTier`
  允许的最大拓扑层级，MVP 可映射为 NUMA 约束
- `unknownTopologyPolicy`
  - `treat-as-worst`: 当作最差 tier
  - `reject`: 直接视为不满足
- `scoreWeight`
  topology score 权重
- `preferLeastDamage`
  单卡分配时尽量保留更好的 GPU 团簇给后续大任务

说明：

- `vendor` 并非远期概念，而是依赖 provider 已上报的归一化 topology metadata
- MVP 默认仍建议走 `auto`，确保在 topology metadata 不完整时平滑退化到 NUMA

### 11.1 与旧配置兼容

当前已有字段：

```go
type GPUNetworkTopologyAwareConfig struct {
    TotalIntranetBandWidthGBps int64 `json:"totalIntranetBandWidthGBps"`
}
```

该字段更偏集群内总带宽保护，不适合作为 GPU interconnect topology 核心配置。

建议：

1. 新配置字段作为主逻辑
2. `TotalIntranetBandWidthGBps` 保留兼容一个版本
3. 在实现中标记为 deprecated
4. 后续统一迁移到新的 topology policy 配置

## 12. Pod 级覆盖

建议支持 Pod annotation 覆盖默认配置：

- `tensor-fusion.ai/gpu-topology-mode`
- `tensor-fusion.ai/gpu-topology-max-tier`
- `tensor-fusion.ai/gpu-topology-source`

优先级：

1. Pod annotation
2. scheduler plugin config
3. 默认值

## 13. Topology Evaluator 设计

建议将拓扑评估逻辑抽成 evaluator：

```go
type Evaluator interface {
    Name() string
    Evaluate(node string, gpus []*tfv1.GPU, req *tfv1.AllocRequest) (*NodeTopologyPlan, error)
}
```

初期实现：

- `NUMAFallbackEvaluator`

后续扩展：

- `GenericTopologyEvaluator`

收益：

- 新增厂商互联时不污染调度主逻辑
- NUMA-only 与统一 topology graph 可独立演进
- `auto` 模式可在 evaluator 层统一选择数据源

### 13.1 AutoEvaluator 建议

建议提供一个 `AutoEvaluator`，策略如下：

1. 若 GPU topology metadata 中存在有效 peer-tier graph，则优先使用 `GenericTopologyEvaluator`
2. 若不存在有效 link tier，但 `NUMANode` 存在，则使用 NUMA evaluator
3. 若两者都不可用，则返回 unknown topology 结果

这样可以保证：

- 当前已存在的 NUMA 数据马上可用
- 后续 provider 补齐不同厂商 topology 后无需重构 scheduler 主流程

## 14. MVP 规则

### 13.1 多卡请求

对请求 `N > 1` 的 Pod：

1. 优先选择同 NUMA 的 `N` 张 GPU
2. 若无同 NUMA 组合：
   - `hard`: 过滤节点
   - `soft`: 允许，但降低分数

### 13.2 单卡请求

对请求 `N = 1` 的 Pod：

1. 不追求“当前最佳单卡”
2. 优先选择对高带宽团簇破坏最小的卡
3. 目标是保留更好的多卡组合给后续请求

### 13.3 NUMA Unknown

NUMA 信息未知时：

- 默认视为 `TierUnknown`
- 行为由 `unknownTopologyPolicy` 决定

## 15. 组合评分

MVP 采用简单且可落地的评分方式。

### 14.1 多卡

对每个候选组合：

1. 先计算其 tier
2. 再计算组合内所有 GPU pair 的亲和得分总和
3. 若分数相同，再比较分配后的碎片度

排序优先级：

1. 更小的 tier
2. 更高的 topology score
3. 更低的碎片度

### 14.2 单卡

单卡请求优先选择“least damage”的 GPU：

- 与其他 GPU 的总体亲和度越低，越优先被单卡任务占用
- 把高互联团簇保留给多卡任务

进一步地，单卡选择应尽量消耗“碎片资源”，而不是“核心资源”。

建议将单卡请求的判定顺序明确为：

1. 优先选择不属于高质量 interconnect 团簇的 GPU
2. 优先选择所在 NUMA 域中已经难以组成多卡组合的“孤卡”
3. 若必须从存在互联关系的域中选择，优先拿走后不会破坏完整多卡组合的 GPU
4. 最后才选择完整高质量 topology 团簇中的成员

也就是说：

- 单卡尽量找非 GPU 互联资源
- 若存在 NUMA 域中只剩单个 GPU 的情况，优先消耗这类孤卡
- 尽量避免提前打散同 NUMA 或同 interconnect domain 的完整组合

可实现为一个单卡 `least-damage` 评分：

1. 是否属于高速互联团簇
2. 所在 NUMA / domain 的剩余卡数
3. 取走后该域还能否组成 2/4/8 卡组合
4. 当前 GPU 与其他 GPU 的平均拓扑亲和度

可操作上的直观优先级为：

- 孤卡
- 非互联卡
- 弱互联卡
- 强互联完整团簇成员

## 16. 组合搜索与复杂度控制

`gputopo` 的核心开销来自“在候选 GPU 集合中搜索最佳组合”。如果不加约束，后续实现容易在多卡节点上出现组合爆炸。

因此，设计中必须明确复杂度控制策略。

### 16.1 设计目标

1. 保证调度路径复杂度可控
2. 在常见 GPU 数量规模下获得稳定结果
3. 优先保证正确性和可解释性，而不是追求理论全局最优

### 16.2 基本约束

MVP 只在单节点内做组合搜索，输入规模通常满足：

- 单节点 GPU 数量一般为 `8/16`
- Pod 单次请求 GPU 数量通常为 `1/2/4/8`

在这一范围内，有限组合枚举是可接受的，但仍需要裁剪。

### 16.3 搜索策略

建议采用“先裁剪、后枚举、最后排序”的三段式策略。

#### 第一步：候选裁剪

对 `GPUResourcesFit` 输出的 `NodeGPUs[nodeName]`，先做轻量裁剪：

1. 去除不满足基础资源条件的 GPU
2. 对单卡请求，仅保留 allocator 初步排序后的前 `K` 个 GPU
3. 对多卡请求，优先保留同 NUMA / 同 tier 域内的 GPU

建议默认：

- 单卡：`K = min(8, candidateCount)`
- 多卡：保留全部同 tier 优先域；只有优先域不足时才扩展到次优域

#### 第二步：组合枚举

对裁剪后的候选集合做组合搜索：

- 请求 `N` 张卡，则枚举 `C(M, N)` 个组合
- `M` 为裁剪后的候选数量

建议限制：

- 若 `C(M, N) <= MaxCombinationSearch`
  则完整枚举
- 若超过阈值，则进入降级策略

默认建议：

```text
MaxCombinationSearch = 256
```

这个阈值足以覆盖大多数 `8/16` 卡场景，同时避免极端情况下搜索过深。

#### 第三步：组合排序

对组合按以下顺序排序：

1. 更小的 `tier`
2. 更高的 topology score
3. 更低的碎片度
4. 更稳定的字典序或 GPU UUID 顺序

最后一项用于保证结果可重复，减少调试困难。

### 16.4 降级策略

当组合数量超过阈值时，不做无界枚举，使用分层降级：

1. 先尝试仅在最优 tier 域内搜索
2. 若仍超阈值，则按 allocator 分数或 index 取前 `K` 张 GPU 再搜索
3. 若仍超阈值，则退化为启发式选择，而非全量枚举

建议启发式策略：

- 多卡：优先从同 domain / 同 NUMA 内选出局部最优集合
- 单卡：直接选择 least-damage GPU

### 16.5 复杂度边界示例

常见场景下组合数如下：

- `8` 选 `2`：`28`
- `8` 选 `4`：`70`
- `8` 选 `8`：`1`
- `16` 选 `4`：`1820`
- `16` 选 `8`：`12870`

可以看到：

- `8` 卡节点通常完整枚举可接受
- `16` 卡及以上场景必须有裁剪和阈值控制

因此本方案不建议“无条件全量枚举”。

### 16.6 MVP 建议实现

MVP 建议采用以下固定策略：

1. 若节点候选 GPU 数量 `<= 8`，直接完整枚举
2. 若候选 GPU 数量 `> 8`，先按 NUMA/tier 进行分域裁剪
3. 若分域后组合数仍超过阈值，则使用启发式选择

这样实现简单、性能可控，并且便于后续替换为更复杂的搜索器。

### 16.7 未来演进

后续如果需要支持更多 GPU 数量或更复杂的多厂商 topology，可演进为：

1. beam search
2. branch and bound
3. 基于 pairwise matrix 的贪心扩展
4. 预计算 clique/domain 候选集

但这些都不作为 MVP 必需项。

### 16.8 对现有流程的零影响要求

为了保证系统可维护性和回归风险可控，必须明确以下约束：

1. 当 `GPUNetworkTopologyAware` 未启用时：
   - 不增加现有调度分支
   - 不改变 `GPUResourcesFit` 的既有行为
   - 不改变 allocator 的默认选卡结果
   - 不改变 reserve / permit / postBind 的既有执行路径
2. 当 `GPUNetworkTopologyAware` 启用但无法获取完整 topology 数据时：
   - 严格按照配置退化
   - 不允许出现影响非 topology 场景的异常行为
3. 新增的 topology 数据结构和 evaluator 必须是可插拔的：
   - 删除或关闭插件时，不需要修改其他核心模块逻辑

因此，建议实现上遵循：

1. 所有 topology 逻辑封装在 `internal/scheduler/gputopo/`
2. `GPUResourcesFit` 只增加一个最小协作点：读取 topology plan
3. 该协作点必须在 plugin 未启用或 state 不存在时直接无损回退到当前逻辑

## 17. 与现有插件的协作方式

### 16.1 与 GPUResourcesFit 的关系

`GPUResourcesFit` 保持现有职责：

- 配额检查
- 候选 GPU 过滤
- allocator 评分
- reserve / unreserve
- 最终 annotation patch

`GPUNetworkTopologyAware` 只新增：

- 候选组合拓扑评估
- 节点级拓扑过滤
- 节点级拓扑打分
- 向 `GPUResourcesFit` 提供推荐 GPU 组合

### 16.2 推荐的改造方式

`GPUResourcesFit` 在最终选卡路径中：

1. 检查 `CycleStateGPUTopologyResult`
2. 若当前选中节点存在 `BestGPUIds`
3. 则优先按 `BestGPUIds` 做 reserve
4. 若没有，则回退到当前 allocator 逻辑

## 18. 与 GPUResourcesFit 的裁决关系

`gputopo` 产出的 `BestGPUIds` 不应只是“建议”，而应成为 `GPUResourcesFit` 在最终选卡时的强约束输入。

原因：

1. 如果只是建议，容易出现 topology 插件和 allocator 最终选择不一致
2. Filter/Score 的意义依赖于最终真的使用该组合
3. 只有把 `BestGPUIds` 作为最终裁决输入，拓扑约束才是闭环的

推荐规则：

1. `gputopo` 为节点选出唯一 `BestGPUIds`
2. `GPUResourcesFit` 在选中该节点后优先直接 reserve 该组合
3. 若该组合在 Reserve 阶段发现已失效，则返回失败并触发重试
4. 不再由 allocator 在该节点上重新自由挑选另一组 GPU

这样可以保证：

- topology 语义不被后续 allocator 覆盖
- plugin 间职责边界清晰
- 调试行为可预测

## 19. 建议代码落点

建议修改：

- `internal/scheduler/gputopo/gpu_network_topo.go`
  - 实现 `PreFilter/Filter/Score`
- `internal/config/scheduler_config.go`
  - 扩展 config 字段
- `internal/scheduler/gpuresources/gpuresources.go`
  - 在选卡路径优先消费 topology plan
- `config/samples/scheduler-config.yaml`
  - 调整 plugin args

建议新增：

- `internal/scheduler/gputopo/evaluator.go`
- `internal/scheduler/gputopo/numa_evaluator.go`
- `internal/scheduler/gputopo/types.go`

后续可新增：

- `internal/scheduler/gputopo/vendor_evaluator.go`

## 20. 测试方案

### 18.1 单元测试

覆盖以下场景：

1. `hard` 模式下存在同 NUMA 组合，Filter 成功
2. `hard` 模式下不存在同 NUMA 组合，Filter 失败
3. `soft` 模式下不存在同 NUMA 组合，Filter 放行但 Score 降低
4. 单卡请求选择 least-damage GPU
5. NUMA unknown 按配置处理

### 18.2 集成测试

基于现有 `test/sched` framework：

1. 同时启用 `GPUResourcesFit` 与 `GPUNetworkTopologyAware`
2. 验证节点选择符合 topology 预期
3. 验证最终 GPU 分配结果与 topology plan 一致

### 18.3 回归测试

1. 非 TensorFusion Pod 不受影响
2. 单卡场景不出现明显性能回退
3. reserve / preemption 不破坏现有行为

## 21. 分阶段实施

### Phase 1

- NUMA-only
- 支持 `hard/soft`
- 实现 `Filter + Score`
- `GPUResourcesFit` 消费 `BestGPUIds`

### Phase 2

- 接入 provider 归一化后的 vendor topology
- 支持 pairwise topology matrix
- tier 从 NUMA 扩展到 interconnect domain

### Phase 3

- 与 gang / clique / 多节点拓扑调度联动
- 串联“节点内 GPU topology”和“节点间 network topology”

## 22. Preemption / Permit / Gang 边界

当前 `GPUResourcesFit` 已经参与：

- preemption 相关校验
- reserve / unreserve
- permit 阶段 gang 协作

因此需要明确 `gputopo` 在这些链路中的边界，避免职责重叠。

### 22.1 Preemption

MVP 中，`gputopo` 不单独实现 preemption 逻辑。

边界如下：

1. `gputopo` 只基于当前可见候选 GPU 做 Filter/Score
2. preemption 后的候选变化仍由 `GPUResourcesFit` 负责重新计算
3. `gputopo` 只消费 preemption 之后重新生成的 `CycleStateGPUSchedulingResult`

这意味着：

- `gputopo` 不负责挑选 victim
- `gputopo` 不维护独立的 preemption 状态
- `gputopo` 只参与 preemption 之后的再次拓扑评估

补充原则：

- 不应因为某组 GPU “有 link” 就默认优先抢占它们
- 高质量 interconnect 团簇默认应被保护，而不是优先被打散
- 只有当更高优先级 Pod 明确需要该 topology，且这是满足调度的必要路径时，才应考虑释放这类资源

换言之：

- 默认保护 topology-rich 资源
- 不做“优先抢占有 link GPU”的策略
- 若未来要增强 preemption，也应以“满足性收益 - 抢占代价 - 拓扑破坏代价”综合决策，而不是只看 link 存在与否

### 22.2 Reserve / Unreserve

`gputopo` 不直接实现 Reserve/Unreserve，而是通过 `BestGPUIds` 影响 `GPUResourcesFit` 的 reserve 结果。

规则如下：

1. `Filter/Score` 阶段为节点生成唯一 `BestGPUIds`
2. `GPUResourcesFit.Reserve` 必须优先消费该组合
3. 如果 reserve 时发现组合已失效，则该次调度失败并重新调度
4. `Unreserve` 仍由 `GPUResourcesFit` 负责

这样可以确保：

- topology 结果与最终分配一致
- scheduler framework 生命周期仍以 `GPUResourcesFit` 为主

### 22.3 Permit / Gang

MVP 中，`gputopo` 不直接参与 gang 协调，也不单独维护 gang 级 topology 语义。

MVP 范围内的规则：

1. `gputopo` 只对单个 Pod 的单节点 GPU 组合做评估
2. gang 的 all-or-nothing 语义仍由现有 gang manager / `GPUResourcesFit` 处理
3. 若多个 gang member 分别进入调度周期，`gputopo` 只对各自当前 node 候选做局部最优选择

这意味着 MVP 不保证：

1. 同一 gang 内多个 Pod 之间的全局 GPU topology 最优
2. gang 级的 inter-pod 拓扑协调
3. 节点内 topology 和节点间 network topology 的联合最优

但需要额外明确一点：

- 即使 MVP 不做 gang-level 全局 topology 联合求解，`gputopo` 的局部 GPU 选择也必须尽量保持 gang-friendly
- 也就是说，插件不能为了当前 member Pod 的局部最优，过早打散高质量 topology 团簇
- 单卡或小规模请求仍应优先使用 `least-damage` 策略，为同 gang 后续 member 保留更完整的优质组合

因此，MVP 在 gang 场景下的原则是：

1. 不做 gang 级联合拓扑求解
2. 保持单 Pod 局部 topology-aware
3. 尽量保护完整的高质量 topology 团簇
4. 避免因为局部贪心选择降低同 gang 后续 Pod 的可调度性

这些能力属于后续阶段。

### 22.4 MVP 明确支持与不支持

MVP 支持：

1. 普通单 Pod 的 topology-aware GPU 选择
2. preemption 之后重新计算拓扑 plan
3. gang 场景下的单 Pod 局部 topology-aware 选择
4. gang-friendly 的局部团簇保护策略
5. 插件关闭时完全保持现有流程不变

MVP 不支持：

1. topology-aware victim selection
2. gang 级全局 topology 联合求解
3. 多节点 network topology 与单节点 GPU topology 联合优化

### 22.5 后续演进方向

后续可逐步增强为：

1. 在 gang manager 中引入 group-level topology preference
2. 在 preemption 评估中考虑释放后可形成的拓扑团簇
3. 与 Koordinator/Volcano 类 network topology 能力联动，实现 node 内和 node 间统一拓扑调度

## 23. 可观测性

为了让 topology 调度可以排查和验证，建议增加以下可观测性输出。

### 22.1 日志

在 `gputopo` 的关键路径输出：

- pod 名称
- node 名称
- 请求 GPU 数量
- 使用的数据源：`numa` / `generic-topology` / `unknown`
- 候选 GPU 数量
- 枚举组合数量
- 是否触发降级策略
- 最终 `BestGPUIds`
- `tier` 和 `score`

### 22.2 Event

建议在以下场景发事件：

- `GPUInterconnectTopologyUnsatisfied`
  - `hard` 模式下拓扑约束无法满足
- `GPUInterconnectTopologyFallback`
  - 从 vendor topology 退化到 NUMA 或 unknown
- `GPUInterconnectTopologySelected`
  - 调试模式下记录最终选择的 topology 组合

### 22.3 Metrics

建议增加指标：

- `gputopo_filter_duration_seconds`
- `gputopo_score_duration_seconds`
- `gputopo_combination_count`
- `gputopo_fallback_total`
- `gputopo_unsatisfied_total`
- `gputopo_search_degraded_total`

### 22.4 调试价值

这些输出可以帮助回答以下问题：

1. 这次调度是否真的用了 topology 数据
2. 为什么某个节点被拒绝
3. 为什么最终选了这组 GPU
4. 是否因为搜索阈值触发了降级

## 24. 可维护性验收标准

从架构和实现验收角度，建议明确以下标准：

1. 代码结构
   - topology 相关逻辑集中在独立目录和独立插件内
   - 不把厂商分支判断散落进调度主流程
2. 依赖方向
   - `gputopo` 依赖 `GPUResourcesFit` 产出的 state
   - `GPUResourcesFit` 不依赖具体厂商 topology 实现
3. 可扩展性
   - 新增厂商支持时，通过新增 provider 映射或 evaluator 扩展完成
   - 不要求重写 scheduler 主流程
4. 可关闭性
   - 关闭插件后，现有测试结果与行为保持一致
   - 不引入隐式 side effect
5. 可回归验证
   - 必须有“插件启用”和“插件关闭”两套测试覆盖
   - 必须验证关闭插件时行为与旧版本一致

## 25. 结论

`gputopo` 最合理的落地方式是：

1. 保持为独立 scheduler plugin
2. 复用 `GPUResourcesFit` 的候选 GPU 结果
3. 借鉴 HAMI 的“先选最佳组合、再打分”的思路
4. 借鉴 Volcano 的“独立 topology 状态和 tier 抽象”的思路
5. 不把拓扑语义并入 `gpufit`

该方案既符合当前代码边界，也适合作为后续 vendor topology 扩展的基础。
