# NVLink 拓扑感知调度（MVP）说明

## 1. 目标

本次改动目标：

1. 仅修改 **一个 CR**：`GPU`（`tensor-fusion.ai/v1, kind=GPU`）
2. `node-discovery` 上报 NVLink 拓扑与带宽信息到 `GPU.status`
3. 调度器在 `Score + Reserve` 阶段支持 NVLink 感知：
   - 单卡/半卡请求：尽量保护高互联带宽 GPU（避免破坏 NVLink 组）
   - 多卡请求：尽量选择同一高带宽互联组
4. 保持兼容：默认关闭（`enableNvlinkAware: false`）

---

## 2. CR 变更（仅 GPU）

### 2.1 新增字段

在 `GPU.status` 新增：

- `nvLink.peerCount`
- `nvLink.totalLinkCount`
- `nvLink.totalBandwidthMBps`
- `nvLink.peers[]`
  - `peerUUID`
  - `linkCount`
  - `linkVersion`
  - `bandwidthMBps`

> 说明：带宽字段采用 `int64`（MB/s），避免 CRD `float` 类型兼容问题。

### 2.2 兼容性

- 字段均为 `omitempty`，历史对象无需迁移
- 未上报拓扑时，调度自动回退到资源分配逻辑

---

## 3. Node Discovery 变更

文件：`cmd/nodediscovery/main.go`

### 3.1 采集流程

1. 首轮遍历 GPU：采集基础信息 + `GetPciInfo()`，建立 `busID -> UUID` 映射
2. 二轮遍历 GPU：按 `NVLINK_MAX_LINKS` 读取每条 link 状态：
   - `GetNvLinkState`
   - `GetNvLinkRemotePciInfo`
   - `GetNvLinkVersion`
3. 按 peer 聚合 `linkCount` 与 `bandwidthMBps`
4. 写入 `GPU.status.nvLink`

### 3.2 降级策略

- NVLink/PCI 查询失败时不报错退出，仅跳过该 link
- 无拓扑数据时 `nvLink=nil`，调度侧自动回退

---

## 4. 调度器改造

核心文件：

- `internal/scheduler/gpuresources/gpuresources.go`
- `internal/scheduler/gpuresources/topology_score.go`

### 4.1 新增调度状态

`GPUSchedulingStateData` 增加：

- `PreferredNodeGPUs map[node][]gpuName`
- `NodeTopologyScore map[node]score(0..100)`

### 4.2 Score 逻辑

- 原资源分：保留（`resourceScore`）
- 若启用 NVLink 感知且节点有拓扑分：
  - `final = resourceWeight * resourceScore + topologyWeight * topologyScore`
- 无拓扑数据时不降级惩罚，沿用原分数

### 4.3 Reserve 选卡逻辑

- 优先使用 `PreFilter` 生成的 `PreferredNodeGPUs`
- 缺失时回退原有资源分选卡

### 4.4 策略细节

- 单卡：在资源分接近时，优先选择 `totalBandwidthMBps` 更低的卡（保护高互联卡）
- 多卡：枚举节点内组合，综合资源分与 pair 拓扑分选最优组合

---

## 5. 可配置项（非 CR）

文件：`internal/config/scheduler_config.go`

`GPUResourcesFit.args` 新增：

- `enableNvlinkAware`（bool）
- `resourceScoreWeight`（float，默认 0.7）
- `topologyScoreWeight`（float，默认 0.3）
- `singleGpuProtectWeight`（float，默认 0.6）

示例已更新：

- `config/samples/scheduler-config.yaml`（示例开启）
- `charts/tensor-fusion/values.yaml`（默认关闭，便于灰度）

---

## 6. 代码级扩展点（减少硬编码）

1. 拓扑评分独立到 `topology_score.go`，与主调度流程解耦
2. `GPU.status.nvLink` 结构可继续扩展（如后续加 NVSwitch/PCIe）
3. 带宽模型集中在 discovery 端估算函数，调度端只消费统一字段
4. 调度启用通过配置控制，无需改代码路径

---

## 7. 测试与验证

### 7.1 已执行并通过

```bash
go test ./cmd/nodediscovery ./internal/scheduler/gpuresources ./internal/gpuallocator ./internal/config
```

### 7.2 已执行但存在环境/存量问题（与本改动无关）

```bash
go test ./...
```

失败原因主要为：

- 现有 `internal/scheduler/expander` 对 k8s scheduler 私有方法依赖导致构建失败
- e2e 依赖 docker daemon，本机权限不足（`/var/run/docker.sock: permission denied`）

---

## 8. 后续建议

1. 若线上灰度：先在单个池开启 `enableNvlinkAware=true`
2. 观察指标：调度耗时、跨卡训练吞吐、单卡请求命中率
3. 后续可将 discovery 端带宽估算改为“实时计数器采样”模式（可选）

