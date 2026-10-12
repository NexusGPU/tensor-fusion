# ACK 环境手工测试用例

在 `e2e/ack` 拉起的阿里云 ACK 集群上，手工把关键功能走一遍。每条用例给出：目的、命令、在 cloud app（app.tensor-fusion.ai）上看什么、预期结果。命令可以整段复制执行。自动化套件里对应的用例 ID 写在标题后面，想直接跑自动化版本用 `make ack-test TIER=<档位> FOCUS='<ID>'`。

约定：

- 在仓库的 `e2e/` 目录下执行；`T4` 指 `ecs.gn6i-c4g1.xlarge` 上的 Tesla T4（产品表里 65 TFLOPS / 15Gi）。
- GPU 节点全部由 Karpenter 拉起，空闲 2 分钟后回收。第一条需要 GPU 的用例要等 4–5 分钟（实例启动 + ACK 装驱动），之后只要节点还在就是秒级。
- 每条用例用自己的 namespace，结尾的 `kubectl delete ns` 就是清理。

## 0. 准备

```bash
cd e2e
export KUBECONFIG=$PWD/ack/.work/kubeconfig     # 集群不在时先 make ack-up（约 10 分钟）
alias k=kubectl

k get nodes -L e2e.tensor-fusion.ai/role,node.kubernetes.io/instance-type
k -n tensor-fusion-sys get pods
k get tensorfusioncluster,gpupool,nodepool,ecsnodeclass,nodeoverlay
k -n tensor-fusion-sys logs deploy/tensor-fusion-sys-controller -c cluster-agent --tail=3   # 应有 Connected to cloud.
```

预期：1 台 `system` 节点 Ready；controller Pod 3/3；NodePool `tf-gpu`、`tf-cpu` 与两个 ECSNodeClass 都是 Ready；NodeOverlay `tensor-fusion-overlay` Ready。

cloud app：用环境所属组织的管理员账号登录，环境列表里的 `ack-us-east-1` 显示已连接。之后每条用例创建的节点、GPU、workload 都应在这个环境下出现。

所有负载脚本放进一个 ConfigMap，后面的用例直接挂载：

```bash
scripts() { k -n "$1" create configmap e2e-scripts --from-file=suite/workloads --dry-run=client -o yaml | k apply -f - >/dev/null; }

# tfpod <ns> <名字> <隔离模式> <算力百分比> <显存> <容器命令>
tfpod() {
cat <<YAML | k apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: $2
  namespace: $1
  labels: {app: $2, tensor-fusion.ai/enabled: "true"}
  annotations:
    tensor-fusion.ai/isolation: $3
    tensor-fusion.ai/compute-percent-request: "$4"
    tensor-fusion.ai/compute-percent-limit: "$4"
    tensor-fusion.ai/vram-request: $5
    tensor-fusion.ai/vram-limit: $5
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 1
  containers:
    - name: main
      image: pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
      command: ["sh", "-c", "$6"]
      volumeMounts: [{name: e2e, mountPath: /e2e}]
  volumes: [{name: e2e, configMap: {name: e2e-scripts}}]
YAML
}
```

## 1. Karpenter 冷启动：从 0 台 GPU 节点到 Pod 运行（KRP-A08、INS-03）

目的：只加 label 和注解的 Pod 让 Karpenter 从 `tf-gpu` NodePool 拉起节点，节点被 TensorFusion 纳管、GPU 被发现。

```bash
k get nodeclaim                      # 起点：没有 tf-gpu 的 NodeClaim
k create ns t1 && scripts t1
date; tfpod t1 cold soft 20 2Gi "python3 /e2e/gpu_probe.py info && sleep 600"

watch -n 5 'kubectl get nodeclaim; kubectl get gpunode,gpu; kubectl -n t1 get pod -o wide'
```

预期，按时间顺序：

1. 约 10 秒内出现 **恰好 1 个** `tf-gpu-*` NodeClaim（机型 `ecs.gn6i-c4g1.xlarge`）。出现第 2、3 个说明 Karpenter 用的不是打过补丁的镜像（见 README 的「provider 补丁」）。
2. 约 4–5 分钟后节点 Ready，随后出现 `hypervisor-<节点名>` Pod、GPUNode（Running）和 1 个 GPU（`Tesla T4`，65 TFLOPS / 15Gi，Running）。
3. Pod `cold` Running，`kubectl -n t1 logs cold` 的 RESULT 行里 `cuda_available: true`、`device_count: 1`，容器内 `nvidia-smi` 看到的总显存是 2048 MiB（配额），不是整卡。

```bash
k get node -l e2e.tensor-fusion.ai/role=gpu -o json | jq '.items[0].status.capacity | with_entries(select(.key|test("index_0|nvidia")))'
k -n t1 get pod cold -o json | jq '.metadata.annotations | with_entries(select(.key|test("gpu-ids|tflops|vram")))'
```

cloud app：环境下出现 1 个节点、1 张 T4，分配量 13 TFLOPS / 2Gi；workload `cold` 在列表里。

清理：`k delete ns t1`。节点空闲 2 分钟后被 Karpenter 回收（用例 7 专门看这一步）。接着做用例 2–5 的话不用等，节点会被复用。

## 2. 切分：一张卡跑三个 Pod（ISO-01）

```bash
k create ns t2 && scripts t2
for i in a b c; do tfpod t2 s-$i soft 20 2Gi "python3 /e2e/gpu_probe.py info && sleep 600"; done
k -n t2 wait pod --all --for=condition=Ready --timeout=600s
k -n t2 get pod -o json | jq -r '.items[] | [.metadata.name, .spec.nodeName, .metadata.annotations["tensor-fusion.ai/gpu-ids"]] | @tsv'
k get gpu -o json | jq '.items[] | {capacity: .status.capacity, available: .status.available, apps: [.status.runningApps[]?.name]}'
```

预期：三个 Pod 的 `gpu-ids` 相同；GPU 的 `available` 是 26 TFLOPS / 9Gi（65 − 3×13，15 − 3×2），几秒内到位；每个容器里 `nvidia-smi` 的总显存是 2048 MiB。

cloud app：这张 GPU 上显示 3 个 workload，已分配约 60% 算力、40% 显存。

清理：`k delete ns t2`，然后 `k get gpu` 的 `available` 回到 65 / 15Gi。

## 3. 显存上限（ISO-03 的 soft 部分）

```bash
k create ns t3 && scripts t3
tfpod t3 vram soft 20 4Gi "python3 /e2e/gpu_probe.py vram --target-mib 6144"
k -n t3 wait pod vram --for=jsonpath='{.status.phase}'=Succeeded --timeout=600s; k -n t3 logs vram | grep RESULT
```

预期：RESULT 里实际分配到约 3840 MiB 后失败（`oom: true`），`total_mib` 是 4096。申请 6 GiB 拿不到，容器看到的就是配额。

清理：`k delete ns t3`。

## 4. 算力限额：soft 模式的实际行为（ISO-02）

目的：看限额是否生效，以及「偏松」的幅度。soft 限流控制的是 GPU 忙时比例（NVML 利用率），不是吞吐本身，所以两个都要看。

```bash
k create ns t4 && scripts t4
for p in 100 50 25; do
  tfpod t4 burn-$p soft $p 4Gi "python3 /e2e/gpu_probe.py burn --seconds 150 --warmup 20 --size 2048 --fp16"
  k -n t4 wait pod burn-$p --for=jsonpath='{.status.phase}'=Succeeded --timeout=900s
  k -n t4 logs burn-$p | grep RESULT | jq -r '"'$p'%: \(.iters_per_s|floor) it/s  p50=\(.iter_s.p50*1000|.*100|floor/100)ms p99=\(.iter_s.p99*1000|.*100|floor/100)ms stalls>=0.4s=\(.stalls_ge_400ms)"'
  k -n t4 delete pod burn-$p --wait
done
```

设备侧的读数（每分钟一行，来自 hypervisor 遥测）：

```bash
sql() { k get --raw "/api/v1/namespaces/greptimedb/services/greptimedb-standalone:4000/proxy/v1/sql?db=public&sql=$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))' "$1")" | jq -c '.output[0].records.rows[]'; }
sql "select greptime_timestamp, compute_percentage, compute_tflops, power_usage, temperature from tf_gpu_usage order by greptime_timestamp desc limit 12"
```

预期（T4 实测，见 TEST-SUMMARY.md）：

| 限额 | 吞吐相对 100% | 设备利用率 | 判定 |
|---|---|---|---|
| 100% | 1.00 | 约 97% | — |
| 50% | 约 0.80 | 约 56% | 利用率在份额 +10 个点以内；吞吐不低于份额的 85% |
| 25% | 约 0.22–0.24 | 约 25–30% | 同上 |

50% 档吞吐高于 50% 是预期内的：T4 在间歇负载下每毫秒完成的计算多于持续满载。不允许的是低于份额（把应用限得太紧），以及出现 ≥0.4 秒的停顿。

cloud app：GPU 的利用率曲线在三段负载期间依次是约 97%、56%、25–30%。

清理：`k delete ns t4`。

## 5. 共调度：两个 LLM 租户各占 50%（PERF-CO-01）

这是切分 + 共调度的核心场景，建议直接跑自动化用例，它会起两个 vLLM、同时压测并出判定：

```bash
make ack-test TIER=perf-quick FOCUS='PERF-CO-01'
python3 tools/perf_history.py record --version <版本> --env ack-us-east-1-t4 results/<run-id>   # 可选：留档
open perf-history/index.html
```

预期：两个租户落在同一张 GPU；合计吞吐不低于独占整卡的 70%；两个租户的吞吐之比不低于 0.7；没有 ≥0.9 秒的 token 间隔。实际数字见 TEST-SUMMARY.md 与查看器的「切分 + 共调度」一节。

手工观察：用例运行期间 `k get pods -A | grep vllm-co`，以及 cloud app 上这张 GPU 同时挂着两个 workload、利用率接近 100%。

## 6. 兼容模式扩容：GPU 不足时由 TensorFusion 扩容器建 NodeClaim（KRP-A02、KRP-A03）

目的：验证「AutoSelect + Karpenter NodePool」的核心闭环。先占满现有 GPU，再来一个 Pod。

```bash
k create ns t6 && scripts t6
# 占满现有的每一张卡（整卡 shared）
n=$(k get gpu --no-headers | wc -l | tr -d ' '); [ "$n" -gt 0 ] || n=1
for i in $(seq 1 $n); do
cat <<YAML | k apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: full-$i
  namespace: t6
  labels: {tensor-fusion.ai/enabled: "true"}
  annotations: {tensor-fusion.ai/isolation: shared, tensor-fusion.ai/gpu-count: "1"}
spec:
  terminationGracePeriodSeconds: 1
  containers: [{name: main, image: public.ecr.aws/docker/library/debian:bookworm-slim, command: ["sleep", "3600"]}]
YAML
done
k -n t6 wait pod --all --for=condition=Ready --timeout=900s
k get nodeclaim; date
tfpod t6 extra soft 20 2Gi "sleep 600"
watch -n 5 'kubectl get nodeclaim -o wide; kubectl -n t6 get pod extra -o wide; kubectl get events -n t6 --field-selector involvedObject.name=extra | tail -5'
```

预期：

1. `extra` 先 Pending，事件里有 `GPUQuotaOrCapacityNotEnough`。
2. 十几秒内新增 **恰好 1 个** NodeClaim，名字是 `tensor-fusion-<10 位>`（扩容器建的，不是 Karpenter 的 `tf-gpu-*`），带 label `karpenter.sh/nodepool=tf-gpu` 和注解 `tensor-fusion.ai/expansion-source`，不带 `karpenter.sh/do-not-disrupt`：
   ```bash
   k get nodeclaim -o json | jq '.items[] | {name: .metadata.name, pool: .metadata.labels["karpenter.sh/nodepool"], src: .metadata.annotations["tensor-fusion.ai/expansion-source"]}'
   k get --raw /api/v1/namespaces/tensor-fusion-sys/services/tensor-fusion-sys:8080/proxy/api/node-scaler | jq .
   ```
3. 新节点 GPU 变为 Running 后，`extra` 在 30 秒内被绑定并 Running（2.17.24 之前这里要等 5 分钟）。

一次扩容接住多个 Pod（KRP-A03）：再起三个 20% 的 Pod，它们应落在同一个新节点、同一张卡，不再新增 NodeClaim。

cloud app：节点数 +1，新节点上出现 `extra`。

清理：`k delete ns t6`。

## 7. 缩容：负载清空后节点被回收（KRP-A09）

接在任何一条用了 GPU 的用例之后，删掉 namespace 即开始计时：

```bash
date; watch -n 10 'kubectl get nodeclaim; kubectl get gpunode,gpu; kubectl get node -l e2e.tensor-fusion.ai/role=gpu'
```

预期：`consolidateAfter=2m` 到点后 NodeClaim 进入删除，Node、GPUNode、GPU 一并消失；ECS 实例被释放：

```bash
aliyun ecs DescribeInstances --RegionId us-east-1 --Tag.1.Key Project --Tag.1.Value tf-e2e-ack | jq -r '.Instances.Instance[] | [.InstanceId,.InstanceType,.Status] | @tsv'
```

只剩 system 节点那一台。cloud app：节点和 GPU 从环境里消失。

## 8. 路径 B：`provisioningMode: Karpenter`，由容量水位驱动（KRP-B02、KRP-B10）

自动化用例会往 TensorFusionCluster 里加一个 `provisioningMode: Karpenter` 的 pool、等它拉起节点、调度一个 Pod 上去、再删掉 pool 并检查没有残留。手工做这件事要改 TensorFusionCluster，建议直接跑：

```bash
make ack-test TIER=karpenter-finops FOCUS='KRP-B02|KRP-B10'
```

运行期间手工观察：

```bash
watch -n 5 'kubectl get gpupool; kubectl get gpunodeclaim; kubectl get nodeclaim -o wide; kubectl get gpunode,gpu'
```

预期：pool `tensor-fusion-karp` 出现 → 1 个 GPUNodeClaim（机型 `ecs.gn6i-c8g1.2xlarge`，带 `tensor-fusion.ai/hourly-pricing` 注解）→ 同名 NodeClaim（owner 是 GPUNodeClaim，带 `karpenter.sh/do-not-disrupt`，请求 `nvidia.com/gpu: 1`）→ 节点 → GPUNodeClaim `Bound` → 指定该 pool 的 Pod 落在这台节点；30 秒内不出现第二个 GPUNodeClaim；删 pool 后 GPUPool、GPUNodeClaim、NodeClaim、节点、GPUNode、GPU 全部消失。

需要带阿里云机型目录的 operator（≥ 包含 #820 的版本）；更早的版本在这里选不出任何机型。

## 9. FinOps：分片与整卡的节点数对比（FIN-01）

```bash
make ack-test TIER=karpenter-finops FOCUS='FIN-01'
cat results/<run-id>/FIN-01.finops.json | jq .
```

预期：3 个 20% 的分片负载只用 1 台节点，3 个整卡负载用 3 台；输出两种方案的节点小时、按量成本（单价取自 `pricing/alibaba-us-east-1.json`）和节省比例。cloud app 的成本视图里同一时间段应能看到节点数的差别。

## 10. 恢复：hypervisor 重启、operator 重启、节点被删（NLC-01、NLC-02、NLC-09）

```bash
k create ns t10 && scripts t10
tfpod t10 app soft 20 2Gi "sleep 1800"
k -n t10 wait pod app --for=condition=Ready --timeout=900s
node=$(k -n t10 get pod app -o jsonpath='{.spec.nodeName}'); k get gpu -o json | jq '.items[].status.available'

# hypervisor 重启
k -n tensor-fusion-sys delete pod hypervisor-$node; date
watch -n 3 "kubectl -n tensor-fusion-sys get pod hypervisor-$node; kubectl get gpu; kubectl -n t10 get pod app"
```

预期：新的 hypervisor 约 20 秒内 Ready，期间 GPU 短暂变 `Pending` 再回到 `Running`；`app` 不重启；`available` 前后一致（52 TFLOPS / 13Gi）。

```bash
# operator 重启
k get --raw /api/v1/namespaces/tensor-fusion-sys/services/tensor-fusion-sys:8080/proxy/api/allocation | jq '.allocation | keys'
k -n tensor-fusion-sys rollout restart deploy/tensor-fusion-sys-controller && k -n tensor-fusion-sys rollout status deploy/tensor-fusion-sys-controller
k get --raw /api/v1/namespaces/tensor-fusion-sys/services/tensor-fusion-sys:8080/proxy/api/allocation | jq '.allocation | keys'
```

预期：两次的 key 集合相同；重启后再起一个 60% 的 Pod 加上已有的 20% 仍能放进同一张卡，再起第二个 60% 的不会落到这张卡。

节点被删（用 Deployment 才会重建）：

```bash
k -n t10 delete pod app
cat <<'YAML' | k apply -f -
apiVersion: apps/v1
kind: Deployment
metadata: {name: web, namespace: t10}
spec:
  replicas: 1
  selector: {matchLabels: {app: web}}
  template:
    metadata:
      labels: {app: web, tensor-fusion.ai/enabled: "true"}
      annotations: {tensor-fusion.ai/isolation: soft, tensor-fusion.ai/compute-percent-request: "20", tensor-fusion.ai/compute-percent-limit: "20", tensor-fusion.ai/vram-request: 2Gi, tensor-fusion.ai/vram-limit: 2Gi}
    spec:
      terminationGracePeriodSeconds: 1
      containers: [{name: main, image: public.ecr.aws/docker/library/debian:bookworm-slim, command: ["sleep", "3600"]}]
YAML
k -n t10 rollout status deploy/web --timeout=900s
node=$(k -n t10 get pod -l app=web -o jsonpath='{.items[0].spec.nodeName}'); k delete node $node; date
watch -n 5 'kubectl get nodeclaim; kubectl get gpunode,gpu; kubectl -n t10 get pod -o wide'
```

预期：被删节点的 GPUNode、GPU 几秒内清理；Deployment 的新 Pod 先 Pending，随后在另一台（已有或新拉起的）GPU 节点上 Running。

cloud app：重启期间环境保持在线（operator 重启时 agent 会短暂重连）；删节点后节点列表随之变化。

清理：`k delete ns t10`。

## 11. 配额、Gang、按型号调度（SCH-10、SCH-11、SCH-04）

这三条在自动化里各十几到几十秒，直接跑并读结果：

```bash
make ack-test TIER=core FOCUS='SCH-10|SCH-11|SCH-04'
```

预期：超出 `GPUResourceQuota` 的 Pod 保持 Pending 且事件含 quota；Gang 的两个成员一起 Running；写了不存在型号的 Pod 一直 Pending 且不触发扩容。

## 12. 指标与控制台（OBS-02、CLD-01）

有负载运行 2 分钟后：

```bash
sql "show tables"
for t in tf_gpu_usage tf_node_metrics tf_pool_metrics tf_worker_resources tf_worker_usage; do echo "$t: $(sql "select count(*) from $t" 2>/dev/null)"; done
sql "select node, instance_type, hourly_price, currency, price_source from tf_node_metrics order by greptime_timestamp desc limit 1"
```

预期：前四张表有数据；`tf_node_metrics` 的 `hourly_price` 是 T4 机型的按量价（1.0941，`price_source=list`）。`tf_worker_usage` 在当前版本没有数据，是已知问题（见 TEST-SUMMARY.md）。

cloud app 上逐项确认：环境在线；节点 / GPU / workload 列表与 `kubectl` 一致；GPU 利用率、显存、功耗、温度曲线有数据；pool 的容量与已分配量一致；成本数字非零。

## 13. 已知不可用的功能（不要当成环境问题）

| 功能 | 现象 | 原因 |
|---|---|---|
| remote 模式（ISO-06） | worker Pod `CrashLoopBackOff`，退出码 0，客户端 `cuda_available: false` | community 版 remote worker 启动时报 `License is not set` 后退出，chart 和 operator 没有下发 License 的配置 |
| hard 隔离（ISO-04、PERF-LLM-02） | sidecar worker 退出，业务进程 `Bus error` | 同上 |
| soft 下 `disable-features: gpu-limiter`（ISO-13） | 30% 限额仍被限到约 27% | webhook 只在 remote / hard 的客户端路径上处理这个注解 |
| 无 libstdc++ 的镜像（ISO-14） | 容器内进程以 127 退出 | soft limiter 经 `ld.so.preload` 注入，依赖 glibc 与 libstdc++ |

想亲眼看一遍 remote / hard 的现象：把用例 0 的 `tfpod` 里隔离模式换成 `hard`，或给 Pod 加 `tensor-fusion.ai/is-local-gpu: "false"`，然后 `kubectl logs <pod> -c tensorfusion-worker`。

## 14. 收尾

```bash
k get ns | grep -E '^t[0-9]+' | awk '{print $1}' | xargs -r kubectl delete ns
make ack-sweep        # 列出这套环境现存的全部云资源
make ack-down         # 销毁并再次清点；最后一行应全部是 clean
```

`ack/guard.sh` 在环境存活期间每 2 分钟检查一次：没有 NodeClaim 对应的 ECS 实例会被终止，GPU 实例超过 4 台时所有 NodePool 的上限被置 0。日志在 `ack/.work/guard.log`。
