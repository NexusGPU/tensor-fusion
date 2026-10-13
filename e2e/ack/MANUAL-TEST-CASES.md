# Manual test cases for the ACK environment

Walk through the key features by hand on the Alibaba Cloud ACK cluster brought up by `e2e/ack`. Each case gives: the purpose, the commands, what to look at in the cloud app (app.tensor-fusion.ai), and the expected result. The commands can be copied and run as whole blocks. The IDs of the corresponding cases in the automated suite are given after each title; to run the automated version directly, use `make ack-test TIER=<tier> FOCUS='<ID>'`.

Conventions:

- Run from the repository's `e2e/` directory; `T4` means the Tesla T4 on `ecs.gn6i-c4g1.xlarge` (65 TFLOPS / 15Gi in the product table).
- All GPU nodes are launched by Karpenter and reclaimed after 2 minutes idle. The first case that needs a GPU waits 4–5 minutes (instance boot + ACK installing the driver); after that it takes seconds as long as the node is still there.
- Each case uses its own namespace, and the `kubectl delete ns` at the end is the cleanup.

## 0. Preparation

```bash
cd e2e
export KUBECONFIG=$PWD/ack/.work/kubeconfig     # if the cluster is not there, run make ack-up first (about 10 minutes)
alias k=kubectl

k get nodes -L e2e.tensor-fusion.ai/role,node.kubernetes.io/instance-type
k -n tensor-fusion-sys get pods
k get tensorfusioncluster,gpupool,nodepool,ecsnodeclass,nodeoverlay
k -n tensor-fusion-sys logs deploy/tensor-fusion-sys-controller -c cluster-agent --tail=3   # should show Connected to cloud.
```

Expected: 1 `system` node Ready; the controller Pod 3/3; NodePools `tf-gpu` and `tf-cpu` and both ECSNodeClasses Ready; NodeOverlay `tensor-fusion-overlay` Ready.

cloud app: log in with an administrator account of the organization that owns the environment; `ack-us-east-1` shows as connected in the environment list. The nodes, GPUs and workloads created by every later case should appear under this environment.

Put all workload scripts into one ConfigMap, which the later cases mount directly:

```bash
scripts() { k -n "$1" create configmap e2e-scripts --from-file=suite/workloads --dry-run=client -o yaml | k apply -f - >/dev/null; }

# tfpod <ns> <name> <isolation mode> <compute percent> <VRAM> <container command>
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

## 1. Karpenter cold start: from 0 GPU nodes to a running Pod (KRP-A08, INS-03)

Purpose: a Pod with only the label and annotations makes Karpenter launch a node from the `tf-gpu` NodePool; the node is managed by TensorFusion and its GPU is discovered.

```bash
k get nodeclaim                      # starting point: no tf-gpu NodeClaim
k create ns t1 && scripts t1
date; tfpod t1 cold soft 20 2Gi "python3 /e2e/gpu_probe.py info && sleep 600"

watch -n 5 'kubectl get nodeclaim; kubectl get gpunode,gpu; kubectl -n t1 get pod -o wide'
```

Expected, in chronological order:

1. Within about 10 seconds **exactly 1** `tf-gpu-*` NodeClaim appears (instance type `ecs.gn6i-c4g1.xlarge`). A 2nd or 3rd one means Karpenter is not using the patched image (see "provider patch" in the README).
2. After about 4–5 minutes the node is Ready, followed by the `hypervisor-<node name>` Pod, the GPUNode (Running) and 1 GPU (`Tesla T4`, 65 TFLOPS / 15Gi, Running).
3. Pod `cold` is Running; the RESULT line of `kubectl -n t1 logs cold` has `cuda_available: true` and `device_count: 1`, and the total VRAM that `nvidia-smi` sees inside the container is 2048 MiB (the quota), not the whole GPU.

```bash
k get node -l e2e.tensor-fusion.ai/role=gpu -o json | jq '.items[0].status.capacity | with_entries(select(.key|test("index_0|nvidia")))'
k -n t1 get pod cold -o json | jq '.metadata.annotations | with_entries(select(.key|test("gpu-ids|tflops|vram")))'
```

cloud app: 1 node and 1 T4 appear under the environment, with 13 TFLOPS / 2Gi allocated; workload `cold` is in the list.

Cleanup: `k delete ns t1`. The node is reclaimed by Karpenter after 2 minutes idle (case 7 looks at this step specifically). If you continue with cases 2–5 there is no need to wait; the node is reused.

## 2. Fractional sharing: three Pods on one GPU (ISO-01)

```bash
k create ns t2 && scripts t2
for i in a b c; do tfpod t2 s-$i soft 20 2Gi "python3 /e2e/gpu_probe.py info && sleep 600"; done
k -n t2 wait pod --all --for=condition=Ready --timeout=600s
k -n t2 get pod -o json | jq -r '.items[] | [.metadata.name, .spec.nodeName, .metadata.annotations["tensor-fusion.ai/gpu-ids"]] | @tsv'
k get gpu -o json | jq '.items[] | {capacity: .status.capacity, available: .status.available, apps: [.status.runningApps[]?.name]}'
```

Expected: the three Pods have the same `gpu-ids`; the GPU's `available` is 26 TFLOPS / 9Gi (65 − 3×13, 15 − 3×2), reached within a few seconds; the total VRAM from `nvidia-smi` in each container is 2048 MiB.

cloud app: this GPU shows 3 workloads, with about 60% of compute and 40% of VRAM allocated.

Cleanup: `k delete ns t2`, after which `available` in `k get gpu` returns to 65 / 15Gi.

## 3. VRAM cap (the soft part of ISO-03)

```bash
k create ns t3 && scripts t3
tfpod t3 vram soft 20 4Gi "python3 /e2e/gpu_probe.py vram --target-mib 6144"
k -n t3 wait pod vram --for=jsonpath='{.status.phase}'=Succeeded --timeout=600s; k -n t3 logs vram | grep RESULT
```

Expected: in RESULT the allocation fails after about 3840 MiB has actually been allocated (`oom: true`), and `total_mib` is 4096. A request for 6 GiB cannot be satisfied, and what the container sees is the quota.

Cleanup: `k delete ns t3`.

## 4. Compute limit: actual behavior in soft mode (ISO-02)

Purpose: see whether the limit takes effect and how far on the loose side it is. soft throttling controls the GPU busy-time fraction (NVML utilization), not throughput itself, so both have to be looked at.

```bash
k create ns t4 && scripts t4
for p in 100 50 25; do
  tfpod t4 burn-$p soft $p 4Gi "python3 /e2e/gpu_probe.py burn --seconds 150 --warmup 20 --size 2048 --fp16"
  k -n t4 wait pod burn-$p --for=jsonpath='{.status.phase}'=Succeeded --timeout=900s
  k -n t4 logs burn-$p | grep RESULT | jq -r '"'$p'%: \(.iters_per_s|floor) it/s  p50=\(.iter_s.p50*1000|.*100|floor/100)ms p99=\(.iter_s.p99*1000|.*100|floor/100)ms stalls>=0.4s=\(.stalls_ge_400ms)"'
  k -n t4 delete pod burn-$p --wait
done
```

Device-side readings (one row per minute, from hypervisor telemetry):

```bash
sql() { k get --raw "/api/v1/namespaces/greptimedb/services/greptimedb-standalone:4000/proxy/v1/sql?db=public&sql=$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))' "$1")" | jq -c '.output[0].records.rows[]'; }
sql "select greptime_timestamp, compute_percentage, compute_tflops, power_usage, temperature from tf_gpu_usage order by greptime_timestamp desc limit 12"
```

Expected (measured on the T4, see TEST-SUMMARY.md):

| Limit | Throughput relative to 100% | Device utilization | Verdict |
|---|---|---|---|
| 100% | 1.00 | about 97% | — |
| 50% | about 0.80 | about 56% | Utilization within share + 10 points; throughput not lower than 85% of the share |
| 25% | about 0.22–0.24 | about 25–30% | Same as above |

Throughput above 50% at the 50% setting is expected: under an intermittent load a T4 completes more computation per millisecond than under sustained full load. What is not allowed is falling below the share (limiting the application too tightly) or stalls ≥0.4 seconds.

cloud app: the GPU's utilization curve is about 97%, 56% and 25–30% in turn during the three load segments.

Cleanup: `k delete ns t4`.

## 5. Co-scheduling: two LLM tenants at 50% each (PERF-CO-01)

This is the core scenario for fractional sharing + co-scheduling. Running the automated case directly is recommended; it starts two vLLM instances, load-tests them at the same time and produces the verdict:

```bash
make ack-test TIER=perf-quick FOCUS='PERF-CO-01'
python3 tools/perf_history.py record --version <version> --env ack-us-east-1-t4 results/<run-id>   # optional: record it in the history
open perf-history/index.html
```

Expected: the two tenants land on the same GPU; combined throughput is not lower than 70% of the whole GPU alone; the ratio of the two tenants' throughput is not lower than 0.7; there are no inter-token gaps ≥0.9 seconds. The actual numbers are in TEST-SUMMARY.md and in the "Fractional sharing + co-scheduling" section of the viewer.

Manual observation: while the case is running, `k get pods -A | grep vllm-co`, and in the cloud app this GPU carries two workloads at the same time with utilization close to 100%.

## 6. Compatibility-mode scale-up: the TensorFusion expander creates a NodeClaim when GPU is short (KRP-A02, KRP-A03)

Purpose: verify the core loop of "AutoSelect + Karpenter NodePool". First fill the existing GPUs, then add one more Pod.

```bash
k create ns t6 && scripts t6
# fill every existing GPU (whole-GPU shared)
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

Expected:

1. `extra` is Pending first, with `GPUQuotaOrCapacityNotEnough` in its events.
2. Within ten-odd seconds **exactly 1** new NodeClaim is added, named `tensor-fusion-<10 characters>` (created by the expander, not Karpenter's `tf-gpu-*`), with the label `karpenter.sh/nodepool=tf-gpu` and the annotation `tensor-fusion.ai/expansion-source`, and without `karpenter.sh/do-not-disrupt`:
   ```bash
   k get nodeclaim -o json | jq '.items[] | {name: .metadata.name, pool: .metadata.labels["karpenter.sh/nodepool"], src: .metadata.annotations["tensor-fusion.ai/expansion-source"]}'
   k get --raw /api/v1/namespaces/tensor-fusion-sys/services/tensor-fusion-sys:8080/proxy/api/node-scaler | jq .
   ```
3. After the GPU on the new node becomes Running, `extra` is bound and Running within 30 seconds (before 2.17.24 this took a 5-minute wait).

One scale-up serving several Pods (KRP-A03): start three more Pods at 20%; they should land on the same new node and the same GPU, with no further NodeClaim added.

cloud app: node count +1, and `extra` appears on the new node.

Cleanup: `k delete ns t6`.

## 7. Scale-down: the node is reclaimed after the workloads are gone (KRP-A09)

Run this after any case that used a GPU; timing starts when the namespace is deleted:

```bash
date; watch -n 10 'kubectl get nodeclaim; kubectl get gpunode,gpu; kubectl get node -l e2e.tensor-fusion.ai/role=gpu'
```

Expected: when `consolidateAfter=2m` elapses the NodeClaim enters deletion, and the Node, GPUNode and GPU disappear with it; the ECS instance is released:

```bash
aliyun ecs DescribeInstances --RegionId us-east-1 --Tag.1.Key Project --Tag.1.Value tf-e2e-ack | jq -r '.Instances.Instance[] | [.InstanceId,.InstanceType,.Status] | @tsv'
```

Only the system node's instance is left. cloud app: the node and the GPU disappear from the environment.

## 8. Path B: `provisioningMode: Karpenter`, driven by the capacity watermark (KRP-B02, KRP-B10)

The automated case adds a pool with `provisioningMode: Karpenter` to the TensorFusionCluster, waits for it to launch a node, schedules a Pod onto it, then deletes the pool and checks that nothing is left behind. Doing this by hand means editing the TensorFusionCluster, so running the automated case directly is recommended:

```bash
make ack-test TIER=karpenter-finops FOCUS='KRP-B02|KRP-B10'
```

Observe by hand while it runs:

```bash
watch -n 5 'kubectl get gpupool; kubectl get gpunodeclaim; kubectl get nodeclaim -o wide; kubectl get gpunode,gpu'
```

Expected: pool `tensor-fusion-karp` appears → 1 GPUNodeClaim (instance type `ecs.gn6i-c8g1.2xlarge`, with the `tensor-fusion.ai/hourly-pricing` annotation) → a NodeClaim of the same name (owned by the GPUNodeClaim, with `karpenter.sh/do-not-disrupt`, requesting `nvidia.com/gpu: 1`) → node → GPUNodeClaim `Bound` → the Pod targeting that pool lands on this node; no second GPUNodeClaim appears within 30 seconds; after the pool is deleted, the GPUPool, GPUNodeClaim, NodeClaim, node, GPUNode and GPU all disappear.

This needs an operator with the Alibaba Cloud instance catalog (≥ a version that includes #820); earlier versions cannot select any instance type here.

## 9. FinOps: node count of fractional vs. whole-GPU workloads (FIN-01)

```bash
make ack-test TIER=karpenter-finops FOCUS='FIN-01'
cat results/<run-id>/FIN-01.finops.json | jq .
```

Expected: 3 fractional workloads at 20% use only 1 node, and 3 whole-GPU workloads use 3; the output gives the node-hours, the on-demand cost (unit prices from `pricing/alibaba-us-east-1.json`) and the savings ratio of the two options. The difference in node count should be visible for the same time range in the cost view of the cloud app.

## 10. Recovery: hypervisor restart, operator restart, node deleted (NLC-01, NLC-02, NLC-09)

```bash
k create ns t10 && scripts t10
tfpod t10 app soft 20 2Gi "sleep 1800"
k -n t10 wait pod app --for=condition=Ready --timeout=900s
node=$(k -n t10 get pod app -o jsonpath='{.spec.nodeName}'); k get gpu -o json | jq '.items[].status.available'

# hypervisor restart
k -n tensor-fusion-sys delete pod hypervisor-$node; date
watch -n 3 "kubectl -n tensor-fusion-sys get pod hypervisor-$node; kubectl get gpu; kubectl -n t10 get pod app"
```

Expected: the new hypervisor is Ready within about 20 seconds, during which the GPU briefly goes `Pending` and then returns to `Running`; `app` does not restart; `available` is the same before and after (52 TFLOPS / 13Gi).

```bash
# operator restart
k get --raw /api/v1/namespaces/tensor-fusion-sys/services/tensor-fusion-sys:8080/proxy/api/allocation | jq '.allocation | keys'
k -n tensor-fusion-sys rollout restart deploy/tensor-fusion-sys-controller && k -n tensor-fusion-sys rollout status deploy/tensor-fusion-sys-controller
k get --raw /api/v1/namespaces/tensor-fusion-sys/services/tensor-fusion-sys:8080/proxy/api/allocation | jq '.allocation | keys'
```

Expected: the two key sets are identical; after the restart, one more Pod at 60% still fits on the same GPU together with the existing 20%, and a second Pod at 60% does not land on this GPU.

Node deleted (the Pod is only re-created when it belongs to a Deployment):

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

Expected: the GPUNode and GPU of the deleted node are cleaned up within a few seconds; the Deployment's new Pod is Pending first and then Running on another GPU node (existing or newly launched).

cloud app: the environment stays online during the restarts (the agent briefly reconnects when the operator restarts); after the node is deleted the node list changes accordingly.

Cleanup: `k delete ns t10`.

## 11. Quota, Gang, scheduling by model (SCH-10, SCH-11, SCH-04)

Each of these three takes between ten-odd seconds and a few tens of seconds in the automated suite; run them directly and read the results:

```bash
make ack-test TIER=core FOCUS='SCH-10|SCH-11|SCH-04'
```

Expected: a Pod exceeding the `GPUResourceQuota` stays Pending and its events mention quota; the two Gang members become Running together; a Pod that names a model that does not exist stays Pending and does not trigger a scale-up.

## 12. Metrics and console (OBS-02, CLD-01)

After a workload has been running for 2 minutes:

```bash
sql "show tables"
for t in tf_gpu_usage tf_node_metrics tf_pool_metrics tf_worker_resources tf_worker_usage; do echo "$t: $(sql "select count(*) from $t" 2>/dev/null)"; done
sql "select node, instance_type, hourly_price, currency, price_source from tf_node_metrics order by greptime_timestamp desc limit 1"
```

Expected: the first four tables have data; `hourly_price` in `tf_node_metrics` is the on-demand price of the T4 instance type (1.0941, `price_source=list`). `tf_worker_usage` has no data in the current version, which is a known issue (see TEST-SUMMARY.md).

Confirm item by item in the cloud app: the environment is online; the node / GPU / workload lists match `kubectl`; the GPU utilization, VRAM, power and temperature curves have data; the pool's capacity and allocated amount are consistent; the cost figures are non-zero.

## 13. Features known not to work (do not treat them as environment problems)

| Feature | Symptom | Cause |
|---|---|---|
| remote mode (ISO-06) | The worker Pod is in `CrashLoopBackOff` with exit code 0, and the client reports `cuda_available: false` | The community remote worker reports `License is not set` at startup and exits; neither the chart nor the operator has any configuration for delivering a License |
| hard isolation (ISO-04, PERF-LLM-02) | The sidecar worker exits and the application process gets `Bus error` | Same as above |
| `disable-features: gpu-limiter` under soft (ISO-13) | A 30% limit is still limited to about 27% | The webhook only handles this annotation on the remote / hard client path |
| Images without libstdc++ (ISO-14) | The process in the container exits with 127 | The soft limiter is injected through `ld.so.preload` and depends on glibc and libstdc++ |

To see the remote / hard symptoms yourself: change the isolation mode in the `tfpod` of case 0 to `hard`, or add `tensor-fusion.ai/is-local-gpu: "false"` to the Pod, then run `kubectl logs <pod> -c tensorfusion-worker`.

## 14. Wrap-up

```bash
k get ns | grep -E '^t[0-9]+' | awk '{print $1}' | xargs -r kubectl delete ns
make ack-sweep        # list all cloud resources of this environment that currently exist
make ack-down         # tear down and check again; the last line should be all clean
```

While the environment is alive, `ack/guard.sh` checks every 2 minutes: ECS instances with no corresponding NodeClaim are terminated, and when there are more than 4 GPU instances the limits of all NodePools are set to 0. The log is in `ack/.work/guard.log`.
