# ACK (Alibaba Cloud) real-GPU end-to-end test summary

2026-10-12 to 13, Alibaba Cloud us-east-1, ACK managed Kubernetes 1.36.2, Karpenter Alibaba Cloud provider 0.2.5 (with the patch from this repository), GPU nodes `ecs.gn6i-c4g1.xlarge` / `c8g1.2xlarge` (one Tesla T4 each), TensorFusion chart from the repository checkout. This is the first time the GPU tier marked "not executed" in `e2e/README.md` has run on real GPUs.

The environment is described in [`README.md`](README.md), the manual reproduction steps are in [`MANUAL-TEST-CASES.md`](MANUAL-TEST-CASES.md), and the charted version of the performance numbers is in `../perf-history/index.html`.

## 1. Conclusions

- **Compatibility mode (path A: `AutoSelect` + Karpenter NodePool) works end to end on Alibaba Cloud.** Cold start, the expander loop, one scale-up serving several Pods, N whole-GPU Pods scaling up exactly N nodes, scale-down, re-creation after a node is deleted, and the FinOps comparison all pass. The precondition is one patch to the provider (see 3.1); without it a single Pending Pod launches 4 GPU nodes.
- **Provisioning mode (path B: `provisioningMode: Karpenter`) works after the fix.** Before the fix the operator's instance catalog had no Alibaba Cloud instance types and the operator could not read `ECSNodeClass`, so the pool never created a GPUNodeClaim (#820, released in v2.17.30).
- **The core behavior of soft isolation holds**: several Pods on one GPU, the VRAM cap, the compute limit, neighbor isolation, and changing the limit at runtime (after fix #822). The compute limit controls the device busy-time fraction, and it does so precisely (measured 53% / 26% for targets of 50% / 25%); throughput is not linear in busy time, see 4.2.
- **Hard isolation and remote mode do not work on the open-source stack**: the community worker reports `License is not set` at startup and exits, and neither the chart nor the operator has any configuration for delivering a License. This is not an environment problem, and it is not something this round could fix.
- Two more known issues were confirmed on real GPUs: `disable-features: gpu-limiter` has no effect under soft isolation; `tf_worker_usage` had no data in the suite run (OBS-02). On the 2026-10-13 bring-up a Deployment-backed soft workload at a 50% limit did produce rows (8 samples in 8 minutes, mean 48.7%, device at 53%), so OBS-02 needs to be re-run to find out which workloads are missed; the cause has not been identified.

## 2. Case results

61 cases in total: **53 pass, 6 known issues (asserted against the correct behavior, which the product does not currently meet), 2 fail**. Case execution time adds up to 225 minutes (re-runs excluded). The durations include the time spent waiting for GPU nodes: the first case that needs a GPU waits about 5 minutes for the cold start.

"Version" is the operator version at the last execution of that case. `dev-ack-20261012-1` = v2.17.28 + #820; `dev-ack-20261012-2` adds #822; the code of both is included in v2.17.30, and the hypervisor did not change between these three versions.

| ID | Case | Status | Duration (s) | Version |
|---|---|---|---|---|
| CLD-01 | cluster-agent sidecar is running (cluster registered with the console) | Pass | 1 | 2.17.28 |
| FIN-01 | FinOps: node-hours and cost of N fractional workloads vs. the whole-GPU baseline | Pass | 1523 | dev-ack-20261012-2 |
| INS-01 | Control plane is ready after Helm install | Pass | 6 | 2.17.28 |
| INS-02 | Default cluster and Pool are reconciled (zero nodes) | Pass | 2 | 2.17.28 |
| INS-03 | GPU nodes and GPUs are discovered automatically (includes KRP-A01, managing Karpenter nodes) | Pass | 14 | dev-ack-20261012-2 |
| INS-04 | Non-GPU nodes are not managed | Pass | 2 | 2.17.28 |
| ISO-01 | soft fractional sharing: several Pods run on one GPU | Pass | 40 | dev-ack-20261012-1 |
| ISO-02 | soft compute limit, measured: throughput proportional to the share, no stalls | Pass | 1032 | dev-ack-20261012-2 |
| ISO-03 | VRAM cap, measured: the container sees only its quota, allocation beyond the limit fails, neighbors are unaffected | Pass | 50 | dev-ack-20261012-2 |
| ISO-04 | hard compute limit, measured (sidecar worker) | Known issue | 99 | dev-ack-20261012-2 |
| ISO-05 | shared whole GPU: once it is taken exclusively, no other Pod lands on that GPU | Pass | 70 | dev-ack-20261012-1 |
| ISO-06 | remote mode: the application Pod is on a CPU node and uses a remote GPU through a TensorFusionConnection | Known issue | 21 | 2.17.30 |
| ISO-13 | disable-features: gpu-limiter turns throttling off and the workload is no longer limited | Known issue | 126 | dev-ack-20261012-2 |
| ISO-14 | What soft mode requires of the application image: in an image without libstdc++ the process cannot start | Known issue | 20 | dev-ack-20261012-1 |
| ISO-15 | Neighbor isolation: a noisy Pod does not push a quiet Pod below its share | Pass | 648 | 2.17.30 |
| ISO-16 | Changing the limit at runtime takes effect (cron scaling rule, i.e. ASC-02) | Pass | 571 | dev-ack-20261012-2 |
| KRP-00 | Karpenter is ready: NodePool / EC2NodeClass Ready, nodeOverlay feature gate on | Pass | 1 | 2.17.28 |
| KRP-01 | NodeOverlay and the operator's NodeClaim RBAC are in place | Pass | 2 | 2.17.28 |
| KRP-A02 | Not enough GPU → the expander creates a NodeClaim → new node → Pod runs (the core loop) | Pass | 338 | dev-ack-20261012-2 |
| KRP-A03 | One scale-up serves several Pending Pods (not one machine per Pod) | Pass | 343 | dev-ack-20261012-2 |
| KRP-A08 | Cold start from 0 GPU nodes: Pending TF Pod → Karpenter NodeClaim → node → GPU registered → Pod runs | Pass | 570 | dev-ack-20261012-2 |
| KRP-A09 | The node is reclaimed after the workloads are gone (should-delete mark + Karpenter consolidation) | Pass | 141 | dev-ack-20261012-2 |
| KRP-A15 | The chart's default NodeOverlay makes Karpenter launch non-TF nodes for TF Pods | Pass | 113 | dev-ack-20261012-2 |
| KRP-A16 | Once the GPU on the scaled-up node is ready, the Pending Pod should be bound promptly (without waiting for the scheduler's 5-minute fallback retry) | Pass | 1 | dev-ack-20261012-2 |
| KRP-A17 | N whole-GPU Pods Pending at the same time: exactly N nodes are added (no more, no fewer) | Pass | 644 | dev-ack-20261012-2 |
| KRP-B02 | Path B: capacity watermark → GPUNodeClaim → NodeClaim of the same name → node → Bound (includes the B01 connection check and the B09 cleanup on Pool deletion) | Pass | 298 | 2.17.30 |
| KRP-B10 | Path B: after the Pool is deleted, the GPUNodeClaim / NodeClaim / node / GPUNode / GPU are all reclaimed (runs after KRP-B02) | Pass | 2 | 2.17.30 |
| KRP-C01 | Karpenter launches and reclaims a non-GPU node (general-purpose NodePool) | Pass | 186 | 2.17.28 |
| NEG-02 | Specifying a pool that does not exist is rejected at admission | Pass | 12 | 2.17.28 |
| NEG-03 | Local hard without sidecar-worker: actually accepted, and the sidecar is added automatically | Pass | 13 | 2.17.28 |
| NEG-04 | partitioned without partition-id is rejected | Pass | 18 | 2.17.28 |
| NEG-05 | A whole-GPU request without gpu-model is rejected | Pass | 12 | 2.17.28 |
| NEG-06 | gang-min-members=1 is rejected | Pass | 12 | 2.17.28 |
| NEG-11 | tflops and compute-percent specified together: actually accepted, tflops takes effect | Pass | 18 | 2.17.28 |
| NEG-12 | soft mode without any resource request is rejected | Pass | 17 | 2.17.28 |
| NEG-13 | Where the error is reported for invalid isolation / qos values | Pass | 12 | 2.17.28 |
| NLC-01 | Hypervisor restart: the application is not interrupted, the GPU goes Pending and then recovers, no accounting is lost | Pass | 50 | dev-ack-20261012-1 |
| NLC-02 | Operator restart: allocation state is rebuilt, no double allocation | Pass | 111 | dev-ack-20261012-1 |
| NLC-09 | State converges after a node is deleted, and the Deployment's Pod is re-created on a new node | Pass | 40 | dev-ack-20261012-2 |
| OBS-01 | The operator metrics endpoint can be scraped | Pass | 1 | 2.17.28 |
| OBS-02 | GPU / worker / node / pool metrics are written to GreptimeDB within the time limit | Known issue | 388 | dev-ack-20261012-1 |
| OBS-06 | GreptimeDB can be queried over HTTP SQL | Pass | 1 | 2.17.28 |
| PERF-CO-01 | Fractional sharing + co-scheduling: two vLLM instances at 50% each, concurrent on the same GPU; throughput and tail latency of each, and combined throughput | Pass | 558 | 2.17.30 |
| PERF-LLM-01 | vLLM streaming inference vs. compute share (soft): throughput proportional, no cliff in TTFT / TPOT tail latency | Pass | 1265 | 2.17.30 |
| PERF-LLM-02 | vLLM streaming inference vs. compute share (hard) | Known issue | 1772 | 2.17.30 |
| PERF-MLP-01 | MLPerf Inference style (LoadGen Offline / Server, shortened duration, not a compliant submission) vs. share | **Fail** | 543 | 2.17.30 |
| PERF-ORT-01 | ONNX Runtime (CUDA EP): QPS and latency percentiles of a CNN / encoder / speech encoder vs. share | Pass | 992 | 2.17.30 |
| PERF-TB-01 | Representative TorchBench subset (conv / transformer / small-kernel-heavy) vs. share | **Fail** | 465 | 2.17.30 |
| SCH-01 | Default bin-packing: small workloads are packed onto one node | Pass | 45 | dev-ack-20261012-2 |
| SCH-04 | Scheduling by GPU model: a matching model is schedulable, a model that does not exist stays Pending | Pass | 85 | dev-ack-20261012-1 |
| SCH-10 | The namespace quota blocks requests over the limit | Pass | 40 | dev-ack-20261012-1 |
| SCH-11 | Gang scheduling (positive): members run together once all are present; annotations and status are visible | Pass | 27 | dev-ack-20261012-1 |
| SCH-12 | The allocation query and expander status APIs are available | Pass | 1 | 2.17.28 |
| SCH-14 | HostPort is assigned automatically | Pass | 18 | 2.17.28 |
| WHK-01 | Adding only the label + annotation is enough for the Pod to be mutated | Pass | 18 | 2.17.28 |
| WHK-02 | Referencing a WorkloadProfile; annotations can override it | Pass | 13 | 2.17.28 |
| WHK-03 | In a multi-container Pod only the specified container is injected | Pass | 13 | 2.17.28 |
| WHK-05 | enabled-replicas onboards only some of the replicas | Pass | 18 | 2.17.28 |
| WHK-06 | A Pod that is not enabled is not modified at all | Pass | 13 | 2.17.28 |
| WHK-07 | After the workload is deleted, the TensorFusionWorkload and the GPU resources are returned | Pass | 24 | dev-ack-20261012-1 |
| WHK-08 | server-side dry-run has no side effects | Pass | 34 | 2.17.28 |

## 3. Problems found and how they were handled

### 3.1 Fixed

| # | Problem | Impact | Handling |
|---|---|---|---|
| 1 | The operator's instance catalog only covers AWS and Azure, so path B cannot select any instance type on Alibaba Cloud | A pool with `provisioningMode: Karpenter` never creates a GPUNodeClaim | #820: adds an Alibaba Cloud ECS GPU instance table (gn6i / T4, gn7i / A10, us-east-1 on-demand list prices) and selects the table by the API group of the NodeClass the pool references. v2.17.30 |
| 2 | The ClusterRole allows `karpenter.k8s.alibabacloud`, while the group of the official provider is `karpenter.alibabacloud.com` | The operator cannot read `ECSNodeClass` | #820: both groups are allowed (marker, `config/rbac`, chart) |
| 3 | The autoscaler waits a 30-minute initial delay before it starts evaluating any workload, which also blocks cron scaling rules | A cron rule whose window falls within the first half hour of a workload has no effect; short-lived workloads are never scaled | #822: a workload with a usable cron rule enters evaluation immediately. v2.17.30. Measured: throughput drops to 26% inside the window (target 30%) and recovers after the window |
| 4 | The NodeClaim capacity returned by the Alibaba Cloud provider does not include the resources declared by the NodeOverlay | One Pending TF Pod launches one more node every 10 seconds, up to the NodePool limit (measured: 1 Pod → 4 T4 nodes) | `patches/karpenter-alibaba-0.2.5-overlay-capacity.patch`, built and imported by `build-karpenter.sh`. With the patch, 1 Pod → 1 node. **An issue / PR should be filed upstream** |
| 5 | The provider's `--feature-gates` flag does not accept core's `NodeOverlay` and panics at startup | Karpenter does not come up | Set it only through the `FEATURE_GATES` environment variable. Upstream problem |
| 6 | Problems in the cases themselves: `NLC-01` took its baseline before the GPU CR had been accounted; `ISO-06` assumed the worker lands on the node of the first GPU; `ISO-15` waited for "all resources returned" while the neighbor was still there; the soft compute criteria used a symmetric throughput band; a 15-second LLM warm-up was not enough; native vLLM taking the whole GPU's VRAM crashes by itself on the T4 | False failures | Corrected one by one, see 4.2 and the commit history |

### 3.2 Not fixed, needs follow-up

| # | Problem | Evidence | Recommendation |
|---|---|---|---|
| 1 | **Hard isolation and remote mode do not work** | `tensor-fusion-worker:v2.25.0` (community) log: `VerifyLicense: License is not set` → `quit server`; under hard the application process then gets `Bus error`, under remote the worker goes into `CrashLoopBackOff` (exit code 0) | Product decision: either the chart / ProviderConfig provides a way to deliver the License and it is documented, or the documentation states clearly that these two modes are not part of the open-source edition. `ISO-04`, `ISO-06` and `PERF-LLM-02` are marked known-fail |
| 2 | **soft is slightly too tight at low shares when measured by throughput** | T4, matmul: at a 25% limit the device utilization is 26% (accurate) and throughput is 21% of full load (about 15% below the share); at 50% utilization is 53% and throughput 79% | The control loop only looks at the busy-time fraction and cannot see how "computation completed per millisecond" changes with the duty cycle. Candidate change (not implemented): in the calibrated branch, give the token rate a floor of `limit × unthrottled token rate`, which can only loosen and never tighten. First export each worker's rate, gain and utilization as metrics, then decide |
| 3 | `disable-features: gpu-limiter` has no effect under soft | 30% limit with this annotation: throughput is still 27.7% of the unlimited value | The webhook only turns it into an environment variable on the remote / hard client path (`compose_client.go`); the soft branch returns before that. `ISO-13` is marked known-fail |
| 4 | `tf_worker_usage` has no data | The table is still empty after the workload has run for 6 minutes; the other four tables have data | Usage alerts, usage-based scaling and the worker-level charts in the console, which depend on it, have no data source. `OBS-02` remains known-fail |
| 5 | **In compatibility mode, NodeClaims created by the expander are not constrained by the NodePool `limits`** | Measured: with the limit of `tf-gpu` set to `nvidia.com/gpu: 1` and 1 GPU node already present, two more whole-GPU Pods were started; the `tensor-fusion-*` NodeClaims created by the expander were `Launched` as usual and the NodePool status became `limits=1, used=2`. Karpenter only checks the limit in its own scheduling | The NodePool cannot be relied on as the cost cap: document that it has to come from the GPUPool capacity configuration or from quotas; or have the expander check the NodePool `limits` before creating a NodeClaim. This environment additionally has `guard.sh` as a backstop |
| 6 | Device telemetry is one row every 60 seconds | `tf_gpu_usage` has one instantaneous sample per GPU per minute | A measurement window shorter than two minutes has only 0–1 samples; the utilization curve in the console also has one-minute resolution |
| 7 | Two upstream problems of vLLM 0.11.0 on the T4 | The architecture check of the FlashInfer sampler throws on GPUs with compute capability 7.x; native, taking 85% of the whole GPU's VRAM with 8 concurrent requests, hits `CUDA error: an illegal memory access` | Unrelated to TensorFusion. The cases turn off the FlashInfer sampler, and the native baseline uses the same VRAM budget as the tenants |
| 9 | Tail latency is amplified under a 50% limit for convolution inference that takes tens of milliseconds per call | TorchBench-mini resnet50: p99 goes from 43.8 ms to 208.7 ms, 2.4 times the proportional expectation (87.6 ms); throughput is 43.8%, which is normal | When the token cost of one call spans several 10 ms refill periods, the waits add up. Read together with item 2: the internal quantities of the control loop need to be observable first. `PERF-TB-01` fails because of this |
| 10 | Batch-style workloads may be limited more tightly than the share | MLPerf-style resnet50 Offline: soft 50% is 38.7% of soft 100% (lower bound 42.5%). In the same run native is only 46% of soft 100%; the short test is very noisy | Re-test with the long window of the full profile before drawing a conclusion. `PERF-MLP-01` fails because of this |
| 8 | An application image without libstdc++ cannot start under soft | busybox: `error while loading shared libraries` | Known issue (G-03), `ISO-14` |

## 4. Performance

### 4.1 Karpenter and node timeline

| Measurement | Value | Source |
|---|---|---|
| CPU node: Pending → NodeClaim | 7.0 s | KRP-C01 |
| CPU node: Pending → node Ready | 63.7 s | KRP-C01 |
| CPU node: Pending → Pod Running | 74.5 s | KRP-C01 |
| CPU node: emptied → node gone (`consolidateAfter=30s`) | 51.0 s | KRP-C01 |
| **GPU node cold start: Pending → NodeClaim** | 6.5 s | KRP-A08 |
| **GPU node cold start: Pending → GPU registered as Running** | 309.7 s | KRP-A08 |
| **GPU node cold start: Pending → Pod Running** | 315.5 s | KRP-A08 |
| Compatibility-mode scale-up: Pending → NodeClaim created by the expander | 13.8 s | KRP-A02 |
| Compatibility-mode scale-up: Pending → Pod Running | 313.5 s | KRP-A02 |
| GPU on the new node Running → Pending Pod bound | 8.0 s | KRP-A02 / A16 (330 s on 2.17.23) |
| 3 Pods at 20% in one scale-up, all Running | 318.1 s, 1 NodeClaim, 1 node, same GPU | KRP-A03 |
| 3 whole-GPU Pods Pending at the same time, all Running | 362.0 s, exactly 3 NodeClaims, 3 nodes | KRP-A17 |
| GPU node: emptied → node gone (`consolidateAfter=2m`) | 128.3 s | KRP-A09 |
| Node deleted → GPUNode / GPU CRs cleaned up | 0.8 s | NLC-09 |
| Node deleted → the Deployment's Pod Running on another node | 7.5 s (the other node already existed) | NLC-09 |
| Path B: pool created → GPUNodeClaim | 23.1 s | KRP-B02 |
| Path B: pool created → Bound, GPU Running | 239.4 s / 240.2 s | KRP-B02 |
| Path B: Pod targeting that pool created → Running | 6.1 s | KRP-B02 |
| Path B: pool deleted → all objects gone | 8.3 s | KRP-B10 |
| hypervisor restart recovery | 18.1 s | NLC-01 |
| operator restart recovery | 23.0 s | NLC-02 |

A GPU node takes about 5 minutes 10 seconds from Pending to usable. Of that, NodeClaim to instance creation is only a few seconds; the rest is instance boot, the ACK join script installing the NVIDIA driver, and the hypervisor being pulled and discovering the device. The `c8g1.2xlarge` of path B is about 70 seconds faster.

FinOps (FIN-01, each scenario held at steady state for 3 minutes, unit prices from `../pricing/alibaba-us-east-1.json`):

| Option | Nodes | Node-hours | On-demand cost | Cold start to all Running |
|---|---|---|---|---|
| 3 fractional workloads (20% compute, 2Gi each) | 1 | 0.133 | $0.146 | 305 s |
| 3 whole-GPU workloads (baseline) | 3 | 0.483 | $0.529 | 602 s |

Savings of 72.4%. At steady state it is 1 node against 3 (67%); this run saved somewhat more because the last two nodes of the whole-GPU option were launched by the expander only after the first one was ready.

### 4.2 soft compute limit (ISO-02, ISO-15, ISO-16, matmul FP16 2048×2048)

| Limit | Throughput | Relative to 100% | Device utilization | Actual compute reading | Power | p99 per-iteration latency |
|---|---|---|---|---|---|---|
| 100% | 1320 it/s | 1.00 | 96% | 62.4 TFLOPS | 70 W | 0.82 ms |
| 50% | 1047 it/s | 0.79 | 53% | 34.5 TFLOPS | 77 W | 6.36 ms |
| 25% | 281 it/s | 0.21 | 26% | 16.9 TFLOPS | 35 W | 10.35 ms |

- Device utilization is the quantity the limiter controls, and all three settings are within 3 points of the target.
- Throughput above 50% at the 50% setting is not slow convergence: run alone for 5 minutes, the 30-second averages are 975, 1019, 1005, 1005... and stay flat throughout. The reason is that a T4 under sustained full load is limited by power / temperature, and under an intermittent load each busy millisecond completes more computation (median per-iteration time 0.52 ms vs. 0.77 ms). The 75% setting reaches 93–95% of full load.
- At the 25% setting throughput is about 15% below the share, see item 2 of 3.2.
- The criteria were therefore changed to: throughput must not be lower than 85% of the share; throughput is not higher than "share + 0.35" (capped at 0.98); when the window has more than two device samples, utilization is not higher than "share + 10 points"; tail latency is allowed the proportional scaling plus a margin of one 12 ms refill period.

Neighbor isolation (ISO-15): a Pod with a 40% share gets 496.7 it/s alone and 503.6 it/s (1.014×) next to a neighbor saturating 60%, with no stalls ≥0.4 s.

Changing the limit at runtime (ISO-16, a cron rule lowers the limit from 100% to 30% for 4 minutes): 1271 it/s before the window, 336 it/s inside the window (26.4%), 1276 it/s after the window.

VRAM cap (ISO-03): requesting 6Gi under a 4Gi limit fails after 3840 MiB has been obtained; the total VRAM seen inside the container is 4096 MiB.

### 4.3 LLM streaming inference (vLLM 0.11.0, Qwen2.5-0.5B-Instruct, FP16)

Each tenant has a 6Gi VRAM cap; 8 concurrent streaming requests saturate it in a closed loop, with 64 output tokens per request; 60 seconds are measured after a 90-second warm-up. The native baseline uses the same VRAM budget as the tenants. operator / hypervisor 2.17.30.

| Configuration | tokens/s | Relative to native | TPM | TTFT p50 / p95 | TPOT p50 / p99 | ITL p99 / max | Stalls ≥0.4s | Device utilization | Power |
|---|---|---|---|---|---|---|---|---|---|
| Native (without TensorFusion) | 361.8 | — | 21706 | 79 / 232 ms | 19.9 / 24.6 ms | 45 / 191 ms | 0 | 82% | 56 W |
| soft 100% | 360.5 | −0.4% | 21632 | 92 / 300 ms | 19.8 / 24.1 ms | 49 / 190 ms | 0 | 82% | 56 W |
| soft 50% | 265.8 | −26.5% | 15950 | 180 / 231 ms | 27.5 / 30.2 ms | 93 / 192 ms | 0 | 60% | 47 W |
| soft 50%, fixed arrival rate (30% of native capacity) | 105.9 | all completed on time | 6354 | 94 / 106 ms | 21.2 / 22.7 ms | 65 / 71 ms | 0 | 46% | 47 W |
| soft 25% | 100.9 | −72.1% | 6054 | 528 / 736 ms | 64.0 / 85.2 ms | 179 / 450 ms | 8 | 39% | 30 W |

- **Without a limit, isolation has no measurable cost**: soft 100% and native differ by 0.4% in throughput, and TPOT is the same.
- A 50% share gets 73.7% of the throughput and 25% gets 28.0%: neither is below the share; latency gets slower with the share but there is no cliff, and all criteria pass.
- When the load is below the capacity of the share (the fixed-arrival-rate row), the limiter does not intervene: TPOT and TTFT are comparable to the unlimited case.
- The 25% setting shows 8 inter-token gaps of 0.4–0.45 seconds (1.3 per thousand tokens; the criterion allows up to 5), none close to 1 second.
- Device utilization has only 1 sample per measurement point (telemetry is one row per minute) and is for reference only.

### 4.4 Fractional sharing + co-scheduling (PERF-CO-01)

Two vLLM instances each get 50% compute and 6Gi VRAM, land on the same T4, and are saturated at the same time.

| Configuration | tokens/s | TTFT p50 / p95 | TPOT p50 / p99 | ITL p99 / max | Stalls ≥0.4s | Device utilization | VRAM used |
|---|---|---|---|---|---|---|---|
| soft 100%, whole GPU alone (reference) | 354.5 | 78 / 144 ms | 20.7 / 25.0 ms | 42 / 193 ms | 0 | 81% | 5.2 GiB |
| Tenant A (50%) | 192.2 | 368 / 429 ms | 35.3 / 40.7 ms | 87 / 255 ms | 0 | 96% (whole GPU) | 9.9 GiB (whole GPU) |
| Tenant B (50%) | 215.3 | 214 / 365 ms | 33.6 / 38.4 ms | 85 / 238 ms | 0 | same as above | same as above |
| **Both tenants combined** | **407.5** | | | | | | |

- Combined throughput is **115%** of a single tenant with the whole GPU to itself: one tenant cannot fill this GPU (81% utilization alone), and two tenants fill it to 96%.
- The ratio of the two tenants' throughput is 0.89, and the slower one gets 47% of the combined total.
- Each tenant's TPOT is about 1.6–1.7 times the value when alone, TTFT p95 is within 0.45 seconds, and there are no stalls.

### 4.5 Other workloads (quick profile, T4)

| Workload · model | Native | soft 100% | soft 50% | soft 50% / soft 100% | Notes |
|---|---|---|---|---|---|
| ONNX Runtime · resnet50 | 365.2 samples/s | 364.4 (−0.2%) | 206.6 | 56.7% | p99 latency 45.0 → 45.1 → 81.0 ms |
| ONNX Runtime · minilm | 968.9 samples/s | 965.7 (−0.3%) | 601.2 | 62.3% | p99 16.9 → 17.0 → 30.6 ms |
| TorchBench-mini · resnet50 (eval) | 23.9 it/s | 23.7 (−0.8%) | 10.4 | 43.8% | **p99 43.3 → 43.8 → 208.7 ms**, 2.4 times the proportional expectation; criterion not met |
| TorchBench-mini · bert_base (eval) | 21.0 it/s | 20.8 (−0.6%) | 14.6 | 70.2% | p99 49.0 → 48.9 → 72.8 ms |
| TorchBench-mini · tiny_mlp (eval) | 1068.8 it/s | 1071.6 (+0.3%) | 1038.6 | 96.9% | This workload by itself keeps the GPU only about 20% busy, so a 50% limit is not a constraint; because the baseline has no device samples, the criterion still judged it as "limit ineffective" and not met, which is a limitation of the judgment |
| MLPerf style · resnet50 Offline | 170.2 samples/s | 371.2 | 143.7 | 38.7% | Below the 42.5% lower bound; criterion not met. In the same run native is half of soft 100%, which shows that this seconds-long Offline short test is very noisy; this row can only be taken as a lead |

- The overhead without a limit is within 1% on all workloads.
- Throughput under a 50% share ranges from 39% to 97%, depending on how much the workload does per busy millisecond and on whether it could fill the GPU in the first place. This is exactly why a soft limit cannot be judged by "throughput = share".
- Two leads that need follow-up: tail latency is clearly amplified under a 50% limit for convolution inference that takes tens of milliseconds per call (TorchBench resnet50); batch-style workloads may be limited more tightly than the share (MLPerf Offline, to be re-tested with the long window of the full profile).


## 5. Environment bring-up / teardown time and cost

The data comes from `../results/ack-ledger.tsv` and the logs of the scripts.

| Phase | Duration | Notes |
|---|---|---|
| VPC + 2 vSwitches | about 20 s | |
| ACK cluster creation to `running` | 176 s | Managed Basic edition, Kubernetes 1.36.2 |
| system node pool creation to node Ready | about 58 s | Of which 49 s is waiting for the node to become Ready |
| Karpenter (RAM role, chart, ECSNodeClass, NodePool Ready) | about 25–35 s | Excludes building and importing the patched image; for the import see the total row |
| TensorFusion chart (`helm --wait`) | 82–95 s | Range over several installs / upgrades |
| **Full bring-up total (`run.sh up`)** | **672 s** | Measured on the scheduled bring-up at 2026-10-13 08:00 (ledger `up-total`), which succeeded on the first attempt. Breakdown: VPC + vSwitches 24 s, cluster 219 s, system node pool 97 s (90 s waiting for node Ready), importing the patched image onto the system node 214 s, Karpenter 27 s, TensorFusion chart 98 s |
| Karpenter launching one CPU node to Pod Running | 74.5 s | Section 4.1 |
| **Karpenter launching one GPU node to Pod Running** | **315.5 s** | Section 4.1; NodeClaim 6.5 s, the rest is instance boot, driver installation and GPU discovery |
| **Full teardown (`run.sh down`)** | **463 s** | Of which deleting the cluster takes 431 s, and the RAM role, vSwitches and VPC about 30 s together; the sweep after teardown reports everything clean, and it was cross-checked with separate describe calls: 0 instances, 0 disks, 0 EIPs, 0 NAT gateways, 0 SLBs in the region |

The environment was alive for about 7 hours 40 minutes (2026-10-12 12:34Z to 20:14Z). This is a cost estimate, not a bill:

| Item | Estimate |
|---|---|
| GPU nodes | about 9 node-hours × $1.09 ≈ $10 (accumulated from the instance count in the cost guard log every 2 minutes; includes the 3 extra nodes launched for about 15 minutes in the first round because of the provider defect) |
| system node (`ecs.g7.xlarge`) | 7.5 h × $0.15 ≈ $1.1 |
| NAT gateway, SLB, EIP, traffic | about $0.5–1 |
| Total | **about $12** |

Case execution time by tier (last complete execution): smoke about 9 minutes, core about 23 minutes, isolation about 45 minutes, karpenter-finops about 68 minutes (path B another 5 minutes), perf-quick about 80 minutes (30 minutes of which is the hard case waiting for a Pod that had already failed; the wait logic has been fixed). The duration of each case is in the table in section 2.

## 6. Merged changes

| PR | Content |
|---|---|
| #820 | fix: Karpenter provisioning mode on Alibaba Cloud (instance catalog, NodeClass RBAC) |
| #821 | test(e2e): ACK tier (bring-up / teardown scripts, provider patch, cost guard, the suite recognizes `ECSNodeClass`) and the first round of case corrections |
| #822 | fix: cron scaling rules are evaluated from the creation of a workload |
| #823 | test(e2e): criteria and case corrections on real GPUs (soft compute criteria, LLM warm-up, VRAM budget of the native baseline, ONNX Runtime load path, known-issue marking), the co-scheduling case `PERF-CO-01`, machine metrics, performance history and viewer, ACK documentation |
| #824 | fix: an empty pool is Running unless the operator provisions it; cluster status totals are refreshed while a pool is not ready. Found on the 2026-10-13 bring-up: with zero GPU nodes the pool and the cluster stayed in `Updating` |

v2.17.29 (with #820) and v2.17.30 (adding #822) were produced automatically by the release process on main.
