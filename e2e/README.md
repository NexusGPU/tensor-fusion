# TensorFusion end-to-end validation in the cloud (AWS EKS + Karpenter)

This directory turns "validating TensorFusion in the cloud" into something that can be run repeatedly:

| Directory | Contents |
|---|---|
| `terraform/` | Disposable EKS + Karpenter environment with an AWS-side timeout dead-man switch, cost queries, and a leftover sweep. See `terraform/README.md` |
| `suite/` | End-to-end test cases in Go + Ginkgo (build tag `e2e`), driven by a kubeconfig, runnable against any cluster (EKS, bare metal) |
| `suite/workloads/` | Probe and load-test scripts that run inside Pods (Python) |
| `mock/` | Build script for the mock accelerator used in rehearsal mode |
| `ack/` | Disposable environment on Alibaba Cloud ACK + Karpenter's Alibaba Cloud provider (real T4). See `ack/README.md` for instructions, `ack/TEST-SUMMARY.md` for conclusions, and `ack/MANUAL-TEST-CASES.md` for manual steps |
| `perf-history/` | Archived performance results per version (`runs/<version>/<env>.json`) and the local viewer `index.html`; written by `tools/perf_history.py` |
| `kwok/` | A tier that needs no cloud account: kind + KWOK fake nodes + Karpenter KWOK provider + chart + fake hypervisor (`fakehv/`). See 6.9 |
| `tools/report.py` | Generates a comparison report from the results of one or more runs (Markdown + CSV + JSON for charts) |
| `pricing/` | Instance price table (from the AWS Price List API), shared by `make cost` and the FinOps cases |
| `GO-LIVE.md` | Real-GPU execution steps for once the GPU quota is in place |

This document maps one-to-one to the actual implementation in `suite/`: **every ID in the tables is a test case that really exists**. Capabilities that are not implemented are listed in section 8 with the reason.

---

## 1. Current state (2026-10-13)

- **The first execution on real GPUs was completed on Alibaba Cloud ACK (T4)**: smoke, core, isolation, karpenter-finops (path A and path B), and perf-quick have all been run. Per-case results, the problems found, and the fixes are in `ack/TEST-SUMMARY.md`. The "Status" column in the tables of this document still records the AWS environment.
- AWS: the GPU quota is 0 (both on-demand and spot G/VT vCPU are 0; the quota increase request is in AWS manual review). **No test case has been run on a real GPU on AWS.**
- Already run on real AWS: environment up/down (3 rounds), the dead-man switch (2 times), the `smoke-nogpu` tier, the `rehearsal` tier (mock GPU), and the `dryrun` tier. Results are in section 9.
- On AWS, the `isolation` and `perf` tiers have only reached dry-run: the manifests passed server-side dry-run, the images and models resolve, and the load-test scripts and judging logic ran end to end on CPU. Their assertion thresholds are design values. **Before the first real execution, do not treat any conclusion of these two tiers as verified.**

## 2. Execution modes: nobody should mistake a rehearsal for hardware validation

The result of every test case carries a `mode` field, and the summaries outside JUnit (`summary.json` / `summary.md` / `report.md`) mark it per case:

| mode | Meaning | Can prove | Cannot prove |
|---|---|---|---|
| `real-gpu` | The Pod runs on a real NVIDIA GPU | Everything | — |
| `mock-gpu` | Rehearsal: CPU nodes + the example accelerator shipped in this repository (mock device, 16 GiB / 312 TFLOPS) | Webhook, scheduling and bin packing, quota, allocation accounting, hypervisor / operator restart recovery, the full Karpenter scale-up and scale-down chain, cleanup | Anything related to CUDA: whether VRAM / compute limits take effect, whether the GPU is usable inside the container, performance |
| `fake-gpu` | KWOK fake nodes (no kubelet, Pods do not actually start) + a fake hypervisor that writes the GPU CRs and `index_*` capacity on their behalf | Control plane only: how many NodeClaims the expander / Karpenter creates, from which NodePool, how long the scheduler takes to bind after the GPU is ready, reclamation and cleanup, the object chain of path B | Everything on the node: the hypervisor itself, device discovery, limiter, device plugin, kubelet, image pulls, the absolute value of any duration (node startup and GPU registration delays are parameters, not measurements) |
| `no-gpu` | Test cases that do not need a GPU | Control plane, admission, API | — |
| `dry-run` | Test cases that need a GPU, taken as far as "only the card is missing" | The manifests are accepted by admission, the images and models exist, the scripts and judging logic are executable, Karpenter created a NodeClaim for the GPU NodePool and EC2 rejected it only because of quota | The feature itself |

Test cases known to always fail on the currently released images are marked `KnownIssue` and shown as `known-fail` in the summaries (they are still failures in JUnit, on purpose).

## 3. Why Go + Ginkgo + kubeconfig

- The repository's existing `test/e2e` is already Ginkgo, and `vendor/` contains Ginkgo, Gomega, client-go, and controller-runtime, so the suite needs neither a `go.mod` change nor `make vendor`.
- Everything uses `unstructured` objects and **does not import the operator's Go types**: the same test binary can judge any released version and bare-metal clusters. Annotation names and label names are written as literals in the tests, so a rename in the code makes the tests fail (they are the public contract promised by the documentation).
- Build tag `e2e`: `go build ./...`, `go test ./...`, and `make lint` do not see this package.
- It shells out to `kubectl` in only two places: `exec` (the repository does not vendor remotecommand). HTTP-style access (controller API, GreptimeDB, metrics) goes through the API server's service / pod proxy, with no port-forward process.
- Ginkgo label filtering does the tiering, and `--junit-report` produces JUnit; the summary is generated by `ReportAfterSuite`.
- Load and measurement logic lives in Python scripts that are delivered through a ConfigMap and run inside the Pod. No load runs on the laptop.

Every test case has: a stable ID, its own namespace (deleted at the end, then waits for the `TensorFusionWorkload` to disappear and for GPU `available == capacity`), a bounded timeout, and its measured numbers written to the summary.

## 4. How to run

```bash
export AWS_PROFILE=tf-e2e
cd e2e

make up TTL=3h            # environment (14–15 minutes), dead-man switch fires after 3 hours
make test-smoke           # smoke-nogpu
make rehearsal            # mock-install + rehearsal tier
make test-dryrun          # dryrun tier
make cost                 # how much was spent, how much per hour right now
make down                 # destroy (about 12 minutes; 5–6 minutes when the dead-man switch already removed the nodes) + leftover sweep

# Real GPU (once the quota is in place, see GO-LIVE.md)
make test-core
make test-isolation
make test-karpenter-finops
make test-perf-quick
E2E_PERF_PROFILE=full make test-perf-full

# Any kubeconfig (bare metal)
make test-core KUBECONFIG=~/.kube/bare-metal
E2E_PERF_PROFILE=full make test-perf-full KUBECONFIG=~/.kube/bare-metal

make test FILTER='core && !disruptive' FOCUS='ISO-0'   # any label filter + focus by ID
make list                                              # list all test cases and tiers
make report RUNS="results/<a> results/<b>"             # side-by-side comparison of multiple runs
```

Results are in `results/<run-id>/` (git-ignored): `junit.xml`, `ginkgo.json`, `summary.json`, `summary.md`, `run.log`, plus per-case attachments (`<ID>.perf.json`, `<ID>.finops.json`, `<ID>.nodes.json`).

Common environment variables:

| Variable | Purpose | Default |
|---|---|---|
| `E2E_GPU_MODE` | `real` / `mock` / `none` / `auto`. The per-tier Makefile targets already set it | Per tier |
| `E2E_FIXED_IMAGES=1` | Set when the images under test already include the pricing initialization fix; enables the path B cases | Unset |
| `E2E_PERF_PROFILE` | `quick` / `full` | `quick` |
| `E2E_IMAGE_TORCH` / `E2E_IMAGE_VLLM` / `E2E_IMAGE_PYTHON` / `E2E_IMAGE_BASE` | Workload images (override when going through a registry mirror) | See `suite/framework_test.go` |
| `E2E_LLM_MODEL` | Selects the LLM; if unset it is chosen automatically from the VRAM limit | Automatic |
| `E2E_NATIVE_RUNTIME_CLASS` | runtimeClass of the native baseline Pod (when the node's default runtime is not nvidia) | Empty |
| `E2E_POOL` / `E2E_MOCK_POOL` | Target GPUPool names | `tensor-fusion-shared` / `tensor-fusion-mock` |

## 5. Tiers: cost and duration

Unit prices (us-west-2, `pricing/us-west-2.json`, 2026-10-11): EKS control plane $0.10/h, t3.large $0.0832/h, g4dn.xlarge on-demand $0.526/h (spot about $0.2–0.25/h), g5.xlarge $1.006/h, g6.xlarge $0.8048/h, t3.medium $0.0416/h. An idle cluster costs about **$0.19/h**.

| Tier (label) | Needs | Cases | Duration | Cost (including idle cluster) | Status |
|---|---|---|---|---|---|
| `smoke-nogpu` | No GPU | 27 | **7 minutes (measured)** | **about $0.03 (estimated from measurement)** | Run |
| `rehearsal` | Mock GPU (1–3 t3.medium spot) | 21 | **about 75 minutes (measured)** | **about $0.35 (estimated from measurement)** | Run |
| `dryrun` | No GPU (1 CPU spot node) | 7 | **10 minutes (measured)** | **about $0.05 (estimated from measurement)** | Run |
| `kwok` | No cloud account needed: one Linux host with docker (kind single node, memory limit 6g) | 16 | Stack up in 72–76 seconds; one full run 1156 seconds (about 19 minutes, measured) | $0 | All 16 cases have been run (9.6): all pass on this branch's operator, of which `KRP-C01` and `NLC-10` passed on a rerun after the test cases were changed; on release images ≤ 2.17.27 `KRP-B02` / `KRP-B10` fail |
| `core` | 1–2 g4dn.xlarge | 13 | about 40 minutes (estimated) | on-demand about $1.0, spot about $0.5 (estimated) | Not run |
| `isolation` | 1 g4dn.xlarge | 6 | about 50 minutes (estimated) | on-demand about $0.7 (estimated) | Not run |
| `karpenter-finops` | Up to 3 g4dn.xlarge (+ 1 g4dn.2xlarge for path B) | 11 | about 90 minutes (estimated; with `consolidateAfter=10m` each wait for node reclamation takes more than 10 minutes) | on-demand about $1.7, spot about $0.9 (estimated) | Not run (9 of them have been rehearsed on mock) |
| `perf` (quick, label `perf-quick`) | 1 g4dn.xlarge | 5 | about 60 minutes (estimated, first round includes the vLLM image pull) | on-demand about $0.75 (estimated) | Not run |
| `perf` (full) | Mainly for the free bare-metal 2×RTX 3090; on EKS only `PERF-LLM-01/02` is recommended | 7 | bare metal about 6 hours; reduced EKS subset about 50 minutes | EKS subset on-demand about $0.6 (estimated) | Not run |
| `disruptive` | Any | 1 (`UPG-03`) | about 5 minutes | — | Passed (no-gpu, uninstall 190s) |

One full pass on EKS (smoke + core + isolation + karpenter-finops + perf-quick): about 4.5 hours, on-demand about **$4.2**, about $2.6 when using spot as much as possible. Even with two or three reruns of each tier it stays within $15, far from the $100 budget. The estimates are the sum of the fixed waits and windows in the test cases and are not backed by data from real GPU runs.

## 6. Test case list

Columns: ID, tier, what it verifies, pass criteria (summary), status. In the Status column: **Passed** = run on real AWS and passed (mode in parentheses); **known-fail** = asserts the correct behavior, which the currently released images do not satisfy; **Not run** = the code is implemented and is waiting for a GPU.

### 6.1 Installation and discovery

| ID | Tier | What it verifies | Pass criteria | Status |
|---|---|---|---|---|
| INS-01 | smoke | Control plane is ready after Helm install | 12 CRDs Established; controller Available; webhook has a caBundle; the leader lease has a holder; GreptimeDB and alertmanager Ready; controller restart count is recorded | Passed (no-gpu) |
| INS-02 | smoke | Default cluster and Pool are reconciled | TFC, default pool, and `tf-scheduling-config` exist; the pool has written its status; it is the default pool; with GPUs it must be `Running`, with zero nodes the actual phase is recorded (measured `Updating`, G-04) | Passed (no-gpu) |
| INS-03 | core / rehearsal / karpenter | GPU nodes and GPUs are discovered automatically, and Karpenter nodes are managed (formerly KRP-A01) | GPU CR capacity > 0; GPUNode `Running`; hypervisor Ready with owners that include both the GPUNode and the Node; the node has 16 `index_*` resources and belongs to a Karpenter NodePool; pool totals are correct | Passed (mock-gpu); not run on real GPU |
| INS-04 | smoke / rehearsal | Non-GPU nodes are not managed | system / cpu nodes have no GPUNode and no pool label | Passed (no-gpu, mock) |

### 6.2 Fractional sharing and isolation

| ID | Tier | What it verifies | Pass criteria | Status |
|---|---|---|---|---|
| ISO-01 | core / rehearsal | Soft fractional sharing: several Pods on one GPU | 3 Pods at 20% have the same `gpu-ids`; the used amount on the GPU CR = 3 requests (the CR accounting delay is recorded); `activeIsolationMode=soft`; on real GPU `nvidia-smi` inside the container is also checked; resources are returned after deletion | Passed (mock-gpu); not run on real GPU |
| ISO-05 | core / rehearsal | shared whole GPU | The GPU's `available` drops to zero; later fractional Pods do not land on this GPU | Passed (mock-gpu); not run on real GPU |
| ISO-06 | core | remote mode | The client is on a non-GPU node with label `component=client`; a `TensorFusionConnection` exists; the worker Pod is on a GPU node; inside the client container `torch.cuda` is available and sees only 1 GPU | Not run |
| ISO-14 | core / rehearsal | Requirements of soft mode on the application image | An image without libstdc++ (busybox) should be able to run under soft | **known-fail** (G-03, reproduced on mock) |
| ISO-02 | isolation | Soft compute limit, measured | matmul throughput at 100% / 50% / 25% share satisfies the proportional bands in section 7.3; no stall ≥0.9s | Not run |
| ISO-03 | isolation | VRAM limit, measured (once for soft, once for hard) | With a 4Gi limit, allocating 6Gi fails; 3.5–4.1Gi is actually obtained; total VRAM seen inside the container ≈ 4Gi; the neighbor on the same GPU does not restart and is not affected | Not run |
| ISO-04 | isolation | Hard compute limit, measured | Same as ISO-02, with the hard proportional bands | Not run |
| ISO-15 | isolation | Neighbor isolation (soft, hard) | The quiet Pod (40%) keeps throughput ≥ 80% of its standalone throughput while a noisy neighbor (60%, fully loaded) is present; no stall ≥0.9s | Not run |
| ISO-16 | isolation | Adjusting the limit at runtime (cron scaling rule, formerly ASC-02) | Within the window throughput drops into the proportional band for a 30% share, and after the window ends it recovers to ≥80% | Not run |
| ISO-13 | isolation | `disable-features: gpu-limiter` | With a 30% limit but throttling disabled, throughput ≥ 85% of the unlimited throughput | Not run |

`partitioned` (MIG) cannot be tested on g4dn / g5 / g6 (T4, A10G, and L4 do not support MIG); it is left to bare metal / other instance types, see section 8.

### 6.3 Scheduling

| ID | Tier | What it verifies | Pass criteria | Status |
|---|---|---|---|---|
| SCH-01 | core / rehearsal / karpenter | Default bin packing (`NodeCompactGPULowLoad`) | With two GPU nodes, all 4 Pods at 20% are on one node | Passed (mock-gpu); not run on real GPU |
| SCH-04 | core / rehearsal | Scheduling by GPU model | A Pod with a matching model is scheduled to that model; a nonexistent model is not scheduled within 60 seconds, and no new NodeClaim is created for it | Passed (mock-gpu); not run on real GPU |
| SCH-10 | smoke / core / rehearsal | Namespace quota | A Pod exceeding `total.requests.tflops` stays Pending with an event that contains quota; with GPUs, Pods within the quota run and `status.used` is correct | Passed (no-gpu, mock) |
| SCH-11 | core / rehearsal | Gang scheduling (positive) | 2 replicas with `gang-enabled` become Running together; `gang-required-members=2`; same group key | Passed (mock-gpu); not run on real GPU |
| SCH-12 | smoke | Allocation query / expander status API | `GET /api/allocation` and `/api/node-scaler` are available | Passed (no-gpu) |
| SCH-14 | smoke | Automatic HostPort allocation | The `port-number` annotation is within 40000–42000 | Passed (no-gpu) |

### 6.4 Webhook and negative cases

| ID | Tier | What it verifies | Pass criteria | Status |
|---|---|---|---|---|
| WHK-01 | smoke / rehearsal | Onboarding with only a label + annotations (includes the former OBS-04: the Pending reason is diagnosable) | `schedulerName`, `component=worker`, pool label, 1 `index_*` limit, the soft init container and environment variables, a `TensorFusionWorkload` is generated; with zero GPUs the event states `no gpu devices in pool` | Passed (no-gpu, mock) |
| WHK-02 | smoke | Referencing a WorkloadProfile, with annotation overrides | Resources and QoS come from the profile; annotations can override them | Passed (no-gpu) |
| WHK-03 | smoke | Multi-container Pod with the injected container specified | Rejected without `inject-container`; with it, only that container carries the index resource | Passed (no-gpu) |
| WHK-05 | smoke | `enabled-replicas` gradual rollout | Exactly 2 of 4 replicas are onboarded | Passed in 2 of three rounds; in 1 round 3 were onboarded, **intermittent known-fail** (G-17) |
| WHK-06 | smoke | Pods that are not enabled are left unchanged | With annotations only and no label: default scheduler, no init container, no workload | Passed (no-gpu) |
| WHK-07 | smoke / core / rehearsal | Resources are returned after the workload is deleted | The workload disappears with its owner; with GPUs, `available == capacity` | Passed (no-gpu, mock) |
| WHK-08 | smoke | Server-side dry-run has no side effects | No `TensorFusionWorkload` appears after a dry-run Pod creation | **known-fail** (G-02) |
| NEG-02 | smoke | Nonexistent pool | Admission rejects it with `gpu pool not found` | Passed |
| NEG-03 | smoke | Local hard without a sidecar | Actual: accepted, and `tensorfusion-worker` is added automatically (G-09) | Passed (asserts current behavior) |
| NEG-04 | smoke | partitioned without partition-id | Admission rejects it (the error text is unrelated to partition, G-12) | Passed |
| NEG-05 | smoke | Whole GPU without `gpu-model` | Admission rejects it | Passed |
| NEG-06 | smoke | `gang-min-members=1` | Admission rejects it | Passed |
| NEG-11 | smoke | `tflops-*` and `compute-percent-*` both set | Actual: accepted, tflops takes effect (G-10) | Passed (asserts current behavior) |
| NEG-12 | smoke | soft with no resources set | Admission rejects it | Passed |
| NEG-13 | smoke | Invalid `isolation` / `qos` | Not accepted (rejected by the CRD enum, G-11) | Passed |

### 6.5 Recovery

| ID | Tier | What it verifies | Pass criteria | Status |
|---|---|---|---|---|
| NLC-01 | core / rehearsal | hypervisor restart | The new hypervisor is Ready and the GPU returns to `Running`; application Pods do not restart; GPU `available` is unchanged; the recovery time and whether `Pending` was observed are recorded | Passed (mock-gpu); not run on real GPU |
| NLC-02 | core / rehearsal | operator restart | The key set of `/api/allocation` is the same before and after the restart; a GPU that is already 60% occupied is not given to another 60% Pod; `available` is correct | Passed (mock-gpu); not run on real GPU |
| NLC-09 | karpenter / rehearsal | A node is deleted | The GPUNode and GPU CR are cleaned up; the Deployment's Pod becomes Running again on another node; both durations are recorded | Passed (mock-gpu); not run on real GPU |

### 6.6 Karpenter compatibility and FinOps

Two mutually exclusive paths (`docs/stack-context.md` §2.4): **compatibility mode (path A)** = GPUPool `AutoSelect` + Karpenter NodePool (chart default); **provisioning mode (path B)** = `provisioningMode: Karpenter`.

| ID | Tier | What it verifies | Pass criteria | Status |
|---|---|---|---|---|
| KRP-00 | smoke | Karpenter is ready | All NodePools / EC2NodeClasses are Ready; `FEATURE_GATES` contains `NodeOverlay=true`; the resolved AMI is recorded | Passed (no-gpu) |
| KRP-01 | smoke | NodeOverlay and RBAC | The overlay contains `index: 512` and `index_0..f: 36` and is accepted by Karpenter; the operator's ServiceAccount can create / delete / list / watch NodeClaims | Passed (no-gpu) |
| KRP-C01 | smoke | Karpenter launches and reclaims a non-GPU node | A new NodeClaim appears in the general-purpose NodePool → Initialized → Pod Running; no GPU node is launched; after deletion the NodeClaim and node disappear; the duration of each stage and the node-hours are recorded | Passed (no-gpu) |
| KRP-A08 | karpenter / rehearsal / dryrun | Path A cold start: 0 nodes → Pending Pod → NodeClaim → node → GPU registration → Pod Running | The NodeClaim nominated by Karpenter belongs to TF's NodePool; the GPU registers and the Pod is Running, with the three durations recorded; **without quota**: the EC2 quota error is captured as evidence and the case is skipped | Passed (mock-gpu, 56s); dry-run: the GPU NodeClaim was created and EC2 rejected it for quota |
| KRP-A15 | karpenter / dryrun | Side effect of the chart's default NodeOverlay | After removing this stack's protective overlay, a Pending TF Pod should not cause a NodeClaim to appear in a non-TF NodePool | **known-fail** (G-01), reproduced in the dry-run tier |
| KRP-A02 | karpenter / rehearsal | Path A core loop: not enough GPU → the expander creates a NodeClaim → new node → Pod Running | One Pending Pod adds only 1 NodeClaim, which belongs to the target NodePool and has no `do-not-disrupt`; the Pod is Running; `expansion-source`, the "node ready → Pod bound" duration, and the node-hours are recorded | mock-gpu: failed once with `consolidateAfter=5m` (G-16), passed with `=10m` (381s, of which 330s was waiting for the scheduler retry); not run on real GPU |
| KRP-A16 | karpenter / rehearsal | A Pending Pod should be bound promptly once the GPU on the scaled-up node is ready | The "node Initialized → Pod bound" value measured by `KRP-A02` is < 120s | **known-fail** (G-16, measured 330s on mock) |
| KRP-A17 | karpenter / kwok | No over-scaling when N whole-GPU Pods are Pending at the same time | Default N=4 (`E2E_BURST`): exactly N new NodeClaims appear over the whole run, all from TF's NodePool; the Pods land on N nodes; no new NodeClaim appears within 60 seconds after all are Running | Passed (fake-gpu, 4 → 4, 45s); not run on real GPU |
| KRP-A03 | karpenter / rehearsal | One scale-up absorbs several Pending Pods | 3 Pods at 20% add only 1 NodeClaim and land on the same GPU of the same node | Passed (mock-gpu); not run on real GPU |
| KRP-A09 | karpenter / rehearsal | Nodes are reclaimed after the workloads are gone | The Node / NodeClaim / GPUNode / GPU CR of every empty GPU node disappear; the reclamation time and whether `should-delete` was applied first are recorded | Passed (mock-gpu); not run on real GPU |
| FIN-01 | karpenter / rehearsal | FinOps: N=3 fractional workloads vs. the whole-GPU baseline | The fractional setup uses only 1 node, and the whole-GPU baseline uses more nodes; the output gives the node count, node-hours, on-demand cost, and savings ratio of both setups (in rehearsal the g4dn.xlarge unit price is substituted and the result is marked `mock-gpu`) | Passed (mock-gpu); not run on real GPU |
| KRP-B02 | karpenter / dryrun / kwok | Path B: capacity watermark → GPUNodeClaim → NodeClaim of the same name → node → GPU registration → Bound → the workload is scheduled to that node (includes the former B01 connection check) | `CloudVendorConnectionReady=True`; the GPUNodeClaim carries `hourly-pricing`; the NodeClaim's owner is the GPUNodeClaim, it carries `do-not-disrupt`, and it requests `nvidia.com/gpu: 1`; Bound; the GPU CR is Running and belongs to that NodeClaim's node, and the GPUNode carries `node-provisioner` and this pool; a fractional Pod that specifies this pool lands on this GPU of this node; no second GPUNodeClaim appears within 30 seconds; any leftover after deleting the pool is a failure | Passed on fake-gpu (this branch's operator); **known-fail** on release images ≤ 2.17.27 (G-08 and the two defects after it, see 9.6), needs `E2E_FIXED_IMAGES=1`; not run on real EC2 |
| KRP-B10 | karpenter / kwok | Path B: reclamation after the Pool is deleted (runs after `KRP-B02` and reports its cleanup time) | The GPUPool, GPUNodeClaim, NodeClaim, node, GPUNode, and GPU all disappear | Passed on fake-gpu (10s); not run on real EC2 |

Spot and on-demand fallback: the NodePool allows both capacity types, and the node ledger (`<ID>.nodes.json`) records the actual `capacity_type` of each node. The fallback path itself has been verified passively on the account without quota: Karpenter tries spot first (`MaxSpotInstanceCountExceeded`) and then switches to on-demand (`VcpuLimitExceeded`), see `GO-LIVE.md`. Once there is quota, a spot capacity shortage cannot be produced reliably, so there is no separate assertion case (section 8).

### 6.7 Observability, console, uninstall

| ID | Tier | What it verifies | Pass criteria | Status |
|---|---|---|---|---|
| OBS-01 | smoke | operator metrics endpoint | Returns `controller_runtime_reconcile_total` | Passed (no-gpu) |
| OBS-06 | smoke | GreptimeDB is queryable | The HTTP SQL API is available; the existing `tf_*` tables are recorded (measured as 0 tables with zero GPUs, G-06) | Passed (no-gpu) |
| OBS-02 | core / rehearsal | Metrics are actually stored | Within 6 minutes after the workload runs, `tf_gpu_usage`, `tf_worker_usage`, `tf_node_metrics`, `tf_pool_metrics`, and `tf_worker_resources` all have rows from the last 15 minutes; the time until the first row arrives is recorded per table | **known-fail** (G-07, depends on the fixed images), reproduced on mock: 3 tables have data, `tf_gpu_usage` and `tf_worker_usage` do not |
| CLD-01 | smoke | Console agent | Runs only when `TF_AGENT_ID` was given at install time: the `cluster-agent` container is Ready; the enroll token does not appear in the logs. Whether the cluster appears in the console UI must be confirmed manually | Skipped (agent not configured) |
| UPG-03 | disruptive | Uninstall script | The 12 CRDs, webhook, namespace, and NodeOverlay disappear; nodes have no `tensor-fusion.ai/*` labels or capacity; ordinary Pods can be created | Passed (no-gpu, uninstall 190s) |

### 6.8 dry-run tier (for the stage when there is no GPU yet)

| ID | What it verifies | Pass criteria | Status |
|---|---|---|---|
| DRY-PERF-01 | LLM measurement and judging pipeline | A CPU stand-in server simulates four throttling behaviors: smooth and proportional → judged as pass; 500ms step release → the tail-latency cliff and bimodality are detected; injected 1s stalls → the stalls are detected; no throttling → "limit has no effect" is detected | Passed (dry-run) |
| DRY-PERF-02 | ORT / TorchBench-mini / MLPerf-LoadGen scripts | Export, session creation, time-limited measurement, and result parsing complete on a CPU node, and all 8 result points have a throughput | Passed (dry-run) |
| DRY-ISO-01 | Manifests of the GPU tiers | 11 manifests (soft / hard / shared probes, vLLM soft / hard / eager, ORT, TorchBench, MLPerf, native baseline) pass server-side dry-run | Passed (dry-run) |
| DRY-IMG-01 | Images and models | Every image used resolves on the registry to a digest and a linux/amd64 compressed size; the `config.json` of the LLM models is accessible without a token | Passed (dry-run) |
| KRP-A08 / KRP-A15 / KRP-B02 | See 6.6 | | |

### 6.9 KWOK tier (no cloud account needed, label `kwok`)

Purpose: to still be able to regression-test the **control plane** of the Karpenter integration without an AWS account and without GPUs — in particular the "Pending Pod is not re-queued after the GPU is ready" issue fixed in 2.17.24 (G-16, 330s before the fix).

Composition (`kwok/up.sh`, parameters in `kwok/lib.sh`):

| Component | Version / source | Notes |
|---|---|---|
| kind single node | `kindest/node:v1.36.4` | The only node with a kubelet, label `e2e.tensor-fusion.ai/role=system`, runs the entire control plane |
| KWOK | v0.8.0 (`kwok.yaml` + `stage-fast.yaml`) | Fake kubelet: keeps the heartbeat of fake nodes alive and marks Pods scheduled to fake nodes as Running / Ready |
| Karpenter | The `kwok/` provider of kubernetes-sigs/karpenter **v1.14.1**, built from source (upstream does not publish an image) | Same Karpenter core version as the EKS stack; `FEATURE_GATES` contains `NodeOverlay=true`; instance types come from `kwok/instance-types.json` (g4dn.xlarge / g4dn.2xlarge / g5.xlarge / g6.xlarge / c5.large / m5.large / t3.large, on-demand prices taken from `pricing/us-west-2.json`, spot at 0.4× of that) |
| NodePool | `tf-gpu` (weight 10, WhenEmpty, `consolidateAfter=1m`), `tf-cpu` (weight 100, 30s), `KWOKNodeClass/tf-kwok` (`nodeRegistrationDelay=10s`) | Same shape as the EKS stack; nodes of the GPU NodePool carry `nvidia.com/gpu.present=true` and `e2e.tensor-fusion.ai/shared-pool=true` |
| TensorFusion | This repository's chart + the operator image (default release image `tensor-fusion-operator:2.17.24`; `TF_OPERATOR_IMAGE` / `TF_RELEASE_TAG` can point to a development image already present in the local docker, and it is not pulled again if present locally) | Same values as rehearsal; differences: GreptimeDB, alertmanager, and the hypervisor's vector sidecar are disabled, and the controller drops `-enable-alert` and `-enable-auto-scale` (they depend on the TSDB); `karpenterNodeOverlay.requirements` is changed to `karpenter.kwok.sh/instance-family In [g4dn, g5, g6]` (the chart default uses the AWS provider's labels, which the KWOK provider does not produce) |
| fake hypervisor | `kwok/fakehv` (Go, build tag `e2e`, runs on the host with kind's kubeconfig) | See below |

What the fake hypervisor writes on behalf of the real one, item by item against the real hypervisor's write points:

| Object | Fields | Corresponding code |
|---|---|---|
| `Node.status.capacity / allocatable` | 16 `tensor-fusion.ai/index_<hex>`, 36 each | `pkg/hypervisor/backend/kubernetes/deviceplugin.go` |
| `GPU` (one per fake device, label `e2e.tensor-fusion.ai/fake=true`) | Labels `managed-by` (`utils.LabelSafeValue(node name)`, the same function as the real hypervisor) / `gpupool`, a controller ownerRef pointing to the GPUNode, `status.capacity`, the initial `available`, `uuid`, `gpuModel`, `vendor=NVIDIA`, `index`, `numaNode`, `isolationMode` / `isolationPolicy` (taken from the environment variables the operator writes into the hypervisor Pod), `nodeSelector`, `usedBy`, `phase=Pending`, `message` | `KubeletBackend.mutateGPUResourceState` |
| `GPUNode.status` | `totalTFlops`, `totalVRAM`, `totalGPUs`, `managedGPUs`, `managedGPUDeviceIDs`, `nodeInfo` | `APIClient.UpdateGPUNodeStatus` |

It only handles nodes that satisfy three conditions at once: they carry `kwok.x-k8s.io/node=fake`, they already have a GPUNode (that is, they are already managed by TensorFusion), and the hypervisor Pod is Running. It also waits `KWOK_FAKE_HV_DELAY` (default 20s, standing for driver initialization + hypervisor startup + device discovery) after that Pod first becomes Running before writing. `phase=Running`, the later `available`, and `runningApps` are still written by the operator, as in production. One GPU per node: g4dn → T4 (65 TFLOPS / 15360Mi), g5 → A10G (125 / 23028Mi), g6 → L4 (121 / 23034Mi).

What this tier **cannot prove**: real devices and drivers, the hypervisor process itself (startup, restart recovery, device plugin registration, worker accounting, metrics), the limiter and isolation, kubelet behavior (image pulls, probes, eviction), and everything on the EC2 side (quota, capacity shortage, spot interruption, AMI). On a fake node "Pod Running" only means the scheduler bound it. Of all the durations, only the control plane's reaction time is a measurement; the node startup and GPU registration delays are parameters.

Test cases (16): `KRP-00`, `KRP-01`, `KRP-C01`, `INS-03`, `KRP-A08`, `KRP-A15`, `KRP-A02`, `KRP-A16`, `KRP-A03`, `KRP-A17`, `KRP-A09`, `FIN-01`, `NLC-09`, `NLC-10`, `KRP-B02`, `KRP-B10`. `NLC-10` is unique to this tier: the KWOK provider names nodes `kwok-<NodeClaim>-<two words>-<n>-<random string>-<uint32>`, so the node names for both the NodeClaims created by the expander (`tensor-fusion-<10 chars>`) and the path B NodeClaims exceed 63 characters (measured 68 and 71). The case first fills the existing GPUs and then starts one more Pod to force such a node, and then asserts that the GPU's `managed-by` label is valid and equals `LabelSafeValue(node name)`, that the full node name is in the ownerReference, that the GPUNode's allocated amount changes with the Pod (which shows the operator's selector can find this GPU), and that the Pod is scheduled to that node. `up.sh` no longer modifies the upstream provider's node naming. The test case code is shared with the other tiers; the only KWOK-specific parts are `E2E_GPU_MODE=fake` (mode recorded as `fake-gpu`), the NodeClass reference for path B (`KWOKNodeClass`), and the steady-state window of `FIN-01` (`E2E_FINOPS_HOLD=45s`). `KRP-A02` now adds `gpu_running_to_pod_bound_seconds` in all tiers (GPU CR becomes Running → Pod bound, sampled every 500ms), and the `KRP-A16` criterion for it is < 30s. `KRP-A15` is no longer skipped on clusters without the protective overlay; it asserts directly against the overlay shipped with the chart.

```bash
make kwok-up kwok-test kwok-down                      # when docker is available locally
KWOK_SSH_HOST=<host> make kwok-up kwok-test kwok-down  # the checkout is copied to that host and run there, results are copied back to results/
FOCUS='KRP-A02|KRP-A16' make kwok-test
TF_RELEASE_TAG=2.17.23 make kwok-up                   # switch the version under test (to compare with before the fix)
```

Registry mirrors, proxies, and host paths are given through environment variables (`KWOK_IMAGE_MIRROR`, `KWOK_K8S_IMAGE_MIRROR`, `KWOK_PULL_TOOL=crane`, `GOPROXY`, `KWOK_HOME`; for remote execution they go in the remote file that `KWOK_REMOTE_ENV` points to) and are not written into the repository.

Host prerequisite: `fs.inotify.max_user_instances` needs headroom (a known kind requirement, ≥ 512 recommended). The kubelet and containerd inside the kind node container consume inotify instances as host root, and the default of 128 may not be enough when another Kubernetes distribution is already running on the host. The symptom is a `kubeadm init` timeout and `error creating fsnotify watcher: too many open files` in the node logs (this is what was seen before raising it; after raising it to 1024 it succeeded on the first try).

Path B: the ClusterRole of chart ≥ 1.8.9 allows the operator to read the NodeClass of each Karpenter provider (including `karpenter.kwok.sh`). Measured: `kubectl auth can-i get/list/watch kwoknodeclasses` is yes and create is no. The former `KWOK_NODECLASS_RBAC=1` bypass in `up.sh` has been removed. The pool used by `KRP-B02` has no `nodeSelector` (`NodeProvisioner` and `NodeSelector` being mutually exclusive is an API convention); the node is attributed to the pool through the `tensor-fusion.ai/node-provisioner` label propagated down from the NodeClaim.

## 7. Design of the performance and isolation-effect tests (`isolation`, `perf`)

### 7.1 What is measured

Not only throughput. A limit being "effective" and "not wrecking service quality" are two different things, so every measurement point records all of the following:

| Workload | Throughput | Latency | Stalls / anomalies |
|---|---|---|---|
| LLM streaming inference (vLLM) | Output tokens/s, TPM, requests/s | TTFT, TPOT, inter-token latency (ITL), total request latency, each at p50 / p95 / p99 / max | Number of ITLs ≥0.4s and ≥0.9s; number that fall near 0.5s and 1s; bimodality coefficient of the ITL; error rate, timeout count |
| ONNX Runtime / TorchBench / MLPerf / matmul | samples/s or iters/s | Per-call latency p50 / p95 / p99 / max | Number of single calls ≥0.4s and ≥0.9s |

Why 0.5s / 1s are watched: the early soft limiter waited for a token with a 1s timeout and refilled tokens every 500ms, so the signature of "starving" a workload was periodic stalls of about 0.5s / 1s. The current limiter refills every 10ms and a single wait is at most 20ms, so such gaps do not occur when it works normally, and these two counters therefore remain a signal that "the limiter has a problem".

The LLM load generator (`workloads/llm_client.py`, standard library only) has two load shapes: closed loop with fixed concurrency (saturation) and open loop with a fixed arrival rate (below capacity there should be no degradation at all).

### 7.2 Workloads and runtimes

How the limiter intercepts kernel launches, CUDA graphs, and VRAM allocation differs by framework, so different execution shapes are covered:

| Runtime | Model / shape | Notes |
|---|---|---|
| vLLM (streaming, CUDA graph / compiled path; also an `--enforce-eager` variant) | `Qwen/Qwen2.5-0.5B-Instruct` (VRAM limit < 10Gi; about 1GiB in fp16, fits into a 4Gi fraction of a T4); `Qwen/Qwen2.5-1.5B-Instruct` (≥ 10Gi) | Apache-2.0, **no Hugging Face token needed** (`DRY-IMG-01` checks this every time). Chosen automatically from the detected GPU VRAM; `E2E_LLM_MODEL` can override it |
| ONNX Runtime GPU (CUDA EP; `--provider tensorrt` is optional and not enabled by default) | `resnet50` (CNN, few large kernels), `mobilenet_v2` (many small kernels), `bert_base` / `minilm` (encoder / embedding shape), `whisper_enc` (speech encoder: conv1d + transformer) | Random weights, `torch.onnx.export` inside the Pod, no datasets or weights downloaded |
| Representative TorchBench subset | `resnet50`, `mobilenet_v2`, `bert_base` (hf_Bert), `gpt2` (hf_GPT2), `tiny_mlp` (launch-bound, corresponds to `drq`, which had the largest overhead in the historical results); eval, plus train in the full tier | Default `--impl mini`: a reimplementation using the same model classes that TorchBench wraps, **not** the pytorch/benchmark harness; `--impl torchbench` uses the real `run.py` (for bare metal, unverified) |
| MLPerf Inference style | Official LoadGen (`mlcommons-loadgen`) + an ONNX Runtime SUT; `resnet50`, `bert`; Offline, Server | Explicitly **not a compliant submission**: random weights, synthetic inputs, `min_duration` on the order of seconds, and the Server target QPS is 60% of the measured Offline value |
| matmul (`gpu_probe.py burn`) | FP16 2048×2048 | A purely GPU-bound reference; the isolation cases use it |

Existing ideas that were carried over: the matmul approach of `gemm.py`, the metric definitions of the vLLM load generator (TTFT / TPOT / percentiles), and the rationale for choosing the TorchBench subset (`drq` +44.8% and the rest ±1% in the 2025 results). Nothing was reused directly; the manifests in this directory were rewritten; no log samples were left for the MLPerf part. The historical numbers can serve as a baseline reference: vLLM in hard mode −1.3% to −2.4% (Qwen2.5-7B, 2025-11); the vast majority of the 84 TorchBench models within ±1%.

### 7.3 Pass criteria (two-sided)

Let `s` be the compute share. The baseline is the result at 100% share for the same mode, the same VRAM limit, and the same workload, and native is the result without going through TensorFusion. The implementation is in `suite/perf_eval_test.go`, with the thresholds concentrated in `criteriaFor()`.

| Criterion | soft | hard | Notes |
|---|---|---|---|
| **(a) Effective: throughput is proportional** `thr(s)/thr(100%)` | `[0.75·s, 1.25·s + 0.05]` | `[0.85·s, 1.15·s + 0.05]` | soft is an in-process token bucket + PID feedback on utilization, refilled every 500ms, and it regulates around the target, so ±25% is an honest band; hard limits directly in the worker that executes the calls, the mechanism is more direct, so ±15% is used |
| (a) How soft is actually judged (changed after the T4 measurements, 2026-10-13) | Throughput ≥ 0.85·s (must not be too tight); throughput ≤ min(s + 0.35, 0.98) (looseness is bounded); when the window has ≥ 2 device samples, device utilization ≤ s + 10 points | Still the symmetric band of the row above | The soft limiter controls the fraction of time the device is busy (NVML utilization), not throughput. On the same T4 with a 50% limit, utilization was 53–56% and matmul throughput was 79–81% of full load, unchanged over five minutes: under intermittent load more computation completes per millisecond than under sustained full load. So looseness is judged by utilization, and throughput only gets a lower bound and a wide upper bound |
| (a) The limit is not decorative | Throughput at the lowest share (≤60%) ≤ 0.85 of the throughput at 100% | Same | A flat curve means nothing was limited |
| (a) Overhead of 100% share relative to native | Throughput loss ≤ 5% | ≤ 15% (shared ≤ 3%) | "Isolation has no cost when unlimited" is the claim to be verified; hard has remoting overhead |
| **(b) No cliff: tail latency** p99 of TPOT (LLM, per request) or of per-call latency (p95 when there are fewer than 300 samples) | ≤ 1.5 × baseline / s + 12 ms | ≤ 1.3 × | Slowing down in proportion is expected; the part beyond proportion is a cliff. The 12 ms for soft is one refill period of the limiter (10 ms refill, a single wait of at most 20 ms); a sub-millisecond call inevitably lands in the gap once per period |
| (b) ITL p99 (LLM, per token) | ≤ 1.5 × baseline ITL p99 / s | ≤ 1.3 × | TPOT is a per-request mean and smooths stalls away; the inter-token latency does not. In the rehearsal's "500ms step release" stand-in, TPOT p50 was fully proportional and ITL p99 was 4.5 times the proportional value |
| (b) p50 | ≤ 1.3 × baseline p50 / s | Same | |
| (b) TTFT p95 | ≤ 1.5 × baseline p95 / s + 0.25s | Same | A fixed term is added to tell "slowed down in proportion" apart from "stuck for a moment before the first token" |
| (b) No stall signature | No gap ≥0.9s; gaps ≥0.4s ≤ 5 per thousand tokens (or per thousand calls) | Same | Judged independently for every limited point, no baseline needed |
| (b) Not bimodal | The ITL p99/p50 does not exceed 3 times that ratio of the baseline | Same | Meeting the average while being bimodal is a cliff |
| Error rate | ≤ 1% | Same | |
| Fixed arrival rate (below the share's capacity) | Completed throughput ≥ 95% of the offered load | Same | |

Sample size: p99 takes part in the judgment only when that distribution has ≥ 300 samples; with 60–299 samples p95 is judged instead, and with fewer the value is only reported. In the quick tier a 60s window gives about 200 requests and about 10,000 tokens: ITL p99 can be judged, TPOT is judged at p95, and TTFT is judged at p95. In the full tier a 180s window gives about 600 requests and more than 30,000 tokens: TPOT p99 can also be judged. All percentiles are written to the results whether or not they take part in the judgment.

These thresholds are design values. `DRY-PERF-01` used a controllable stand-in to prove that the judging logic reaches the correct conclusion for the four throttling behaviors, but whether the thresholds themselves are appropriate has to be calibrated after the first real GPU run.

### 7.4 Factors and experiment design

Factors: isolation mode (native / shared / soft / hard) × compute share × VRAM limit (tight / loose) × workload and runtime × co-tenancy (exclusive / noisy neighbor) × load intensity (saturation / fixed arrival rate). The full Cartesian product is over a thousand combinations and is not run.

- **A full sweep is done only where "proportionality" is the question itself**: the share sweep of vLLM under soft (`PERF-LLM-01`) and hard (`PERF-LLM-02`), plus one fixed arrival rate.
- **The rest uses a pairwise design** (`PERF-PW-01`, `perfDesign()`): any two levels of any two factors appear together at least once, the main effect of every factor can still be estimated, and what is given up is interactions of third order and above. 5 factors with 180 combinations take about 17 runs. Impossible combinations are excluded (whole-GPU shared has no share, no neighbor, and no VRAM limit; 100% share has no neighbor).
- Every measurement is time-limited: a short warm-up + a fixed window. Models and pip packages are cached in a hostPath on the node (`/var/lib/tf-e2e/cache`) and downloaded only once per node.
- The server is rebuilt per share instead of having its limit changed in place: apart from autoscaling, TensorFusion has no interface to change a limit in place (`ISO-16` verifies in-place adjustment separately with a cron rule); the model comes from the node cache, and a rebuild takes about 1–2 minutes.
- Parallelism: the test cases run serially within one process so that measurements do not interfere with each other. To shorten wall-clock time, start one `make test FILTER=perf FOCUS=…` on each of two nodes / two GPUs (with different `RUN_ID`s); this is not automated as a fan-out.

| ID | Tier | Contents |
|---|---|---|
| PERF-LLM-01 | perf / perf-quick | vLLM: native baseline + soft share sweep (quick: 100 / 50 / 25; full: 100 / 75 / 50 / 25) + fixed arrival rate at 50% share |
| PERF-CO-01 | perf / perf-quick | Splitting + co-scheduling: soft 100% exclusive as the reference, then two vLLM instances at 50% each land on the same GPU and are saturated at the same time. Judgment: same GPU, combined throughput ≥ 70% of exclusive, ratio of the two tenants' throughput ≥ 0.7, no stalls |
| PERF-LLM-02 | perf / perf-quick | vLLM: native + hard share sweep (quick: 100 / 50) |
| PERF-LLM-03 | perf | vLLM variants: whole-GPU shared; VRAM tight / loose × soft 100 / 50; `--enforce-eager` |
| PERF-ORT-01 | perf / perf-quick | ONNX Runtime: native, soft 100, soft 50 (full adds soft 25, shared, hard 100 / 50, soft 50 + noisy neighbor) |
| PERF-TB-01 | perf / perf-quick | TorchBench subset, same configurations as above |
| PERF-MLP-01 | perf / perf-quick | MLPerf style, same configurations as above (quick: resnet50 Offline; full: + bert, + Server) |
| PERF-PW-01 | perf | Pairwise design |

Two profiles:

| profile | Purpose | Per measurement point | Estimate |
|---|---|---|---|
| `perf-quick` (default) | Run every time, suitable for EKS spot nodes | 15s warm-up + 60s window | about 60 minutes on 1 T4, about $0.75 (on-demand) |
| `perf-full` (`E2E_PERF_PROFILE=full`) | The complete design, mainly for the free bare-metal 2×RTX 3090; on EKS only `FOCUS='PERF-LLM-0[12]'` is recommended, as a cross-hardware confirmation | 30s warm-up + 180s window | bare metal about 6 hours; reduced EKS subset about 50 minutes, about $0.6 |

Every measurement point also carries a set of machine metrics (`machine`): this GPU's utilization, actual compute, VRAM usage, power, and temperature within the measurement window, taken from the hypervisor's device telemetry (`tf_gpu_usage` in GreptimeDB, one row per minute), not the values seen inside the container. When the window is shorter than two minutes there may be only one sample; the sample count is recorded together with the metrics.

### 7.5 Output

Each run produces one machine-readable result (`summary.json`) plus a `<ID>.perf.json` per test case (all metrics and each individual criterion for every measurement point). `make report` generates:

- `report.md`: a side-by-side table of test case status; "metric vs. share" tables grouped by workload (one row per mode, with the ratios relative to 100% and relative to native, pass / fail, and the failed criteria); a FinOps table;
- `perf.csv`: one row per measurement point;
- `perf.json`: series for plotting (one per run × workload × model × mode × metric, with points as `[share, value, passed]`).

Passing the result directories of bare metal and EKS together to `make report RUNS="…"` gives a side-by-side comparison; passing the results of two versions together shows regressions.

**Archiving per version**: `results/` is not committed. For a run that should be kept long term, use

```bash
python3 tools/perf_history.py record --version 2.17 --env ack-us-east-1-t4 --label "Alibaba Cloud ACK us-east-1 · T4 (ecs.gn6i)" results/<run-id> ...
open perf-history/index.html
```

This writes `perf-history/runs/<version>/<env>.json` (committed; node names and GPU UUIDs are stripped) and regenerates `perf-history/data.js`. The viewer is a page opened directly from the file system: LLM throughput / TTFT / TPOT / ITL, the two co-scheduled tenants, the other workloads, machine metrics, the trend across versions on the same environment, and the change relative to the previous version. Archiving the same version on the same environment again overwrites it.

## 8. Test cases that are not automated, and why

The parts of the earlier list (101 entries, written from reading the code) that did not make it into the suite. **IDs from the list are not reused**; a later implementation keeps the original ID.

| Original ID | Contents | Why it is not automated |
|---|---|---|
| ISO-08, SCH-03, SCH-05, SCH-07, SCH-13, INS-07, NLC-07 | Multiple GPUs on one node: dynamic isolation, multi-GPU Pods, specifying GPU indexes, topology, whole-GPU reservation regression, coexistence with the native device plugin, defrag | This stack only has single-GPU instance types; multi-GPU is covered on bare-metal 2×RTX 3090. The suite works with any kubeconfig, but these cases have not been written yet |
| `partitioned` (MIG) positive | MIG partitions | T4 / A10G / L4 do not support MIG; in addition the chart's default dynamic policy rejects partitioned, so a separate deployment with a static policy is needed |
| ISO-07, NLC-10 | remote + hard, remote connection self-healing | `ISO-06` has to pass on real GPU first |
| ISO-10, ISO-11 | Real training / inference, performance baseline | Replaced by the `perf` tier |
| ISO-12 | Compute oversubscription ratio | The expected behavior is unclear and has to be observed on real GPU first |
| SCH-02 | Switching the placement strategy | Requires changing the global `SchedulingConfigTemplate`, which affects other cases running in parallel; suited to a separate environment |
| SCH-06 | nodeSelector / taints | Low priority, not written |
| SCH-08, SCH-09 | Priority preemption, eviction protection | Preemption and node scale-up compete (a Pending high-priority Pod triggers the expander at the same time), so it is only stable after the NodePool is pushed to its limit; also the corresponding tests in the repository are `PIt` (not enabled) |
| SCH-11 negative | With insufficient resources no gang member is bound, and timeout | With Karpenter, "insufficient resources" does not hold (it scales up); needs to be combined with quota or a NodePool limit |
| WHK-04, NEG-07 | Automatic migration of native `nvidia.com/gpu`; blast radius when the operator is unavailable | Requires changing a chart value and restarting the controller, and removes the webhook's `objectSelector` (affecting Pod creation across the whole cluster). Destructive, suited to manual execution in a dedicated window |
| ASC-01, ASC-03, ASC-04, ASC-05 | Usage-based autoscaling and its variants | Depends on usage metrics being stored (G-07), and there is no data on the release images; `ISO-16` already covers cron rules |
| OBS-03 | Built-in alerts firing | Depends on metrics (G-07); the column queried by `GPUVRAMFull` is always 0 (G-29) |
| OBS-05 | Extra application labels on metrics | Depends on metrics (G-07) |
| NLC-03, NLC-04 | Terminating Pods preserved across restarts, no leak under high-frequency create and delete | Not written; NLC-04 takes a long time and is suited to nightly runs |
| NLC-05, NLC-06 | TF compaction marking, manual trigger, do-not-disturb | `KRP-A09` records whether `should-delete` appeared first, with no separate assertion: new nodes have a 5-minute protection period, and Karpenter's `consolidateAfter` expires first |
| NLC-08 | Batched rolling update of the hypervisor | Needs two usable hypervisor image tags, not written |
| KRP-A04, A05, A07, A10, A12, A13, A14 | Variants of path A (Pods with the same name, instance type / availability zone constraints, situations that should not scale up, consolidation with running workloads, coexistence with non-TF workloads, concurrency limit, restarting the operator during a scale-up) | Not written. Part of A12 is covered by `KRP-C01` (a CPU workload does not trigger a GPU node) and `KRP-A15`; the "do not scale up for unsatisfiable requests" part of A07 is covered by `SCH-04` |
| KRP-A06 | Fallback by preference when capacity is insufficient | `InsufficientCapacityError` cannot be produced reliably; the spot → on-demand fallback sequence was observed passively on the account without quota |
| KRP-A11, KRP-B07 | Spot interruption (FIS), spot and propagated parameters for path B | Needs a FIS role and an interruption queue (`enable_spot_interruption_queue`), not written |
| KRP-B03 to B06, B08 | Scale-up, limits, reclamation, retries, and orphan marking for path B | Not written. Path B is unusable on release images ≤ 2.17.27 (G-08 and the two items listed in 9.6); on this branch `KRP-B02` / `KRP-B10` already pass in the KWOK tier, so these can be written next |
| NEG-01, NEG-08, NEG-09, NEG-10 | Oversized requests, misconfigured NodeClass, no Karpenter CRDs, GPUs in a managed node group | Not written; NEG-09 needs a cluster without Karpenter, NEG-10 needs `gpu_static_count > 0` |
| CLD-01 UI part, CLD-02 | The cluster appears in the console, creating configuration from a console template | Needs an account and a person looking at the UI; the cluster side of `CLD-01` is automated |
| UPG-01, UPG-02 | Helm upgrade within the same architecture, v1 → v2 upgrade | Needs images of two versions / a separate v1 environment |
| UPG-04 | No leftovers before destroy | Handled by the `sweep` of `make down` |
| Non-NVIDIA vendors, NVLink, GPU falling off the bus, `Provisioned` mode, replica-count scaling, reBalancer / auto-freeze | | No corresponding hardware or API on EKS, or the feature is not implemented (see the reasons in section 6 of the earlier list, which still hold) |

## 9. Results of what has been run

The data is in `results/` (not committed). Runtime environment: EKS 1.36.4, Karpenter 1.14.1, chart 1.8.8 (repository checkout), operator / hypervisor `2.17.23`, `vgpu-provider-nvidia:1.3.9`.

### 9.1 Environment and dead-man switch

See "Measured durations and cost" in `terraform/README.md`. Summary: `make up` 13m43s and 15m21s (rounds 2 and 3, succeeded on the first try); `make down` 12m07s with nodes still present and 5–6 minutes when the nodes were already removed; in two of the three rounds `sweep` was empty right after destroy, and round 2 left 2 launch templates that are not billed (fixed); the dead-man switch removed all compute resources within 6–8 minutes after expiry both times; total spend over the three rounds is estimated at about $1.1.

### 9.2 `smoke-nogpu` (three rounds, about 7 minutes each)

| Run | Result |
|---|---|
| smoke-1 (first round, including problems in the tests themselves) | 22 passed, 1 known-fail, 3 failed. The 3 failures: `KRP-C01` (disturbed by a leftover CPU node launched because of G-01; this stack's protective overlay was added afterwards), `NEG-11` (the expectation was wrong; for the actual behavior see G-10), `OBS-06` (with zero GPUs there are no `tf_*` tables at all, see G-06; changed to record the current state) |
| smoke-2 | **25 passed, 1 known-fail (`WHK-08`)** |
| smoke-3 (round 2 environment) | 24 passed, 2 failed: `WHK-08` (known) and `WHK-05` (in this round 3 of the 4 replicas were onboarded, G-17; marked as intermittent known-fail since then) |

`CLD-01` (from smoke-3 on): the agent is not configured, so it is skipped by design.

### 9.3 `rehearsal` (mock GPU, CPU instances)

Three runs: rehearsal-1 was aborted because the tests did not wait for the GPU CR accounting; rehearsal-2 is a complete round (18 cases, 68 minutes); rehearsal-3 reran the 3 scale-up related cases after raising `consolidateAfter` to 10m, and also ran the 2 cases added later.

| ID | Result | Measured |
|---|---|---|
| INS-03 | Passed | The mock GPU reports 312 TFLOPS / 16GiB; the hypervisor's owners include the GPUNode and the Node; the node has 16 index resources |
| ISO-01 | Passed | 3 soft Pods at 20% on the same GPU; GPU CR accounting delay 5.4s; returned 10.7s after deletion |
| ISO-05 | Passed | After the whole GPU is taken, fractional Pods do not land on this GPU |
| ISO-14 | known-fail | busybox exits with 127 under soft: `libstdc++.so.6: cannot open shared object file` |
| SCH-01 | Passed | With two nodes, all 4 Pods at 20% are on one node |
| SCH-04 | Passed | The Pod with a matching model is scheduled; the nonexistent model is not scheduled and no NodeClaim is created (event `NodeExpansionFailed: no potential GPU nodes can fit`) |
| SCH-10 | Passed | The Pod over quota is blocked; the Pod within quota runs and `status.used` is correct |
| SCH-11 | Passed | The 2 gang members become Running together after 7.5s; `status.gang.phase=Running` |
| WHK-01, WHK-07, INS-04 | Passed | The workload is reclaimed in 0.2s and the GPU resources are returned |
| NLC-01 | Passed | The hypervisor recovers in 23s; the GPU goes through `Pending` in the meantime; application Pods did not restart and the allocated amount is unchanged |
| NLC-02 | Passed | The operator recovers in 21s; `/api/allocation` is the same before and after the restart; no duplicate allocation |
| NLC-09 | Passed | After the node is deleted the GPUNode / GPU CR are cleaned up within 0.6s; the Pod is Running on a new node after 59s |
| OBS-02 | known-fail | `tf_node_metrics`, `tf_pool_metrics`, and `tf_worker_resources` have data; `tf_gpu_usage` and `tf_worker_usage` have no data within 6 minutes |
| KRP-A08 | Passed | Cold start: NodeClaim 6s, GPU registration 51s, Pod Running 56s |
| KRP-A02 | Failed at 5m, passed at 10m | The expander creates the NodeClaim in 12.5s (owner is the NodePool, carries `expansion-source`, no `do-not-disrupt`), exactly 1; with `consolidateAfter=5m` the node was reclaimed first and the Pod timed out after 12 minutes Pending; with `=10m` the Pod was Running after 381s, of which "node Initialized → Pod bound" was 330s |
| KRP-A16 | known-fail | 330s (required < 120s) |
| KRP-A03 | Passed | 3 Pods add only 1 NodeClaim, same node and same GPU, all Running in 62.6s (this time the scheduling retry happened to come in time) |
| KRP-A09 | Passed | The empty node disappears within 46s; TF applied `should-delete` first |
| FIN-01 | Passed | See below |

Another path A measurement: when the spot quota of the mock NodePool was exhausted, the expander's first NodeClaim failed for lack of capacity (event `NodeExpansionCandidatesExhausted`), and after the scheduler re-queued the Pod the next NodeClaim created by the expander landed on **on-demand** (t3a.medium) — the spot → on-demand fallback also works on TF's expansion path.

FinOps (`FIN-01`, mock, substituting g4dn.xlarge on-demand at $0.526/h, each scenario held at steady state for 3 minutes):

| Setup | Nodes | Node-hours | Cost | Cold start until all Running |
|---|---|---|---|---|
| 3 fractional workloads (20% compute, 2Gi each) | **1** | 0.149 | $0.079 | 364s (of which about 5 minutes is the scheduling retry wait of G-16) |
| 3 whole-GPU workloads (baseline) | **3** | 0.219 | $0.115 | 126s |

At steady state it is 1 node versus 3 (67% savings). The savings measured in this short window are only 31.7%, because the node-hours of the fractional setup include the 5 minutes of idle waiting caused by G-16. This should be measured again on real GPU.

### 9.4 `dryrun`

| ID | Result | Measured |
|---|---|---|
| DRY-PERF-01 | Passed | Stand-in with smooth throttling: 372 → 189 → 95 tokens/s (100% / 50% / 25%), TPOT and ITL scale by 1/s, all criteria pass. Step release @25%: throughput 79 (proportion met), TPOT p50 80.7ms (proportional), but ITL p99 399ms (p50 20ms) and TTFT p95 436ms → bimodality and an ITL cliff are detected. Injected 1s stalls: 29 occurrences ≥0.9s → detected. No throttling @25%: throughput 367 → "not proportional" and "limit has no effect" are detected |
| DRY-PERF-02 | Passed | 8 result points on CPU in total: ORT (tiny_mlp, minilm), torch mini (tiny_mlp, mobilenet_v2 × eval / train), MLPerf LoadGen (Offline, Server); pip install + run takes about 3.5 minutes |
| DRY-ISO-01 | Passed | All 11 GPU manifests pass admission; the dry-run leaves 10 `TensorFusionWorkload` objects behind (that is, G-02) |
| DRY-IMG-01 | Passed | PyTorch 3161 MiB, vLLM **11877 MiB**, operator 102 MiB, hypervisor 71 MiB, worker 33 MiB, client 31 MiB, vgpu-provider 4 MiB (all linux/amd64 compressed sizes); the `config.json` of both Qwen models is accessible without a token |
| KRP-A08 (dry-run) | Skipped with the dry-run conclusion | TF Pod without a nodeSelector → Karpenter creates a NodeClaim for the `tf-gpu` NodePool → spot: `MaxSpotInstanceCountExceeded`; on-demand: `VcpuLimitExceeded … current vCPU limit of 0` |
| KRP-A15 | known-fail | After removing the protective overlay a `tf-cpu` NodeClaim (c5.large) appears within 38s |
| KRP-B02 | Skipped | Needs the fixed images |

There is also `make gpu-dryrun`: `RunInstances --dry-run` can launch all 16 combinations of 4 instance types × 2 AZs × on-demand / spot.

### 9.5 `disruptive`

`UPG-03`: the uninstall script finishes in 190s; the 12 CRDs, webhook, namespace, and NodeOverlay all disappear, and no `tensor-fusion.ai/*` labels or capacity are left on the nodes.

### 9.6 `kwok` (fake-gpu, 2026-10-12)

Environment: kind v0.33.0 / Kubernetes 1.36.4, KWOK v0.8.0, Karpenter v1.14.1 KWOK provider (unmodified upstream source, node names are no longer shortened), the chart is the repository checkout (1.8.9), hypervisor image tag `2.17.27` (it does not run on fake nodes). Two rounds: in the first round (morning) the operator was release image `2.17.24` / `2.17.25` and node names were shortened; in the second round the operator was a **development image built from this branch** (the runtime layers of release image `2.17.27` + this branch's `/manager`), and the full tier of 16 cases ran in one pass in 1156 seconds, with two more cases rerun separately. `up.sh` takes 72–76 seconds from zero to usable. The fake hypervisor delay is 20s, `nodeRegistrationDelay=10s`, and the GPU NodePool has `consolidateAfter=1m`. Everything here is a control-plane conclusion (mode `fake-gpu`, see section 2); durations come from 5-second polling and have 5-second precision.

| ID | Result | Measured |
|---|---|---|
| KRP-A02 | Passed (twice on 2.17.24, once on 2.17.25, once on this branch) | One Pending Pod gives exactly 1 NodeClaim (created by the expander, NodePool `tf-gpu`); NodeClaim 10–15s, Pod Running 45s; **GPU CR Running → Pod bound < 1s** (500ms sampling, condition timestamps with 1s precision, the reading is 0); node Initialized → Pod bound 19–20s, equal to the fake hypervisor delay |
| KRP-A16 | Passed | The same two numbers, with criteria < 120s and < 30s. Before the fix (2.17.23, mock, EKS) it was 330s; this tier was not compared against 2.17.23 |
| KRP-A17 | Passed (once each on 2.17.24, 2.17.25, and this branch) | 4 whole-GPU Pods → exactly 4 NodeClaims and 4 nodes, all Running in 45s, and no extra NodeClaim in the following 60s |
| KRP-A09 | Passed | This branch: 1 empty node disappears in 65s; 2.17.25: 4 in 90s; 2.17.24: 1 in 75s. In all of them the NodePool's `consolidateAfter` expired first and `should-delete` did not appear |
| KRP-A15 | Passed (once each on 2.17.24, 2.17.25, and this branch) | After the overlay's requirements are restricted to the GPU instance families, a Pending TF Pod did not cause a NodeClaim to appear in `tf-cpu` within 90s. The chart's default requirements use the AWS provider's labels, and this has still not been retested on AWS |
| KRP-A08 | Passed (this branch, once) | Cold start from 0 GPU nodes: Pending → Karpenter's own NodeClaim 5s → GPU registration 35s → Pod Running 35s (total case duration 221s; the rest is waiting for the previous case's node to be reclaimed) |
| KRP-A03 | Passed (this branch, once) | 3 Pods at 20% → 1 NodeClaim, 1 node, the same GPU, all Running in 45s; afterwards the node-scaler's in-flight state is empty |
| FIN-01 | Passed (this branch, once) | 3 fractional workloads use 1 node (all Running in 35s), and the whole-GPU baseline uses 3 nodes (75s); converted by node-hours × on-demand price, the savings are 71.5%. The steady-state window is only 45s and the node-hours are 0.022h versus 0.077h: the ratio is meaningful, the absolute amount is not |
| NLC-09 | Passed (this branch, once) | After the node is deleted the GPUNode / GPU CR are cleaned up within 5s, and the Deployment's Pod is Running on another node within 5s (the other node already existed at that time; there is no real rebuild time on fake nodes) |
| KRP-C01 | Passed (this branch; failed the first time, passed on a rerun after the test case was changed) | `tf-cpu` produces a NodeClaim in 5s, the node is Initialized and the Pod Running in 15s (t3.large, spot), and the node disappears 50s after deletion. The first failure was because the case's premise did not hold: it asserted that the GPU NodePool had no NodeClaim at all, while in the full-tier run the previous case had left a GPU node behind; it was changed to "no new GPU NodeClaim is added" |
| NLC-10 | Passed (this branch; skipped in the full-tier run, passed on a rerun after the test case was changed) | Node `kwok-tensor-fusion-<10 chars>-<word>-17-<random string>-<uint32>`, 68 characters in total: the GPU registers successfully, `managed-by` is the truncated prefix plus a 10-character hash, and the Pod is Running in 45s. It was skipped in the full-tier run because the case only looked for a long name among the existing nodes, and the cold-start node (`kwok-tf-gpu-…`) is under 63 characters; it was changed to have the expander force one |
| KRP-B02 | **Release image 2.17.27 + chart 1.8.9: failed**; this branch: passed (twice) | 2.17.27: the GPUNodeClaim, the NodeClaim of the same name, and the node appear in turn within 20s, and the node carries the `node-provisioner` label; after that there is no GPUNode and no hypervisor Pod, and the GPUNodeClaim's phase stays empty (see below). This branch: watermark → GPUNodeClaim 5s / 20s → Bound with the GPU Running 35s / 55s (the difference between the two runs is which cycle the pool's first reconcile fell into) → a fractional Pod that specifies this pool is Running in 5s, on this GPU of this node (names of 68 / 71 characters) |
| KRP-B10 | Passed (this branch, twice) | Within 10s after the pool is deleted, the GPUPool, GPUNodeClaim, NodeClaim, node, GPUNode, and GPU all disappear. On 2.17.27 the GPUNodeClaim is still there 152s after the same operation (with a finalizer and a deletionTimestamp) |
| INS-03 | Passed | Cold start to GPU Running 35s; the node carries 16 `index_*` and belongs to the NodePool |
| KRP-00, KRP-01 | Passed | NodePool / KWOKNodeClass Ready, `NodeOverlay=true`; the overlay is accepted by Karpenter, and the NodeClaim RBAC is in place |

The three path B problems found in this tier, none of which is a KWOK artifact:

1. The chart's ClusterRole had no read permission on the Karpenter NodeClass (fixed in chart 1.8.9).
2. `NodeReconciler` attributes nodes to a pool only by the pool's `nodeSelector`. A provisioning pool (`Karpenter` / `Provisioned`) has no `nodeSelector`, so the nodes it launches match no pool, and therefore there is no GPUNode, no hypervisor, and the GPUNodeClaim never becomes Bound. Fix: when a node carries the `node-provisioner` label, it is attributed to that GPUNodeClaim's pool, with `nodeSelector` matching as the fallback. A real provider is no different from KWOK on this point — in both, the label is propagated from the NodeClaim to the node. The controller's envtest has always been green because it configures the provisioning pool with both a `nodeSelector` and matching `gpuNodeLabels`.
3. The pool's finalizer deleted its GPUNodeClaims but only waited for the GPUNodes to disappear; with no GPUNode the pool was deleted immediately, and the GPUNodeClaim controller returns right away when the pool does not exist, so nobody ever removes the claim's finalizer. Fix: the pool waits until its GPUNodeClaims are gone as well.

Node names over 63 characters: the hypervisor puts `tensor-fusion.ai/managed-by=<node name>` on every GPU CR, and label values are limited to 63 bytes. In the morning round the fake hypervisor (which writes the same way as the real hypervisor) was rejected by the API server: `metadata.labels: Invalid value: "kwok-tensor-fusion-…": must be no more than 63 bytes`. After the fix the label value is `utils.LabelSafeValue(node name)`: the node name itself when it fits (existing objects do not change), and a prefix plus a 10-character SHA-256 when it does not. The writer (hypervisor) and the three selectors (GPUNode controller, the allocator's capacity refresh, the hypervisor's cleanup of stale GPUs) use the same function, and readers that need the node name now take it from the GPU's GPUNode ownerReference. **The real hypervisor binary has not been run with long node names**: the writer in this tier is the fake hypervisor, so what is proven is the operator side and the API server's validation; the hypervisor side only has unit tests (100 and 253 characters). Similar cases that were not changed: the pool name appears in the label key `tensor-fusion.ai/pool-<pool name>` (key names are limited to 63, so pool names are limited to 58) and in several label values; the values of `tensor-fusion.ai/workload` and `worker-name` are workload / Pod names, which are read back as names in several places. When these are too long they are still rejected by the API server.

### 9.7 Conclusion on image pulls

No ECR pull-through cache was set up: the TensorFusion images are on Docker Hub, and pull-through requires putting Docker Hub credentials into Secrets Manager; these images total less than 250 MiB, and on a mock node it takes about 15 seconds from node ready to hypervisor Running. Only the workload images are worth caring about: PyTorch 3.2 GiB and vLLM 11.9 GiB. The countermeasures are to pull only once per node (the perf cases run serially on the same node), to raise the root volume throughput of GPU nodes to 300 MiB/s, and to cache models and pip packages in a hostPath on the node. The real pull durations will be recorded by Pod events, and whether pre-pulling is worthwhile will be decided after the first GPU run.
