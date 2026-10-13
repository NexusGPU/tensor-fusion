# Real-GPU go-live runbook

Once the GPU quota is in place, run this document from start to finish. All commands are run from the `e2e/` directory.
Written on 2026-10-12: at this point the account's G/VT quotas (on-demand, spot) are both 0 and the quota increase request is in AWS manual review. **None of the GPU steps below has been executed.**

## 0. Preflight checks (2 minutes, $0)

```bash
export AWS_PROFILE=tf-e2e
cd e2e
make preflight
make quotas          # L-DB2E81BA (on-demand G/VT) and L-3819A6DF (spot G/VT) at least 8, 16 recommended; L-1216C47A (standard on-demand) ≥ 8
```

| Quota | Minimum | Notes |
|---|---|---|
| `L-DB2E81BA` on-demand G/VT | 8 vCPU | 2 g4dn.xlarge; the whole-GPU baseline of `karpenter-finops` needs 3 (12); path B adds 1 more g4dn.2xlarge (8) |
| `L-3819A6DF` spot G/VT | 0 also works | When it is 0 Karpenter falls back to on-demand automatically; with 8–16 most nodes will be spot and the cost is halved |
| `L-1216C47A` standard on-demand | 8 vCPU | It is currently 8, which is enough |

With only 8 on-demand G vCPUs: `make up TF_ARGS='-var gpu_node_limit=2'`, and skip the whole-GPU baseline of `FIN-01` (remove it from `FOCUS`) and `KRP-B02`.

## 1. Bring up the environment (14–15 minutes, about $0.05)

```bash
# When using the fixed images, pass the tags as parameters (every image reference is a parameter, nothing is hardcoded in the repository)
export TF_RELEASE_TAG=<tag of the operator and hypervisor>        # unset = 2.17.23 (released)
export TF_VGPU_PROVIDER_TAG=<tag of vgpu-provider-nvidia>    # unset = 1.3.9
# Optional: register with the console (keep the values only in the current shell)
# export TF_AGENT_ID=… TF_ENROLL_TOKEN=… TF_CLOUD_ENDPOINT=…

make up TTL=6h
export KUBECONFIG=$PWD/terraform/.kube/config
make mock-uninstall    # if a rehearsal was done before: leaving the mock pool in place makes Pods of the real pool launch mock nodes
make test-smoke        # about 7 minutes; first confirm that the control plane and Karpenter are healthy
```

When the images already include the pricing initialization fix and the hypervisor metrics fix, also add `export E2E_FIXED_IMAGES=1` (enables `KRP-B02`). `OBS-02`, `WHK-08`, `ISO-14`, and `KRP-A15` report `known-fail` before the fixes and should turn into passes after them.

## 2. The tiers (one command per tier)

| Order | Command | Needs | Estimated duration | Estimated cost (including the idle cluster at $0.19/h) |
|---|---|---|---|---|
| 1 | `make test-core` | 1–2 g4dn.xlarge | 40 minutes | on-demand $1.0 / spot $0.5 |
| 2 | `make test-isolation` | 1 g4dn.xlarge | 50 minutes | on-demand $0.7 |
| 3 | `make test-karpenter-finops` | Up to 3 g4dn.xlarge (+ 1 g4dn.2xlarge for path B) | 90 minutes | on-demand $1.7 / spot $0.9 |
| 4 | `make test-perf-quick` | 1 g4dn.xlarge | 60 minutes (first round includes the vLLM image pull, 11.9GB compressed) | on-demand $0.75 |
| 5 (optional) | `E2E_PERF_PROFILE=full make test FILTER=perf FOCUS='PERF-LLM-0[12]'` | 1 node | 50 minutes | on-demand $0.6 |
| 6 (optional, switch the GPU model for a cross-hardware confirmation) | Before `make test-perf-quick`, leave only `g5.xlarge` or `g6.xlarge` in the NodePool: `TF_VAR_gpu_instance_types='["g5.xlarge"]' make up` | 1 g5.xlarge (A10G) / g6.xlarge (L4) | 60 minutes | g5 $1.2 / g6 $1.0 |
| Last | `make test FILTER=disruptive` (`UPG-03` uninstall) → `make cost` → `make down` | — | 5 + 12 minutes | $0.06 |

Full pass (1–4 + wrap-up): about **4.7 hours, on-demand about $4.4**, about $2.7 with spot quota. About $6.2 when steps 5 and 6 are added. The budget is $100.
The durations and costs are computed from the fixed waits and measurement windows in the test cases; there is no data from real GPU runs. The mock rehearsal tier measured about 75 minutes.

During the run:

```bash
make deadman           # how much time the dead-man switch has left
make extend TTL=3h     # extend it if that is not enough
make cost              # how much has been spent
make status            # nodes / NodeClaims / GPUs
```

After each tier, look at `results/<run-id>/summary.md`. Rerun failed test cases individually: `make test FILTER=core FOCUS='ISO-01'`. After everything has run:

```bash
make report RUNS="results/<core> results/<isolation> results/<karpenter> results/<perf>"
```

Bare metal (2×RTX 3090) uses the same commands plus `KUBECONFIG=…`; passing the bare-metal and EKS result directories together to `make report` gives the side-by-side comparison.

## 3. What the rehearsal has already proven (real AWS, 2026-10-11 to 12)

Everything below was obtained by running on a real account, with the mode noted in parentheses.

### 3.1 Environment and cost safety

- `make up`: rounds 2 and 3 each succeeded on the first try, in 13m43s and 15m21s (EKS control plane 9–10.5 minutes, node group about 2 minutes, Karpenter and NodePools about 1 minute, TensorFusion 105s). `make down`: 12m07s with nodes still present, 5–6 minutes when they had already been removed by the dead-man switch. The per-round record of the three rounds (including the apply rerun in round 1 and the cause and fix of the 2 leftover launch templates in round 2) is in `terraform/README.md`.
- Dead-man switch (actually triggered twice, TTL 2 minutes): all compute resources were gone 5m58s and 8m12s after expiry, including the instance Karpenter relaunched after the first cleanup.
- `sweep` was empty after the last destroy; this was also cross-checked with direct describe calls: in the region, 0 instances, 0 volumes, 0 ENIs, 0 EKS clusters, 0 non-default VPCs, 0 Lambdas, 0 log groups, 0 SSM parameters, 0 launch templates, 0 `tf-e2e*` IAM roles.
- Spend: the three rounds together are estimated at about $1.1.

### 3.2 GPU launch path: only the quota is missing

- `make gpu-dryrun`: the NVIDIA AL2023 AMI for EKS 1.36 resolves (`amazon-eks-node-al2023-x86_64-nvidia-1.36-v20261003`); for all 16 combinations of g4dn.xlarge / g4dn.2xlarge / g5.xlarge / g6.xlarge × two AZs × on-demand / spot, `ec2 run-instances --dry-run` returns "would launch" (parameters, subnets, security groups, Karpenter's instance profile, tags, and volumes are all accepted). dry-run does not check quota.
- A real NodeClaim attempt (mode `dry-run`): after a TF Pod without a nodeSelector (label + annotations only) went Pending, Karpenter nominated and created a NodeClaim for the **`tf-gpu` NodePool**, and EC2 answered:
  - spot: `InsufficientCapacityError … UnfulfillableCapacity … MaxSpotInstanceCountExceeded: Max spot instance count exceeded`
  - then it switched to on-demand automatically: `InsufficientCapacityError … VcpuLimitExceeded: You have requested more vCPU capacity than your current vCPU limit of 0 allows for the instance bucket that the specified instance type belongs to.`
  
  This also shows that the spot → on-demand fallback order works. Once the quota is in place, the part after this step (node join, hypervisor, GPU registration) is the same chain that already ran end to end on mock; the only difference is that the device is real.

### 3.3 Control chain (mock GPU, `rehearsal` tier)

The `rehearsal` tier was run on mock GPUs (EKS 1.36 + Karpenter 1.14.1 + operator 2.17.23, nodes are t3.medium spot); per-case results are in `README.md` 9.3. Key points:

- **Cold start (path A)**: Pending fractional Pod → Karpenter NodeClaim 6s → node joins → hypervisor registers the mock GPU 51s → GPU / GPUNode CR → Pod Running 56s.
- **Scale-up (path A core loop)**: create another Pod after the GPU is full → the expander creates a NodeClaim in 12.5s (owner is the NodePool, carries `expansion-source`, no `do-not-disrupt`) → node → Pod Running. **But there is a race (G-16)**: the Pod is only bound after the scheduler's 5-minute fallback retry (measured 330s); with `consolidateAfter` ≤ 5m the node is reclaimed by Karpenter first.
- **Bin packing**: 3 Pending fractional Pods add only 1 NodeClaim and land on the same GPU of the same node; with two nodes, all 4 small Pods are on one node.
- **Scale-down**: an empty node is first marked `should-delete` by TF and then deleted by Karpenter consolidation, and the GPUNode / GPU CR are cleaned up with it.
- **spot → on-demand**: when the spot quota is exhausted, the expander's NodeClaim fails and then automatically switches to on-demand.
- **Node deletion**: the GPUNode / GPU CR are cleaned up, and the Deployment's Pod is recreated on a new node after 59s.
- **Recovery**: hypervisor restart 23s, operator restart 21s; allocation state is not lost and there is no duplicate allocation.
- **FinOps**: 3 fractional workloads use 1 node, and the whole-GPU baseline uses 3 (cost computed by substituting the g4dn.xlarge unit price).
- **Path B was not rehearsed**: it is unusable on the release images (G-08), and mock cannot cover it either (instance types come from the built-in GPU price table, and the NodeClaim requests `nvidia.com/gpu`).

The product problems confirmed in the rehearsal will occur on real GPUs as well.

### 3.4 How far the GPU-requiring test cases got (`dryrun` tier)

- `DRY-ISO-01`: all 11 manifests that the `isolation` and `perf` tiers will create (soft / hard / shared probes, vLLM soft / hard / eager, ONNX Runtime, TorchBench, MLPerf, native baseline) pass server-side dry-run.
- `DRY-IMG-01`: all images resolve on the registry (PyTorch 3161 MiB, vLLM 11877 MiB, linux/amd64 compressed sizes); Qwen2.5-0.5B / 1.5B need no token.
- `DRY-PERF-01`: the LLM load generator, metric collection, judging logic, and report ran end to end on a CPU stand-in and reached the correct pass and fail conclusions for the four behaviors "smooth and proportional / 500ms step release / 1s stalls / no throttling".
- `DRY-PERF-02`: the three scripts for ONNX Runtime, TorchBench-mini, and MLPerf LoadGen completed export, session creation, time-limited measurement, and result parsing on CPU.
- What was not reached: not a single line of the CUDA paths of these scripts (`CUDAExecutionProvider`, `torch.cuda`, vLLM itself) has been executed.

## 4. What has not been proven yet (focus on these in the first real run)

1. **Every conclusion related to CUDA**: whether the GPU is usable inside the container, whether VRAM and compute limits take effect, neighbor isolation, changing limits at runtime, and the proportionality of performance to share and tail latency. The assertion thresholds of the `isolation` and `perf` tiers are design values; the first-round results will very likely require calibrating thresholds or fixing tests, so do not treat first-round failures directly as product defects.
2. **Time scales of real GPU nodes**: NVIDIA AMI boot, pulling the hypervisor and `vgpu-provider` images, and pulling the application images (PyTorch and vLLM, about 3161 MiB and 11877 MiB compressed respectively). A mock node takes about 1 minute from Pending to Pod Running; real GPU nodes are expected to take 3–6 minutes.
   **Scale-up on path A has a race**: after the GPU on the new node is registered, the Pod that previously failed scheduling is not re-queued immediately and has to wait for the scheduler's 5-minute fallback retry; during this time the new node looks empty to Karpenter. In the rehearsal, "the node is reclaimed first and the Pod stays Pending" occurred once each with `consolidateAfter` at 1m and at 5m. This stack therefore sets the default `consolidateAfter` of the GPU / mock NodePools to 10m. `KRP-A02` records `node_initialized_to_pod_bound_seconds` and `KRP-A16` asserts on it (currently known-fail). If `KRP-A02` is still unstable on real GPUs, look at this number first.
3. **Path B** (`provisioningMode: Karpenter`): unusable on the release images, and it cannot be rehearsed on mock (instance types come from the built-in GPU price table). It will be run for the first time once the images are fixed.
4. **Storing hypervisor metrics** (`OBS-02`): on mock it was confirmed that `tf_node_metrics`, `tf_pool_metrics`, and `tf_worker_resources` have data and that `tf_gpu_usage` and `tf_worker_usage` do not (known issue). After the images are fixed all of them should arrive.
5. **remote mode** (`ISO-06`) and **hard isolation** (`ISO-03`, `ISO-04`, `PERF-LLM-02`): they need the NVIDIA worker and cannot run on mock.
6. **Native baseline** (mode `native`) depends on the node's default container runtime being nvidia (it should be on the EKS AL2023 NVIDIA AMI); if it is not, set `E2E_NATIVE_RUNTIME_CLASS=nvidia`.
7. **The vLLM image tag** (`vllm/vllm-openai:v0.11.0`) and the PyTorch image were only verified to exist on the registry and have not actually been started on a T4; T4 does not support bf16, and the scripts already use `--dtype half`.
8. **TorchBench's `--impl torchbench`** (the real pytorch/benchmark harness) has not been run and is only used in the full tier on bare metal.
9. The Cost Explorer per-tag number in `make cost`: the tag can only be activated about 24 hours after AWS billing discovers it, and it was still $0 when this rehearsal ended; the live estimate part works.

## 5. When something goes wrong

| Symptom | Action |
|---|---|
| An AWS call reports an expired session | `aws sso login --profile tf-e2e`, then rerun the last command (`make up` / `make down` are both re-entrant) |
| `terraform destroy` fails midway | Just run `make down` again; if that still fails, use `make sweep` to see the leftovers and `make sweep-delete` to remove them |
| The person has left, the agent has died | Nothing to do. When the time comes, the dead-man switch scales the node groups to 0 and terminates all instances; 6 hours later it deletes the control plane. Finish with `make down` afterwards |
| The dead-man switch has already fired and you want to continue | `make up TTL=3h` (extends the deadline first, then restores the node group) |
| The GPU NodeClaim stays at `InsufficientCapacityError` | `make quotas`; or `TF_VAR_gpu_capacity_types='["on-demand"]' make up` |
| A test case left a namespace behind | `kubectl delete ns -l tf-e2e/run` |
