# Disposable EKS + Karpenter environment for TensorFusion e2e

One command to bring it up and one command to destroy it, with an AWS-side timeout dead-man switch. It has a single purpose: to give `e2e/suite` a repeatable cloud environment.
Design priorities: **(1) leave no billable resource behind, (2) reliable up and down, (3) cost per hour**.

> Status (2026-10-12): 3 rounds of `make up` → tests → `make down` were completed on a real account; per-round results are in "Measured durations and cost".
> The GPU quota is 0, so **the GPU NodePool is only verified up to the point where EC2 returns the quota error** (evidence in `../GO-LIVE.md`). The measured numbers are in "Measured durations and cost" in this document.

## Quick start

```bash
# Dependencies: aws cli v2, kubectl, helm 3, jq, bc, curl, unzip. terraform is downloaded into .tools/ by make tools (pinned version, SHA256 verified, no brew / sudo)
export AWS_PROFILE=tf-e2e            # a named IAM Identity Center profile; the scripts and Terraform only honor this environment variable
cd e2e/terraform

make quotas              # EC2 vCPU quotas and pending quota increase requests
make up TTL=3h           # infrastructure + Karpenter + TensorFusion; the dead-man switch fires after 3 hours
export KUBECONFIG=$PWD/.kube/config   # a separate kubeconfig, ~/.kube/config is not touched
make status

make extend TTL=2h       # change the self-destruct time to "2 hours from now"
make cost                # cost already incurred (Cost Explorer) + a live estimate of the resources currently running
make down                # destroy (about 12 minutes when nodes are still present) + leftover check (the sweep output is the evidence of "no leftovers")
```

`e2e/Makefile` forwards these targets unchanged, so they can also be run from the `e2e/` directory.

## Architecture

```
VPC 10.42.0.0/16 (2 public subnets / 2 AZs, IGW, no NAT, no private subnets, no VPC endpoints)
├─ Dead-man switch: EventBridge rule (every 5 minutes) → Lambda tf-e2e-deadman → SSM parameter /tf-e2e/deadman/deadline
└─ EKS 1.36 (API authentication mode, Pod Identity, no OIDC / KMS / control plane logs)
   ├─ Managed node group system   1 × t3.large On-Demand   CoreDNS, Karpenter, TF controller, GreptimeDB, alertmanager
   ├─ Karpenter 1.14.1 (featureGates.nodeOverlay=true)
   │   ├─ NodePool tf-cpu   weight 100  general-purpose CPU nodes (non-GPU test Pods, remote mode client, load generator)  limits cpu=4
   │   ├─ NodePool tf-mock  weight 50   "fake GPU" nodes for rehearsal mode (CPU instance types + mock accelerator)        limits cpu=6
   │   ├─ NodePool tf-gpu   weight 10   g4dn.xlarge / g4dn.2xlarge / g5.xlarge / g6.xlarge, spot first with fallback to on-demand, limits nvidia.com/gpu=4
   │   ├─ EC2NodeClass tf-gpu (80Gi gp3 @300MiB/s) / tf-cpu (30Gi gp3), AMI alias al2023@latest (resolves automatically to the NVIDIA variant for GPU instance types)
   │   └─ NodeOverlay tf-e2e-no-index (weight 100): zeroes the tensor-fusion.ai/index* capacity of non-TF NodePools, see "Pitfalls hit"
   ├─ NVIDIA k8s-device-plugin (runs only on real GPU nodes; a path B NodeClaim needs nvidia.com/gpu before it becomes Initialized)
   └─ TensorFusion Helm chart (stage 3, installed separately, image tags passed explicitly every time)
       ├─ GPUPool tensor-fusion-shared (AutoSelect, selects nodes with e2e.tensor-fusion.ai/shared-pool=true = tf-gpu)
       └─ GPUPool tensor-fusion-mock (exists only after `make mock-install`, selects nodes with e2e.tensor-fusion.ai/mock-gpu=true = tf-mock)
```

Three stages:

| Stage | Contents | Executed by |
| --- | --- | --- |
| 1 | Dead-man switch, VPC, EKS, system node group, IAM, Pod Identity | Terraform resources |
| 2 | Karpenter, EC2NodeClass / NodePool, the NodeOverlay that prevents accidental scale-up, NVIDIA device plugin | `scripts/install.sh` (called by `terraform_data.install` at the end of apply; `make install` reruns it on its own) |
| 3 | hostPath PV for GreptimeDB, TensorFusion chart | `scripts/install-tf.sh` (called at the end of `make up`; `make tf-install` reruns it with parameters; `SKIP_TF=1 make up` skips it) |

The helm / kubernetes providers are not used: they make `terraform destroy` fail outright when the cluster is unreachable or half deleted. With scripts, the Terraform state contains only AWS resources.

## Version choices and rationale

| Component | Version | Rationale |
| --- | --- | --- |
| EKS | **1.36** | The operator embeds the kube-scheduler of `k8s.io/kubernetes v1.36.1` (`go.mod`). The Kubernetes version skew policy does not allow the scheduler to be newer than the apiserver, so the apiserver must be at least 1.36. 1.36 is also the current EKS default version, with standard support until 2027-08 (`aws eks describe-cluster-versions`, 2026-10-11) |
| Karpenter | **1.14.1** | Karpenter compatibility matrix (`website/content/en/docs/upgrading/compatibility.md`, 2026-10-11): K8s 1.34 needs ≥ 1.6, 1.35 needs ≥ 1.9, **1.36 needs ≥ 1.13**, and 1.37 only has 1.15.x. 1.14.1 is the latest patch of the 1.14 line (2026-08-21); 1.15.0 was released only the day before |
| Compatibility with the operator's dependency | `sigs.k8s.io/karpenter v1.8.2` | The operator only uses its API types. The CRDs of v1.8.2 and v1.14.1 were compared: NodeClaim / NodePool in `karpenter.sh/v1` only have additions (`Gte` / `Lte` operators, list-type markers, relaxed CEL) and no removed or changed fields; `NodeOverlay` is still `v1alpha1` and still behind the `nodeOverlay` feature gate, and its schema matches the chart template. Measured: a NodeClaim created by operator 2.17.23 was accepted by 1.14.1 and a node was launched |
| 1.35 + 1.9.2 from the earlier draft | Dropped | A valid combination in the matrix, but apiserver 1.35 is one minor version older than the embedded scheduler |
| terraform | 1.16.5 (`make tools`) | `.tools/terraform`, SHA256 verified on download |
| terraform-aws-modules/eks//modules/karpenter | ~> 21.29 | Only its IAM / Pod Identity / instance profile are used; the controller policy of 21.29.0 (2026-10-07) already includes `ec2:DescribePlacementGroups`, which 1.14 needs |
| Images | operator / hypervisor `2.17.23`, `vgpu-provider-nvidia:1.3.9` | The latest release on Docker Hub (2026-10-11). The chart's default tags are empty or behind, so `install-tf.sh` **passes the tags explicitly every time** |

## Cost safety (hard requirement)

1. **Tags**: all resources carry `Project=tf-e2e` (provider `default_tags`, the `tag_specifications` of the managed node group's launch template, the EC2NodeClass `tags`). Karpenter instances additionally have `eks:eks-cluster-name` and `karpenter.sh/nodeclaim`.
2. **Dead-man switch (does not depend on any local process)**:
   - An EventBridge rule invokes the Lambda `tf-e2e-deadman` every 5 minutes, and the Lambda reads the deadline from an SSM parameter.
   - After expiry: all managed node groups are scaled to 0 → all instances carrying this stack's tags are terminated (those from Karpenter and those created directly by TF path B both count) → orphaned EBS volumes are deleted.
   - Once expiry is exceeded by more than `deadman_cluster_grace` (default 6h): the node groups and the EKS control plane ($0.10/h) are deleted. The VPC, IAM, and Lambda that remain afterwards are not billed.
   - **Level-triggered** rather than a one-shot timer: as long as the deadline has passed, it cleans up again every 5 minutes, so instances relaunched by Karpenter or the ASG do not survive the next round.
   - The EKS cluster `depends_on` the dead-man switch: it is created before the first billable resource and destroyed after the last one.
   - `make up TTL=…` resets the deadline before apply; `make extend TTL=…` changes it at any time; `make deadman` shows the status and the most recent executions.
   - Measured (2026-10-12): twice, with the deadline set to 2 minutes later both times, after which only read-only polling was done locally.
     - First time (system node only): the execution 58s after expiry scaled the node group to 0; the graceful drain of the node group had not finished after more than 5 minutes, and the next execution terminated the instance directly. **All compute resources were gone 5m58s after expiry.**
     - Second time (system node + 1 Karpenter spot node): the execution 3m12s after expiry scaled the node group to 0 and terminated the Karpenter instance; Karpenter was still alive and immediately launched another one; the next execution (8m12s after expiry) terminated that one too. **Everything was gone 8m12s after expiry**, which is exactly why it is level-triggered.
     - The worst case is "expiry + two periods", about 10 minutes.
3. **`make cost`**: Cost Explorer month-to-date by the `Project` tag + a per-service breakdown for the whole account (CE has a delay of up to 24h, and the tag must first be discovered by AWS before it can be activated as a cost allocation tag; the script tries to activate it); plus a live estimate computed from the running resources (spot at the current spot price, on-demand from `../pricing/us-west-2.json`, EKS control plane, public IPv4, gp3). Each CE API call costs $0.01.
4. **`make down`**: records a cost snapshot before destroying, runs `terraform destroy` (retried once automatically on failure), and then runs `sweep`. `sweep` checks EKS, EC2 instances, launch templates, VPC / ENI / LB / SG / NAT, EBS volumes and snapshots, EIPs, IAM, Lambda, SSM, log groups, SQS, EventBridge, and Budgets by tag and name prefix, and exits with code 1 when there are leftovers. It does not depend on the Terraform state: `make sweep NAME=tf-e2e REGION=us-west-2`; `make sweep-delete` is the last resort.
5. NodePool `limits` (`cpu_limit`, `mock_cpu_limit`, `gpu_node_limit`) cap how much Karpenter can launch; path B NodeClaims are not subject to NodePool limits and rely on the GPUPool `maxResources` and the dead-man switch.

The only thing kept on purpose: `AWSServiceRoleForEC2Spot` (an account-level service-linked role, free; without it spot fails on a new account; `install.sh` creates it only if missing, and `make down` does not delete it).

## TensorFusion install parameters (stage 3)

All are read from environment variables, and image tags are always passed to helm explicitly:

| Variable | Meaning | Default |
| --- | --- | --- |
| `TF_CHART` | Chart source. Empty = this repository's `charts/tensor-fusion` (the branch under test); `tensor-fusion/tensor-fusion` = the published chart | Repository checkout |
| `TF_CHART_VERSION` | Version of the published chart | Latest |
| `TF_RELEASE_TAG` | Version of the operator and hypervisor (same tag for the same release) | `2.17.23` |
| `TF_OPERATOR_TAG` / `TF_OPERATOR_IMAGE` | Override the operator's tag / repository separately | Follows `TF_RELEASE_TAG` |
| `TF_HYPERVISOR_TAG` / `TF_HYPERVISOR_IMAGE` | Override the hypervisor separately | Follows `TF_RELEASE_TAG` |
| `TF_VGPU_PROVIDER_TAG` | Tag of `vgpu-provider-nvidia` | `1.3.9` |
| `TF_CLIENT_TAG` / `TF_WORKER_TAG` | remote client / worker | Chart default |
| `TF_AGENT_ID` / `TF_ENROLL_TOKEN` / `TF_CLOUD_ENDPOINT` | Console agent. Read only from the environment and passed to helm through a temporary values file with mode 0600; never on the command line, never echoed, never stored in the repository | Disabled |
| `TF_HELM_ARGS` | Arguments appended to `helm upgrade --install` | Empty |

```bash
make tf-install                                   # repository chart + released images
TF_RELEASE_TAG=2.17.24 make tf-install            # once the fixed images are out
TF_CHART=tensor-fusion/tensor-fusion TF_CHART_VERSION=1.8.8 make tf-install
TF_AGENT_ID=… TF_ENROLL_TOKEN=… make tf-install   # register with the console (values come from your shell, do not write them into a file)
make tf-uninstall                                 # the repository's scripts/uninstall.sh
```

## Rehearsal mode: exercising the control chain without a GPU

`make mock-install` turns the CPU nodes of the `tf-mock` NodePool (t3.medium, spot about $0.014/h) into TensorFusion GPU nodes with mock devices:

1. Put the example provider source in the repository (`provider/example`) and `e2e/mock/{mock_patch.h,build.sh}` into a ConfigMap;
2. Run a one-off Job on the system node that compiles it with the public `gcc:13` image and prints the artifact to the log as base64;
3. The script stores it as the ConfigMap `tf-mock-accelerator` (binaryData); the hypervisor of the mock pool gets an extra init container that copies it to `/run/tensor-fusion`;
4. Create a `ProviderConfig` (vendor `STUB`) and the GPUPool `mock` (`defaultVendor: STUB`, selects the mock nodes).

**No local Docker is needed, and no test image has to be built or pushed.** The changes to the example provider are only three lines (applied with sed at build time, failing if the pattern is not found): the device UUID includes the node's short hostname (the original hardcodes `example-device-0..3`, and since GPU CRs are cluster-scoped objects, two mock nodes would overwrite each other); the device count is read from `TF_MOCK_GPU_COUNT` (default 1, matching the single-GPU instance types of the real GPU NodePool). Each mock device reports 16 GiB / 312 TFLOPS.

For what a rehearsal can and cannot prove, see "Execution modes" in `../README.md`.

## Measured durations and cost (us-west-2)

| Round | `make up` | Of which | `make down` | `sweep` after destroy |
| --- | --- | --- | --- | --- |
| 1 | about 16.5 minutes, in two attempts | EKS control plane 10m28s; system node group 2m12s; stage 2 (Karpenter, NodePool) 53s; TensorFusion 109s. The first apply failed at the final install hook: I modified `scripts/install.sh` while apply was in progress, so `filesha256` was inconsistent before and after. Rerunning `make up` converged in 53s | 6m05s (destroy 332s) | Empty |
| 2 | **13m43s**, succeeded on the first try | Stages 1 + 2 together 715s (EKS 9m03s, node group 1m50s); TensorFusion 105s (helm 96s) | 4m54s (destroy 263s) | 2 Karpenter launch templates left (not billed). Cause: during the dead-man switch test Karpenter was terminated between creating the LT and launching the instance, and the destroy hook looked for LTs by the old tag `karpenter.k8s.aws/cluster`, while 1.14 uses `eks:eks-cluster-name`. Fixed; after the fixed hook ran against these two real leftovers, `sweep` was empty |
| 3 | **15m21s**, succeeded on the first try | Stages 1 + 2 together 812s (EKS 10m36s, node group 1m51s); TensorFusion 105s (helm 98s) | **12m07s** (destroy 695s) | Empty |

The 5–6 minutes of `make down` (rounds 1 and 2) were measured after the dead-man switch had already removed the nodes; with nodes still present (round 3) it has to wait for the node group to drain and be deleted, about 12 minutes.

Karpenter (general-purpose NodePool, c5.large spot): Pending → NodeClaim 6s → node ready 39s → Pod Running 39s; the node disappears 87s after it is emptied (`consolidateAfter=30s`).
mock GPU node (t3.medium spot): Pending → NodeClaim 6s → mock GPU registered 51s → Pod Running 56s.

Cost: the three rounds together are estimated at about **$1.1** (EKS control plane about 4.2h × $0.10; t3.large about 3.5h; Karpenter's small spot nodes about 5.5 node-hours in total; about 10 Cost Explorer API calls). Cost Explorer lags, and at the time of writing month-to-date showed $0.04; the `Project` tag can only be activated as a cost allocation tag after about 24 hours. The idle cluster measured **$0.216/h** (including 1 mock node), and $0.19/h with no Karpenter nodes.

## Pitfalls hit (defects of the earlier draft)

| # | Defect | Resolution |
| --- | --- | --- |
| 1 | The versions chosen were EKS 1.35 + Karpenter 1.9.2, without considering that the embedded scheduler is 1.36 | Changed to 1.36 + 1.14.1, rationale above |
| 2 | Default `gpu_static_count=1`: GPU nodes were launched by a managed node group. TF's expander only clones nodes whose "owner is a Karpenter NodeClaim", so such a node cannot serve as a scale-up template; with a quota of 0 it also makes apply hang for 25 minutes | Default 0, all GPUs come from Karpenter; the variable is kept for NEG-10 |
| 3 | No timeout self-destruct at all; the README said "it is recommended to have an agent run make down on a timer" | Added the Lambda dead-man switch, `make extend`, and `make deadman` |
| 4 | No `make cost` | Added |
| 5 | The instances, volumes, and ENIs of the managed node group did not carry the `Project` tag (`default_tags` do not propagate to instances launched by the ASG) | The node group now uses a launch template with `tag_specifications` |
| 6 | Only a GPU NodePool existed: ordinary Pods that did not fit on the system node made Karpenter launch GPU nodes | Added the `tf-cpu` NodePool with a higher weight than GPU |
| 7 | **Adding the CPU NodePool exposed a product problem**: the chart's NodeOverlay applies to all NodePools, so a Pending TF Pod makes Karpenter launch a CPU node, and that node never becomes Initialized and is never reclaimed (`../DOCS-GAPS.md` G-01) | Added the higher-weight `NodeOverlay/tf-e2e-no-index`; `KRP-A15` reproduces this problem specifically |
| 8 | The EBS CSI addon was installed for a single GreptimeDB PVC (one more addon, one IAM role, about 1.5 minutes of up/down time, and the class of leftovers where "the volume outlives the cluster") | Removed. The PVC has no storageClassName, so pre-creating a hostPath PV to bind to it is enough |
| 9 | A new account has no `AWSServiceRoleForEC2Spot`, so spot requests fail | `install.sh` creates it only if missing |
| 10 | The TensorFusion install and the infrastructure were tied together in one script, and image tags relied on the chart defaults (behind / empty) | Split into stage 3, tags passed explicitly |
| 11 | AZs were chosen by the first GPU instance type only | AZs are sorted by "number of GPU instance types offered" and the top two are taken |
| 12 | With NodePool `consolidateAfter` set to 1m and to 5m, nodes added by the TF expander were both seen being reclaimed by Karpenter as empty nodes before the Pod landed on them (the Pod has to wait for the scheduler's 5-minute fallback retry before it is scheduled again, see `../DOCS-GAPS.md` G-16) | Default 10m; `KRP-A02` records the "node ready → Pod bound" duration and `KRP-A16` asserts on it (known-fail) |
| 13 | The load balancer item in `sweep` gave a false positive when everything was empty (the string was a single space) | Fixed |
| 15 | The destroy hook and `sweep` looked for Karpenter's launch templates by the `karpenter.k8s.aws/cluster` tag; Karpenter 1.14 applies `eks:eks-cluster-name` / `kubernetes.io/cluster/<name>`, so LTs left behind when Karpenter was killed midway could not be deleted (2 left after the round 2 destroy) | Both the old and new tags are checked, and only those whose name starts with `karpenter.k8s.aws/` are deleted |
| 14 | Modifying `scripts/install.sh` while `terraform apply` is in progress makes `filesha256` inconsistent before and after, and apply fails | Do not change the scripts during a run; after a failure, rerunning `make up` converges |

## Quotas

The unit is vCPU; view them with `make quotas`:

| Quota | Code | Needed by this stack |
| --- | --- | --- |
| Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) | `L-1216C47A` | 8 is enough: system 2 + tf-cpu ≤ 4 (when on-demand); spot is counted separately |
| All Standard Spot Instance Requests | `L-34B43A08` | 8 is enough for rehearsal: at most 3 t3.medium mock nodes + 1 CPU node |
| Running On-Demand G and VT | `L-DB2E81BA` | The real GPU tiers need ≥ 8 (2 g4dn.xlarge); 16 is recommended for a full pass |
| All G and VT Spot Instance Requests | `L-3819A6DF` | Same as above when using spot; when it is 0 Karpenter falls back to on-demand |

## Remote state

The default is local state (`terraform.tfstate`, git-ignored; it contains the account ID, do not commit it). When several people share it, add an S3 backend in `versions.tf` (`use_lockfile = true`). If the local state is lost, clean up with `make sweep-delete`.
