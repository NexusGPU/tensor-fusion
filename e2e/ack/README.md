# Disposable ACK + Karpenter environment for TensorFusion e2e

An Alibaba Cloud ACK managed cluster + Karpenter's Alibaba Cloud provider, with real T4s as GPU nodes. It serves the same purpose as `../terraform` (EKS): a repeatable cloud environment for `../suite`. This is the first environment in which the GPU tier runs on real GPUs.

- Test conclusions, durations and cost: [`TEST-SUMMARY.md`](TEST-SUMMARY.md)
- Manual test steps: [`MANUAL-TEST-CASES.md`](MANUAL-TEST-CASES.md)

## Quick start

```bash
# Requires: aliyun CLI (already set up with aliyun configure), kubectl, helm 3, jq, curl; building the patched image also needs docker, git, go
cd e2e
make ack-up                              # VPC + cluster + Karpenter + TensorFusion + cost guard, about 10 minutes
export KUBECONFIG=$PWD/ack/.work/kubeconfig
make ack-test TIER=smoke                 # any tier: smoke | core | isolation | karpenter-finops | perf-quick
make ack-test TIER=core FOCUS='ISO-01'   # a single case
make ack-sweep                           # list the cloud resources of this environment that currently exist
make ack-down                            # tear down and check for leftovers
```

All parameters are in `lib.sh` and are overridden with environment variables (region, instance types, NodePool limits, versions). Credentials come only from the aliyun CLI profile and are not written into the repository; Karpenter inside the cluster assumes a RAM role through RRSA, and there is no AccessKey in the cluster.

## Components

```
VPC 10.43.0.0/16 (2 vSwitches / 2 zones, NAT gateway + EIP for egress)
└─ ACK managed Basic edition, Kubernetes 1.36 (Flannel, RRSA, public API endpoint)
   ├─ Node pool system   1 × ecs.g7.xlarge, Alibaba Cloud Linux 3 container-optimized (cgroup v2)
   ├─ Karpenter: AliyunContainerService/karpenter-provider-alibabacloud 0.2.5 (core 1.8, NodeOverlay feature gate on)
   │   ├─ NodePool tf-cpu   weight 100  ecs.c7.large / ecs.g7.large, limits cpu=8, reclaimed after 30 seconds
   │   ├─ NodePool tf-gpu   weight 10   ecs.gn6i-c4g1.xlarge / c8g1.2xlarge (one T4 each), on-demand, limits nvidia.com/gpu=3, reclaimed after 2 minutes idle
   │   └─ ECSNodeClass tf-gpu / tf-cpu: the cluster's own node image, security group, worker RAM role
   ├─ ACK's built-in nvidia device plugin (GPU nodes report nvidia.com/gpu; the NodeClaims of path B rely on it to become Initialized)
   └─ TensorFusion chart (this repository's checkout); the default pool selects nodes with e2e.tensor-fusion.ai/shared-pool=true
```

| Script | What it does |
|---|---|
| `run.sh up` / `down` | Full bring-up / teardown; this is what `make ack-up` / `ack-down` call |
| `up.sh` | VPC, vSwitches, cluster, system node pool, kubeconfig |
| `install.sh` | Karpenter's RAM role, chart, ECSNodeClass, NodePool |
| `install-tf.sh` | TensorFusion chart, with the image tag passed explicitly; the console agent's credentials go from environment variables into a Secret |
| `build-karpenter.sh` | Builds the provider with `patches/` and imports it onto the system node |
| `load-image.sh` | Imports a local image (or a `docker save` archive) into containerd on the system node |
| `guard.sh` | Cost guard |
| `down.sh` | Deletes all resources by name and tag, then checks for leftovers |

`run.sh up` reads three local inputs under `ack/.work/` (none of them is committed): `images/*.tar` (development builds that no image registry hosts), `stack.env` (e.g. `KARPENTER_ALIBABA_IMAGE=`, `TF_OPERATOR_TAG=`), and `agent.env` (`TF_AGENT_ID` / `TF_ENROLL_TOKEN` of the console environment, mode 600).

The vGPU worker license, which hard isolation and remote mode need, is read from `TF_LICENSE` / `TF_LICENSE_SIGN` in the environment or from the file `TF_LICENSE_FILE` (default `~/.config/tensor-fusion/license.env`, mode 600; set another path in `stack.env`). `install-tf.sh` stores it in the Secret `tensor-fusion-license` and points the worker at it; the suite puts the same Secret into every case namespace. Without a license the stack is installed as before and the cases that need one are skipped with that reason. See `tools/tf-license.sh`.

## Cost safety

- All resources carry the tag `Project=tf-e2e-ack`.
- NodePool `limits`: at most 3 GPUs, and at most 8 cores in the CPU pool.
- `guard.sh` (started in the background by `run.sh up`, log in `ack/.work/guard.log`) does the following every 2 minutes: it terminates ECS instances that have no NodeClaim and do not belong to a node pool (only once they have existed for more than 5 minutes); when there are more than 4 GPU instances, it sets the limits of all NodePools to 0 and deletes the GPUNodeClaims of path B. Lifting the freeze is a human decision (re-run `install.sh`).
- The cost guard is a local process and does not work while the machine is asleep. **This environment has no cloud-side timeout self-destruct like the EKS stack has**; run `make ack-down` when you are done.
- Idle cost: system node + NAT gateway + SLB + EIP, about $0.25/h; each T4 node is about $1.09/h (us-east-1 on-demand list price, see `../pricing/alibaba-us-east-1.json`).

## Pitfalls encountered

| # | Symptom | Handling |
|---|---|---|
| 1 | `CreateCluster` fails with `ErrKubernetesAuditRoleNotAttach` / `AddonRoleNotAuthorized`: the account has not authorized ACK's service roles | A one-time account-level authorization (RAM console, or follow the link in the error). It is not part of this stack and `down.sh` does not delete it |
| 2 | Node pool creation fails with `CgroupV1NotSupport`: Kubernetes ≥ 1.35 does not accept cgroup v1 nodes, and the `AliyunLinux3` image is cgroup v1 | Use `AliyunLinux3ContainerOptimized` as the node image; the node pool is created separately, so on failure only the node pool has to be replaced |
| 3 | Karpenter panics at startup: `unknown feature gate "NodeOverlay"` | The chart's `settings.featureGates` also becomes the `--feature-gates` flag, which the provider parses with its own list. Set it only through the `FEATURE_GATES` environment variable |
| 4 | `RunInstances` fails with `InvalidUser.PassRoleForbidden` | `ram:PassRole` scoped by role ARN is not accepted; use `Resource: *` with the condition `acs:Service=ecs.aliyuncs.com` |
| 5 | One NodeClaim corresponds to two ECS instances | When `RunInstances` succeeds and the following `DescribeInstances` fails, the provider retries directly and does not reclaim the first instance. The trigger is replacing the RAM policy while running; the script now updates the policy version in place, and the cost guard is the backstop |
| 6 | **One Pending TensorFusion Pod launches 4 GPU nodes** | The NodeClaim capacity returned by the provider does not include the resources declared by the NodeOverlay, so on every round Karpenter concludes that the in-flight node cannot fit the Pod. The patch in `patches/` adds the overlay capacity; `build-karpenter.sh` builds and imports it |
| 7 | The ACK console flags Karpenter's nodes as "stray nodes" | Expected behavior: the provider registers nodes with the cluster join script and does not go through an ACK node pool. Managed upgrades and self-healing of node pools do not apply to these nodes |
| 8 | `KRP-C01` never sees the CPU NodePool become empty | ACK's CoreDNS has 2 replicas with anti-affinity, and the second replica occupies Karpenter's CPU node. `install.sh` scales it down to 1 replica |
| 9 | Development images cannot be pushed to any image registry | The controller and Karpenter are both pinned to the system node, and `load-image.sh` imports directly into that node's containerd. The hypervisor runs on nodes launched by Karpenter, so this approach does not apply to it; it needs an image registry |

## Differences from the EKS stack

| | EKS (`../terraform`) | ACK (this directory) |
|---|---|---|
| Bring-up / teardown | Terraform | shell + aliyun CLI |
| Timeout self-destruct | Lambda, cloud-side | None; the local cost guard only handles instance leaks and limits |
| Capacity type | spot first, falling back to on-demand | on-demand only |
| GPU driver | NVIDIA AMI | Installed by the ACK join script on first boot; node ready in about 4–5 minutes |
| Storage | No CSI, hostPath PV | Same (ACK installs CSI by default, but there is no default StorageClass) |
| NodeOverlay scope | `tf-e2e-no-index` zeroes out non-TF NodePools | The chart's `karpenterNodeOverlay.requirements` scopes it directly to `karpenter.alibabacloud.com/instance-gpu-manufacturer=nvidia` |
