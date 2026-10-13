# TensorFusion Full-Stack Context Index

This document explains where this repository sits in the overall TensorFusion stack: which components it deploys, which repository builds each of them, and how they communicate with the console (Cloud Console). Its purpose is to let an engineer or AI agent who has only this repository open find the rest of the context.

- Compiled on 2026-10-11. Content about anything other than this repository comes from a read-only survey of the neighbouring repositories; items marked "inferred" have no direct evidence.
- This is a public repository. Do **not** write any keys, tokens, connection strings, internal addresses, or cloud account IDs here, and do not write exploitation details of unfixed security issues. Put that kind of content in local, uncommitted `tmp.*` files (`.gitignore` already ignores `tmp*`).

## 1. Component overview

```
                         ┌──────────────────────────────────────────────┐
                         │ Cloud Console  app.tensor-fusion.ai          │
                         │ (tensor-fusion-cloud, private)               │
                         └───────────────▲──────────────────────────────┘
                                         │ WebSocket (the cluster dials out, no inbound access needed)
┌────────────────────────────────────────┼─────────────────────────────────────────────────┐
│ User Kubernetes cluster                │                                                 │
│  ┌─────────────────────────────────────┴──────────────────────────────┐                  │
│  │ controller Deployment (this repository, cmd/main.go)               │                  │
│  │  ├ controller: CRD controllers + Pod webhook + embedded scheduler  │                  │
│  │  ├ vector: metrics and logs → GreptimeDB                           │                  │
│  │  └ cluster-agent (optional sidecar, from the cloud repository)     │                  │
│  └────────────────────────────────────────────────────────────────────┘                  │
│  ┌─────────────────────────────────┐   ┌───────────────────────────────────────────────┐ │
│  │ GreptimeDB (tf_* metric tables) │   │ Karpenter / cloud provider API (node scaling) │ │
│  └─────────────────────────────────┘   └───────────────────────────────────────────────┘ │
│                                                                                          │
│  Each GPU node:                                                                          │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ hypervisor DaemonSet Pod (this repository, cmd/hypervisor, Go, CGO disabled)       │  │
│  │  ├ loads the vendor accelerator library through purego (vgpu-provider)             │  │
│  │  ├ device plugin / kubelet checkpoint integration                                  │  │
│  │  └ writes each Pod's shared memory limit file /run/tensor-fusion/shm/<ns>/<pod>/   │  │
│  ├────────────────────────────────────────────────────────────────────────────────────┤  │
│  │ Workload Pod (injected by the webhook)                                             │  │
│  │  ├ soft: /etc/ld.so.preload → libcuda_limiter.so (vgpu-provider)                   │  │
│  │  ├ hard / sidecar: client library + worker sidecar (tensor-fusion-core)            │  │
│  │  └ remote: client library ──TensorFusionConnection──▶ standalone worker Pod        │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. This repository

Go module `github.com/NexusGPU/tensor-fusion` (the repository on GitHub has the same name, `NexusGPU/tensor-fusion`; the old name `tensor-fusion-operator` still redirects). Kubebuilder layout, depending on `k8s.io/kubernetes v1.36.1` (vendored, then patched), `controller-runtime v0.24.0`, and `sigs.k8s.io/karpenter v1.8.2`.

### 2.1 Directory index

| Path | Content |
| --- | --- |
| `cmd/main.go` | controller entry point: controllers, webhook, scheduler, and HTTP server all run in one process |
| `cmd/sched/` | Assembly of the embedded kube-scheduler |
| `cmd/hypervisor/` | Node hypervisor entry point; `shm_init/` is the subcommand that mounts shm |
| `cmd/hypervisor-tui/` | Terminal debugging UI for the hypervisor |
| `api/v1/` | CRD types, a separate Go module (`github.com/NexusGPU/tensor-fusion/api`) |
| `internal/controller/` | Controllers for each CRD; for node scaling see `gpupool_node_provision.go`, `gpunodeclaim_controller.go`, `gpupool_compaction_controller.go`, `gpupool_defrag.go` |
| `internal/webhook/v1/` | Pod mutating webhook; annotation parsing is in `tf_parser.go` |
| `internal/utils/compose*.go` | Generates the Pod fragments injected for the hypervisor / worker / client / limiter (`compose.go`, `compose_client.go`, `compose_worker.go`, `compose_limiter.go`); the place where all images, environment variables, and mounts are assembled |
| `internal/scheduler/` | Scheduler plugins: `gpuresources` (GPU resource filtering and reservation), `gputopo` (topology), `expander` (scale-up triggered by Pending Pods) |
| `internal/gpuallocator/` | GPU allocation state machine, shared by the scheduler and the controllers |
| `internal/cloudprovider/` | Node provisioning: `karpenter/`, `aws/`, `alibaba/`, `pricing/` |
| `internal/autoscaler/` | Vertical autoscaling (limit recommendation and in-place adjustment) |
| `internal/quota/`, `internal/gang/` | Namespace quota, gang scheduling |
| `internal/metrics/`, `internal/alert/` | Metric encoding and writing (through vector to GreptimeDB) and alert evaluation |
| `internal/server/` | HTTP API of the controller, including `/api/connection`, which the client uses to fetch connection information |
| `internal/provider/` | ProviderConfig management: resolves images and hardware information per vendor |
| `pkg/hypervisor/` | hypervisor implementation, see 2.3 |
| `pkg/constants/` | Annotation names, environment variable names, vendor mapping (`vendors.go`) |
| `provider/` | C ABI header files for the accelerator / limiter and an example implementation (`make build-provider`) |
| `charts/tensor-fusion/` | Helm chart; `values-cn.yaml`, `values-production.yaml`, and `values-multi-vendor.yaml` are overlay files |
| `config/` | kustomize manifests, CRDs, samples |
| `patches/`, `scripts/patch-scheduler.sh` | Patches to the vendored kube-scheduler, applied by `make vendor` |
| `test/e2e/` (Kind), `test/sched/` (scheduler integration and benchmarks) | In-repository tests |
| `e2e/` | End-to-end suite (`suite/`) and the environments it runs on: AWS EKS (`terraform/`), Alibaba Cloud ACK with real GPUs (`ack/`), KWOK (`kwok/`); recorded performance results in `perf-history/` |
| `docs/`, `design/` | Design and operations documents, see section 6 |

### 2.2 CRDs (`tensor-fusion.ai/v1`)

`TensorFusionCluster` → `GPUPool` → `GPUNode` → `GPU` is the resource hierarchy; `GPUNodeClaim` and `GPUNodeClass` are used for node provisioning; `TensorFusionWorkload`, `WorkloadProfile`, and `TensorFusionConnection` describe workloads; `SchedulingConfigTemplate`, `GPUResourceQuota`, and `ProviderConfig` are policy and vendor configuration.

Key enums:
- Isolation mode (`api/v1/workloadprofile_types.go`): `shared`, `soft`, `hard`, `partitioned`.
- Node provisioning mode of a pool (`api/v1/gpupool_types.go`): `Provisioned`, `AutoSelect`, `Karpenter`.
- QoS: `low`, `medium`, `high`, `critical`.

### 2.3 hypervisor (`pkg/hypervisor/`)

| Subdirectory | Responsibility |
| --- | --- |
| `device/` | Calls the vendor accelerator library through purego: device discovery, metrics, partitioning (MIG) |
| `backend/kubernetes/` | device plugin, Pod cache, kubelet checkpoint, reporting of GPU / GPUNode CRs |
| `backend/single_node/` | Non-Kubernetes single-node mode with file-based state; imported as a library by `gpu-go` |
| `worker/` | worker lifecycle and device allocation; `state/` is the shared memory layout of the soft limiter; `computing/quota_controller.go` is the compute throttling control loop |
| `server/` | Node HTTP API; `handlers/legacy.go` keeps the interfaces from the vgpu.rs era |
| `metrics/`, `tui/`, `framework/`, `api/` | Metrics, debugging UI, interface definitions |

`pkg/hypervisor` and `api/v1` are depended on by `gpu-go` as Go libraries; changing the exported interfaces of either is equivalent to changing a public API.

### 2.4 Two Karpenter-compatible paths

The two are mutually exclusive and are selected by the GPUPool configuration:
- **`AutoSelect` + Karpenter NodePool** (chart default): GPU nodes are managed by Karpenter; when `internal/scheduler/expander` finds Pending Pods it creates Karpenter `NodeClaim`s directly; on scale-down it only labels the node with `tensor-fusion.ai/should-delete=true` and leaves the actual reclamation to Karpenter consolidation.
- **`provisioningMode: Karpenter`**: TensorFusion creates `GPUNodeClaim`s according to the pool's capacity configuration and then generates a `NodeClaim` of the same name (carrying `karpenter.sh/do-not-disrupt`); reclamation is handled by TensorFusion's own compaction.

Code entry points: `internal/cloudprovider/karpenter/`, `internal/controller/gpunodeclaim_controller.go`, `charts/tensor-fusion/templates/node-overlay.yaml`.

### 2.5 Build, test, release

- `make build` / `make build-hypervisor` / `make build-provider`; `make vendor` re-applies the scheduler patches.
- `make test` (envtest; the coverage threshold is in the Makefile), `make ut F=<file>`, `make ut-sched`, `make test-e2e` (Kind), `make lint`.
- `.github/workflows/release.yml` publishes both the operator and hypervisor images (with the same version number); `release-helm.yml` publishes the chart (tags look like `tensor-fusion-helm-<ver>`); `release-api.yml` publishes the `api/` module.

## 3. Component origins: who builds what

| Image / artifact | Referenced in | Source repository |
| --- | --- | --- |
| `tensorfusion/tensor-fusion-operator` | `values.yaml` `controller.image` | This repository, `dockerfile/operator.Dockerfile` |
| `tensorfusion/tensor-fusion-hypervisor` (2.x) | `values.yaml` `cluster.hypervisorImage` | This repository, `dockerfile/hypervisor.Dockerfile` |
| `tensorfusion/vgpu-provider-nvidia` | `values.yaml` `providerConfigs.nvidia.images.middleware` | `vgpu-provider` |
| `tensorfusion/tensor-fusion-client`, `tensor-fusion-worker` (v2.x) | `values.yaml` `providerConfigs.nvidia.images.remoteClient` / `remoteWorker` | `tensor-fusion-core` (private) |
| `tensorfusion/tensor-fusion-agent` | `values.yaml` `agent.image` | `agent/` in `tensor-fusion-cloud` |
| vector, GreptimeDB, alertmanager, kube-webhook-certgen | `values.yaml` | Third party |
| `libaccelerator_example.so` | Default value in `cmd/hypervisor/main.go` | `provider/` in this repository |

The mapping from vendor to library file is in `pkg/constants/vendors.go`:

| Vendor | accelerator library | soft limiter library | Source |
| --- | --- | --- | --- |
| NVIDIA | `libaccelerator_nvidia.so` | `libcuda_limiter.so` | `vgpu-provider` |
| AMD | `libaccelerator_amd.so` | None | `vgpu-provider` (skeleton only) |
| Ascend | `libaccelerator_ascend.so` | `libascend_limiter.so` | `vgpu-provider-internal` (private) |
| MooreThreads | `libaccelerator_musa.so` | `libmusa_limiter.so` | Same as above |
| PPU | `libaccelerator_ppu.so` | `libppu_limiter.so` | Same as above |

`/build/*` in the middleware image has three consumers: it is copied into the hypervisor image at build time; the `init-runtime` container of the hypervisor Pod copies it to `/run/tensor-fusion` on the host; and the `init-soft-limiter` container of the workload Pod copies it to an emptyDir for preload.

## 4. Related repositories

For local development these repositories are usually placed next to this one in the same parent directory. Repositories marked "private" are not accessible to external readers; this document only describes their interface relationship with this repository and not their internal implementation. The images they publish are public.

### 4.1 Data plane

| Repository | Role | Language | Status | Relationship to this repository |
| --- | --- | --- | --- | --- |
| `vgpu-provider` (private) | Per-vendor accelerator libraries, the NVIDIA soft limiter, the `nvidia-smi` wrapper script | C/C++ | Active | `provider/accelerator.h` and `provider/limiter.h` are manual copies, not a submodule; the shared memory layout is synchronised by hand across three places (see 5.1) |
| `vgpu.rs` | The original Rust hypervisor + CUDA limiter | Rust | Its hypervisor has been replaced by the Go implementation in this repository (2026-01); it still publishes `bootstrap`, `libadd_path.so`, and the Rust limiter inside the worker image | The shared memory layout, the `/api/v1/pod` and `/api/v1/process` interfaces, and the ERL parameters follow its definitions |
| `tensor-fusion-core` (private) | Client library and worker for API remoting, the current v2.x | C/C++ | Active | The `client` / `worker` images; reads environment variables such as `TENSOR_FUSION_OPERATOR_GET_CONNECTION_URL`, `TF_CUDA_SM_PERCENT_LIMIT`, and `TF_GPU_MEMORY_LIMIT` |
| `tensor-fusion-worker`, `tensor-fusion-client`, `tensor-fusion-common`, `TensorFusion` | The v1-era client / worker and the earlier monolithic repository | C/C++ | Replaced by `tensor-fusion-core` (inferred) | No direct dependency; `compose.go` still recognises the old container names |
| `ngpu` | Experimental branch that runs the client and worker in the same process | C/C++ | Stalled (inferred) | Constants such as `TENSOR_FUSION_NGPU_PATH` may be read only by it |

Injection for the three workload shapes (`internal/utils/compose*.go`):
- Local soft: only the vgpu-provider limiter is injected, with no worker.
- Local hard or explicit sidecar: client init container + worker sidecar, communicating through `/dev/shm`.
- Remote: client init container + a standalone worker Pod, paired through `TensorFusionConnection`.
- `partitioned`, `shared`: nothing is injected for either.

### 4.2 Console and operations

| Repository | Role | Relationship to this repository |
| --- | --- | --- |
| `tensor-fusion-cloud` (private) | The console `app.tensor-fusion.ai` and the cluster-side `cluster-agent` | See 5.2. Its CRD types are generated from `config/crd/bases` in this repository; the TensorFusionCluster template and the Helm install command it provides must stay consistent with the CRDs and chart values of this repository |
| `auto-ops` (private) | Infrastructure code for deployment and testing, benchmark cases | Installs the chart from this repository; is the source of the benchmark and demo manifests; is responsible for syncing images to registries in mainland China for use by `values-cn.yaml` |
| `tensor-fusion-website` (private) | Official website, product documentation (the `tensor-fusion-engine` part), accounts and authentication | The documentation must be updated after changes to Helm values, metrics, annotations, or alert rules |
| `tensor-fusion-site` | The earlier public documentation site | Its CRD reference is generated from `config/crd/bases` in this repository |
| `gpu-go` | CLI and agent of the "GPU Go" product | Imports `pkg/hypervisor` and `api/v1` from this repository as libraries (single-node mode) |
| `benchmark` | Helm chart and results for the vGPU benchmark | Not consumed by this repository |

### 4.3 Third-party repositories commonly placed alongside

`kubernetes` (for reading the scheduler source and writing `patches/`), `kube-scheduler-simulator`, `alibaba-cloud-csi-driver`, `libnvidia-container`, `iceoryx2`, and others are clones kept for reference and are not part of this stack.

## 5. Cross-repository contracts

### 5.1 Shared memory for soft isolation

- Path: `/run/tensor-fusion/shm/<namespace>/<pod>/shm` on the host, mounted into the workload container with `subPathExpr`; inside the container `TF_SHM_PATH` points to it.
- The layout is defined in three places that must be changed together: `pkg/hypervisor/worker/state/shm_layout.go` in this repository (token operations are in `shm_erl.go`), the shared memory layout header in `vgpu-provider`, and `crates/utils/src/shared_memory/mod.rs` in `vgpu.rs`. The layout has no magic or version field.
- The hypervisor writes the limits, heartbeat, GPU memory usage, and token bucket parameters; the limiter reads them in-process and deducts tokens. The control loop driver is in `pkg/hypervisor/worker/computing/quota_controller.go`, rate control in `quota_rate.go`, metering and calibration in `quota_meter.go`, attribution in `quota_attribution.go`, refill in `quota_refill.go`, and configuration in `quota_config.go`.
- The limit fields `up_limit` and `mem_limit` can be rewritten in place while the mapping is alive (in-place adjustment by the autoscaler, changes to the ProviderConfig hardware table). There are three kinds of write occasions: when the mapping is opened or reopened; on the two events of worker update and device capacity update; and on every round of the 500 ms sync loop (when the value has not changed, nothing is written and no memory is allocated). Every write re-derives the value inside the same critical section from the worker and device capacity at that moment, so the order in which events arrive does not affect the result, and an inconsistency from any cause is corrected within one sync period. The limiter must read from the mapping every time and must not read only once at startup.
- The two beats of the token bucket: every 500 ms the control loop adjusts `erl_token_refill_rate` and `erl_token_capacity` according to utilization; every 10 ms (`erlRefillInterval`) the refill loop issues tokens at the current rate, so that a throttled process slows down evenly instead of running for a stretch and stopping for a stretch in each control cycle. The bucket capacity is `rate × 0.1 s` (`burstWindow`): an idle worker accumulates at most the amount of work equal to 0.1 seconds of its quota and spends it all at once when it resumes.
- Burst window W (the full rules are in the comment above `burstWindow` in `quota_config.go`): W is the maximum number of seconds, measured at its own refill rate, that a worker can accumulate while idle; bucket capacity = `rate × W`. Order of precedence: the Pod annotation `tensor-fusion.ai/compute-burst-window` (Go duration, such as `500ms` or `2s`; it can be changed in place on a running Pod, takes effect within one control cycle, and needs no restart); otherwise `elasticRateLimitParameters.burstWindow` of the pool-level `SchedulingConfigTemplate` (seconds, as a string; read at hypervisor startup through `TF_HYPERVISOR_SCHEDULING_CONFIG`, not included in the template hash, so the hypervisor Pods must be recreated after a change for it to take effect); otherwise the default of 0.1 s compiled into the hypervisor. The range is 0 to 10 s (`constants.ComputeBurstWindowMax`); the bucket always keeps at least two refill intervals (20 ms) worth, so W = 0 means "accumulate almost nothing". The webhook rejects malformed, negative, or over-limit values when the Pod is created; the hypervisor validates the annotation once more when it is changed afterwards (out-of-range values are clamped to the nearest bound, unparseable values are treated as unset, and each case is logged once). The long-term quota does not change with W: a burst spends only the tokens the worker itself accumulated at its quota rate, the rate is not raised because of W, and the rate of an idle worker does not rise either; in any T seconds a worker gets at most `quota × (T + W)` of device time. Once its bucket is full, a worker with quota s that has the device to itself can run at full speed for about `s × W / (1 − s)` seconds (with W = 2 s: about 0.67 s at 25%, about 2 s at 50%) and then returns to even throttling; filling the bucket takes W seconds of idleness. The overdraft rule is unchanged (see `debtBound` in the next item). When W is increased, the bucket fills gradually at the rate; when W is decreased, the part above the new capacity is removed at the next refill and the rest is left untouched. Nothing needs to change on the limiter side: the capacity is just a number in the mapping.
- Token refill protocol (`SharedDeviceInfoV2.RefillERLTokens`; the layout is unchanged and only atomic operations on existing fields are used): read `erl_last_token_update` and compute the elapsed time on the same clock; advance the timestamp to the current moment with a CAS; only the party whose CAS succeeds adds `rate × elapsed time` to `erl_current_tokens`, without exceeding the capacity. The timestamp is never moved backwards; a timestamp far in the future (more than 5 s) is reset to the current moment and no tokens are issued; a single refill credits at most 5 s. Every writer that refills tokens must follow this: the two loops of the hypervisor, and a limiter process that refills by itself when it finds that nobody is refilling (timestamp stale by about 100 ms or more). This way a stretch of time is credited only once, and nothing is lost or duplicated at a handover. The timestamp is in Unix seconds (f64). The shared definition of the arithmetic is the test vector file `pkg/hypervisor/worker/state/testdata/erl_refill_vectors.json` (executed on the Go side by `TestRefillVectors`; the tests on the limiter side should use the same file); the byte layout of the fields is pinned by the golden values in `soft_limiter_refill_test.go` in the same directory.
- Balance convention (shared debt): `erl_current_tokens` is an account shared by all processes in the Pod. When the balance is greater than 0 the limiter admits the work and deducts its full cost, so the balance can become negative (bounded overdraft); when the balance is not greater than 0 no work is admitted, and refilled tokens repay the debt first. The hypervisor treats any finite value as a legal balance; only NaN / Inf are considered invalid and handled as an empty bucket. The overdraft bound is `max(capacity, rate × 60 s × 100 / up_limit)` (the cost of the whole GPU running at full speed for 60 seconds, and not less than one bucket); the full rules are in the comment above `debtBound` in `pkg/hypervisor/worker/computing/quota_rate.go`, and the limiter and the hypervisor share this text: limiter charges never take the balance below `-bound`; the hypervisor only raises a balance that is below `-bound` up to `-bound`, and the bound does not shrink while the debt is outstanding (it is the largest value published since the balance was last non-negative). When the balance is not a number, has risen by more than the amount the hypervisor itself refilled (plus the amount the limiter self-refills at the published rate), or is below the overdraft bound, the bucket is treated as "unmetered" for the current cycle and the following 20 cycles; as long as one limited worker on a device is unmetered, the token attribution for the whole device is considered unknown in that cycle (rates are held and refill continues), and the device utilization is not distributed to the remaining workers. Clock rollback: when the refill loop finds the timestamp more than 50 ms in the future three times in a row, it restarts timing from that timestamp without issuing back tokens (`RestartERLRefillClock`). The hypervisor meters a worker's consumption as "previous balance + the amount it refilled in the interval − current balance"; a cycle in which it finds that someone else (limiter self-refill) advanced the timestamp is not metered.
- Heartbeat: `last_heartbeat` (Unix seconds) is written by a separate 500 ms loop (`pkg/hypervisor/worker/heartbeat.go`), which performs a single atomic write and goes through no provider call or API call; it is also written once immediately when the mapping is opened. The hypervisor writes no field on a normal exit: after the heartbeat expires naturally, the limiter refills by itself at the published rate, and the control loop after the restart takes over the published rate and leaves it unadjusted for 10 seconds.
- GPU memory sample time `pod_memory_sampled_at[i]` (V2 segment only): it occupies the first 128 bytes of the 512-byte padding at the tail of V2, one 8-byte word per device slot, located at `34992 + 8 × i` from the start of the segment (same index as `devices[i]`; the `pod_memory_used` of that slot is at `96 + 136 × i`). The segment size is still 35504 bytes, the other offsets are unchanged, the remaining padding starts at 35120, and there is still no version field. The value is the f64 bit pattern of Unix seconds, on the same clock and with the same encoding as `erl_last_token_update`; 0 means that no time was published. It is the wall clock read before the process query that produced this `pod_memory_used` began, not after the query and not the moment of the write; the limiter uses it to add "GPU memory allocated by this process after the sample" to the sampled value. There is no sequence field; the protocol is simply the order of three atomic writes per slot by the single writer, the hypervisor: write 0 to the time → write `pod_memory_used` → write the time. The limiter reads the time, the usage, and the time in that order, and treats (usage, time) as one sample only when the two times are equal and non-zero; otherwise it rereads, and after several attempts it uses the usage alone. On the same slot the time of a new sample is always different from the one published before and never goes backwards: when the clock has not advanced or has been rolled back, `nextafter(previous, +∞)` is published, where "previous" is taken from the mapping itself, so this still holds after a hypervisor restart; when the mapping already holds the same (usage, time) pair, it is not written again. When sampling fails, and when the device's process list cannot be read and the retention period has not yet ended, neither field is written. A usage that does not come from a query carries no time: for a newly created or recreated segment, for `SetPodMemoryUsed`, and for the write that records 0 after a device has been unreadable for 60 s in a row, the time is 0, and the limiter compares using the usage alone, the same as when this field did not exist. Compatibility: an old limiter treats these 128 bytes as padding and does not read them; an old hypervisor (including `vgpu.rs`, which currently does not have this field) always writes them as 0, and a new limiter that reads 0 works the original way, so both old/new combinations behave the same as before the extension and the two sides can be upgraded separately. The Go-side implementation is `SetPodMemorySample` / `GetPodMemorySample` in `soft_limiter_shm.go`, called from `publishPodMemoryUsed` in `controller_shm.go`; the offsets are pinned by `TestPodMemorySampleFieldsKeepTheLimiterLayout` in `soft_limiter_memory_sample_test.go`, and the limiter side has a corresponding layout test.
- Environment variables on the workload container side: `TF_ISOLATION_MODE`, `TF_SHM_PATH`, `HYPERVISOR_IP`, `HYPERVISOR_PORT`, `POD_NAME`, `POD_NAMESPACE`, `CONTAINER_NAME`, defined in `pkg/constants/env.go`.
- Design background: `docs/dynamic-gpu-isolation-design.md`, `docs/soft-isolation-preload.md`.

### 5.1.1 TFLOPS units and device capacity

- TensorFusion has a single definition of TFLOPS: the FP16 TFLOPS published by the vendor, taken from `hardwareMetadata[].fp16TFlops` in the ProviderConfig. Requests, limits, `status.capacity.tflops` on the GPU CR, and the GPUNode / GPUPool totals all use this unit.
- The `maxTflops` reported by the accelerator library is its own theoretical estimate (for NVIDIA it is FP32: CUDA core count × maximum SM clock × 2, which is 44.08 for the RTX 3090 and about 19.5 for the A100) and is not comparable with FP16 (71.2 for the 3090, 312 for the A100). Inside the hypervisor it is kept only as `DeviceInfo.ProviderEstimatedTflops` and serves as the capacity only when the hardware table does not contain the model.
- Inside the hypervisor each device has exactly one capacity, `DeviceInfo.TflopsCapacity`, which the device controller resolves once when a device is discovered / refreshed (`pkg/hypervisor/device/tflops_capacity.go`); when the hardware table changes it is resolved again and all subscribers are notified. GPU CR reporting, node totals, HTTP / TUI, and limit conversion all read this field, and nothing divides directly by the vendor library's estimate any more.
- The hardware table is delivered through the hypervisor Pod's environment variable `TF_PROVIDER_HARDWARE_METADATA`; after a ProviderConfig change the operator recreates the hypervisor Pods according to the pool's rolling policy. The single-node backend (`backend/single_node`, used by `gpu-go`) has no hardware table, so the capacity is the vendor library's estimate and the behaviour is the same as before.
- The capacity source is written to the annotation `tensor-fusion.ai/tflops-capacity-source` on the GPU CR: `provider-config` means FP16; `provider-estimate` means that the model is not in the hardware table and the value fell back to the vendor library's estimate. In the latter case the hypervisor log also warns once per device. If both values appear in one pool, the pool mixes two units and the missing models need to be added to the ProviderConfig.
- There is a single function for TFLOPS → percent, `utils.TflopsToComputePercent` (`internal/utils/resource.go`): it rounds up to a whole percent in the range 1–100. Both the scheduler (which in hard mode writes the `tensor-fusion.ai/effective-hard-sm-percent` annotation) and the hypervisor (`computeUpLimit`, which in soft mode writes `up_limit` in shared memory) call it, with the same capacity as the divisor. Hard mode prefers the annotation written by the scheduler; soft mode converts inside the hypervisor, so no second writer is needed when a limit is adjusted in place.

### 5.1.2 "Not provided" markers of the accelerator library

`provider/accelerator.h` is kept verbatim identical to `vgpu-provider/accelerator/accelerator.h`.

Optional features are negotiated through `AccelSetFeatures(requested, &enabled)`. This function was added later; the hypervisor resolves it as an optional symbol (`pkg/hypervisor/device/accelerator_unix.go` / `accelerator_windows.go`) and calls it once after every successful `AccelInit` and before polling starts (`AcceleratorInterface.initialize`); `AccelShutdown` clears the selection. The only place where the hypervisor declares which features it understands is `requestedFeatures` in `pkg/hypervisor/device/accelerator.go`, and the negotiation result is stored on `AcceleratorInterface.TelemetryMarkers()`, not in a global variable.

When `ACCEL_FEATURE_TELEMETRY_MARKERS` is not enabled (an old library does not have the symbol, the call fails, or the library does not know the bit): `computeUtilizationPercent` is always 0 and carries no information, the Go side marks every process as `ComputeUtilizationUnavailable`, and the throttling control loop uses device utilization and token consumption instead; the library also returns no markers. When it is enabled, content that cannot be read is returned as in-band markers, which are recognised only in `accelerator.go` and never flow out as numeric values:

- `ProcessInformation.computeUtilizationPercent == -1.0`: the device has no per-process utilization for this sample (0.0 means measured idle). The Go side converts it to `ComputeUtilizationUnavailable`; the workers on that device use device utilization for this beat, without treating it as 0 and without changing any remembered state.
- A row with an empty `processId`: the device's process list could not be read this time (this is not "no processes"). The Go side converts it to `ProcessInformation.ProcessListUnavailable`; the `pod_memory_used` of each worker on that device keeps its previous value, 0 is not written, and a log line is recorded when the state changes.
- `DeviceMetrics.utilizationPercent == UINT32_MAX`, `DeviceMetrics.memoryUsedBytes == UINT64_MAX`: the corresponding single field was not read and the other fields are valid. The Go side converts them to `GPUUsageMetrics.ComputeUnavailable` / `MemoryUnavailable`, and the corresponding field is omitted from the `tf_gpu_usage` row; when both are marked, the device cannot be opened and the sample is discarded.
- When the process list does not fit, the library returns `ACCEL_ERROR_RESOURCE_EXHAUSTED` and no rows; the Go side doubles the buffer and retries, up to 16384 rows, and the buffer is reused between calls.

With markers enabled, `AccelGetProcessInformation` takes about 4 ms per GPU that has processes. It is called only once per round by the 500 ms sync loop (`WorkerController.processInformation`), and other readers reuse that sample; the 10 ms refill loop and the heartbeat loop make no provider calls.

A single per-process utilization reading is very noisy (the driver attributes each short sample entirely to one process, and a process holding about 20% reads exactly 0 in about one sixth of its readings), and it describes the past second. The control loop does not use single readings directly: each kind of reading has its own calibration (`utilizationCalibration`) that learns "how much utilization one token per second corresponds to"; a worker's utilization is that ratio multiplied by the tokens it actually consumed, and the rate is taken directly as `target share ÷ ratio`. A new worker is first handled by the device-utilization calibration (accurate immediately) and gradually switches to the per-process calibration once enough per-process readings have accumulated.

### 5.2 Console and cluster

- `cluster-agent` is an optional sidecar in the controller Deployment and is enabled only when `agent.agentId` is set (`charts/tensor-fusion/templates/controller-deployment.yaml`). `agentId` has the form `<organization>:<environment name>` and is generated, together with `agent.enrollToken`, by the console's onboarding wizard.
- The agent's credentials do not appear in the Pod spec: the chart writes `agent.enrollToken` and `agent.agentId` into the Secret `<release>-agent`, and the sidecar reads them through `secretKeyRef`; alternatively `agent.existingSecret` (`name`, `key`, and the optional `agentIdKey`) can point to a Secret you manage yourself. When the chart-managed Secret changes, the `checksum/agent-secret` annotation on the Pod template changes with it and the controller Pod restarts; after rotating a self-managed Secret a manual restart is needed. `hack/test-chart-secrets.sh` (`make test-chart`) checks these rendering results with `helm template`.
- The image of the vector sidecar is pinned to `timberio/vector:0.56.0-alpine` (the same in both the controller and the hypervisor, each with a memory limit of 512Mi). With GreptimeDB Cloud, the vector configuration (ConfigMap) only references the environment variable `GREPTIME_PASSWORD`: for the controller's sidecar the chart injects it from the Secret `<release>-greptimedb-secret`; for the hypervisor's sidecar the operator injects it when assembling the Pod (`composeVectorContainer`), and the Secret name is passed in through the controller's environment variable `TSDB_PASSWORD_SECRET_NAME`.
- `total_scale_up_cnt` / `total_scale_down_cnt` in `tf_system_metrics` are cumulative values "since the operator process started"; they return to zero after an operator restart or leader change, and consumers should treat them as resettable counters. The time index column of the `tf_*` tables is `greptime_timestamp`; the operator's own SQL probes the column name per table, and user-written alert rules use `{{ .TimeColumn }}`.
- The agent actively connects to `agent.cloudEndpoint` (default `wss://app.tensor-fusion.ai`). All reads and writes from the console to the cluster are proxied over this WebSocket to the in-cluster API server, using the permissions of the controller's ServiceAccount, and the agent side has a path allowlist. Changes to `charts/tensor-fusion/templates/rbac.yaml` therefore change what the console can do.
- Metrics do not leave the cluster: the operator and the hypervisor write tables such as `tf_gpu_usage`, `tf_node_metrics`, `tf_pool_metrics`, `tf_worker_resources`, `tf_worker_usage`, and `tf_system_metrics` into the in-cluster GreptimeDB; when the console draws charts it runs the queries inside the cluster through the agent. Table schema changes require the console's metric queries to be updated in step.
- Resources the console operates on directly: all `tensor-fusion.ai/v1` CRDs, plus Pod, Deployment, Namespace, ConfigMap, and Event.
- The console's Helm install command uses `https://download.tensor-fusion.ai` as the chart repository, which is a mirror of `helm.tensor-fusion.ai`.

### 5.3 client and worker

- The client queries the worker address from the controller's `/api/connection` through `TENSOR_FUSION_OPERATOR_GET_CONNECTION_URL` (`internal/server/`).
- The `TensorFusionConnection` controller selects the worker with `internal/worker/worker.go`.
- Hard limits are delivered through environment variables of the worker container and enforced by the worker from `tensor-fusion-core`.

## 6. Index of in-repository documents

| Document | Content |
| --- | --- |
| `docs/installation-guide.md` | Installation and Helm values |
| `docs/cross-vendor-test-matrix.md` | Test matrix of vendors × isolation modes |
| `docs/dynamic-gpu-isolation-design.md` | Design of dynamic isolation modes |
| `docs/soft-isolation-preload.md` | Preload mechanism of the soft limiter |
| `docs/v1-to-v2-upgrade-best-practice.md` | Upgrading from v1 to v2 |
| `design/gputopo_scheduler_design_cn.md` | GPU topology scheduling design |
| `design/namespaced_gpu_resource_quota.md` | Namespace quota design |
| `e2e/README.md` | End-to-end suite, case list, and the EKS demo |
| `e2e/ack/README.md` | The ACK test stack with real GPU nodes; results in `e2e/ack/TEST-SUMMARY.md` |
| `docs/pod-security.md` | Pod security requirements of the injected containers |
| `e2e/terraform/README.md` | Terraform notes for the EKS test stack |

## 7. Known inconsistencies

Confirm the current state before modifying the related parts:

1. `appVersion` in `charts/tensor-fusion/Chart.yaml` is maintained by hand and lags behind the latest operator tag.
2. The hypervisor image tags in `config/samples` are inconsistent with each other. (The chart's default `hypervisorImage` is empty and resolves to the chart `appVersion`.)
3. The image name `tensorfusion/tensor-fusion-hypervisor` is published both by `vgpu.rs` (1.x, Rust) and by this repository (2.x, Go), with incompatible entry points and environment variables.
4. The chart pins the version of `vgpu-provider-nvidia`, while `dockerfile/hypervisor.Dockerfile` takes `:latest` at build time, so the libraries inside the hypervisor image and the libraries copied by the init container may not be the same version.
5. The `tensor-fusion-node-discovery` image is still referenced by the chart templates and samples, but the corresponding code has been deleted and no workflow builds it.
6. The hypervisor-facing `Limiter*` functions in `provider/limiter.h` have no callers on the Go side, and the header of the same name in `vgpu-provider` does not contain these declarations either. (`provider/accelerator.h` has been synchronised with `vgpu-provider`, see 5.1.2.)
