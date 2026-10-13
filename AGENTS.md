# TensorFusion

A Kubernetes operator for GPU virtualization and pooling. This repository builds two binaries:

- **controller** (`cmd/main.go`): CRD controllers, the Pod mutating webhook, and an embedded kube-scheduler, in one process.
- **hypervisor** (`cmd/hypervisor`): the per-node agent that discovers devices and enforces fractional GPU limits. It is written in Go and replaced the Rust hypervisor from `vgpu.rs`.

The vendor libraries it loads (`vgpu-provider`), the client and worker used for hard isolation and remote GPUs (`tensor-fusion-core`), and the console (`tensor-fusion-cloud`) live in other repositories. [`docs/stack-context.md`](docs/stack-context.md) maps the whole stack and the contracts between repositories; read it before touching images, the limiter, client/worker, or anything the console consumes.

Write code, comments, commit messages, and documentation in English.

## Where things are

| To change | Go to |
| --- | --- |
| CRD fields | `api/v1/`, then `make manifests generate`; the chart's copies are in `charts/tensor-fusion/crds/` |
| What gets injected into a Pod | `internal/webhook/v1/`, `internal/utils/compose.go` |
| Scheduling and GPU allocation | `internal/scheduler/`, `internal/gpuallocator/` |
| Node scaling, Karpenter | `internal/controller/gpupool_node_provision.go`, `gpunodeclaim_controller.go`, `internal/cloudprovider/`, `internal/scheduler/expander/` |
| Enforcing compute and VRAM limits | `pkg/hypervisor/worker/` (shared memory layout in `state/`, control loop in `computing/`) |
| Device discovery, vendors | `pkg/hypervisor/device/`, `provider/`, `pkg/constants/vendors.go` |
| Annotation and environment variable names | `pkg/constants/` |
| Metrics and alerts | `internal/metrics/`, `internal/alert/` |
| Helm chart | `charts/tensor-fusion/` |
| End-to-end tests on real clusters | `e2e/` (EKS: `e2e/terraform/`, ACK with real GPUs: `e2e/ack/`) |

Design documents: `docs/dynamic-gpu-isolation-design.md`, `docs/soft-isolation-preload.md`, `design/`. Installation and Helm values: `docs/installation-guide.md`.

## Commands

```bash
make build              # controller
make build-hypervisor   # hypervisor (CGO disabled, purego)
make build-provider     # example accelerator library
make test               # full envtest suite
make ut F=<file>        # one test file under internal/controller
make ut-sched           # scheduler tests
make test-chart         # render the chart and check what it ships
make lint
make vendor             # re-vendor and apply the scheduler patches in patches/
```

## Constraints

- `vendor/` holds a patched kube-scheduler. Do not edit it by hand: change the patch in `patches/`, then run `make vendor`.
- The shared memory layout for soft isolation is defined in this repository, in `vgpu-provider`, and in `vgpu.rs`. It has no version field, so all three must change together.
- `pkg/hypervisor` and `api/v1` are imported as libraries by `gpu-go`. Changing an exported interface there is a breaking change.
- `provider/*.h` are hand-copied from `vgpu-provider`.
- A change to CRDs, Helm values, metric tables, or annotations needs a matching change in the console (`tensor-fusion-cloud`) and the documentation (`tensor-fusion-website`).

## Sensitive information

This repository is public. Never commit keys, tokens, kubeconfigs, connection strings, internal addresses, cloud account IDs, or exploit details of an unfixed security issue. Keep local notes in `tmp.*` files, which `.gitignore` excludes.
