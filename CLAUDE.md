# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorFusion is a GPU virtualization and pooling solution for Kubernetes that optimizes GPU cluster utilization through fractional vGPU, remote GPU sharing over network, GPU-first scheduling, and oversubscription capabilities. It's built as a Kubernetes operator using the Kubebuilder framework.

## Build and Development Commands

### Primary Commands
- `make build` - Build the tensor-fusion operator binary to `bin/manager`
- `make run` - Run the controller locally (requires kube config)
- `make docker-build` - Build Docker image for the operator
- `make test` - Run comprehensive test suite with coverage
- `make test-serial` - Run tests serially (for debugging)
- `make lint` - Run golangci-lint
- `make lint-fix` - Run linter with automatic fixes

### Code Generation
- `make generate` - Generate DeepCopy methods for API types
- `make manifests` - Generate CRDs, RBAC, and webhook configurations

### Testing Commands
- `make ut F=<filename>` - Run unit tests for specific file (e.g., `make ut F=gpu_controller_test.go`)
- `make test-e2e` - Run end-to-end tests (requires Kind cluster)

### Deployment
- `make install` - Install CRDs to current kubectl context
- `make deploy` - Deploy operator to current kubectl context
- `make uninstall` - Remove CRDs from cluster
- `make undeploy` - Remove operator deployment

## Architecture Overview

### Core Components

**Main Entry Point**: `cmd/main.go` - Orchestrates all components including:
- Kubernetes controller manager
- Custom scheduler integration
- GPU allocator
- Port allocator
- HTTP API server
- Admission webhooks
- Metrics collection

**API Types** (`api/v1/`):
- `GPU` - Individual GPU resource representation
- `GPUNode` - Node-level GPU information
- `GPUPool` - Grouped GPU resources
- `GPUNodeClaim` - GPU provisioning requests
- `GPUResourceQuota` - Resource quotas for GPU usage
- `TensorFusionCluster` - Cluster configuration
- `TensorFusionWorkload` - Workload specifications
- `WorkloadProfile` - Workload template definitions

**GPU Allocator** (`internal/gpuallocator/`):
- Central component for GPU scheduling decisions
- Implements filtering and scoring strategies
- Handles quota management and resource allocation
- Strategies: compact-first, low-load distribution

**Controllers** (`internal/controller/`):
- Kubernetes controllers for each CRD type
- Pod controller for GPU assignment and lifecycle management
- Node controller for GPU discovery and status management

**Scheduler Integration** (`internal/scheduler/`):
- Custom Kubernetes scheduler plugins
- GPU resource fit plugin for resource matching
- GPU topology plugin for optimal placement

### Key Architectural Patterns

**Resource Management**: Uses a centralized allocator pattern where the `GpuAllocator` maintains the authoritative state of all GPU allocations and availability.

**Filtering Pipeline**: GPU selection uses a multi-stage filtering system:
1. Phase filter (GPU state validation)
2. Resource filter (capacity matching)  
3. GPU model filter (hardware requirements)
4. Node affinity filter (placement constraints)

**Metrics and Observability**: Comprehensive metrics collection through:
- Time-series database integration (GreptimeDB)
- OpenTelemetry support
- Custom metrics encoding (InfluxDB, JSON, OTEL formats)

**Configuration Management**: Dynamic configuration reloading for:
- GPU info and pricing data
- Global settings
- Alert rules
- Scheduler configurations

## Development Guidelines

### Testing Strategy
- Unit tests use Ginkgo framework (`github.com/onsi/ginkgo/v2`)
- Controller tests use envtest for Kubernetes API simulation
- Test files follow `*_test.go` naming convention
- Integration tests require real or Kind Kubernetes cluster

### Code Organization
- API types in `api/v1/` with kubebuilder markers for CRD generation
- Business logic in `internal/` packages
- Controllers follow the standard controller-runtime pattern
- Utilities and helpers in `internal/utils/`

### Configuration Files
- GPU information: `/etc/tensor-fusion/gpu-info.yaml`
- Dynamic config: `/etc/tensor-fusion/config.yaml` 
- Scheduler config: `/etc/tensor-fusion/scheduler-config.yaml`
- These paths can be overridden via command-line flags

### Dependencies
- Built on controller-runtime v0.21.0
- Kubernetes API v0.33.3
- Uses Karpenter integration for node provisioning
- Cloud provider integrations (AWS, Alibaba Cloud)
- Time-series database support (GreptimeDB, MySQL)

## Common Development Tasks

### Adding New CRD
1. Define types in `api/v1/` with kubebuilder markers
2. Run `make generate` to create DeepCopy methods
3. Run `make manifests` to generate CRD YAML
4. Create controller in `internal/controller/`
5. Add controller setup in `cmd/main.go`

### Running Tests
- Individual controller tests: `make ut F=<controller_test.go>`
- Full test suite: `make test`  
- With coverage: Tests automatically generate `cover.out`

### Local Development
1. Ensure Kind cluster is running for e2e tests
2. Set KUBECONFIG environment variable
3. Use `make run` to start controller locally
4. Use sample configurations in `config/samples/`

### Building and Deployment
- Container builds use configurable tool (docker/nerdctl)
- Supports multi-platform builds via buildx
- Helm charts available in `charts/tensor-fusion/`
- Kustomize configurations in `config/` directory

## TODO
TODO 文件夹下存放了所有需要执行的task，阅读以及评估合理性并且执行