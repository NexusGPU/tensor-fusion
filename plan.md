# AMD GPU Remote Execution Support Implementation Plan

## Architecture Overview

TensorFusion uses a **provider pattern** for vendor-specific GPU support. Each vendor implements a shared library (`.so`) that conforms to the [`accelerator.h`](provider/accelerator.h) interface. The system consists of:

<img width="516" height="202" alt="{B6F05EE1-844D-4F11-A33A-19E8218225F7}" src="https://gist.github.com/user-attachments/assets/a7a6a3e2-41bf-41e3-9c60-1d222576d9ba" />

## Current State

   - ✅ **Accelerator interface defined**: [`provider/accelerator.h`](provider/accelerator.h)
   - ✅ **Example provider exists**: [`provider/example/accelerator.c`](provider/example/accelerator.c) using AMD SMI-style mocks
   - ✅ **AMD vendor constants defined**: [`internal/constants/vendors.go`](internal/constants/vendors.go)
   - ✅ **Multi-vendor support in GPUPool**: [`api/v1/gpupool_types.go`](api/v1/gpupool_types.go) with `ProviderImage` map
   - ❌ **No actual AMD provider implementation**
   - ❌ **No AMD-specific Docker images**
   - ❌ **No Helm chart configuration for AMD**

## Implementation Plan

### Phase 1: AMD Provider Library Implementation

**Goal**: Create `libaccelerator_amd.so` that implements the accelerator interface using amd-smi

#### 1.1 Create AMD Provider C Implementation

**File**: [`provider/amd/accelerator.c`](provider/amd/accelerator.c)

**Key Dependencies**:

   - `amd_smi/amdsmi.h` - Modern AMD SMI library
   - `hip/hip_runtime.h` - HIP runtime for device enumeration
   - Link against: `-lamd_smi -lamdhip64`

**Required API Implementations**:

1. **Initialization**:

   - `VirtualGPUInit()`: Call `amdsmi_init(AMDSMI_INIT_AMD_GPUS)`

2. **Device Discovery** (Critical):

   - `GetDeviceCount()`: Use `amdsmi_get_socket_handles()` and `amdsmi_get_processor_handles()`
   - `GetAllDevices()`: Query device properties using:
   - `amdsmi_get_gpu_asic_info()` - Get model name, vendor
   - `amdsmi_get_gpu_vram_info()` - Get total memory
   - `amdsmi_get_gpu_compute_process_gpus()` - Get compute capabilities
   - Generate UUID from PCI bus ID
   - `GetDeviceTopology()`: Use `amdsmi_topo_get_numa_node_number()`

3. **Metrics Collection** (Critical):

   - `GetProcessInformation()`: Use `amdsmi_get_gpu_process_info()` to get per-process metrics
   - `GetDeviceMetrics()`: Collect:
   - GPU utilization: `amdsmi_get_gpu_activity()`
   - Memory usage: `amdsmi_get_gpu_memory_usage()`
   - Power: `amdsmi_get_power_info()`
   - Temperature: `amdsmi_get_temp_metric(AMDSMI_TEMPERATURE_TYPE_EDGE)`
   - PCIe throughput: `amdsmi_get_gpu_pci_throughput()`

4. **Virtualization APIs** (Return NOT_SUPPORTED):

   - `AssignPartition()`: Return `false` (no MIG-like support yet)
   - `RemovePartition()`: Return `false`
   - `SetMemHardLimit()`: Return `RESULT_ERROR_NOT_SUPPORTED`
   - `SetComputeUnitHardLimit()`: Return `RESULT_ERROR_NOT_SUPPORTED`
   - `Snapshot()`, `Resume()`: Return `RESULT_ERROR_NOT_SUPPORTED`

5. **Utility**:

   - `GetVendorMountLibs()`: Return ROCm library paths:
   - `/opt/rocm/lib` → `/usr/local/rocm/lib`
   - `/opt/rocm/bin` → `/usr/local/rocm/bin`

**Error Handling**:

   - Check all amd-smi return codes (`amdsmi_status_t`)
   - Use `RegisterLogCallback()` for logging
   - Map amd-smi errors to `Result` enum

**Reference Implementation**: Study [`provider/example/accelerator.c`](provider/example/accelerator.c) for structure

#### 1.2 Update Provider Makefile

**File**: [`provider/Makefile`](provider/Makefile)

Add AMD build target:

```makefile
# AMD provider
AMD_DIR := $(PROVIDER_DIR)/amd
AMD_LIB := $(BUILD_DIR)/libaccelerator_amd.so
AMD_SRC := $(AMD_DIR)/accelerator.c
AMD_OBJ := $(BUILD_DIR)/accelerator_amd.o

# ROCm paths (TheRock installs to /opt/rocm via symlink)
ROCM_PATH ?= /opt/rocm
AMD_CFLAGS := -I$(ROCM_PATH)/include
AMD_LDFLAGS := -L$(ROCM_PATH)/lib -lamd_smi -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib

.PHONY: amd test-amd-run

amd: $(AMD_LIB)

$(AMD_LIB): $(AMD_OBJ) | $(BUILD_DIR)
	$(CC) $(LDFLAGS) -o $@ $< $(AMD_LDFLAGS)

$(AMD_OBJ): $(AMD_SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(AMD_CFLAGS) -c -o $@ $<

# Test AMD provider
test-amd-run: amd
	LD_LIBRARY_PATH=$(ROCM_PATH)/lib:$(BUILD_DIR):$$LD_LIBRARY_PATH \
	$(BUILD_DIR)/test_accelerator $(AMD_LIB)
```

#### 1.3 Create AMD Provider Test

**File**: [`provider/test/test_amd_provider.c`](provider/test/test_amd_provider.c)

Simple test program to verify:

   - Library loads successfully
   - Device discovery works
   - Metrics can be collected
   - All unsupported features return proper error codes

### Phase 2: Docker Image Creation

**Goal**: Package AMD provider in container images for deployment

#### 2.1 Create AMD Provider Dockerfile

**File**: [`dockerfile/amd-provider.Dockerfile`](dockerfile/amd-provider.Dockerfile)

```dockerfile
FROM ubuntu:22.04 AS builder

WORKDIR /workspace

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    gfortran git ninja-build g++ pkg-config xxd patchelf automake libtool python3-venv python3-dev libegl1-mesa-dev texinfo bison flex \
    && rm -rf /var/lib/apt/lists/*

# Copy and run TheRock installation script
COPY scripts/install_rocm_tarball.sh /tmp/
ARG ROCM_VERSION=7.10.0
ARG AMDGPU_FAMILY=gfx94X-dcgpu
ARG RELEASE_TYPE=stable

RUN bash /tmp/install_rocm_tarball.sh ${ROCM_VERSION} ${AMDGPU_FAMILY} ${RELEASE_TYPE}

# Set environment for build (script creates /opt/rocm symlink)
ENV ROCM_PATH=/opt/rocm
ENV PATH=${ROCM_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:${LD_LIBRARY_PATH}

# Copy provider source
COPY provider/ provider/

# Build AMD provider
WORKDIR /workspace/provider
RUN make amd

# Create deployment image
FROM ubuntu:22.04

WORKDIR /build

# Copy TheRock installation (includes /opt/rocm symlink and /opt/rocm-VERSION)
COPY --from=builder /opt/rocm-* /opt/
COPY --from=builder /opt/rocm /opt/rocm

# Copy built provider library
COPY --from=builder /workspace/provider/build/libaccelerator_amd.so /build/lib/

# Create metadata file
ARG ROCM_VERSION=7.10.0
RUN echo "version: 1.0.0\n\
hardwareVendor: AMD\n\
releaseDate: \"2026-01-23\"\n\
isolationModes:\n\
  - shared\n\
rocmDistribution: TheRock\n\
rocmVersion: ${ROCM_VERSION}" > /build/metadata.yaml

# Set runtime environment (uses standard /opt/rocm path)
ENV ROCM_PATH=/opt/rocm
ENV PATH=${ROCM_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:${LD_LIBRARY_PATH}

ENTRYPOINT ["sleep", "infinity"]
```

**Key Points**:

   - Uses Ubuntu base + TheRock installed via official script
   - TheRock installs to `/opt/rocm-{VERSION}` with `/opt/rocm` symlink
   - Multi-stage build to keep final image smaller
   - Provider library in `/build/lib/`
   - TheRock binaries/libs accessible via standard `/opt/rocm` path

**TheRock Build Arguments**:

   - `ROCM_VERSION`: Version to install (default: `7.10.0`)
   - `AMDGPU_FAMILY`: GPU architecture (default: `gfx94X-dcgpu` for MI325X)
   - `RELEASE_TYPE`: Release channel (default: `stable`, options: `nightlies`, `prereleases`, `devreleases`)

**Script Benefits**:

   - Handles download from correct URL based on release type
   - Validates installation after extraction
   - Creates standard `/opt/rocm` symlink automatically
   - Same script works in bare metal and Docker

#### 2.2 Update Main Makefile for AMD Images

**File**: [`Makefile`](Makefile)

Add Docker build targets:

```makefile
.PHONY: docker-build-amd-provider
docker-build-amd-provider:
	$(CONTAINER_TOOL) build -f dockerfile/amd-provider.Dockerfile \
		-t $(IMG_REGISTRY)/tensor-fusion-amd-provider:$(VERSION) .

.PHONY: docker-push-amd-provider
docker-push-amd-provider:
	$(CONTAINER_TOOL) push $(IMG_REGISTRY)/tensor-fusion-amd-provider:$(VERSION)
```

### Phase 3: Helm Chart Integration

**Goal**: Enable AMD GPU deployment via Helm charts

#### 3.1 Create AMD-Specific Values File

**File**: [`charts/tensor-fusion/values-amd.yaml`](charts/tensor-fusion/values-amd.yaml)

```yaml
# AMD GPU Configuration
initialGpuNodeLabelSelector: "amdgpu.com/gpu.present=true"

controller:
  image:
    repository: tensorfusion/tensor-fusion-operator
    tag: "1.48.6"

# Configure AMD-specific provider image
defaultProviderImages:
  AMD: "tensorfusion/tensor-fusion-amd-provider:1.0.0"

# Example GPUPool configuration for AMD
exampleGPUPool:
  enabled: true
  spec:
    nodeManagerConfig:
      defaultVendor: "AMD"
      nodeSelector:
        nodeSelectorTerms:
          - matchExpressions:
              - key: amdgpu.com/gpu.present
                operator: Exists
    componentConfig:
      hypervisor:
        image: "tensorfusion/tensor-fusion-operator:1.48.6"
        providerImage:
          AMD: "tensorfusion/tensor-fusion-amd-provider:1.0.0"
      worker:
        image: "tensorfusion/tensor-fusion-operator:1.48.6"
        providerImage:
          AMD: "tensorfusion/tensor-fusion-amd-provider:1.0.0"

schedulerConfig:
  profiles:
    - schedulerName: tensor-fusion-scheduler
      pluginConfig:
        - name: NodeResourcesFit
          args:
            ignoredResourceGroups:
              - "tensor-fusion.ai"
            scoringStrategy:
              resources:
                - name: cpu
                  weight: 1
                - name: memory
                  weight: 1
                - name: amd.com/gpu
                  weight: 5
```

#### 3.2 Update Helm Templates

**File**: [`charts/tensor-fusion/templates/gpu-public-gpu-info.yaml`](charts/tensor-fusion/templates/gpu-public-gpu-info.yaml)

Add AMD vendor information to the ConfigMap that provides GPU model mappings.

### Phase 4: Kubernetes Deployment Configuration

**Goal**: Configure cluster for AMD GPU support

#### 4.1 Create AMD Node Labeling Documentation

**File**: `docs/amd-gpu-setup.md` (new)

Document how to:

1. Install ROCm on nodes
2. Label nodes with `amdgpu.com/gpu.present=true`
3. Verify GPU accessibility
4. Configure device plugin compatibility

#### 4.2 Sample GPUPool for AMD

**File**: [`config/samples/v1_gpupool_amd.yaml`](config/samples/v1_gpupool_amd.yaml)

```yaml
apiVersion: tensor-fusion.ai/v1
kind: GPUPool
metadata:
  name: amd-gpu-pool
spec:
  defaultUsingLocalGPU: true
  
  capacityConfig:
    minResources:
      gpu: 1
    maxResources:
      gpu: 8
    oversubscription:
      vramExpandToHostMem: 50
      tflopsOversellRatio: 200
  
  nodeManagerConfig:
    provisioningMode: AutoSelect
    defaultVendor: AMD
    nodeSelector:
      nodeSelectorTerms:
        - matchExpressions:
            - key: amdgpu.com/gpu.present
              operator: Exists
  
  componentConfig:
    hypervisor:
      image: tensorfusion/tensor-fusion-operator:1.48.6
      providerImage:
        AMD: tensorfusion/tensor-fusion-amd-provider:1.0.0
      enableVector: true
    
    worker:
      image: tensorfusion/tensor-fusion-operator:1.48.6
      providerImage:
        AMD: tensorfusion/tensor-fusion-amd-provider:1.0.0
```

### Phase 5: Testing & Validation

#### 5.1 Local Provider Testing

```bash
# Build provider
cd provider
make amd

# Run unit tests
make test-amd-run

# Test with hypervisor locally
cd ..
make build-hypervisor
./bin/hypervisor \
  -accelerator-lib ./provider/build/libaccelerator_amd.so \
  -vendor AMD \
  -isolation-mode shared \
  -backend-type simple
```

#### 5.2 Cluster Integration Testing

**Create test script**: `scripts/test-amd-cluster.sh`

Steps:

1. Label nodes: `kubectl label nodes <node-name> amdgpu.com/gpu.present=true`
2. Install Helm chart with AMD values
3. Verify GPUPool status shows AMD GPUs discovered
4. Deploy test workload requesting `tensor-fusion.ai/vram` resources
5. Verify GPU allocation and metrics collection

#### 5.3 Metrics Validation

Verify these metrics are collected in GreptimeDB:

   - `tf_gpu_usage`: GPU utilization, memory, temperature
   - `tf_worker_usage`: Per-workload GPU assignment
   - `tf_node_metrics`: Node-level aggregated metrics

## Key Technical Considerations

### TheRock Installation (Official Script)

**Important**: User is using **TheRock** (AMD's next-gen ROCm distribution) instead of standard ROCm packages.

**Installation Script**: `scripts/install_rocm_tarball.sh`

```bash
./install_rocm_tarball.sh <VERSION> <AMDGPU_FAMILY> [RELEASE_TYPE]

# Production installation for MI325X:
./install_rocm_tarball.sh 7.10.0 gfx94X-dcgpu stable
```

**Script Features**:

   - Installs to `/opt/rocm-{VERSION}` (e.g., `/opt/rocm-7.10.0`)
   - Creates symlink `/opt/rocm` → `/opt/rocm-{VERSION}` for compatibility
   - Supports release types: `stable`, `nightlies`, `prereleases`, `devreleases`
   - Validates installation (checks bin, lib, include directories)
   - Works in both bare metal and Docker environments

**Download URLs**:

   - **Stable**: `https://repo.amd.com/rocm/tarball/therock-dist-linux-gfx94X-dcgpu-7.10.0.tar.gz`
   - **Nightlies**: `https://rocm.nightlies.amd.com/tarball/...`
   - **Prereleases**: `https://rocm.prereleases.amd.com/tarball/...`

**Path Standardization**:

✅ **Good News**: Because the script creates `/opt/rocm` symlink, we can use **standard ROCm paths**:

   - Libraries: `/opt/rocm/lib`
   - Binaries: `/opt/rocm/bin`
   - Headers: `/opt/rocm/include`

This means **no special path detection needed** - TheRock looks exactly like standard ROCm to our provider!

### AMD SMI vs ROCm SMI

**Use amd-smi (modern)** instead of rocm-smi:

   - **Header**: `amd_smi/amdsmi.h` (not `rocm_smi/rocm_smi.h`)
   - **Library**: `-lamd_smi` (not `-lrocm_smi64`)
   - **Init**: `amdsmi_init(AMDSMI_INIT_AMD_GPUS)`
   - **Better API design**: Unified handles, cleaner error codes
   - **Location in TheRock**: `$ROCM_PATH/lib/libamd_smi.so`

### GPU Model Detection for MI325X

```c
// Get ASIC info
amdsmi_asic_info_t asic_info;
amdsmi_get_gpu_asic_info(processor_handle, &asic_info);

// MI325X will report:
// - asic_info.market_name might contain "MI325X" 
// - Use asic_info.device_id to identify specific ASIC
// - Compute units from asic_info.num_shader_engines * asic_info.num_shader_arrays_per_engine
```

### Resource Naming

Follow AMD device plugin convention:

   - Use `amd.com/gpu` for whole GPU resources
   - TensorFusion will manage fractional resources via `tensor-fusion.ai/vram`, `tensor-fusion.ai/tflops`

### Library Path Management (Simplified with TheRock)

The provider's `GetVendorMountLibs()` implementation is **simple** because TheRock uses standard paths:

```c
Result GetVendorMountLibs(Mount* mounts, size_t maxCount, size_t* mountCount) {
    if (!mounts || !mountCount || maxCount < 2) {
        return RESULT_ERROR_INVALID_PARAM;
    }
    
    // TheRock installs to /opt/rocm (via symlink) - same as standard ROCm
    snprintf(mounts[0].hostPath, MAX_MOUNT_PATH, "/opt/rocm/lib");
    snprintf(mounts[0].guestPath, MAX_MOUNT_PATH, "/usr/local/rocm/lib");
    
    snprintf(mounts[1].hostPath, MAX_MOUNT_PATH, "/opt/rocm/bin");
    snprintf(mounts[1].guestPath, MAX_MOUNT_PATH, "/usr/local/rocm/bin");
    
    *mountCount = 2;
    return RESULT_SUCCESS;
}
```

**Why This Works**:

   - TheRock installation script creates `/opt/rocm` symlink
   - No runtime path detection needed
   - Works with both standard ROCm and TheRock installations
   - Same implementation as NVIDIA provider (simple and reliable)

These paths are mounted into worker containers so HIP applications can find ROCm runtime.

## Rollout Strategy

1. **Week 1-2**: Implement and test AMD provider locally

   - Write `provider/amd/accelerator.c`
   - Test on bare metal with MI325X
   - Verify all metrics APIs work

2. **Week 3**: Docker images and build system

   - Create Dockerfiles
   - Build and push images
   - Test image in Kubernetes

3. **Week 4**: Helm integration and cluster testing

   - Create values-amd.yaml
   - Deploy to test cluster
   - Run workload tests

4. **Week 5**: Documentation and refinement

   - Write deployment guide
   - Performance testing
   - Bug fixes

### Phase 6: Remote GPU Execution (GPU-over-IP)

**Goal**: Enable containers to use AMD GPUs on remote nodes via API forwarding

**Approach**: Phased implementation after monitoring is stable, with basic HIP API coverage

#### How Remote GPU Execution Works (Architecture Overview)

Remote GPU execution makes physical GPUs on one node **transparently accessible** to containers on other nodes. The application code doesn't change - it's completely invisible except for latency.

**Bidirectional Request-Response Flow**:

<img width="417" height="512" alt="{EAB11A4D-F97A-451B-92E4-17DBD2E97055}" src="https://gist.github.com/user-attachments/assets/c90a1742-421e-4d3f-941a-20f76b53ef50" />

**Key Properties**:

1. **Transparent to Application**:

   - No code changes required
   - Application calls standard HIP APIs
   - Same APIs as if GPU were local

2. **Bidirectional Communication**:

   - **Request Path**: App → Stub → Network → Worker → GPU
   - **Response Path**: GPU → Worker → Network → Stub → App
   - Every API call gets a response (synchronous or async)

3. **State Synchronization**:

   - Memory allocations tracked on both sides
   - Device pointers maintained in mapping tables
   - Kernel modules cached on worker

4. **Data Handling**:

   - Small data (pointers, scalars): Serialized in protocol
   - Large data (buffers): Sent as binary payload
   - `hipMemcpy(H2D)`: Client sends data over network
   - `hipMemcpy(D2H)`: Worker sends data back

5. **Only User-Visible Difference**:

   - **Latency**: Network RTT + execution time
   - **Local GPU**: ~microseconds
   - **Remote GPU**: ~milliseconds (acceptable for ML workloads)
   - Performance overhead typically < 20% for training/inference

**What Gets Forwarded**:

| HIP API | Request Data | Response Data |
|---------|--------------|---------------|
| `hipMalloc` | Size | Device pointer |
| `hipFree` | Device pointer | Status |
| `hipMemcpy(H2D)` | Pointer + Data (1GB) | Status |
| `hipMemcpy(D2H)` | Pointer + Size | Data (1GB) + Status |
| `hipLaunchKernel` | Function, grid, block, args | Status |
| `hipStreamSynchronize` | Stream handle | Status |

#### 6.1 Research AMD Remote GPU Solutions

**Action Items**:

1. Research AMD's existing remote GPU technologies:

   - AMD MxGPU (hardware virtualization)
   - AMD ROCm-aware networking solutions
   - Any AMD-supported API remoting tools

2. Study existing open-source HIP API interceptors:

   - Check for HIP equivalents of CUDA interception tools
   - Review vGPU.rs if it has HIP support patterns
   - Look at AMD's virtualization documentation

3. Analyze TensorFusion's NVIDIA remote GPU implementation (if accessible):

   - Understand the protocol format
   - Learn serialization approach
   - Study worker service architecture

**Deliverable**: Research document comparing approaches and recommending implementation strategy

#### 6.2 Enable Remote GPU Support in Provider

**File**: [`provider/amd/accelerator.c`](provider/amd/accelerator.c)

Update the AMD provider to report remote GPU capability:

```c
// In GetAllDevices()
devices[i].virtualizationCapabilities.supportsRemoting = true;
```

This signals to TensorFusion that remote execution is available for AMD GPUs.

#### 6.3 Implement HIP Client Stub (Interception Library)

**File**: `provider/amd/hip_client_stub.c` (new)

Create a shared library that intercepts HIP API calls and forwards them to a remote worker.

**Key Components**:

1. **HIP API Interception**:
   ```c
   // Intercept memory allocation
   hipError_t hipMalloc(void** devPtr, size_t size) {
       return remote_hip_malloc(devPtr, size, worker_connection);
   }
   
   // Intercept memory copy
   hipError_t hipMemcpy(void* dst, const void* src, size_t count, hipMemcpyKind kind) {
       return remote_hip_memcpy(dst, src, count, kind, worker_connection);
   }
   
   // Intercept kernel launch
   hipError_t hipLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, 
                               void** args, size_t sharedMem, hipStream_t stream) {
       return remote_hip_launch_kernel(func, gridDim, blockDim, args, 
                                       sharedMem, stream, worker_connection);
   }
   ```

2. **Network Communication**:

   - Use TCP sockets for initial implementation (best effort performance)
   - Protocol: Serialize HIP API calls as messages
   - Format: `[op_code][request_id][payload_length][payload]`

3. **Basic APIs to Support** (Phase 1):

   - Device management: `hipGetDeviceCount`, `hipSetDevice`, `hipGetDeviceProperties`
   - Memory: `hipMalloc`, `hipFree`, `hipMemcpy`, `hipMemcpyAsync`
   - Kernels: `hipLaunchKernel`, `hipModuleLoad`, `hipModuleGetFunction`
   - Streams: `hipStreamCreate`, `hipStreamDestroy`, `hipStreamSynchronize`

4. **Connection Management**:

   - Connect to worker specified by TensorFusion (via env var)
   - Automatic reconnection on failure
   - Keep-alive heartbeat

**Dockerfile**: [`dockerfile/amd-client-stub.Dockerfile`](dockerfile/amd-client-stub.Dockerfile)

```dockerfile
FROM rocm/dev-ubuntu-22.04:6.3.1-complete AS builder

WORKDIR /workspace
COPY provider/ provider/

RUN cd provider && make hip-client-stub

FROM ubuntu:22.04

WORKDIR /build
COPY --from=builder /workspace/provider/build/libhip_client_stub.so /build/lib/

# Minimal HIP headers for application linking
RUN mkdir -p /build/include
COPY --from=builder /opt/rocm/include/hip /build/include/hip

ENTRYPOINT ["sleep", "infinity"]
```

#### 6.4 Implement Worker Service (Remote GPU Executor)

**File**: `provider/amd/hip_worker_service.c` (new)

Create a service that receives HIP API calls and executes them on physical GPUs.

**Architecture**:

<img width="527" height="118" alt="{BF842B60-4E26-44B8-9034-E2A4F055A4CA}" src="https://gist.github.com/user-attachments/assets/171ffcf7-ba67-438e-93df-763801592250" />


**Key Components**:

1. **Network Listener**:

   - Listen on configurable port (default: 50051)
   - Accept multiple client connections
   - Per-client session management

2. **Request Dispatcher**:

   - Parse incoming messages
   - Route to appropriate handler
   - Manage request/response lifecycle

3. **API Handlers**:

   - Memory operations: Allocate/free GPU memory
   - Data transfers: Handle host↔device copies
   - Kernel execution: Load and launch kernels
   - Stream management: Async operation handling

4. **Resource Tracking**:

   - Track memory allocations per client
   - Maintain kernel module cache
   - Monitor GPU utilization

5. **Error Handling**:

   - Convert HIP errors to network responses
   - Graceful client disconnect handling
   - Resource cleanup on failure

**Integration with TensorFusion Worker**:

   - Worker service runs alongside TensorFusion worker pod
   - Hypervisor reports remote GPU capability
   - TensorFusion scheduler assigns remote GPU workers to client pods

#### 6.5 Protocol Design

**Message Format** (simple binary protocol):

```
Header (16 bytes):
 - magic: 0x48495052 ('HIPR') [4 bytes]
 - version: 0x0001 [2 bytes]
 - op_code: API operation [2 bytes]
 - request_id: Unique ID [4 bytes]
 - payload_length: Data size [4 bytes]

Payload (variable):
 - API-specific data (msgpack serialization)
```

**Operation Codes**:

```c
enum HIPRemoteOp {
    HIP_REMOTE_GET_DEVICE_COUNT = 0x0001,
    HIP_REMOTE_MALLOC = 0x0010,
    HIP_REMOTE_FREE = 0x0011,
    HIP_REMOTE_MEMCPY_H2D = 0x0020,
    HIP_REMOTE_MEMCPY_D2H = 0x0021,
    HIP_REMOTE_LAUNCH_KERNEL = 0x0030,
    HIP_REMOTE_STREAM_SYNC = 0x0040,
    // ... more ops
};
```

**Example: hipMalloc**:

```
Request:
  op_code: HIP_REMOTE_MALLOC
  payload: { size: 1048576 }

Response:
  payload: { status: 0, devPtr: 0x7f8a40000000 }
```

#### 6.6 Update Helm Chart for Remote GPU

**File**: [`charts/tensor-fusion/values-amd.yaml`](charts/tensor-fusion/values-amd.yaml)

Add remote GPU configuration:

```yaml
componentConfig:
  # Client stub injected into user containers
  client:
    image: "tensorfusion/tensor-fusion-operator:1.48.6"
    providerImage:
      AMD: "tensorfusion/tensor-fusion-amd-client-stub:1.0.0"
  
  # Worker service that executes HIP calls
  worker:
    image: "tensorfusion/tensor-fusion-operator:1.48.6"
    providerImage:
      AMD: "tensorfusion/tensor-fusion-amd-worker:1.0.0"
    # Worker service listens on this port
    workerServicePort: 50051
  
  # Hypervisor manages worker lifecycle
  hypervisor:
    image: "tensorfusion/tensor-fusion-operator:1.48.6"
    providerImage:
      AMD: "tensorfusion/tensor-fusion-amd-provider:1.0.0"
```

#### 6.7 Testing Remote GPU Execution

**Test Script**: `scripts/test-amd-remote-gpu.sh`

```bash
#!/bin/bash
# Test remote GPU execution

# 1. Deploy worker on GPU node
kubectl label node gpu-node-1 tensor-fusion.ai/gpu-worker=true

# 2. Deploy client on non-GPU node  
kubectl run test-client \
  --image=rocm/pytorch:latest \
  --overrides='{"spec": {"nodeSelector": {"kubernetes.io/hostname": "cpu-node-1"}}}' \
  --env="TF_REMOTE_GPU=true" \
  -- python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 3. Verify remote GPU is used
kubectl logs test-client | grep "True"
kubectl logs test-client | grep "MI325X"
```

**Validation**:

   - Client container on non-GPU node can use GPU
   - Performance overhead < 20% for typical ML workloads
   - Stable under sustained load (no memory leaks)
   - Proper error handling when worker disconnects

#### 6.8 Performance Optimization (Future)

For production deployments requiring lower latency:

1. **RDMA Support**:

   - Use ROCm's RDMA-aware communication
   - Requires InfiniBand/RoCE network
   - Reduces CPU overhead for data transfers

2. **Protocol Optimization**:

   - Batch small API calls
   - Compress large data transfers
   - Zero-copy where possible

3. **Connection Pooling**:

   - Reuse connections across requests
   - Load balance across multiple workers

4. **Kernel Caching**:

   - Cache compiled kernels on worker
   - Avoid redundant module loads

## Success Criteria

### Phase 1-5 (Monitoring):

✅ AMD GPUs discovered and registered in GPUPool

✅ Real-time metrics visible in TensorFusion dashboard

✅ Workloads can request fractional GPU resources

✅ GPU utilization tracked per workload

✅ No crashes or memory leaks under sustained load

✅ Compatible with existing ROCm workloads

### Phase 6 (Remote GPU):

✅ Container on non-GPU node can execute HIP code on remote GPU

✅ Basic HIP APIs (malloc, memcpy, kernel launch) work remotely

✅ Performance overhead acceptable for ML training workloads

✅ Automatic failover when worker becomes unavailable

✅ Multiple clients can share same remote GPU worker

## Implementation Timeline

**Immediate (Weeks 1-5)**: Phases 1-5 (Monitoring foundation)

**After Monitoring Stable (Weeks 6-10)**: Phase 6 (Remote GPU execution)

## Open Questions for User

Before starting implementation, please confirm:

1. **ROCm Version**: Which ROCm version is installed on your MI325X nodes? (6.x recommended)
2. **Device Plugin**: Are you currently using any AMD device plugin (e.g., AMD GPU Device Plugin)?
3. **Node Access**: Can you provide SSH access to one MI325X node for testing?
4. **Multi-GPU**: How many MI325X GPUs per node in your cluster?
5. **Network**: What network fabric connects your nodes? (TCP/IP, InfiniBand, RoCE?)
6. **Workload Types**: What frameworks will use remote GPU? (PyTorch, TensorFlow, custom HIP code?)