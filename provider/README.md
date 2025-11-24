# Accelerator Provider Interface

This directory contains the abstract ABI (Application Binary Interface) for vGPU vendor accelerator libraries.

## Overview

The accelerator interface abstracts vGPU vendor-specific implementations into a unified API, supporting four isolation modes:

- **Shared Mode**: Oversubscription, high elasticity, no resource control (equivalent to NVIDIA timeslicing)
- **Soft Mode**: Oversubscription, high elasticity, time-sharing resource control via hooks and limiter
- **Hard Mode**: No oversubscription, medium elasticity, space-sharing via one-time resource limits
- **Partitioned Mode**: No oversubscription, low elasticity, hardware/driver-level partitioning (e.g., MIG)

## Structure

```
provider/
├── accelerator.h          # Main interface definition
├── limiter.h             # Limiter.so API (not vendor-implemented)
├── Makefile              # Build scripts
├── stub/
│   └── accelerator.c     # Stub implementation for testing
├── ascend/
│   ├── accelerator.cpp   # Huawei Ascend implementation (dcmi vNPU flow)
│   └── tests/test_ascend.cpp
└── test/
    ├── ascend_partition_fake.c # C sanity check for Ascend lib
    └── test_accelerator.c     # Stub test suite
```

## Building

### Build Stub Implementation

```bash
cd provider
make stub
```

### Build Ascend Implementation

```bash
cd provider
make ascend
```

### Run Tests

```bash
cd provider
make test-run
# Ascend fake vNPU path
make test-ascend-run
# Hypervisor integration smoke test (loads stub .so)
cd ..
go test ./internal/hypervisor/device -run TestControllerLoadsStubLibrary -count=1
```

> NOTE: `make ascend`/`test-ascend-run` require libdcmi built for the current architecture (e.g., aarch64 on Ascend hosts). The sample libdcmi shipped under `/usr/local/dcmi` in this environment is aarch64 and cannot be linked on x86; run the build on the target Ascend node.
>
> The Ascend provider dlopens libdcmi at runtime. Override the path with `DCMI_LIB_PATH=/path/to/libdcmi.so` if needed; when the library cannot be loaded, device count is reported as zero so tests will skip.

## Interface Categories

### 1. DeviceInfo APIs

- `GetDeviceCount()`: Number of accelerators
- `GetAllDevices()`: Enumerate accelerators and capabilities
- `GetPartitionTemplates()`: Hardware partition templates (MIG/vNPU)
- `GetDeviceTopology()`: Device topology (NVLink, IB NIC, etc.)

### 2. Virtualization APIs

#### Partitioned Isolation
- `AssignPartition()`: Assign hardware partition (returns partitionOverhead)
- `RemovePartition()`: Remove partition

#### Hard Isolation
- `SetMemHardLimit()`: Set hard memory limit (one-time)
- `SetComputeUnitHardLimit()`: Set hard compute limit (one-time)

#### Snapshot/Migration
- `Snapshot()`: Snapshot device state for processes
- `Resume()`: Resume device state for processes

### 3. Metrics APIs

- `GetProcessComputeUtilization()`: Get compute utilization per process
- `GetProcessMemoryUtilization()`: Get memory utilization per process
- `GetDeviceMetrics()`: Basic device metrics (power, PCIe, SM active, TC usage)
- `GetExtendedDeviceMetrics()`: Extended metrics (NVLink bandwidth, etc.)

## Vendor Implementations

### Stub Implementation

The stub implementation (`stub/accelerator.c`) provides a reference implementation for testing and development.

### Ascend Implementation

The Ascend implementation (`ascend/accelerator.cpp`) uses libdcmi to enumerate devices and create/destroy vNPU partitions:

- Enumerates devices via `dcmi_get_all_device_count` + logicId→card/device mapping
- Returns vir01/vir02/vir02_1c/vir04/vir04_3c/vir04_3c_ndvpp/vir04_4c_dvpp templates
- Creates/destroys vnpu via `dcmi_create_vdevice` / `dcmi_set_destroy_vdevice` (vgroup from `ASCEND_VGROUP`, default 0)
- Metrics/Hard/Snapshot remain stubbed; swap to real APIs as needed

## Usage in Hypervisor

The hypervisor uses the accelerator library via CGO bindings:

```go
import "github.com/NexusGPU/tensor-fusion/internal/hypervisor/device"

mgr, err := device.NewManager("path/to/libaccelerator.so", 30*time.Second)
```

See `internal/hypervisor/device/` for the Go bindings and device manager implementation.

## Testing

- `make test-run`: runs the stub C test suite against `libaccelerator_stub.so`
- `make test-ascend-run`: builds and runs the Ascend fake vNPU checks
- `meson test -C build`: runs the gtest-based Ascend provider tests (if you built with Meson)
- `go test ./internal/hypervisor/device -run TestControllerLoadsStubLibrary`: loads the stub .so through the hypervisor controller to validate the vendor-agnostic path

## Notes

- All struct parameters are carefully designed with key attributes
- Memory management: Use provided cleanup functions to free allocated memory
- Thread safety: Vendor implementations should be thread-safe
- Error handling: All APIs return Result enum for error handling
