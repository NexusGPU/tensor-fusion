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
├── example/
│   └── accelerator.c     # Example implementation for testing
└── test/
    └── test_accelerator.c # Test suite
```

## Building

### Build Example Implementation

```bash
cd provider
make example
```

### Run Tests

```bash
cd provider
make test-run
```

## Interface Categories

### 0. Initialization

- `AccelInit()`: Initialize the accelerator library (must be called before other APIs)
- `AccelShutdown()`: Shutdown the accelerator library

### 1. DeviceInfo APIs

- `AccelGetDeviceCount()`: Get the number of available devices
- `AccelGetAllDevices()`: Get all available devices information (returns `ExtendedDeviceInfo` with basic info, properties, and virtualization capabilities)
- `AccelGetAllDevicesTopology()`: Get device topology including NVLink, IB NIC, and other interconnects

### 2. Virtualization APIs

#### Partitioned Isolation
- `AccelAssignPartition()`: Assign a partition to a device using a template (e.g., create MIG instance)
- `AccelRemovePartition()`: Remove a partition from a device

#### Hard Isolation
- `AccelSetMemHardLimit()`: Set hard memory limit for a worker (one-time, called at worker start by limiter.so)
- `AccelSetComputeUnitHardLimit()`: Set hard compute unit limit for a worker (one-time, called at worker start)

#### Snapshot/Migration
- `AccelSnapshot()`: Snapshot device state for processes (lock processes, checkpoint state)
- `AccelResume()`: Resume device state for processes (unlock processes, restore state)

### 3. Metrics APIs

- `AccelGetProcessInformation()`: Get process information (compute and memory utilization) for all processes on all devices. Combines compute and memory utilization into a single call (AMD SMI style API design)
- `AccelGetDeviceMetrics()`: Get basic device metrics (power, temperature, PCIe bandwidth, utilization, memory usage, and extra vendor-specific metrics)
- `AccelGetVendorMountLibs()`: Get vendor mount paths for additional device driver or runtime libraries

### 4. Utility APIs

- `AccelRegisterLogCallback()`: Register a log callback function for library logging

## Key Data Structures

### Device Information
- `DeviceBasicInfo`: UUID, vendor, model, driver/firmware versions, memory, compute units, PCIe info
- `VirtualizationCapabilities`: Partitioning, soft/hard isolation, snapshot, metrics, remoting support
- `ExtendedDeviceInfo`: Combines basic info, properties, and capabilities

### Virtualization
- `PartitionAssignment`: Partition request with template ID, device UUID, and optional env vars
- `WorkerInfo`: Worker identifier, device UUID, process ID, and resource limits
- `ProcessArray`: Array of process IDs for snapshot/resume operations

### Metrics
- `ProcessInformation`: Per-process compute and memory utilization
- `DeviceMetrics`: Device-level power, temperature, PCIe, utilization, and extra metrics

## Result Codes

All APIs return an `AccelResult` enum:
- `ACCEL_SUCCESS`: Operation succeeded
- `ACCEL_ERROR_INVALID_PARAM`: Invalid parameter
- `ACCEL_ERROR_NOT_FOUND`: Resource not found
- `ACCEL_ERROR_NOT_SUPPORTED`: Operation not supported
- `ACCEL_ERROR_RESOURCE_EXHAUSTED`: Resource exhausted
- `ACCEL_ERROR_OPERATION_FAILED`: Operation failed
- `ACCEL_ERROR_INTERNAL`: Internal error

## Vendor Implementations

### Example Implementation

The stub implementation (`example/accelerator.c`) provides a reference implementation for testing and development.

## Usage in Hypervisor

The hypervisor uses the accelerator library via CGO bindings:

```go
import "github.com/NexusGPU/tensor-fusion/pkg/hypervisor/device"

mgr, err := device.NewManager("path/to/libaccelerator.so", 30*time.Second)
```

See `internal/hypervisor/device/` for the Go bindings and device manager implementation.

## ProviderConfig extraEnv Note

When using `ProviderConfig.spec.hypervisor.extraEnv`, values are appended to the
hypervisor process environment and may be forwarded to worker container env vars
through partition assignment results.

For Ascend partition mode:

- `TF_ASCEND_PARTITION_RUNTIME_OPTIONS` is intended for static vNPU workflows
  that require `ASCEND_RUNTIME_OPTIONS`.
- In dynamic partition workflows (for example, using `ASCEND_VISIBLE_DEVICES`
  with `ASCEND_VNPU_SPECS`), this is usually not needed.

## Testing

All tests pass successfully:

```bash
$ make test-run
```

## Vendor Artifacts

- /build/vgpu/lib_{vendor}.so: Implements partitioned, soft, hard isolation, need to be loaded before app running, and used by hypervisor for GPU discovery and monitoring
- /build/metadata.yaml: package metadata

TensorFusion will create hostPath and mount to Hypervisor, copy vendor artifacts from /build/** to the host path, and then mount to worker containers.

```yaml
version: 1.2.3
hardwareVendor: NVIDIA
releaseDate: "2025-01-01"
isolationModes:
  - partitioned
  - soft
  - hard
```

## Notes

- All struct parameters are carefully designed with fixed-size arrays for ABI stability
- Memory management: Caller allocates output buffers, library fills them
- Thread safety: Vendor implementations should be thread-safe
- Error handling: All APIs return `AccelResult` enum for error handling
- Logging: Use `AccelRegisterLogCallback()` to receive library log messages
