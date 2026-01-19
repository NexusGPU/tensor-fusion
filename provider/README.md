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

- `VirtualGPUInit()`: Initialize the virtual GPU library (must be called before other APIs)

### 1. DeviceInfo APIs

- `GetDeviceCount()`: Get the number of available devices
- `GetAllDevices()`: Get all available devices information (returns `ExtendedDeviceInfo` with basic info, properties, and virtualization capabilities)
- `GetDeviceTopology()`: Get device topology including NVLink, IB NIC, and other interconnects

### 2. Virtualization APIs

#### Partitioned Isolation
- `AssignPartition()`: Assign a partition to a device using a template (e.g., create MIG instance)
- `RemovePartition()`: Remove a partition from a device

#### Hard Isolation
- `SetMemHardLimit()`: Set hard memory limit for a worker (one-time, called at worker start by limiter.so)
- `SetComputeUnitHardLimit()`: Set hard compute unit limit for a worker (one-time, called at worker start)

#### Snapshot/Migration
- `Snapshot()`: Snapshot device state for processes (lock processes, checkpoint state)
- `Resume()`: Resume device state for processes (unlock processes, restore state)

### 3. Metrics APIs

- `GetProcessInformation()`: Get process information (compute and memory utilization) for all processes on all devices. Combines compute and memory utilization into a single call (AMD SMI style API design)
- `GetDeviceMetrics()`: Get basic device metrics (power, temperature, PCIe bandwidth, utilization, memory usage, and extra vendor-specific metrics)
- `GetVendorMountLibs()`: Get vendor mount paths for additional device driver or runtime libraries

### 4. Utility APIs

- `RegisterLogCallback()`: Register a log callback function for library logging

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

All APIs return a `Result` enum:
- `RESULT_SUCCESS`: Operation succeeded
- `RESULT_ERROR_INVALID_PARAM`: Invalid parameter
- `RESULT_ERROR_NOT_FOUND`: Resource not found
- `RESULT_ERROR_NOT_SUPPORTED`: Operation not supported
- `RESULT_ERROR_RESOURCE_EXHAUSTED`: Resource exhausted
- `RESULT_ERROR_OPERATION_FAILED`: Operation failed
- `RESULT_ERROR_INTERNAL`: Internal error

## Vendor Implementations

### Example Implementation

The stub implementation (`example/accelerator.c`) provides a reference implementation for testing and development.

## Usage in Hypervisor

The hypervisor uses the accelerator library via CGO bindings:

```go
import "github.com/NexusGPU/tensor-fusion/internal/hypervisor/device"

mgr, err := device.NewManager("path/to/libaccelerator.so", 30*time.Second)
```

See `internal/hypervisor/device/` for the Go bindings and device manager implementation.

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
- Error handling: All APIs return `Result` enum for error handling
- Logging: Use `RegisterLogCallback()` to receive library log messages
