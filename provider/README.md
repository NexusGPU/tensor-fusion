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
│   └── accelerator.c     # Huawei Ascend implementation
└── test/
    └── test_accelerator.c # Test suite
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
```

## Interface Categories

### 1. DeviceInfo APIs

- `getDeviceInfo()`: Get device information (capabilities, basic info, NUMA, etc.)
- `getPartitionTemplates()`: Get hardware partition templates (e.g., MIG)
- `getDeviceTopology()`: Get device topology (NVLink, IB NIC, etc.)

### 2. Virtualization APIs

#### Partitioned Isolation
- `assignPartition()`: Assign hardware partition (returns partitionOverhead)
- `removePartition()`: Remove partition

#### Hard Isolation
- `setMemHardLimit()`: Set hard memory limit (one-time)
- `setComputeUnitHardLimit()`: Set hard compute limit (one-time)

#### Snapshot/Migration
- `snapshot()`: Snapshot device state for processes
- `resume()`: Resume device state for processes

### 3. Metrics APIs

- `getProcessComputeUtilization()`: Get compute utilization per process
- `getProcessMemoryUtilization()`: Get memory utilization per process
- `getDeviceMetrics()`: Get basic device metrics (power, PCIe, SM active, TC usage)
- `getExtendedDeviceMetrics()`: Get extended metrics (NVLink bandwidth, etc.)

## Vendor Implementations

### Stub Implementation

The stub implementation (`stub/accelerator.c`) provides a reference implementation for testing and development.

### Ascend Implementation

The Ascend implementation (`ascend/accelerator.c`) provides support for Huawei Ascend accelerators:

- Supports Soft and Hard isolation modes
- Does not support hardware partitioning (MIG-like features)
- Uses HCCS (Huawei Cache Coherent System) for device interconnects
- Typical device: Ascend 910 with 32GB memory, 2 AI cores, 320 TFLOPS (FP16)

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
========================================
Accelerator Library Test Suite
========================================
Total tests:  47
Passed:       47
Failed:       0
All tests passed! ✓
```

## Notes

- All struct parameters are carefully designed with key attributes
- Memory management: Use provided cleanup functions to free allocated memory
- Thread safety: Vendor implementations should be thread-safe
- Error handling: All APIs return Result enum for error handling

