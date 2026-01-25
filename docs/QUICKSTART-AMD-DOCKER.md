# Quick Start: AMD Provider Docker Testing

Test the AMD provider implementation in Docker on your MI325X machine.

## Prerequisites

- Docker installed
- AMD GPU kernel driver loaded (`lsmod | grep amdgpu`)
- GPU devices available (`ls /dev/kfd /dev/dri/`)

## One-Command Test

```bash
cd /home/sai/tensor-fusion

# Build image and run tests
./scripts/test-amd-docker.sh --build
```

This will:
1. Build the Docker image with TheRock ROCm 7.11.0rc0 (prereleases)
2. Run the complete test suite
3. Report results

## Manual Testing

### Step 1: Build Image

```bash
make docker-build-amd-provider
```

This builds with TheRock 7.11.0rc0 (prereleases) by default.

### Step 2: Run Tests

```bash
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --cap-add=SYS_PTRACE \
  tensorfusion/tensor-fusion-amd-provider:latest \
  /build/bin/test_amd_provider
```

### Step 3: Interactive Testing

```bash
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --cap-add=SYS_PTRACE \
  tensorfusion/tensor-fusion-amd-provider:latest \
  /bin/bash

# Inside container:
# - Run tests: /build/bin/test_amd_provider
# - Check GPUs: /opt/rocm/bin/rocm-smi
# - View metadata: cat /build/metadata.yaml
```

## Expected Results

```
========================================
AMD Provider Test Suite
========================================
[INFO] AMD SMI initialized successfully
[INFO] Discovered 8 device(s)

Device 0:
  UUID: AMD-GPU-0000:41:00.0
  Vendor: AMD
  Model: MI325X
  Memory: 196608000000 bytes (183.11 GB)
  Compute Units: 304
  ...

========================================
Test Summary
========================================
Passed: 45
Failed: 0
Total:  45

✓ All tests passed!
```

## Troubleshooting

**Issue**: "Cannot access /dev/kfd"
- **Solution**: Ensure user is in `video` group: `sudo usermod -a -G video $USER`
- Then logout/login

**Issue**: "No GPUs detected"
- **Check**: `lsmod | grep amdgpu` (driver loaded?)
- **Check**: `ls -la /dev/dri /dev/kfd` (devices exist?)
- **Try**: Run with `--privileged` flag (testing only)

**Issue**: Build fails
- **Check**: Internet connectivity
- **Try**: Different ROCm version: `--build-arg ROCM_VERSION=7.10.0 --build-arg RELEASE_TYPE=stable`

## Full Documentation

See `docs/amd-bare-metal-testing.md` for comprehensive testing guide.
