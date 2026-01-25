# AMD Provider Docker Testing Guide

This guide covers testing the AMD provider implementation in Docker on MI325X hardware, which closely mirrors the actual Kubernetes deployment environment.

## Prerequisites

### 1. Hardware Requirements
- AMD MI325X GPU(s) installed
- AMD GPU kernel driver loaded (amdgpu module)
- Sufficient system memory (recommended: 32GB+)

### 2. Software Requirements
- Ubuntu 24.04 (or 22.04) host OS
- Docker or Podman installed
- AMD GPU kernel driver on host (amdgpu)
- GPU devices accessible at `/dev/dri/` and `/dev/kfd`

### 3. Verify Host GPU Access

Before building Docker images, ensure GPUs are visible on the host:

```bash
# Check AMD GPU kernel driver is loaded
lsmod | grep amdgpu

# List GPU devices
ls -la /dev/dri/
ls -la /dev/kfd

# Check PCI devices
lspci | grep -i amd | grep -i vga

# If you have rocm-smi installed on host (optional)
rocm-smi
```

## Docker Testing Steps

### Step 1: Build AMD Provider Docker Image

From the TensorFusion repository root:

```bash
cd /home/sai/tensor-fusion

# Build the AMD provider image
# This will install TheRock ROCm and build the provider library
make docker-build-amd-provider
```

This command will:
- Use Ubuntu 24.04 as base
- Download and install TheRock ROCm 7.10.0
- Build `libaccelerator_amd.so`
- Package everything in a container image

The build takes ~10-15 minutes depending on network speed.

Expected output:
```
[+] Building ...
 => [builder 1/8] FROM docker.io/library/ubuntu:24.04
 => [builder 2/8] WORKDIR /workspace
 => [builder 3/8] RUN apt-get update && apt-get install -y ...
 => [builder 4/8] COPY scripts/install_rocm_tarball.sh /tmp/
 => [builder 5/8] RUN bash /tmp/install_rocm_tarball.sh 7.10.0 gfx94X-dcgpu stable
 => [builder 6/8] COPY provider/ provider/
 => [builder 7/8] WORKDIR /workspace/provider
 => [builder 8/8] RUN make amd
 => [stage-1 1/4] FROM docker.io/library/ubuntu:24.04
 => [stage-1 2/4] COPY --from=builder /opt/rocm-* /opt/
 => [stage-1 3/4] COPY --from=builder /workspace/provider/build/libaccelerator_amd.so /build/lib/
 => [stage-1 4/4] RUN echo "version: 1.0.0..." > /build/metadata.yaml
 => exporting to image
 => => naming to tensorfusion/tensor-fusion-amd-provider:latest
```

### Step 2: Run Container with GPU Access

Run the container with GPU device access:

```bash
# Run container with GPU access
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --cap-add=SYS_PTRACE \
  tensorfusion/tensor-fusion-amd-provider:latest \
  /bin/bash
```

**Note**: The `--device` flags expose AMD GPU devices to the container.

You should now be in a shell inside the container.

### Step 3: Verify ROCm Installation in Container

Inside the container, verify ROCm is accessible:

```bash
# Check ROCm environment
echo $ROCM_PATH
echo $LD_LIBRARY_PATH

# Verify ROCm libraries exist
ls -la /opt/rocm/lib/libamd_smi.so
ls -la /opt/rocm/lib/libamdhip64.so

# Check provider library
ls -la /build/lib/libaccelerator_amd.so

# Check metadata
cat /build/metadata.yaml
```

Expected output:
```
/opt/rocm
/opt/rocm/lib:
-rwxr-xr-x 1 root root ... /opt/rocm/lib/libamd_smi.so
-rwxr-xr-x 1 root root ... /opt/rocm/lib/libamdhip64.so
-rwxr-xr-x 1 root root ... /build/lib/libaccelerator_amd.so

version: 1.0.0
hardwareVendor: AMD
releaseDate: "2024-01-24"
isolationModes:
  - shared
rocmDistribution: TheRock
rocmVersion: 7.10.0
```

### Step 4: Test GPU Detection with ROCm Tools

Inside the container, use ROCm tools to verify GPU access:

```bash
# Use rocm-smi to detect GPUs
/opt/rocm/bin/rocm-smi

# Or use amd-smi
/opt/rocm/bin/amd-smi list

# Check device info
/opt/rocm/bin/amd-smi static

# Monitor GPU usage
/opt/rocm/bin/amd-smi monitor
```

You should see output like:
```
========================= ROCm System Management Interface =========================
================================== Concise Info ==================================
GPU  Temp (DieEdge)  AvgPwr  SCLK    MCLK     Fan  Perf    PwrCap  VRAM%  GPU%  
0    35.0c           150W    800Mhz  1600Mhz  0%   manual  500W    0%     0%    
1    36.0c           149W    800Mhz  1600Mhz  0%   manual  500W    0%     0%    
...
====================================================================================
```

### Step 5: Build and Run Unit Tests in Container

Now let's build and run the test suite inside the container:

```bash
# Install build tools (if needed for test compilation)
apt-get update && apt-get install -y gcc make

# Create test binary
cd /workspace
mkdir -p provider/test

# Copy test source (this should already be in image if we update Dockerfile)
# For now, we'll create a simple test

cat > /tmp/test_provider.c << 'EOF'
#include <stdio.h>
#include <dlfcn.h>

int main() {
    printf("Loading AMD provider library...\n");
    
    void* handle = dlopen("/build/lib/libaccelerator_amd.so", RTLD_NOW);
    if (!handle) {
        printf("Failed to load library: %s\n", dlerror());
        return 1;
    }
    
    printf("✓ Library loaded successfully\n");
    
    // Get function pointers
    typedef int (*InitFunc)(void);
    typedef int (*GetDeviceCountFunc)(size_t*);
    
    InitFunc init = (InitFunc)dlsym(handle, "VirtualGPUInit");
    GetDeviceCountFunc getCount = (GetDeviceCountFunc)dlsym(handle, "GetDeviceCount");
    
    if (!init || !getCount) {
        printf("Failed to get function pointers\n");
        dlclose(handle);
        return 1;
    }
    
    printf("✓ Function pointers resolved\n");
    
    // Initialize
    printf("\nInitializing AMD SMI...\n");
    int result = init();
    if (result != 0) {
        printf("✗ Initialization failed with code %d\n", result);
        dlclose(handle);
        return 1;
    }
    printf("✓ AMD SMI initialized\n");
    
    // Get device count
    printf("\nDetecting GPUs...\n");
    size_t deviceCount = 0;
    result = getCount(&deviceCount);
    if (result != 0) {
        printf("✗ GetDeviceCount failed with code %d\n", result);
        dlclose(handle);
        return 1;
    }
    
    printf("✓ Found %zu AMD GPU(s)\n", deviceCount);
    
    dlclose(handle);
    
    printf("\n========================================\n");
    printf("Basic Test: PASSED\n");
    printf("========================================\n");
    
    return 0;
}
EOF

# Compile test
gcc -o /tmp/test_provider /tmp/test_provider.c -ldl

# Run test
/tmp/test_provider
```

Expected output:
```
Loading AMD provider library...
✓ Library loaded successfully
✓ Function pointers resolved

Initializing AMD SMI...
✓ AMD SMI initialized

Detecting GPUs...
✓ Found 8 AMD GPU(s)

========================================
Basic Test: PASSED
========================================
```

### Step 6: Test with HIP (Optional)

Test that HIP runtime can access GPUs:

```bash
cat > /tmp/test_hip.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    
    if (err != hipSuccess) {
        printf("hipGetDeviceCount failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    
    printf("Found %d HIP device(s)\n\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  Compute Units: %d\n", prop.multiProcessorCount);
        printf("  Max Clock: %d MHz\n", prop.clockRate / 1000);
        printf("  PCIe: Gen%d x%d\n", prop.pciDomainID, prop.pciBusID);
        printf("\n");
    }
    
    return 0;
}
EOF

# Install hipcc if not present
apt-get install -y /opt/rocm/bin/hipcc || true

# Compile with hipcc
/opt/rocm/bin/hipcc /tmp/test_hip.cpp -o /tmp/test_hip

# Run
/tmp/test_hip
```

## Advanced Testing: Run Full Test Suite

To run the complete unit test suite, we need to include it in the Docker image. Let me update the Dockerfile to include the tests.

### Updated Testing Process

Exit the current container and rebuild with tests included:

```bash
# Exit container
exit

# We'll update the Dockerfile to copy tests, then rebuild
# (I'll create an updated Dockerfile in the next step)
```

## Verification Checklist

### ✓ Docker Image Build
- [ ] Image builds without errors
- [ ] TheRock ROCm installed successfully
- [ ] Provider library compiled and included
- [ ] Metadata file created

### ✓ Container Runtime
- [ ] Container starts with GPU devices
- [ ] `/dev/kfd` and `/dev/dri/*` accessible
- [ ] ROCm libraries present and loadable
- [ ] Provider library present

### ✓ GPU Access
- [ ] `rocm-smi` shows all GPUs
- [ ] `amd-smi` lists devices
- [ ] Device info matches hardware (MI325X)
- [ ] HIP runtime detects GPUs

### ✓ Provider Functionality
- [ ] Library loads successfully
- [ ] AMD SMI initializes
- [ ] Device count accurate
- [ ] All GPUs detected

### ✓ Expected Values for MI325X
- [ ] Memory: ~192 GB per GPU
- [ ] Compute Units: 304 per GPU
- [ ] Model: MI325X or similar
- [ ] Vendor: AMD

## Troubleshooting

### Issue: "Cannot access /dev/kfd: Permission denied"

**Solution**: Ensure proper device permissions and groups:
```bash
# On host, check device ownership
ls -la /dev/kfd /dev/dri/

# Run container with additional permissions
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --group-add render \
  --cap-add=SYS_PTRACE \
  --privileged \
  tensorfusion/tensor-fusion-amd-provider:latest \
  /bin/bash
```

### Issue: "amdsmi_init failed" or "No GPUs detected"

**Possible causes**:
1. GPU devices not properly passed to container
2. Driver mismatch between host and container
3. Insufficient permissions

**Solution**:
```bash
# Check host has working driver
sudo dmesg | grep amdgpu

# Verify devices exist on host
ls -la /dev/dri/ /dev/kfd

# Try with --privileged flag (for testing only)
docker run -it --rm --privileged \
  tensorfusion/tensor-fusion-amd-provider:latest \
  /bin/bash
```

### Issue: Docker build fails during TheRock installation

**Solution**: Check network connectivity and try different release type:
```bash
# Build with nightlies if stable fails
docker build -f dockerfile/amd-provider.Dockerfile \
  --build-arg RELEASE_TYPE=nightlies \
  -t tensorfusion/tensor-fusion-amd-provider:latest .
```

### Issue: "hipcc: command not found"

**Solution**: Use full path:
```bash
/opt/rocm/bin/hipcc /tmp/test_hip.cpp -o /tmp/test_hip
```

## Performance Validation

### Memory Bandwidth Test in Container

```bash
# Simple memory copy test
cat > /tmp/bandwidth_test.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <sys/time.h>

int main() {
    const size_t size = 1024 * 1024 * 1024; // 1GB
    void *d_ptr;
    void *h_ptr = malloc(size);
    
    hipMalloc(&d_ptr, size);
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    hipMemcpy(d_ptr, h_ptr, size, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    
    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_usec - start.tv_usec) / 1000000.0;
    double bandwidth = (size / (1024.0 * 1024.0 * 1024.0)) / elapsed;
    
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth);
    
    hipFree(d_ptr);
    free(h_ptr);
    
    return 0;
}
EOF

/opt/rocm/bin/hipcc /tmp/bandwidth_test.cpp -o /tmp/bandwidth_test
/tmp/bandwidth_test
```

Expected: ~50-100 GB/s for PCIe Gen5 x16

## Next Steps

Once Docker-based testing is successful:

1. ✓ **Phase 1-2 Complete**: AMD provider works in Docker
2. → **Continue to Phase 3**: Helm chart integration for Kubernetes
3. → **Phase 4**: Deploy to actual Kubernetes cluster
4. → **Phase 5**: Documentation
5. → **Phase 6**: Remote GPU execution (GPU-over-IP)

## Saving Test Results

To save logs and results from container:

```bash
# Run container with volume mount
docker run -it --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  -v /tmp/test-results:/results \
  tensorfusion/tensor-fusion-amd-provider:latest \
  /bin/bash

# Inside container, save results
/opt/rocm/bin/rocm-smi > /results/rocm-smi-output.txt
/opt/rocm/bin/amd-smi static > /results/amd-smi-static.txt
# ... run tests and save outputs to /results/

# Exit container
exit

# View results on host
cat /tmp/test-results/*.txt
```

## Docker Compose Alternative (Optional)

For easier testing, create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  amd-provider-test:
    image: tensorfusion/tensor-fusion-amd-provider:latest
    devices:
      - /dev/kfd
      - /dev/dri
    security_opt:
      - seccomp:unconfined
    cap_add:
      - SYS_PTRACE
    group_add:
      - video
    volumes:
      - /tmp/test-results:/results
    stdin_open: true
    tty: true
    command: /bin/bash
```

Then run:
```bash
docker-compose run amd-provider-test
```
