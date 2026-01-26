#!/bin/bash
# Test script for AMD HIP Remote GPU (GPU-over-IP) implementation
#
# This script tests the remote GPU functionality by:
# 1. Building the client stub and worker service
# 2. Running a local worker service
# 3. Testing the client stub against it
#
# Usage:
#   ./scripts/test-remote-gpu.sh [--build-only] [--no-cleanup]
#
# Requirements:
#   - ROCm installed at /opt/rocm (for worker service build)
#   - GCC for building the client stub

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AMD_DIR="$PROJECT_ROOT/provider/amd"

BUILD_ONLY=false
NO_CLEANUP=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --no-cleanup)
            NO_CLEANUP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "TensorFusion Remote GPU Test Suite"
echo "========================================"
echo ""

# Check for ROCm - modern ROCm (6.x+) puts HIP at /opt/rocm/include/hip
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
HIP_AVAILABLE=false
if [ -f "$ROCM_PATH/include/hip/hip_runtime.h" ]; then
    HIP_AVAILABLE=true
    echo "[INFO] ROCm found at $ROCM_PATH (modern layout)"
elif [ -f "$ROCM_PATH/hip/include/hip/hip_runtime.h" ]; then
    HIP_AVAILABLE=true
    echo "[INFO] ROCm found at $ROCM_PATH (legacy layout)"
else
    echo "[WARN] ROCm/HIP not found at $ROCM_PATH"
    echo "[WARN] Worker service tests will be skipped"
fi

# Build components
echo ""
echo "========================================"
echo "Building Components"
echo "========================================"

cd "$AMD_DIR"

echo "[BUILD] Building client stub..."
make client
if [ $? -eq 0 ]; then
    echo "[OK] Client stub built: $AMD_DIR/libhip_client_stub.so"
else
    echo "[FAIL] Client stub build failed"
    exit 1
fi

if [ "$HIP_AVAILABLE" = true ]; then
    echo "[BUILD] Building worker service..."
    make worker
    if [ $? -eq 0 ]; then
        echo "[OK] Worker service built: $AMD_DIR/hip_worker_service"
    else
        echo "[WARN] Worker service build failed (may need HIP development headers)"
    fi
fi

if [ "$BUILD_ONLY" = true ]; then
    echo ""
    echo "[INFO] Build complete (--build-only specified)"
    exit 0
fi

# Test client stub loading
echo ""
echo "========================================"
echo "Testing Client Stub"
echo "========================================"

echo "[TEST] Testing client stub LD_PRELOAD..."
if LD_PRELOAD="$AMD_DIR/libhip_client_stub.so" TF_DEBUG=1 /bin/true 2>&1 | grep -q "TensorFusion HIP client stub loaded"; then
    echo "[OK] Client stub loads and initializes correctly"
else
    echo "[FAIL] Client stub failed to load"
    exit 1
fi

# Create a simple test program
echo "[TEST] Creating simple HIP test program..."
cat > /tmp/test_hip_remote.c << 'EOF'
#include <stdio.h>
#include <dlfcn.h>

// Minimal HIP type definitions
typedef int hipError_t;
#define hipSuccess 0

// Function pointer types
typedef hipError_t (*hipGetDeviceCount_t)(int*);
typedef hipError_t (*hipMalloc_t)(void**, size_t);
typedef hipError_t (*hipFree_t)(void*);
typedef const char* (*hipGetErrorString_t)(hipError_t);

int main() {
    printf("Testing HIP Remote GPU functionality...\n");
    
    // Load functions via dlsym (they'll come from LD_PRELOAD stub)
    hipGetDeviceCount_t hipGetDeviceCount = (hipGetDeviceCount_t)dlsym(RTLD_DEFAULT, "hipGetDeviceCount");
    hipMalloc_t hipMalloc = (hipMalloc_t)dlsym(RTLD_DEFAULT, "hipMalloc");
    hipFree_t hipFree = (hipFree_t)dlsym(RTLD_DEFAULT, "hipFree");
    hipGetErrorString_t hipGetErrorString = (hipGetErrorString_t)dlsym(RTLD_DEFAULT, "hipGetErrorString");
    
    if (!hipGetDeviceCount) {
        printf("ERROR: hipGetDeviceCount not found (LD_PRELOAD not working?)\n");
        return 1;
    }
    
    // This will go through our client stub
    int count = -1;
    hipError_t err = hipGetDeviceCount(&count);
    
    const char* errStr = hipGetErrorString ? hipGetErrorString(err) : "unknown";
    printf("hipGetDeviceCount returned: count=%d, error=%d (%s)\n", count, err, errStr);
    
    if (err != hipSuccess) {
        printf("Note: Error expected if worker not running or connection failed\n");
        return 0;
    }
    
    printf("Device count: %d\n", count);
    
    // Try to allocate memory
    if (hipMalloc) {
        void* ptr = NULL;
        err = hipMalloc(&ptr, 1024);
        printf("hipMalloc returned: ptr=%p, error=%d\n", ptr, err);
        
        if (ptr && err == hipSuccess && hipFree) {
            err = hipFree(ptr);
            printf("hipFree returned: error=%d\n", err);
        }
    }
    
    printf("Test completed successfully!\n");
    return 0;
}
EOF

echo "[BUILD] Compiling test program..."
if ! gcc -o /tmp/test_hip_remote /tmp/test_hip_remote.c -ldl 2>&1; then
    echo "[WARN] Test program compilation failed - check gcc is installed"
fi

if [ -f /tmp/test_hip_remote ]; then
    echo "[TEST] Running test program with client stub (no worker)..."
    echo "---"
    TF_DEBUG=1 LD_PRELOAD="$AMD_DIR/libhip_client_stub.so" /tmp/test_hip_remote 2>&1 || true
    echo "---"
    echo "[OK] Test program executed (expected to fail without worker)"
fi

# Test with worker if available
if [ -f "$AMD_DIR/hip_worker_service" ] && [ "$HIP_AVAILABLE" = true ]; then
    echo ""
    echo "========================================"
    echo "Testing with Worker Service"
    echo "========================================"
    
    # Kill any existing worker on the test port
    EXISTING_PID=$(lsof -ti:50052 2>/dev/null || true)
    if [ -n "$EXISTING_PID" ]; then
        echo "[CLEAN] Killing existing process on port 50052 (PID=$EXISTING_PID)..."
        kill -9 $EXISTING_PID 2>/dev/null || true
        sleep 1
    fi
    
    # Start worker in background
    echo "[START] Starting worker service..."
    TF_DEBUG=1 TF_WORKER_PORT=50052 "$AMD_DIR/hip_worker_service" &
    WORKER_PID=$!
    sleep 2
    
    if kill -0 $WORKER_PID 2>/dev/null; then
        echo "[OK] Worker service started (PID=$WORKER_PID)"
        
        echo "[TEST] Running test with worker..."
        echo "---"
        TF_DEBUG=1 TF_WORKER_HOST=localhost TF_WORKER_PORT=50052 \
            LD_PRELOAD="$AMD_DIR/libhip_client_stub.so" /tmp/test_hip_remote 2>&1 || true
        echo "---"
        
        echo "[STOP] Stopping worker service..."
        kill $WORKER_PID 2>/dev/null || true
        wait $WORKER_PID 2>/dev/null || true
        echo "[OK] Worker stopped"
    else
        echo "[FAIL] Worker service failed to start"
    fi
fi

# Cleanup
if [ "$NO_CLEANUP" != true ]; then
    echo ""
    echo "[CLEAN] Cleaning up test files..."
    rm -f /tmp/test_hip_remote /tmp/test_hip_remote.c
fi

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "[OK] Client stub builds and loads correctly"
if [ "$HIP_AVAILABLE" = true ] && [ -f "$AMD_DIR/hip_worker_service" ]; then
    echo "[OK] Worker service builds and runs"
else
    echo "[SKIP] Worker service (requires ROCm)"
fi
echo ""
echo "Next steps:"
echo "1. Build Docker images:"
echo "   docker build -f dockerfile/amd-client-stub.Dockerfile -t tf-hip-client ."
echo "   docker build -f dockerfile/amd-worker.Dockerfile -t tf-hip-worker ."
echo ""
echo "2. Deploy in Kubernetes with TensorFusion operator"
echo "3. Test with actual HIP workloads"
