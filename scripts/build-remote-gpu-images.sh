#!/bin/bash
# Build and push AMD remote GPU Docker images for TensorFusion
#
# Images built:
#   - AMD Provider: Contains ROCm, provider library, and HIP client stub
#                   Used by TensorFusion init container to inject libraries
#   - HIP Worker:   Contains ROCm and HIP worker service
#                   Runs on GPU nodes to execute remote HIP calls
#
# Note: Operator and Hypervisor images are built by build-amd-images.sh
#
# Usage:
#   ./scripts/build-remote-gpu-images.sh [--push] [--registry REGISTRY]
#
# Options:
#   --push              Push images to registry after building
#   --registry REG      Registry prefix (default: ghcr.io/saienduri)
#   --tag TAG           Image tag (default: latest)
#   --provider-only     Build only the provider image
#   --worker-only       Build only the worker service image
#   --rocm-version VER  ROCm version (default: 7.11.0rc0)
#   --amdgpu-family FAM AMDGPU family (default: gfx94X-dcgpu)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default configuration
REGISTRY="ghcr.io/saienduri"
TAG="latest"
PUSH=false
BUILD_PROVIDER=true
BUILD_WORKER=true
ROCM_VERSION="7.11.0rc0"
AMDGPU_FAMILY="gfx94X-dcgpu"
RELEASE_TYPE="prereleases"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --provider-only)
            BUILD_WORKER=false
            shift
            ;;
        --worker-only)
            BUILD_PROVIDER=false
            shift
            ;;
        --rocm-version)
            ROCM_VERSION="$2"
            shift 2
            ;;
        --amdgpu-family)
            AMDGPU_FAMILY="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--push] [--registry REG] [--tag TAG]"
            echo ""
            echo "Builds AMD remote GPU images for TensorFusion:"
            echo "  - Provider image: For init container library injection"
            echo "  - Worker image:   For executing remote HIP calls on GPU nodes"
            echo ""
            echo "Note: Operator and Hypervisor are built by build-amd-images.sh"
            echo ""
            echo "Options:"
            echo "  --push              Push images to registry after building"
            echo "  --registry REG      Registry prefix (default: ghcr.io/saienduri)"
            echo "  --tag TAG           Image tag (default: latest)"
            echo "  --provider-only     Build only the provider image"
            echo "  --worker-only       Build only the worker service image"
            echo "  --rocm-version VER  ROCm version (default: 7.11.0rc0)"
            echo "  --amdgpu-family FAM AMDGPU family (default: gfx94X-dcgpu)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Image names
PROVIDER_IMAGE="${REGISTRY}/tensor-fusion-amd-provider:${TAG}"
WORKER_IMAGE="${REGISTRY}/tensor-fusion-hip-worker:${TAG}"

echo "========================================"
echo "TensorFusion AMD Remote GPU Builder"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Registry:      ${REGISTRY}"
echo "  Tag:           ${TAG}"
echo "  ROCm Version:  ${ROCM_VERSION}"
echo "  AMDGPU Family: ${AMDGPU_FAMILY}"
echo "  Push:          ${PUSH}"
echo ""

cd "$PROJECT_ROOT"

# Build AMD provider image (includes HIP client stub)
if [ "$BUILD_PROVIDER" = true ]; then
    echo "========================================"
    echo "Building AMD Provider Image"
    echo "========================================"
    echo "Image: ${PROVIDER_IMAGE}"
    echo "Contains: ROCm, provider library, HIP client stub"
    echo ""
    
    docker build \
        -f dockerfile/amd-provider.Dockerfile \
        --build-arg ROCM_VERSION="${ROCM_VERSION}" \
        --build-arg AMDGPU_FAMILY="${AMDGPU_FAMILY}" \
        --build-arg RELEASE_TYPE="${RELEASE_TYPE}" \
        -t "${PROVIDER_IMAGE}" \
        .
    
    echo ""
    echo "[OK] Provider image built: ${PROVIDER_IMAGE}"
    
    # Show image size
    docker images "${PROVIDER_IMAGE}" --format "Size: {{.Size}}"
    echo ""
fi

# Build worker service image
if [ "$BUILD_WORKER" = true ]; then
    echo "========================================"
    echo "Building HIP Worker Image"
    echo "========================================"
    echo "Image: ${WORKER_IMAGE}"
    echo "Contains: ROCm, HIP worker service"
    echo ""
    
    docker build \
        -f dockerfile/amd-worker.Dockerfile \
        --build-arg ROCM_VERSION="${ROCM_VERSION}" \
        --build-arg AMDGPU_FAMILY="${AMDGPU_FAMILY}" \
        --build-arg RELEASE_TYPE="${RELEASE_TYPE}" \
        -t "${WORKER_IMAGE}" \
        .
    
    echo ""
    echo "[OK] Worker image built: ${WORKER_IMAGE}"
    
    # Show image size
    docker images "${WORKER_IMAGE}" --format "Size: {{.Size}}"
    echo ""
fi

# Push images if requested
if [ "$PUSH" = true ]; then
    echo "========================================"
    echo "Pushing Images to Registry"
    echo "========================================"
    
    if [ "$BUILD_PROVIDER" = true ]; then
        echo "Pushing ${PROVIDER_IMAGE}..."
        docker push "${PROVIDER_IMAGE}"
        echo "[OK] Provider image pushed"
    fi
    
    if [ "$BUILD_WORKER" = true ]; then
        echo "Pushing ${WORKER_IMAGE}..."
        docker push "${WORKER_IMAGE}"
        echo "[OK] Worker image pushed"
    fi
    
    echo ""
fi

# Summary
echo "========================================"
echo "Build Summary"
echo "========================================"

if [ "$BUILD_PROVIDER" = true ]; then
    echo "Provider: ${PROVIDER_IMAGE}"
    echo "  - Used by: TensorFusion init container"
    echo "  - Contains: libhip_client_stub.so, libaccelerator_amd.so"
fi

if [ "$BUILD_WORKER" = true ]; then
    echo "Worker:   ${WORKER_IMAGE}"
    echo "  - Used by: Worker pods on GPU nodes"
    echo "  - Contains: hip_worker_service"
fi

echo ""
echo "Quick test commands:"
echo ""

if [ "$BUILD_PROVIDER" = true ]; then
    echo "# Verify provider image contents:"
    echo "docker run --rm ${PROVIDER_IMAGE} ls -la /build/lib/"
    echo ""
fi

if [ "$BUILD_WORKER" = true ]; then
    echo "# Test worker (requires GPU):"
    echo "docker run --rm --device=/dev/kfd --device=/dev/dri -e TF_DEBUG=1 ${WORKER_IMAGE}"
    echo ""
fi

echo ""
echo "Note: For operator and hypervisor images, use:"
echo "  ./scripts/build-amd-images.sh"
echo ""

if [ "$PUSH" = false ]; then
    echo "To push images, run with --push flag:"
    echo "  $0 --push"
fi
