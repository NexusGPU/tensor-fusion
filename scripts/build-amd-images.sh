#!/bin/bash
set -e

# Build script for TensorFusion AMD support
# This builds both the operator and hypervisor images with AMD support

OPERATOR_IMG="ghcr.io/saienduri/tensor-fusion-operator:amd-support"
HYPERVISOR_IMG="ghcr.io/saienduri/tensor-fusion-hypervisor:amd-support"
VERSION="amd-support"

echo "========================================="
echo "Building TensorFusion AMD Support Images"
echo "========================================="

cd "$(dirname "$0")/.."

# Build operator image (for controller in control plane)
echo ""
echo "1. Building operator image..."
docker build \
  -f dockerfile/operator.Dockerfile \
  --build-arg TARGETOS=linux \
  --build-arg TARGETARCH=amd64 \
  --build-arg GO_LDFLAGS="-X 'github.com/NexusGPU/tensor-fusion/internal/version.BuildVersion=${VERSION}'" \
  -t "${OPERATOR_IMG}" \
  .

echo "✓ Operator image built: ${OPERATOR_IMG}"

# Build hypervisor image (for GPU nodes)
echo ""
echo "2. Building hypervisor image..."
docker build \
  -f dockerfile/hypervisor.Dockerfile \
  --build-arg TARGETOS=linux \
  --build-arg TARGETARCH=amd64 \
  -t "${HYPERVISOR_IMG}" \
  .

echo "✓ Hypervisor image built: ${HYPERVISOR_IMG}"

# Push images
echo ""
echo "3. Pushing images to registry..."
docker push "${OPERATOR_IMG}"
docker push "${HYPERVISOR_IMG}"

echo ""
echo "========================================="
echo "✓ All images built and pushed successfully!"
echo "========================================="
echo ""
echo "Operator image:    ${OPERATOR_IMG}"
echo "Hypervisor image:  ${HYPERVISOR_IMG}"
echo ""
echo "Note: AMD provider image should already be built:"
echo "  ghcr.io/saienduri/tensor-fusion-amd-provider:latest"
echo ""
