#!/bin/bash
# Copyright 2024.
# Quick test script for AMD provider in Docker

set -e

echo "========================================="
echo "AMD Provider Docker Test Script"
echo "========================================="
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

# Check for GPU devices
echo "Checking for AMD GPU devices..."
if [ ! -e /dev/kfd ]; then
    echo "Warning: /dev/kfd not found. AMD GPU may not be available."
fi

if [ ! -d /dev/dri ]; then
    echo "Warning: /dev/dri not found. AMD GPU may not be available."
else
    echo "Found devices:"
    ls -la /dev/dri/ | grep -E "card|render"
fi
echo ""

# Build image if requested
if [ "$1" == "--build" ] || [ "$1" == "-b" ]; then
    echo "Building AMD provider Docker image..."
    echo ""
    cd "$(dirname "$0")/.."
    make docker-build-amd-provider
    echo ""
    echo "Build complete!"
    echo ""
fi

# Run tests in container
echo "Running AMD provider tests in Docker container..."
echo "========================================="
echo ""

docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --cap-add=SYS_PTRACE \
  ghcr.io/saienduri/tensor-fusion-amd-provider \
  /build/bin/test_amd_provider

TEST_RESULT=$?

echo ""
echo "========================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✓ All tests PASSED"
else
    echo "✗ Tests FAILED with exit code $TEST_RESULT"
fi
echo "========================================="

exit $TEST_RESULT
