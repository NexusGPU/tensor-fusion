#!/bin/bash
# Copyright 2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TheRock ROCm Installation Script
# Installs AMD's next-gen ROCm distribution from tarball

set -e

# Parse arguments
ROCM_VERSION="${1:-7.11.0rc0}"
AMDGPU_FAMILY="${2:-gfx94X-dcgpu}"
RELEASE_TYPE="${3:-prereleases}"

echo "========================================="
echo "TheRock ROCm Installation"
echo "========================================="
echo "Version: $ROCM_VERSION"
echo "GPU Family: $AMDGPU_FAMILY"
echo "Release Type: $RELEASE_TYPE"
echo "========================================="

# Determine download URL based on release type
case "$RELEASE_TYPE" in
    stable)
        BASE_URL="https://repo.amd.com/rocm/tarball"
        ;;
    nightlies)
        BASE_URL="https://rocm.nightlies.amd.com/tarball"
        ;;
    prereleases)
        BASE_URL="https://rocm.prereleases.amd.com/tarball"
        ;;
    devreleases)
        BASE_URL="https://rocm.devreleases.amd.com/tarball"
        ;;
    *)
        echo "Error: Unknown release type '$RELEASE_TYPE'"
        echo "Valid types: stable, nightlies, prereleases, devreleases"
        exit 1
        ;;
esac

# Construct tarball filename and URL
TARBALL_NAME="therock-dist-linux-${AMDGPU_FAMILY}-${ROCM_VERSION}.tar.gz"
DOWNLOAD_URL="${BASE_URL}/${TARBALL_NAME}"

echo "Download URL: $DOWNLOAD_URL"
echo ""

# Create temp directory for download
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Download tarball
echo "Downloading TheRock tarball..."
cd "$TEMP_DIR"
if ! wget -q --show-progress "$DOWNLOAD_URL"; then
    echo "Error: Failed to download tarball from $DOWNLOAD_URL"
    echo "Please verify the version, GPU family, and release type are correct."
    exit 1
fi

echo "Download complete."
echo ""

# Extract to /opt
echo "Extracting to /opt/rocm-${ROCM_VERSION}..."
sudo mkdir -p "/opt/rocm-${ROCM_VERSION}"

# Extract tarball - it extracts everything to current directory, so extract to temp first
EXTRACT_DIR=$(mktemp -d)
if ! sudo tar -xzf "$TARBALL_NAME" -C "$EXTRACT_DIR"; then
    echo "Error: Failed to extract tarball"
    sudo rm -rf "$EXTRACT_DIR"
    exit 1
fi

# Move contents to proper location
sudo mv "$EXTRACT_DIR"/* "/opt/rocm-${ROCM_VERSION}/" 2>/dev/null || \
sudo cp -r "$EXTRACT_DIR"/* "/opt/rocm-${ROCM_VERSION}/"
sudo rm -rf "$EXTRACT_DIR"

echo "Extraction complete."
echo ""

# Create symlink /opt/rocm -> /opt/rocm-VERSION
echo "Creating symlink /opt/rocm -> /opt/rocm-${ROCM_VERSION}..."
sudo rm -f /opt/rocm
sudo ln -s "/opt/rocm-${ROCM_VERSION}" /opt/rocm

echo "Symlink created."
echo ""

# Validate installation
echo "Validating installation..."
VALIDATION_FAILED=false

if [ ! -d "/opt/rocm/bin" ]; then
    echo "  ✗ Missing /opt/rocm/bin directory"
    VALIDATION_FAILED=true
else
    echo "  ✓ /opt/rocm/bin exists"
fi

if [ ! -d "/opt/rocm/lib" ]; then
    echo "  ✗ Missing /opt/rocm/lib directory"
    VALIDATION_FAILED=true
else
    echo "  ✓ /opt/rocm/lib exists"
fi

if [ ! -d "/opt/rocm/include" ]; then
    echo "  ✗ Missing /opt/rocm/include directory"
    VALIDATION_FAILED=true
else
    echo "  ✓ /opt/rocm/include exists"
fi

# Check for key libraries
if [ ! -f "/opt/rocm/lib/libamd_smi.so" ] && [ ! -f "/opt/rocm/lib64/libamd_smi.so" ]; then
    echo "  ✗ Missing libamd_smi.so"
    VALIDATION_FAILED=true
else
    echo "  ✓ libamd_smi.so found"
fi

if [ ! -f "/opt/rocm/lib/libamdhip64.so" ] && [ ! -f "/opt/rocm/lib64/libamdhip64.so" ]; then
    echo "  ✗ Missing libamdhip64.so"
    VALIDATION_FAILED=true
else
    echo "  ✓ libamdhip64.so found"
fi

echo ""

if [ "$VALIDATION_FAILED" = true ]; then
    echo "✗ Installation validation failed!"
    echo "Some expected files or directories are missing."
    exit 1
fi

echo "========================================="
echo "✓ TheRock ROCm ${ROCM_VERSION} installed successfully!"
echo "========================================="
echo ""
echo "Installation location: /opt/rocm-${ROCM_VERSION}"
echo "Symlink: /opt/rocm -> /opt/rocm-${ROCM_VERSION}"
echo ""
echo "To use ROCm, add to your environment:"
echo "  export ROCM_PATH=/opt/rocm"
echo "  export PATH=\$ROCM_PATH/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=\$ROCM_PATH/lib:\$LD_LIBRARY_PATH"
echo ""
