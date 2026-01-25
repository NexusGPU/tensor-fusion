#!/bin/bash
# Copyright 2024.
# TensorFusion Init Container Entrypoint
# Sets up GPU interception for soft/hard isolation modes
# For shared mode, no interception is needed

set -e

echo "TensorFusion: Starting init for ${HARDWARE_VENDOR:-UNKNOWN}"

# Directories
LIBS_DIR="${TF_LIBS_DIR:-/tensor-fusion}"
CONF_DIR="${TF_CONF_DIR:-/tensor-fusion-conf}"

# Create target directories
mkdir -p "${LIBS_DIR}"
mkdir -p "${CONF_DIR}"

# Get isolation mode from environment (set by webhook)
ISOLATION_MODE="${TF_ISOLATION_MODE:-shared}"

echo "Isolation mode: ${ISOLATION_MODE}"

if [ "${ISOLATION_MODE}" = "shared" ]; then
    # Shared mode: No interception needed, application uses GPU directly
    echo "Shared isolation mode - no GPU call interception needed"
    echo "Application will use native ROCm/CUDA libraries directly"
    
    # Create empty config files so mounts don't fail
    touch "${CONF_DIR}/tensor-fusion.conf"
    touch "${CONF_DIR}/ld.so.preload"
    
    echo "TensorFusion: Init complete (pass-through mode)"
else
    # Soft/Hard modes: Need interception (requires limiter library, not provider library)
    echo "TODO: Soft/hard isolation modes require limiter library (not yet implemented for AMD)"
    echo "For now, falling back to shared mode behavior"
    
    # Create empty config files
    touch "${CONF_DIR}/tensor-fusion.conf"
    touch "${CONF_DIR}/ld.so.preload"
    
    echo "TensorFusion: Init complete (pass-through mode)"
fi

