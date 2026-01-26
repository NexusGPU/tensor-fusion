# AMD HIP Worker Service Container
#
# This container runs the TensorFusion worker service that executes
# HIP API calls on behalf of remote clients.
#
# Usage:
#   Build:  docker build -f dockerfile/amd-worker.Dockerfile -t tensor-fusion/amd-worker .
#   Run:    docker run --device=/dev/kfd --device=/dev/dri -p 50051:50051 tensor-fusion/amd-worker
#
# Environment variables:
#   TF_WORKER_PORT - Port to listen on (default: 50051)
#   TF_DEVICE_ID   - GPU device ID to use (default: 0)
#   TF_DEBUG       - Enable debug logging (set to 1)

FROM ubuntu:24.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    python3-venv \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Copy and run TheRock installation script
COPY scripts/install_rocm_tarball.sh /tmp/

ARG ROCM_VERSION=7.11.0rc0
ARG AMDGPU_FAMILY=gfx94X-dcgpu
ARG RELEASE_TYPE=prereleases

RUN bash /tmp/install_rocm_tarball.sh ${ROCM_VERSION} ${AMDGPU_FAMILY} ${RELEASE_TYPE}

# Set environment for build
ENV ROCM_PATH=/opt/rocm
ENV PATH=${ROCM_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib

# Copy source files
WORKDIR /build
COPY provider/amd/hip_remote_protocol.h .
COPY provider/amd/hip_worker_service.c .
COPY provider/amd/Makefile .

# Build the worker service
RUN make worker

# Final image - minimal runtime with ROCm
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="TensorFusion AMD HIP Worker Service"
LABEL org.opencontainers.image.description="Worker service for remote HIP API execution"
LABEL org.opencontainers.image.vendor="TensorFusion"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libatomic1 \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy TheRock installation (includes /opt/rocm symlink and /opt/rocm-VERSION)
COPY --from=builder /opt /opt

# Create directory for TensorFusion
RUN mkdir -p /usr/lib/tensor-fusion

# Copy the worker service binary
COPY --from=builder /build/hip_worker_service /usr/lib/tensor-fusion/

# Set runtime environment
ENV ROCM_PATH=/opt/rocm
ENV PATH=${ROCM_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib
# Use same port as NVIDIA TensorFusion workers (8000)
ENV TF_WORKER_PORT=8000
ENV TF_DEVICE_ID=0

# Expose the worker port
EXPOSE 8000

# Health check - just verify port is listening (our protocol is binary, not text)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD nc -z localhost ${TF_WORKER_PORT} || exit 1

# Start the worker service (runs as root - container has explicit device access)
ENTRYPOINT ["/usr/lib/tensor-fusion/hip_worker_service"]
