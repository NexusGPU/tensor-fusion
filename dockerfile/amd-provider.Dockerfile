# AMD Provider Dockerfile
# Builds the AMD accelerator provider library with TheRock ROCm distribution
# Also includes the HIP client stub for remote GPU (GPU-over-IP) mode

FROM ubuntu:24.04 AS builder

WORKDIR /workspace

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    gfortran \
    git \
    ninja-build \
    g++ \
    pkg-config \
    xxd \
    patchelf \
    automake \
    libtool \
    python3-venv \
    python3-dev \
    libegl1-mesa-dev \
    texinfo \
    bison \
    flex \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Copy TheRock installation script
COPY scripts/install_rocm_tarball.sh /tmp/

# Install TheRock
ARG ROCM_VERSION=7.11.0rc0
ARG AMDGPU_FAMILY=gfx94X-dcgpu
ARG RELEASE_TYPE=prereleases

RUN bash /tmp/install_rocm_tarball.sh ${ROCM_VERSION} ${AMDGPU_FAMILY} ${RELEASE_TYPE}

# Set environment for build (script creates /opt/rocm symlink)
ENV ROCM_PATH=/opt/rocm
ENV PATH=${ROCM_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib

# Copy provider source and tests
COPY provider/ provider/

# Build AMD provider library
WORKDIR /workspace/provider
RUN make amd

# Build HIP client stub for remote GPU mode (pure C, no ROCm dependencies)
RUN make -C amd client

# Build test binary
RUN gcc -Wall -Wextra -std=c11 \
    -I. -I${ROCM_PATH}/include \
    -o build/test_amd_provider test/test_amd_provider.c \
    -L${ROCM_PATH}/lib -L./build \
    -laccelerator_amd -lamd_smi -lamdhip64 \
    -Wl,-rpath,${ROCM_PATH}/lib -Wl,-rpath,/build/lib

# Create deployment image
FROM ubuntu:24.04

# Install runtime dependencies (python3 for amd-smi)
RUN apt-get update && apt-get install -y \
    python3 \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy TheRock installation (includes /opt/rocm symlink and /opt/rocm-VERSION)
COPY --from=builder /opt /opt

# Copy built provider library and test binary
COPY --from=builder /workspace/provider/build/libaccelerator_amd.so /build/lib/
COPY --from=builder /workspace/provider/build/test_amd_provider /build/bin/

# Copy HIP client stub for remote GPU mode
# Makefile outputs to current directory
COPY --from=builder /workspace/provider/amd/libhip_client_stub.so /build/lib/

# Copy init container entrypoint script
COPY scripts/inject-libs.sh /build/bin/
RUN chmod +x /build/bin/inject-libs.sh

# Create metadata file
ARG ROCM_VERSION=7.11.0rc0
RUN echo "version: 1.0.0\n\
hardwareVendor: AMD\n\
releaseDate: \"$(date -I)\"\n\
isolationModes:\n\
  - shared\n\
  - remote\n\
rocmDistribution: TheRock\n\
rocmVersion: ${ROCM_VERSION}" > /build/metadata.yaml

# Set runtime environment (uses standard /opt/rocm path)
ENV ROCM_PATH=/opt/rocm
ENV PATH=${ROCM_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=/build/lib:${ROCM_PATH}/lib
ENV HARDWARE_VENDOR=AMD

# Init container entrypoint - copies libraries to shared volumes
# For hypervisor/standalone mode, can override to use default sleep
ENTRYPOINT ["/build/bin/inject-libs.sh"]
