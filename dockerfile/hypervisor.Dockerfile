# Stage 1: pull vendor accelerator libraries from the vgpu-provider image.
# init-runtime container will copy /build/* to /run/tensor-fusion at pod start,
# which is how libaccelerator_nvidia.so reaches the hypervisor's lib path.
ARG VGPU_PROVIDER_IMAGE=tensorfusion/vgpu-provider-nvidia:latest
FROM ${VGPU_PROVIDER_IMAGE} AS provider

# Stage 2: final hypervisor image
FROM ubuntu:24.04
ARG TARGETARCH

WORKDIR /
COPY bin/hypervisor-linux-${TARGETARCH} /usr/local/bin/hypervisor

# Vendor .so files used by init-runtime to populate /run/tensor-fusion at pod start.
COPY --from=provider /build /build

USER 65532:65532

ENTRYPOINT ["hypervisor"]
