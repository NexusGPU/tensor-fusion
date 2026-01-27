# Copy pre-built binary from GitHub Actions
FROM ubuntu:24.04
ARG TARGETARCH

WORKDIR /
COPY bin/hypervisor-linux-${TARGETARCH} /hypervisor

USER 65532:65532

ENTRYPOINT ["/hypervisor"]
