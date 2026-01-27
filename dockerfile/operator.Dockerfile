# Copy pre-built binary from GitHub Actions
FROM ubuntu:24.04
ARG TARGETARCH

WORKDIR /
COPY bin/manager-linux-${TARGETARCH} /manager

USER 65532:65532

ENTRYPOINT ["/manager"]
