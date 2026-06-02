# Copy pre-built binary from GitHub Actions
FROM ubuntu:26.04
ARG TARGETARCH

WORKDIR /
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY bin/manager-linux-${TARGETARCH} /manager

USER 65532:65532

ENTRYPOINT ["/manager"]
