#!/usr/bin/env bash
# ACK tier helper: build Karpenter's Alibaba Cloud provider with the patches of
# ack/patches/ and load it onto the system node. The published image
# (${KARPENTER_ALIBABA_VERSION}) launches one more instance every provisioning
# round for a pod that requests a NodeOverlay resource; see the patch.
#
#   ack/build-karpenter.sh     then: KARPENTER_ALIBABA_IMAGE=<printed image> ack/install.sh

source "$(dirname "$0")/lib.sh"
need docker git go

: "${KARPENTER_ALIBABA_BASE_IMAGE:=registry-cn-hangzhou.ack.aliyuncs.com/acs/karpenter-provider-alibabacloud:v0.2.5-4a6a244}"
image="tf-e2e/karpenter-provider-alibabacloud:v${KARPENTER_ALIBABA_VERSION}-tf1"

# Outside the repository: `make manifests` walks every Go file below it.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
log "cloning ${KARPENTER_ALIBABA_REPO} v${KARPENTER_ALIBABA_VERSION}"
git clone -q --depth 1 --branch "v${KARPENTER_ALIBABA_VERSION}" "$KARPENTER_ALIBABA_REPO" "$work/src" 2>/dev/null
for p in "${ACK_DIR}"/patches/karpenter-alibaba-"${KARPENTER_ALIBABA_VERSION}"-*.patch; do
  log "applying $(basename "$p")"
  git -C "$work/src" apply "$p"
done
log "building the controller (linux/amd64)"
(cd "$work/src" && GOFLAGS=-mod=mod CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o "$work/ctx/controller" ./cmd/controller)
printf 'FROM %s\nCOPY controller /controller\n' "$KARPENTER_ALIBABA_BASE_IMAGE" >"$work/ctx/Dockerfile"
docker build -q --platform linux/amd64 -t "$image" "$work/ctx" >/dev/null
bash "${ACK_DIR}/load-image.sh" "$image"
echo "$image"
