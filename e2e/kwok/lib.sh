#!/usr/bin/env bash
# Shared settings of the KWOK tier. Sourced by up.sh / down.sh / test.sh.
# Everything is a parameter; nothing here names a host, a mirror or a path
# outside the repository.

set -euo pipefail

KWOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${KWOK_DIR}/../.." && pwd)"

: "${KWOK_CLUSTER:=tfkwok}"                 # kind cluster name; containers and the docker network carry it as prefix
: "${KWOK_HOME:=${KWOK_DIR}/.work}"          # binaries, sources, kubeconfig, logs (git-ignored)
: "${KWOK_NODE_IMAGE:=kindest/node:v1.36.4}" # Kubernetes of the EKS stack (1.36); must be a tag of the kind release in use
: "${KWOK_NODE_MEMORY:=6g}"                  # memory limit of the kind node container
: "${KWOK_VERSION:=v0.8.0}"                  # kubernetes-sigs/kwok release (fake kubelet)
: "${KARPENTER_VERSION:=v1.14.1}"            # kubernetes-sigs/karpenter tag; its kwok/ provider is built from source
: "${KARPENTER_REPO:=https://github.com/kubernetes-sigs/karpenter}"
: "${KARPENTER_SRC:=${KWOK_HOME}/src/karpenter}"
: "${KWOK_MANIFEST_DIR:=${KWOK_HOME}/manifests}"
: "${KWOK_RELEASE_URL:=https://github.com/kubernetes-sigs/kwok/releases/download}"
: "${KWOK_IMAGE_MIRROR:=}"                   # registry that mirrors Docker Hub, e.g. mirror.example.com ("" = Docker Hub)
: "${KWOK_K8S_IMAGE_MIRROR:=registry.k8s.io}" # registry (and path prefix) that mirrors registry.k8s.io
: "${KWOK_PULL_TOOL:=docker}"                # docker | crane (crane: pull without the docker daemon, then docker load)
: "${TF_RELEASE_TAG:=2.17.24}"               # operator + hypervisor release under test
: "${TF_OPERATOR_IMAGE:=tensorfusion/tensor-fusion-operator}"
: "${TF_HYPERVISOR_IMAGE:=tensorfusion/tensor-fusion-hypervisor:${TF_RELEASE_TAG}}"
: "${TF_CHART:=${REPO_ROOT}/charts/tensor-fusion}"
: "${KWOK_GPU_CONSOLIDATE:=1m}"              # consolidateAfter of the GPU NodePool (EKS stack: 10m, forced by G-16)
: "${KWOK_REGISTRATION_DELAY:=10s}"          # KWOKNodeClass.nodeRegistrationDelay: instance boot
: "${KWOK_FAKE_HV_DELAY:=20s}"               # fake hypervisor: driver + hypervisor start-up before GPUs register
: "${KWOK_FAKE_GPUS_PER_NODE:=1}"
: "${GO:=go}"

ROLE_LABEL="e2e.tensor-fusion.ai/role"
SHARED_POOL_LABEL="e2e.tensor-fusion.ai/shared-pool"
GPU_NODEPOOL="tf-gpu"
CPU_NODEPOOL="tf-cpu"
NODECLASS="tf-kwok"
TF_NAMESPACE="tensor-fusion-sys"
TF_RELEASE="tensor-fusion-sys"

mkdir -p "${KWOK_HOME}/bin" "${KWOK_HOME}/logs" "${KWOK_HOME}/images"
export PATH="${KWOK_HOME}/bin:${PATH}"
# A dedicated kubeconfig: the user's default kubeconfig is never read or written.
export KUBECONFIG="${KWOK_HOME}/kubeconfig"
export KIND_EXPERIMENTAL_DOCKER_NETWORK="${KWOK_CLUSTER}"
export HELM_REPOSITORY_CONFIG="${KWOK_HOME}/helm/repositories.yaml"
export HELM_REPOSITORY_CACHE="${KWOK_HOME}/helm/cache"

log() { printf '[tf-kwok %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

need() {
  local bin
  for bin in "$@"; do
    command -v "$bin" >/dev/null 2>&1 || { log "missing required tool: $bin (put it on PATH or into ${KWOK_HOME}/bin)"; exit 1; }
  done
}

wait_for() { # wait_for <timeout s> <what> <command...>
  local timeout="$1" what="$2" start
  shift 2
  start=$(date +%s)
  until "$@" >/dev/null 2>&1; do
    if [ $(($(date +%s) - start)) -ge "$timeout" ]; then
      log "TIMEOUT after ${timeout}s waiting for ${what}"
      return 1
    fi
    sleep 3
  done
  log "${what} ($(($(date +%s) - start))s)"
}

# mirror_ref <canonical image>: where to pull it from.
mirror_ref() {
  local ref="${1#docker.io/}"
  case "$ref" in
    registry.k8s.io/*) echo "${KWOK_K8S_IMAGE_MIRROR}/${ref#registry.k8s.io/}" ;;
    *.*/*) echo "$ref" ;; # some other registry: as is
    *) if [ -n "$KWOK_IMAGE_MIRROR" ]; then echo "${KWOK_IMAGE_MIRROR}/${ref}"; else echo "$ref"; fi ;;
  esac
}

# ensure_image <canonical image>: the docker daemon has it under its canonical name.
ensure_image() {
  local ref="$1" src tar
  docker image inspect "$ref" >/dev/null 2>&1 && return 0
  src="$(mirror_ref "$ref")"
  log "pulling ${ref} (from ${src%%/*})"
  if [ "$KWOK_PULL_TOOL" = "crane" ]; then
    need crane
    tar="${KWOK_HOME}/images/$(echo "$ref" | tr '/:' '__').tar"
    crane pull --platform linux/amd64 "$src" "$tar"
    docker load -q -i "$tar" >/dev/null
    rm -f "$tar"
  else
    docker pull -q "$src" >/dev/null
  fi
  [ "$src" = "$ref" ] || docker tag "$src" "$ref"
}

cluster_exists() { kind get clusters 2>/dev/null | grep -qx "$KWOK_CLUSTER"; }
