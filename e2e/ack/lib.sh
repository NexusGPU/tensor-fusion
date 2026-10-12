#!/usr/bin/env bash
# Shared settings of the ACK tier (Alibaba Cloud ACK + Karpenter's Alibaba Cloud
# provider). Sourced by up.sh / install.sh / down.sh / sweep.sh.
# Everything is a parameter; credentials come from the aliyun CLI profile and
# are never written into the repository.

set -euo pipefail

ACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ACK_DIR}/../.." && pwd)"

: "${ACK_NAME:=tf-e2e-ack}"                    # cluster name, name prefix and Project tag of everything this tier creates
: "${ACK_REGION:=us-east-1}"
: "${ACK_HOME:=${ACK_DIR}/.work}"               # kubeconfig, state, helm cache (git-ignored)
: "${ACK_K8S_VERSION:=1.36.2-aliyun.1}"         # the operator embeds the 1.36 kube-scheduler: the apiserver must be >= 1.36
: "${ACK_RUNTIME_VERSION:=2.3.4}"               # containerd
: "${ACK_CLUSTER_SPEC:=ack.standard}"           # ack.standard = managed Basic (free control plane), ack.pro.small = Pro
: "${ACK_VPC_CIDR:=10.43.0.0/16}"
: "${ACK_POD_CIDR:=172.20.0.0/16}"              # Flannel
: "${ACK_SERVICE_CIDR:=172.21.0.0/20}"
: "${ACK_SYSTEM_INSTANCE_TYPE:=ecs.g7.xlarge}"  # 4 vCPU / 16 GiB: ACK add-ons + Karpenter + TF controller + GreptimeDB
: "${ACK_IMAGE_TYPE:=AliyunLinux3ContainerOptimized}" # node image family; must be cgroup v2 (Kubernetes >= 1.35)
: "${ACK_KEY_PAIR:=tensor-fusion-shared}"       # existing ECS key pair of the region (node login)
: "${ACK_GPU_INSTANCE_TYPES:=ecs.gn6i-c4g1.xlarge ecs.gn6i-c8g1.2xlarge}" # 1 x T4 each
: "${ACK_CPU_INSTANCE_TYPES:=ecs.c7.large ecs.g7.large}"
: "${ACK_GPU_NODE_LIMIT:=3}"                    # NodePool limit nvidia.com/gpu (FIN-01 needs 3 whole-GPU nodes); guard.sh enforces a hard ceiling
: "${ACK_CPU_LIMIT:=8}"                         # NodePool limit cpu of the general-purpose pool
: "${ACK_GPU_CONSOLIDATE:=2m}"                  # consolidateAfter of the GPU NodePool
: "${ACK_GPU_DISK_GB:=120}"
: "${KARPENTER_ALIBABA_VERSION:=0.2.5}"         # AliyunContainerService/karpenter-provider-alibabacloud chart version
: "${KARPENTER_ALIBABA_CHART_REF:=67a27fd141a777527ac8ce0c66f1e98ab12b04d3}" # commit that carries helm-charts/karpenter-<version>.tgz (added after the tag)
: "${KARPENTER_ALIBABA_REPO:=https://github.com/AliyunContainerService/karpenter-provider-alibabacloud}"
: "${KARPENTER_ROLE:=${ACK_NAME}-karpenter}"    # RAM role assumed by the Karpenter ServiceAccount (RRSA)

ROLE_LABEL="e2e.tensor-fusion.ai/role"
SHARED_POOL_LABEL="e2e.tensor-fusion.ai/shared-pool"
GPU_NODEPOOL="tf-gpu"
CPU_NODEPOOL="tf-cpu"
GPU_NODECLASS="tf-gpu"
CPU_NODECLASS="tf-cpu"
TF_NAMESPACE="tensor-fusion-sys"
TF_RELEASE="tensor-fusion-sys"
STATE="${ACK_HOME}/state.env"

mkdir -p "${ACK_HOME}"
# A dedicated kubeconfig: the user's default kubeconfig is never read or written.
export KUBECONFIG="${ACK_HOME}/kubeconfig"
export HELM_REPOSITORY_CONFIG="${ACK_HOME}/helm/repositories.yaml"
export HELM_REPOSITORY_CACHE="${ACK_HOME}/helm/cache"

log() { printf '[tf-ack %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

need() {
  local bin
  for bin in "$@"; do
    command -v "$bin" >/dev/null 2>&1 || { log "missing required tool: $bin"; exit 1; }
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
    sleep 10
  done
  log "${what} ($(($(date +%s) - start))s)"
}

ali() { aliyun "$@" --region "$ACK_REGION"; }

# ledger <step> <key=value>...: one line per stage into results/ack-ledger.tsv
# (git-ignored), the source of the timings quoted in the documentation.
ledger() {
  local step="$1"
  shift
  mkdir -p "${ACK_DIR}/../results"
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$ACK_NAME" "$step" "$*" >>"${ACK_DIR}/../results/ack-ledger.tsv"
}

# state_get <key> / state_set <key> <value>: ids of what up.sh created.
state_get() { [ -f "$STATE" ] && sed -n "s/^$1=//p" "$STATE" | tail -1 || true; }
state_set() {
  touch "$STATE"
  grep -v "^$1=" "$STATE" >"${STATE}.tmp" || true
  echo "$1=$2" >>"${STATE}.tmp"
  mv "${STATE}.tmp" "$STATE"
}

cluster_id() {
  ali cs GET /api/v1/clusters --name "$ACK_NAME" | jq -r '.clusters[]? | select(.state != "deleted") | .cluster_id' | head -1
}

json_list() { printf '%s\n' "$@" | jq -R . | jq -sc .; } # words -> JSON array
