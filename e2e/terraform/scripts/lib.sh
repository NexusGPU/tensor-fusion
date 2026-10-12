#!/usr/bin/env bash
# Shared helpers for the tf-e2e scripts. Sourced, not executed.
# All inputs arrive as TF_E2E_* environment variables (set by Terraform's
# local-exec, or by `make` from `terraform output install_env`).

set -euo pipefail

: "${TF_E2E_CLUSTER:?TF_E2E_CLUSTER is required}"
: "${TF_E2E_REGION:?TF_E2E_REGION is required}"

if [ -n "${TF_E2E_AWS_PROFILE:-}" ]; then
  export AWS_PROFILE="$TF_E2E_AWS_PROFILE"
fi
export AWS_REGION="$TF_E2E_REGION"
export AWS_DEFAULT_REGION="$TF_E2E_REGION"
export AWS_PAGER=""

# Private Helm repo config/cache: the scripts must not depend on (or modify) the
# user's global `helm repo` state, which is often stale.
TF_E2E_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HELM_REPOSITORY_CONFIG="${TF_E2E_HOME}/.helm/repositories.yaml"
export HELM_REPOSITORY_CACHE="${TF_E2E_HOME}/.helm/cache"
mkdir -p "$HELM_REPOSITORY_CACHE"

log() { printf '[tf-e2e %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

need() {
  for bin in "$@"; do
    command -v "$bin" >/dev/null 2>&1 || { log "missing required tool: $bin"; exit 1; }
  done
}

# wait_for <timeout seconds> <what> <command...>: retries the command every 15 s
# (the laptop running this is busy; no tight polling loops).
wait_for() {
  local timeout="$1" what="$2" start
  shift 2
  start=$(date +%s)
  until "$@" >/dev/null 2>&1; do
    if [ $(($(date +%s) - start)) -ge "$timeout" ]; then
      log "TIMEOUT after ${timeout}s waiting for ${what}"
      return 1
    fi
    sleep 15
  done
  log "${what} ($(($(date +%s) - start))s)"
}

# Writes a dedicated kubeconfig (never touches ~/.kube/config) and exports KUBECONFIG.
setup_kubeconfig() {
  : "${TF_E2E_KUBECONFIG:?TF_E2E_KUBECONFIG is required}"
  mkdir -p "$(dirname "$TF_E2E_KUBECONFIG")"
  aws eks update-kubeconfig --name "$TF_E2E_CLUSTER" --region "$TF_E2E_REGION" \
    --kubeconfig "$TF_E2E_KUBECONFIG" --alias "$TF_E2E_CLUSTER" >/dev/null
  chmod 600 "$TF_E2E_KUBECONFIG"
  export KUBECONFIG="$TF_E2E_KUBECONFIG"
}

# no_index_overlay <nodepool>...: (re)creates NodeOverlay tf-e2e-no-index, which
# zeroes the tensor-fusion.ai/index* capacity on the named NodePools.
#
# Why: the TensorFusion chart ships NodeOverlay/tensor-fusion-overlay with
# `requirements: []`, i.e. EVERY instance type of EVERY NodePool is advertised
# as providing the index_* resources TF worker pods request. Karpenter then
# answers a pending TF pod by launching a node from the highest-weight NodePool
# (a CPU node here), which never registers the resource, so the NodeClaim stays
# Initialized=Unknown forever and is never consolidated. Seen live on
# Karpenter 1.14.1 + chart 1.8.8 (case KRP-A15 reproduces it on purpose). A
# higher-weight overlay wins per resource, so this confines TF pods to the
# NodePools that really host TensorFusion nodes.
no_index_overlay() {
  local pools
  pools=$(printf '"%s", ' "$@")
  {
    cat <<YAML
apiVersion: karpenter.sh/v1alpha1
kind: NodeOverlay
metadata:
  name: tf-e2e-no-index
spec:
  weight: 100
  requirements:
    - key: karpenter.sh/nodepool
      operator: In
      values: [${pools%, }]
  capacity:
    tensor-fusion.ai/index: 0
YAML
    for i in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do echo "    tensor-fusion.ai/index_${i}: 0"; done
  } | kubectl apply -f - >/dev/null
}
