#!/usr/bin/env bash
# The vGPU worker license of the test environments. Hard isolation (worker
# sidecar) and remote GPU mode (worker Pod) need it: without one the worker
# quits with "License is not set".
#
# The license never lives in this repository. It is read from, in this order:
#   1. TF_LICENSE and TF_LICENSE_SIGN in the environment (CI: repository secrets);
#   2. the file TF_LICENSE_FILE, default ~/.config/tensor-fusion/license.env,
#      mode 600, with the two lines TF_LICENSE='...' and TF_LICENSE_SIGN='...'.
# The values reach the cluster only as a Secret, created through stdin; they are
# never on a command line and never printed.
#
# A license other than the "ee" edition lists GPU UUIDs; the worker refuses to
# start on a node that has any other GPU ("Device UUID ... is not found in the
# license"). Use one file per environment when the environments need different
# licenses.
#
# Sourced by the install scripts (tf_license_load, tf_license_secret), or run:
#
#   tools/tf-license.sh apply [namespace...]
#       For a cluster that is already installed (a bare-metal lab, for example):
#       creates the Secret in the given namespaces (default: "default" and every
#       namespace that has a TensorFusionWorkload) and points the worker of every
#       pool of every TensorFusionCluster at it. Replaces a license written into
#       the cluster object as plain values.
#   tools/tf-license.sh status
#       Says where a license would be read from, its edition and expiry, and
#       which namespaces of the current cluster have the Secret.
#
#   KUBECTL   the kubectl command, default "kubectl". For a cluster reached over
#             SSH: KUBECTL="ssh lab sudo -n kubectl".

: "${TF_LICENSE_FILE:=${HOME}/.config/tensor-fusion/license.env}"
: "${TF_LICENSE_SECRET:=tensor-fusion-license}"
: "${KUBECTL:=kubectl}"

# tf_license_load: 0 when both values are available (exported), 1 when not.
tf_license_load() {
  if [ -z "${TF_LICENSE:-}" ] && [ -f "$TF_LICENSE_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$TF_LICENSE_FILE"
    set +a
  fi
  [ -n "${TF_LICENSE:-}" ] && [ -n "${TF_LICENSE_SIGN:-}" ]
}

# tf_license_secret <namespace>...: creates or updates the Secret there.
tf_license_secret() {
  local ns
  for ns in "$@"; do
    jq -n --arg name "$TF_LICENSE_SECRET" --arg ns "$ns" --arg l "$TF_LICENSE" --arg s "$TF_LICENSE_SIGN" \
      '{apiVersion: "v1", kind: "Secret", metadata: {name: $name, namespace: $ns}, stringData: {TF_LICENSE: $l, TF_LICENSE_SIGN: $s}}' |
      $KUBECTL apply -f - >/dev/null
  done
}

# tf_license_summary: edition and expiry, nothing that identifies the license.
tf_license_summary() {
  local edition expiry devices
  devices=${TF_LICENSE%%|*}
  edition=$(printf '%s' "$TF_LICENSE" | cut -d'|' -f2)
  expiry=$(printf '%s' "$TF_LICENSE" | cut -d'|' -f3)
  expiry=$(date -u -r "$((expiry / 1000))" +%Y-%m-%d 2>/dev/null || date -u -d "@$((expiry / 1000))" +%Y-%m-%d)
  if [ -n "$devices" ]; then
    printf 'edition=%s expires=%s bound to %s GPU(s)' "$edition" "$expiry" "$(printf '%s' "$devices" | tr ',' '\n' | grep -c .)"
  else
    printf 'edition=%s expires=%s not bound to GPUs' "$edition" "$expiry"
  fi
}

# Executed, not sourced.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
  if ! tf_license_load; then
    echo "no license: set TF_LICENSE and TF_LICENSE_SIGN, or write them to ${TF_LICENSE_FILE}" >&2
    exit 1
  fi
  with_secret() {
    $KUBECTL get secret -A --field-selector "metadata.name=${TF_LICENSE_SECRET}" -o json | jq -r '.items[].metadata.namespace'
  }
  case "${1:-}" in
    status)
      echo "license: $(tf_license_summary)"
      echo "namespaces with Secret ${TF_LICENSE_SECRET}: $(with_secret | tr '\n' ' ')"
      ;;
    apply)
      shift
      if [ $# -eq 0 ]; then
        # shellcheck disable=SC2046
        set -- $({ echo default; $KUBECTL get tensorfusionworkload -A -o json | jq -r '.items[].metadata.namespace'; } | sort -u)
      fi
      tf_license_secret "$@"
      env_ref='[{name: "TF_LICENSE", valueFrom: {secretKeyRef: {name: $secret, key: "TF_LICENSE", optional: true}}},
                {name: "TF_LICENSE_SIGN", valueFrom: {secretKeyRef: {name: $secret, key: "TF_LICENSE_SIGN", optional: true}}}]'
      for cluster in $($KUBECTL get tensorfusioncluster -o json | jq -r '.items[].metadata.name'); do
        $KUBECTL get tensorfusioncluster "$cluster" -o json |
          jq --arg secret "$TF_LICENSE_SECRET" "
            .spec.gpuPools[].specTemplate.componentConfig.worker.podTemplate.template.spec.containers |=
              ((. // [{name: \"tensorfusion-worker\"}]) | map(
                if .name == \"tensorfusion-worker\"
                then .env = ((.env // []) | map(select(.name != \"TF_LICENSE\" and .name != \"TF_LICENSE_SIGN\"))) + ${env_ref}
                else . end))
            | {spec: {gpuPools: .spec.gpuPools}}" |
          $KUBECTL patch tensorfusioncluster "$cluster" --type merge --patch-file /dev/stdin >/dev/null
        echo "TensorFusionCluster ${cluster}: workers read the license from Secret ${TF_LICENSE_SECRET}"
      done
      echo "license ($(tf_license_summary)) in namespaces: $*"
      ;;
    *)
      echo "usage: tf-license.sh apply [namespace...] | status" >&2
      exit 2
      ;;
  esac
fi
