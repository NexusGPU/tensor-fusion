#!/usr/bin/env bash
# ACK tier, one command for the whole stack.
#
#   ack/run.sh up      up.sh + registry.sh + install.sh + install-tf.sh + the cost guard in the background
#   ack/run.sh down    stop the guard + down.sh
#
# Optional local inputs under ${ACK_HOME} (git-ignored), picked up when present:
#   images/*.tar    `docker save` archives loaded onto the system node before the
#                   installs (development builds no registry carries). images/karpenter.tar
#                   (the patched provider, KARPENTER_ALIBABA_IMAGE) is built by
#                   build-karpenter.sh when it is missing.
#   stack.env       plain KEY=VALUE lines exported for every stage, e.g.
#                   TF_RELEASE_TAG=..., TF_OPERATOR_TAG=..., TF_AGENT_TAG=...
#   agent.env       TF_AGENT_ID / TF_ENROLL_TOKEN of the cloud console environment
#                   (mode 600; never printed)

source "$(dirname "$0")/lib.sh"

load_env() {
  local f
  for f in "${ACK_HOME}/stack.env" "${ACK_HOME}/agent.env"; do
    [ -f "$f" ] || continue
    set -a
    # shellcheck disable=SC1090
    . "$f"
    set +a
  done
}

case "${1:-}" in
  up)
    start=$(date +%s)
    load_env
    # The published provider image over-provisions GPU nodes (README pitfall 6):
    # build the patched one before anything costs money.
    if [ "${KARPENTER_ALIBABA_IMAGE#tf-e2e/}" != "${KARPENTER_ALIBABA_IMAGE}" ] && [ ! -s "${ACK_HOME}/images/karpenter.tar" ]; then
      bash "${ACK_DIR}/build-karpenter.sh" --no-load >/dev/null
    fi
    bash "${ACK_DIR}/up.sh"
    for tar in "${ACK_HOME}"/images/*.tar; do
      [ -f "$tar" ] && bash "${ACK_DIR}/load-image.sh" "$tar"
    done
    # The image cache first: everything installed after it is pulled through it.
    bash "${ACK_DIR}/registry.sh"
    bash "${ACK_DIR}/install.sh"
    bash "${ACK_DIR}/install-tf.sh"
    nohup bash "${ACK_DIR}/guard.sh" >>"${ACK_HOME}/guard.log" 2>&1 &
    echo $! >"${ACK_HOME}/guard.pid"
    ledger up-total "seconds=$(($(date +%s) - start))"
    log "stack up in $(($(date +%s) - start))s, guard running (pid $(cat "${ACK_HOME}/guard.pid")). export KUBECONFIG=${KUBECONFIG}"
    ;;
  down)
    if [ -f "${ACK_HOME}/guard.pid" ]; then
      kill "$(cat "${ACK_HOME}/guard.pid")" 2>/dev/null || true
      rm -f "${ACK_HOME}/guard.pid"
    fi
    bash "${ACK_DIR}/down.sh"
    ;;
  *)
    log "usage: run.sh up | down"
    exit 2
    ;;
esac
