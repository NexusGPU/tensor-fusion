#!/usr/bin/env bash
# KWOK tier, stage "down": stop the fake hypervisor and delete the kind cluster.
# Built binaries, sources and pulled images stay (KWOK_PURGE=1 removes the
# work directory and the images this tier built).

source "$(dirname "$0")/lib.sh"
need docker kind

if [ -f "${KWOK_HOME}/fakehv.pid" ]; then
  kill "$(cat "${KWOK_HOME}/fakehv.pid")" 2>/dev/null || true
  rm -f "${KWOK_HOME}/fakehv.pid"
fi
if cluster_exists; then
  log "deleting kind cluster ${KWOK_CLUSTER}"
  kind delete cluster --name "$KWOK_CLUSTER" --kubeconfig "$KUBECONFIG" >/dev/null
fi
docker network rm "$KWOK_CLUSTER" >/dev/null 2>&1 || true
rm -f "$KUBECONFIG"
if [ "${KWOK_PURGE:-}" = "1" ]; then
  docker image rm "${KWOK_CLUSTER}/karpenter-kwok:${KARPENTER_VERSION}-shortnames" >/dev/null 2>&1 || true
  rm -rf "${KWOK_HOME}"
fi
left=$(docker ps -a --filter "name=^${KWOK_CLUSTER}" --format '{{.Names}}' | tr '\n' ' ')
log "down. containers left with prefix ${KWOK_CLUSTER}: ${left:-none}"
