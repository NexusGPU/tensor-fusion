#!/usr/bin/env bash
# ACK tier helper: keep one GPU node warm, e.g. before a demo. A placeholder
# Deployment (default/tf-warm) holds a token share of one GPU in soft mode, so
# the pool brings up a T4 node through Karpenter, the hypervisor starts on it,
# and its init containers pull ${WARM_IMAGES} onto the node (through the image
# cache of registry.sh). A workload that comes later starts in seconds instead
# of the 12 to 20 minutes of a cold node.
#
#   ack/warm.sh up       apply the placeholder and wait until it is ready
#   ack/warm.sh down     delete it; the GPU node is reclaimed after ${ACK_GPU_CONSOLIDATE}
#   ack/warm.sh status
#   ack/warm.sh manifest print the Deployment only
#
# The node costs money while the placeholder exists (about $1.09/h for a T4).

source "$(dirname "$0")/lib.sh"
need kubectl jq

: "${WARM_IMAGES:=vllm/vllm-openai:v0.11.0 tensorfusion/torch2.7-cuda12.8-ssh:v4}"
: "${WARM_POOL:=tensor-fusion-shared}"
: "${WARM_TIMEOUT:=1800}"                       # a cold T4 node with a 12 GB image: 12 to 20 minutes
WARM_NS=default
WARM_NAME=tf-warm

manifest() {
  local i=0 img
  cat <<YAML
apiVersion: apps/v1
kind: Deployment
metadata: {name: ${WARM_NAME}, namespace: ${WARM_NS}, labels: {app: ${WARM_NAME}}}
spec:
  replicas: 1
  selector: {matchLabels: {app: ${WARM_NAME}}}
  template:
    metadata:
      labels: {app: ${WARM_NAME}, tensor-fusion.ai/enabled: "true"}
      annotations:
        tensor-fusion.ai/gpupool: ${WARM_POOL}
        tensor-fusion.ai/isolation: soft
        tensor-fusion.ai/is-local-gpu: "true"
        tensor-fusion.ai/tflops-request: "1"
        tensor-fusion.ai/tflops-limit: "1"
        tensor-fusion.ai/vram-request: 256Mi
        tensor-fusion.ai/vram-limit: 256Mi
        tensor-fusion.ai/inject-container: hold
    spec:
      terminationGracePeriodSeconds: 1
      initContainers:
YAML
  for img in $WARM_IMAGES; do
    printf '        - {name: image-%d, image: "%s", command: ["true"]}\n' "$i" "$img"
    i=$((i + 1))
  done
  cat <<YAML
      containers:
        - name: hold
          image: ${REGISTRY_HELPER_IMAGE:-busybox:1.36}
          command: ["sleep", "infinity"]
          resources: {requests: {cpu: 10m, memory: 16Mi}, limits: {memory: 64Mi}}
YAML
}

case "${1:-status}" in
  up)
    start=$(date +%s)
    log "placeholder ${WARM_NS}/${WARM_NAME} (pool ${WARM_POOL}, images: ${WARM_IMAGES})"
    manifest | kubectl apply -f - >/dev/null
    wait_for "$WARM_TIMEOUT" "placeholder ready" bash -c "[ \"\$(kubectl -n ${WARM_NS} get deploy ${WARM_NAME} -o jsonpath='{.status.readyReplicas}')\" = 1 ]"
    ledger warm "seconds=$(($(date +%s) - start))"
    kubectl -n "$WARM_NS" get pods -l "app=${WARM_NAME}" -o wide
    ;;
  down)
    kubectl -n "$WARM_NS" delete deploy "$WARM_NAME" --ignore-not-found
    ;;
  status)
    kubectl -n "$WARM_NS" get deploy,pods -l "app=${WARM_NAME}" -o wide
    kubectl get nodes -l "${ROLE_LABEL}=gpu" -o wide
    ;;
  manifest) manifest ;;
  *)
    log "usage: warm.sh up | down | status | manifest"
    exit 2
    ;;
esac
