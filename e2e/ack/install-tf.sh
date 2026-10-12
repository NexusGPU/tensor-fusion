#!/usr/bin/env bash
# ACK tier, stage 3: TensorFusion from this checkout's chart, configured like the
# EKS stack (terraform/scripts/install-tf.sh). Image tags are always explicit.
#
#   TF_RELEASE_TAG        operator + hypervisor release under test (default 2.17.28)
#   TF_OPERATOR_TAG       operator tag alone, for a development build (default: TF_RELEASE_TAG).
#                         A tag no registry has must already be in containerd on the system node.
#   TF_VGPU_PROVIDER_TAG  tensorfusion/vgpu-provider-nvidia tag     (default 1.3.12)
#   TF_AGENT_ID / TF_ENROLL_TOKEN / TF_CLOUD_ENDPOINT
#                         cloud console agent. Read from the environment only and
#                         stored in a Secret created through stdin; never on a
#                         command line, never echoed, never written into the repo.
#   TF_HELM_ARGS          extra arguments appended to `helm upgrade --install`.

source "$(dirname "$0")/lib.sh"
need kubectl helm jq

: "${TF_RELEASE_TAG:=2.17.28}"
: "${TF_OPERATOR_IMAGE:=tensorfusion/tensor-fusion-operator}"
: "${TF_OPERATOR_TAG:=${TF_RELEASE_TAG}}"
: "${TF_HYPERVISOR_IMAGE:=tensorfusion/tensor-fusion-hypervisor:${TF_RELEASE_TAG}}"
: "${TF_VGPU_PROVIDER_TAG:=1.3.12}"
: "${TF_CHART:=${REPO_ROOT}/charts/tensor-fusion}"
AGENT_SECRET=tensor-fusion-cloud-agent

# GreptimeDB's claim has no storageClassName and the cluster has no CSI add-on:
# a hostPath PersistentVolume on the system node, pre-bound to the claim.
if [ "$(kubectl get pv tf-e2e-greptimedb -o jsonpath='{.status.phase}' 2>/dev/null)" = "Released" ]; then
  kubectl delete pv tf-e2e-greptimedb --timeout=60s
fi
kubectl apply -f - >/dev/null <<YAML
apiVersion: v1
kind: PersistentVolume
metadata:
  name: tf-e2e-greptimedb
spec:
  capacity:
    storage: 20Gi
  accessModes: ["ReadWriteOnce"]
  persistentVolumeReclaimPolicy: Retain
  volumeMode: Filesystem
  claimRef:
    namespace: greptimedb
    name: datanode-${TF_RELEASE}-greptimedb-standalone-0
  hostPath:
    path: /var/lib/tf-e2e/greptimedb
    type: DirectoryOrCreate
  nodeAffinity:
    required:
      nodeSelectorTerms:
        - matchExpressions:
            - key: ${ROLE_LABEL}
              operator: In
              values: ["system"]
YAML

agent_args=()
if [ -n "${TF_AGENT_ID:-}" ]; then
  : "${TF_ENROLL_TOKEN:?TF_ENROLL_TOKEN must be set together with TF_AGENT_ID}"
  kubectl create namespace "$TF_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  jq -n --arg id "$TF_AGENT_ID" --arg tok "$TF_ENROLL_TOKEN" --arg name "$AGENT_SECRET" --arg ns "$TF_NAMESPACE" \
    '{apiVersion: "v1", kind: "Secret", metadata: {name: $name, namespace: $ns}, stringData: {"agent-id": $id, "enroll-token": $tok}}' |
    kubectl apply -f - >/dev/null
  log "cloud console agent enabled (agent id and token taken from the environment)"
fi
if kubectl -n "$TF_NAMESPACE" get secret "$AGENT_SECRET" >/dev/null 2>&1; then
  agent_args=(--set-string "agent.existingSecret.name=${AGENT_SECRET}" --set-string "agent.existingSecret.agentIdKey=agent-id")
  [ -n "${TF_CLOUD_ENDPOINT:-}" ] && agent_args+=(--set-string "agent.cloudEndpoint=${TF_CLOUD_ENDPOINT}")
fi

log "installing TensorFusion: chart=${TF_CHART} operator=${TF_OPERATOR_IMAGE}:${TF_OPERATOR_TAG} hypervisor=${TF_HYPERVISOR_IMAGE} middleware=vgpu-provider-nvidia:${TF_VGPU_PROVIDER_TAG}"
start=$(date +%s)
# karpenterNodeOverlay.requirements: the chart default selects on labels of
# Karpenter's AWS provider; the Alibaba Cloud provider's own GPU label is used.
# shellcheck disable=SC2086  # TF_HELM_ARGS is intentionally word-split
helm upgrade --install "$TF_RELEASE" "$TF_CHART" --namespace "$TF_NAMESPACE" --create-namespace \
  --wait --timeout 10m ${agent_args[@]+"${agent_args[@]}"} ${TF_HELM_ARGS:-} -f - >/dev/null <<YAML
cluster:
  hypervisorImage: ${TF_HYPERVISOR_IMAGE}
  pool:
    nodeSelectorKey: ${SHARED_POOL_LABEL}
controller:
  image:
    repository: ${TF_OPERATOR_IMAGE}
    tag: "${TF_OPERATOR_TAG}"
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: ${ROLE_LABEL}
                operator: In
                values: ["system"]
providerConfigs:
  nvidia:
    images:
      middleware: tensorfusion/vgpu-provider-nvidia:${TF_VGPU_PROVIDER_TAG}
karpenterNodeOverlay:
  requirements:
    - key: karpenter.alibabacloud.com/instance-gpu-manufacturer
      operator: In
      values: ["nvidia"]
YAML
log "helm release ready ($(($(date +%s) - start))s)"
ledger install-tf "seconds=$(($(date +%s) - start)) operator=${TF_OPERATOR_TAG} hypervisor=${TF_HYPERVISOR_IMAGE##*:}"
kubectl -n "$TF_NAMESPACE" get deploy,sts,pods -o wide || true
kubectl get tensorfusioncluster,gpupool,nodeoverlay 2>/dev/null || true
