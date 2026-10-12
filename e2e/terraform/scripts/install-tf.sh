#!/usr/bin/env bash
# Stage 3: TensorFusion. Separate from the infrastructure on purpose: chart
# source and image tags are what changes between runs.
#
# Parameters (environment, all optional):
#   TF_CHART             chart reference. Default: this checkout's charts/tensor-fusion
#                        (the branch under test). `tensor-fusion/tensor-fusion` = the
#                        released chart from the public Helm repo.
#   TF_CHART_VERSION     chart version, only for a repo chart.
#   TF_RELEASE_TAG       operator + hypervisor version (one release publishes both images
#                        under the same tag). Default 2.17.23, the latest release when this
#                        was written. Image tags are always passed explicitly to helm.
#   TF_OPERATOR_TAG      tensorfusion/tensor-fusion-operator tag   (default: TF_RELEASE_TAG)
#   TF_OPERATOR_IMAGE    operator image repository override
#   TF_HYPERVISOR_TAG    tensorfusion/tensor-fusion-hypervisor tag (default: TF_RELEASE_TAG)
#   TF_HYPERVISOR_IMAGE  full hypervisor image reference override
#   TF_VGPU_PROVIDER_TAG tensorfusion/vgpu-provider-nvidia tag     (default: 1.3.9)
#   TF_CLIENT_TAG / TF_WORKER_TAG   remote client / worker image tags (default: chart values)
#   TF_AGENT_ID / TF_ENROLL_TOKEN / TF_CLOUD_ENDPOINT
#                        cloud console agent. Read from the environment only, handed to
#                        helm through a 0600 temp file (never on a command line, never
#                        echoed, never written into the repo).
#   TF_HELM_ARGS         extra arguments appended to `helm upgrade --install`.
#
#   scripts/install-tf.sh            install / upgrade
#   scripts/install-tf.sh uninstall  remove the release, CRs and CRDs (scripts/uninstall.sh of the repo)

source "$(dirname "$0")/lib.sh"
need aws kubectl helm

setup_kubeconfig

RELEASE=tensor-fusion-sys
NAMESPACE=tensor-fusion-sys
REPO_ROOT="$(cd "${TF_E2E_HOME}/../.." && pwd)"
CHART="${TF_CHART:-${REPO_ROOT}/charts/tensor-fusion}"

if [ "${1:-}" = "uninstall" ]; then
  log "uninstalling TensorFusion with the repo's scripts/uninstall.sh"
  NAMESPACE="$NAMESPACE" HELM_RELEASE="$RELEASE" "${REPO_ROOT}/scripts/uninstall.sh"
  kubectl delete pv tf-e2e-greptimedb --ignore-not-found --timeout=60s
  exit 0
fi

# --- GreptimeDB volume ----------------------------------------------------------
# The chart's GreptimeDB StatefulSet has a 20Gi volumeClaimTemplate without a
# storageClassName. Instead of the EBS CSI driver (addon + IAM role + a volume
# that can outlive the cluster) it gets a hostPath PersistentVolume on the
# system node, pre-bound to the claim the StatefulSet will create.
default_sc=$(kubectl get storageclass -o jsonpath='{range .items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")]}{.metadata.name}{" "}{end}')
if [ -n "$default_sc" ]; then
  log "WARNING: default StorageClass present (${default_sc}); the GreptimeDB claim will use it instead of the hostPath volume"
fi
if [ "$(kubectl get pv tf-e2e-greptimedb -o jsonpath='{.status.phase}' 2>/dev/null)" = "Released" ]; then
  kubectl delete pv tf-e2e-greptimedb --timeout=60s
fi
kubectl apply -f - <<YAML
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
    name: datanode-${RELEASE}-greptimedb-standalone-0
  hostPath:
    path: /var/lib/tf-e2e/greptimedb
    type: DirectoryOrCreate
  nodeAffinity:
    required:
      nodeSelectorTerms:
        - matchExpressions:
            - key: ${TF_E2E_ROLE_LABEL_KEY}
              operator: In
              values: ["system"]
YAML

# --- values ----------------------------------------------------------------------
values="$(mktemp)"
secrets="$(mktemp)"
chmod 600 "$values" "$secrets"
trap 'rm -f "$values" "$secrets"' EXIT

# Image tags are ALWAYS passed explicitly: the chart's own defaults (appVersion,
# cluster.hypervisorImage) lag behind or are empty, and a run must record
# exactly which images it tested.
release_tag="${TF_RELEASE_TAG:-2.17.23}"
operator_tag="${TF_OPERATOR_TAG:-${release_tag}}"
operator_repo="${TF_OPERATOR_IMAGE:-tensorfusion/tensor-fusion-operator}"
hypervisor_image="${TF_HYPERVISOR_IMAGE:-tensorfusion/tensor-fusion-hypervisor:${TF_HYPERVISOR_TAG:-${release_tag}}}"
middleware_image="tensorfusion/vgpu-provider-nvidia:${TF_VGPU_PROVIDER_TAG:-1.3.9}"

cat >"$values" <<YAML
# The default (AutoSelect) pool only takes nodes carrying this label: the
# Karpenter GPU NodePool. Mock-GPU and TF-provisioned nodes stay out of it.
cluster:
  pool:
    nodeSelectorKey: ${TF_E2E_SHARED_POOL_LABEL_KEY}
controller:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: ${TF_E2E_ROLE_LABEL_KEY}
                operator: In
                values: ["system"]
YAML

set_args=(
  --set-string "controller.image.repository=${operator_repo}"
  --set-string "controller.image.tag=${operator_tag}"
  --set-string "cluster.hypervisorImage=${hypervisor_image}"
  --set-string "providerConfigs.nvidia.images.middleware=${middleware_image}"
)
[ -n "${TF_CLIENT_TAG:-}" ] && set_args+=(--set-string "providerConfigs.nvidia.images.remoteClient=tensorfusion/tensor-fusion-client:${TF_CLIENT_TAG}")
[ -n "${TF_WORKER_TAG:-}" ] && set_args+=(--set-string "providerConfigs.nvidia.images.remoteWorker=tensorfusion/tensor-fusion-worker:${TF_WORKER_TAG}")

if [ -n "${TF_AGENT_ID:-}" ]; then
  : "${TF_ENROLL_TOKEN:?TF_ENROLL_TOKEN must be set together with TF_AGENT_ID}"
  {
    printf 'agent:\n  agentId: %s\n  enrollToken: %s\n' "$(printf '%s' "$TF_AGENT_ID" | jq -Rs .)" "$(printf '%s' "$TF_ENROLL_TOKEN" | jq -Rs .)"
    [ -n "${TF_CLOUD_ENDPOINT:-}" ] && printf '  cloudEndpoint: %s\n' "$(printf '%s' "$TF_CLOUD_ENDPOINT" | jq -Rs .)"
  } >"$secrets"
  log "cloud console agent enabled (agent id and token taken from the environment)"
else
  echo '{}' >"$secrets"
fi

version_args=()
case "$CHART" in
  tensor-fusion/*)
    helm repo add tensor-fusion https://nexusgpu.github.io/tensor-fusion --force-update >/dev/null
    helm repo update tensor-fusion >/dev/null
    [ -n "${TF_CHART_VERSION:-}" ] && version_args=(--version "${TF_CHART_VERSION}")
    ;;
esac

# The chart only renders NodeOverlay/tensor-fusion-overlay when the Karpenter
# NodeOverlay CRD already exists (helm `lookup`).
kubectl get crd nodeoverlays.karpenter.sh >/dev/null 2>&1 ||
  log "WARNING: nodeoverlays.karpenter.sh CRD not found - run 'make install' first, or the NodeOverlay will be missing"

log "installing TensorFusion: chart=${CHART} operator=${operator_repo}:${operator_tag} hypervisor=${hypervisor_image} middleware=${middleware_image}"
start=$(date +%s)
# shellcheck disable=SC2086  # TF_HELM_ARGS is intentionally word-split
helm upgrade --install "$RELEASE" "$CHART" \
  ${version_args[@]+"${version_args[@]}"} \
  --namespace "$NAMESPACE" --create-namespace \
  --wait --timeout 10m \
  -f "$values" -f "$secrets" \
  ${set_args[@]+"${set_args[@]}"} ${TF_HELM_ARGS:-}
log "helm release ready ($(($(date +%s) - start))s)"

kubectl -n "$NAMESPACE" get deploy,sts,pods -o wide || true
kubectl get tensorfusioncluster,gpupool,nodeoverlay 2>/dev/null || true
