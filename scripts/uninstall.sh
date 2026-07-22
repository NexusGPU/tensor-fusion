#!/usr/bin/env bash

# Cluster-wide TensorFusion uninstaller.
# Supports Helm, rendered Helm manifests, and Kustomize installs.

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

NAMESPACE=${NAMESPACE:-tensor-fusion-sys}
KUSTOMIZE_NAMESPACE=${KUSTOMIZE_NAMESPACE:-tensor-fusion}
HELM_RELEASE=${HELM_RELEASE:-tensor-fusion-sys}
RESOURCE_PREFIX=${RESOURCE_PREFIX:-${HELM_RELEASE}}
LABEL_DOMAIN=${LABEL_DOMAIN:-tensor-fusion.ai}
WAIT_TIMEOUT_SECONDS=${WAIT_TIMEOUT_SECONDS:-600}
POLL_INTERVAL_SECONDS=${POLL_INTERVAL_SECONDS:-2}
CHART_DIR=${CHART_DIR:-${ROOT_DIR}/charts/tensor-fusion}
KUSTOMIZE_DIR=${KUSTOMIZE_DIR:-${ROOT_DIR}/config/default}

ALL_CRDS=(
    gpunodeclaims.tensor-fusion.ai
    gpunodeclasses.tensor-fusion.ai
    gpunodes.tensor-fusion.ai
    gpupools.tensor-fusion.ai
    gpuresourcequotas.tensor-fusion.ai
    gpus.tensor-fusion.ai
    providerconfigs.tensor-fusion.ai
    schedulingconfigtemplates.tensor-fusion.ai
    tensorfusionclusters.tensor-fusion.ai
    tensorfusionconnections.tensor-fusion.ai
    tensorfusionworkloads.tensor-fusion.ai
    workloadprofiles.tensor-fusion.ai
)

NODE_LABEL_SUFFIXES=(
    watch-node isolationMode hardware-vendor node-selector-hash
    node-provisioner orphan expansion-source should-delete
)

TENSORFUSION_PVS=()

log() {
    printf '[tensor-fusion uninstall] %s\n' "$*"
}

die() {
    printf '[tensor-fusion uninstall] ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    printf '%s\n' \
        'Usage: ./scripts/uninstall.sh [--help]' \
        '' \
        'Completely removes TensorFusion from the current kubectl cluster.' \
        'Supported installs: Helm, helm-rendered manifests, and Kustomize.' \
        '' \
        'Environment variables:' \
        '  NAMESPACE              Helm/manifest namespace (default: tensor-fusion-sys)' \
        '  KUSTOMIZE_NAMESPACE    Kustomize namespace (default: tensor-fusion)' \
        '  HELM_RELEASE           Helm release name (default: tensor-fusion-sys)' \
        '  RESOURCE_PREFIX        Rendered resource/PVC prefix (default: HELM_RELEASE)' \
        '  LABEL_DOMAIN           Node label/resource domain (default: tensor-fusion.ai)' \
        '  WAIT_TIMEOUT_SECONDS   Finalizer/delete timeout (default: 600)' \
        '  CHART_DIR              Local Helm chart path' \
        '  KUSTOMIZE_DIR          Local Kustomize path' \
        '' \
        'The script deletes all TensorFusion CRs/CRDs, control-plane resources,' \
        'TensorFusion namespaces, PVCs/PVs, Node labels, taints, NodeOverlay, and' \
        'schedulable tensor-fusion.ai/index* capacity. Zero-valued index entries may' \
        'remain in kubelet checkpoints until kubelet restarts. Vendor labels such' \
        'as nvidia.com/gpu.present and huawei.com/npu.present are preserved.'
}

is_namespaced_crd() {
    case "$1" in
        gpuresourcequotas.tensor-fusion.ai|tensorfusionconnections.tensor-fusion.ai|tensorfusionworkloads.tensor-fusion.ai|workloadprofiles.tensor-fusion.ai)
            return 0
            ;;
        *) return 1 ;;
    esac
}

crd_exists() {
    kubectl get crd "$1" >/dev/null 2>&1
}

delete_cr_instances() {
    local crd=$1
    local resource=${crd%%.*}
    local args=(delete "$crd" --all --ignore-not-found=true --wait=false)

    if ! crd_exists "$crd"; then
        log "CRD $crd is not installed; skipping"
        return
    fi
    is_namespaced_crd "$crd" && args+=(--all-namespaces)

    log "Marking all $resource resources for deletion"
    kubectl "${args[@]}"
}

wait_for_cr_instances() {
    local crd=$1
    local resource=${crd%%.*}
    local deadline=$((SECONDS + WAIT_TIMEOUT_SECONDS))
    local args=(get "$crd")
    local remaining

    crd_exists "$crd" || return 0
    is_namespaced_crd "$crd" && args+=(--all-namespaces)
    args+=(-o name --ignore-not-found)

    while true; do
        remaining=$(kubectl "${args[@]}") || die "failed to list $resource"
        if [[ -z "$remaining" ]]; then
            log "All $resource resources have been deleted"
            return
        fi
        if ((SECONDS >= deadline)); then
            printf '%s\n' "$remaining" >&2
            die "timed out waiting for $resource; controller is still required for finalizers"
        fi
        sleep "$POLL_INTERVAL_SECONDS"
    done
}

delete_and_wait_for_crs() {
    local crd
    for crd in "$@"; do delete_cr_instances "$crd"; done
    for crd in "$@"; do wait_for_cr_instances "$crd"; done
}

delete_tensorfusion_crs() {
    log "Deleting namespaced CRs while the controller is running"
    delete_and_wait_for_crs \
        tensorfusionconnections.tensor-fusion.ai \
        tensorfusionworkloads.tensor-fusion.ai \
        workloadprofiles.tensor-fusion.ai \
        gpuresourcequotas.tensor-fusion.ai

    log "Stopping Cluster reconciliation before deleting Pools and Nodes"
    delete_cr_instances tensorfusionclusters.tensor-fusion.ai
    delete_cr_instances gpupools.tensor-fusion.ai
    delete_cr_instances gpunodeclaims.tensor-fusion.ai
    delete_cr_instances gpunodes.tensor-fusion.ai
    delete_cr_instances gpus.tensor-fusion.ai

    wait_for_cr_instances gpunodeclaims.tensor-fusion.ai
    wait_for_cr_instances gpus.tensor-fusion.ai
    wait_for_cr_instances gpunodes.tensor-fusion.ai
    wait_for_cr_instances gpupools.tensor-fusion.ai
    wait_for_cr_instances tensorfusionclusters.tensor-fusion.ai

    delete_and_wait_for_crs \
        gpunodeclasses.tensor-fusion.ai \
        schedulingconfigtemplates.tensor-fusion.ai \
        providerconfigs.tensor-fusion.ai
}

cleanup_helm_install() {
    command -v helm >/dev/null 2>&1 || return 0

    if helm status "$HELM_RELEASE" -n "$NAMESPACE" >/dev/null 2>&1; then
        log "Uninstalling Helm release $HELM_RELEASE"
        helm uninstall "$HELM_RELEASE" -n "$NAMESPACE" \
            --wait --timeout "${WAIT_TIMEOUT_SECONDS}s"
    fi

    if [[ -d "$CHART_DIR" ]]; then
        log "Deleting resources from a rendered Helm manifest"
        helm template "$HELM_RELEASE" "$CHART_DIR" -n "$NAMESPACE" \
            --set greptime.installStandalone=false \
            --set cluster.enabled=false \
            --set providerConfigs.nvidia.enabled=false |
            kubectl delete -f - --ignore-not-found=true --wait=true \
                --timeout "${WAIT_TIMEOUT_SECONDS}s"
    fi
}

cleanup_kustomize_install() {
    [[ -f "$KUSTOMIZE_DIR/kustomization.yaml" ]] || return 0
    log "Deleting resources described by $KUSTOMIZE_DIR"
    kubectl delete -k "$KUSTOMIZE_DIR" --ignore-not-found=true --wait=true \
        --timeout "${WAIT_TIMEOUT_SECONDS}s"
}

cleanup_fixed_name_resources() {
    local namespaced_resources=(
        "deployment/${RESOURCE_PREFIX}-controller"
        "statefulset/${RESOURCE_PREFIX}-alert-manager"
        "service/${RESOURCE_PREFIX}"
        "service/${RESOURCE_PREFIX}-webhook"
        service/alert-manager
        service/alert-manager-headless
        "serviceaccount/${RESOURCE_PREFIX}"
        serviceaccount/tensor-fusion-hypervisor-sa
        "serviceaccount/${RESOURCE_PREFIX}-webhook-job"
        "configmap/${RESOURCE_PREFIX}-config"
        "configmap/${RESOURCE_PREFIX}-public-gpu-info"
        "configmap/${RESOURCE_PREFIX}-vector-config"
        "configmap/${RESOURCE_PREFIX}-alert-manager-config"
        configmap/tensor-fusion-operator-leader-info
        secret/tensor-fusion-webhook-secret
        secret/tf-cloud-vendor-credentials
        "secret/${RESOURCE_PREFIX}-greptimedb-secret"
        "job/${RESOURCE_PREFIX}-add-hook-crt"
        "job/${RESOURCE_PREFIX}-patch-admission-webhook"
        "role/${RESOURCE_PREFIX}-webhook-job"
        "rolebinding/${RESOURCE_PREFIX}-webhook-job"
    )
    local cluster_resources=(
        "clusterrole/${RESOURCE_PREFIX}-role"
        clusterrole/tensor-fusion-hypervisor-role
        "clusterrole/${RESOURCE_PREFIX}-webhook-job"
        "clusterrolebinding/${RESOURCE_PREFIX}-rolebinding"
        clusterrolebinding/tensor-fusion-hypervisor-rolebinding
        "clusterrolebinding/${RESOURCE_PREFIX}-webhook-job"
        "mutatingwebhookconfiguration/${RESOURCE_PREFIX}-mutating-webhook"
        priorityclass/tensor-fusion-critical
        priorityclass/tensor-fusion-high
        priorityclass/tensor-fusion-medium
    )

    log "Deleting fixed-name resources used by manifest installs"
    kubectl -n "$NAMESPACE" delete "${namespaced_resources[@]}" \
        --ignore-not-found=true --wait=true --timeout "${WAIT_TIMEOUT_SECONDS}s"
    kubectl delete "${cluster_resources[@]}" --ignore-not-found=true \
        --wait=true --timeout "${WAIT_TIMEOUT_SECONDS}s"
    kubectl -n "$NAMESPACE" delete pod \
        -l "${LABEL_DOMAIN}/component=hypervisor" --ignore-not-found=true
}

capture_tensorfusion_pvs() {
    local namespace pvc pv

    for namespace in "$NAMESPACE" "$KUSTOMIZE_NAMESPACE" greptimedb; do
        kubectl get namespace "$namespace" >/dev/null 2>&1 || continue
        while IFS=$'\t' read -r pvc pv; do
            case "$pvc" in
                alertmanager-storage-"${RESOURCE_PREFIX}"-alert-manager-*|datanode-"${RESOURCE_PREFIX}"-greptimedb-standalone-*)
                    [[ -n "$pv" ]] && TENSORFUSION_PVS+=("$pv")
                    ;;
            esac
        done < <(kubectl -n "$namespace" get pvc \
            -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.volumeName}{"\n"}{end}')
    done
}

cleanup_greptime_and_pvcs() {
    local namespace pvc

    if kubectl get namespace greptimedb >/dev/null 2>&1; then
        log "Deleting TensorFusion standalone GreptimeDB"
        kubectl -n greptimedb delete \
            "statefulset/${RESOURCE_PREFIX}-greptimedb-standalone" \
            "configmap/${RESOURCE_PREFIX}-greptimedb-standalone" \
            service/greptimedb-standalone --ignore-not-found=true \
            --wait=true --timeout "${WAIT_TIMEOUT_SECONDS}s"
    fi

    log "Deleting TensorFusion-owned PVCs"
    for namespace in "$NAMESPACE" "$KUSTOMIZE_NAMESPACE" greptimedb; do
        kubectl get namespace "$namespace" >/dev/null 2>&1 || continue
        while IFS= read -r pvc; do
            case "$pvc" in
                persistentvolumeclaim/alertmanager-storage-"${RESOURCE_PREFIX}"-alert-manager-*|persistentvolumeclaim/datanode-"${RESOURCE_PREFIX}"-greptimedb-standalone-*)
                    kubectl -n "$namespace" delete "$pvc" --ignore-not-found=true
                    ;;
            esac
        done < <(kubectl -n "$namespace" get pvc -o name)
    done
}

cleanup_node_resources() {
    local suffix key node
    local capacity_entries=""
    local allocatable_entries=""
    local patch=""
    local remove_labels=()
    local resource_suffixes=(index index_{0..15} index_{a..f})

    if kubectl get crd nodeoverlays.karpenter.sh >/dev/null 2>&1; then
        log "Deleting Karpenter TensorFusion NodeOverlay"
        kubectl delete nodeoverlay.karpenter.sh tensor-fusion-overlay \
            --ignore-not-found=true --wait=true --timeout "${WAIT_TIMEOUT_SECONDS}s"
    fi

    for suffix in "${NODE_LABEL_SUFFIXES[@]}"; do
        remove_labels+=("${LABEL_DOMAIN}/${suffix}-")
    done
    log "Removing TensorFusion Node labels; vendor *.present labels are preserved"
    kubectl label nodes --all "${remove_labels[@]}" --overwrite

    key="${LABEL_DOMAIN}/used-by"
    kubectl taint nodes --all "${key}-" >/dev/null 2>&1 || true

    for suffix in "${NODE_LABEL_SUFFIXES[@]}"; do
        key="${LABEL_DOMAIN}/${suffix}"
        [[ -z $(kubectl get nodes -l "$key" -o name) ]] || die "Node label $key still exists"
    done

    for suffix in "${resource_suffixes[@]}"; do
        key="${LABEL_DOMAIN}/${suffix}"
        capacity_entries+="\"${key}\":null,"
        allocatable_entries+="\"${key}\":null,"
    done
    patch="{\"status\":{\"capacity\":{${capacity_entries%,}},\"allocatable\":{${allocatable_entries%,}}}}"

    log "Removing TensorFusion extended resources from Node status"
    while IFS= read -r node; do
        kubectl patch "$node" --subresource=status --type=merge -p "$patch" >/dev/null
    done < <(kubectl get nodes -o name)

    sleep 5
    node=$(kubectl get nodes -o json)
    if grep -Eq "\"${LABEL_DOMAIN}/index[^\"]*\"[[:space:]]*:[[:space:]]*\"[1-9]" <<<"$node"; then
        die "non-zero TensorFusion extended resources still exist in Node status"
    fi
    if grep -q "\"${LABEL_DOMAIN}/index" <<<"$node"; then
        log "Zero-valued TensorFusion index resources remain in kubelet checkpoints; kubelet restart is required to remove their keys"
    fi
}

delete_crds() {
    local crd
    log "Deleting all current TensorFusion CRDs"
    for crd in "${ALL_CRDS[@]}"; do
        kubectl delete crd "$crd" --ignore-not-found=true --wait=true \
            --timeout "${WAIT_TIMEOUT_SECONDS}s"
    done
    for crd in "${ALL_CRDS[@]}"; do
        crd_exists "$crd" && die "CRD $crd still exists"
    done
    return 0
}

delete_namespaces() {
    local namespace
    for namespace in "$NAMESPACE" "$KUSTOMIZE_NAMESPACE" greptimedb; do
        log "Deleting namespace $namespace"
        kubectl delete namespace "$namespace" --ignore-not-found=true \
            --wait=true --timeout "${WAIT_TIMEOUT_SECONDS}s"
    done
}

delete_tensorfusion_pvs() {
    local pv
    for pv in "${TENSORFUSION_PVS[@]}"; do
        log "Deleting TensorFusion persistent volume $pv"
        kubectl delete pv "$pv" --ignore-not-found=true --wait=true \
            --timeout "${WAIT_TIMEOUT_SECONDS}s"
    done
}

main() {
    local context

    case "${1:-}" in
        -h | --help)
            usage
            return 0
            ;;
        '') ;;
        *)
            usage >&2
            die "unknown argument: $1"
            ;;
    esac

    command -v kubectl >/dev/null 2>&1 || die "kubectl command not found"
    kubectl version --request-timeout=10s >/dev/null || die "cannot connect to Kubernetes"
    context=$(kubectl config current-context) || die "cannot determine kubectl context"

    log "kubectl context: $context"
    log "starting full cluster-wide uninstall"

    capture_tensorfusion_pvs
    delete_tensorfusion_crs
    cleanup_helm_install
    cleanup_kustomize_install
    cleanup_fixed_name_resources
    cleanup_greptime_and_pvcs
    cleanup_node_resources
    delete_crds
    delete_namespaces
    delete_tensorfusion_pvs

    log "TensorFusion cluster-wide uninstall completed successfully"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
