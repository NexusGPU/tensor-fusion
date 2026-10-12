#!/usr/bin/env bash
# Rehearsal mode: make CPU nodes look like TensorFusion GPU nodes.
#
# Nodes of the Karpenter NodePool tf-mock (label e2e.tensor-fusion.ai/mock-gpu=true)
# join a second GPUPool `mock` whose hypervisor loads the repo's example
# accelerator provider (provider/example) instead of the NVIDIA one. Every such
# node then reports TF_MOCK_GPU_COUNT (default 1) mock devices (vendor STUB, 16 GiB,
# 312 TFLOPS), so scheduling, bin-packing, quota, node expansion through
# Karpenter and cleanup run for real on $0.04/h instances. Nothing here proves
# anything about CUDA, VRAM or compute limits.
#
# How the library gets onto the node without a test image or local Docker:
#   1. the provider sources + e2e/mock/{mock_patch.h,build.sh} go into a ConfigMap;
#   2. a one-off Job on the system node compiles them in a stock gcc image and
#      prints the .so base64-encoded;
#   3. that becomes ConfigMap tf-mock-accelerator (binaryData), which an extra
#      init container of the mock pool's hypervisor copies to /run/tensor-fusion.
#
#   scripts/mock-gpu.sh install      idempotent
#   scripts/mock-gpu.sh uninstall

source "$(dirname "$0")/lib.sh"
need aws kubectl jq
setup_kubeconfig

NS=tensor-fusion-sys
TFC="${TF_E2E_TFC_NAME:-tensor-fusion}"
POOL=mock
REPO_ROOT="$(cd "${TF_E2E_HOME}/../.." && pwd)"
MOCK_DIR="${TF_E2E_HOME}/../mock"
GCC_IMAGE="${TF_E2E_GCC_IMAGE:-public.ecr.aws/docker/library/gcc:13}"
BUSYBOX_IMAGE="${TF_E2E_BUSYBOX_IMAGE:-public.ecr.aws/docker/library/busybox:1.37}"
# 1 device per node mirrors the single-GPU instance types of the real GPU
# NodePool (g4dn.xlarge, g5.xlarge, g6.xlarge), so bin-packing and node counts
# in the rehearsal are the ones the real run will see.
GPU_COUNT="${TF_MOCK_GPU_COUNT:-1}"

remove_pool() {
  kubectl get tensorfusioncluster "$TFC" -o json |
    jq --arg pool "$POOL" 'del(.status, .metadata.managedFields, .metadata.resourceVersion, .metadata.uid, .metadata.generation, .metadata.creationTimestamp)
      | .spec.gpuPools |= map(select(.name != $pool))' |
    kubectl apply -f - >/dev/null
}

if [ "${1:-install}" = "uninstall" ]; then
  log "removing mock pool, ProviderConfig and accelerator ConfigMaps"
  remove_pool || true
  wait_for 180 "GPUPool ${TFC}-${POOL} deleted" sh -c "! kubectl get gpupool ${TFC}-${POOL}" || true
  kubectl delete providerconfig stub-provider --ignore-not-found
  kubectl -n "$NS" delete configmap tf-mock-src tf-mock-accelerator --ignore-not-found
  kubectl -n "$NS" delete job tf-mock-build --ignore-not-found
  no_index_overlay "${TF_E2E_CPU_NODEPOOL}" "${TF_E2E_MOCK_NODEPOOL}"
  exit 0
fi

# --- 1+2. build the library in the cluster ---------------------------------------
src_hash=$(cat "${REPO_ROOT}/provider/accelerator.h" "${REPO_ROOT}/provider/limiter.h" \
  "${REPO_ROOT}/provider/example/accelerator.c" "${REPO_ROOT}/provider/example/device_mock/driver_mock.c" \
  "${REPO_ROOT}/provider/example/device_mock/driver_mock.h" "${MOCK_DIR}/mock_patch.h" "${MOCK_DIR}/build.sh" |
  shasum -a 256 | cut -c1-16)
have_hash=$(kubectl -n "$NS" get configmap tf-mock-accelerator -o jsonpath='{.metadata.annotations.tf-e2e/source-hash}' 2>/dev/null || true)

if [ "$src_hash" != "$have_hash" ]; then
  log "building the mock accelerator in the cluster (${GCC_IMAGE})"
  kubectl -n "$NS" create configmap tf-mock-src --dry-run=client -o yaml \
    --from-file=accelerator.h="${REPO_ROOT}/provider/accelerator.h" \
    --from-file=limiter.h="${REPO_ROOT}/provider/limiter.h" \
    --from-file=accelerator.c="${REPO_ROOT}/provider/example/accelerator.c" \
    --from-file=driver_mock.c="${REPO_ROOT}/provider/example/device_mock/driver_mock.c" \
    --from-file=driver_mock.h="${REPO_ROOT}/provider/example/device_mock/driver_mock.h" \
    --from-file=mock_patch.h="${MOCK_DIR}/mock_patch.h" \
    --from-file=build.sh="${MOCK_DIR}/build.sh" | kubectl apply -f - >/dev/null

  kubectl -n "$NS" delete job tf-mock-build --ignore-not-found >/dev/null
  kubectl apply -f - >/dev/null <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: tf-mock-build
  namespace: ${NS}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        ${TF_E2E_ROLE_LABEL_KEY}: system
      containers:
        - name: build
          image: ${GCC_IMAGE}
          command: ["sh", "/src/build.sh"]
          volumeMounts:
            - {name: src, mountPath: /src}
      volumes:
        - name: src
          configMap: {name: tf-mock-src}
YAML
  wait_for 420 "mock accelerator build job" sh -c \
    "kubectl -n $NS get job tf-mock-build -o jsonpath='{.status.succeeded}{.status.failed}' | grep -q 1"
  if [ "$(kubectl -n "$NS" get job tf-mock-build -o jsonpath='{.status.succeeded}')" != "1" ]; then
    kubectl -n "$NS" logs job/tf-mock-build | tail -30 >&2
    log "mock accelerator build FAILED"
    exit 1
  fi
  lib="$(mktemp)"
  kubectl -n "$NS" logs job/tf-mock-build |
    sed -n '/-----BEGIN MOCK ACCELERATOR-----/,/-----END MOCK ACCELERATOR-----/p' | sed '1d;$d' | base64 -d >"$lib"
  file "$lib" 2>/dev/null | grep -q 'ELF 64-bit' || { log "build output is not an ELF library"; exit 1; }
  kubectl -n "$NS" create configmap tf-mock-accelerator --dry-run=client -o yaml \
    --from-file=libaccelerator_example.so="$lib" |
    kubectl annotate --local -f - "tf-e2e/source-hash=${src_hash}" -o yaml | kubectl apply -f - >/dev/null
  log "mock accelerator built: $(wc -c <"$lib" | tr -d ' ') bytes"
  rm -f "$lib"
  kubectl -n "$NS" delete job tf-mock-build --ignore-not-found >/dev/null
else
  log "mock accelerator ConfigMap is up to date"
fi

# --- 3. ProviderConfig for the example vendor --------------------------------------
# The device reports vendor "STUB" (provider/example/accelerator.c); GPU.status.vendor
# is that string and the scheduler compares it literally, so STUB is used everywhere.
# deviceMountRule "false": the example lib names /dev/example<i> as device nodes,
# which do not exist on the host.
kubectl apply -f - >/dev/null <<YAML
apiVersion: tensor-fusion.ai/v1
kind: ProviderConfig
metadata:
  name: stub-provider
spec:
  vendor: STUB
  images:
    middleware: ${BUSYBOX_IMAGE}
  hypervisor:
    deviceMount:
      deviceMountRule: "false"
YAML

# --- 4. the mock pool ---------------------------------------------------------------
# Copy componentConfig / schedulingConfigTemplate of the default pool, then swap
# the accelerator. The hypervisor container list is left exactly as the chart
# renders it (the controller resets the template when the count is unexpected).
kubectl get tensorfusioncluster "$TFC" -o json |
  jq --arg pool "$POOL" \
    --arg mockLabel "$TF_E2E_MOCK_LABEL_KEY" \
    --arg busybox "$BUSYBOX_IMAGE" \
    --arg count "$GPU_COUNT" '
    del(.status, .metadata.managedFields, .metadata.resourceVersion, .metadata.uid, .metadata.generation, .metadata.creationTimestamp)
    | (.spec.gpuPools[] | select(.isDefault == true)) as $default
    | .spec.gpuPools |= (
        map(select(.name != $pool)) + [
          ( $default
            | .name = $pool
            | .isDefault = false
            # 0 keeps a pool with no node out of "Updating"; max is a generous cap.
            | .specTemplate.capacityConfig.minResources  = { cpu: "0", memory: "0", tflops: "0",    vram: "0"     }
            | .specTemplate.capacityConfig.warmResources = { cpu: "0", memory: "0", tflops: "0",    vram: "0"     }
            | .specTemplate.capacityConfig.maxResources  = { cpu: "0", memory: "0", tflops: "5000", vram: "256Gi" }
            | .specTemplate.nodeManagerConfig.defaultVendor = "STUB"
            | .specTemplate.nodeManagerConfig.provisioningMode = "AutoSelect"
            | .specTemplate.nodeManagerConfig.nodeCompaction.period = "1m"
            | .specTemplate.nodeManagerConfig.nodeSelector = {
                nodeSelectorTerms: [ { matchExpressions: [ { key: $mockLabel, operator: "In", values: ["true"] } ] } ]
              }
            | .specTemplate.componentConfig.hypervisor.podTemplate.template.spec |= (
                .volumes = ((.volumes // []) | map(select(.name != "tf-mock-accelerator")))
                  + [ { name: "tf-mock-accelerator", configMap: { name: "tf-mock-accelerator" } } ]
                | .initContainers = ((.initContainers // []) | map(select(.name != "install-mock-accelerator")))
                  + [ {
                      name: "install-mock-accelerator",
                      image: $busybox,
                      command: ["sh", "-c", "mkdir -p /run/tensor-fusion && install -m 755 /mock/libaccelerator_example.so /run/tensor-fusion/libaccelerator_example.so"],
                      securityContext: { privileged: true, runAsUser: 0 },
                      volumeMounts: [
                        { name: "tf-mock-accelerator", mountPath: "/mock" },
                        { name: "tf-data", mountPath: "/run/tensor-fusion" }
                      ]
                    } ]
                | .containers[0].env = (
                    ((.containers[0].env // []) | map(select(.name != "TF_ACCELERATOR_LIB_PATH" and .name != "TF_HARDWARE_VENDOR" and .name != "TF_MOCK_GPU_COUNT")))
                    + [ { name: "TF_ACCELERATOR_LIB_PATH", value: "/run/tensor-fusion/libaccelerator_example.so" },
                        { name: "TF_HARDWARE_VENDOR", value: "STUB" },
                        { name: "TF_MOCK_GPU_COUNT", value: $count } ]
                  )
                | .containers[0].securityContext = ((.containers[0].securityContext // {}) + { privileged: true, runAsUser: 0, runAsGroup: 0 })
              )
          )
        ]
      )' |
  kubectl apply -f - >/dev/null

wait_for 120 "GPUPool ${TFC}-${POOL} created" kubectl get gpupool "${TFC}-${POOL}"
# Mock nodes are TensorFusion nodes from now on: give them back the index_* capacity.
no_index_overlay "${TF_E2E_CPU_NODEPOOL}"
log "mock pool ready: GPUPool ${TFC}-${POOL}, ${GPU_COUNT} mock GPU(s) per node of NodePool ${TF_E2E_MOCK_NODEPOOL}"
log "nodes appear on demand (Karpenter). Run the rehearsal with: make -C .. test-rehearsal"
kubectl get gpupool
