#!/usr/bin/env bash
# KWOK tier, stage "up": a throw-away kind cluster in which Karpenter launches
# FAKE nodes (KWOK) that look like the GPU instances of the EKS stack, plus the
# TensorFusion chart configured like the EKS rehearsal, plus the fake hypervisor
# (fakehv/) that registers GPUs for those nodes. No cloud account, no GPU.
#
#   kind (1 real node, role=system)
#     ├─ kwok-controller           fake kubelet: node heartbeats, pod phases
#     ├─ karpenter (kwok provider) NodePools tf-gpu / tf-cpu -> fake nodes
#     └─ tensor-fusion controller  the code under test, unmodified release image
#   host process: fakehv           GPU CRs + index_* capacity for claimed fake nodes
#
# Idempotent; parameters in lib.sh. Needs docker, kind, kubectl, helm, go, git.

source "$(dirname "$0")/lib.sh"
need docker kind kubectl helm "$GO"

node="${KWOK_CLUSTER}-control-plane"
karpenter_image="${KWOK_CLUSTER}/karpenter-kwok:${KARPENTER_VERSION}-shortnames"
kwok_image="registry.k8s.io/kwok/kwok:${KWOK_VERSION}"
certgen_image="$(sed -n 's/^ *image: *\(registry.k8s.io\/ingress-nginx\/kube-webhook-certgen:[^ ]*\).*/\1/p' "${TF_CHART}/values.yaml" | head -1)"
vector_image="$(sed -n 's/^ *vectorAgentImage: *\([^ ]*\).*/\1/p' "${TF_CHART}/values.yaml" | head -1)"
operator_image="${TF_OPERATOR_IMAGE}:${TF_RELEASE_TAG}"

# --- 1. images ---------------------------------------------------------------------
# The hypervisor image is not loaded: pods on KWOK nodes are never started, so
# it is only a string in the pod spec.
for img in "$KWOK_NODE_IMAGE" "$kwok_image" "$operator_image" "$certgen_image" "$vector_image"; do
  ensure_image "$img"
done

if ! docker image inspect "$karpenter_image" >/dev/null 2>&1; then
  if [ ! -d "${KARPENTER_SRC}/kwok" ]; then
    need git
    log "cloning ${KARPENTER_REPO} ${KARPENTER_VERSION}"
    git clone -q --depth 1 --branch "$KARPENTER_VERSION" "$KARPENTER_REPO" "$KARPENTER_SRC"
  fi
  # The provider names nodes kwok-<nodeclaim>-<two words>-<n>-<random>-<uint32>,
  # 65+ characters for a NodeClaim of the TensorFusion expander. The hypervisor
  # (and so the fake one) labels each GPU with tensor-fusion.ai/managed-by=<node
  # name>, and a label value is limited to 63 bytes: such a node can never
  # register a GPU. Node names are shortened to kwok-<nodeclaim>-<uint32>; this
  # is the only change to the upstream source.
  if grep -q 'kwok-%s-%s-%d' "${KARPENTER_SRC}/kwok/cloudprovider/cloudprovider.go"; then
    sed -i.orig 's|fmt.Sprintf("kwok-%s-%s-%d", nodeClaim.Name, kwokutils.RandomName(), rand.Uint32())|fmt.Sprintf("kwok-%s-%d", nodeClaim.Name, rand.Uint32())|' \
      "${KARPENTER_SRC}/kwok/cloudprovider/cloudprovider.go"
    sed -i.orig2 '/kwokutils "sigs.k8s.io\/karpenter\/kwok\/utils"/d' "${KARPENTER_SRC}/kwok/cloudprovider/cloudprovider.go"
    rm -f "${KWOK_HOME}/bin/karpenter-kwok"
  fi
  if [ ! -x "${KWOK_HOME}/bin/karpenter-kwok" ]; then
    log "building the Karpenter KWOK provider ${KARPENTER_VERSION} from source"
    (cd "$KARPENTER_SRC" && CGO_ENABLED=0 GOFLAGS=-mod=mod "$GO" build -o "${KWOK_HOME}/bin/karpenter-kwok" ./kwok)
  fi
  ctx="$(mktemp -d)"
  cp "${KWOK_HOME}/bin/karpenter-kwok" "$ctx/karpenter"
  printf 'FROM scratch\nCOPY karpenter /karpenter\nUSER 65532\nENTRYPOINT ["/karpenter"]\n' >"$ctx/Dockerfile"
  docker build -q -t "$karpenter_image" "$ctx" >/dev/null
  rm -rf "$ctx"
fi

log "building the fake hypervisor"
(cd "$REPO_ROOT" && CGO_ENABLED=0 "$GO" build -tags e2e -mod=vendor -o "${KWOK_HOME}/bin/fakehv" ./e2e/kwok/fakehv)

# --- 2. kind -----------------------------------------------------------------------
if ! cluster_exists; then
  log "creating kind cluster ${KWOK_CLUSTER} (${KWOK_NODE_IMAGE}, memory ${KWOK_NODE_MEMORY})"
  kind create cluster --name "$KWOK_CLUSTER" --image "$KWOK_NODE_IMAGE" --kubeconfig "$KUBECONFIG" --wait 120s --config - <<YAML
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    labels:
      ${ROLE_LABEL}: system
YAML
  docker update --memory "$KWOK_NODE_MEMORY" --memory-swap "$KWOK_NODE_MEMORY" "$node" >/dev/null
else
  kind export kubeconfig --name "$KWOK_CLUSTER" --kubeconfig "$KUBECONFIG" >/dev/null 2>&1
fi
chmod 600 "$KUBECONFIG"
log "loading images into the kind node"
kind load docker-image --name "$KWOK_CLUSTER" "$kwok_image" "$karpenter_image" "$operator_image" "$certgen_image" "$vector_image" >/dev/null

# --- 3. KWOK: the fake kubelet --------------------------------------------------------
for f in kwok.yaml stage-fast.yaml; do
  if [ ! -s "${KWOK_MANIFEST_DIR}/${f}" ]; then
    need curl
    mkdir -p "$KWOK_MANIFEST_DIR"
    curl -fsSL -o "${KWOK_MANIFEST_DIR}/${f}" "${KWOK_RELEASE_URL}/${KWOK_VERSION}/${f}"
  fi
done
log "installing KWOK ${KWOK_VERSION}"
kubectl apply --server-side -f "${KWOK_MANIFEST_DIR}/kwok.yaml" >/dev/null
# Nothing of the control plane may land on a fake node.
kubectl -n kube-system patch deploy kwok-controller --type merge \
  -p "{\"spec\":{\"template\":{\"spec\":{\"nodeSelector\":{\"${ROLE_LABEL}\":\"system\"}}}}}" >/dev/null
kubectl -n kube-system patch deploy coredns --type merge \
  -p "{\"spec\":{\"template\":{\"spec\":{\"nodeSelector\":{\"${ROLE_LABEL}\":\"system\"}}}}}" >/dev/null
wait_for 60 "KWOK Stage CRD" kubectl get crd stages.kwok.x-k8s.io
kubectl apply -f "${KWOK_MANIFEST_DIR}/stage-fast.yaml" >/dev/null
kubectl -n kube-system rollout status deploy/kwok-controller --timeout=180s >/dev/null

# --- 4. Karpenter with the KWOK cloud provider ---------------------------------------
log "installing Karpenter ${KARPENTER_VERSION} (kwok provider)"
kubectl apply --server-side -f "${KARPENTER_SRC}/kwok/charts/crds" >/dev/null
kubectl -n kube-system create configmap "${KWOK_CLUSTER}-instance-types" \
  --from-file=instance-types.json="${KWOK_DIR}/instance-types.json" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
# featureGates: the chart's FEATURE_GATES line has no NodeOverlay entry, so the
# variable is set again (the later definition wins); the TensorFusion chart's
# NodeOverlay is ignored by Karpenter without the gate.
helm upgrade --install karpenter "${KARPENTER_SRC}/kwok/charts" --namespace kube-system --skip-crds \
  --wait --timeout 5m -f - >/dev/null <<YAML
replicas: 1
nodeSelector:
  ${ROLE_LABEL}: system
controller:
  image:
    repository: ${karpenter_image%:*}
    tag: ${karpenter_image##*:}
  resources:
    requests: {cpu: 200m, memory: 256Mi}
    limits: {memory: 1Gi}
  env:
    - name: FEATURE_GATES
      value: "NodeOverlay=true,NodeRepair=false,ReservedCapacity=true,SpotToSpotConsolidation=false,StaticCapacity=false,CapacityBuffer=false"
    - name: INSTANCE_TYPES_FILE_PATH
      value: /etc/kwok/instance-types.json
  extraVolumeMounts:
    - name: instance-types
      mountPath: /etc/kwok
extraVolumes:
  - name: instance-types
    configMap:
      name: ${KWOK_CLUSTER}-instance-types
settings:
  featureGates:
    staticCapacity: false
    capacityBuffer: false
    spotToSpotConsolidation: false
    nodeRepair: false
YAML

log "applying KWOKNodeClass and NodePools"
nodepool() { # nodepool <name> <weight> <instance json> <limits> <policy> <after> <labels>
  cat <<YAML
---
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: $1
spec:
  weight: $2
  template:
    metadata:
      labels: $7
    spec:
      nodeClassRef:
        group: karpenter.kwok.sh
        kind: KWOKNodeClass
        name: ${NODECLASS}
      expireAfter: Never
      requirements:
        - key: node.kubernetes.io/instance-type
          operator: In
          values: $3
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: kubernetes.io/os
          operator: In
          values: ["linux"]
  limits: $4
  disruption:
    consolidationPolicy: $5
    consolidateAfter: $6
YAML
}
{
  cat <<YAML
apiVersion: karpenter.kwok.sh/v1alpha1
kind: KWOKNodeClass
metadata:
  name: ${NODECLASS}
spec:
  nodeRegistrationDelay: ${KWOK_REGISTRATION_DELAY}
YAML
  # Same shape as the EKS stack (terraform/scripts/install.sh): the general-purpose
  # pool has the higher weight, so a pod that could go anywhere never costs a GPU
  # node; GPU nodes carry the labels the chart (initialGpuNodeLabelSelector) and
  # the default GPUPool select on.
  nodepool "$CPU_NODEPOOL" 100 '["c5.large", "m5.large", "t3.large"]' "{cpu: 16}" WhenEmptyOrUnderutilized 30s \
    "{${ROLE_LABEL}: cpu}"
  nodepool "$GPU_NODEPOOL" 10 '["g4dn.xlarge", "g4dn.2xlarge", "g5.xlarge", "g6.xlarge"]' "{cpu: 64}" WhenEmpty "$KWOK_GPU_CONSOLIDATE" \
    "{${ROLE_LABEL}: gpu, nvidia.com/gpu.present: \"true\", ${SHARED_POOL_LABEL}: \"true\"}"
} | kubectl apply -f - >/dev/null

# --- 5. TensorFusion ------------------------------------------------------------------
# Values of the EKS stack (terraform/scripts/install-tf.sh) where they apply.
# Differences, all forced by the environment:
#   - GreptimeDB, alertmanager and the hypervisor's vector sidecar are off, and so
#     are -enable-alert / -enable-auto-scale (they need the TSDB);
#   - karpenterNodeOverlay.requirements: the chart default selects on labels of
#     Karpenter's AWS provider, which the KWOK provider does not set. The KWOK
#     equivalent of "GPU instance types only" is the instance family.
log "installing TensorFusion: chart=${TF_CHART} operator=${operator_image} hypervisor=${TF_HYPERVISOR_IMAGE}"
helm upgrade --install "$TF_RELEASE" "$TF_CHART" --namespace "$TF_NAMESPACE" --create-namespace \
  --wait --timeout 8m -f - ${TF_HELM_ARGS:-} >/dev/null <<YAML
cluster:
  hypervisorImage: ${TF_HYPERVISOR_IMAGE}
  enableVector: false
  pool:
    nodeSelectorKey: ${SHARED_POOL_LABEL}
controller:
  image:
    repository: ${TF_OPERATOR_IMAGE}
    tag: "${TF_RELEASE_TAG}"
  command:
    - /manager
    - -metrics-bind-address
    - :9000
    - -leader-elect
    - -enable-auto-expander
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
          - matchExpressions:
              - key: ${ROLE_LABEL}
                operator: In
                values: ["system"]
greptime:
  installStandalone: false
alert:
  enabled: false
karpenterNodeOverlay:
  requirements:
    - key: karpenter.kwok.sh/instance-family
      operator: In
      values: ["g4dn", "g5", "g6"]
YAML
wait_for 120 "default GPUPool" kubectl get gpupool tensor-fusion-shared

# Path B (provisioningMode: Karpenter) reads the NodeClass named in the pool, and
# the chart's ClusterRole has no rule for any Karpenter NodeClass kind, so the
# GPUNodeClaim controller is refused and never creates the NodeClaim (KRP-B02
# fails on the chart as shipped). KWOK_NODECLASS_RBAC=1 adds the missing read
# permission so that the rest of path B can be exercised; default off, because
# the default run must show what a user gets.
if [ "${KWOK_NODECLASS_RBAC:-0}" = "1" ]; then
  log "granting the operator read access to KWOKNodeClass (workaround, see KRP-B02)"
  kubectl apply -f - >/dev/null <<YAML
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ${KWOK_CLUSTER}-nodeclass-reader
rules:
  - apiGroups: ["karpenter.kwok.sh"]
    resources: ["kwoknodeclasses"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: ${KWOK_CLUSTER}-nodeclass-reader
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: ${KWOK_CLUSTER}-nodeclass-reader
subjects:
  - kind: ServiceAccount
    name: $(kubectl -n "$TF_NAMESPACE" get deploy "${TF_RELEASE}-controller" -o jsonpath='{.spec.template.spec.serviceAccountName}')
    namespace: ${TF_NAMESPACE}
YAML
else
  kubectl delete clusterrolebinding "${KWOK_CLUSTER}-nodeclass-reader" --ignore-not-found >/dev/null 2>&1 || true
fi

# --- 6. fake hypervisor ---------------------------------------------------------------
if [ -f "${KWOK_HOME}/fakehv.pid" ] && kill -0 "$(cat "${KWOK_HOME}/fakehv.pid")" 2>/dev/null; then
  kill "$(cat "${KWOK_HOME}/fakehv.pid")" 2>/dev/null || true
fi
nohup "${KWOK_HOME}/bin/fakehv" -delay "$KWOK_FAKE_HV_DELAY" -gpus-per-node "$KWOK_FAKE_GPUS_PER_NODE" -tf-namespace "$TF_NAMESPACE" \
  >>"${KWOK_HOME}/logs/fakehv.log" 2>&1 &
echo $! >"${KWOK_HOME}/fakehv.pid"
log "fake hypervisor running (pid $(cat "${KWOK_HOME}/fakehv.pid"), delay ${KWOK_FAKE_HV_DELAY}, log ${KWOK_HOME}/logs/fakehv.log)"

log "up. export KUBECONFIG=${KUBECONFIG}"
kubectl get nodes -L "${ROLE_LABEL}" || true
kubectl get nodepool,kwoknodeclass,nodeoverlay,gpupool || true
kubectl -n "$TF_NAMESPACE" get pods || true
