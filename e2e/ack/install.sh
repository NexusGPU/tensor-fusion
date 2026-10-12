#!/usr/bin/env bash
# ACK tier, stage 2: Karpenter with the Alibaba Cloud provider
# (AliyunContainerService/karpenter-provider-alibabacloud) and the NodePools the
# suite expects, in the shape of the EKS stack (terraform/scripts/install.sh):
#
#   RAM role ${KARPENTER_ROLE}   assumed by ServiceAccount kube-system/karpenter through RRSA (no AccessKey in the cluster)
#   ECSNodeClass tf-gpu / tf-cpu the cluster's own node image, security group and worker RAM role
#   NodePool tf-cpu  weight 100  general-purpose nodes
#   NodePool tf-gpu  weight 10   single-T4 instances, on-demand, limit nvidia.com/gpu=${ACK_GPU_NODE_LIMIT}
#
# Idempotent; parameters in lib.sh.

source "$(dirname "$0")/lib.sh"
need aliyun kubectl helm jq

start=$(date +%s)
cid=$(state_get CLUSTER_ID)
[ -n "$cid" ] || { log "no cluster in ${STATE}; run up.sh first"; exit 1; }
detail=$(ali cs GET "/clusters/${cid}")
uid=$(aliyun sts GetCallerIdentity | jq -r .AccountId)
sg=$(echo "$detail" | jq -r .security_group_id)
worker_role=$(echo "$detail" | jq -r .worker_ram_role_name)
endpoint=$(echo "$detail" | jq -r '.master_url | fromjson | .intranet_api_server_endpoint')
image=$(ali cs GET "/clusters/${cid}/nodepools" | jq -r '.nodepools[] | select(.nodepool_info.name == "system") | .scaling_group.image_id')
log "cluster ${cid}: endpoint ${endpoint}, node image ${image}, worker role ${worker_role}"

# --- RAM role for the controller (RRSA) ------------------------------------------------
oidc=$(aliyun ims ListOIDCProviders | jq -c --arg n "ack-rrsa-${cid}" '.OIDCProviders.OIDCProvider[] | select(.OIDCProviderName == $n)')
[ -n "$oidc" ] || { log "no OIDC provider ack-rrsa-${cid}: RRSA is not enabled on the cluster"; exit 1; }
oidc_arn=$(echo "$oidc" | jq -r .Arn)
trust=$(jq -nc --arg arn "$oidc_arn" --arg iss "$(echo "$oidc" | jq -r .IssuerUrl)" '{Version: "1", Statement: [{
  Effect: "Allow", Action: "sts:AssumeRole", Principal: {Federated: [$arn]},
  Condition: {StringEquals: {"oidc:aud": "sts.aliyuncs.com", "oidc:iss": $iss, "oidc:sub": "system:serviceaccount:kube-system:karpenter"}}}]}')
# The provider's documented policy, plus PassRole for the worker role the nodes
# get. PassRole scoped to the role's ARN is refused by RunInstances
# (InvalidUser.PassRoleForbidden, tried with both spellings of the name); the
# service condition is the scoping ECS honours.
policy=$(jq -nc '{Version: "1", Statement: [
  {Effect: "Allow", Resource: "*", Action: ["cs:DescribeClusterAttachScripts", "cs:GetClusterAddonInstance", "cs:DescribeClusterDetail"]},
  {Effect: "Allow", Resource: "*", Action: ["ecs:RunInstances", "ecs:DescribeInstances", "ecs:DeleteInstances", "ecs:TagResources",
    "ecs:CreateLaunchTemplate", "ecs:DescribeLaunchTemplates", "ecs:DeleteLaunchTemplate", "ecs:DescribeInstanceTypeResource",
    "ecs:DescribeImages", "ecs:DescribeSecurityGroups", "ecs:DescribeCapacityReservations", "ecs:DescribePrice",
    "ecs:DescribeInstanceTypes", "ecs:DescribeZones", "ecs:DescribeAvailableResource"]},
  {Effect: "Allow", Resource: "*", Action: ["vpc:DescribeVSwitches"]},
  {Effect: "Allow", Resource: "*", Action: ["ram:GetRole"]},
  {Effect: "Allow", Resource: "*", Action: ["ram:PassRole"], Condition: {StringEquals: {"acs:Service": "ecs.aliyuncs.com"}}}]}')
if aliyun ram GetRole --RoleName "$KARPENTER_ROLE" >/dev/null 2>&1; then
  aliyun ram UpdateRole --RoleName "$KARPENTER_ROLE" --NewAssumeRolePolicyDocument "$trust" >/dev/null
else
  log "creating RAM role ${KARPENTER_ROLE}"
  aliyun ram CreateRole --RoleName "$KARPENTER_ROLE" --AssumeRolePolicyDocument "$trust" --Description "Karpenter controller of ACK cluster ${ACK_NAME} (tf e2e)" >/dev/null
fi
# An existing policy is updated in place, never deleted and recreated: in that
# window a RunInstances that succeeded is followed by a refused DescribeInstances,
# and the provider (0.2.5) retries the launch without removing the first
# instance, leaving a second ECS instance for the same NodeClaim.
if aliyun ram GetPolicy --PolicyType Custom --PolicyName "$KARPENTER_ROLE" >/dev/null 2>&1; then
  aliyun ram CreatePolicyVersion --PolicyName "$KARPENTER_ROLE" --PolicyDocument "$policy" --SetAsDefault true \
    --RotateStrategy DeleteOldestNonDefaultVersionWhenLimitExceeded >/dev/null
else
  aliyun ram CreatePolicy --PolicyName "$KARPENTER_ROLE" --PolicyDocument "$policy" >/dev/null
fi
aliyun ram ListPoliciesForRole --RoleName "$KARPENTER_ROLE" | jq -e --arg n "$KARPENTER_ROLE" '.Policies.Policy[] | select(.PolicyName == $n)' >/dev/null ||
  aliyun ram AttachPolicyToRole --PolicyType Custom --PolicyName "$KARPENTER_ROLE" --RoleName "$KARPENTER_ROLE" >/dev/null
role_arn="acs:ram::${uid}:role/${KARPENTER_ROLE}"

# --- Karpenter ---------------------------------------------------------------------------
# The packaged chart, not a clone: Go sources under the repository would be
# picked up by `make manifests` (controller-gen walks ./...).
chart="${ACK_HOME}/charts/karpenter-${KARPENTER_ALIBABA_VERSION}.tgz"
if [ ! -s "$chart" ]; then
  need curl
  mkdir -p "$(dirname "$chart")"
  log "downloading the Karpenter chart ${KARPENTER_ALIBABA_VERSION} (${KARPENTER_ALIBABA_REPO})"
  curl -fsSL -o "$chart" "${KARPENTER_ALIBABA_REPO}/raw/${KARPENTER_ALIBABA_CHART_REF}/helm-charts/karpenter-${KARPENTER_ALIBABA_VERSION}.tgz"
fi
# KARPENTER_ALIBABA_IMAGE: a patched build that build-karpenter.sh loaded onto the system node.
image_values=""
if [ -n "${KARPENTER_ALIBABA_IMAGE:-}" ]; then
  image_values="  image: {repository: \"${KARPENTER_ALIBABA_IMAGE%:*}\", tag: \"${KARPENTER_ALIBABA_IMAGE##*:}\", pullPolicy: IfNotPresent}"
fi
log "installing Karpenter (Alibaba Cloud provider ${KARPENTER_ALIBABA_VERSION}${KARPENTER_ALIBABA_IMAGE:+, image ${KARPENTER_ALIBABA_IMAGE}})"
helm show crds "$chart" | kubectl apply --server-side --force-conflicts -f - >/dev/null
# NodeOverlay gate: without it Karpenter ignores the NodeOverlay of the
# TensorFusion chart and never scales out for TF worker pods. It is set through
# the FEATURE_GATES environment variable only: the chart's settings.featureGates
# also becomes the --feature-gates argument, which the provider (0.2.5) parses
# with its own gate list and dies on ("unknown feature gate NodeOverlay").
helm upgrade --install karpenter "$chart" --namespace kube-system --skip-crds \
  --wait --timeout 5m -f - >/dev/null <<YAML
replicaCount: 1
controller:
${image_values}
  nodeSelector:
    ${ROLE_LABEL}: system
  resources:
    requests: {cpu: 200m, memory: 512Mi}
    limits: {cpu: "2", memory: 1Gi}
  env:
    - name: FEATURE_GATES
      value: "NodeOverlay=true"
    - name: ALIBABA_CLOUD_REGION
      value: ${ACK_REGION}
    - name: ALIBABA_CLOUD_ROLE_ARN
      value: ${role_arn}
    - name: ALIBABA_CLOUD_OIDC_PROVIDER_ARN
      value: ${oidc_arn}
    - name: ALIBABA_CLOUD_OIDC_TOKEN_FILE
      value: /var/run/secrets/ack.alibabacloud.com/rrsa-tokens/token
settings:
  clusterID: ${cid}
  clusterName: ${ACK_NAME}
  clusterEndpoint: ${endpoint}
  region: ${ACK_REGION}
YAML

log "applying ECSNodeClasses and NodePools"
nodeclass() { # nodeclass <name> <disk GiB>
  cat <<YAML
---
apiVersion: karpenter.alibabacloud.com/v1alpha1
kind: ECSNodeClass
metadata:
  name: $1
spec:
  clusterID: ${cid}
  vSwitchSelectorTerms:
    - tags:
        karpenter.sh/discovery: ${ACK_NAME}
  securityGroupSelectorTerms:
    - id: ${sg}
  imageSelectorTerms:
    - id: ${image}
  role: ${worker_role}
  systemDisk:
    category: cloud_essd
    size: $2
    performanceLevel: PL0
  tags:
    Project: ${ACK_NAME}
YAML
}
nodepool() { # nodepool <name> <nodeclass> <weight> <instance json> <limits> <policy> <after> <labels>
  cat <<YAML
---
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: $1
spec:
  weight: $3
  template:
    metadata:
      labels: $8
    spec:
      nodeClassRef:
        group: karpenter.alibabacloud.com
        kind: ECSNodeClass
        name: $2
      expireAfter: Never
      requirements:
        - key: node.kubernetes.io/instance-type
          operator: In
          values: $4
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand"]
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: kubernetes.io/os
          operator: In
          values: ["linux"]
  limits: $5
  disruption:
    consolidationPolicy: $6
    consolidateAfter: $7
YAML
}
# shellcheck disable=SC2086  # the instance type lists are word-split on purpose
{
  nodeclass "$GPU_NODECLASS" "$ACK_GPU_DISK_GB"
  nodeclass "$CPU_NODECLASS" 40
  # weight: a pod that could go anywhere lands on the general-purpose pool, never on a GPU node.
  nodepool "$CPU_NODEPOOL" "$CPU_NODECLASS" 100 "$(json_list $ACK_CPU_INSTANCE_TYPES)" "{cpu: ${ACK_CPU_LIMIT}}" \
    WhenEmptyOrUnderutilized 30s "{${ROLE_LABEL}: cpu}"
  # GPU nodes carry the labels the chart (initialGpuNodeLabelSelector) and the default GPUPool select on.
  nodepool "$GPU_NODEPOOL" "$GPU_NODECLASS" 10 "$(json_list $ACK_GPU_INSTANCE_TYPES)" "{nvidia.com/gpu: ${ACK_GPU_NODE_LIMIT}}" \
    WhenEmpty "$ACK_GPU_CONSOLIDATE" "{${ROLE_LABEL}: gpu, nvidia.com/gpu.present: \"true\", ${SHARED_POOL_LABEL}: \"true\"}"
} | kubectl apply -f - >/dev/null

# ACK runs two CoreDNS replicas with anti-affinity; with one system node the
# second one keeps a Karpenter CPU node alive for good (KRP-C01 needs that
# NodePool empty). One replica is enough for a test cluster.
kubectl -n kube-system scale deploy coredns --replicas=1 >/dev/null

for nc in "$GPU_NODECLASS" "$CPU_NODECLASS"; do
  wait_for 180 "ECSNodeClass/${nc} Ready" kubectl wait ecsnodeclass "$nc" --for=condition=Ready --timeout=10s
done
ledger install-karpenter "seconds=$(($(date +%s) - start))"
log "stage 2 done"
kubectl get nodepool,ecsnodeclass || true
