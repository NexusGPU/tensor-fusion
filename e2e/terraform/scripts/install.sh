#!/usr/bin/env bash
# Stage 2: cluster add-ons that TensorFusion depends on.
#   Karpenter -> EC2NodeClasses -> NodePools (gpu / cpu / mock) -> NVIDIA device plugin
# Idempotent. Run by `terraform apply` (terraform_data.install) or `make install`.
# TensorFusion itself is stage 3: scripts/install-tf.sh (`make tf-install`).

source "$(dirname "$0")/lib.sh"
need aws kubectl helm

setup_kubeconfig

log "waiting for system node(s) to be Ready"
wait_for 300 "system node Ready" \
  kubectl wait node -l "${TF_E2E_ROLE_LABEL_KEY}=system" --for=condition=Ready --timeout=10s

# --- EC2 Spot service-linked role ---------------------------------------------
# A fresh account has no AWSServiceRoleForEC2Spot and Karpenter's first spot
# CreateFleet then fails with AuthFailure.ServiceLinkedRoleCreationNotPermitted.
# Account-global, free, shared with anything else that uses spot: created here
# if missing and deliberately NOT removed by `make down`.
if ! aws iam get-role --role-name AWSServiceRoleForEC2Spot >/dev/null 2>&1; then
  log "creating the EC2 Spot service-linked role (one-time, account-wide)"
  aws iam create-service-linked-role --aws-service-name spot.amazonaws.com >/dev/null 2>&1 ||
    log "WARNING: could not create AWSServiceRoleForEC2Spot; spot launches will fall back to on-demand"
fi

# --- Karpenter ----------------------------------------------------------------
# featureGates.nodeOverlay: the TensorFusion chart creates a karpenter.sh/v1alpha1
# NodeOverlay (tensor-fusion.ai/index_* capacity) when that CRD exists; without the
# gate Karpenter ignores it and never scales out for TF worker pods.
log "installing Karpenter ${TF_E2E_KARPENTER_VERSION}"
helm upgrade --install karpenter oci://public.ecr.aws/karpenter/karpenter \
  --version "${TF_E2E_KARPENTER_VERSION}" \
  --namespace kube-system \
  --wait --timeout 5m \
  -f - <<YAML
replicas: 1
nodeSelector:
  ${TF_E2E_ROLE_LABEL_KEY}: system
settings:
  clusterName: ${TF_E2E_CLUSTER}
  interruptionQueue: "${TF_E2E_KARPENTER_QUEUE}"
  featureGates:
    nodeOverlay: true
controller:
  resources:
    requests:
      cpu: 200m
      memory: 512Mi
    limits:
      memory: 1Gi
YAML

log "applying EC2NodeClasses and NodePools"
# JSON is valid YAML flow syntax, so the *_JSON variables are inlined as-is.
nodeclass() { # nodeclass <name> <disk GiB> <throughput MiB/s>
  cat <<YAML
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: $1
spec:
  # The al2023 alias resolves to the NVIDIA (accelerated) AL2023 AMI for GPU
  # instance types and to the standard one otherwise.
  amiFamily: AL2023
  amiSelectorTerms:
    - alias: ${TF_E2E_AMI_ALIAS}
  # Terraform-owned instance profile (see karpenter.tf for why not spec.role).
  instanceProfile: ${TF_E2E_KARPENTER_PROFILE}
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: ${TF_E2E_CLUSTER}
  securityGroupSelectorTerms:
    - id: ${TF_E2E_CLUSTER_SG_ID}
  associatePublicIPAddress: true
  metadataOptions:
    httpEndpoint: enabled
    httpTokens: required
    httpPutResponseHopLimit: 2
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: $2Gi
        volumeType: gp3
        iops: 3000
        throughput: $3
        encrypted: true
        deleteOnTermination: true
  tags: ${TF_E2E_TAGS_JSON}
YAML
}

nodepool() { # nodepool <name> <nodeclass> <weight> <instance json> <capacity json> <limits yaml> <policy> <after> <labels yaml>
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
      labels: $9
    spec:
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: $2
      expireAfter: Never
      requirements:
        - key: node.kubernetes.io/instance-type
          operator: In
          values: $4
        - key: karpenter.sh/capacity-type
          operator: In
          values: $5
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: kubernetes.io/os
          operator: In
          values: ["linux"]
  # Cost guard for everything this NodePool launches.
  limits: $6
  disruption:
    consolidationPolicy: $7
    consolidateAfter: $8
YAML
}

{
  # GPU root volumes get 300 MiB/s (+$0.01/h per node): unpacking multi-GB CUDA
  # images is throughput-bound at the gp3 default of 125 MiB/s.
  nodeclass "${TF_E2E_GPU_NODECLASS}" "${TF_E2E_GPU_DISK_GB}" 300
  nodeclass "${TF_E2E_CPU_NODECLASS}" 30 125

  # weight: Karpenter tries NodePools in descending weight, so a pod that could
  # go anywhere lands on the general-purpose pool, never on a GPU node.
  nodepool "${TF_E2E_CPU_NODEPOOL}" "${TF_E2E_CPU_NODECLASS}" 100 \
    "${TF_E2E_CPU_INSTANCE_JSON}" "${TF_E2E_CPU_CAPACITY_JSON}" \
    "{cpu: ${TF_E2E_CPU_LIMIT}}" WhenEmptyOrUnderutilized 30s \
    "{${TF_E2E_ROLE_LABEL_KEY}: cpu}"

  # CPU nodes that the rehearsal mode turns into TensorFusion nodes with mock
  # GPUs (make mock-install). Untainted, like real TF GPU nodes; the weight keeps
  # generic pods on the general-purpose pool. They carry nvidia.com/gpu.present
  # because the operator only reconciles nodes matching the chart's
  # initialGpuNodeLabelSelector; the mock-gpu label is what the mock GPUPool
  # selects, so they never join the real pool.
  nodepool "${TF_E2E_MOCK_NODEPOOL}" "${TF_E2E_CPU_NODECLASS}" 50 \
    "${TF_E2E_MOCK_INSTANCE_JSON}" "${TF_E2E_MOCK_CAPACITY_JSON}" \
    "{cpu: ${TF_E2E_MOCK_CPU_LIMIT}}" WhenEmpty "${TF_E2E_GPU_CONSOLIDATE}" \
    "{${TF_E2E_ROLE_LABEL_KEY}: mock, ${TF_E2E_MOCK_LABEL_KEY}: \"true\", ${TF_E2E_GPU_PRESENT_LABEL_KEY}: \"true\"}"

  nodepool "${TF_E2E_GPU_NODEPOOL}" "${TF_E2E_GPU_NODECLASS}" 10 \
    "${TF_E2E_GPU_INSTANCE_JSON}" "${TF_E2E_GPU_CAPACITY_JSON}" \
    "{nvidia.com/gpu: ${TF_E2E_GPU_NODE_LIMIT}}" WhenEmpty "${TF_E2E_GPU_CONSOLIDATE}" \
    "{${TF_E2E_ROLE_LABEL_KEY}: gpu, ${TF_E2E_GPU_PRESENT_LABEL_KEY}: \"true\", ${TF_E2E_SHARED_POOL_LABEL_KEY}: \"true\"}"
} | kubectl apply -f -

# Keep pending TensorFusion pods from pulling up non-TF nodes (see lib.sh). The
# mock pool is excluded too until `make mock-install` turns its nodes into TF nodes.
if kubectl get gpupool "tensor-fusion-mock" >/dev/null 2>&1; then
  no_index_overlay "${TF_E2E_CPU_NODEPOOL}"
else
  no_index_overlay "${TF_E2E_CPU_NODEPOOL}" "${TF_E2E_MOCK_NODEPOOL}"
fi

for nc in "${TF_E2E_GPU_NODECLASS}" "${TF_E2E_CPU_NODECLASS}"; do
  wait_for 120 "EC2NodeClass/${nc} Ready" \
    kubectl wait ec2nodeclass "${nc}" --for=condition=Ready --timeout=10s
done

# --- NVIDIA device plugin -----------------------------------------------------
# Not required by TensorFusion itself. It makes GPU nodes advertise nvidia.com/gpu,
# which Karpenter waits for before a NodeClaim that requests nvidia.com/gpu
# (what TF creates in provisioningMode=Karpenter) becomes Initialized.
# It runs zero pods while the cluster has no real GPU node.
if [ "${TF_E2E_NVDP_ENABLED}" = "true" ]; then
  log "installing NVIDIA device plugin ${TF_E2E_NVDP_VERSION}"
  helm upgrade --install nvdp nvidia-device-plugin \
    --repo https://nvidia.github.io/k8s-device-plugin \
    --version "${TF_E2E_NVDP_VERSION}" \
    --namespace nvidia-device-plugin --create-namespace -f - >/dev/null <<YAML
# Real GPU nodes only: mock-GPU nodes also carry the gpu.present label.
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: ${TF_E2E_GPU_PRESENT_LABEL_KEY}
              operator: In
              values: ["true"]
            - key: ${TF_E2E_MOCK_LABEL_KEY}
              operator: DoesNotExist
YAML
fi

log "stage 2 done. export KUBECONFIG=${TF_E2E_KUBECONFIG}"
kubectl get nodes -L node.kubernetes.io/instance-type,karpenter.sh/capacity-type,"${TF_E2E_ROLE_LABEL_KEY}" || true
kubectl get nodepool,ec2nodeclass || true
