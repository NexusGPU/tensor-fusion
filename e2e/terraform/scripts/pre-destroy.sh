#!/usr/bin/env bash
# Destroy hook #1 (terraform_data.cluster_cleanup). Runs BEFORE Terraform touches
# the node groups / cluster / Karpenter IAM. Goal: nothing that an in-cluster
# controller created in AWS survives the cluster.
#
#   phase A (best effort, needs a working Kubernetes API):
#     stop the controllers that create AWS resources, let the cloud controller
#     delete load balancers.
#   phase B (authoritative, EC2 API only):
#     terminate every Karpenter-launched instance (NodePool nodes and NodeClaims
#     TensorFusion created itself), delete Karpenter's launch templates.
#
# Never fails the destroy: every step is best effort and the tag sweep
# (scripts/sweep.sh) is the final check.

source "$(dirname "$0")/lib.sh"
need aws

CLUSTER="$TF_E2E_CLUSTER"

# ---------------------------------------------------------------- phase A ----
k8s_ok=false
if command -v kubectl >/dev/null 2>&1 &&
  aws eks describe-cluster --name "$CLUSTER" >/dev/null 2>&1 &&
  setup_kubeconfig 2>/dev/null &&
  kubectl --request-timeout=15s get ns kube-system >/dev/null 2>&1; then
  k8s_ok=true
fi

if $k8s_ok; then
  log "phase A: quiescing in-cluster controllers"
  # Nobody may ask for new nodes any more: TF controller first (it creates
  # NodeClaims itself), then Karpenter.
  kubectl -n tensor-fusion-sys scale deployment --all --replicas=0 --timeout=30s >/dev/null 2>&1 || true
  kubectl -n kube-system scale deployment karpenter --replicas=0 --timeout=30s >/dev/null 2>&1 || true

  # Service type=LoadBalancer -> ELB/NLB + security groups.
  if command -v jq >/dev/null 2>&1; then
    kubectl get svc -A -o json 2>/dev/null |
      jq -r '.items[] | select(.spec.type=="LoadBalancer") | "\(.metadata.namespace) \(.metadata.name)"' |
      while read -r ns name; do
        log "deleting LoadBalancer service $ns/$name"
        kubectl -n "$ns" delete svc "$name" --timeout=120s || true
      done
  fi
  kubectl delete ingress -A --all --timeout=60s >/dev/null 2>&1 || true
else
  log "phase A skipped: Kubernetes API not reachable (cluster gone or no kubectl)"
fi

# ---------------------------------------------------------------- phase B ----
log "phase B: terminating Karpenter-launched instances of cluster $CLUSTER"
for pass in 1 2 3; do
  ids=""
  for f in "Name=tag:eks:eks-cluster-name,Values=${CLUSTER}" "Name=tag:kubernetes.io/cluster/${CLUSTER},Values=owned"; do
    ids="$ids $(aws ec2 describe-instances \
      --filters "$f" "Name=tag-key,Values=karpenter.sh/nodeclaim" \
      "Name=instance-state-name,Values=pending,running,stopping,stopped" \
      --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null | sed 's/None//' || true)"
  done
  ids=$(echo "$ids" | tr -s ' \t' '\n' | sort -u | tr '\n' ' ' | sed 's/^ *//; s/ *$//')
  if [ -z "$ids" ]; then
    log "pass $pass: no Karpenter instances left"
    break
  fi
  log "pass $pass: terminating $ids"
  # shellcheck disable=SC2086
  aws ec2 terminate-instances --instance-ids $ids >/dev/null || true
  # shellcheck disable=SC2086
  aws ec2 wait instance-terminated --instance-ids $ids || true
done

# Karpenter deletes its launch templates itself, unless it dies between creating
# one and launching from it (seen after the dead-man switch killed its node).
# Karpenter >= 1.x tags them eks:eks-cluster-name / kubernetes.io/cluster/<name>
# (plus the EC2NodeClass tags); the old karpenter.k8s.aws/cluster tag is kept for
# older versions. The node groups' own launch templates are Terraform's and are
# named <stack>-<group>, so only karpenter.k8s.aws/* names are touched here.
lts=""
for f in "Name=tag:eks:eks-cluster-name,Values=${CLUSTER}" "Name=tag:kubernetes.io/cluster/${CLUSTER},Values=owned" "Name=tag:karpenter.k8s.aws/cluster,Values=${CLUSTER}"; do
  lts="$lts $(aws ec2 describe-launch-templates --filters "$f" \
    --query 'LaunchTemplates[?starts_with(LaunchTemplateName, `karpenter.k8s.aws/`)].LaunchTemplateId' --output text 2>/dev/null | sed 's/None//' || true)"
done
for lt in $(echo "$lts" | tr -s ' \t' '\n' | sort -u); do
  log "deleting Karpenter launch template $lt"
  aws ec2 delete-launch-template --launch-template-id "$lt" >/dev/null || true
done

log "pre-destroy cleanup finished"
exit 0
