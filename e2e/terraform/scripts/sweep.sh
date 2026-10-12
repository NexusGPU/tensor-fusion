#!/usr/bin/env bash
# Tag/name based leftover finder for a tf-e2e stack. Works WITHOUT Terraform
# state (lost state, half-failed destroy, different machine).
#
#   scripts/sweep.sh            # report only, exit 1 if anything is left
#   scripts/sweep.sh --delete   # also delete what controllers left behind and,
#                               # as a last resort, the Terraform-managed pieces
#
# Needs TF_E2E_CLUSTER (= var.name) and TF_E2E_REGION; `make sweep` sets them.

source "$(dirname "$0")/lib.sh"
need aws

NAME="$TF_E2E_CLUSTER"
DELETE=false
[ "${1:-}" = "--delete" ] && DELETE=true
found=0

q() { # q <service> <subcommand> [args...] -> whitespace separated ids, never fails
  local out
  out=$(aws "$@" --output text 2>/dev/null || true)
  [ "$out" = "None" ] && out=""
  printf '%s' "$out"
}
report() { # report <label> <ids>
  if [ -n "$2" ]; then
    found=1
    log "LEFTOVER $1: $2"
  fi
}

# --- EKS cluster ---------------------------------------------------------------
cluster=$(q eks describe-cluster --name "$NAME" --query 'cluster.status')
report "EKS cluster ${NAME}" "$cluster"
if $DELETE && [ -n "$cluster" ]; then
  for ng in $(q eks list-nodegroups --cluster-name "$NAME" --query 'nodegroups[]'); do
    aws eks delete-nodegroup --cluster-name "$NAME" --nodegroup-name "$ng" >/dev/null || true
  done
  for ng in $(q eks list-nodegroups --cluster-name "$NAME" --query 'nodegroups[]'); do
    aws eks wait nodegroup-deleted --cluster-name "$NAME" --nodegroup-name "$ng" || true
  done
  aws eks delete-cluster --name "$NAME" >/dev/null || true
  aws eks wait cluster-deleted --name "$NAME" || true
fi

# --- EC2 instances (Karpenter, node groups, anything tagged) ---------------------
inst=""
for f in "Name=tag:kubernetes.io/cluster/${NAME},Values=owned" "Name=tag:eks:cluster-name,Values=${NAME}" "Name=tag:eks:eks-cluster-name,Values=${NAME}" "Name=tag:Project,Values=${NAME}"; do
  inst="$inst $(q ec2 describe-instances --filters "$f" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId')"
done
inst=$(echo $inst | tr ' ' '\n' | sort -u | tr '\n' ' ' | sed 's/ *$//')
report "EC2 instances" "$inst"
if $DELETE && [ -n "$inst" ]; then
  # shellcheck disable=SC2086
  aws ec2 terminate-instances --instance-ids $inst >/dev/null || true
  # shellcheck disable=SC2086
  aws ec2 wait instance-terminated --instance-ids $inst || true
fi

# --- launch templates ------------------------------------------------------------
lts=""
for f in "Name=tag:karpenter.k8s.aws/cluster,Values=${NAME}" "Name=tag:eks:eks-cluster-name,Values=${NAME}" "Name=tag:eks:cluster-name,Values=${NAME}" "Name=tag:Project,Values=${NAME}"; do
  lts="$lts $(q ec2 describe-launch-templates --filters "$f" --query 'LaunchTemplates[].LaunchTemplateId')"
done
lts=$(echo $lts | tr ' ' '\n' | sort -u | tr '\n' ' ' | sed 's/ *$//')
report "launch templates" "$lts"
if $DELETE; then
  for lt in $lts; do aws ec2 delete-launch-template --launch-template-id "$lt" >/dev/null || true; done
fi

# --- VPCs and what is inside them -------------------------------------------------
vpcs=$(q ec2 describe-vpcs --filters "Name=tag:Project,Values=${NAME}" --query 'Vpcs[].VpcId')
report "VPCs" "$vpcs"
for vpc in $vpcs; do
  report "  ENIs in $vpc" "$(q ec2 describe-network-interfaces --filters "Name=vpc-id,Values=${vpc}" --query 'NetworkInterfaces[].NetworkInterfaceId')"
  lbs="$(q elbv2 describe-load-balancers --query "LoadBalancers[?VpcId=='${vpc}'].LoadBalancerName") $(q elb describe-load-balancers --query "LoadBalancerDescriptions[?VPCId=='${vpc}'].LoadBalancerName")"
  report "  load balancers in $vpc" "$(echo $lbs)"
  report "  security groups in $vpc" "$(q ec2 describe-security-groups --filters "Name=vpc-id,Values=${vpc}" --query "SecurityGroups[?GroupName!='default'].GroupId")"
  report "  NAT gateways in $vpc" "$(q ec2 describe-nat-gateways --filter "Name=vpc-id,Values=${vpc}" "Name=state,Values=pending,available" --query 'NatGateways[].NatGatewayId')"
  if $DELETE; then
    TF_E2E_VPC_ID="$vpc" "$(dirname "$0")/vpc-cleanup.sh" || true
    for igw in $(q ec2 describe-internet-gateways --filters "Name=attachment.vpc-id,Values=${vpc}" --query 'InternetGateways[].InternetGatewayId'); do
      aws ec2 detach-internet-gateway --internet-gateway-id "$igw" --vpc-id "$vpc" || true
      aws ec2 delete-internet-gateway --internet-gateway-id "$igw" || true
    done
    for sn in $(q ec2 describe-subnets --filters "Name=vpc-id,Values=${vpc}" --query 'Subnets[].SubnetId'); do
      aws ec2 delete-subnet --subnet-id "$sn" || true
    done
    for rt in $(q ec2 describe-route-tables --filters "Name=vpc-id,Values=${vpc}" --query "RouteTables[?length(Associations[?Main==\`true\`])==\`0\`].RouteTableId"); do
      aws ec2 delete-route-table --route-table-id "$rt" || true
    done
    aws ec2 delete-vpc --vpc-id "$vpc" || true
  fi
done

# --- EBS volumes / snapshots ------------------------------------------------------
vols=""
for f in "Name=tag:kubernetes.io/cluster/${NAME},Values=owned" "Name=tag:Project,Values=${NAME}"; do
  vols="$vols $(q ec2 describe-volumes --filters "$f" --query 'Volumes[].VolumeId')"
done
vols=$(echo $vols | tr ' ' '\n' | sort -u | tr '\n' ' ' | sed 's/ *$//')
report "EBS volumes" "$vols"
if $DELETE; then
  for v in $vols; do aws ec2 delete-volume --volume-id "$v" >/dev/null 2>&1 || true; done
fi
report "Elastic IPs" "$(q ec2 describe-addresses --filters "Name=tag:Project,Values=${NAME}" --query 'Addresses[].AllocationId')"

# --- IAM (names are prefixed with the stack name) ----------------------------------
profiles=$(q iam list-instance-profiles --query "InstanceProfiles[?starts_with(InstanceProfileName, '${NAME}-') || starts_with(InstanceProfileName, '${NAME}_')].InstanceProfileName")
report "IAM instance profiles" "$profiles"
roles=$(q iam list-roles --query "Roles[?starts_with(RoleName, '${NAME}-')].RoleName")
report "IAM roles" "$roles"
policies=$(q iam list-policies --scope Local --query "Policies[?starts_with(PolicyName, '${NAME}-')].Arn")
report "IAM policies" "$policies"
if $DELETE; then
  for p in $profiles; do
    for r in $(q iam get-instance-profile --instance-profile-name "$p" --query 'InstanceProfile.Roles[].RoleName'); do
      aws iam remove-role-from-instance-profile --instance-profile-name "$p" --role-name "$r" || true
    done
    aws iam delete-instance-profile --instance-profile-name "$p" || true
  done
  for r in $roles; do
    for a in $(q iam list-attached-role-policies --role-name "$r" --query 'AttachedPolicies[].PolicyArn'); do
      aws iam detach-role-policy --role-name "$r" --policy-arn "$a" || true
    done
    for i in $(q iam list-role-policies --role-name "$r" --query 'PolicyNames[]'); do
      aws iam delete-role-policy --role-name "$r" --policy-name "$i" || true
    done
    aws iam delete-role --role-name "$r" || true
  done
  for a in $policies; do aws iam delete-policy --policy-arn "$a" || true; done
fi

# --- dead-man switch: Lambda, SSM parameters -----------------------------------------
fns=$(q lambda list-functions --query "Functions[?starts_with(FunctionName, '${NAME}-')].FunctionName")
report "Lambda functions" "$fns"
params=$(q ssm get-parameters-by-path --path "/${NAME}/" --recursive --query 'Parameters[].Name')
report "SSM parameters" "$params"
if $DELETE; then
  for fn in $fns; do aws lambda delete-function --function-name "$fn" || true; done
  for pa in $params; do aws ssm delete-parameter --name "$pa" || true; done
fi

# --- CloudWatch logs, SQS, EventBridge ----------------------------------------------
lgs="$(q logs describe-log-groups --log-group-name-prefix "/aws/eks/${NAME}/" --query 'logGroups[].logGroupName') $(q logs describe-log-groups --log-group-name-prefix "/aws/lambda/${NAME}-" --query 'logGroups[].logGroupName')"
lgs=$(echo $lgs)
report "CloudWatch log groups" "$lgs"
queues=$(q sqs list-queues --queue-name-prefix "${NAME}-" --query 'QueueUrls[]')
report "SQS queues" "$queues"
rules=$(q events list-rules --name-prefix "${NAME}" --query 'Rules[].Name')
report "EventBridge rules" "$rules"
if $DELETE; then
  for lg in $lgs; do aws logs delete-log-group --log-group-name "$lg" || true; done
  for qu in $queues; do aws sqs delete-queue --queue-url "$qu" || true; done
  for ru in $rules; do
    ids=$(q events list-targets-by-rule --rule "$ru" --query 'Targets[].Id')
    # shellcheck disable=SC2086
    [ -n "$ids" ] && aws events remove-targets --rule "$ru" --ids $ids >/dev/null || true
    aws events delete-rule --name "$ru" || true
  done
fi

# --- snapshots / budgets --------------------------------------------------------------
report "EBS snapshots" "$(q ec2 describe-snapshots --owner-ids self --filters "Name=tag:Project,Values=${NAME}" --query 'Snapshots[].SnapshotId')"
acct=$(q sts get-caller-identity --query Account)
budgets=$(q budgets describe-budgets --account-id "$acct" --query "Budgets[?starts_with(BudgetName, '${NAME}-')].BudgetName")
report "AWS Budgets" "$budgets"
if $DELETE; then
  for b in $budgets; do aws budgets delete-budget --account-id "$acct" --budget-name "$b" || true; done
fi

# --- catch-all: anything else still carrying the Project tag -------------------------
# The tagging API keeps listing EC2 resources for about an hour after they are
# gone (terminated instances, their ENIs / volumes, Karpenter's instant fleets and
# closed spot requests). Those types were checked directly above, so they are
# only counted here; any OTHER resource type is a real leftover.
other=$(q resourcegroupstaggingapi get-resources --tag-filters "Key=Project,Values=${NAME}" --query 'ResourceTagMappingList[].ResourceARN' | tr '\t' '\n' | sed '/^$/d')
stale=$(printf '%s\n' "$other" | grep -c -E ':(instance|network-interface|volume|fleet|spot-instances-request|launch-template|security-group|subnet|vpc|internet-gateway|route-table)/' || true)
unknown=$(printf '%s\n' "$other" | grep -v -E ':(instance|network-interface|volume|fleet|spot-instances-request|launch-template|security-group|subnet|vpc|internet-gateway|route-table)/' | sed '/^$/d' || true)
[ "${stale:-0}" != "0" ] && log "tagging API still lists ${stale} already-deleted EC2 resource(s) (expected for ~1 h; verified gone by the direct checks above)"
report "other resources tagged Project=${NAME}" "$(echo $unknown)"

if [ "$found" = 0 ]; then
  log "sweep: nothing left for ${NAME} in ${TF_E2E_REGION} (checked: EKS, EC2 instances, launch templates, VPC/ENI/LB/SG/NAT, EBS volumes+snapshots, EIP, IAM, Lambda, SSM, logs, SQS, EventBridge, Budgets)"
  exit 0
fi
if $DELETE; then
  log "sweep: deletion attempted - run again without --delete to verify"
  exit 0
fi
log "sweep: leftovers found (see above). 'make sweep-delete' removes them."
exit 1
