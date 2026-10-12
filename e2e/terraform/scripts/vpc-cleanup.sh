#!/usr/bin/env bash
# Destroy hook #2 (terraform_data.vpc_cleanup). Runs AFTER node groups and the
# EKS cluster are gone and BEFORE subnets / IGW / VPC are deleted. The VPC is
# dedicated to this stack, so everything still inside it is a leftover.
# Also callable by hand: scripts/sweep.sh --delete runs it for a tagged VPC.

source "$(dirname "$0")/lib.sh"
need aws

VPC="${TF_E2E_VPC_ID:?TF_E2E_VPC_ID is required}"
CLUSTER="$TF_E2E_CLUSTER"

# 1. Load balancers created by Service type=LoadBalancer.
for arn in $(aws elbv2 describe-load-balancers \
  --query "LoadBalancers[?VpcId=='${VPC}'].LoadBalancerArn" --output text 2>/dev/null || true); do
  [ "$arn" = "None" ] && continue
  log "deleting load balancer $arn"
  aws elbv2 delete-load-balancer --load-balancer-arn "$arn" || true
done
for name in $(aws elb describe-load-balancers \
  --query "LoadBalancerDescriptions[?VPCId=='${VPC}'].LoadBalancerName" --output text 2>/dev/null || true); do
  [ "$name" = "None" ] && continue
  log "deleting classic load balancer $name"
  aws elb delete-load-balancer --load-balancer-name "$name" || true
done
for arn in $(aws elbv2 describe-target-groups \
  --query "TargetGroups[?VpcId=='${VPC}'].TargetGroupArn" --output text 2>/dev/null || true); do
  [ "$arn" = "None" ] && continue
  aws elbv2 delete-target-group --target-group-arn "$arn" >/dev/null 2>&1 || true
done

# 2. Network interfaces (leaked VPC-CNI ENIs, ELB ENIs that are still detaching).
for attempt in $(seq 1 18); do
  all=$(aws ec2 describe-network-interfaces --filters "Name=vpc-id,Values=${VPC}" \
    --query 'NetworkInterfaces[].NetworkInterfaceId' --output text 2>/dev/null || true)
  if [ -z "$all" ] || [ "$all" = "None" ]; then
    break
  fi
  for eni in $(aws ec2 describe-network-interfaces \
    --filters "Name=vpc-id,Values=${VPC}" "Name=status,Values=available" \
    --query 'NetworkInterfaces[].NetworkInterfaceId' --output text 2>/dev/null || true); do
    [ "$eni" = "None" ] && continue
    log "deleting leaked ENI $eni"
    aws ec2 delete-network-interface --network-interface-id "$eni" >/dev/null 2>&1 || true
  done
  log "waiting for network interfaces in $VPC to disappear (attempt $attempt/18)"
  sleep 10
done

# 3. Security groups created by Kubernetes (k8s-elb-*, ...). The EKS cluster
#    security group is removed by EKS itself together with the cluster.
sgs=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=${VPC}" \
  --query "SecurityGroups[?GroupName!='default'].GroupId" --output text 2>/dev/null || true)
for sg in $sgs; do
  [ "$sg" = "None" ] && continue
  # Drop rules first so groups referencing each other can be deleted.
  for dir in ingress egress; do
    rules=$(aws ec2 describe-security-group-rules --filters "Name=group-id,Values=${sg}" \
      --query "SecurityGroupRules[?IsEgress==\`$([ "$dir" = egress ] && echo true || echo false)\`].SecurityGroupRuleId" \
      --output text 2>/dev/null || true)
    if [ -n "$rules" ] && [ "$rules" != "None" ]; then
      # shellcheck disable=SC2086
      aws ec2 "revoke-security-group-${dir}" --group-id "$sg" --security-group-rule-ids $rules >/dev/null 2>&1 || true
    fi
  done
done
for sg in $sgs; do
  [ "$sg" = "None" ] && continue
  log "deleting leftover security group $sg"
  aws ec2 delete-security-group --group-id "$sg" >/dev/null 2>&1 || true
done

# 4. EBS volumes provisioned by the CSI driver that outlived their PVC cleanup.
for filter in "Name=tag:kubernetes.io/cluster/${CLUSTER},Values=owned" "Name=tag:Project,Values=${CLUSTER}"; do
  for vol in $(aws ec2 describe-volumes --filters "$filter" "Name=status,Values=available" \
    --query 'Volumes[].VolumeId' --output text 2>/dev/null || true); do
    [ "$vol" = "None" ] && continue
    log "deleting leftover EBS volume $vol"
    aws ec2 delete-volume --volume-id "$vol" >/dev/null 2>&1 || true
  done
done

log "vpc cleanup finished"
exit 0
