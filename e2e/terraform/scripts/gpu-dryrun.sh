#!/usr/bin/env bash
# Proves as much of the GPU launch path as AWS allows without GPU quota:
#   - the NVIDIA AL2023 EKS AMI for the cluster's Kubernetes version resolves;
#   - every GPU instance type is offered in the stack's AZs;
#   - `ec2 run-instances --dry-run` (AMI + subnet + security group + Karpenter's
#     instance profile + the same tags and volume Karpenter uses) is accepted for
#     on-demand and spot. A dry run validates parameters and IAM only; it does
#     NOT evaluate quotas, so the quota answer comes from the real NodeClaim
#     attempt of the suite (case DRY-KRP-01) and from `make quotas`.
#
# Exit code 0 = every combination would launch (quota permitting).

source "$(dirname "$0")/lib.sh"
need aws jq

types=$(echo "$TF_E2E_GPU_INSTANCE_JSON" | jq -r '.[]')
k8s=$(aws eks describe-cluster --name "$TF_E2E_CLUSTER" --query cluster.version --output text)
vpc=$(aws eks describe-cluster --name "$TF_E2E_CLUSTER" --query cluster.resourcesVpcConfig.vpcId --output text)
ami=$(aws ssm get-parameter --name "/aws/service/eks/optimized-ami/${k8s}/amazon-linux-2023/x86_64/nvidia/recommended/image_id" --query Parameter.Value --output text)
ami_name=$(aws ec2 describe-images --image-ids "$ami" --query 'Images[0].Name' --output text)
log "EKS ${k8s} NVIDIA AMI: ${ami} (${ami_name})"

subnets=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=${vpc}" "Name=tag:karpenter.sh/discovery,Values=${TF_E2E_CLUSTER}" \
  --query 'Subnets[].[SubnetId,AvailabilityZone]' --output text)
rc=0
printf '%-14s %-12s %-12s %-10s %s\n' TYPE AZ SUBNET CAPACITY RESULT
for t in $types; do
  offered=$(aws ec2 describe-instance-type-offerings --location-type availability-zone \
    --filters "Name=instance-type,Values=${t}" --query 'InstanceTypeOfferings[].Location' --output text)
  while read -r subnet az; do
    [ -z "$subnet" ] && continue
    if ! echo "$offered" | grep -qw "$az"; then
      printf '%-14s %-12s %-12s %-10s %s\n' "$t" "$az" "${subnet:0:12}" - "not offered in this AZ (Karpenter uses the other subnet)"
      continue
    fi
    for cap in on-demand spot; do
      market=()
      [ "$cap" = spot ] && market=(--instance-market-options 'MarketType=spot')
      out=$(aws ec2 run-instances --dry-run --image-id "$ami" --instance-type "$t" --subnet-id "$subnet" \
        --security-group-ids "$TF_E2E_CLUSTER_SG_ID" --iam-instance-profile "Name=${TF_E2E_KARPENTER_PROFILE}" \
        --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${TF_E2E_GPU_DISK_GB},VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=${TF_E2E_CLUSTER}}]" \
        --count 1 ${market[@]+"${market[@]}"} 2>&1 || true)
      if echo "$out" | grep -q DryRunOperation; then
        result="OK (would launch)"
      else
        result="FAIL: $(echo "$out" | tr '\n' ' ' | sed -E 's/[0-9]{12}/<account>/g' | cut -c1-160)"
        rc=1
      fi
      printf '%-14s %-12s %-12s %-10s %s\n' "$t" "$az" "${subnet:0:12}" "$cap" "$result"
    done
  done <<<"$subnets"
done

echo
log "quotas (vCPU):"
for q in L-DB2E81BA L-3819A6DF; do
  aws service-quotas get-service-quota --service-code ec2 --quota-code "$q" --query '[Quota.QuotaCode,Quota.Value,Quota.QuotaName]' --output text
done
exit $rc
