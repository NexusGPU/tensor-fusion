#!/usr/bin/env bash
# ACK tier, stage "down": remove everything up.sh / install.sh created, then
# print what is still there (the sweep; exit 1 when something billable is left).
#
#   1. NodePools and NodeClaims  -> Karpenter terminates its ECS instances
#   2. ECS instances and launch templates still tagged Project=${ACK_NAME} (Karpenter killed mid-launch, path B leftovers)
#   3. the ACK cluster (with its SLBs and the NAT gateway / EIP ACK created)
#   4. RAM role + policy of the Karpenter controller
#   5. vSwitches, security groups, VPC
#
# Works without the state file and without a reachable cluster: everything is
# found by name and tag. `down.sh sweep` only lists. The account-level ACK
# service roles and the console environment are not touched.

source "$(dirname "$0")/lib.sh"
need aliyun jq

instances() {
  ali ecs DescribeInstances --RegionId "$ACK_REGION" --PageSize 100 --Tag.1.Key Project --Tag.1.Value "$ACK_NAME" |
    jq -r '.Instances.Instance[] | select(.Status != "Released") | .InstanceId'
}
launch_templates() {
  ali ecs DescribeLaunchTemplates --RegionId "$ACK_REGION" --PageSize 50 |
    jq -r --arg n "$ACK_NAME" --arg c "$(cluster_id)" '.LaunchTemplateSets.LaunchTemplateSet[]? | select((.LaunchTemplateName | test($n)) or ($c != "" and (.LaunchTemplateName | test($c))) or ([.Tags.InstanceTag[]? | select(.TagKey == "Project" and .TagValue == $n)] | length > 0)) | .LaunchTemplateId'
}
vpc_id() { ali vpc DescribeVpcs --RegionId "$ACK_REGION" --VpcName "$ACK_NAME" | jq -r '.Vpcs.Vpc[0].VpcId // empty'; }

sweep() {
  local vpc left=0 out
  vpc=$(vpc_id)
  report() { # report <what> <ids>
    if [ -n "$2" ]; then left=1; log "LEFT  $1: $(echo "$2" | tr '\n' ' ')"; else log "clean $1"; fi
  }
  report "ACK cluster" "$(cluster_id)"
  report "ECS instances (Project=${ACK_NAME})" "$(instances)"
  report "launch templates" "$(launch_templates)"
  report "ECS disks (Project=${ACK_NAME})" "$(ali ecs DescribeDisks --RegionId "$ACK_REGION" --Tag.1.Key Project --Tag.1.Value "$ACK_NAME" | jq -r '.Disks.Disk[].DiskId')"
  if [ -n "$vpc" ]; then
    report "NAT gateways" "$(ali vpc DescribeNatGateways --RegionId "$ACK_REGION" --VpcId "$vpc" | jq -r '.NatGateways.NatGateway[].NatGatewayId')"
    report "load balancers" "$(ali slb DescribeLoadBalancers --RegionId "$ACK_REGION" --VpcId "$vpc" | jq -r '.LoadBalancers.LoadBalancer[].LoadBalancerId')"
    report "network interfaces" "$(ali ecs DescribeNetworkInterfaces --RegionId "$ACK_REGION" --VpcId "$vpc" | jq -r '.NetworkInterfaceSets.NetworkInterfaceSet[].NetworkInterfaceId')"
    report "security groups" "$(ali ecs DescribeSecurityGroups --RegionId "$ACK_REGION" --VpcId "$vpc" | jq -r '.SecurityGroups.SecurityGroup[].SecurityGroupId')"
    report "vSwitches" "$(ali vpc DescribeVSwitches --RegionId "$ACK_REGION" --VpcId "$vpc" | jq -r '.VSwitches.VSwitch[].VSwitchId')"
  fi
  report "VPC" "$vpc"
  out=$(ali vpc DescribeEipAddresses --RegionId "$ACK_REGION" --PageSize 100 | jq -r --arg n "$ACK_NAME" '.EipAddresses.EipAddress[] | select(.Status == "Available" or ((.Name // "") | test($n))) | .AllocationId + "(" + .Status + ")"')
  [ -z "$out" ] || log "note  EIPs of the region that are unassociated or named after this stack: $(echo "$out" | tr '\n' ' ')"
  report "RAM role ${KARPENTER_ROLE}" "$(aliyun ram GetRole --RoleName "$KARPENTER_ROLE" 2>/dev/null | jq -r '.Role.RoleName // empty')"
  report "RAM policy ${KARPENTER_ROLE}" "$(aliyun ram GetPolicy --PolicyType Custom --PolicyName "$KARPENTER_ROLE" 2>/dev/null | jq -r '.Policy.PolicyName // empty')"
  return $left
}

if [ "${1:-}" = "sweep" ]; then
  sweep
  exit $?
fi

start=$(date +%s)
cid=$(cluster_id)

# --- 1. let Karpenter terminate its own nodes --------------------------------------------
if [ -n "$cid" ] && [ -s "$KUBECONFIG" ] && kubectl get nodepool >/dev/null 2>&1; then
  need kubectl
  log "deleting GPUNodeClaims, NodePools and NodeClaims"
  kubectl delete gpunodeclaim --all --wait=false >/dev/null 2>&1 || true
  kubectl delete nodepool --all --wait=false >/dev/null 2>&1 || true
  kubectl delete nodeclaim --all --wait=false >/dev/null 2>&1 || true
  wait_for 300 "NodeClaims gone" bash -c '[ -z "$(kubectl get nodeclaim -o name 2>/dev/null)" ]' || true
fi

# --- 2. instances and launch templates the cluster deletion would not take ----------------
system_ids=""
if [ -n "$cid" ]; then
  system_ids=$(ali cs GET "/clusters/${cid}/nodes" | jq -r '.nodes[]? | .instance_id' | tr '\n' ' ')
fi
for id in $(instances); do
  case " $system_ids " in *" $id "*) continue ;; esac # node pool instances go with the cluster
  log "terminating instance ${id}"
  ali ecs DeleteInstance --InstanceId "$id" --Force true >/dev/null || true
done
for lt in $(launch_templates); do
  ali ecs DeleteLaunchTemplate --RegionId "$ACK_REGION" --LaunchTemplateId "$lt" >/dev/null || true
done

# --- 3. the cluster -----------------------------------------------------------------------
if [ -n "$cid" ]; then
  log "deleting ACK cluster ${cid}"
  ali cs DELETE "/clusters/${cid}?retain_all_resources=false" --header "Content-Type=application/json" \
    --body '{"delete_options":[{"resource_type":"SLB","delete_mode":"delete"},{"resource_type":"ALB","delete_mode":"delete"},{"resource_type":"SLS_Data","delete_mode":"delete"},{"resource_type":"SLS_ControlPlane","delete_mode":"delete"},{"resource_type":"PrivateZone","delete_mode":"delete"}]}' >/dev/null
  wait_for 1500 "cluster ${cid} deleted" bash -c "[ -z \"\$(aliyun cs GET /api/v1/clusters --name ${ACK_NAME} --region ${ACK_REGION} | jq -r '.clusters[]? | select(.state != \"deleted\") | .cluster_id')\" ]"
fi
rm -f "$KUBECONFIG"

# --- 4. RAM ---------------------------------------------------------------------------------
if aliyun ram GetRole --RoleName "$KARPENTER_ROLE" >/dev/null 2>&1; then
  log "deleting RAM role and policy ${KARPENTER_ROLE}"
  aliyun ram DetachPolicyFromRole --PolicyType Custom --PolicyName "$KARPENTER_ROLE" --RoleName "$KARPENTER_ROLE" >/dev/null 2>&1 || true
  aliyun ram DeleteRole --RoleName "$KARPENTER_ROLE" >/dev/null || true
fi
if aliyun ram GetPolicy --PolicyType Custom --PolicyName "$KARPENTER_ROLE" >/dev/null 2>&1; then
  for v in $(aliyun ram ListPolicyVersions --PolicyType Custom --PolicyName "$KARPENTER_ROLE" | jq -r '.PolicyVersions.PolicyVersion[] | select(.IsDefaultVersion | not) | .VersionId'); do
    aliyun ram DeletePolicyVersion --PolicyName "$KARPENTER_ROLE" --VersionId "$v" >/dev/null || true
  done
  aliyun ram DeletePolicy --PolicyName "$KARPENTER_ROLE" >/dev/null || true
fi

# --- 5. network -----------------------------------------------------------------------------
vpc=$(vpc_id)
if [ -n "$vpc" ]; then
  # What ACK created for snat_entry and did not remove with the cluster.
  for nat in $(ali vpc DescribeNatGateways --RegionId "$ACK_REGION" --VpcId "$vpc" | jq -r '.NatGateways.NatGateway[].NatGatewayId'); do
    eips=$(ali vpc DescribeEipAddresses --RegionId "$ACK_REGION" --AssociatedInstanceType Nat --AssociatedInstanceId "$nat" | jq -r '.EipAddresses.EipAddress[].AllocationId')
    log "deleting NAT gateway ${nat}"
    ali vpc DeleteNatGateway --RegionId "$ACK_REGION" --NatGatewayId "$nat" --Force true >/dev/null || true
    wait_for 300 "NAT gateway ${nat} deleted" bash -c "[ -z \"\$(aliyun vpc DescribeNatGateways --RegionId ${ACK_REGION} --NatGatewayId ${nat} | jq -r '.NatGateways.NatGateway[].NatGatewayId')\" ]" || true
    for eip in $eips; do
      log "releasing EIP ${eip}"
      ali vpc ReleaseEipAddress --RegionId "$ACK_REGION" --AllocationId "$eip" >/dev/null || true
    done
  done
  for sg in $(ali ecs DescribeSecurityGroups --RegionId "$ACK_REGION" --VpcId "$vpc" | jq -r '.SecurityGroups.SecurityGroup[].SecurityGroupId'); do
    wait_for 180 "security group ${sg} deleted" ali ecs DeleteSecurityGroup --RegionId "$ACK_REGION" --SecurityGroupId "$sg" || true
  done
  for vsw in $(ali vpc DescribeVSwitches --RegionId "$ACK_REGION" --VpcId "$vpc" | jq -r '.VSwitches.VSwitch[].VSwitchId'); do
    wait_for 180 "vSwitch ${vsw} deleted" ali vpc DeleteVSwitch --RegionId "$ACK_REGION" --VSwitchId "$vsw" || true
  done
  wait_for 180 "VPC ${vpc} deleted" ali vpc DeleteVpc --RegionId "$ACK_REGION" --VpcId "$vpc" || true
fi
rm -f "$STATE"

ledger down "seconds=$(($(date +%s) - start))"
log "down in $(($(date +%s) - start))s; sweep:"
sweep
