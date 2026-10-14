#!/usr/bin/env bash
# ACK tier, stage 1: a throw-away ACK managed cluster for the e2e suite.
#
#   VPC ${ACK_VPC_CIDR} (2 vSwitches / 2 zones, NAT gateway + EIP for egress)
#   │  ACK creates the NAT gateway and its EIP (snat_entry): 200 Mbit/s, pay by
#   │  traffic. Raising the bandwidth is refused for this account (OrderError).
#   └─ ACK managed ${ACK_K8S_VERSION} (Flannel, RRSA on, public API endpoint)
#      │  no CSI add-ons: nothing here needs cloud disks (GreptimeDB gets a hostPath
#      │  volume), and a disk is one more thing that can outlive the cluster
#      └─ node pool "system"  1 x ${ACK_SYSTEM_INSTANCE_TYPE} (${ACK_IMAGE_TYPE})  label e2e.tensor-fusion.ai/role=system
#
# GPU nodes are not part of this stage: they all come from Karpenter (install.sh),
# because TensorFusion's expander only clones nodes owned by a Karpenter NodeClaim.
# ACK's account-level service roles (ACK_SERVICE_ROLES) are created first when
# missing; they are what the console's one-time authorization creates, and they
# stay in the account.
# Idempotent; parameters in lib.sh. Everything is tagged Project=${ACK_NAME}.

source "$(dirname "$0")/lib.sh"
need aliyun kubectl jq

start=$(date +%s)
tags_kv=(--Tag.1.Key Project --Tag.1.Value "$ACK_NAME" --Tag.2.Key karpenter.sh/discovery --Tag.2.Value "$ACK_NAME")

# --- 0. ACK service roles (account level, kept) -------------------------------------------
# Without them CreateCluster fails with ErrKubernetesAuditRoleNotAttach / AddonRoleNotAuthorized.
created=()
for role in $ACK_SERVICE_ROLES; do
  aliyun ram GetRole --RoleName "$role" >/dev/null 2>&1 && continue
  log "creating ACK service role ${role}"
  aliyun ram CreateRole --RoleName "$role" --Description "ACK service role" \
    --AssumeRolePolicyDocument '{"Version":"1","Statement":[{"Effect":"Allow","Action":"sts:AssumeRole","Principal":{"Service":["cs.aliyuncs.com"]}}]}' >/dev/null
  aliyun ram AttachPolicyToRole --PolicyType System --PolicyName "${role}Policy" --RoleName "$role" >/dev/null
  created+=("$role")
done
if [ ${#created[@]} -gt 0 ]; then
  printf '%s created ACK service roles: %s\n' "$(date -u +%FT%TZ)" "${created[*]}" >>"${ACK_HOME}/account-changes.log"
fi

# --- 1. VPC and vSwitches ------------------------------------------------------------
vpc=$(ali vpc DescribeVpcs --RegionId "$ACK_REGION" --VpcName "$ACK_NAME" | jq -r '.Vpcs.Vpc[0].VpcId // empty')
if [ -z "$vpc" ]; then
  log "creating VPC ${ACK_NAME} (${ACK_VPC_CIDR})"
  vpc=$(ali vpc CreateVpc --RegionId "$ACK_REGION" --VpcName "$ACK_NAME" --CidrBlock "$ACK_VPC_CIDR" "${tags_kv[@]}" | jq -r .VpcId)
  wait_for 120 "VPC ${vpc} Available" bash -c "aliyun vpc DescribeVpcAttribute --RegionId $ACK_REGION --VpcId $vpc | jq -e '.Status == \"Available\"'"
fi
state_set VPC_ID "$vpc"

# Zones: the ones that sell every GPU instance type, first two.
zones=$(for t in $ACK_GPU_INSTANCE_TYPES; do
  ali ecs DescribeAvailableResource --RegionId "$ACK_REGION" --DestinationResource InstanceType --InstanceChargeType PostPaid --InstanceType "$t" |
    jq -r '.AvailableZones.AvailableZone[]? | select(.StatusCategory == "WithStock") | .ZoneId'
done | sort | uniq -c | sort -rn | awk '{print $2}' | head -2)
[ -n "$zones" ] || { log "no zone of ${ACK_REGION} sells ${ACK_GPU_INSTANCE_TYPES}"; exit 1; }

vsws=()
i=0
for z in $zones; do
  name="${ACK_NAME}-${z}"
  id=$(ali vpc DescribeVSwitches --RegionId "$ACK_REGION" --VpcId "$vpc" --VSwitchName "$name" | jq -r '.VSwitches.VSwitch[0].VSwitchId // empty')
  if [ -z "$id" ]; then
    cidr="$(echo "$ACK_VPC_CIDR" | cut -d. -f1-2).$((i * 16)).0/20"
    log "creating vSwitch ${name} (${cidr})"
    id=$(ali vpc CreateVSwitch --RegionId "$ACK_REGION" --VpcId "$vpc" --ZoneId "$z" --CidrBlock "$cidr" --VSwitchName "$name" "${tags_kv[@]}" | jq -r .VSwitchId)
  fi
  vsws+=("$id")
  i=$((i + 1))
done
state_set VSWITCH_IDS "${vsws[*]}"
vsw_json=$(json_list "${vsws[@]}")

# --- 2. cluster ----------------------------------------------------------------------
cid=$(cluster_id)
if [ -z "$cid" ]; then
  log "creating ACK cluster ${ACK_NAME} (${ACK_CLUSTER_SPEC}, Kubernetes ${ACK_K8S_VERSION})"
  body=$(jq -n \
    --arg name "$ACK_NAME" --arg region "$ACK_REGION" --arg spec "$ACK_CLUSTER_SPEC" --arg ver "$ACK_K8S_VERSION" \
    --arg vpc "$vpc" --argjson vsws "$vsw_json" \
    --arg pod "$ACK_POD_CIDR" --arg svc "$ACK_SERVICE_CIDR" '{
      name: $name, cluster_type: "ManagedKubernetes", cluster_spec: $spec, region_id: $region,
      kubernetes_version: $ver, vpcid: $vpc, vswitch_ids: $vsws,
      container_cidr: $pod, service_cidr: $svc, node_cidr_mask: "25", proxy_mode: "ipvs",
      snat_entry: true, endpoint_public_access: true, is_enterprise_security_group: true,
      deletion_protection: false, enable_rrsa: true, timezone: "UTC",
      addons: [{name: "flannel"}],
      tags: [{key: "Project", value: $name}]
    }')
  cid=$(ali cs POST /clusters --header "Content-Type=application/json" --body "$body" | jq -r .cluster_id)
  [ -n "$cid" ] && [ "$cid" != "null" ] || { log "CreateCluster returned no cluster id"; exit 1; }
fi
state_set CLUSTER_ID "$cid"
log "cluster ${cid}: waiting for state running"
until s=$(ali cs GET "/clusters/${cid}" | jq -r .state) && [ "$s" = "running" ]; do
  [ "$s" = "failed" ] && { log "cluster creation failed"; ali cs GET "/clusters/${cid}/events" | jq -r '.events[-5:][]? | .message' >&2 || true; exit 1; }
  [ $(($(date +%s) - start)) -gt 2400 ] && { log "TIMEOUT waiting for the cluster (state ${s})"; exit 1; }
  sleep 30
done
log "cluster running ($(($(date +%s) - start))s)"
ledger up-cluster "seconds=$(($(date +%s) - start))"

# The security group ACK created is what Karpenter's nodes join; tag it for discovery.
sg=$(ali cs GET "/clusters/${cid}" | jq -r .security_group_id)
state_set SECURITY_GROUP_ID "$sg"
ali ecs TagResources --RegionId "$ACK_REGION" --ResourceType securitygroup --ResourceId.1 "$sg" "${tags_kv[@]}" >/dev/null

# --- 3. system node pool -----------------------------------------------------------------
# Its own step (not part of CreateCluster) so that a failed pool can be replaced
# without recreating the cluster. Kubernetes >= 1.35 refuses cgroup v1 nodes
# (CgroupV1NotSupport), which rules out the plain AliyunLinux3 image.
np=$(ali cs GET "/clusters/${cid}/nodepools" | jq -c '.nodepools[] | select(.nodepool_info.name == "system")')
if [ -n "$np" ] && [ "$(echo "$np" | jq -r '.status.healthy_nodes - .status.failed_nodes')" -lt 1 ] && [ "$(echo "$np" | jq -r .status.failed_nodes)" -gt 0 ]; then
  npid=$(echo "$np" | jq -r .nodepool_info.nodepool_id)
  log "node pool system (${npid}) has only failed nodes: deleting it"
  ali cs DELETE "/clusters/${cid}/nodepools/${npid}?force=true" >/dev/null
  wait_for 600 "node pool ${npid} deleted" bash -c "! aliyun cs GET /clusters/${cid}/nodepools --region ${ACK_REGION} | jq -e --arg id ${npid} '.nodepools[] | select(.nodepool_info.nodepool_id == \$id)'"
  np=""
fi
if [ -z "$np" ]; then
  log "creating node pool system (1 x ${ACK_SYSTEM_INSTANCE_TYPE}, ${ACK_IMAGE_TYPE})"
  body=$(jq -n --argjson vsws "$vsw_json" --arg itype "$ACK_SYSTEM_INSTANCE_TYPE" --arg img "$ACK_IMAGE_TYPE" \
    --arg key "$ACK_KEY_PAIR" --arg role "$ROLE_LABEL" --arg rtver "$ACK_RUNTIME_VERSION" --arg name "$ACK_NAME" \
    --argjson disk "$ACK_SYSTEM_DISK_GB" '{
      nodepool_info: {name: "system"},
      scaling_group: {
        vswitch_ids: $vsws, instance_types: [$itype], instance_charge_type: "PostPaid",
        system_disk_category: "cloud_essd", system_disk_size: $disk, key_pair: $key,
        desired_size: 1, image_type: $img,
        tags: [{key: "Project", value: $name}]
      },
      kubernetes_config: {runtime: "containerd", runtime_version: $rtver, labels: [{key: $role, value: "system"}]},
      management: {enable: false}
    }')
  ali cs POST "/clusters/${cid}/nodepools" --header "Content-Type=application/json" --body "$body" >/dev/null
fi

# --- 4. kubeconfig -------------------------------------------------------------------
(umask 077 && ali cs GET "/k8s/${cid}/user_config" --TemporaryDurationMinutes 2880 | jq -r .config >"$KUBECONFIG")
wait_for 900 "system node Ready" kubectl wait node -l "${ROLE_LABEL}=system" --for=condition=Ready --timeout=10s

ledger up-stage1 "seconds=$(($(date +%s) - start))"
log "stage 1 done in $(($(date +%s) - start))s. export KUBECONFIG=${KUBECONFIG}"
kubectl get nodes -o wide -L "${ROLE_LABEL}" || true
