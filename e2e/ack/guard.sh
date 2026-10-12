#!/usr/bin/env bash
# ACK tier cost guard: a loop that keeps a provisioning bug from turning into a
# bill. Every ${ACK_GUARD_INTERVAL}s it
#   - terminates ECS instances of the stack that no NodeClaim and no ACK node
#     pool accounts for (the provider can launch two instances for one NodeClaim),
#     once they are older than 5 minutes;
#   - when more than ${ACK_MAX_GPU_INSTANCES} GPU instances exist, sets the limits of
#     every NodePool to zero (nothing new launches) and deletes the GPUNodeClaims
#     of path B, then reports it; lifting the freeze is a decision for a person
#     (rerun install.sh);
#   - exits when the cluster is gone.
#
#   ack/guard.sh            run in the foreground (start it in the background and keep the log)
#   ack/guard.sh once       one pass

source "$(dirname "$0")/lib.sh"
need aliyun kubectl jq

: "${ACK_MAX_GPU_INSTANCES:=4}"   # hard ceiling over NodePool tf-gpu (limit ${ACK_GPU_NODE_LIMIT}) plus path B
: "${ACK_GUARD_INTERVAL:=120}"

pass() {
  local instances claimed pool_ids gpu now
  instances=$(ali ecs DescribeInstances --RegionId "$ACK_REGION" --PageSize 100 --Tag.1.Key Project --Tag.1.Value "$ACK_NAME" |
    jq -c '[.Instances.Instance[] | select(.Status != "Released") | {id: .InstanceId, type: .InstanceType, status: .Status, created: .CreationTime}]') || return 0
  claimed=$(kubectl get nodeclaim -o json 2>/dev/null | jq -r '.items[].status.providerID // empty | split(".") | last') || return 0
  pool_ids=$(ali cs GET "/clusters/$1/nodes" | jq -r '.nodes[]?.instance_id') || return 0
  now=$(date -u +%s)
  for row in $(echo "$instances" | jq -r '.[] | .id + "," + .created'); do
    id=${row%%,*}
    created=$(date -u -j -f '%Y-%m-%dT%H:%MZ' "${row##*,}" +%s 2>/dev/null || date -u -d "${row##*,}" +%s)
    echo "$claimed" "$pool_ids" | tr ' ' '\n' | grep -qx "$id" && continue
    [ $((now - created)) -lt 300 ] && continue # may still be on its way into a NodeClaim
    log "GUARD: instance ${id} belongs to no NodeClaim and no node pool: terminating it"
    ali ecs DeleteInstance --InstanceId "$id" --Force true >/dev/null || true
  done
  # Instances on their way out (Stopping / Stopped) no longer cost GPU time and
  # stay listed for minutes: only the ones that run or are about to count.
  gpu=$(echo "$instances" | jq '[.[] | select(.type | test("^ecs\\.(gn|ebmgn)")) | select(.status == "Running" or .status == "Starting" or .status == "Pending")] | length')
  if [ "$gpu" -gt "$ACK_MAX_GPU_INSTANCES" ]; then
    log "GUARD: ${gpu} GPU instances exceed the ceiling of ${ACK_MAX_GPU_INSTANCES}: freezing every NodePool"
    for np in $(kubectl get nodepool -o name); do
      kubectl patch "$np" --type merge -p '{"spec":{"limits":{"cpu":"0","nvidia.com/gpu":"0"}}}' >/dev/null || true
    done
    kubectl delete gpunodeclaim --all --wait=false >/dev/null 2>&1 || true
  fi
  echo "$(date -u +%FT%TZ) instances=$(echo "$instances" | jq length) gpu=${gpu}" >>"${ACK_HOME}/guard.tsv"
}

cid=$(cluster_id)
[ -n "$cid" ] || { log "no cluster"; exit 0; }
if [ "${1:-}" = "once" ]; then
  pass "$cid"
  exit 0
fi
log "guard running: ceiling ${ACK_MAX_GPU_INSTANCES} GPU instances, every ${ACK_GUARD_INTERVAL}s"
while [ -n "$(cluster_id)" ]; do
  pass "$cid"
  sleep "$ACK_GUARD_INTERVAL"
done
log "cluster gone, guard exits"
