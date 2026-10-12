#!/usr/bin/env bash
# Dead-man switch control. The switch itself lives in AWS (deadman.tf); this
# only moves or shows its deadline.
#
#   scripts/deadman.sh arm <ttl>     set the deadline to now + ttl (45m, 3h). No-op when the
#                                    stack (its SSM parameter) does not exist yet: Terraform
#                                    creates it armed on the first apply.
#   scripts/deadman.sh status        deadline, time left, last teardown action, recent ticks
#   scripts/deadman.sh fire          invoke the Lambda now (it still only acts after the deadline)

source "$(dirname "$0")/lib.sh"
need aws jq

PARAM="/${TF_E2E_CLUSTER}/deadman/deadline"
STATUS="/${TF_E2E_CLUSTER}/deadman/last-action"
FN="${TF_E2E_CLUSTER}-deadman"

ttl_seconds() {
  case "$1" in
    *h) echo $((${1%h} * 3600)) ;;
    *m) echo $((${1%m} * 60)) ;;
    *) log "TTL must look like 45m or 3h (got '$1')"; exit 2 ;;
  esac
}

iso_after() { # seconds from now -> RFC 3339 UTC (BSD and GNU date)
  date -u -v+"$1"S +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "+$1 seconds" +%Y-%m-%dT%H:%M:%SZ
}

epoch_of() {
  date -u -j -f %Y-%m-%dT%H:%M:%SZ "$1" +%s 2>/dev/null || date -u -d "$1" +%s
}

case "${1:-status}" in
  arm)
    ttl="${2:?usage: deadman.sh arm <ttl>}"
    secs=$(ttl_seconds "$ttl")
    if ! aws ssm get-parameter --name "$PARAM" >/dev/null 2>&1; then
      log "dead-man switch not created yet; terraform will arm it"
      exit 0
    fi
    deadline=$(iso_after "$secs")
    aws ssm put-parameter --name "$PARAM" --type String --overwrite --value "$deadline" >/dev/null
    log "dead-man deadline set to ${deadline} (now + ${ttl})"
    ;;
  status)
    deadline=$(aws ssm get-parameter --name "$PARAM" --query Parameter.Value --output text 2>/dev/null || true)
    if [ -z "$deadline" ]; then
      log "no dead-man switch found for ${TF_E2E_CLUSTER} in ${TF_E2E_REGION}"
      exit 1
    fi
    left=$(($(epoch_of "$deadline") - $(date -u +%s)))
    if [ "$left" -gt 0 ]; then
      echo "deadline : ${deadline}  (in $((left / 60)) min)"
    else
      echo "deadline : ${deadline}  (EXPIRED $((-left / 60)) min ago - compute is being torn down)"
    fi
    echo "rule     : $(aws events describe-rule --name "$FN" --query '[State,ScheduleExpression]' --output text 2>/dev/null || echo missing)"
    echo "last action: $(aws ssm get-parameter --name "$STATUS" --query Parameter.Value --output text 2>/dev/null || echo '{}')"
    echo "recent ticks:"
    aws logs filter-log-events --log-group-name "/aws/lambda/${FN}" \
      --start-time $((($(date +%s) - 1800) * 1000)) --filter-pattern '"checked_at"' \
      --query 'events[-4:].message' --output text 2>/dev/null | tr '\t' '\n' | grep -v '^[[:space:]]*$' | sed 's/^/  /' || true
    ;;
  fire)
    out="$(mktemp)"
    aws lambda invoke --function-name "$FN" --cli-binary-format raw-in-base64-out --payload '{}' "$out" >/dev/null
    jq . "$out"
    rm -f "$out"
    ;;
  *)
    log "usage: deadman.sh arm <ttl> | status | fire"
    exit 2
    ;;
esac
