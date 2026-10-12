#!/usr/bin/env bash
# What has this stack cost, and what is it costing right now?
#
#   1. Cost Explorer, month to date, filtered by the Project tag (authoritative,
#      but lags by up to 24 h and needs the tag activated as a cost allocation
#      tag - attempted here), plus the account-wide month-to-date by service.
#   2. A live estimate from what is running: every tagged instance (spot at the
#      current spot price, on-demand from e2e/pricing/<region>.json), the EKS
#      control plane, public IPv4 addresses and EBS volumes, with $/h now and
#      the amount accrued since each resource started.
#
#   scripts/cost.sh                 print both
#   scripts/cost.sh --ledger FILE   also append "<time> <$/h> <accrued $>" to FILE
#
# Each Cost Explorer API call is billed $0.01.

source "$(dirname "$0")/lib.sh"
need aws jq

NAME="$TF_E2E_CLUSTER"
PRICES="${TF_E2E_HOME}/../pricing/${TF_E2E_REGION}.json"
LEDGER=""
[ "${1:-}" = "--ledger" ] && LEDGER="${2:?--ledger needs a file}"

now=$(date -u +%s)
epoch_of() { # RFC 3339 with any offset (2026-10-11T23:36:21.790000+08:00, ...Z) -> epoch; BSD and GNU date
  local t
  t=$(printf '%s' "$1" | sed -E 's/\.[0-9]+//; s/Z$/+0000/; s/([+-][0-9]{2}):([0-9]{2})$/\1\2/')
  date -j -f %Y-%m-%dT%H:%M:%S%z "$t" +%s 2>/dev/null || date -d "$1" +%s
}

# ------------------------------------------------------------ Cost Explorer ----
month_start=$(date -u +%Y-%m-01)
tomorrow=$(date -u -v+1d +%Y-%m-%d 2>/dev/null || date -u -d tomorrow +%Y-%m-%d)

echo "== Cost Explorer, ${month_start}..today (data lags up to 24 h) =="
aws ce update-cost-allocation-tags-status \
  --cost-allocation-tags-status TagKey=Project,Status=Active >/dev/null 2>&1 ||
  echo "   (cost allocation tag 'Project' cannot be activated yet: AWS only lists a tag key ~24 h after its first use)"
tagged=$(aws ce get-cost-and-usage --time-period "Start=${month_start},End=${tomorrow}" \
  --granularity MONTHLY --metrics UnblendedCost \
  --filter "{\"Tags\":{\"Key\":\"Project\",\"Values\":[\"${NAME}\"]}}" \
  --query 'ResultsByTime[].Total.UnblendedCost.Amount' --output text 2>/dev/null || echo "n/a")
echo "   Project=${NAME}                : \$${tagged}"
echo "   whole account, by service      :"
aws ce get-cost-and-usage --time-period "Start=${month_start},End=${tomorrow}" \
  --granularity MONTHLY --metrics UnblendedCost --group-by Type=DIMENSION,Key=SERVICE \
  --query 'ResultsByTime[].Groups[].[Keys[0],Metrics.UnblendedCost.Amount]' --output text 2>/dev/null |
  awk -F'\t' '$2+0 >= 0.005 {printf "     %-48s $%.2f\n", $1, $2; s+=$2} END {printf "     %-48s $%.2f\n", "TOTAL (rows >= $0.005)", s}'

# ------------------------------------------------------------ live estimate ----
echo
echo "== Live estimate from running resources (${TF_E2E_REGION}) =="
rate_total=0
accrued_total=0
add() { # add <$/h> <hours>
  rate_total=$(echo "$rate_total + $1" | bc -l)
  accrued_total=$(echo "$accrued_total + $1 * $2" | bc -l)
}

cp_price=$(jq -r '.eks_control_plane_per_hour' "$PRICES")
ip_price=$(jq -r '.public_ipv4_per_hour' "$PRICES")
gb_month=$(jq -r '.gp3_per_gb_month' "$PRICES")

created=$(aws eks describe-cluster --name "$NAME" --query 'cluster.createdAt' --output text 2>/dev/null || true)
if [ -n "$created" ] && [ "$created" != "None" ]; then
  hours=$(echo "($now - $(epoch_of "$created")) / 3600" | bc -l)
  printf "   %-22s %-14s %-10s \$%.4f/h  up %5.2f h  \$%.3f\n" "EKS control plane" "$NAME" "-" "$cp_price" "$hours" "$(echo "$cp_price * $hours" | bc -l)"
  add "$cp_price" "$hours"
else
  echo "   EKS control plane      none"
fi

instances="$(mktemp)"
trap 'rm -f "$instances"' EXIT
for f in "Name=tag:Project,Values=${NAME}" "Name=tag:eks:cluster-name,Values=${NAME}" "Name=tag:eks:eks-cluster-name,Values=${NAME}" "Name=tag:kubernetes.io/cluster/${NAME},Values=owned"; do
  aws ec2 describe-instances --filters "$f" "Name=instance-state-name,Values=pending,running" \
    --query 'Reservations[].Instances[].[InstanceId,InstanceType,InstanceLifecycle||`on-demand`,LaunchTime,Placement.AvailabilityZone,PublicIpAddress||`-`,(Tags[?Key==`karpenter.sh/nodepool`].Value)[0]||(Tags[?Key==`eks:nodegroup-name`].Value)[0]||`-`]' \
    --output text 2>/dev/null >>"$instances" || true
done
sort -u "$instances" -o "$instances"

count=0
while IFS=$'\t' read -r id type lifecycle launched az ip owner; do
  [ -z "$id" ] && continue
  count=$((count + 1))
  if [ "$lifecycle" = "spot" ]; then
    price=$(aws ec2 describe-spot-price-history --instance-types "$type" --availability-zone "$az" \
      --product-descriptions "Linux/UNIX" --max-items 1 --query 'SpotPriceHistory[0].SpotPrice' --output text 2>/dev/null | head -1)
  else
    price=$(jq -r --arg t "$type" '.on_demand[$t] // empty' "$PRICES")
  fi
  if [ -z "$price" ] || [ "$price" = "None" ]; then
    echo "   $id $type $lifecycle: no price (run scripts/refresh-prices.sh $type); NOT counted"
    continue
  fi
  [ "$ip" != "-" ] && price=$(echo "$price + $ip_price" | bc -l)
  hours=$(echo "($now - $(epoch_of "$launched")) / 3600" | bc -l)
  printf "   %-22s %-14s %-10s \$%.4f/h  up %5.2f h  \$%.3f  (%s)\n" "$id" "$type" "$lifecycle" "$price" "$hours" "$(echo "$price * $hours" | bc -l)" "$owner"
  add "$price" "$hours"
done <"$instances"
[ "$count" = 0 ] && echo "   EC2 instances          none"

gb=$(aws ec2 describe-volumes --filters "Name=tag:Project,Values=${NAME}" --query 'sum(Volumes[].Size)' --output text 2>/dev/null || echo 0)
[ "$gb" = "None" ] && gb=0
vol_rate=$(echo "$gb * $gb_month / 730" | bc -l)
printf "   %-22s %-14s %-10s \$%.4f/h  (in the burn rate, not in the accrued total)\n" "EBS gp3" "${gb} GiB" "-" "$vol_rate"
rate_total=$(echo "$rate_total + $vol_rate" | bc -l)

printf "\n   burn rate now          : \$%.3f/h\n" "$rate_total"
printf "   accrued by what exists : \$%.3f   (resources already deleted are not in this number; see the ledger / Cost Explorer)\n" "$accrued_total"

if [ -n "$LEDGER" ]; then
  mkdir -p "$(dirname "$LEDGER")"
  printf "%s\t%s\tcost\trate_per_hour=%.4f\taccrued_usd=%.4f\tinstances=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$NAME" "$rate_total" "$accrued_total" "$count" >>"$LEDGER"
fi
