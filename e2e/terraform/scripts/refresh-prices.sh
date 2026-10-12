#!/usr/bin/env bash
# Regenerates e2e/pricing/<region>.json (Linux on-demand $/h) from the AWS
# Price List API. The file is committed: `make cost` and the FinOps cases of the
# test suite read it, so cost numbers in reports are reproducible.
#   scripts/refresh-prices.sh [instance-type ...]

source "$(dirname "$0")/lib.sh"
need aws jq

out="${TF_E2E_HOME}/../pricing/${TF_E2E_REGION}.json"
types=("$@")
if [ ${#types[@]} -eq 0 ]; then
  types=(t3.medium t3a.medium t3.large t3a.large m5.large m5a.large m6i.large c5.large c6i.large
    g4dn.xlarge g4dn.2xlarge g5.xlarge g6.xlarge)
fi

tmp="$(mktemp)"
echo '{}' >"$tmp"
for t in "${types[@]}"; do
  price=$(aws pricing get-products --region us-east-1 --service-code AmazonEC2 --max-items 1 \
    --filters "Type=TERM_MATCH,Field=instanceType,Value=${t}" \
    "Type=TERM_MATCH,Field=regionCode,Value=${TF_E2E_REGION}" \
    "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
    "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
    "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" \
    "Type=TERM_MATCH,Field=capacitystatus,Value=Used" \
    --query 'PriceList[0]' --output text |
    jq -r '[.terms.OnDemand[].priceDimensions[].pricePerUnit.USD][0]')
  log "${t}: ${price}"
  jq --arg t "$t" --argjson p "$price" '.[$t] = $p' "$tmp" >"$tmp.n" && mv "$tmp.n" "$tmp"
done

jq -S --arg region "$TF_E2E_REGION" --arg date "$(date -u +%Y-%m-%d)" '{
  region: $region,
  source: "AWS Price List API, Linux, shared tenancy, on-demand, USD per hour",
  refreshed: $date,
  eks_control_plane_per_hour: 0.10,
  public_ipv4_per_hour: 0.005,
  gp3_per_gb_month: 0.08,
  on_demand: .
}' "$tmp" >"$out"
rm -f "$tmp"
log "wrote $out"
