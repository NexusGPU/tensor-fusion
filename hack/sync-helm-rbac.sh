#!/usr/bin/env bash
# Sync the operator ClusterRole rules from the controller-gen output
# (config/rbac/role.yaml) into the Helm chart template. `make manifests` only
# copies CRDs to the chart, not RBAC, so the chart's rbac.yaml drifts behind the
# generated role. Run this after `make manifests` to keep them in sync.
set -euo pipefail
ROLE=config/rbac/role.yaml
CHART=charts/tensor-fusion/templates/rbac.yaml
rules=$(python3 -c "import yaml;d=yaml.safe_load(open('$ROLE'));print(yaml.dump({'rules':d['rules']},default_flow_style=False,sort_keys=False),end='')")
header=$(sed -n '1,/^rules:/p' "$CHART" | sed '$d')
tail=$(sed -n '/^---/,$p' "$CHART")
{ printf '%s\n' "$header"; printf '%s\n' "$rules"; printf '%s\n' "$tail"; } > "$CHART.tmp" && mv "$CHART.tmp" "$CHART"
echo "synced operator RBAC: config/rbac/role.yaml -> $CHART"
