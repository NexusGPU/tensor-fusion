#!/usr/bin/env bash
# Renders the chart with `helm template` and checks the Karpenter NodeOverlay
# it ships. The overlay advertises the tensor-fusion.ai/index* capacity, so it
# must never apply to every instance type:
#   - by default it is restricted to GPU instance types;
#   - values that lack the block (an upgrade with --reuse-values) get the same
#     restriction;
#   - the requirements can be replaced through the values;
#   - an empty requirement list, or disabling it, renders no overlay at all;
#   - it is only rendered when the NodeOverlay CRD is known, unless forced.
set -euo pipefail

CHART=${CHART:-charts/tensor-fusion}
CRD_API=karpenter.sh/v1alpha1/NodeOverlay
failures=0

render() { helm template release "$CHART" -n tensor-fusion-sys "$@"; }

check() { # check <description> <command...>
    local description=$1
    shift
    if "$@" >/dev/null 2>&1; then
        echo "ok   $description"
    else
        echo "FAIL $description"
        failures=$((failures + 1))
    fi
}

# overlay <rendered>: the NodeOverlay documents, comments removed.
overlay() {
    RENDERED="$1" python3 - <<'PY'
import os
for document in os.environ["RENDERED"].split("\n---"):
    lines = [line for line in document.splitlines() if not line.lstrip().startswith("#")]
    if any(line.strip() == "kind: NodeOverlay" for line in lines):
        print("\n".join(lines))
PY
}

# requirements <overlay>: one line per requirement, "<key> <operator> <values>".
requirements() {
    OVERLAY="$1" python3 - <<'PY'
import os, re
lines = os.environ["OVERLAY"].splitlines()
inside, current, out = False, None, []
for line in lines:
    if re.match(r"^  requirements:", line):
        inside = True
        continue
    if inside and re.match(r"^  \S", line):
        break
    if not inside:
        continue
    text = line.strip()
    # Requirements are list items at the indentation of the block; the items
    # nested deeper are a requirement's values.
    if text.startswith("- ") and len(line) - len(line.lstrip()) == 4:
        current = {"values": []}
        out.append(current)
        text = text[2:].strip()
    if current is None or not text:
        continue
    if text.startswith("key:"):
        current["key"] = text.split(":", 1)[1].strip().strip('"')
    elif text.startswith("operator:"):
        current["operator"] = text.split(":", 1)[1].strip().strip('"')
    elif text.startswith("- "):
        current["values"].append(text[2:].strip().strip('"'))
for r in out:
    print(r.get("key", "?"), r.get("operator", "?"), ",".join(r["values"]))
PY
}

contains() { [[ "$1" == *"$2"* ]]; }
lacks() { [[ "$1" != *"$2"* ]]; }
is_empty() { [ -z "$1" ]; }
is_set() { [ -n "$1" ]; }
same() { [ "$1" = "$2" ]; }
has_line() { grep -qxF -- "$2" <<<"$1"; }

gpu_only=$'karpenter.k8s.aws/instance-gpu-count Gt 0\nkarpenter.k8s.aws/instance-gpu-manufacturer In nvidia'

# Without the CRD the chart must install on a cluster that has no Karpenter.
out=$(overlay "$(render)")
check "no overlay when the NodeOverlay CRD is not known" is_empty "$out"

out=$(overlay "$(render --api-versions "$CRD_API")")
reqs=$(requirements "$out")
check "the overlay is rendered when the CRD is known" is_set "$out"
check "by default it applies to instance types with GPUs only" has_line "$reqs" "karpenter.k8s.aws/instance-gpu-count Gt 0"
check "and only to the GPU vendor the chart defaults to" has_line "$reqs" "karpenter.k8s.aws/instance-gpu-manufacturer In nvidia"
check "the default requirements are exactly those" same "$reqs" "$gpu_only"
check "it is not a catch-all" lacks "$out" "requirements: []"
check "it advertises the index resources" contains "$out" "tensor-fusion.ai/index_f: 36"
check "and the v1 resource" contains "$out" "tensor-fusion.ai/index: 512"
check "no weight unless the values set one" lacks "$out" "weight:"

out=$(overlay "$(render --set karpenterNodeOverlay.enabled=true)")
check "enabled=true renders it without the CRD" same "$(requirements "$out")" "$gpu_only"

# Values of a release installed before the block existed.
out=$(overlay "$(render --api-versions "$CRD_API" --set karpenterNodeOverlay=null)")
check "values without the block get the GPU-only default" same "$(requirements "$out")" "$gpu_only"
out=$(overlay "$(render --api-versions "$CRD_API" --set karpenterNodeOverlay.requirements=null)")
check "and so do values without the requirements" same "$(requirements "$out")" "$gpu_only"

out=$(overlay "$(render --api-versions "$CRD_API" \
    --set-json 'karpenterNodeOverlay.requirements=[{"key":"karpenter.sh/nodepool","operator":"In","values":["gpu","gpu-spot"]}]' \
    --set karpenterNodeOverlay.weight=10)")
check "the requirements can be replaced" same "$(requirements "$out")" "karpenter.sh/nodepool In gpu,gpu-spot"
check "the default is not mixed in" lacks "$out" "karpenter.k8s.aws"
check "the weight is rendered" contains "$out" "weight: 10"

out=$(overlay "$(render --set karpenterNodeOverlay.enabled=true --set-json 'karpenterNodeOverlay.requirements=[]')")
check "an empty requirement list renders no overlay, not a catch-all" is_empty "$out"

out=$(overlay "$(render --api-versions "$CRD_API" --set karpenterNodeOverlay.enabled=false)")
check "enabled=false renders no overlay" is_empty "$out"

if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed"
