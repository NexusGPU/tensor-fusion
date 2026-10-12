#!/usr/bin/env bash
# Renders the chart with `helm template` and checks where the pods of its Helm
# hooks may be scheduled. Helm waits for a hook Job, so a hook pod on a node
# that cannot pull or run the image holds the install or upgrade until the
# timeout:
#   - by default every hook pod selects Linux nodes and has the affinity and
#     the tolerations of the controller;
#   - controller.admissionWebhooks.patch.nodeSelector / affinity / tolerations
#     replace them, kubernetes.io/os: linux staying unless the selector names
#     that key;
#   - values that lack the keys (an upgrade with --reuse-values) render the
#     defaults;
#   - with values-cn.yaml every hook image comes from the mirror.
# Hook pods are found by the helm.sh/hook annotation, so a hook added later is
# checked as well.
set -euo pipefail

CHART=${CHART:-charts/tensor-fusion}
CN_VALUES=${CN_VALUES:-$CHART/values-cn.yaml}
PATCH=controller.admissionWebhooks.patch
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

# pods <rendered> <hook|controller> <field>: one line per pod template,
# "<kind>/<name> <value>". The value is the block of that pod spec field with
# its lines joined by " | ", empty when the pod spec does not have the field.
# The field "images" gives one line per container instead, "<kind>/<name>
# <image reference>".
pods() {
    RENDERED="$1" WHICH="$2" FIELD="$3" python3 - <<'PY'
import os, re

def indent(line):
    return len(line) - len(line.lstrip())

which, field = os.environ["WHICH"], os.environ["FIELD"]
for document in os.environ["RENDERED"].split("\n---"):
    lines = [l for l in document.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    text = "\n".join(lines)
    containers = [l for l in lines if l.strip() == "containers:"]
    if not containers:
        continue
    kind = re.search(r"^kind: (\S+)", text, re.M).group(1)
    name = re.search(r"^  name: (\S+)", text, re.M).group(1)
    is_hook = re.search(r'^\s+"?helm\.sh/hook"?:', text, re.M) is not None
    is_controller = kind == "Deployment" and name.endswith("-controller")
    if (which == "hook" and not is_hook) or (which == "controller" and not is_controller):
        continue
    if field == "images":
        for l in lines:
            if re.match(r"\s*(- )?image:", l):
                print(kind + "/" + name, l.split(":", 1)[1].strip().strip("\"'"))
        continue
    else:
        # The fields of the pod spec are at the indentation of `containers:`.
        base, value, inside = indent(containers[0]), [], False
        for l in lines:
            if inside and indent(l) <= base:
                break
            if inside:
                value.append(l[base + 2:])
            elif l == " " * base + field + ":":
                inside = True
    print(kind + "/" + name, " | ".join(value))
PY
}

# values <pods output>: the distinct values, names removed.
values() { sed -E 's/^[^ ]+ ?//' <<<"$1" | sort -u; }
# each <pods output> <expected>: every pod has exactly that value.
each() { [ -n "$1" ] && [ "$(values "$1")" = "$2" ]; }
# image_prefix <pods output> <prefix>: every image of every pod starts with it.
image_prefix() { [ -n "$1" ] && ! values "$1" | awk -v prefix="$2" 'index($0, prefix) != 1' | grep -q .; }
# names <pods output>: the distinct pods.
names() { cut -d' ' -f1 <<<"$1" | sort -u; }
has_line() { grep -qF -- "$2" <<<"$1"; }
lacks() { [[ "$1" != *"$2"* ]]; }
is_set() { [ -n "$1" ]; }
same() { [ "$1" = "$2" ]; }

linux="kubernetes.io/os: linux"

echo "== default values"
out=$(render)
selector=$(pods "$out" hook nodeSelector)
check "the certificate hook pod is found" has_line "$selector" "Job/release-add-hook-crt "
check "the webhook patch hook pod is found" has_line "$selector" "Job/release-patch-admission-webhook "
check "the hooks are exactly those two" same "$(wc -l <<<"$selector" | tr -d ' ')" "2"
check "every hook pod selects Linux nodes, and nothing else" each "$selector" "$linux"
controller_affinity=$(values "$(pods "$out" controller affinity)")
check "the controller has an affinity by default" is_set "$controller_affinity"
check "every hook pod has the controller's affinity" each "$(pods "$out" hook affinity)" "$controller_affinity"
check "no tolerations, as the controller has none" each "$(pods "$out" hook tolerations)" ""

echo "== the controller's constraints are followed"
out=$(render \
    --set-json 'controller.affinity={"nodeAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":{"nodeSelectorTerms":[{"matchExpressions":[{"key":"node-role/system","operator":"Exists"}]}]}}}' \
    --set-json 'controller.tolerations=[{"key":"node-role/system","operator":"Exists","effect":"NoSchedule"}]')
controller_affinity=$(values "$(pods "$out" controller affinity)")
controller_tolerations=$(values "$(pods "$out" controller tolerations)")
check "the controller gets the affinity of the values" has_line "$controller_affinity" "node-role/system"
check "the controller gets the tolerations of the values" has_line "$controller_tolerations" "node-role/system"
check "every hook pod has that affinity" each "$(pods "$out" hook affinity)" "$controller_affinity"
check "and those tolerations" each "$(pods "$out" hook tolerations)" "$controller_tolerations"
check "and still selects Linux nodes" each "$(pods "$out" hook nodeSelector)" "$linux"

echo "== values of the hooks replace them"
out=$(render \
    --set-json 'controller.tolerations=[{"key":"node-role/system","operator":"Exists","effect":"NoSchedule"}]' \
    --set-json "$PATCH.nodeSelector={\"kubernetes.io/arch\":\"amd64\"}" \
    --set-json "$PATCH.affinity={\"nodeAffinity\":{\"requiredDuringSchedulingIgnoredDuringExecution\":{\"nodeSelectorTerms\":[{\"matchExpressions\":[{\"key\":\"pool\",\"operator\":\"In\",\"values\":[\"hooks\"]}]}]}}}" \
    --set-json "$PATCH.tolerations=[{\"key\":\"pool\",\"operator\":\"Equal\",\"value\":\"hooks\",\"effect\":\"NoSchedule\"}]")
check "the node selector of the values, with Linux added" each "$(pods "$out" hook nodeSelector)" \
    "kubernetes.io/arch: amd64 | $linux"
affinity=$(pods "$out" hook affinity)
check "the affinity of the values" has_line "$affinity" "key: pool"
check "is on every hook pod" same "$(values "$affinity" | wc -l | tr -d ' ')" "1"
check "the controller's affinity is not mixed in" lacks "$affinity" "podAntiAffinity"
check "the tolerations of the values, the controller's not mixed in" each "$(pods "$out" hook tolerations)" \
    "- effect: NoSchedule |   key: pool |   operator: Equal |   value: hooks"
check "the controller keeps its own tolerations" has_line "$(pods "$out" controller tolerations)" "node-role/system"
check "and its own affinity" has_line "$(pods "$out" controller affinity)" "podAntiAffinity"

out=$(render --set-json "$PATCH.nodeSelector={\"kubernetes.io/os\":\"windows\",\"pool\":\"hooks\"}")
check "a selector that names kubernetes.io/os is left as it is" each "$(pods "$out" hook nodeSelector)" \
    "kubernetes.io/os: windows | pool: hooks"

echo "== values without the keys"
# Values of a release installed before the keys existed.
default_pods=$(for field in nodeSelector affinity tolerations; do pods "$(render)" hook "$field"; done)
out=$(render --set "$PATCH.nodeSelector=null" --set "$PATCH.affinity=null" --set "$PATCH.tolerations=null")
check "the chart renders" is_set "$out"
check "with the defaults" same \
    "$(for field in nodeSelector affinity tolerations; do pods "$out" hook "$field"; done)" "$default_pods"
out=$(render --set "$PATCH.nodeSelector=null" --set "$PATCH.affinity=null" --set "$PATCH.tolerations=null" \
    --set controller.affinity=null --set controller.tolerations=null)
check "without the controller's keys either: Linux nodes" each "$(pods "$out" hook nodeSelector)" "$linux"
check "and no affinity" each "$(pods "$out" hook affinity)" ""

echo "== values-cn.yaml"
default_images=$(pods "$(render)" hook images)
check "default values: every hook image is the upstream one" image_prefix "$default_images" \
    "registry.k8s.io/ingress-nginx/kube-webhook-certgen:"
out=$(render -f "$CN_VALUES")
cn_images=$(pods "$out" hook images)
check "both hook pods are rendered" same "$(names "$cn_images" | wc -l | tr -d ' ')" "2"
check "with their three containers" same "$(wc -l <<<"$cn_images" | tr -d ' ')" "3"
check "every hook image comes from the mirror" image_prefix "$cn_images" \
    "k8s.m.daocloud.io/ingress-nginx/kube-webhook-certgen:"
check "in the version of the default values" same \
    "$(values "$cn_images" | sed -E 's|.*/||')" "$(values "$default_images" | sed -E 's|.*/||')"
check "the hook pods select Linux nodes with the overlay too" each "$(pods "$out" hook nodeSelector)" "$linux"

if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed"
