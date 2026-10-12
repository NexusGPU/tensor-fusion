#!/usr/bin/env bash
# Renders the chart with `helm template -f values-cn.yaml` and checks the
# images an installation in mainland China would pull:
#   - no image comes from Docker Hub, named explicitly (docker.io/...) or by
#     an unqualified reference (tensorfusion/..., which resolves to it);
#   - every image names its registry;
#   - the images are the same versions as with the default values, so the
#     overlay cannot fall behind values.yaml unnoticed;
#   - the hypervisor image is the release named by appVersion. The template
#     takes a full image reference or defaults to the Docker Hub path, so the
#     overlay has to repeat the tag;
#   - the chart lints with the overlay.
# Images are read from the pod specs and from the TensorFusionCluster and
# ProviderConfig resources the chart renders.
set -euo pipefail

CHART=${CHART:-charts/tensor-fusion}
CN_VALUES=${CN_VALUES:-$CHART/values-cn.yaml}
MIRROR=registry.cn-hangzhou.aliyuncs.com/tensorfusion
failures=0

# References that may remain on Docker Hub in a render with the overlay, one
# per line as "<image reference>  # <reason>". There are none: every image the
# chart renders can be set from the values.
ALLOWED_DOCKER_HUB=""

# The cluster agent sidecar is only rendered with an agent id and a token.
render() {
    helm template release "$CHART" -n tensor-fusion-sys \
        --set agent.agentId=example-org:example-env --set agent.enrollToken=example-enroll-token "$@"
}

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

# images <rendered>: one line per image reference, "<key> <reference>". Keys
# are image, *Image, and the entries of an `images:` map.
images() {
    RENDERED="$1" python3 - <<'PY'
import os, re
entry = re.compile(r"^(\s*)(?:- )?([A-Za-z]+):\s*(\S.*)$")
block_indent = None
for line in os.environ["RENDERED"].splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    indent = len(line) - len(line.lstrip())
    if block_indent is not None and indent <= block_indent:
        block_indent = None
    if line.strip() == "images:":
        block_indent = indent
        continue
    match = entry.match(line)
    if not match:
        continue
    key, value = match.group(2), match.group(3).strip().strip("\"'")
    if block_indent is not None or key == "image" or key.endswith("Image"):
        print(key, value)
PY
}

# versions <images>: "<key> <name>:<tag>" without the registry and path.
versions() { sed -E 's|^([^ ]+) .*/([^/]+)$|\1 \2|' <<<"$1" | sort; }

# unqualified <images>: references whose first component is not a registry
# host (no dot, no port, not localhost); the container runtime resolves those
# to Docker Hub.
unqualified() {
    awk '{ n = split($2, part, "/"); if (n == 1 || (part[1] !~ /[.:]/ && part[1] != "localhost")) print }' <<<"$1"
}

# allowed: the references of ALLOWED_DOCKER_HUB, reasons removed.
allowed() { sed -E 's/[[:space:]]*#.*$//; /^[[:space:]]*$/d' <<<"$ALLOWED_DOCKER_HUB"; }

# docker_hub <images>: references that resolve to Docker Hub, named explicitly
# or unqualified, except the allowed ones.
docker_hub() {
    { grep -E ' (index\.|registry-1\.)?docker\.io/' <<<"$1" || true; unqualified "$1"; } |
        while read -r key reference; do
            allowed | grep -qxF -- "$reference" || echo "$key $reference"
        done
}

contains() { grep -qF -- "$2" <<<"$1"; }
has_line() { grep -qxF -- "$2" <<<"$1"; }
is_empty() { [ -z "$1" ]; }
same() { [ "$1" = "$2" ]; }

# none <description> <found>: passes when nothing was found, lists it otherwise.
none() {
    check "$1" is_empty "$2"
    if [ -n "$2" ]; then
        while IFS= read -r line; do echo "       $line"; done <<<"$2"
    fi
}

app_version=$(sed -n 's/^appVersion: *"\(.*\)"$/\1/p' "$CHART/Chart.yaml")
default_images=$(images "$(render)")
out=$(render -f "$CN_VALUES")
cn_images=$(images "$out")

echo "== the images are found in a render"
check "the chart has an appVersion" test -n "$app_version"
check "default values: the hypervisor image of the pool" has_line "$default_images" \
    "image tensorfusion/tensor-fusion-hypervisor:$app_version"
check "default values: the images of the ProviderConfig" contains "$default_images" "middleware tensorfusion/vgpu-provider-nvidia:"
check "default values: the cluster agent sidecar" contains "$default_images" "tensor-fusion-agent:"

echo "== values-cn.yaml: nothing from Docker Hub"
# Whole render, comments aside: also catches an image in a field this script
# does not know, a ConfigMap or an environment variable.
text=$(grep -v '^[[:space:]]*#' <<<"$out")
while IFS= read -r reference; do
    [ -z "$reference" ] || text=$(grep -vF -- "$reference" <<<"$text" || true)
done < <(allowed)
none "the render does not mention docker.io" "$(grep -n 'docker\.io' <<<"$text" || true)"
none "the render has no unqualified tensorfusion/ image" \
    "$(grep -nE '(^|[^/A-Za-z0-9._-])tensorfusion/' <<<"$text" || true)"
none "no image reference resolves to Docker Hub, named or unqualified" "$(docker_hub "$cn_images")"
none "TensorFusion images come from $MIRROR" \
    "$(grep -E '/tensor-?fusion[^ ]*$|/vgpu-provider-' <<<"$cn_images" | grep -vF " $MIRROR/" || true)"

echo "== values-cn.yaml: same versions as the default values"
# "<" is the default values, ">" the overlay.
# One exception: the cluster agent. The mirror only has its 1.0.0 release, so
# the overlay stays there until a newer one is mirrored (see the note in the
# overlay); that release does not know the namespace lists.
none "every image has the name and tag it has with the default values (the agent excepted)" \
    "$(diff <(versions "$default_images") <(versions "$cn_images") | grep '^[<>]' | grep -v 'tensor-fusion-agent' || true)"
hypervisor=$(sed -n 's|^image \(.*/tensor-fusion-hypervisor:.*\)$|\1|p' <<<"$cn_images")
check "the hypervisor image is the appVersion release ($app_version), got ${hypervisor:-none}: cluster.hypervisorImage in the overlay is bumped with appVersion" \
    same "$hypervisor" "$MIRROR/tensor-fusion-hypervisor:$app_version"
check "the controller image is that release too" has_line "$cn_images" "image $MIRROR/tensor-fusion-operator:$app_version"
check "the Vector image is the fixed version, for the controller" has_line "$cn_images" \
    "image docker.m.daocloud.io/timberio/vector:0.56.0-alpine"
check "and for the hypervisor sidecar" has_line "$cn_images" \
    "vectorImage docker.m.daocloud.io/timberio/vector:0.56.0-alpine"

echo "== values-cn.yaml: lint"
check "the chart lints with the overlay" helm lint "$CHART" -f "$CN_VALUES"

if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed"
