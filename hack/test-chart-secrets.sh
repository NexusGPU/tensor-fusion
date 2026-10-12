#!/usr/bin/env bash
# Renders the chart with `helm template` and checks that credentials never
# appear as values in a pod spec or a ConfigMap:
#   - the cluster agent's enroll token and id come from a Secret, created by the
#     chart or named by the user;
#   - rotating the chart-managed token changes the controller pod template;
#   - a release upgraded with --reuse-values from a chart that had the token
#     inline renders (its values lack every key added since);
#   - the Alertmanager configuration is a Secret.
# It also checks what the hypervisor pods depend on:
#   - the Vector ConfigMap they mount is rendered whatever the values are, and
#     never holds the database password;
#   - the chart sets no rate limiter parameters unless the values do;
#   - the default hypervisor image is the controller's release.
set -euo pipefail

CHART=${CHART:-charts/tensor-fusion}
TOKEN=example-enroll-token
AGENT_ID=example-org:example-env
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

# doc <rendered> <kind> <name>: the one document of that kind and name.
doc() {
    RENDERED="$1" python3 - "$2" "$3" <<'PY'
import os, sys
kind, name = sys.argv[1], sys.argv[2]
for document in os.environ["RENDERED"].split("\n---"):
    lines = document.splitlines()
    if f"kind: {kind}" in lines and any(line.strip() == f"name: {name}" for line in lines):
        print(document)
        sys.exit(0)
sys.exit(1)
PY
}

contains() { grep -qF -- "$2" <<<"$1"; }
fails() { ! "$@"; }
lacks() { ! grep -qF -- "$2" <<<"$1"; }
b64() { printf '%s' "$1" | base64 | tr -d '\n'; }

echo "== token set: the chart creates the Secret"
out=$(render --set agent.agentId="$AGENT_ID" --set agent.enrollToken="$TOKEN")
deployment=$(doc "$out" Deployment release-controller)
secret=$(doc "$out" Secret release-agent)
check "cluster-agent sidecar is present" contains "$deployment" "name: cluster-agent"
check "pod spec does not contain the token" lacks "$deployment" "$TOKEN"
check "pod spec does not contain the agent id" lacks "$deployment" "$AGENT_ID"
check "token is referenced from the Secret" contains "$deployment" "key: enroll-token"
check "agent id is referenced from the Secret" contains "$deployment" "key: agent-id"
check "Secret holds the token" contains "$secret" "enroll-token: \"$(b64 "$TOKEN")\""
check "Secret holds the agent id" contains "$secret" "agent-id: \"$(b64 "$AGENT_ID")\""
check "pod template tracks the Secret" contains "$deployment" "checksum/agent-secret:"
checksum=$(grep -F "checksum/agent-secret:" <<<"$deployment")
rotated=$(doc "$(render --set agent.agentId="$AGENT_ID" --set agent.enrollToken=rotated-token)" \
    Deployment release-controller)
check "a rotated token changes the pod template" lacks "$rotated" "$checksum"
other=$(doc "$(render --set agent.agentId="$AGENT_ID" --set agent.enrollToken="$TOKEN" \
    --set controller.replicaCount=2)" Deployment release-controller)
check "an unrelated value does not change the checksum" contains "$other" "$checksum"

echo "== namespace scope of the cluster agent"
out=$(render --set agent.agentId="$AGENT_ID" --set agent.enrollToken="$TOKEN")
deployment=$(doc "$out" Deployment release-controller)
check "no namespace lists by default" lacks "$deployment" "AGENT_NAMESPACE_"
check "the default agent image is 1.1.0" contains "$deployment" "tensor-fusion-agent:1.1.0"
out=$(render --set agent.agentId="$AGENT_ID" --set agent.enrollToken="$TOKEN" \
    --set 'agent.namespaceAllowlist={team-a,team-b-*}' --set 'agent.namespaceDenylist={kube-system}')
deployment=$(doc "$out" Deployment release-controller)
check "the allowlist reaches the sidecar" contains "$deployment" 'value: "team-a,team-b-*"'
check "under its variable" contains "$deployment" "name: AGENT_NAMESPACE_ALLOWLIST"
check "the denylist reaches the sidecar" contains "$deployment" 'value: "kube-system"'
check "under its variable" contains "$deployment" "name: AGENT_NAMESPACE_DENYLIST"
out=$(render --set agent.agentId="$AGENT_ID" --set agent.enrollToken="$TOKEN" \
    --set agent.namespaceAllowlist=null --set agent.namespaceDenylist=null)
check "values without the keys (an upgrade with --reuse-values) render" \
    contains "$(doc "$out" Deployment release-controller)" "name: cluster-agent"
out=$(render --set 'agent.namespaceAllowlist={team-a}')
check "the lists are not rendered without the sidecar" lacks "$out" "AGENT_NAMESPACE_"
for bad in 'Team-A' 'team_a' '*' 'team-*-x' '-team' 'a,b' ''; do
    check "an invalid entry ($bad) is refused when the chart is rendered" \
        fails render --set agent.agentId="$AGENT_ID" --set agent.enrollToken="$TOKEN" \
        --set-json "agent.namespaceAllowlist=[\"$bad\"]"
    check "in the denylist as well ($bad)" \
        fails render --set agent.agentId="$AGENT_ID" --set agent.enrollToken="$TOKEN" \
        --set-json "agent.namespaceDenylist=[\"$bad\"]"
done

echo "== existing secret: the chart creates none"
out=$(render --set agent.agentId="$AGENT_ID" --set agent.existingSecret.name=my-agent --set agent.existingSecret.key=token)
deployment=$(doc "$out" Deployment release-controller)
check "cluster-agent sidecar is present" contains "$deployment" "name: cluster-agent"
check "no Secret is created" lacks "$out" "name: release-agent"
check "token is referenced from the named Secret" contains "$deployment" "name: my-agent"
check "with the named key" contains "$deployment" "key: token"
check "no checksum for a Secret the chart cannot see" lacks "$deployment" "checksum/agent-secret:"
out=$(render --set agent.existingSecret.name=my-agent --set agent.existingSecret.agentIdKey=id \
    --set agent.enrollToken="$TOKEN")
deployment=$(doc "$out" Deployment release-controller)
check "agent id from the named Secret enables the sidecar" contains "$deployment" "key: id"
check "an existing Secret takes precedence over enrollToken" lacks "$out" "$(b64 "$TOKEN")"

echo "== neither: no sidecar"
out=$(render)
deployment=$(doc "$out" Deployment release-controller)
check "cluster-agent sidecar is absent" lacks "$deployment" "name: cluster-agent"
check "no Secret is created" lacks "$out" "name: release-agent"
check "no checksum annotation" lacks "$deployment" "checksum/agent-secret:"
out=$(render --set agent.enrollToken="$TOKEN")
check "a token without an agent id renders nothing of it" lacks "$out" "$(b64 "$TOKEN")"

echo "== upgrade with --reuse-values from a release that had the token inline"
# Such a release carries the values of the chart it was installed from: the
# keys added since do not exist in them. Setting a key to null removes it.
out=$(render --set agent.agentId="$AGENT_ID" --set agent.enrollToken="$TOKEN" --set agent.existingSecret=null)
deployment=$(doc "$out" Deployment release-controller)
check "renders without the new keys" contains "$deployment" "name: cluster-agent"
check "pod spec does not contain the token" lacks "$deployment" "$TOKEN"
check "token is referenced from the Secret" contains "$deployment" "key: enroll-token"
check "Secret is created" contains "$out" "name: release-agent"

echo "== other credentials"
out=$(render --set alert.enabled=true)
check "Alertmanager configuration is a Secret" doc "$out" Secret release-alert-manager-config
check "and not a ConfigMap" lacks "$(doc "$out" ConfigMap release-alert-manager-config || true)" "alertmanager.yml"
out=$(render --set greptime.isCloud=true --set greptime.password=example-db-password \
    --set agent.agentId="$AGENT_ID" --set agent.enrollToken="$TOKEN")
deployment=$(doc "$out" Deployment release-controller)
check "pod spec does not contain the database password" lacks "$deployment" "example-db-password"
config=$(doc "$out" ConfigMap tensor-fusion-sys-vector-config)
check "Vector configuration does not contain the database password" lacks "$config" "example-db-password"
check "Vector configuration reads it from the environment" contains "$config" '${GREPTIME_PASSWORD}'
check "controller's Vector sidecar gets it from the Secret" contains "$deployment" "name: GREPTIME_PASSWORD"
check "operator is told the Secret for the hypervisors' sidecars" contains "$deployment" "name: TSDB_PASSWORD_SECRET_NAME"
check "nothing but the Secret holds the password" lacks \
    "$(grep -v "password: \"$(b64 example-db-password)\"" <<<"$out")" "$(b64 example-db-password)"
out=$(render)
deployment=$(doc "$out" Deployment release-controller)
check "without a database password no Vector sidecar refers to one" lacks "$deployment" "GREPTIME_PASSWORD"
check "the Vector image is a fixed version" contains "$deployment" "timberio/vector:0.56.0-alpine"
check "and the same one for the hypervisor sidecar" contains "$out" 'vectorImage: "docker.io/timberio/vector:0.56.0-alpine"'
check "no floating Vector tag is left" lacks "$out" "vector:latest"

echo "== Vector configuration"
# The controller's sidecar and every hypervisor pod mount this ConfigMap by
# name from the release namespace, whether or not the chart creates the pool
# and whether or not the pool runs the sidecar.
vector_config_cases=(
    ""
    "--set cluster.enabled=false"
    "--set cluster.enableVector=false"
    "--set schedulingConfigTemplate.enabled=false"
    "--set greptime.installStandalone=false"
    "--set greptime.password=example-db-password"
    "--set greptime.isCloud=true --set greptime.user=example-user --set greptime.password=example-db-password"
)
for args in "${vector_config_cases[@]}"; do
    label=${args:-default values}
    # shellcheck disable=SC2086 # args is a list of options
    out=$(render $args)
    config=$(doc "$out" ConfigMap tensor-fusion-sys-vector-config || true)
    check "[$label] the Vector ConfigMap is rendered" contains "$config" "name: tensor-fusion-sys-vector-config"
    check "[$label] in the namespace of the release" contains "$config" "namespace: tensor-fusion-sys"
    check "[$label] with the hypervisor sidecar's configuration" contains "$config" "vector-hypervisor.yaml: |"
    check "[$label] and the controller sidecar's" contains "$config" "vector-operator.yaml: |"
    check "[$label] without the database password" lacks "$config" "example-db-password"
    check "[$label] not encoded either" lacks "$config" "$(b64 example-db-password)"
    # The hypervisor sidecar: its metrics are posted one line per request, so
    # that a row the database rejects costs that row only, and the
    # hypervisor's own log file is shipped to the log table.
    hypervisor_config=${config#*vector-hypervisor.yaml: |}
    metrics_sink=${hypervisor_config#*sink_greptimedb_hypervisor_metrics:}
    metrics_sink=${metrics_sink%%sink_greptimedb_hypervisor_worker_logs:*}
    logs_sink=${hypervisor_config#*sink_greptimedb_hypervisor_worker_logs:}
    check "[$label] hypervisor metrics are posted one line per request" contains "$metrics_sink" "max_events: 1"
    check "[$label] the hypervisor's log file is a source" contains "$hypervisor_config" "- /logs/pods/*/hypervisor*.log"
    check "[$label] its rows carry the release namespace" contains "$hypervisor_config" '.namespace = "tensor-fusion-sys"'
    check "[$label] and go to the log table" contains "$logs_sink" "- prepare_hypervisor_logs"
    check "[$label] which is tf_system_log" contains "$logs_sink" "table: tf_system_log"
done

echo "== rate limiter parameters"
# They are built into the hypervisor. A value written by the chart would
# replace them in every installation.
out=$(render)
template=$(doc "$out" SchedulingConfigTemplate tf-scheduling-config)
check "the SchedulingConfigTemplate is rendered" contains "$template" "autoFreezeAndResume:"
check "the chart sets no rate limiter parameters" lacks "$out" "elasticRateLimitParameters"
template=$(doc "$(render --set-string schedulingConfigTemplate.elasticRateLimitParameters.burstWindow=0.2)" \
    SchedulingConfigTemplate tf-scheduling-config)
check "a parameter from the values is rendered" contains "$template" 'burstWindow: "0.2"'
check "and no other" lacks "$template" "capacityMin"
template=$(doc "$(render --set schedulingConfigTemplate.elasticRateLimitParameters=null)" \
    SchedulingConfigTemplate tf-scheduling-config || true)
check "renders for a release whose values lack the key" contains "$template" "autoFreezeAndResume:"

echo "== images"
app_version=$(sed -n 's/^appVersion: *"\(.*\)"$/\1/p' "$CHART/Chart.yaml")
out=$(render)
check "the chart has an appVersion" test -n "$app_version"
check "the controller image is that release" contains "$out" "tensor-fusion-operator:$app_version\""
check "and so is the default hypervisor image" contains "$out" "tensor-fusion-hypervisor:$app_version\""

if [ "$failures" -ne 0 ]; then
    echo "$failures check(s) failed"
    exit 1
fi
echo "all checks passed"
