#!/usr/bin/env bash
# KWOK tier, stage "test": build the suite and run the cases labelled `kwok`
# against the cluster of up.sh. Runs on the machine that holds the kind cluster.
#
#   FOCUS='KRP-A02|KRP-A16'   only these cases
#   RUN_ID=kwok-1             results land in e2e/results/<RUN_ID>/

source "$(dirname "$0")/lib.sh"
need kubectl "$GO"
cluster_exists || { log "kind cluster ${KWOK_CLUSTER} not found: run up.sh first"; exit 1; }
if ! { [ -f "${KWOK_HOME}/fakehv.pid" ] && kill -0 "$(cat "${KWOK_HOME}/fakehv.pid")" 2>/dev/null; }; then
  log "fake hypervisor is not running: run up.sh again"
  exit 1
fi

RUN_ID="${RUN_ID:-kwok-$(date -u +%Y%m%d-%H%M%S)}"
RESULTS="${RESULTS:-${REPO_ROOT}/e2e/results/${RUN_ID}}"
TEST_BIN="${KWOK_HOME}/bin/suite.test"
mkdir -p "$RESULTS"

log "building the suite"
(cd "$REPO_ROOT" && "$GO" test -tags e2e -mod=vendor -c -o "$TEST_BIN" ./e2e/suite)

log "tier kwok -> ${RESULTS}"
cd "${REPO_ROOT}/e2e/suite"
set +e
E2E_RUN_ID="$RUN_ID" E2E_RESULTS_DIR="$RESULTS" E2E_GPU_MODE=fake \
  E2E_FIXED_IMAGES="${E2E_FIXED_IMAGES:-1}" E2E_FINOPS_HOLD="${E2E_FINOPS_HOLD:-45s}" \
  E2E_GPU_NODECLASS="$NODECLASS" E2E_GPU_NODEPOOL="$GPU_NODEPOOL" E2E_CPU_NODEPOOL="$CPU_NODEPOOL" \
  "$TEST_BIN" -test.timeout "${TIMEOUT:-90m}" -test.v \
  -ginkgo.label-filter="${FILTER:-kwok}" ${FOCUS:+-ginkgo.focus="$FOCUS"} \
  -ginkgo.junit-report="${RESULTS}/junit.xml" -ginkgo.json-report="${RESULTS}/ginkgo.json" \
  -ginkgo.no-color -ginkgo.seed=1 -ginkgo.timeout="${TIMEOUT:-90m}" 2>&1 | tee "${RESULTS}/run.log" |
  grep -E '^(•|\s*\[|  measure|  note|Ran |PASS|FAIL|ok|---|summary:)|STEP'
rc=${PIPESTATUS[0]}
set -e
cp "${KWOK_HOME}/logs/fakehv.log" "${RESULTS}/fakehv.log" 2>/dev/null || true
kubectl -n "$TF_NAMESPACE" logs deploy/"${TF_RELEASE}-controller" -c controller --tail=-1 >"${RESULTS}/operator.log" 2>/dev/null || true
kubectl -n kube-system logs deploy/karpenter --tail=-1 >"${RESULTS}/karpenter.log" 2>/dev/null || true
echo
head -60 "${RESULTS}/summary.md" 2>/dev/null || true
exit "$rc"
