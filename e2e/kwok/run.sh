#!/usr/bin/env bash
# Entry point of `make kwok-up | kwok-test | kwok-down`.
#
# Local (default): runs the stage on this machine (needs docker here).
# Remote: KWOK_SSH_HOST=<ssh host> copies the checkout to that host
# (KWOK_REMOTE_DIR, default tf-kwok/repo under the remote home) and runs the same
# stage there; test results are copied back to e2e/results/. KWOK_*, TF_*, E2E_*,
# KARPENTER_*, GO* variables and FOCUS / FILTER / RUN_ID / TIMEOUT are forwarded.
# KWOK_REMOTE_ENV names a file on the remote host that is sourced first
# (mirrors, proxies and paths of that host live there, not in this repository).

set -euo pipefail
stage="${1:?usage: run.sh up|test|down}"
dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "${dir}/../.." && pwd)"

if [ -z "${KWOK_SSH_HOST:-}" ]; then
  exec bash "${dir}/${stage}.sh"
fi

remote="${KWOK_REMOTE_DIR:-tf-kwok/repo}"
ssh "$KWOK_SSH_HOST" "mkdir -p '${remote}'"
rsync -az --delete --exclude .git --exclude 'tmp*' --exclude '/bin' --exclude 'e2e/results' --exclude 'e2e/kwok/.work' \
  --exclude 'e2e/terraform/.terraform' --exclude 'e2e/terraform/.kube' --exclude 'e2e/terraform/terraform.tfstate*' --exclude 'e2e/terraform/*.tfvars' \
  "${repo}/" "${KWOK_SSH_HOST}:${remote}/"

[ "$stage" = "test" ] && export RUN_ID="${RUN_ID:-kwok-$(date -u +%Y%m%d-%H%M%S)}"
fwd=""
while IFS='=' read -r name value; do
  case "$name" in
    KWOK_SSH_HOST | KWOK_REMOTE_DIR | KWOK_REMOTE_ENV) ;;
    KWOK_* | TF_* | E2E_* | KARPENTER_* | GOPROXY | GOSUMDB | GOTOOLCHAIN | GOFLAGS | FOCUS | FILTER | RUN_ID | TIMEOUT)
      fwd+="export ${name}=$(printf '%q' "$value"); " ;;
  esac
done < <(env)

rc=0
ssh "$KWOK_SSH_HOST" 'bash -s' <<REMOTE || rc=$?
set -e
[ -n "${KWOK_REMOTE_ENV:-}" ] && . "${KWOK_REMOTE_ENV:-/dev/null}"
${fwd}
cd "${remote}" && bash e2e/kwok/${stage}.sh
REMOTE
if [ "$stage" = "test" ]; then
  mkdir -p "${repo}/e2e/results/${RUN_ID}"
  rsync -az "${KWOK_SSH_HOST}:${remote}/e2e/results/${RUN_ID}/" "${repo}/e2e/results/${RUN_ID}/" || true
  echo "results: e2e/results/${RUN_ID}/"
fi
exit "$rc"
