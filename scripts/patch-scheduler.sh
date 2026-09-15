#!/bin/bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

for patch in \
  patches/scheduler-csi-capacity-1.patch \
  patches/scheduler-csi-capacity-2.patch \
  patches/scheduler-csi-capacity-3.patch \
  patches/scheduler-pdb-1.patch \
  patches/scheduler-pdb-2.patch \
  patches/scheduler-sched-one.patch; do
  if git apply --check "$patch" 2>/dev/null; then
    git apply "$patch"
    echo "Applied: $patch"
  elif git apply --reverse --check "$patch" 2>/dev/null; then
    echo "Already applied: $patch"
  else
    echo "Cannot apply $patch to the current vendor sources." >&2
    git apply --check "$patch"
    exit 1
  fi
done
