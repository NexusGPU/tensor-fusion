#!/usr/bin/env bash
# Publish a built smprobe (./build.sh) to an OSS bucket, so that a pod can fetch
# it with curl instead of kubectl cp:
#
#   SMPROBE_BUCKET=<bucket> ./publish.sh
#   -> https://<bucket>.oss-<region>.aliyuncs.com/smprobe/v<version>/smprobe (and smprobe.sha256)
#
# The bucket stays private; only the two objects are public-read. It is created
# when missing. Credentials come from the aliyun CLI profile. The bucket is not
# part of the ACK test stack: `make ack-down` does not touch it.
#
#   SMPROBE_BUCKET   bucket name (required; names are global, pick one with a random suffix)
#   SMPROBE_REGION   default us-east-1
set -euo pipefail
cd "$(dirname "$0")"

: "${SMPROBE_BUCKET:?set SMPROBE_BUCKET to the bucket name}"
: "${SMPROBE_REGION:=us-east-1}"
version=$(sed -n 's/^#define SMPROBE_VERSION "\(.*\)"/\1/p' smprobe.c)
[ -x smprobe ] && [ -f smprobe.sha256 ] || { echo "build first: ./build.sh" >&2; exit 1; }
{ sha256sum -c smprobe.sha256 >/dev/null 2>&1 || shasum -a 256 -c smprobe.sha256 >/dev/null 2>&1; } ||
  { echo "smprobe does not match smprobe.sha256: rebuild" >&2; exit 1; }

if ! aliyun oss stat "oss://${SMPROBE_BUCKET}" --region "$SMPROBE_REGION" >/dev/null 2>&1; then
  echo "creating bucket ${SMPROBE_BUCKET} (${SMPROBE_REGION}, private)" >&2
  aliyun oss mb "oss://${SMPROBE_BUCKET}" --region "$SMPROBE_REGION" --acl private >/dev/null
fi
prefix="oss://${SMPROBE_BUCKET}/smprobe/v${version}"
for f in smprobe smprobe.sha256; do
  aliyun oss cp "$f" "${prefix}/${f}" --region "$SMPROBE_REGION" -f --acl public-read >/dev/null
done
echo "https://${SMPROBE_BUCKET}.oss-${SMPROBE_REGION}.aliyuncs.com/smprobe/v${version}/smprobe"
