#!/usr/bin/env bash
# ACK tier: an image cache inside the cluster, so that nodes do not pull from the
# public internet. A new GPU node pulls a 12 GB model server image through the
# VPC's NAT gateway (200 Mbit/s, shared) in about 12 minutes; from this cache it
# comes over the VPC network.
#
#   ack/registry.sh            install or update (idempotent; run.sh up calls it)
#   ack/registry.sh status     what is running, what the cache holds
#   ack/registry.sh warm <image>...   read images through the cache once, so that
#                              the first node that needs them does not wait
#   ack/registry.sh purge      delete the bucket, the RAM user and its policy
#
# What it sets up:
#   - On the system node, one pod (namespace tf-registry, host network):
#       * CNCF Distribution (registry) as a pull-through cache, one instance per
#         upstream registry (docker.io, quay.io, registry.k8s.io, ghcr.io). It
#         keeps what it served on the node's local disk and answers from there;
#       * persistent storage: an OSS bucket, written through its S3-compatible
#         API over the internal network. A sidecar (rclone) copies new layers to
#         the bucket, and an init container restores the local disk from it. The
#         bucket outlives the cluster (down.sh keeps it), so a new stack starts
#         with a filled cache.
#         (The registry does not write to the bucket itself: as a pull-through
#         cache with the S3 driver, registry 3.0.0 cancels the upload when the
#         response ends and answers 500, and containerd then pulls from upstream.)
#   - On every node, by a DaemonSet: /etc/containerd/certs.d/<registry>/hosts.toml
#     that makes containerd try the cache first. containerd reads these files on
#     every pull, nothing is restarted. The upstream registry stays the fallback,
#     so a node can still pull when the cache is down (and the cache pod itself
#     can be pulled when the cluster is new).
#
# Credentials: a RAM user that may only use this bucket. Its access key is created
# once, kept in ${ACK_HOME}/registry.env (mode 600, git-ignored) and stored in the
# cluster as a Secret through stdin; it is never printed and never on a command line.

source "$(dirname "$0")/lib.sh"
need kubectl aliyun jq

: "${REGISTRY_IMAGE:=registry:3.0.0}"
: "${REGISTRY_SYNC_IMAGE:=rclone/rclone:1.68}"
: "${REGISTRY_WARM_IMAGE:=quay.io/skopeo/stable:v1.17}"
: "${REGISTRY_SYNC_SECONDS:=60}"             # how often new layers are copied to the bucket
: "${REGISTRY_HELPER_IMAGE:=busybox:1.36}"
REGISTRY_NS=tf-registry
REGISTRY_USER="${ACK_NAME}-registry"
REGISTRY_ENV="${ACK_HOME}/registry.env"
OSS_REGION="oss-${ACK_REGION}"
# name=upstream=port on the node address
MIRRORS="docker.io=https://registry-1.docker.io=5000 quay.io=https://quay.io=5001 registry.k8s.io=https://registry.k8s.io=5002 ghcr.io=https://ghcr.io=5003"

account_id() { aliyun sts GetCallerIdentity | jq -r .AccountId; }
bucket_name() { # stable per account: bucket names are global
  [ -n "${REGISTRY_BUCKET:-}" ] || REGISTRY_BUCKET="${ACK_NAME}-registry-$(account_id | shasum | cut -c1-8)"
  echo "$REGISTRY_BUCKET"
}
system_node_ip() {
  kubectl get nodes -l "${ROLE_LABEL}=system" -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}'
}

ensure_bucket() {
  local bucket=$1
  if aliyun oss stat "oss://${bucket}" --region "$ACK_REGION" >/dev/null 2>&1; then
    return
  fi
  log "creating OSS bucket ${bucket} (${ACK_REGION}, private; kept by down.sh)"
  aliyun oss mb "oss://${bucket}" --region "$ACK_REGION" --acl private >/dev/null
}

ensure_credentials() {
  local bucket=$1 policy
  if [ -f "$REGISTRY_ENV" ] && grep -q '^REGISTRY_ACCESS_KEY_SECRET=' "$REGISTRY_ENV"; then
    return
  fi
  policy=$(jq -cn --arg b "$bucket" '{Version: "1", Statement: [{Effect: "Allow", Action: "oss:*", Resource: ["acs:oss:*:*:\($b)", "acs:oss:*:*:\($b)/*"]}]}')
  aliyun ram GetUser --UserName "$REGISTRY_USER" >/dev/null 2>&1 ||
    aliyun ram CreateUser --UserName "$REGISTRY_USER" --Comments "image cache of the ${ACK_NAME} test stack: bucket ${bucket} only" >/dev/null
  aliyun ram GetPolicy --PolicyType Custom --PolicyName "$REGISTRY_USER" >/dev/null 2>&1 ||
    aliyun ram CreatePolicy --PolicyName "$REGISTRY_USER" --PolicyDocument "$policy" >/dev/null
  aliyun ram AttachPolicyToUser --PolicyType Custom --PolicyName "$REGISTRY_USER" --UserName "$REGISTRY_USER" >/dev/null 2>&1 || true
  # The secret of a key can only be read when the key is created: without the
  # local file the old keys are useless.
  for key in $(aliyun ram ListAccessKeys --UserName "$REGISTRY_USER" | jq -r '.AccessKeys.AccessKey[].AccessKeyId'); do
    aliyun ram DeleteAccessKey --UserName "$REGISTRY_USER" --UserAccessKeyId "$key" >/dev/null
  done
  log "creating an access key for RAM user ${REGISTRY_USER} (bucket ${bucket} only), kept in ${REGISTRY_ENV}"
  (
    umask 077
    aliyun ram CreateAccessKey --UserName "$REGISTRY_USER" |
      jq -r '"REGISTRY_ACCESS_KEY_ID=\(.AccessKey.AccessKeyId)\nREGISTRY_ACCESS_KEY_SECRET=\(.AccessKey.AccessKeySecret)"' >"$REGISTRY_ENV"
  )
  printf '%s created RAM user and policy %s, OSS bucket %s\n' "$(date -u +%FT%TZ)" "$REGISTRY_USER" "$bucket" >>"${ACK_HOME}/account-changes.log"
  sleep 10 # a new key is not accepted everywhere at once
}

registry_config() { # registry_config <name> <upstream> <port>
  cat <<YAML
version: 0.1
log:
  level: info
storage:
  filesystem:
    rootdirectory: /var/lib/registry/$1
  delete:
    enabled: true
  cache:
    blobdescriptor: inmemory
http:
  addr: :$3
proxy:
  remoteurl: $2
  ttl: 0
YAML
}

hosts_toml() { # hosts_toml <upstream> <node ip> <port>
  cat <<TOML
server = "$1"

[host."http://$2:$3"]
  capabilities = ["pull", "resolve"]
TOML
}

install() {
  local bucket ip entry name upstream port rest files=() containers=""
  bucket=$(bucket_name)
  ensure_bucket "$bucket"
  ensure_credentials "$bucket"
  ip=$(system_node_ip)
  [ -n "$ip" ] || { log "no node with ${ROLE_LABEL}=system"; exit 1; }

  kubectl create namespace "$REGISTRY_NS" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  (
    set -a
    # shellcheck disable=SC1090
    . "$REGISTRY_ENV"
    set +a
    jq -n --arg ns "$REGISTRY_NS" --arg endpoint "${OSS_REGION}-internal.aliyuncs.com" '{apiVersion: "v1", kind: "Secret", metadata: {name: "registry-oss", namespace: $ns},
      stringData: {RCLONE_CONFIG_OSS_TYPE: "s3", RCLONE_CONFIG_OSS_PROVIDER: "Alibaba", RCLONE_CONFIG_OSS_ENDPOINT: $endpoint,
        RCLONE_CONFIG_OSS_ACCESS_KEY_ID: env.REGISTRY_ACCESS_KEY_ID, RCLONE_CONFIG_OSS_SECRET_ACCESS_KEY: env.REGISTRY_ACCESS_KEY_SECRET}}' |
      kubectl apply -f - >/dev/null
  )

  local work
  work=$(mktemp -d)
  trap 'rm -rf "$work"' RETURN
  mkdir -p "${work}/hosts"
  for entry in $MIRRORS; do
    name=${entry%%=*}
    rest=${entry#*=}
    upstream=${rest%%=*}
    port=${rest##*=}
    registry_config "$name" "$upstream" "$port" >"${work}/${name}.yml"
    files+=(--from-file="${name}.yml=${work}/${name}.yml")
    hosts_toml "$upstream" "$ip" "$port" >"${work}/hosts/${name}"
    containers+="
        - name: registry-$(echo "$name" | tr . -)
          image: ${REGISTRY_IMAGE}
          args: [\"serve\", \"/etc/tf-registry/${name}.yml\"]
          env: [{name: OTEL_TRACES_EXPORTER, value: none}]
          resources: {requests: {cpu: 20m, memory: 64Mi}, limits: {memory: 1Gi}}
          readinessProbe: {httpGet: {path: /v2/, port: ${port}}, periodSeconds: 5}
          volumeMounts: [{name: config, mountPath: /etc/tf-registry}, {name: data, mountPath: /var/lib/registry}]"
  done
  kubectl -n "$REGISTRY_NS" create configmap registry-config "${files[@]}" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  kubectl -n "$REGISTRY_NS" create configmap registry-hosts --from-file="${work}/hosts" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  local sum
  sum=$(cat "${work}"/*.yml "${work}"/hosts/* | shasum | cut -c1-16)

  kubectl apply -f - >/dev/null <<YAML
apiVersion: apps/v1
kind: Deployment
metadata: {name: registry, namespace: ${REGISTRY_NS}}
spec:
  replicas: 1
  strategy: {type: Recreate}
  selector: {matchLabels: {app: registry}}
  template:
    metadata:
      labels: {app: registry}
      annotations: {tf-e2e/config: "${sum}"}
    spec:
      hostNetwork: true
      nodeSelector: {${ROLE_LABEL}: system}
      tolerations: [{operator: Exists}]
      terminationGracePeriodSeconds: 10
      initContainers:
        # A new system node (a new stack): the local disk gets what the bucket holds.
        - name: restore
          image: ${REGISTRY_SYNC_IMAGE}
          args: ["copy", "oss:${bucket}/registry", "/var/lib/registry", "--transfers", "16", "--checkers", "32", "--size-only", "--stats-one-line", "--stats", "30s"]
          envFrom: [{secretRef: {name: registry-oss}}]
          volumeMounts: [{name: data, mountPath: /var/lib/registry}]
      containers:${containers}
        # New layers go to the bucket; nothing is ever deleted there. Uploads in progress are left out.
        - name: sync
          image: ${REGISTRY_SYNC_IMAGE}
          command:
            - sh
            - -c
            - |
              while true; do
                rclone copy /var/lib/registry "oss:${bucket}/registry" --exclude "**/_uploads/**" --transfers 8 --checkers 32 --size-only --min-age 15s -q || echo "sync failed, will try again"
                sleep ${REGISTRY_SYNC_SECONDS}
              done
          envFrom: [{secretRef: {name: registry-oss}}]
          resources: {requests: {cpu: 20m, memory: 64Mi}, limits: {memory: 1Gi}}
          volumeMounts: [{name: data, mountPath: /var/lib/registry}]
      volumes:
        - {name: config, configMap: {name: registry-config}}
        - {name: data, hostPath: {path: /var/lib/tf-e2e/registry, type: DirectoryOrCreate}}
---
# Every node, the ones Karpenter adds included: containerd tries the cache first.
apiVersion: apps/v1
kind: DaemonSet
metadata: {name: registry-hosts, namespace: ${REGISTRY_NS}}
spec:
  selector: {matchLabels: {app: registry-hosts}}
  updateStrategy: {rollingUpdate: {maxUnavailable: "100%"}}
  template:
    metadata:
      labels: {app: registry-hosts}
      annotations: {tf-e2e/config: "${sum}"}
    spec:
      priorityClassName: system-node-critical
      tolerations: [{operator: Exists}]
      terminationGracePeriodSeconds: 1
      containers:
        - name: hosts
          image: ${REGISTRY_HELPER_IMAGE}
          command:
            - sh
            - -c
            - |
              while true; do
                for f in /hosts/*; do
                  [ -f "\$f" ] || continue
                  d=/certs.d/\$(basename "\$f")
                  mkdir -p "\$d"
                  cmp -s "\$f" "\$d/hosts.toml" || { cp "\$f" "\$d/hosts.toml.tmp" && mv "\$d/hosts.toml.tmp" "\$d/hosts.toml"; }
                done
                sleep 60
              done
          resources: {requests: {cpu: 1m, memory: 8Mi}, limits: {memory: 32Mi}}
          volumeMounts:
            - {name: hosts, mountPath: /hosts}
            - {name: certs, mountPath: /certs.d}
      volumes:
        - {name: hosts, configMap: {name: registry-hosts}}
        - {name: certs, hostPath: {path: /etc/containerd/certs.d, type: DirectoryOrCreate}}
YAML
  kubectl -n "$REGISTRY_NS" rollout status deploy/registry --timeout=30m >/dev/null
  kubectl -n "$REGISTRY_NS" rollout status ds/registry-hosts --timeout=5m >/dev/null
  log "image cache ready on ${ip} (ports 5000-5003), persistent copy in bucket ${bucket}"
  ledger registry "bucket=${bucket} node=${ip}"
}

status() {
  local bucket ip
  bucket=$(bucket_name)
  ip=$(system_node_ip)
  kubectl -n "$REGISTRY_NS" get deploy,ds,pods -o wide 2>&1 || true
  echo "bucket ${bucket}: $(aliyun oss du "oss://${bucket}" --region "$ACK_REGION" 2>/dev/null | grep -E 'total object sum size|total du size' | tr '\n' ' ')"
  kubectl -n "$REGISTRY_NS" exec deploy/registry -c sync -- sh -c 'echo "local disk: $(du -sh /var/lib/registry 2>/dev/null | cut -f1) of $(df -h /var/lib/registry | tail -1 | tr -s " " | cut -d" " -f2,4 | sed "s/ / total, /") free"; for d in /var/lib/registry/*; do [ -d "$d/docker/registry/v2/repositories" ] && echo "$(basename "$d"): $(cd "$d/docker/registry/v2/repositories" && find . -name _manifests -maxdepth 4 | sed "s#^\./##; s#/_manifests##" | tr "\n" " ")"; done' 2>&1 || true
  echo "nodes with the containerd hosts.toml: $(kubectl -n "$REGISTRY_NS" get ds registry-hosts -o jsonpath='{.status.numberReady}/{.status.desiredNumberScheduled}')"
  echo "cache address: http://${ip}:5000 (docker.io) 5001 (quay.io) 5002 (registry.k8s.io) 5003 (ghcr.io)"
}

# cache_ref <image>: the same image addressed through the cache, <node ip>:<port>/<repository>:<tag>.
cache_ref() {
  local image=$1 ip=$2 host repo entry
  case "${image%%/*}" in
    *.* | *:*) host=${image%%/*} repo=${image#*/} ;;
    *) host=docker.io repo=$image ;;
  esac
  [ "$host" != docker.io ] || case "$repo" in */*) ;; *) repo="library/${repo}" ;; esac
  for entry in $MIRRORS; do
    [ "${entry%%=*}" = "$host" ] && { echo "${ip}:${entry##*=}/${repo}"; return; }
  done
  log "no cache for registry ${host} (image ${image})"
  return 1
}

# warm <image>...: a Job that reads the images through the cache and throws the
# copy away, so the layers are in the cache (local disk, then the bucket) before
# a node needs them. It does not go through kubelet: an image pulled that way
# would be unpacked on the system node as well.
warm() {
  [ $# -gt 0 ] || { log "usage: registry.sh warm <image>..."; exit 2; }
  local ip script="set -e" image ref start
  ip=$(system_node_ip)
  for image in "$@"; do
    ref=$(cache_ref "$image" "$ip")
    script+="; echo ${image}; skopeo copy -q --src-tls-verify=false --override-os linux --override-arch amd64 docker://${ref} dir:/scratch/i; rm -rf /scratch/i"
  done
  kubectl -n "$REGISTRY_NS" delete job registry-warm --ignore-not-found >/dev/null
  start=$(date +%s)
  kubectl apply -f - >/dev/null <<YAML
apiVersion: batch/v1
kind: Job
metadata: {name: registry-warm, namespace: ${REGISTRY_NS}}
spec:
  backoffLimit: 2
  ttlSecondsAfterFinished: 600
  template:
    spec:
      restartPolicy: Never
      hostNetwork: true
      nodeSelector: {${ROLE_LABEL}: system}
      tolerations: [{operator: Exists}]
      containers:
        - name: warm
          image: ${REGISTRY_WARM_IMAGE}
          command: ["sh", "-c", "${script}"]
          resources: {requests: {cpu: 50m, memory: 128Mi}, limits: {memory: 2Gi}}
          volumeMounts: [{name: scratch, mountPath: /scratch}]
      volumes: [{name: scratch, emptyDir: {}}]
YAML
  kubectl -n "$REGISTRY_NS" wait job/registry-warm --for=condition=Complete --timeout=60m >/dev/null
  log "warmed $# image(s) in $(($(date +%s) - start))s"
  ledger registry-warm "seconds=$(($(date +%s) - start)) images=$#"
}

purge() {
  local bucket
  bucket=$(bucket_name)
  log "deleting bucket ${bucket}, RAM user and policy ${REGISTRY_USER}"
  aliyun oss rm "oss://${bucket}" --region "$ACK_REGION" -r -f >/dev/null 2>&1 || true
  aliyun oss rm "oss://${bucket}" --region "$ACK_REGION" -b -f >/dev/null 2>&1 || true
  for key in $(aliyun ram ListAccessKeys --UserName "$REGISTRY_USER" 2>/dev/null | jq -r '.AccessKeys.AccessKey[].AccessKeyId'); do
    aliyun ram DeleteAccessKey --UserName "$REGISTRY_USER" --UserAccessKeyId "$key" >/dev/null || true
  done
  aliyun ram DetachPolicyFromUser --PolicyType Custom --PolicyName "$REGISTRY_USER" --UserName "$REGISTRY_USER" >/dev/null 2>&1 || true
  aliyun ram DeleteUser --UserName "$REGISTRY_USER" >/dev/null 2>&1 || true
  aliyun ram DeletePolicy --PolicyName "$REGISTRY_USER" >/dev/null 2>&1 || true
  rm -f "$REGISTRY_ENV"
}

case "${1:-install}" in
  install) install ;;
  status) status ;;
  warm)
    shift
    warm "$@"
    ;;
  purge) purge ;;
  *)
    log "usage: registry.sh [install | status | warm <image>... | purge]"
    exit 2
    ;;
esac
