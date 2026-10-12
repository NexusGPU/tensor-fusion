#!/usr/bin/env bash
# ACK tier helper: put a locally built image into containerd on the system node,
# for development builds that no registry carries. Only pods pinned to the
# system node (TensorFusion controller, Karpenter) can use such an image.
#
#   ack/load-image.sh <image in the local docker daemon>...
#   ack/load-image.sh <archive.tar>          a `docker save` archive (no docker daemon needed)

source "$(dirname "$0")/lib.sh"
need kubectl

[ $# -gt 0 ] || { log "usage: load-image.sh <image>... | <archive.tar>"; exit 2; }
if [ $# -eq 1 ] && [ -f "$1" ]; then
  tar="$1"
  trap 'kubectl -n kube-system delete pod tf-e2e-image-loader --wait=false >/dev/null 2>&1' EXIT
else
  need docker
  tar="$(mktemp -d)/images.tar"
  trap 'kubectl -n kube-system delete pod tf-e2e-image-loader --wait=false >/dev/null 2>&1; rm -f "$tar"' EXIT
  docker save "$@" -o "$tar"
fi
kubectl -n kube-system delete pod tf-e2e-image-loader --ignore-not-found >/dev/null
kubectl apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: tf-e2e-image-loader
  namespace: kube-system
spec:
  nodeSelector:
    ${ROLE_LABEL}: system
  hostPID: true
  restartPolicy: Never
  containers:
    - name: loader
      image: public.ecr.aws/docker/library/debian:bookworm-slim
      command: ["sleep", "3600"]
      securityContext:
        privileged: true
      volumeMounts:
        - name: images
          mountPath: /images
  volumes:
    - name: images
      hostPath:
        path: /var/lib/tf-e2e/images
        type: DirectoryOrCreate
YAML
kubectl -n kube-system wait pod/tf-e2e-image-loader --for=condition=Ready --timeout=180s >/dev/null
kubectl -n kube-system cp "$tar" tf-e2e-image-loader:/images/images.tar
# The host's own ctr, in the host's namespaces.
kubectl -n kube-system exec tf-e2e-image-loader -- nsenter -t 1 -m -u -n -i -- \
  sh -c 'ctr -n k8s.io images import /var/lib/tf-e2e/images/images.tar >/dev/null && rm -f /var/lib/tf-e2e/images/images.tar'
log "loaded into containerd on the system node: $*"
