# Pod security for TensorFusion workloads

[中文](pod-security.zh.md)

TensorFusion admission keeps each workload pod inside its own directory of the
node-level data volume. It does not replace a cluster pod security policy: who
may create privileged pods, declare `hostPath` volumes, or open a shell in a
pod is decided by the cluster administrator. This page lists both sides and
gives the namespace labels to start from.

Audience: cluster administrators who run TensorFusion in a cluster shared by
several teams.

## What the operator enforces

Local GPU pods get a `hostPath` volume named `tf-data` (`/run/tensor-fusion` on
the node), unless their isolation mode is `shared` or `partitioned`. That
leaves `soft`, the default, and `hard`. The limiter and the node's hypervisor
exchange state through the volume, and it holds one directory per pod.

| When | Rule |
| --- | --- |
| Pod creation (mutating webhook) | Each container and init container keeps at most one mount of `tf-data`: `mountPath: /run/tensor-fusion/shm` with `subPathExpr: shm/$(TF_SHM_POD_NAMESPACE)/$(TF_SHM_POD_NAME)`. Any other mount of `tf-data` in the pod spec is removed. |
| Pod creation (mutating webhook) | In a container with that mount, `TF_SHM_POD_NAMESPACE` and `TF_SHM_POD_NAME` are reserved. Whatever the pod spec defines for them is replaced with the downward API values `metadata.namespace` and `metadata.name`. |
| Update of `pods/ephemeralcontainers` (validating webhook) | In a pod that has a data volume, an ephemeral container that mounts a data volume, or sets one of the two reserved variables to anything but its downward API value, is refused. Other ephemeral containers are admitted, so `kubectl debug` keeps working. |

### Which pods the ephemeral container rule covers

The rule covers every pod that has a data volume. It does not read the
`tensor-fusion.ai/enabled` label.

- A data volume is a volume named `tf-data`, or a `hostPath` volume whose path
  is `/run/tensor-fusion` or a directory below it, whatever the volume is
  called. `/run/tensor-fusion-other` is a different directory and does not
  count.
- The volumes of a pod cannot be changed after the pod is created, so the rule
  holds for as long as the pod exists. Changing the pod's labels or annotations
  has no effect on it.
- TensorFusion's own pods that have the `tf-data` volume, such as the
  hypervisor pods, are covered as well.
- The webhook receives every update of `pods/ephemeralcontainers` in the
  cluster, in all namespaces. For a pod without a data volume it admits the
  request without further checks.

### When the operator is unavailable

Each webhook has its own failure policy:

| Webhook | Helm value | Default | While the operator is unavailable |
| --- | --- | --- | --- |
| Mutating, pod creation | `controller.admissionWebhooks.failurePolicy` | `Fail` | Pods labeled `tensor-fusion.ai/enabled: "true"` cannot be created. |
| Validating, ephemeral containers | `controller.admissionWebhooks.ephemeralContainers.failurePolicy` | `Ignore` | Ephemeral containers are admitted without the check, for every pod, including pods with a data volume. |

The API server waits
`controller.admissionWebhooks.ephemeralContainers.timeoutSeconds` (default `3`)
for the validating webhook. A request that gets no answer in that time is
handled by the failure policy.

The default is `Ignore` because the webhook sees every pod: with `Fail`, an
operator outage stops `kubectl debug` in all namespaces, `kube-system`
included, until the operator is back. Set `Fail` if the rule has to hold during
an outage too, for example when teams that share the cluster can add ephemeral
containers to their own pods:

```yaml
controller:
  admissionWebhooks:
    ephemeralContainers:
      failurePolicy: Fail
```

## What stays with the cluster administrator

The operator does not check the following. Cover them with Pod Security
Admission, a policy engine, and RBAC.

| Area | Why it matters | Control |
| --- | --- | --- |
| `hostPath` volumes a pod declares itself, under any name or path | They reach node directories directly, including `/run/tensor-fusion`. The ephemeral container rule covers such a volume when its path is `/run/tensor-fusion` or below, but the pod's own containers mount it as declared. A parent directory such as `/run`, or a PersistentVolume backed by a node path, is not recognised as a data volume. | Pod Security Admission or a policy engine |
| Privileged containers, `hostPID`, `hostNetwork`, `hostIPC`, added capabilities | They give access to the node and to other pods on it. This applies to ephemeral containers too. | Pod Security Admission or a policy engine |
| `pods/exec`, `pods/attach`, `pods/ephemeralcontainers` | They run commands inside a workload pod with the pod's mounts. | RBAC |

## Recommended policy

### Namespaces without the `tf-data` volume

Enforce the `baseline` Pod Security Standard and report against `restricted`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: team-a
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/enforce-version: latest
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/audit: restricted
```

`baseline` refuses `hostPath` volumes, privileged containers and host
namespaces, for ephemeral containers as well.

Before you apply the label to an existing namespace, list the pods it would
refuse:

```bash
kubectl label --dry-run=server --overwrite namespace team-a \
  pod-security.kubernetes.io/enforce=baseline
```

### Namespaces with `soft` or `hard` isolation pods

`enforce: baseline` cannot be used here. Pod Security Admission evaluates the
pod after the mutating webhook has added `tf-data`, and refuses it:

```text
violates PodSecurity "baseline:latest": hostPath volumes (volume "tf-data")
```

Pod Security Admission has no per-volume exemption, so in these namespaces:

1. Keep `baseline` as a warning and in the audit log:

   ```yaml
   pod-security.kubernetes.io/warn: baseline
   pod-security.kubernetes.io/audit: baseline
   ```

2. Enforce the same restrictions with a policy engine or a
   `ValidatingAdmissionPolicy` that allows a `hostPath` volume only when it is
   the injected one (name `tf-data`, path `/run/tensor-fusion`, on a pod
   labeled `tensor-fusion.ai/enabled: "true"`) and refuses privileged
   containers and host namespaces. TensorFusion does not ship this policy.

### RBAC

Grant `pods/ephemeralcontainers`, `pods/exec` and `pods/attach` only to the
people who debug workloads in that namespace. Check a user's access with:

```bash
kubectl auth can-i patch pods --subresource=ephemeralcontainers -n team-a --as jane
kubectl auth can-i create pods --subresource=exec -n team-a --as jane
```

The controller's ClusterRole also grants `get` on `pods/log` in every
namespace. The operator does not read pod logs itself. The grant is for the
`cluster-agent` sidecar, which runs in the controller pod under the same
ServiceAccount and serves container logs to the TensorFusion console. Logs
opened in the console are therefore read with the controller's identity, not
the viewer's: whoever may view logs of this cluster in the console can read the
logs of any pod in any namespace, whatever their own Kubernetes RBAC allows.
To restrict this:

- Give console access to this cluster only to people who may read all pod logs.
- If the cluster is not connected to the console, set neither `agent.agentId`
  nor `agent.existingSecret.agentIdKey`. The chart then does not add the
  sidecar.
- To limit the namespaces, remove the `pods/log` rule from the ClusterRole and
  grant it to the controller's ServiceAccount with a Role and RoleBinding in
  each namespace whose logs the console may show. The chart has no value for
  this. Apply the change with a Helm post-renderer; a manual edit is undone by
  the next `helm upgrade`.

## Next steps

- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Enforce Pod Security Standards with namespace labels](https://kubernetes.io/docs/tasks/configure-pod-container/enforce-standards-namespace-labels/)
- [Local soft isolation and application preload libraries](soft-isolation-preload.md)
