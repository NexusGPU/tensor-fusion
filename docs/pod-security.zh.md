# TensorFusion 工作负载的 Pod 安全

[English](pod-security.md)

TensorFusion 的准入控制保证每个业务 Pod 只能访问节点级数据卷中属于自己的目录。它不能代替集群的 Pod 安全策略：谁可以创建特权 Pod、声明 `hostPath` 卷、进入 Pod 执行命令，由集群管理员决定。本文列出两边各自负责的内容，并给出可以直接使用的命名空间标签。

读者：在多团队共用的集群中运行 TensorFusion 的集群管理员。

## Operator 负责的部分

本地 GPU 模式的 Pod 会得到一个名为 `tf-data` 的 `hostPath` 卷（节点上的 `/run/tensor-fusion`），隔离模式为 `shared` 或 `partitioned` 时除外。也就是默认的 `soft` 和 `hard` 两种模式。限制器和节点上的 hypervisor 通过这个卷交换状态，每个 Pod 在其中有一个自己的目录。

| 时机 | 规则 |
| --- | --- |
| 创建 Pod（mutating webhook） | 每个容器和 init 容器最多保留一个 `tf-data` 挂载：`mountPath: /run/tensor-fusion/shm`，`subPathExpr: shm/$(TF_SHM_POD_NAMESPACE)/$(TF_SHM_POD_NAME)`。Pod 定义中对 `tf-data` 的其他挂载会被移除。 |
| 创建 Pod（mutating webhook） | 带有该挂载的容器中，`TF_SHM_POD_NAMESPACE` 和 `TF_SHM_POD_NAME` 是保留变量。Pod 定义中的同名变量会被替换为 downward API 的 `metadata.namespace` 和 `metadata.name`。 |
| 更新 `pods/ephemeralcontainers`（validating webhook） | Pod 带有数据卷时，临时容器挂载数据卷，或把两个保留变量设成 downward API 以外的值，请求会被拒绝。其他临时容器正常放行，`kubectl debug` 可以照常使用。 |

### 临时容器规则作用于哪些 Pod

规则作用于所有带数据卷的 Pod，不看 `tensor-fusion.ai/enabled` 标签。

- 数据卷指名为 `tf-data` 的卷，或者路径是 `/run/tensor-fusion` 及其子目录的 `hostPath` 卷，卷叫什么名字都算。`/run/tensor-fusion-other` 是另一个目录，不算。
- Pod 创建后不能修改卷，所以只要 Pod 还在，规则就一直生效。修改 Pod 的标签或注解不影响规则。
- TensorFusion 自己带 `tf-data` 卷的 Pod（比如 hypervisor Pod）同样受规则约束。
- 集群内所有命名空间对 `pods/ephemeralcontainers` 的更新都会发到这个 webhook。Pod 没有数据卷时，webhook 直接放行，不做其他检查。

### Operator 不可用时

两个 webhook 各有自己的失败策略：

| Webhook | Helm 配置项 | 默认值 | Operator 不可用期间 |
| --- | --- | --- | --- |
| Mutating，创建 Pod | `controller.admissionWebhooks.failurePolicy` | `Fail` | 带 `tensor-fusion.ai/enabled: "true"` 标签的 Pod 无法创建。 |
| Validating，临时容器 | `controller.admissionWebhooks.ephemeralContainers.failurePolicy` | `Ignore` | 临时容器不经检查直接放行，所有 Pod 都是如此，带数据卷的 Pod 也不例外。 |

API server 等待 validating webhook 的时间由 `controller.admissionWebhooks.ephemeralContainers.timeoutSeconds` 决定（默认 `3` 秒）。超时未收到答复的请求按失败策略处理。

默认值是 `Ignore`，因为这个 webhook 会收到所有 Pod 的请求：设为 `Fail` 后，Operator 一旦不可用，所有命名空间（包括 `kube-system`）都无法使用 `kubectl debug`，直到 Operator 恢复。如果规则在 Operator 不可用期间也必须生效，比如共用集群的各个团队都能向自己的 Pod 添加临时容器，就设为 `Fail`：

```yaml
controller:
  admissionWebhooks:
    ephemeralContainers:
      failurePolicy: Fail
```

## 集群管理员负责的部分

下面这些 Operator 不检查，需要用 Pod Security Admission、策略引擎和 RBAC 覆盖。

| 范围 | 原因 | 控制手段 |
| --- | --- | --- |
| Pod 自己声明的 `hostPath` 卷，不论名称和路径 | 可以直接访问节点目录，包括 `/run/tensor-fusion`。路径是 `/run/tensor-fusion` 及其子目录时，临时容器规则会覆盖这个卷，但 Pod 自己的容器仍按声明挂载。`/run` 这样的上级目录，以及由节点路径提供的 PersistentVolume，不会被识别为数据卷。 | Pod Security Admission 或策略引擎 |
| 特权容器、`hostPID`、`hostNetwork`、`hostIPC`、额外的 capabilities | 可以访问节点和节点上的其他 Pod，临时容器同样适用。 | Pod Security Admission 或策略引擎 |
| `pods/exec`、`pods/attach`、`pods/ephemeralcontainers` | 可以在业务 Pod 内执行命令，并使用该 Pod 的挂载。 | RBAC |

## 推荐策略

### 没有 `tf-data` 卷的命名空间

强制执行 `baseline` 级别的 Pod Security Standard，并按 `restricted` 级别给出警告和审计记录：

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

`baseline` 会拒绝 `hostPath` 卷、特权容器和主机命名空间，对临时容器同样生效。

给已有命名空间打标签之前，先查看哪些 Pod 会被拒绝：

```bash
kubectl label --dry-run=server --overwrite namespace team-a \
  pod-security.kubernetes.io/enforce=baseline
```

### 运行 `soft` 或 `hard` 隔离 Pod 的命名空间

这里不能使用 `enforce: baseline`。Pod Security Admission 在 mutating webhook 添加 `tf-data` 之后检查 Pod，会拒绝它：

```text
violates PodSecurity "baseline:latest": hostPath volumes (volume "tf-data")
```

Pod Security Admission 不支持按卷豁免，所以这类命名空间按下面的方式处理：

1. 把 `baseline` 保留为警告和审计：

   ```yaml
   pod-security.kubernetes.io/warn: baseline
   pod-security.kubernetes.io/audit: baseline
   ```

2. 用策略引擎或 `ValidatingAdmissionPolicy` 强制执行同样的限制：只允许注入的那个 `hostPath` 卷（名称 `tf-data`，路径 `/run/tensor-fusion`，且 Pod 带 `tensor-fusion.ai/enabled: "true"` 标签），拒绝特权容器和主机命名空间。TensorFusion 目前不提供这条策略。

### RBAC

只把 `pods/ephemeralcontainers`、`pods/exec`、`pods/attach` 授予需要在该命名空间调试业务的人。用下面的命令查看某个用户的权限：

```bash
kubectl auth can-i patch pods --subresource=ephemeralcontainers -n team-a --as jane
kubectl auth can-i create pods --subresource=exec -n team-a --as jane
```

控制器的 ClusterRole 还授予了所有命名空间内 `pods/log` 的 `get` 权限。Operator 自己不读 Pod 日志，这条权限是给 `cluster-agent` sidecar 用的：它运行在控制器 Pod 里，使用同一个 ServiceAccount，向 TensorFusion 控制台提供容器日志。因此，在控制台里打开的日志是以控制器的身份读取的，而不是查看者的身份：只要能在控制台查看这个集群的日志，就能读到任意命名空间中任意 Pod 的日志，不受本人 Kubernetes RBAC 的限制。收紧的办法：

- 只把这个集群的控制台访问权限给可以阅读全部 Pod 日志的人。
- 集群不接入控制台时，`agent.agentId` 和 `agent.existingSecret.agentIdKey` 都不要设置，Chart 就不会添加这个 sidecar。
- 需要限定命名空间时，从 ClusterRole 中删除 `pods/log` 规则，再在允许控制台查看日志的每个命名空间里，用 Role 和 RoleBinding 把它授予控制器的 ServiceAccount。Chart 没有对应的配置项，请用 Helm post-renderer 做这个修改；手工修改会被下一次 `helm upgrade` 还原。

## 下一步

- [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/)
- [Enforce Pod Security Standards with namespace labels](https://kubernetes.io/docs/tasks/configure-pod-container/enforce-standards-namespace-labels/)
- [Local soft isolation and application preload libraries](soft-isolation-preload.md)
