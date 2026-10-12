"""Dead-man switch of a tf-e2e stack.

Invoked every few minutes by EventBridge. Level-triggered: as long as the
deadline in SSM has passed it keeps removing compute, so nothing a dying local
process (or a still-running Karpenter) leaves or re-creates survives.

  stage 1 (now > deadline):
      scale every managed node group of the cluster to 0, terminate every
      instance that carries one of the stack's tags (Karpenter and
      TensorFusion-provisioned nodes included), delete detached stack volumes.
  stage 2 (now > deadline + grace, optional):
      delete the node groups, then the EKS control plane.

What is left afterwards costs nothing: VPC, IAM, this function. `make down`
removes those; `make up` / `make extend` re-arm and bring the nodes back.
"""

import datetime
import json
import os

import boto3
from botocore.exceptions import ClientError

CLUSTER = os.environ["CLUSTER"]
PROJECT = os.environ["PROJECT"]
PARAM_DEADLINE = os.environ["PARAM_DEADLINE"]
PARAM_STATUS = os.environ["PARAM_STATUS"]
GRACE_SECONDS = int(os.environ.get("GRACE_SECONDS", "-1"))

ec2 = boto3.client("ec2")
eks = boto3.client("eks")
ssm = boto3.client("ssm")

LIVE_STATES = ["pending", "running", "stopping", "stopped"]


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _parse(ts):
    return datetime.datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))


def _tag_filters():
    return [
        [{"Name": "tag:Project", "Values": [PROJECT]}],
        [{"Name": "tag:eks:cluster-name", "Values": [CLUSTER]}],
        [{"Name": "tag:eks:eks-cluster-name", "Values": [CLUSTER]}],
        [{"Name": f"tag:kubernetes.io/cluster/{CLUSTER}", "Values": ["owned"]}],
    ]


def _instances():
    """instance id -> managed node group name ('' when not in one)."""
    found = {}
    for filters in _tag_filters():
        pages = ec2.get_paginator("describe_instances").paginate(
            Filters=filters + [{"Name": "instance-state-name", "Values": LIVE_STATES}]
        )
        for page in pages:
            for res in page["Reservations"]:
                for inst in res["Instances"]:
                    tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
                    found[inst["InstanceId"]] = tags.get("eks:nodegroup-name", "")
    return found


def _nodegroups():
    try:
        names = eks.list_nodegroups(clusterName=CLUSTER)["nodegroups"]
    except ClientError as err:
        if err.response["Error"]["Code"] == "ResourceNotFoundException":
            return None  # cluster is gone
        raise
    return {n: eks.describe_nodegroup(clusterName=CLUSTER, nodegroupName=n)["nodegroup"] for n in names}


def _scale_down(groups, actions):
    """Returns the node groups whose instances the ASG is (about to be) removing itself."""
    draining = set()
    for name, ng in groups.items():
        sc = ng["scalingConfig"]
        if ng["status"] in ("DELETING", "CREATING"):
            draining.add(name)
            continue
        if sc["desiredSize"] == 0 and sc["minSize"] == 0:
            continue
        draining.add(name)
        if ng["status"] != "ACTIVE":
            continue  # UPDATING: retried on the next tick
        try:
            eks.update_nodegroup_config(
                clusterName=CLUSTER,
                nodegroupName=name,
                scalingConfig={"minSize": 0, "desiredSize": 0, "maxSize": max(1, sc["maxSize"])},
            )
            actions.append(f"scaled node group {name} to 0")
        except ClientError as err:
            actions.append(f"scale node group {name} failed: {err.response['Error']['Code']}")
    return draining


def _terminate(instances, skip_groups, actions):
    # Instances of a node group that is being scaled down right now are left to
    # the ASG for one tick: terminating them first only makes it launch replacements.
    ids = [i for i, ng in instances.items() if ng not in skip_groups or ng == ""]
    for start in range(0, len(ids), 50):
        chunk = ids[start : start + 50]
        try:
            ec2.terminate_instances(InstanceIds=chunk)
            actions.append(f"terminated {' '.join(chunk)}")
        except ClientError as err:
            actions.append(f"terminate {' '.join(chunk)} failed: {err.response['Error']['Code']}")


def _delete_volumes(actions):
    seen = set()
    for filters in _tag_filters():
        vols = ec2.describe_volumes(Filters=filters + [{"Name": "status", "Values": ["available"]}])["Volumes"]
        for vol in vols:
            vid = vol["VolumeId"]
            if vid in seen:
                continue
            seen.add(vid)
            try:
                ec2.delete_volume(VolumeId=vid)
                actions.append(f"deleted volume {vid}")
            except ClientError as err:
                actions.append(f"delete volume {vid} failed: {err.response['Error']['Code']}")


def _delete_cluster(groups, actions):
    for name, ng in groups.items():
        if ng["status"] != "DELETING":
            try:
                eks.delete_nodegroup(clusterName=CLUSTER, nodegroupName=name)
                actions.append(f"deleting node group {name}")
            except ClientError as err:
                actions.append(f"delete node group {name} failed: {err.response['Error']['Code']}")
    if groups:
        return
    try:
        eks.delete_cluster(name=CLUSTER)
        actions.append(f"deleting cluster {CLUSTER}")
    except ClientError as err:
        code = err.response["Error"]["Code"]
        if code != "ResourceNotFoundException":
            actions.append(f"delete cluster failed: {code}")


def handler(event, context):
    now = _now()
    deadline = _parse(ssm.get_parameter(Name=PARAM_DEADLINE)["Parameter"]["Value"])
    status = {"checked_at": now.isoformat(timespec="seconds"), "deadline": deadline.isoformat(timespec="seconds")}

    if now < deadline:
        status.update(state="armed", remaining_seconds=int((deadline - now).total_seconds()))
        print(json.dumps(status))
        return status

    actions = []
    groups = _nodegroups()
    draining = _scale_down(groups, actions) if groups else set()
    instances = _instances()
    _terminate(instances, draining, actions)
    _delete_volumes(actions)

    stage = 1
    if GRACE_SECONDS >= 0 and now > deadline + datetime.timedelta(seconds=GRACE_SECONDS) and groups is not None:
        stage = 2
        _delete_cluster(groups, actions)

    status.update(
        state="expired",
        stage=stage,
        live_instances=len(instances),
        node_groups=sorted(groups) if groups else [],
        cluster_exists=groups is not None,
        actions=actions,
    )
    print(json.dumps(status))
    # Only overwrite the stored status when something was done, so `make deadman`
    # keeps showing the tick that actually tore things down.
    if actions:
        ssm.put_parameter(Name=PARAM_STATUS, Value=json.dumps(status)[:4000], Type="String", Overwrite=True)
    return status
